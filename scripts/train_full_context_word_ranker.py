"""Train a word selector on complete candidate-inserted document contexts."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import resource
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from hanglish.models.sentence_ranker import SentenceRanker
from hanglish.models.full_context_word_ranker import FullContextWordRanker
from scripts.train_sentence_ranker import _optimizer, _set_trainable_layers

DATA = ROOT / 'data/natural_context/current-context-v1'
MANIFEST_SHA = 'b280bd12081c90fdd6bf53a3356b0a5f0ff0ccf83b905ca26b027c9d09027989'
PARENT = ROOT / 'checkpoints/sentence_ranker'
RECIPE = dict(seed=20260920, epochs=1, top_layers=2, head_lr=5e-5,
              encoder_lr=1e-5, weight_decay=.01, group_batch=4,
              max_length=288, base_weight=.15, threads=2)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(value):
    return hashlib.sha256(('full-context-word-v1\0' + value).encode()).hexdigest()


def prepare():
    if digest(DATA / 'manifest.json') != MANIFEST_SHA:
        raise ValueError('changed source manifest')
    manifest = json.loads((DATA / 'manifest.json').read_text())
    pools = {}
    pins = {str((DATA / 'manifest.json').relative_to(ROOT)): MANIFEST_SHA}
    for role, fit_limit, hold_limit in [('natural_train', 1280, 512),
                                       ('retention_train', 640, 128)]:
        path = DATA / f'{role}.jsonl'
        expected = manifest['outputs'][role]['sha256']
        if digest(path) != expected:
            raise ValueError('changed training data')
        pins[str(path.relative_to(ROOT))] = expected
        selected = {'fit': [], 'hold': []}
        for line in path.read_text().splitlines():
            row = json.loads(line)
            split = 'hold' if int(key(row['group_id']), 16) % 5 == 0 else 'fit'
            groups = sorted((g for g in row['groups'] if len(g['candidates']) > 1),
                            key=lambda g: (key(g['id']), g['id']))[:2]
            for group in groups:
                count = min(8, len(group['candidates']))
                selected[split].append({
                    'id': group['id'], 'case_id': row['id'], 'group_id': row['group_id'],
                    'role': role, 'context': group['context'], 'source': group['source'],
                    'candidates': group['candidates'][:count],
                    'base_scores': group['base_scores'][:count],
                    'teacher_scores': group['teacher_scores'][:count],
                    'expected': group['expected'],
                })
        for split, limit in [('fit', fit_limit), ('hold', hold_limit)]:
            pools[f'{role}:{split}'] = sorted(selected[split], key=lambda g: (key(g['id']), g['id']))[:limit]
    fit_groups = {r['group_id'] for name, rows in pools.items() if name.endswith(':fit') for r in rows}
    hold_groups = {r['group_id'] for name, rows in pools.items() if name.endswith(':hold') for r in rows}
    if fit_groups & hold_groups:
        raise ValueError('document groups cross partitions')
    return pools, pins


def choose(candidates, scores):
    return min(range(len(candidates)), key=lambda i: (-scores[i], candidates[i]))


def positives(row):
    return [i for i, value in enumerate(row['candidates']) if value in row['expected']]


def group_loss(scores, rows):
    losses = []
    cursor = 0
    for row in rows:
        count = len(row['candidates'])
        group = scores[cursor:cursor+count] + scores.new_tensor(row['base_scores']) * .15
        cursor += count
        indices = positives(row)
        if not indices or len(indices) == count:
            raise ValueError('uninformative training group')
        losses.append(torch.logsumexp(group, 0) - torch.logsumexp(group[indices], 0))
    if cursor != len(scores):
        raise ValueError('unconsumed logits')
    return torch.stack(losses).mean()


def evaluate(ranker, pools, guard):
    outcomes, metrics = [], {}
    for name, rows in pools.items():
        if not name.endswith(':hold'):
            continue
        counts = Counter(rows=len(rows))
        for row in rows:
            candidates = row['candidates']
            base = [a + .15*b for a, b in zip(row['teacher_scores'], row['base_scores'])]
            scores = (ranker.score_pairs([row['context']]*len(candidates), candidates)
                      if all(ranker.can_score(row['context'], c) for c in candidates) else None)
            total = [a+.15*b for a,b in zip(scores,row['base_scores'])] if scores is not None else base
            before, after = candidates[choose(candidates, base)], candidates[choose(candidates, total)]
            a, b = before in row['expected'], after in row['expected']
            counts.update(baseline=int(a), correct=int(b), corrections=int(b and not a),
                          regressions=int(a and not b), missing=int(not positives(row)),
                          fallback=int(scores is None))
            outcomes.append(dict(id=row['id'], pool=name, baseline=before, choice=after,
                                 baseline_correct=a, correct=b, scores=scores))
            guard()
        metrics[name] = dict(counts)
    return dict(metrics=metrics, outcomes=outcomes)


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(RECIPE['seed'])
    random.seed(RECIPE['seed'])
    def rss():
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value if sys.platform == 'darwin' else value*1024
    def guard():
        if rss() > 3*1024**3 or time.monotonic()-started > 1200:
            raise RuntimeError('full-context training resource limit')
    pools, pins = prepare()
    for path in [Path(__file__), ROOT/'hanglish/models/full_context_word_ranker.py',
                 ROOT/'hanglish/models/semantic_ranker.py', ROOT/'hanglish/models/sentence_ranker.py',
                 ROOT/'scripts/train_sentence_ranker.py', *[p for p in PARENT.iterdir() if p.is_file()]]:
        pins[str(path.relative_to(ROOT))] = digest(path)
    (output/'selected-groups.json').write_text(json.dumps(pools, ensure_ascii=False)+'\n')
    sentence = SentenceRanker.load(PARENT)
    sentence.max_length = RECIPE['max_length']
    ranker = FullContextWordRanker(sentence)
    before = evaluate(ranker, pools, guard)
    (output/'before.json').write_text(json.dumps(before, ensure_ascii=False, indent=2)+'\n')
    schedule = [r for name, rows in pools.items() if name.endswith(':fit') for r in rows
                if 0 < len(positives(r)) < len(r['candidates'])
                and all(ranker.can_score(r['context'], c) for c in r['candidates'])]
    random.Random(RECIPE['seed']).shuffle(schedule)
    if not schedule:
        raise ValueError('no informative training examples')
    layers = _set_trainable_layers(sentence.model, 2)
    optimizer = _optimizer(sentence.model, SimpleNamespace(**RECIPE), head_only=False)
    initial_weights = {name: p.detach().clone() for name,p in sentence.model.named_parameters() if p.requires_grad}
    sentence.model.train()
    train_start = time.monotonic()
    losses = []
    with (output/'updates.jsonl').open('x') as stream:
        for start in range(0, len(schedule), 4):
            batch = schedule[start:start+4]
            encoded = ranker.encode_pairs([r['context'] for r in batch for _ in r['candidates']],
                                          [c for r in batch for c in r['candidates']])
            optimizer.zero_grad(set_to_none=True)
            logits = sentence.model(**encoded).logits.reshape(-1)
            loss = group_loss(logits, batch)
            if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(sentence.model.parameters(),1.)
            if not torch.isfinite(norm): raise RuntimeError('nonfinite gradients')
            optimizer.step()
            losses.append(float(loss.detach()))
            stream.write(json.dumps(dict(step=len(losses),loss=losses[-1],ids=[r['id'] for r in batch],rss_bytes=rss()))+'\n')
            stream.flush()
            guard()
            if len(losses)==8:
                projected=(time.monotonic()-train_start)/8*((len(schedule)+3)//4) + 2*(train_start-started)
                print(json.dumps(dict(resource_probe_steps=8,projected_seconds=projected,rss_bytes=rss())),flush=True)
                if projected>1200: raise RuntimeError('projected training resource limit')
            if len(losses)%32==0:
                print(json.dumps(dict(step=len(losses),groups=len(schedule),elapsed_seconds=time.monotonic()-started,rss_bytes=rss())),flush=True)
    sentence.model.eval()
    changed = [name for name,p in sentence.model.named_parameters() if name in initial_weights and not torch.equal(p.detach(),initial_weights[name])]
    if not changed: raise RuntimeError('no neural weights changed')
    checkpoint=output/'checkpoint'
    sentence.model.save_pretrained(checkpoint, safe_serialization=True)
    sentence.tokenizer.save_pretrained(checkpoint)
    metadata=dict(sentence.metadata)
    metadata.update(max_length=288,inference_batch_size=8,experimental_input='full-active-word-context-v1',
                    artifact_hashes={p.name:digest(p) for p in checkpoint.iterdir() if p.is_file()},
                    training_recipe=RECIPE,training_sources=pins)
    (checkpoint/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    sample=schedule[0]
    memory=ranker.score_pairs([sample['context']]*len(sample['candidates']),sample['candidates'])
    reloaded=FullContextWordRanker(SentenceRanker.load(checkpoint))
    saved=reloaded.score_pairs([sample['context']]*len(sample['candidates']),sample['candidates'])
    delta=max(abs(a-b) for a,b in zip(memory,saved))
    if delta>1e-5: raise RuntimeError('reload parity failed')
    after=evaluate(reloaded,pools,guard)
    for path,value in pins.items():
        if digest(ROOT/path)!=value: raise RuntimeError('training source changed')
    natural=after['metrics']['natural_train:hold']; retention=after['metrics']['retention_train:hold']
    comparison = {name: dict(
        cached_semantic_correct=values['baseline'],
        full_context_before_training=before['metrics'][name]['correct'],
        full_context_after_training=values['correct'],
        training_delta=values['correct']-before['metrics'][name]['correct'],
        rows=values['rows']) for name,values in after['metrics'].items()}
    report=dict(completed=True,recipe=RECIPE,selected_counts={k:len(v) for k,v in pools.items()},
                trained_groups=len(schedule),steps=len(losses),layers=layers,changed_parameters=len(changed),
                before=before,after=after,comparison=comparison,reload_max_delta=delta,source_hashes=pins,
                selected_groups_sha256=digest(output/'selected-groups.json'),
                next_runtime_evaluation_allowed=natural['correct']>natural['baseline'] and retention['correct']>=retention['baseline'],
                elapsed_seconds=time.monotonic()-started,peak_rss_bytes=rss(),production_changed=False,
                generalization_claimed=False,new_dev_or_final_read=False)
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('before','after','source_hashes')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    try: run(args.output)
    except Exception as error:
        if args.output.exists():
            (args.output/'failure.json').write_text(json.dumps(dict(error_type=type(error).__name__,error=str(error)),indent=2)+'\n')
        raise
