"""Train row/column context contrasts while retaining production word decisions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from hanglish.models.semantic_ranker import SemanticRanker, encode_pairs
from scripts.train_semantic_ranker import _optimizer, _set_trainable_layers
from scripts.train_full_context_word_ranker import prepare as prepare_retention
from scripts.evaluate_ensemble_regression import file_sha256, rss_bytes, verify_pins

PARENT = ROOT / 'checkpoints/semantic_ranker'
AUDIT = ROOT / '.omx/artifacts/semantic-context-learning-audit-v1'
AUDIT_SHA = '4e72d39119512282017affd43a17adf1f349fd92aad9a9be4277a8b7633278e4'
RECIPE = dict(seed=20260920, epochs=2, top_layers=2, batch_groups=4,
              head_lr=5e-5, encoder_lr=1e-5, weight_decay=.01,
              base_weight=.15, temperature=2., column_weight=.5)


def key(text):
    return hashlib.sha256(('context-contrast-v1\0' + text).encode()).hexdigest()


def semantic_split(rows):
    by_target = {}
    for row in rows:
        by_target.setdefault((row['input'], row['expected'][0]), []).append(row)
    fit, hold = [], []
    if len(by_target) != 20:
        raise ValueError('expected20 TRAIN senses')
    for group in by_target.values():
        ordered = sorted(group, key=lambda r: key(r['context']))
        if len(ordered) != 12:
            raise ValueError('expected12 contexts per sense')
        fit.extend(ordered[:8]); hold.extend(ordered[8:])
    return fit, hold


def prepare():
    if file_sha256(AUDIT / 'report.json') != AUDIT_SHA:
        raise ValueError('audit report changed')
    audit = json.loads((AUDIT / 'report.json').read_text())
    if file_sha256(AUDIT / 'capture.jsonl') != audit['capture_sha256']:
        raise ValueError('audit capture changed')
    records = [r for r in map(json.loads, (AUDIT / 'capture.jsonl').read_text().splitlines()) if r['arm'] == 'parent']
    fit, hold = semantic_split(records)
    pairs = []
    for row in fit:
        donor = min((r for r in fit if r['input'] == row['input'] and r['expected'] != row['expected']),
                    key=lambda r: key(row['id'] + '\0' + r['id']))
        candidates = [row['expected'][0], donor['expected'][0]]
        scores = {o['text']: o['score'] for o in row['options']}
        if any(c not in row['own']['calls'][0]['candidates'] for c in candidates):
            raise ValueError('contrast candidate not in actual runtime pool')
        pairs.append({'kind': 'pair', 'id': row['id'], 'donor_id': donor['id'],
                      'contexts': [row['own']['calls'][0]['contexts'][0], donor['own']['calls'][0]['contexts'][0]],
                      'candidates': candidates, 'base_scores': [scores[c] for c in candidates]})
    pools, pins = prepare_retention()
    regular = [dict(r, kind='retention') for r in
               pools['natural_train:fit'][:512] + pools['retention_train:fit'][:256]]
    semantic_hold = []
    for row in hold:
        call = row['own']['calls'][0]
        options = {o['text']: o['score'] for o in row['options']}
        semantic_hold.append({'id': row['id'], 'context': call['contexts'][0],
                              'candidates': call['candidates'], 'expected': row['expected'],
                              'base_scores': [options[c] for c in call['candidates']]})
    evaluation = {'semantic_hold': semantic_hold, 'natural_hold': pools['natural_train:hold'],
                  'retention_hold': pools['retention_train:hold']}
    fit_ids = {r['id'] for r in fit}
    if fit_ids & {r['id'] for r in hold} or any(p['donor_id'] not in fit_ids for p in pairs):
        raise ValueError('semantic context split leakage')
    pins.update({str((AUDIT / 'report.json').relative_to(ROOT)): AUDIT_SHA,
                 str((AUDIT / 'capture.jsonl').relative_to(ROOT)): audit['capture_sha256']})
    return pairs + regular, evaluation, pins


def flatten(units):
    contexts, candidates = [], []
    for unit in units:
        if unit['kind'] == 'pair':
            contexts.extend([context for context in unit['contexts'] for _ in unit['candidates']])
            candidates.extend(unit['candidates'] * 2)
        else:
            contexts.extend([unit['context']] * len(unit['candidates']))
            candidates.extend(unit['candidates'])
    return contexts, candidates


def objective(logits, units):
    losses, cursor = [], 0
    for unit in units:
        if unit['kind'] == 'pair':
            matrix = logits[cursor:cursor+4].reshape(2, 2)
            matrix = matrix + logits.new_tensor(unit['base_scores']) * RECIPE['base_weight']
            target = torch.tensor([0, 1], device=logits.device)
            losses.append(torch.nn.functional.cross_entropy(matrix, target)
                          + RECIPE['column_weight'] * torch.nn.functional.cross_entropy(matrix.T, target))
            cursor += 4
        else:
            count = len(unit['candidates'])
            scores = logits[cursor:cursor+count] + logits.new_tensor(unit['base_scores']) * RECIPE['base_weight']
            teacher = logits.new_tensor(unit['teacher_scores']) + logits.new_tensor(unit['base_scores']) * RECIPE['base_weight']
            temperature = RECIPE['temperature']
            loss = temperature**2 * torch.nn.functional.kl_div(
                torch.log_softmax(scores / temperature, 0), torch.softmax(teacher / temperature, 0), reduction='sum')
            positives = [i for i, c in enumerate(unit['candidates']) if c in unit['expected']]
            if positives:
                loss = loss + torch.logsumexp(scores, 0) - torch.logsumexp(scores[positives], 0)
            losses.append(loss)
            cursor += count
    if cursor != len(logits):
        raise ValueError('loss did not consume all candidate logits')
    return torch.stack(losses).mean()


def evaluate(ranker, pools, guard):
    ranker.model.eval()
    result = {}
    for role, rows in pools.items():
        predictions = []
        for row in rows:
            scores = ranker.score_pairs([row['context']] * len(row['candidates']), row['candidates'])
            adjusted = [s + RECIPE['base_weight'] * b for s, b in zip(scores, row['base_scores'])]
            chosen = min(range(len(scores)), key=lambda i: (-adjusted[i], row['candidates'][i]))
            predictions.append({'id': row['id'], 'scores': scores, 'result': row['candidates'][chosen],
                                'correct': row['candidates'][chosen] in row['expected']})
            guard()
        result[role] = {'rows': len(rows), 'correct': sum(r['correct'] for r in predictions), 'predictions': predictions}
    return result


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    def guard():
        if rss_bytes() > 3 * 1024**3 or time.monotonic() - started > 1200:
            raise RuntimeError('training exceeded3GiB/1200seconds')
    try:
        torch.set_num_threads(2); torch.set_num_interop_threads(1)
        torch.manual_seed(RECIPE['seed']); random.seed(RECIPE['seed'])
        units, pools, pins = prepare()
        files = [Path(__file__), ROOT / 'scripts/train_full_context_word_ranker.py',
                 ROOT / 'scripts/train_semantic_ranker.py', ROOT / 'hanglish/models/semantic_ranker.py',
                 ROOT / 'scripts/evaluate_ensemble_regression.py', *PARENT.iterdir()]
        pins.update({str(p.relative_to(ROOT)): file_sha256(p) for p in files if p.is_file()})
        verify_pins(pins)
        selection = {'fit': units, 'evaluation': pools}
        (output / 'selection.json').write_text(json.dumps(selection, ensure_ascii=False) + '\n')
        ranker = SemanticRanker.load(PARENT)
        original = {name: tensor.detach().clone() for name, tensor in ranker.model.state_dict().items()}
        before = evaluate(ranker, pools, guard)
        configuration = _set_trainable_layers(ranker.model, RECIPE['top_layers'])
        trainable_names = {name for name, p in ranker.model.named_parameters() if p.requires_grad}
        optimizer = _optimizer(ranker.model, SimpleNamespace(**RECIPE), head_only=False)
        updates, losses = 0, []
        training_started = time.monotonic()
        with (output / 'updates.jsonl').open('x') as stream:
            for epoch in range(RECIPE['epochs']):
                order = list(range(len(units)))
                random.Random(RECIPE['seed'] + epoch).shuffle(order)
                ranker.model.train()
                for offset in range(0, len(order), RECIPE['batch_groups']):
                    batch = [units[i] for i in order[offset:offset+RECIPE['batch_groups']]]
                    contexts, candidates = flatten(batch)
                    encoded = encode_pairs(ranker.tokenizer, contexts, candidates, ranker.max_length)
                    optimizer.zero_grad(set_to_none=True)
                    loss = objective(ranker.model(**encoded).logits.reshape(-1), batch)
                    if not torch.isfinite(loss):
                        raise ValueError('non-finite training loss')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(ranker.model.parameters(), 1.)
                    optimizer.step(); updates += 1
                    value = float(loss.detach()); losses.append(value)
                    stream.write(json.dumps({'epoch': epoch, 'update': updates, 'ids': [u['id'] for u in batch], 'loss': value}) + '\n')
                    if updates == 3:
                        estimated = (time.monotonic() - training_started) / updates * RECIPE['epochs'] * ((len(units)+3)//4)
                        resource = {'updates': 3, 'peak_rss_bytes': rss_bytes(), 'estimated_training_s': estimated}
                        (output / 'resource-pilot.json').write_text(json.dumps(resource, indent=2))
                        if estimated > 900:
                            raise RuntimeError('resource pilot exceeds estimated900seconds')
                    guard()
                stream.flush()
                print(json.dumps({'epoch': epoch+1, 'updates': updates, 'elapsed_s': time.monotonic()-started}), flush=True)
        ranker.model.eval()
        changed = [name for name, value in ranker.model.state_dict().items() if not torch.equal(original[name], value)]
        if not changed or any(name not in trainable_names for name in changed):
            raise ValueError('changed weights do not match trainable scope')
        checkpoint = output / 'checkpoint'
        ranker.model.save_pretrained(checkpoint, safe_serialization=True)
        ranker.tokenizer.save_pretrained(checkpoint)
        hashes = {p.name: file_sha256(p) for p in checkpoint.iterdir() if p.is_file()}
        metadata = {'format_version': 1, 'max_length': ranker.max_length,
                    'inference_batch_size': ranker.batch_size, 'artifact_hashes': hashes,
                    'experimental': True, 'recipe': RECIPE, 'source_pins': pins}
        (checkpoint / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        after = evaluate(ranker, pools, guard)
        loaded = SemanticRanker.load(checkpoint)
        replay = evaluate(loaded, pools, guard)
        if after != replay:
            raise ValueError('saved checkpoint replay differs')
        verify_pins(pins)
        metrics = {}
        for role in pools:
            prior, new = before[role]['predictions'], after[role]['predictions']
            metrics[role] = {'rows': len(prior), 'before': before[role]['correct'], 'after': after[role]['correct'],
                             'fix_ids': [b['id'] for a, b in zip(prior, new) if not a['correct'] and b['correct']],
                             'loss_ids': [b['id'] for a, b in zip(prior, new) if a['correct'] and not b['correct']]}
        advance = (metrics['semantic_hold']['after'] > metrics['semantic_hold']['before']
                   and metrics['natural_hold']['after'] >= metrics['natural_hold']['before']
                   and metrics['retention_hold']['after'] == 128)
        predictions = {'before': before, 'after': after}
        (output / 'predictions.json').write_text(json.dumps(predictions, ensure_ascii=False) + '\n')
        report = {'schema': 'counterfactual-context-training-v1', 'completed': True, 'training_run': True,
                  'production_changed': False, 'new_generalization': False, 'advance_to_broader_evaluation': advance,
                  'recipe': RECIPE, 'configuration': configuration, 'updates': updates, 'fit_units': len(units),
                  'changed_parameter_names': changed, 'unchanged_tensor_count': len(original)-len(changed),
                  'mean_loss': sum(losses)/len(losses), 'metrics': metrics, 'reload_replay_equal': True,
                  'pins': pins, 'selection_sha256': file_sha256(output/'selection.json'),
                  'predictions_sha256': file_sha256(output/'predictions.json'),
                  'updates_sha256': file_sha256(output/'updates.jsonl'),
                  'checkpoint_sha256': file_sha256(checkpoint/'model.safetensors'),
                  'metadata_sha256': file_sha256(checkpoint/'metadata.json'),
                  'elapsed_s': time.monotonic()-started, 'peak_rss_bytes': rss_bytes()}
        (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        print(json.dumps({k: report[k] for k in ('completed','advance_to_broader_evaluation','updates','metrics','elapsed_s','peak_rss_bytes')}, ensure_ascii=False), flush=True)
    except BaseException as exc:
        (output/'failure.json').write_text(json.dumps({'completed': False, 'error': repr(exc)}, indent=2))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
