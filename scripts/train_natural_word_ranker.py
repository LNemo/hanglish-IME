"""Bounded word-ranker continuation on actual, source-only runtime features."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
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
from scripts.prepare_natural_word_ranker import OUTPUT, validate_word_data, source_hashes, file_sha256
from scripts.train_natural_sentence_ranker import balanced_schedule, validate_mixed_split, rss_bytes
from scripts.train_sentence_ranker import _optimizer, _set_trainable_layers

PARENT = ROOT / 'checkpoints/semantic_ranker'
PARENT_WEIGHT = '54ca7d24b2bf17d1ba8f5181476ce91032acc94149a2baf401fe47e859f2907c'
RECIPE = {'name': 'runtime-context-top2-distill-v1', 'seed': 20260914, 'epochs': 1,
          'top_layers': 2, 'head_lr': 5e-5, 'encoder_lr': 1e-5, 'weight_decay': .01,
          'group_batch': 4, 'candidate_limit': 8, 'source_group_limit': 2048,
          'selection_salt': 'natural-word-source-groups-v1', 'base_weight': .15,
          'distillation_weight': 1., 'temperature': 2.,
          'rss_limit_bytes': 3 * 1024**3, 'training_seconds_limit': 1800}


@dataclass(frozen=True)
class WordExample:
    context: str
    candidates: tuple[str, ...]
    positives: tuple[int, ...]
    base_scores: tuple[float, ...]
    teacher_scores: tuple[float, ...]
    case_id: str
    origin: str


def select_groups(rows, recipe):
    groups = [(row['origin'], group) for row in rows for group in row['groups']]
    # Select before consulting coverage/labels, so failures cannot change the sample.
    return sorted(groups, key=lambda item: hashlib.sha256(
        (recipe['selection_salt'] + ':' + item[1]['id']).encode()).hexdigest())[:recipe['source_group_limit']]


def bounded_example(origin, group, recipe):
    positives = group['positive_indices']
    if not positives or len(group['candidates']) < 2:
        return None
    negatives = sorted((i for i in range(len(group['candidates'])) if i not in positives),
                       key=lambda i: (-(group['teacher_scores'][i] + recipe['base_weight'] * group['base_scores'][i]),
                                      group['candidates'][i]))
    chosen = (list(positives) + negatives)[:max(recipe['candidate_limit'], len(positives))]
    return WordExample(group['context'], tuple(group['candidates'][i] for i in chosen),
                       tuple(i for i, original in enumerate(chosen) if original in positives),
                       tuple(group['base_scores'][i] for i in chosen),
                       tuple(group['teacher_scores'][i] for i in chosen), group['id'], origin)


def word_loss(model, tokenizer, examples, max_length, recipe):
    contexts, candidates, groups = [], [], []
    for example in examples:
        start = len(candidates)
        candidates.extend(example.candidates)
        contexts.extend([example.context] * len(example.candidates))
        groups.append((start, len(candidates), example))
    scores = model(**encode_pairs(tokenizer, contexts, candidates, max_length)).logits.reshape(-1)
    losses = []
    for start, end, example in groups:
        bias = scores.new_tensor(example.base_scores) * recipe['base_weight']
        group = scores[start:end] + bias
        positive = torch.tensor(example.positives, device=group.device)
        loss = torch.logsumexp(group, 0) - torch.logsumexp(group[positive], 0)
        if example.origin == 'retention':
            teacher = group.new_tensor(example.teacher_scores) + bias
            temperature = recipe['temperature']
            loss = loss + recipe['distillation_weight'] * temperature**2 * torch.nn.functional.kl_div(
                torch.log_softmax(group / temperature, dim=0),
                torch.softmax(teacher / temperature, dim=0), reduction='sum')
        losses.append(loss)
    return torch.stack(losses).mean()


def evaluate_groups(ranker, rows, *, check_teacher=False):
    outcomes = []
    max_teacher_delta = 0.
    scored_groups = 0
    for row in rows:
        contexts, candidates = [], []
        for group in row['groups']:
            contexts.extend([group['context']] * len(group['candidates']))
            candidates.extend(group['candidates'])
        # Preserve runtime batch boundaries, even for effectively unambiguous groups.
        scores = ranker.score_pairs(contexts, candidates)
        cursor = 0
        for group in row['groups']:
            n = len(group['candidates']); scored_groups += 1
            current = scores[cursor:cursor+n]; cursor += n
            if check_teacher:
                max_teacher_delta = max(max_teacher_delta, max(abs(a-b) for a,b in zip(current, group['teacher_scores'])))
            if n < 2:
                continue
            chosen = min(range(n), key=lambda i: (-(current[i] + RECIPE['base_weight'] * group['base_scores'][i]),
                                                  group['candidates'][i]))
            outcomes.append({'id': group['id'], 'correct': chosen in group['positive_indices'],
                             'covered': bool(group['positive_indices']), 'result': group['candidates'][chosen]})
    if check_teacher and max_teacher_delta > 1e-5:
        raise ValueError('captured teacher differs from actual word-ranker replay')
    return {'cases': len(rows), 'scored_groups': scored_groups, 'ambiguous_groups': len(outcomes),
            'covered': sum(r['covered'] for r in outcomes), 'correct': sum(r['correct'] for r in outcomes),
            'max_teacher_delta': max_teacher_delta if check_teacher else None, 'outcomes': outcomes}


def training_hashes():
    result = source_hashes()
    for relative in ('scripts/train_natural_word_ranker.py', 'scripts/train_natural_sentence_ranker.py',
                     'scripts/train_sentence_ranker.py'):
        result[relative] = file_sha256(ROOT / relative)
    return result


def train(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest, data = validate_word_data(args.prepared, args.manifest_sha256)
    mixed_guard = validate_mixed_split(data)
    if file_sha256(PARENT / 'model.safetensors') != PARENT_WEIGHT:
        raise ValueError('word-ranker parent differs from explicit baseline')
    before = training_hashes()
    torch.set_num_threads(2)
    random.seed(RECIPE['seed']); torch.manual_seed(RECIPE['seed'])
    ranker = SemanticRanker.load(PARENT)
    args.output.mkdir(parents=True)
    def log(event, **values):
        message = json.dumps({'event': event, **values}, ensure_ascii=False)
        with (args.output / 'training-events.jsonl').open('a') as stream: stream.write(message + '\n')
        print(message, flush=True)
    initial = evaluate_groups(ranker, data['natural_dev'], check_teacher=True)
    log('initial_natural_dev', **{k:v for k,v in initial.items() if k != 'outcomes'})
    examples, eligibility = {}, {}
    for role in ('natural_train', 'retention_train'):
        selected = select_groups(data[role], RECIPE)
        eligible = [example for origin, group in selected
                    if (example := bounded_example(origin, group, RECIPE)) is not None]
        examples[role] = eligible
        eligibility[role] = {'source_cases': len(data[role]), 'total_groups': sum(len(r['groups']) for r in data[role]),
                             'selected_before_labels': len(selected), 'eligible': len(eligible)}
    schedule = balanced_schedule(examples['natural_train'], examples['retention_train'], RECIPE['seed'])
    layer_info = _set_trainable_layers(ranker.model, RECIPE['top_layers'])
    optimizer = _optimizer(ranker.model, SimpleNamespace(**RECIPE), head_only=False)
    log('training_start', recipe=RECIPE, eligibility=eligibility, scheduled_examples=len(schedule), layer_info=layer_info)
    started = time.monotonic(); losses = []
    ranker.model.train()
    for start in range(0, len(schedule), RECIPE['group_batch']):
        optimizer.zero_grad(set_to_none=True)
        loss = word_loss(ranker.model, ranker.tokenizer, schedule[start:start+RECIPE['group_batch']], ranker.max_length, RECIPE)
        if not bool(torch.isfinite(loss)): raise RuntimeError('nonfinite training loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(ranker.model.parameters(), 1.)
        if not bool(torch.isfinite(norm)): raise RuntimeError('nonfinite gradients')
        optimizer.step(); losses.append(float(loss.detach()))
        if rss_bytes() > RECIPE['rss_limit_bytes'] or time.monotonic()-started > RECIPE['training_seconds_limit']:
            raise RuntimeError('word training resource gate exceeded')
        if len(losses) % 32 == 0:
            log('training_step', step=len(losses), loss=losses[-1], elapsed_seconds=round(time.monotonic()-started, 2), peak_rss_bytes=rss_bytes())
    training_seconds = time.monotonic()-started
    ranker.model.eval()
    final = evaluate_groups(ranker, data['natural_dev'])
    validate_word_data(args.prepared, args.manifest_sha256)
    if before != training_hashes(): raise RuntimeError('word training sources changed')
    bundle = args.output / 'checkpoint'; bundle.mkdir()
    ranker.model.save_pretrained(bundle, safe_serialization=True)
    ranker.tokenizer.save_pretrained(bundle)
    hashes = {p.name: file_sha256(p) for p in bundle.iterdir() if p.is_file()}
    metadata = {k:ranker.metadata[k] for k in ('base_model', 'base_revision', 'base_source', 'source_hashes',
                                               'training_data_guard', 'candidate_generator') if k in ranker.metadata}
    metadata.update(format_version=1, created_at=datetime.now(timezone.utc).isoformat(),
        artifact_hashes=hashes, max_length=ranker.max_length, inference_batch_size=ranker.batch_size,
        parent_checkpoint_sha256=PARENT_WEIGHT, parent_runtime_version=manifest['runtime_version'],
        natural_word_preparation={'path': str(args.prepared), 'manifest_sha256': args.manifest_sha256},
        augmentation_data_guard={'kind': 'natural-word-runtime-v1', **mixed_guard},
        frozen_source_hashes=before, trainer_sha256=before['scripts/train_natural_word_ranker.py'],
        training=RECIPE, training_eligibility=eligibility, trainable_layers=layer_info, test_data_read=False)
    (bundle / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    loaded = SemanticRanker.load(bundle)
    if loaded.score_pairs(['[MASK]'], ['검증']) != ranker.score_pairs(['[MASK]'], ['검증']):
        raise RuntimeError('saved word model differs from in-memory inference')
    old = {r['id']:r for r in initial['outcomes']}
    report = {'recipe': RECIPE, 'initial': initial, 'final': final, 'eligibility': eligibility,
        'corrections': [r['id'] for r in final['outcomes'] if r['correct'] and not old[r['id']]['correct']],
        'regressions': [r['id'] for r in final['outcomes'] if not r['correct'] and old[r['id']]['correct']],
        'optimizer_steps': len(losses), 'mean_train_loss': sum(losses)/len(losses),
        'training_seconds': training_seconds, 'peak_rss_bytes': rss_bytes(),
        'checkpoint_sha256': hashes['model.safetensors'], 'checkpoint': str(bundle),
        'test_data_read': False, 'deployed': False,
        'next_gate': 'Actual hybrid natural128 and original dev140/60/40; only then exposed final regression.'}
    (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    (args.output / 'trainer-source.py').write_bytes(Path(__file__).read_bytes())
    log('complete', before=initial['correct'], after=final['correct'], total=final['ambiguous_groups'],
        training_seconds=training_seconds, peak_rss_bytes=rss_bytes())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, default=OUTPUT)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, default=ROOT / '.omx/artifacts/natural-word-ranker-v1')
    train(parser.parse_args())
