"""One bounded continuation fit on pinned natural and retained original candidates."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
from hanglish.models.sentence_ranker import SentenceRanker, encode_sentence_pairs
from scripts.prepare_natural_ranker import validate_prepared, frozen_hashes, file_sha256, OUTPUT
from scripts.train_sentence_ranker import SentenceExample, _loss, _optimizer, _set_trainable_layers

PARENT = ROOT / 'checkpoints/sentence_ranker'
PARENT_WEIGHT = 'b146dcbed6d9a37b8370e0c2d7d63d507be910c7e20e8c4157fdbe5d73d8607e'
RECIPE = {'seed': 20260914, 'epochs': 1, 'top_layers': 2,
          'head_lr': 5e-5, 'encoder_lr': 1e-5, 'weight_decay': .01,
          'group_batch': 4, 'candidate_limit': 8, 'natural_retention_ratio': '1:1',
          'rss_limit_bytes': 3 * 1024**3, 'training_seconds_limit': 1800}
HEAD_RECIPE = {**RECIPE, 'name': 'distill-head-v2', 'top_layers': 0,
               'head_lr': 5e-5, 'distillation_weight': 1., 'temperature': 2.}
TOP2_DISTILL_RECIPE = {**HEAD_RECIPE, 'name': 'distill-top2-v3', 'top_layers': 2}


def rss_bytes():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == 'darwin' else rss * 1024


def encoder_hash(model):
    digest = hashlib.sha256()
    for name, parameter in model.base_model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def distillation_loss(model, tokenizer, examples, args, teacher_scores, recipe):
    sentences, contexts, groups = [], [], []
    for example in examples:
        start = len(sentences)
        sentences.extend(example.candidates)
        contexts.extend([example.context] * len(example.candidates))
        groups.append((start, len(sentences), example))
    encoded = encode_sentence_pairs(tokenizer, sentences, contexts, args.max_length)
    scores = model(**{key: value.to(args.device) for key, value in encoded.items()}).logits.reshape(-1)
    losses = []
    for start, end, example in groups:
        group = scores[start:end]
        positive = torch.tensor(example.positives, device=group.device)
        loss = torch.logsumexp(group, 0) - torch.logsumexp(group[positive], 0)
        if example.category == 'retention':
            teacher = torch.tensor(teacher_scores[example.case_id], device=group.device)
            if teacher.shape != group.shape:
                raise ValueError('teacher candidate order/length mismatch')
            temperature = recipe['temperature']
            divergence = torch.nn.functional.kl_div(
                torch.log_softmax(group / temperature, dim=0),
                torch.softmax(teacher / temperature, dim=0), reduction='sum')
            loss = loss + recipe['distillation_weight'] * temperature**2 * divergence
        losses.append(loss)
    return torch.stack(losses).mean()


def bounded_example(row, limit=8):
    positives = row['positive_indices']
    if not positives or len(row['candidates']) < 2:
        return None
    selected = list(positives)
    selected.extend(i for i in range(len(row['candidates'])) if i not in positives)
    selected = selected[:max(limit, len(positives))]
    return SentenceExample(row['context'], tuple(row['candidates'][i] for i in selected),
                           tuple(i for i, original in enumerate(selected) if original in positives),
                           row['id'], row['origin'])


def balanced_schedule(natural, retention, seed):
    if not natural or not retention:
        raise ValueError('both training sources need covered rankable examples')
    rng = random.Random(seed)
    left, right = list(natural), list(retention)
    rng.shuffle(left); rng.shuffle(right)
    # Retain every eligible example; cycle the smaller source, never dev rows.
    return [item for i in range(max(len(left), len(right)))
            for item in (left[i % len(left)], right[i % len(right)])]


def validate_mixed_split(data):
    training = data['natural_train'] + data['retention_train']
    dev_targets = {text for row in data['natural_dev'] for text in row['expected']}
    overlaps = [row['id'] for row in training if any(
        target in row['context'] or any(target in gold for gold in row['expected'])
        for target in dev_targets)]
    if overlaps:
        raise ValueError('natural dev target appears in mixed training text/context')
    return {'dev_target_in_mixed_train_overlap_rows': 0}


def evaluate_runtime_pool(ranker, rows):
    correct = covered = fallbacks = 0
    outcomes = []
    for row in rows:
        candidates = row['candidates']
        # Same sentence-stage fallback as the production hybrid converter.
        can_rank = (len(row['input'].split()) > 1 and len(candidates) > 1
                    and all(ranker.can_score(c) for c in candidates))
        if can_rank:
            scores = ranker.score_pairs([row['context']] * len(candidates), candidates)
            choice = min(range(len(candidates)), key=lambda i: (-scores[i], candidates[i]))
        else:
            choice = 0
            fallbacks += 1
        hit = choice in row['positive_indices']
        correct += hit
        covered += bool(row['positive_indices'])
        outcomes.append({'id': row['id'], 'correct': hit, 'result': candidates[choice],
                         'covered': bool(row['positive_indices']), 'fallback': not can_rank})
    return {'rows': len(rows), 'correct': correct, 'covered': covered,
            'fallbacks': fallbacks, 'outcomes': outcomes}


def train(args):
    recipe = {'natural-v1': RECIPE, 'distill-head-v2': HEAD_RECIPE,
              'distill-top2-v3': TOP2_DISTILL_RECIPE}[args.recipe]
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest, data = validate_prepared(args.prepared, args.manifest_sha256)
    mixed_guard = validate_mixed_split(data)
    if file_sha256(PARENT / 'model.safetensors') != PARENT_WEIGHT:
        raise ValueError('parent weights differ from pinned deployed sentence ranker')
    before = frozen_hashes()
    trainer_hash = file_sha256(Path(__file__))
    torch.set_num_threads(2)
    random.seed(recipe['seed']); torch.manual_seed(recipe['seed'])
    ranker = SentenceRanker.load(PARENT)
    ranker.model.to('cpu')
    args.output.mkdir(parents=True)
    log_path = args.output / 'training-events.jsonl'
    # Logs stay outside the model bundle accepted by the safe loader.
    bundle = args.output / 'checkpoint'
    def log(event, **values):
        record = {'event': event, **values}
        with log_path.open('a') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(json.dumps(record, ensure_ascii=False), flush=True)
    initial = evaluate_runtime_pool(ranker, data['natural_dev'])
    log('initial_natural_dev', **{k: v for k, v in initial.items() if k != 'outcomes'})
    groups = {}
    eligibility = {}
    for role in ('natural_train', 'retention_train'):
        examples = []
        counts = {'rows': len(data[role]), 'uncovered_or_single': 0, 'too_long': 0}
        for row in data[role]:
            example = bounded_example(row, recipe['candidate_limit'])
            if example is None:
                counts['uncovered_or_single'] += 1
            elif not all(ranker.can_score(c) for c in row['candidates']):
                counts['too_long'] += 1
            else:
                examples.append(example)
        counts['eligible'] = len(examples)
        eligibility[role] = counts
        groups[role] = examples
    schedule = balanced_schedule(groups['natural_train'], groups['retention_train'], recipe['seed'])
    options = SimpleNamespace(device='cpu', max_length=ranker.max_length,
                              head_lr=recipe['head_lr'], encoder_lr=recipe['encoder_lr'],
                              weight_decay=recipe['weight_decay'])
    teacher_scores = {}
    encoder_before = None
    if recipe.get('distillation_weight', 0) > 0:
        for example in groups['retention_train']:
            teacher_scores[example.case_id] = ranker.score_pairs(
                [example.context] * len(example.candidates), list(example.candidates))
        (args.output / 'teacher-scores.json').write_text(json.dumps(teacher_scores, sort_keys=True) + '\n')
    if recipe['top_layers'] == 0:
        encoder_before = encoder_hash(ranker.model)
        for name, parameter in ranker.model.named_parameters():
            parameter.requires_grad = not name.startswith(ranker.model.base_model_prefix + '.')
        layer_info = {'unfrozen_encoder_layers': 0,
                      'trainable_parameters': sum(p.numel() for p in ranker.model.parameters() if p.requires_grad),
                      'total_parameters': sum(p.numel() for p in ranker.model.parameters())}
    else:
        layer_info = _set_trainable_layers(ranker.model, recipe['top_layers'])
    optimizer = _optimizer(ranker.model, options, head_only=recipe['top_layers'] == 0)
    log('training_start', recipe=recipe, eligibility=eligibility,
        scheduled_examples=len(schedule), layer_info=layer_info)
    losses = []
    started = time.monotonic()
    ranker.model.train()
    for start in range(0, len(schedule), recipe['group_batch']):
        batch = schedule[start:start + recipe['group_batch']]
        optimizer.zero_grad(set_to_none=True)
        loss = (distillation_loss(ranker.model, ranker.tokenizer, batch, options, teacher_scores, recipe)
                if teacher_scores else _loss(ranker.model, ranker.tokenizer, batch, options))
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('nonfinite training loss')
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(ranker.model.parameters(), 1.)
        if not bool(torch.isfinite(gradient_norm)):
            raise RuntimeError('nonfinite gradients')
        optimizer.step()
        losses.append(float(loss.detach()))
        elapsed = time.monotonic() - started
        if rss_bytes() > recipe['rss_limit_bytes'] or elapsed > recipe['training_seconds_limit']:
            raise RuntimeError('bounded training resource gate exceeded')
        if len(losses) == 5 or len(losses) % 16 == 0:
            log('training_step', step=len(losses), loss=losses[-1],
                elapsed_seconds=round(elapsed, 2), peak_rss_bytes=rss_bytes())
    training_seconds = time.monotonic() - started
    encoder_after = encoder_hash(ranker.model) if encoder_before is not None else None
    if encoder_before is not None and encoder_before != encoder_after:
        raise RuntimeError('frozen encoder weights changed')
    ranker.model.eval()
    final = evaluate_runtime_pool(ranker, data['natural_dev'])
    log('final_natural_dev', **{k: v for k, v in final.items() if k != 'outcomes'})
    validate_prepared(args.prepared, args.manifest_sha256)
    if before != frozen_hashes() or trainer_hash != file_sha256(Path(__file__)):
        raise RuntimeError('source code or model artifacts changed during training')
    bundle.mkdir()
    ranker.model.save_pretrained(bundle, safe_serialization=True)
    ranker.tokenizer.save_pretrained(bundle)
    artifact_hashes = {p.name: file_sha256(p) for p in bundle.iterdir() if p.is_file()}
    parent_metadata = ranker.metadata
    metadata = {key: parent_metadata[key] for key in (
        'base_model', 'base_revision', 'base_source', 'source_hashes', 'training_data_guard',
        'candidate_generator') if key in parent_metadata}
    metadata.update(format_version=1, created_at=datetime.now(timezone.utc).isoformat(),
        max_length=ranker.max_length, inference_batch_size=ranker.batch_size,
        artifact_hashes=artifact_hashes, test_data_read=False,
        parent_checkpoint_sha256=PARENT_WEIGHT,
        natural_preparation={'path': str(args.prepared), 'manifest_sha256': args.manifest_sha256,
                             'source_selection': manifest['limits'], 'output_hashes': manifest['outputs']},
        augmentation_data_guard={'kind': 'natural-context-pilot-v1',
                                 'manifest_sha256': args.manifest_sha256, **mixed_guard},
        frozen_source_hashes=before, trainer_sha256=trainer_hash,
        training=recipe, training_eligibility=eligibility, trainable_layers=layer_info,
        encoder_hash_before=encoder_before, encoder_hash_after=encoder_after,
        teacher_scores_sha256=file_sha256(args.output / 'teacher-scores.json') if teacher_scores else None,
        natural_dev_before={k: v for k, v in initial.items() if k != 'outcomes'},
        natural_dev_after={k: v for k, v in final.items() if k != 'outcomes'})
    (bundle / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    loaded = SentenceRanker.load(bundle)
    if loaded.score_pairs([''], ['검증 문장']) != ranker.score_pairs([''], ['검증 문장']):
        raise RuntimeError('saved model differs from in-memory inference')
    old = {r['id']: r for r in initial['outcomes']}
    report = {'recipe': recipe, 'prepared_manifest_sha256': args.manifest_sha256,
        'initial': initial, 'final': final, 'eligibility': eligibility,
        'corrections': [r['id'] for r in final['outcomes'] if r['correct'] and not old[r['id']]['correct']],
        'regressions': [r['id'] for r in final['outcomes'] if not r['correct'] and old[r['id']]['correct']],
        'optimizer_steps': len(losses), 'mean_train_loss': sum(losses) / len(losses),
        'training_seconds': training_seconds, 'peak_rss_bytes': rss_bytes(),
        'encoder_hash_before': encoder_before, 'encoder_hash_after': encoder_after,
        'checkpoint': str(bundle), 'checkpoint_sha256': artifact_hashes['model.safetensors'],
        'no_gold_injection': True, 'test_data_read': False, 'deployed': False,
        'next_gate': 'Actual hybrid original development 140/60/40, only if natural dev improves; then exposed final regression.'}
    (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    log('complete', checkpoint=str(bundle), before=initial['correct'], after=final['correct'],
        total=final['rows'], peak_rss_bytes=rss_bytes(), training_seconds=training_seconds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, default=OUTPUT)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--recipe', choices=('natural-v1', 'distill-head-v2', 'distill-top2-v3'), default='natural-v1')
    parser.add_argument('--output', type=Path, default=ROOT / '.omx/artifacts/natural-sentence-ranker-v1')
    train(parser.parse_args())
