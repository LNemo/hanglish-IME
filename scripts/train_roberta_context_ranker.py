"""Supervised local RoBERTa ranking on freshly captured runtime word contexts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version as package_version
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
from scripts.prepare_context_training import OUTPUT, validate_context_data, source_hashes
from scripts.probe_roberta_ranker_resources import (
    load_cross_encoder, validate_source, sha256, memory, synchronize, REVISION)
from scripts.train_natural_word_ranker import bounded_example, select_groups, evaluate_groups
from scripts.train_natural_sentence_ranker import balanced_schedule
from scripts.train_sentence_ranker import _set_trainable_layers, _optimizer

RECIPE = {'name': 'roberta-supervised-word-top2-v1', 'seed': 20260914,
          'epochs': 1, 'head_warmup_steps': 128, 'top_layers': 2,
          'head_lr': 5e-5, 'encoder_lr': 1e-5, 'weight_decay': .01,
          'group_batch': 2, 'candidate_limit': 8, 'source_group_limit': 8192,
          'selection_salt': 'roberta-context-source-groups-v1',
          'base_weight': .15, 'distillation_weight': 1., 'temperature': 2.,
          'max_length': 96, 'attention_implementation': 'eager',
          'resource_limit_bytes': 3 * 1024**3, 'training_seconds_limit': 1800}


def training_hashes():
    result = source_hashes()
    for relative in ('scripts/train_roberta_context_ranker.py', 'scripts/probe_roberta_ranker_resources.py',
                     'scripts/train_natural_word_ranker.py', 'scripts/train_natural_sentence_ranker.py',
                     'scripts/train_sentence_ranker.py'):
        result[relative] = sha256(ROOT / relative)
    return result


def ranking_loss(scores, examples, recipe=RECIPE):
    losses, cursor = [], 0
    for example in examples:
        n = len(example.candidates)
        bias = scores.new_tensor(example.base_scores) * recipe['base_weight']
        group = scores[cursor:cursor+n] + bias
        cursor += n
        positive = torch.tensor(example.positives, device=group.device)
        loss = torch.logsumexp(group, 0) - torch.logsumexp(group[positive], 0)
        if example.origin == 'retention':
            teacher = group.new_tensor(example.teacher_scores) + bias
            temperature = recipe['temperature']
            loss = loss + recipe['distillation_weight'] * temperature**2 * torch.nn.functional.kl_div(
                torch.log_softmax(group / temperature, 0),
                torch.softmax(teacher / temperature, 0), reduction='sum')
        losses.append(loss)
    if cursor != len(scores):
        raise ValueError('candidate groups do not consume all logits')
    return torch.stack(losses).mean()


def train(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest, data = validate_context_data(args.prepared, args.manifest_sha256)
    before, base_sources = training_hashes(), validate_source()
    pilot = json.loads(args.resource_report.read_text())
    if (not pilot['passed'] or pilot['device'] != args.device or pilot['batch_size'] != 16
            or pilot['source_hashes'] != base_sources or pilot['attention_implementation'] != 'eager'
            or pilot['script_sha256'] != sha256(ROOT / 'scripts/probe_roberta_ranker_resources.py')):
        raise ValueError('matching resource pilot required before training')
    torch.set_num_threads(2)
    random.seed(RECIPE['seed']); torch.manual_seed(RECIPE['seed'])
    args.output.mkdir(parents=True)
    def log(event, **values):
        message = json.dumps({'event': event, **values}, ensure_ascii=False)
        with (args.output / 'training-events.jsonl').open('a') as stream:
            stream.write(message + '\n')
        print(message, flush=True)
    parent = SemanticRanker.load(ROOT / 'checkpoints/semantic_ranker')
    initial = evaluate_groups(parent, data['natural_dev'], check_teacher=True)
    del parent
    examples, eligibility = {}, {}
    for role in ('natural_train', 'retention_train'):
        selected = (select_groups(data[role], RECIPE) if role == 'natural_train' else
                    [(row['origin'], group) for row in data[role] for group in row['groups']])
        examples[role] = [example for origin, group in selected
                          if (example := bounded_example(origin, group, RECIPE)) is not None]
        if any(len(example.candidates) > RECIPE['candidate_limit'] for example in examples[role]):
            raise ValueError('positive candidate set exceeds measured batch bounds')
        eligibility[role] = {'source_rows': len(data[role]), 'selected_groups': len(selected),
                             'eligible_groups': len(examples[role])}
    schedule = balanced_schedule(examples['natural_train'], examples['retention_train'], RECIPE['seed'])
    model, tokenizer, loading = load_cross_encoder()
    model.to(args.device)
    layers = _set_trainable_layers(model, RECIPE['top_layers'])
    optimizer = _optimizer(model, SimpleNamespace(**RECIPE), head_only=False)
    log('training_start', recipe=RECIPE, eligibility=eligibility, layer_info=layers,
        scheduled_groups=len(schedule), initial_correct=initial['correct'])
    started = time.monotonic()
    losses = []
    peak = 0
    for phase in ('head_warmup', 'top2'):
        for parameter in model.base_model.parameters():
            parameter.requires_grad = False
        if phase == 'top2':
            _set_trainable_layers(model, RECIPE['top_layers'])
        phase_examples = (schedule[:RECIPE['head_warmup_steps'] * RECIPE['group_batch']]
                          if phase == 'head_warmup' else schedule)
        model.train()
        for start in range(0, len(phase_examples), RECIPE['group_batch']):
            batch = phase_examples[start:start+RECIPE['group_batch']]
            contexts = [e.context for e in batch for _ in e.candidates]
            candidates = [candidate for e in batch for candidate in e.candidates]
            encoded = encode_pairs(tokenizer, contexts, candidates, RECIPE['max_length'])
            encoded = tokenizer.pad(encoded, padding='max_length', max_length=96, return_tensors='pt')
            encoded = {key: value.to(args.device) for key, value in encoded.items()}
            optimizer.zero_grad(set_to_none=True)
            scores = model(**encoded).logits.reshape(-1)
            loss = ranking_loss(scores, batch)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('nonfinite ranking loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not bool(torch.isfinite(norm)):
                raise RuntimeError('nonfinite gradients')
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            current = memory(args.device)
            peak = max(peak, current['conservative_combined_bytes'])
            elapsed = time.monotonic() - started
            if peak > RECIPE['resource_limit_bytes'] or elapsed > RECIPE['training_seconds_limit']:
                raise RuntimeError(f'training resource bound exceeded: {current}, {elapsed}')
            if args.device == 'mps' and len(losses) % 64 == 0:
                torch.mps.empty_cache()
            if len(losses) % 64 == 0:
                log('training_step', phase=phase, step=len(losses), loss=losses[-1],
                    elapsed_seconds=round(elapsed, 2), memory=current)
    synchronize(args.device)
    seconds = time.monotonic() - started
    if seconds > RECIPE['training_seconds_limit']:
        raise RuntimeError('training duration exceeded cap after device synchronization')
    del optimizer, loss, scores, encoded
    model.to('cpu').eval()
    if args.device == 'mps':
        torch.mps.empty_cache()
    bundle = args.output / 'checkpoint'
    model.save_pretrained(bundle, safe_serialization=True)
    tokenizer.save_pretrained(bundle)
    hashes = {path.name: sha256(path) for path in bundle.iterdir() if path.is_file()}
    base_directory = ROOT / 'data/external/klue-roberta-small'
    source_metadata = json.loads((base_directory / 'SOURCE.json').read_text())
    license_metadata = {**source_metadata['license'],
        'attribution': 'KLUE-benchmark contributors; adapted from klue/roberta-small',
        'model_url': f'https://huggingface.co/klue/roberta-small/tree/{REVISION}',
        'text': (base_directory / source_metadata['license']['file']).read_text()}
    metadata = {'format_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
                'base_model': 'klue/roberta-small', 'base_revision': REVISION,
                'base_source': str(ROOT / 'data/external/klue-roberta-small'),
                'source_hashes': base_sources, 'artifact_hashes': hashes,
                'license': license_metadata,
                'library_versions': {name: package_version(name) for name in
                                     ('torch', 'transformers', 'tokenizers', 'safetensors')},
                'max_length': 96, 'inference_batch_size': 8,
                'parent_runtime_version': manifest['runtime_version'],
                'context_preparation': {'path': str(args.prepared), 'manifest_sha256': args.manifest_sha256},
                'frozen_source_hashes': before, 'training': RECIPE,
                'training_eligibility': eligibility, 'test_data_read': False}
    (bundle / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    # Evaluate exactly the safe loader and CPU path that production uses.
    in_memory = SemanticRanker(model, tokenizer)
    probe_before = in_memory.score_pairs(['주변 [MASK] 문맥'] * 2, ['검증', '검정'])
    loaded = SemanticRanker.load(bundle)
    probe_after = loaded.score_pairs(['주변 [MASK] 문맥'] * 2, ['검증', '검정'])
    delta = max(abs(a-b) for a, b in zip(probe_before, probe_after))
    if delta > 1e-4:
        raise RuntimeError('saved model loading differs from trained model')
    del model, in_memory
    final = evaluate_groups(loaded, data['natural_dev'])
    validate_context_data(args.prepared, args.manifest_sha256)
    if before != training_hashes() or base_sources != validate_source():
        raise RuntimeError('training inputs changed')
    old = {row['id']: row for row in initial['outcomes']}
    report = {'recipe': RECIPE, 'initial': initial, 'final': final, 'eligibility': eligibility,
              'corrections': [r['id'] for r in final['outcomes'] if r['correct'] and not old[r['id']]['correct']],
              'regressions': [r['id'] for r in final['outcomes'] if not r['correct'] and old[r['id']]['correct']],
              'device': args.device, 'optimizer_steps': len(losses), 'mean_loss': sum(losses)/len(losses),
              'training_seconds': seconds, 'peak_conservative_combined_bytes': peak,
              'checkpoint_sha256': hashes['model.safetensors'], 'save_load_score_max_delta': delta,
              'resource_report_sha256': sha256(args.resource_report), 'loading_info': loading,
              'test_data_read': False, 'deployed': False,
              'next_gate': 'Actual hybrid natural128 and original dev140/60/40 before any final regression.'}
    (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (args.output / 'trainer-source.py').write_bytes(Path(__file__).read_bytes())
    log('complete', initial=initial['correct'], final=final['correct'], training_seconds=seconds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, default=OUTPUT)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--resource-report', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'mps'), default='mps')
    parser.add_argument('--output', type=Path, required=True)
    train(parser.parse_args())
