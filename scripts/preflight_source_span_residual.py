"""Frozen-recipe FIT-only source-span residual learning/resource pilot."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCHEMA = 'source-span-residual-pilot-v1'
DATA = ROOT / '.omx/artifacts/source-correspondence-fit-v2'
SELECTION = ROOT / '.omx/artifacts/recombination-paired-training-v1/selection.json'
PLAN = ROOT / '.omx/plans/source-span-residual-pilot-v1.md'
PARENT = ROOT / 'checkpoints/sentence_ranker'
FIXED = {
    DATA / 'independent-verification.json': '449a8f286a60cc4a7c542612054c4ca762378caedfb0cbc9ca350801030f2de2',
    DATA / 'fit-paths.jsonl': 'f1cd66da23cacc6025e8aa2203944f664f0d74d916a0bde135cac65ef56afd8c',
    SELECTION: 'cbac8994a33d2a8b4d0f0c4596a1a07e206d6b75562c3416ff49877864829fd8',
}
PARENT_HASHES = {
    'config.json': '8703d582f1e16300ab44e501e4a131b15f0a133b7b74bbcfb51e7e53009fe05a',
    'metadata.json': '8a6a1f9736fab0173003751e70c5e7ef0fa139a0e773444986ede044e0e55d14',
    'model.safetensors': 'a68f3bd26733e9bc897a182f78b1decafefa3582fa82b6b2cfcaa34672412ffa',
    'special_tokens_map.json': '5d5b662e421ea9fac075174bb0688ee0d9431699900b90662acd44b2a350503a',
    'tokenizer.json': '5ee39bd824ba876371d68c5e958a8169c29b29060b8db19e6269403f3d3c7999',
    'tokenizer_config.json': '5a85ad076d4ae965ccff1888927079a924e3ffe401be44da855ff08a0ad0c45d',
    'vocab.txt': '6e886927dfcecd22029b1ba80c10a1374740259c1067fc3a28d964b7ae2d55a7',
}
LIMITS = {'whole_seconds': 600., 'update_seconds': 60.,
          'rss_bytes': 3 * 1024**3, 'swap_growth_bytes': 512 * 1024**2}
SEED = 20260921


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def select_cohort(eligible, count=16):
    require(count == 16, 'frozen pilot size is16')
    ordered = sorted(eligible, key=lambda r: (hashlib.sha256(r['id'].encode()).hexdigest(), r['id']))
    selected, components = [], set()
    for name, pool, quota in [
        ('multiple_paths', [r for r in ordered if r['multiple_paths']], 2),
        ('unresolved_ascii', [r for r in ordered if r['unresolved_ascii']], 4),
        ('long_source', sorted(ordered, key=lambda r: (-len(r['input']), r['id'])), 2),
        ('hash', ordered, 8),
    ]:
        accepted = 0
        for row in pool:
            if row['component_id'] in components:
                continue
            selected.append(dict(row, stratum=name)); components.add(row['component_id'])
            accepted += 1
            if accepted == quota:
                break
    for row in ordered:
        if len(selected) == count:
            break
        if row['component_id'] not in components:
            selected.append(dict(row, stratum='hash_fill')); components.add(row['component_id'])
    require(len(selected) == count, 'insufficient distinct FIT components')
    return selected


def prepare():
    from hanglish.core.source_correspondence import expand_path
    require(all(sha(p) == h for p, h in FIXED.items()), 'frozen FIT inputs changed')
    proof = read(DATA / 'independent-verification.json')
    require(proof['verified'] is True and proof['counts']['rows'] == 1642
            and proof['capture_sha256'] == FIXED[DATA / 'fit-paths.jsonl'], 'FIT authorization differs')
    selected = read(SELECTION)['rows']
    require(len(selected) == 1483 and len({r['id'] for r in selected}) == 1483, 'shared FIT size differs')
    by_id = {r['id']: r for r in selected}
    eligible = []; records = {}
    for record in rows(DATA / 'fit-paths.jsonl'):
        row = by_id.get(record['id'])
        if row is None:
            continue
        require(record['input'] == row['input'] and record['component_id'] == row['component_id'],
                'FIT path/label identity differs')
        arm = row['arms']['b_union']; by_text = defaultdict(list)
        for path in record['paths']:
            if path['text'] in arm['candidates']:
                by_text[path['text']].append(path)
        require(set(by_text) == set(arm['candidates']), 'selected candidate alignment missing')
        positives = arm['positive_indices']
        require(2 <= len(arm['candidates']) <= 8 and positives
                and len(set(positives)) == len(positives)
                and all(type(i) is int and 0 <= i < len(arm['candidates']) for i in positives)
                and len(positives) < len(arm['candidates']), 'FIT positive/negative domain differs')
        opaque = any(s.unresolved_ascii_output for variants in by_text.values()
                     for p in variants for s in expand_path(record, p).spans)
        eligible.append({'id': row['id'], 'input': row['input'], 'component_id': row['component_id'],
                         'candidates': arm['candidates'], 'positive_indices': positives,
                         'multiple_paths': any(len(v) > 1 for v in by_text.values()),
                         'unresolved_ascii': opaque})
        # Keep compact selected paths only; edge table is shared for this row.
        records[row['id']] = dict(record, paths=[p for variants in by_text.values() for p in variants])
    require(len(eligible) == 1483, 'eligible FIT projection incomplete')
    cohort = select_cohort(eligible)
    return cohort, {r['id']: records[r['id']] for r in cohort}


def tensor_hashes(state):
    return {n: hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for n, v in state.items()}


def train_loss(scores, positives):
    import torch
    return torch.logsumexp(scores, dim=0) - torch.logsumexp(scores[positives], dim=0)


def run(output):
    import torch
    from safetensors.torch import save_file, load_file
    from hanglish.core.source_correspondence import expand_path
    from hanglish.models.sentence_ranker import SentenceRanker, encode_sentence_pairs
    from hanglish.models.source_span_residual import SourceSpanResidualRanker
    from hanglish.models.span_token_bridge import encode_aligned_candidates
    from scripts.preflight_pko_seq2seq import ResourceWatchdog, append_jsonl, system_swap_bytes

    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    random.seed(SEED); torch.manual_seed(SEED)
    cohort, records = prepare()
    pins = {str(p): h for p, h in FIXED.items()}
    for name, h in PARENT_HASHES.items():
        require(sha(PARENT / name) == h, 'frozen parent changed: ' + name)
        pins[str(PARENT / name)] = h
    for p in [Path(__file__).resolve(), PLAN, ROOT / 'hanglish/core/source_correspondence.py',
              ROOT / 'hanglish/models/source_span_residual.py', ROOT / 'hanglish/models/span_token_bridge.py',
              ROOT / 'hanglish/models/sentence_ranker.py', ROOT / 'scripts/preflight_pko_seq2seq.py']:
        pins[str(p)] = sha(p)
    write(output / 'cohort.json', cohort)
    write(output / 'source-inputs.json', [records[r['id']] for r in cohort])
    write(output / 'pins.json', pins)
    baseline_swap = system_swap_bytes()

    class Watchdog(ResourceWatchdog):
        def _monitor(self):
            while not self.stop_event.wait(self.interval_s):
                try:
                    append_jsonl(self.trace, self._sample('watchdog'), self.lock)
                except BaseException as error:
                    self.failure = f'{type(error).__name__}: {error}'
                    os.kill(os.getpid(), signal.SIGUSR1)
                    if not self.stop_event.wait(5):
                        os._exit(124)
                    return

    watchdog = Watchdog(started, baseline_swap, output / 'resources.jsonl', limits=LIMITS, interval_s=2.)
    try:
        with watchdog:
            with watchdog.operation('load_parent_and_residual', update=True):
                parent = SentenceRanker.load(PARENT)
                model = SourceSpanResidualRanker(parent)
                reference_ranker = SentenceRanker(parent.model, parent.tokenizer,
                                                   max_length=256, batch_size=parent.batch_size)
                parent_before = tensor_hashes(model.parent.state_dict())
                residual_before = tensor_hashes(model.residual_state_dict())
            cache = {}; groups = {}; feature_tensors = {}; before = []
            for index, row in enumerate(cohort):
                record = records[row['id']]
                paths = defaultdict(list)
                for p in record['paths']:
                    paths[p['text']].append(expand_path(record, p))
                groups[row['id']] = [paths[text] for text in row['candidates']]
                with watchdog.operation('features:' + row['id'], update=True):
                    encoded, _ = encode_aligned_candidates(parent.tokenizer, row['candidates'], 256)
                    canonical = encode_sentence_pairs(parent.tokenizer, row['candidates'], [''] * len(row['candidates']), 256)
                    require(set(encoded) == set(canonical)
                            and all(torch.equal(encoded[k], canonical[k]) for k in encoded),
                            'real tokenizer differs from parent')
                    feature = model.extract_features(row['candidates']); cache[row['id']] = feature
                    reference_scores = feature.logits.new_tensor(reference_ranker.score_pairs(
                        [''] * len(row['candidates']), row['candidates']))
                    require(torch.equal(reference_scores, feature.logits),
                            'frozen hidden-state extraction changed parent scores')
                    with torch.no_grad():
                        initial = model(row['input'], groups[row['id']], features=feature)
                    require(torch.equal(initial['scores'], feature.logits)
                            and torch.count_nonzero(initial['residuals']).item() == 0,
                            'initial model does not reproduce frozen parent')
                    feature_tensors[f'row{index}.logits'] = feature.logits.cpu().contiguous()
                    feature_tensors[f'row{index}.hidden'] = feature.hidden.cpu().contiguous()
                    before.append({'id': row['id'], 'scores': feature.logits.tolist(),
                                   'parent_reference_scores': reference_scores.tolist(),
                                   'token_spans': feature.token_spans})
            save_file(feature_tensors, output / 'features.safetensors')
            write(output / 'before.json', before)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001, weight_decay=.01)
            model.train(); updates = 0
            gradient_groups = ('source_embedding.', 'source_encoder.', 'segment_head.0.', 'segment_head.2.')
            gradient_seen = {key: False for key in gradient_groups}
            for epoch in range(8):
                for row in cohort:
                    with watchdog.operation(f'update:{updates + 1}', update=True):
                        optimizer.zero_grad(set_to_none=True)
                        result = model(row['input'], groups[row['id']], features=cache[row['id']])
                        loss = train_loss(result['scores'], row['positive_indices'])
                        require(torch.isfinite(loss).item(), 'nonfinite training loss')
                        loss.backward()
                        group_norms = {key: sum(
                            p.grad.detach().float().square().sum().item()
                            for name, p in model.named_parameters()
                            if name.startswith(key) and p.grad is not None) ** .5
                            for key in gradient_groups}
                        for key, value in group_norms.items():
                            if updates >= 1 and value > 0:
                                gradient_seen[key] = True
                        norm = torch.nn.utils.clip_grad_norm_(
                            [p for p in model.parameters() if p.requires_grad], 1.)
                        require(torch.isfinite(norm).item(), 'nonfinite gradient norm')
                        optimizer.step(); updates += 1
                        append_jsonl(output / 'updates.jsonl', {'update': updates, 'epoch': epoch,
                                     'id': row['id'], 'loss': loss.item(), 'gradient_norm': norm.item(),
                                     'gradient_group_norms': group_norms})
            require(updates == 128, 'pilot update denominator differs')
            require(all(gradient_seen.values()), 'upstream residual gradients remained zero')
            parent_after = tensor_hashes(model.parent.state_dict())
            require(parent_before == parent_after and not model.parent.training,
                    'frozen parent changed during training')
            residual_after = tensor_hashes(model.residual_state_dict())
            changed = sorted(n for n in residual_before if residual_before[n] != residual_after[n])
            require(any(n.startswith('source_encoder.') for n in changed)
                    and any(n.startswith('segment_head.') for n in changed), 'source/head did not learn')
            model.eval(); after = []
            for row in cohort:
                with watchdog.operation('after:' + row['id'], update=True), torch.no_grad():
                    result = model(row['input'], groups[row['id']], features=cache[row['id']])
                    after.append({'id': row['id'], 'scores': result['scores'].tolist(),
                                  'residuals': result['residuals'].tolist(),
                                  'loss': train_loss(result['scores'], row['positive_indices']).item()})
            with watchdog.operation('save_reload', update=True):
                save_file(model.residual_state_dict(), output / 'residual.safetensors')
                identity = {'schema': SCHEMA, 'architecture': model.architecture,
                            'parent_artifact_hashes': PARENT_HASHES,
                            'token_bridge_sha256': pins[str(ROOT / 'hanglish/models/span_token_bridge.py')],
                            'residual_module_sha256': pins[str(ROOT / 'hanglish/models/source_span_residual.py')],
                            'source_correspondence_sha256': pins[str(ROOT / 'hanglish/core/source_correspondence.py')],
                            'residual_sha256': sha(output / 'residual.safetensors'),
                            'seed': SEED, 'learning_rate': .001, 'weight_decay': .01,
                            'clip_norm': 1., 'epochs': 8, 'cohort_sha256': sha(output / 'cohort.json')}
                write(output / 'checkpoint.json', identity)
                fresh = SourceSpanResidualRanker(parent)
                require(read(output / 'checkpoint.json') == identity
                        and fresh.architecture == identity['architecture']
                        and all(sha(PARENT / name) == h for name, h in identity['parent_artifact_hashes'].items())
                        and sha(output / 'residual.safetensors') == identity['residual_sha256'],
                        'checkpoint identity changed before reload')
                fresh.load_residual_state_dict(load_file(output / 'residual.safetensors'))
                fresh.eval(); reloaded = []
                for row, expected in zip(cohort, after):
                    with torch.no_grad():
                        scores = fresh(row['input'], groups[row['id']], features=cache[row['id']])['scores']
                    target = scores.new_tensor(expected['scores'])
                    require(torch.allclose(scores, target, atol=1e-5, rtol=0)
                            and scores.argmax().item() == target.argmax().item(), 'residual reload parity differs')
                    reloaded.append({'id': row['id'], 'scores': scores.tolist()})
            final_resources = watchdog._sample('final')
        require(all(sha(p) == h for p, h in pins.items()), 'pinned inputs changed during pilot')
        write(output / 'after.json', after); write(output / 'reloaded.json', reloaded)
        correct = lambda values: sum(max(range(len(v['scores'])), key=v['scores'].__getitem__) in r['positive_indices']
                                     for r, v in zip(cohort, values))
        report = {'schema': SCHEMA, 'completed': True, 'rows': 16, 'updates': updates, 'seed': SEED,
                  'pins': pins, 'limits': LIMITS, 'baseline_swap_bytes': baseline_swap,
                  'final_resources': final_resources, 'elapsed_s': time.monotonic() - started,
                  'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                  'parent_before': parent_before, 'parent_after': parent_after,
                  'residual_before': residual_before, 'residual_after': residual_after,
                  'changed_residual_names': changed,
                  'upstream_nonzero_gradients_after_first_update': gradient_seen,
                  'fit_diagnostic': {'before_correct': correct(before), 'after_correct': correct(after),
                                     'before_mean_loss': sum(train_loss(torch.tensor(v['scores']), r['positive_indices']).item()
                                                             for r, v in zip(cohort, before)) / 16,
                                     'after_mean_loss': sum(r['loss'] for r in after) / 16},
                  'contracts': {'parent_frozen': True, 'zero_init_parent_parity': True,
                                'actual_tokenizer_parent_parity': True, 'saved_reload_parity': True,
                                'training_labels_fit_only': True, 'hold_opened': False, 'dev_opened': False,
                                'test_opened': False, 'production_changed': False, 'quality_improvement_claimed': False},
                  'outputs': {p.name: sha(p) for p in output.iterdir() if p.is_file()}}
        write(output / 'report.json', report)
        return report
    except BaseException as error:
        if not (output / 'report.json').exists():
            write(output / 'report.json', {'schema': SCHEMA, 'completed': False,
                                         'error': f'{type(error).__name__}: {error}', 'production_changed': False})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({k: result[k] for k in ('completed', 'rows', 'updates', 'elapsed_s',
                                          'trainable_parameters', 'fit_diagnostic')}))
