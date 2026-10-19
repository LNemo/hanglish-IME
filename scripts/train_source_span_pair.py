"""Matched FIT training: source-local pooling versus whole-source pooling.

Both arms use the same frozen parent, actual candidate paths, initial residual
weights, row order and optimizer recipe. No evaluation labels are read here.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, OrderedDict
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.preflight_source_span_residual import (
    DATA, FIXED, PARENT, PARENT_HASHES, SEED, read, require, rows, sha,
    tensor_hashes, train_loss, write,
)

SCHEMA = 'source-span-paired-training-v1'
PLAN = ROOT / '.omx/plans/source-span-paired-training-v1.md'
ARMS = ('whole_source', 'source_local')
EPOCHS = 2
LIMITS = {'whole_seconds': 1800., 'update_seconds': 60.,
          'rss_bytes': 3 * 1024**3, 'swap_growth_bytes': 512 * 1024**2}
CACHE_BYTES = 1536 * 1024**2
RECIPE = {'seed': SEED, 'epochs': EPOCHS, 'learning_rate': .001,
          'weight_decay': .01, 'clip_norm': 1., 'threads': 2,
          'dtype': 'float32', 'optimizer': 'AdamW', 'accumulation': 1}


def whole_source_hook(module, inputs, output):
    """Control retains position/action fields but removes local source content."""
    states, final = output
    return states.mean(dim=1, keepdim=True).expand_as(states), final


def install_arm(model, arm):
    require(arm in ARMS, 'unknown source pooling arm')
    return (model.source_encoder.register_forward_hook(whole_source_hook)
            if arm == 'whole_source' else None)


def training_data():
    from scripts.preflight_source_span_residual import SELECTION
    from hanglish.core.source_correspondence import expand_path
    require(all(sha(p) == h for p, h in FIXED.items()), 'frozen FIT inputs changed')
    require(read(DATA / 'independent-verification.json')['verified'] is True,
            'FIT proof missing')
    selection = read(SELECTION)['rows']
    require(len(selection) == 1483 and len({r['id'] for r in selection}) == 1483,
            'FIT selection denominator differs')
    selected = {r['id']: r for r in selection}
    records = {}; groups = {}
    for record in rows(DATA / 'fit-paths.jsonl'):
        row = selected.get(record['id'])
        if row is None:
            continue
        require(record['id'] not in records and row['input'] == record['input']
                and row['component_id'] == record['component_id'], 'FIT identity differs')
        arm = row['arms']['b_union']; texts = arm['candidates']
        positives = arm['positive_indices']
        require(2 <= len(texts) <= 8 and len(set(texts)) == len(texts)
                and positives and len(set(positives)) == len(positives)
                and all(type(i) is int and 0 <= i < len(texts) for i in positives)
                and len(positives) < len(texts), 'invalid FIT candidate/target domain')
        by_text = defaultdict(list)
        for path in record['paths']:
            if path['text'] in texts:
                by_text[path['text']].append(expand_path(record, path))
        require(set(by_text) == set(texts), 'FIT candidate path missing')
        groups[row['id']] = [by_text[t] for t in texts]
        records[row['id']] = {'id': row['id'], 'input': row['input'],
                             'component_id': row['component_id'],
                             'candidates': texts, 'positive_indices': positives}
    require(set(records) == set(selected), 'FIT projection incomplete')
    # Preserve the already frozen, seeded paired selection order for both arms.
    return [records[r['id']] for r in selection], groups


class FeatureCache:
    """Immutable on-disk frozen features with a bounded in-memory LRU."""
    def __init__(self, directory, limit=32):
        require(type(limit) is int and limit > 0, 'positive cache limit required')
        self.directory = Path(directory)
        self.directory.mkdir(exist_ok=False)
        self.limit = limit
        self.entries = {}; self.memory = OrderedDict(); self.bytes = 0

    def put(self, index, feature):
        from safetensors.torch import save_file
        require(index not in self.entries, 'duplicate feature cache index')
        path = self.directory / f'{index:04d}.safetensors'
        save_file({'logits': feature.logits.cpu().contiguous(),
                   'hidden': feature.hidden.cpu().contiguous()}, path)
        metadata = {'candidates': list(feature.candidates), 'token_spans': feature.token_spans,
                    'tensors_sha256': sha(path)}
        meta_path = path.with_suffix('.json'); write(meta_path, metadata)
        self.bytes += path.stat().st_size + meta_path.stat().st_size
        require(self.bytes <= CACHE_BYTES, 'feature cache disk bound exceeded')
        self.entries[index] = {'tensors_sha256': metadata['tensors_sha256'],
                               'metadata_sha256': sha(meta_path)}

    def get(self, index):
        from safetensors.torch import load_file
        from hanglish.models.source_span_residual import FrozenCandidates
        require(index in self.entries, 'missing cached features')
        if index in self.memory:
            self.memory.move_to_end(index)
            return self.memory[index]
        path = self.directory / f'{index:04d}.safetensors'; meta_path = path.with_suffix('.json')
        identity = self.entries[index]
        require(sha(path) == identity['tensors_sha256']
                and sha(meta_path) == identity['metadata_sha256'], 'cached feature changed')
        meta = read(meta_path); tensors = load_file(path)
        feature = FrozenCandidates(tuple(meta['candidates']), tensors['logits'], tensors['hidden'],
            tuple(tuple(None if s is None else tuple(s) for s in token_row)
                  for token_row in meta['token_spans']))
        self.memory[index] = feature
        if len(self.memory) > self.limit:
            self.memory.popitem(last=False)
        return feature


def code_pins():
    files = [Path(__file__).resolve(), PLAN,
             ROOT / 'scripts/preflight_source_span_residual.py',
             ROOT / 'scripts/preflight_pko_seq2seq.py',
             ROOT / 'scripts/source_span_resources.py',
             ROOT / 'hanglish/models/source_span_residual.py',
             ROOT / 'hanglish/models/span_token_bridge.py',
             ROOT / 'hanglish/models/sentence_ranker.py',
             ROOT / 'hanglish/core/source_correspondence.py']
    pins = {str(p): h for p, h in FIXED.items()}
    for name, expected in PARENT_HASHES.items():
        require(sha(PARENT / name) == expected, 'frozen parent changed: ' + name)
        pins[str(PARENT / name)] = expected
    pins.update({str(p): sha(p) for p in files})
    return pins


def run(output):
    import torch
    from safetensors.torch import load_file, save_file
    from hanglish.models.sentence_ranker import SentenceRanker, encode_sentence_pairs
    from hanglish.models.source_span_residual import SourceSpanResidualRanker
    from hanglish.models.span_token_bridge import encode_aligned_candidates
    from scripts.preflight_pko_seq2seq import append_jsonl, system_swap_bytes
    from scripts.source_span_resources import AtomicResourceWatchdog

    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(SEED); torch.manual_seed(SEED)
    pins = code_pins(); write(output / 'pins.json', pins)
    data, groups = training_data(); write(output / 'selection.json', data)
    baseline_swap = system_swap_bytes()
    watchdog = AtomicResourceWatchdog(started, baseline_swap, output / 'resources.jsonl',
                                     interval_s=2., limits=LIMITS)
    try:
        with watchdog:
            with watchdog.operation('load_parent_and_initial', update=True):
                parent = SentenceRanker.load(PARENT)
                torch.manual_seed(SEED)
                model = SourceSpanResidualRanker(parent)
                reference_ranker = SentenceRanker(parent.model, parent.tokenizer,
                    max_length=256, batch_size=parent.batch_size)
                parent_before = tensor_hashes(model.parent.state_dict())
                initial = model.residual_state_dict()
                initial_hashes = tensor_hashes(initial)
                save_file(initial, output / 'initial.safetensors')
                architecture = model.architecture
            cache = FeatureCache(output / 'features')
            before_correct = 0; before_loss = 0.
            for index, row in enumerate(data):
                with watchdog.operation('features:' + row['id'], update=True):
                    encoded, _ = encode_aligned_candidates(parent.tokenizer, row['candidates'], 256)
                    canonical = encode_sentence_pairs(parent.tokenizer, row['candidates'],
                                                       [''] * len(row['candidates']), 256)
                    require(set(encoded) == set(canonical)
                            and all(torch.equal(encoded[k], canonical[k]) for k in encoded),
                            'candidate token bridge differs from parent input')
                    feature = model.extract_features(row['candidates'])
                    reference_scores = feature.logits.new_tensor(reference_ranker.score_pairs(
                        [''] * len(row['candidates']), row['candidates']))
                    require(torch.equal(reference_scores, feature.logits),
                            'frozen features differ from independent parent scores')
                    with torch.no_grad():
                        result = model(row['input'], groups[row['id']], features=feature)
                    require(torch.equal(result['scores'], feature.logits)
                            and torch.count_nonzero(result['residuals']).item() == 0,
                            'initial scores differ from parent')
                    loss = train_loss(feature.logits, row['positive_indices']).item()
                    correct = int(feature.logits.argmax().item() in row['positive_indices'])
                    before_correct += correct; before_loss += loss
                    cache.put(index, feature)
                    append_jsonl(output / 'before.jsonl', {'id': row['id'],
                        'scores': feature.logits.tolist(),
                        'parent_reference_scores': reference_scores.tolist(),
                        'canonical_token_input_equal': True, 'loss': loss, 'correct': correct})
            write(output / 'feature-manifest.json', cache.entries)
            del model
            arm_reports = {}
            for arm in ARMS:
                directory = output / arm; directory.mkdir()
                torch.manual_seed(SEED)
                model = SourceSpanResidualRanker(parent)
                model.load_residual_state_dict(initial)
                require(tensor_hashes(model.residual_state_dict()) == initial_hashes,
                        'matched arm initialization differs')
                hook = install_arm(model, arm)
                optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                    lr=RECIPE['learning_rate'], weight_decay=RECIPE['weight_decay'])
                model.train(); updates = 0
                gradient_groups = ('source_embedding.', 'source_encoder.',
                                   'segment_head.0.', 'segment_head.2.')
                maxima = dict.fromkeys(gradient_groups, 0.)
                epoch_losses = []
                for epoch in range(EPOCHS):
                    total = 0.
                    for index, row in enumerate(data):
                        with watchdog.operation(f'{arm}:update:{updates + 1}', update=True):
                            optimizer.zero_grad(set_to_none=True)
                            scores = model(row['input'], groups[row['id']], features=cache.get(index))['scores']
                            loss = train_loss(scores, row['positive_indices'])
                            require(torch.isfinite(loss).item(), 'nonfinite training loss')
                            loss.backward()
                            norms = {key: sum(p.grad.detach().square().sum().item()
                                for name, p in model.named_parameters()
                                if name.startswith(key) and p.grad is not None) ** .5
                                for key in gradient_groups}
                            norm = torch.nn.utils.clip_grad_norm_(
                                [p for p in model.parameters() if p.requires_grad],
                                RECIPE['clip_norm'], error_if_nonfinite=True)
                            optimizer.step(); updates += 1; total += loss.item()
                            if updates > 1:
                                maxima = {key: max(maxima[key], norms[key]) for key in maxima}
                            append_jsonl(directory / 'updates.jsonl', {'update': updates,
                                'epoch': epoch, 'id': row['id'], 'loss': loss.item(),
                                'gradient_norm': norm.item(), 'gradient_groups': norms})
                    epoch_losses.append(total / len(data))
                require(all(v > 0 for v in maxima.values()), 'upstream gradient group never trained')
                require(tensor_hashes(model.parent.state_dict()) == parent_before
                        and all(p.grad is None for p in model.parent.parameters()), 'frozen parent changed')
                model.eval(); after_correct = 0; after_loss = 0.
                # Full FIT scores are diagnostics, never checkpoint selection.
                for index, row in enumerate(data):
                    with watchdog.operation(arm + ':fit:' + row['id'], update=True), torch.no_grad():
                        result = model(row['input'], groups[row['id']], features=cache.get(index))
                        scores = result['scores']; loss = train_loss(scores, row['positive_indices']).item()
                        correct = int(scores.argmax().item() in row['positive_indices'])
                        after_correct += correct; after_loss += loss
                        append_jsonl(directory / 'after.jsonl', {'id': row['id'],
                            'scores': scores.tolist(), 'residuals': result['residuals'].tolist(),
                            'loss': loss, 'correct': correct})
                with watchdog.operation(arm + ':save_reload', update=True):
                    state = model.residual_state_dict(); save_file(state, directory / 'residual.safetensors')
                    metadata = {'schema': SCHEMA, 'arm': arm, 'architecture': architecture,
                                'recipe': RECIPE, 'pins': pins, 'selection_sha256': sha(output / 'selection.json'),
                                'initial_sha256': sha(output / 'initial.safetensors'),
                                'residual_sha256': sha(directory / 'residual.safetensors')}
                    write(directory / 'checkpoint.json', metadata)
                    fresh = SourceSpanResidualRanker(parent)
                    fresh.load_residual_state_dict(load_file(directory / 'residual.safetensors'))
                    fresh_hook = install_arm(fresh, arm); fresh.eval()
                    reload_rows = []
                    for index in sorted(set([0, len(data) - 1] + list(range(0, len(data), 100)))):
                        row = data[index]; feature = cache.get(index)
                        with torch.no_grad():
                            original = model(row['input'], groups[row['id']], features=feature)['scores']
                            reloaded = fresh(row['input'], groups[row['id']], features=feature)['scores']
                        require(torch.equal(original, reloaded), 'checkpoint reload score mismatch')
                        reload_rows.append({'id': row['id'], 'scores': original.tolist(),
                                            'reloaded': reloaded.tolist()})
                    write(directory / 'reload.json', reload_rows)
                    if fresh_hook is not None:
                        fresh_hook.remove()
                    del fresh
                arm_reports[arm] = {'updates': updates, 'initial_hashes': initial_hashes,
                    'final_hashes': tensor_hashes(state), 'upstream_gradient_maxima': maxima,
                    'epoch_mean_loss': epoch_losses, 'fit_correct': after_correct,
                    'fit_mean_loss': after_loss / len(data), 'reload_rows': len(reload_rows),
                    'checkpoint_sha256': sha(directory / 'checkpoint.json')}
                if hook is not None:
                    hook.remove()
                del model, optimizer
            final_resources = watchdog._sample('final')
        require(all(sha(Path(p)) == h for p, h in pins.items()), 'pinned input changed during training')
        report = {'schema': SCHEMA, 'completed': True, 'rows': len(data), 'recipe': RECIPE,
            'limits': LIMITS, 'arms': arm_reports, 'parent_before': parent_before,
            'parent_after': tensor_hashes(parent.model.state_dict()), 'pins': pins,
            'before_fit_correct': before_correct, 'before_fit_mean_loss': before_loss / len(data),
            'cache_bytes': cache.bytes, 'baseline_swap_bytes': baseline_swap,
            'final_resources': final_resources, 'elapsed_s': time.monotonic() - started,
            'contracts': {'fit_only': True, 'hold_opened': False, 'dev_opened': False,
                          'test_opened': False, 'production_changed': False,
                          'matched_initialization_data_order_recipe': True},
            'outputs': {str(p.relative_to(output)): sha(p) for p in output.rglob('*')
                        if p.is_file() and 'features' not in p.relative_to(output).parts}}
        write(output / 'report.json', report)
        print(json.dumps({'completed': True, 'elapsed_s': report['elapsed_s'],
                          'before_fit_correct': before_correct, 'arms': arm_reports}, ensure_ascii=False))
    except BaseException as error:
        write(output / 'failure.json', {'schema': SCHEMA, 'completed': False,
              'error_type': type(error).__name__, 'error': str(error),
              'elapsed_s': time.monotonic() - started})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.prepare_only:
        selected, paths = training_data()
        summary = {'rows': len(selected), 'groups': len(paths),
                   'candidate_count': sum(len(r['candidates']) for r in selected),
                   'max_source_bytes': max(len(r['input'].encode()) for r in selected),
                   'updates_per_arm': len(selected) * EPOCHS, 'recipe': RECIPE}
        write(args.output, summary); print(json.dumps(summary))
    else:
        run(args.output)
