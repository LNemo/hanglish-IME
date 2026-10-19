"""Fixed-model FIT audit: rank the same forward with full and cap-8 domains."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.preflight_source_span_residual import DATA, FIXED, PARENT, PARENT_HASHES, read, require, rows, sha, write, tensor_hashes
from scripts.train_source_span_pair import training_data
from scripts.train_shared_edge_residual import LIMITS, TARGET_CHUNK, CACHE_BYTES, token_census, plain

SCHEMA = 'shared-edge-full-fit-audit-v1'
AUTH_SCHEMA = SCHEMA + '-authorization'
TRAIN = ROOT / '.omx/artifacts/shared-edge-residual-training-v1'
CENSUS = ROOT / '.omx/artifacts/recombination-fresh-census-v1'
PLAN = ROOT / '.omx/plans/shared-edge-full-fit-audit-v1.md'
FROZEN = {
    TRAIN / 'report.json': '481ff33a1ae6e9f0e8f3d5c24566f5e076a42d29dd02145e1ad964fd6a06e5da',
    TRAIN / 'independent-verification.json': 'e4d48a8be5373198e83a11bc88793f6262520f6a2b3b075da2d3a1d8e28cf0e1',
    CENSUS / 'capture.jsonl': '847a8d6603dbd66ed87d2f9863b4e20581989887da76679b1a5586bb6d552154',
    CENSUS / 'capture.complete.json': '934b832bccf7b89c6f5a9700d2b3468dcf886caeb92be5a5a9ee0f88f5a06dcb',
    CENSUS / 'independent-verification.json': 'a744c8324fca8c2a1ae170fb9e51aa0ce06f9e0a6310446aecbe93b11e5deb29',
}


def project(selected, records, captures):
    from hanglish.core.source_correspondence import expand_path
    prepared = []; groups = {}; targets = set()
    for chosen in selected:
        key = chosen['id']; record = records[key]; capture = captures[key]
        require(record['input'] == chosen['input'] == capture['input']
                and record['component_id'] == chosen['component_id'] == capture['component_id']
                and capture['split'] == 'fit', 'FIT identity differs')
        require('expected' not in capture and len(capture['score_texts']) == len(capture['scores'])
                and len(set(capture['score_texts'])) == len(capture['score_texts']), 'source-only score domain differs')
        by_text = defaultdict(list)
        for compact in record['paths']:
            if compact['text'] != chosen['input']:
                path = expand_path(record, compact); by_text[path.text].append(path)
        candidates = sorted(by_text)
        require(set(chosen['candidates']).issubset(by_text), 'selected candidate missing from full domain')
        score_map = dict(zip(capture['score_texts'], capture['scores']))
        require(set(candidates) == set(score_map)
                and all(type(v) in (int, float) and math.isfinite(v) for v in score_map.values()),
                'full nonraw parent score domain differs')
        positives = {chosen['candidates'][i] for i in chosen['positive_indices']}
        require(positives and positives.issubset(chosen['candidates']), 'FIT positive domain differs')
        index = {text: i for i, text in enumerate(candidates)}
        groups[key] = [sorted(by_text[text], key=lambda p: p.path_id) for text in candidates]
        targets.update(s.text for group in groups[key] for path in group for s in path.spans)
        prepared.append({'id': key, 'input': chosen['input'], 'component_id': chosen['component_id'],
            'candidates': candidates, 'positive_indices': [index[t] for t in sorted(positives)],
            'selected_indices': [index[t] for t in chosen['candidates']],
            'base_scores': [score_map[t] for t in candidates],
            'candidate_path_counts': [len(groups[key][i]) for i in range(len(candidates))]})
    return prepared, groups, sorted(targets)


def prepare():
    require(all(sha(p) == h for p, h in FROZEN.items()), 'frozen diagnostic input changed')
    training = read(TRAIN / 'report.json'); proof = read(TRAIN / 'independent-verification.json')
    require(training['completed'] and proof['verified']
            and proof['report_sha256'] == FROZEN[TRAIN / 'report.json'], 'shared training proof differs')
    pins = dict(training['pins']); pins.update({str(p): h for p, h in FROZEN.items()})
    require(all(sha(p) == h for p, h in pins.items()), 'shared training input changed')
    for name, digest in training['outputs'].items():
        require(sha(TRAIN / name) == digest, 'shared training output changed')
        pins[str(TRAIN / name)] = digest
    census_proof = read(CENSUS / 'independent-verification.json')
    require(census_proof['verified'] and census_proof['capture_sha256'] == FROZEN[CENSUS / 'capture.jsonl']
            and census_proof['contracts']['source_scores_checked_before_labels'] is True,
            'closed source census proof differs')
    expected_parent = {str(PARENT / name): digest for name, digest in PARENT_HASHES.items()}
    require(census_proof['model_pins'] == expected_parent
            and all(pins.get(p) == h and sha(p) == h for p, h in expected_parent.items()),
            'census parent differs from trained parent')
    closure = read(CENSUS / 'capture.complete.json')
    require(closure == {'schema': 'recombination-fresh-training-parent-census-v1-source-closure',
            'source_only': True, 'targets_joined': False, 'rows': 2048,
            'capture_sha256': FROZEN[CENSUS / 'capture.jsonl'], 'input_truncation': False, 'score_context': ''},
            'source census closure differs')
    selected, old_groups = training_data(); ids = {r['id'] for r in selected}
    records = {r['id']: r for r in rows(DATA / 'fit-paths.jsonl') if r['id'] in ids}
    captures = {}
    for capture in rows(CENSUS / 'capture.jsonl'):
        # Mixed capture is source-only. Do not retain non-FIT candidate records.
        if capture['split'] == 'fit' and capture['id'] in ids:
            require(capture['id'] not in captures, 'duplicate source census ID')
            captures[capture['id']] = capture
    require(set(records) == set(captures) == ids, 'FIT source denominator differs')
    prepared, groups, targets = project(selected, records, captures)
    for row in prepared:
        legacy = old_groups[row['id']]
        for j, i in enumerate(row['selected_indices']):
            require({p.path_id for p in groups[row['id']][i]} == {p.path_id for p in legacy[j]},
                    'selected candidate path domain changed')
        row['selected_path_expansion_count'] = 0
    require(len(prepared) == 1483 and sum(len(r['candidates']) for r in prepared) == 202603,
            'fixed full FIT domain differs')
    files = [PLAN, Path(__file__).resolve(), ROOT / 'scripts/train_shared_edge_residual.py',
             ROOT / 'scripts/train_source_span_pair.py', ROOT / 'hanglish/core/source_correspondence.py',
             ROOT / 'hanglish/models/shared_edge_residual.py', ROOT / 'scripts/source_span_resources.py']
    pins.update({str(p): sha(p) for p in files})
    return prepared, groups, targets, pins


def summarize_row(row, scores, known_targets, groups):
    import math
    require(len(scores) == len(row['candidates']) and all(math.isfinite(v) for v in scores), 'score domain differs')
    positives = set(row['positive_indices']); selected = set(row['selected_indices'])
    require(positives.issubset(selected), 'positive escaped restricted domain')
    rank = lambda indices: sorted(indices, key=lambda i: (-scores[i], row['candidates'][i]))
    full = rank(range(len(scores))); short = rank(selected)
    full_hit = int(full[0] in positives); short_hit = int(short[0] in positives)
    full_top5 = int(bool(set(full[:4]) & positives)); short_top5 = int(bool(set(short[:4]) & positives))
    require(full_hit <= short_hit and full_top5 <= short_top5, 'expansion gained despite all positives selected')
    best_positive = max(scores[i] for i in positives)
    excluded = [i for i in full if i not in selected]
    winning_texts = {s.text for path in groups[full[0]] for s in path.spans}
    return {'restricted_top1': short_hit, 'full_top1': full_hit,
            'restricted_top5': short_top5, 'full_top5': full_top5,
            'top1_expansion_loss': short_hit - full_hit, 'top5_expansion_loss': short_top5 - full_top5,
            'winner_outside_selected': int(full[0] not in selected),
            'winner_has_new_target_type': int(bool(winning_texts - known_targets)),
            'excluded_above_best_positive': sum(scores[i] > best_positive for i in excluded)}


def authorization(prepared, targets, pins):
    return {'schema': AUTH_SCHEMA, 'rows': len(prepared), 'candidates': sum(len(r['candidates']) for r in prepared),
            'targets': len(targets), 'limits': LIMITS, 'pins': pins}


def run(output, auth_path):
    started = time.monotonic(); prepared, groups, targets, pins = prepare()
    require(read(auth_path) == authorization(prepared, targets, pins), 'consumed prelaunch authorization differs')
    pins[str(Path(auth_path).resolve())] = sha(auth_path)
    import torch
    from safetensors.torch import load_file, save_file
    from hanglish.models.sentence_ranker import SentenceRanker
    from hanglish.models.source_span_residual import FrozenCandidates
    from hanglish.models.shared_edge_residual import FrozenEdgeFeatures, SharedEdgeResidualRanker
    from scripts.preflight_pko_seq2seq import append_jsonl, system_swap_bytes
    from scripts.source_span_resources import AtomicResourceWatchdog
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    write(output / 'pins.json', pins); write(output / 'selection.json', prepared); write(output / 'targets.json', targets)
    torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    baseline = system_swap_bytes(); watchdog = AtomicResourceWatchdog(started, baseline,
        output / 'resources.jsonl', interval_s=2., limits=LIMITS)
    try:
        with watchdog:
            with watchdog.operation('load_frozen_model_and_cache', update=True):
                model = SharedEdgeResidualRanker(SentenceRanker.load(PARENT))
                model.load_residual_state_dict(load_file(TRAIN / 'residual.safetensors')); model.eval()
                state_before = tensor_hashes(model.residual_state_dict()); parent_before = tensor_hashes(model.parent.state_dict())
                old_targets = read(TRAIN / 'targets.json'); old_cache = load_file(TRAIN / 'target-features.safetensors')
                old_index = {t: i for i, t in enumerate(old_targets)}; known = set(old_targets)
                old_cache_hashes = tensor_hashes(old_cache)
                require(old_cache_hashes == read(TRAIN / 'report.json')['target_cache_tensor_hashes'],
                        'old isolated target cache tensors differ')
            with watchdog.operation('token_census', update=True):
                census = token_census(model.tokenizer, prepared, targets); write(output / 'tokens.json', census)
            new_targets = [t for t in targets if t not in known]; new_vectors = []; new_flags = []
            for start in range(0, len(new_targets), TARGET_CHUNK):
                with watchdog.operation(f'new_targets:{start}', update=True):
                    features = model.extract_edge_features(new_targets[start:start + TARGET_CHUNK])
                    new_vectors.extend(features.vectors.cpu().unbind()); new_flags.extend(features.has_tokens.cpu().unbind())
            with watchdog.operation('close_target_cache', update=True):
                new_index = {t: i for i, t in enumerate(new_targets)}
                vectors = torch.stack([old_cache['vectors'][old_index[t]] if t in known else new_vectors[new_index[t]] for t in targets])
                flags = torch.stack([old_cache['has_tokens'][old_index[t]] if t in known else new_flags[new_index[t]] for t in targets])
                target_index = {t: i for i, t in enumerate(targets)}
                reuse = [t for t in targets if t in known]
                require(torch.equal(vectors[[target_index[t] for t in reuse]], old_cache['vectors'][[old_index[t] for t in reuse]])
                        and torch.equal(flags[[target_index[t] for t in reuse]], old_cache['has_tokens'][[old_index[t] for t in reuse]]),
                        'known isolated target tensors were not reused exactly')
                merged_state = {'vectors': vectors, 'has_tokens': flags}; cache_hashes = tensor_hashes(merged_state)
                save_file({'vectors': vectors, 'has_tokens': flags}, output / 'target-features.safetensors')
                require((output / 'target-features.safetensors').stat().st_size <= CACHE_BYTES, 'cache disk bound exceeded')
                write(output / 'target-features.json', {'schema': SCHEMA,
                    'old_target_file_sha256': sha(TRAIN / 'targets.json'),
                    'old_tensor_file_sha256': sha(TRAIN / 'target-features.safetensors'),
                    'old_tensor_hashes': old_cache_hashes, 'tensor_hashes': cache_hashes,
                    'tensor_file_sha256': sha(output / 'target-features.safetensors'),
                    'tokens_sha256': sha(output / 'tokens.json'),
                    'reused_targets': len(reuse), 'new_targets': len(new_targets),
                    'known_reuse_exact': True, 'chunk_size': TARGET_CHUNK,
                    'tensor_bytes': vectors.numel() * vectors.element_size() + flags.numel() * flags.element_size()})
                del new_vectors, new_flags
            historical_rows = list(rows(TRAIN / 'after.jsonl')); historical = {r['id']: r for r in historical_rows}
            require(len(historical_rows) == len(historical) == len(prepared)
                    and set(historical) == {r['id'] for r in prepared}, 'historical FIT denominator differs')
            totals = Counter(); max_drift = 0.; max_base_drift = 0.; max_residual_drift = 0.
            for row in prepared:
                with watchdog.operation('full_fit:' + row['id'], update=True), torch.no_grad():
                    ids = sorted({target_index[s.text] for group in groups[row['id']] for path in group for s in path.spans})
                    target = FrozenEdgeFeatures(tuple(targets[i] for i in ids), vectors[ids], flags[ids])
                    base = FrozenCandidates(tuple(row['candidates']), torch.tensor(row['base_scores'], dtype=torch.float32), torch.empty(0), ())
                    result = model(row['input'], groups[row['id']], base_features=base, edge_features=target)
                    scores = result['scores'].tolist(); stats = summarize_row(row, scores, known, groups[row['id']])
                    previous = historical[row['id']]; chosen = row['selected_indices']
                    stats['historical_selected_correct'] = previous['correct']
                    stats['historical_correctness_disagreement'] = int(previous['correct'] != stats['restricted_top1'])
                    drift = max(abs(scores[i] - previous['scores'][j]) for j, i in enumerate(chosen))
                    base_drift = max(abs(row['base_scores'][i] - previous['base_scores'][j]) for j, i in enumerate(chosen))
                    residual_drift = max(abs(result['residuals'][i].item() - previous['residuals'][j]) for j, i in enumerate(chosen))
                    max_drift = max(max_drift, drift); max_base_drift = max(max_base_drift, base_drift); totals.update(stats)
                    max_residual_drift = max(max_residual_drift, residual_drift)
                    append_jsonl(output / 'predictions.jsonl', {'id': row['id'], 'scores': scores,
                        'residuals': plain(result['residuals']), 'edge_ids': plain(result['edge_ids']),
                        'edge_scores': plain(result['edge_scores']), 'winning_path_ids': plain(result['winning_path_ids']),
                        'counts': stats, 'historical_selected_score_max_abs_drift': drift,
                        'historical_selected_base_max_abs_drift': base_drift,
                        'historical_selected_residual_max_abs_drift': residual_drift})
            require(state_before == tensor_hashes(model.residual_state_dict())
                    and parent_before == tensor_hashes(model.parent.state_dict()), 'fixed model changed')
            require(old_cache_hashes == tensor_hashes(old_cache)
                    and cache_hashes == tensor_hashes(merged_state), 'frozen target cache mutated')
            final = watchdog.sample_and_append('final')
        require(all(sha(p) == h for p, h in pins.items()), 'pinned input changed')
        elapsed = time.monotonic() - started; require(elapsed <= LIMITS['whole_seconds'], 'whole audit time exceeded')
        report = {'schema': SCHEMA, 'completed': True, 'rows': len(prepared), 'counts': dict(totals),
            'candidates': sum(len(r['candidates']) for r in prepared), 'targets': len(targets),
            'reused_targets': len(set(targets) & known), 'new_targets': len(new_targets),
            'historical_selected_score_max_abs_drift': max_drift, 'historical_selected_base_max_abs_drift': max_base_drift,
            'historical_selected_residual_max_abs_drift': max_residual_drift,
            'full_path_count': sum(sum(r['candidate_path_counts']) for r in prepared),
            'selected_path_expansion_count': sum(r['selected_path_expansion_count'] for r in prepared),
            'target_cache_tensor_hashes': cache_hashes, 'old_target_cache_unchanged': True,
            'model_before': state_before, 'model_after': tensor_hashes(model.residual_state_dict()),
            'parent_before': parent_before, 'parent_after': tensor_hashes(model.parent.state_dict()),
            'scope': {'fit_labels_only': True, 'optimizer_updates': 0, 'dev_targets_opened': False,
                'test_targets_opened': False, 'inner_hold_labels_opened': False, 'production_changed': False,
                'full_and_restricted_use_same_forward': True, 'source_only_mixed_census_filtered': True},
            'pins': pins, 'limits': LIMITS, 'baseline_swap_bytes': baseline, 'final_resources': final,
            'elapsed_s': elapsed, 'outputs': {p.name: sha(p) for p in output.iterdir() if p.is_file()}}
        write(output / 'report.json', report); print(json.dumps({'completed': True, 'counts': dict(totals), 'elapsed_s': elapsed}), flush=True)
    except BaseException as error:
        write(output / 'failure.json', {'schema': SCHEMA, 'error': str(error), 'elapsed_s': time.monotonic() - started}); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--prepare-only', action='store_true'); parser.add_argument('--authorization', type=Path)
    args = parser.parse_args()
    if args.prepare_only:
        prepared, _, targets, pins = prepare(); value = authorization(prepared, targets, pins)
        write(args.output, value); print(json.dumps({k: value[k] for k in ('rows', 'candidates', 'targets')}))
    else:
        require(args.authorization is not None, 'prelaunch authorization required'); run(args.output, args.authorization)
