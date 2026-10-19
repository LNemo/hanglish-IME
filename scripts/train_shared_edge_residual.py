"""Fixed FIT-only training of candidate-independent source-edge residuals."""
from __future__ import annotations

import argparse
from collections import OrderedDict
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.preflight_source_span_residual import (
    PARENT, PARENT_HASHES, SEED, read, require, rows, sha, tensor_hashes, train_loss, write,
)
from scripts.train_source_span_pair import training_data
from scripts.prepare_shared_edge_fit import project

SCHEMA = 'shared-edge-residual-training-v1'
AUTH_SCHEMA = 'shared-edge-residual-prelaunch-authorization-v1'
OLD = ROOT / '.omx/artifacts/source-span-paired-training-v1'
INVENTORY = ROOT / '.omx/artifacts/shared-edge-fit-inventory-v1'
PLAN = ROOT / '.omx/plans/shared-edge-residual-v1.md'
RECIPE = {'seed': SEED, 'epochs': 2, 'learning_rate': .001, 'weight_decay': .01,
          'clip_norm': 1., 'threads': 2, 'dtype': 'float32', 'optimizer': 'AdamW', 'accumulation': 1}
LIMITS = {'whole_seconds': 1800., 'update_seconds': 60.,
          'rss_bytes': 3 * 1024**3, 'swap_growth_bytes': 512 * 1024**2}
TARGET_CHUNK = 256
CACHE_BYTES = 256 * 1024**2


def authenticate_inputs():
    old = read(OLD / 'report.json'); proof = read(OLD / 'independent-verification.json')
    require(old['schema'] == 'source-span-paired-training-v1' and old['completed'] is True
            and old['rows'] == 1483 and old['recipe'] == RECIPE
            and proof['schema'] == 'source-span-paired-training-independent-verification-v1'
            and proof['verified'] is True and proof['run_report_sha256'] == sha(OLD / 'report.json'),
            'historical control identity differs')
    require(all(sha(p) == h for p, h in old['pins'].items())
            and all(sha(OLD / p) == h for p, h in old['outputs'].items()), 'historical control changed')
    require(all(sha(PARENT / p) == h for p, h in PARENT_HASHES.items()), 'parent bundle changed')
    inventory = read(INVENTORY / 'report.json')
    require(inventory['schema'] == 'shared-edge-fit-inventory-v1'
            and inventory['scope'] == 'authorized_fixed_fit_only'
            and all(sha(p) == h for p, h in inventory['pins'].items())
            and inventory['targets_sha256'] == sha(INVENTORY / 'targets.json')
            and inventory['rows_sha256'] == sha(INVENTORY / 'rows.jsonl'), 'FIT inventory changed')
    data, groups = training_data()
    targets, projected, counts = project(data, groups)
    require(targets == read(INVENTORY / 'targets.json')
            and projected == list(rows(INVENTORY / 'rows.jsonl')) and counts == inventory['counts'],
            'FIT inventory does not reconstruct from frozen selection/paths')
    require(counts['rows'] == 1483 and counts['unique_target_texts'] == 16814
            and counts['edges'] == 47282 and counts['candidates'] == 11793,
            'frozen inventory counts differ')
    before = list(rows(OLD / 'before.jsonl')); control = list(rows(OLD / 'source_local/after.jsonl'))
    require([r['id'] for r in data] == [r['id'] for r in before] == [r['id'] for r in control],
            'historical score order differs')
    files = [PLAN, Path(__file__).resolve(), ROOT / 'hanglish/models/shared_edge_residual.py',
        ROOT / 'scripts/prepare_shared_edge_fit.py', INVENTORY / 'report.json',
        INVENTORY / 'targets.json', INVENTORY / 'rows.jsonl', OLD / 'report.json',
        OLD / 'independent-verification.json', OLD / 'initial.safetensors',
        OLD / 'feature-manifest.json', OLD / 'before.jsonl', OLD / 'source_local/after.jsonl']
    pins = dict(old['pins']); pins.update({str(p): sha(p) for p in files})
    return data, groups, targets, projected, before, control, pins


def consume_authorization(path, pins):
    """Validate a post-freeze authorization and pin it for the whole run."""
    path = Path(path).resolve()
    authorization = read(path)
    require(isinstance(authorization, dict)
            and set(authorization) == {'schema', 'rows', 'targets', 'recipe', 'pins'}
            and authorization['schema'] == AUTH_SCHEMA
            and authorization['rows'] == 1483
            and authorization['targets'] == 16814
            and authorization['recipe'] == RECIPE
            and authorization['pins'] == pins
            and all(sha(Path(p)) == digest for p, digest in pins.items()),
            'prelaunch authorization differs')
    return {**pins, str(path): sha(path)}


class LogitCache:
    """Read only cached base logits; never load full-candidate hidden tensors."""
    def __init__(self):
        self.entries = {int(k): v for k, v in read(OLD / 'feature-manifest.json').items()}
        self.memory = OrderedDict()

    def get(self, index, candidates):
        if index not in self.memory:
            import torch
            from safetensors import safe_open
            from hanglish.models.source_span_residual import FrozenCandidates
            path = OLD / 'features' / f'{index:04d}.safetensors'; meta_path = path.with_suffix('.json')
            identity = self.entries[index]
            require(sha(path) == identity['tensors_sha256']
                    and sha(meta_path) == identity['metadata_sha256'], 'base logit cache changed')
            meta = read(meta_path)
            require(meta['tensors_sha256'] == identity['tensors_sha256'], 'base cache metadata differs')
            with safe_open(path, framework='pt', device='cpu') as stream:
                logits = stream.get_tensor('logits')
            # The inherited feature container is used for candidate identity/logits
            # only. Empty placeholders prevent a candidate hidden-state dependency.
            self.memory[index] = FrozenCandidates(tuple(meta['candidates']), logits, torch.empty(0), ())
        value = self.memory[index]; self.memory.move_to_end(index)
        while len(self.memory) > 32: self.memory.popitem(last=False)
        require(value.candidates == tuple(candidates), 'base candidate order differs')
        return value


def token_census(tokenizer, data, targets):
    reserved = tokenizer.num_special_tokens_to_add(pair=True)
    require(type(reserved) is int and 0 <= reserved <= 256, 'pair-special budget differs')
    def length(text):
        ids = tokenizer(' '.join(text.strip().split()), add_special_tokens=False,
                        truncation=False)['input_ids']
        require(isinstance(ids, list) and all(type(i) is int for i in ids), 'tokenizer IDs malformed')
        count = len(ids) + reserved
        require(count <= 256, 'complete candidate or isolated target exceeds256tokens')
        return count
    target_counts = [length(text) for text in targets]
    candidate_counts = [[length(text) for text in row['candidates']] for row in data]
    return {'reserved_tokens': reserved, 'target_counts': target_counts,
            'candidate_counts': candidate_counts, 'max_target_tokens': max(target_counts),
            'max_candidate_tokens': max(max(v) for v in candidate_counts)}


def audit_forward(result, row):
    """Check actual tensor contributions against the authenticated path inventory."""
    import torch
    edges = [(e['input_start'], e['input_end'], e['text']) for e in row['edges']]
    require(list(result['edge_ids']) == edges
            and result['edge_scores'].shape == (len(edges),)
            and torch.isfinite(result['edge_scores']).all().item(),
            'model edge domain/order differs')
    seen = {}; audits = result['path_audit']
    require(len(audits) == len(row['paths'])
            and result['residuals'].shape == (len(row['paths']),)
            and len(result['winning_path_ids']) == len(row['paths']),
            'candidate path inventory differs')
    for candidate_index, (expected_paths, paths) in enumerate(zip(row['paths'], audits)):
        require(len(expected_paths) == len(paths), 'distinct path count differs')
        path_residuals = []
        for expected, path in zip(expected_paths, paths):
            require(path['path_id'] == expected['path_id']
                    and list(path['edge_ids']) == expected['edge_indices'], 'path identity differs')
            weights = result['edge_scores'].new_tensor([n / expected['weight_denominator']
                                                     for n in expected['weight_numerators']])
            require(torch.equal(path['weights'], weights)
                    and abs(weights.sum().item() - 1.) <= 1e-6, 'source-length weights differ')
            values = result['edge_scores'][list(path['edge_ids'])] * weights
            require(torch.equal(path['contributions'], values)
                    and torch.equal(path['residual'], values.sum()), 'weighted contributions differ')
            path_residuals.append(path['residual'])
            for key, contribution in zip(path['edge_ids'], path['contributions']):
                if key in seen:
                    require(torch.equal(seen[key], contribution), 'shared edge contribution changed')
                else: seen[key] = contribution.detach()
        stacked = torch.stack(path_residuals); maximum = stacked.max()
        winners = tuple(path['path_id'] for path in paths
                        if bool((path['residual'].detach() == maximum.detach()).item()))
        require(torch.equal(result['residuals'][candidate_index], maximum)
                and tuple(result['winning_path_ids'][candidate_index]) == winners,
                'candidate max path reduction differs')
    require(len(seen) == len(edges), 'an edge was not audited')


def plain(value):
    import torch
    if isinstance(value, torch.Tensor): return value.detach().cpu().tolist()
    if isinstance(value, dict): return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [plain(v) for v in value]
    return value


def subset_targets(row, targets, vectors, has_tokens):
    from hanglish.models.shared_edge_residual import FrozenEdgeFeatures
    ids = sorted({edge['target_index'] for edge in row['edges']})
    return FrozenEdgeFeatures(tuple(targets[i] for i in ids), vectors[ids], has_tokens[ids])


def run(output, authorization):
    started = time.monotonic()
    data, groups, targets, inventory, before, control, pins = authenticate_inputs()
    pins = consume_authorization(authorization, pins)
    import torch
    from safetensors.torch import load_file, save_file
    from hanglish.models.sentence_ranker import SentenceRanker
    from hanglish.models.shared_edge_residual import FrozenEdgeFeatures, SharedEdgeResidualRanker
    from scripts.preflight_pko_seq2seq import append_jsonl, system_swap_bytes
    from scripts.source_span_resources import AtomicResourceWatchdog

    output = Path(output); output.mkdir(parents=True, exist_ok=False); write(output / 'pins.json', pins)
    write(output / 'selection.json', data); write(output / 'targets.json', targets)
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True); random.seed(SEED); torch.manual_seed(SEED)
    baseline_swap = system_swap_bytes()
    watchdog = AtomicResourceWatchdog(started, baseline_swap, output / 'resources.jsonl',
                                     interval_s=2., limits=LIMITS)
    try:
        with watchdog:
            with watchdog.operation('load_parent_and_initial', update=True):
                parent = SentenceRanker.load(PARENT)
                torch.manual_seed(SEED); model = SharedEdgeResidualRanker(parent)
                initial = load_file(OLD / 'initial.safetensors'); model.load_residual_state_dict(initial)
                require(tensor_hashes(model.residual_state_dict()) == tensor_hashes(initial),
                        'initial tensors differ from matched control')
                parent_before = tensor_hashes(model.parent.state_dict())
                save_file(model.residual_state_dict(), output / 'initial.safetensors')
            with watchdog.operation('token_census', update=True):
                census = token_census(parent.tokenizer, data, targets); write(output / 'tokens.json', census)
            vectors = []; flags = []; parent_batches = 0
            for start in range(0, len(targets), TARGET_CHUNK):
                with watchdog.operation(f'isolated_targets:{start}', update=True):
                    texts = targets[start:start + TARGET_CHUNK]
                    extracted = model.extract_edge_features(texts)
                    require(extracted.texts == tuple(texts), 'isolated feature text order differs')
                    expected_flags = torch.tensor([v > census['reserved_tokens']
                        for v in census['target_counts'][start:start + len(texts)]], dtype=torch.bool)
                    require(torch.equal(extracted.has_tokens.cpu(), expected_flags), 'target token flags differ')
                    vectors.append(extracted.vectors.cpu()); flags.append(extracted.has_tokens.cpu())
                    parent_batches += (len(texts) + model.feature_batch_size - 1) // model.feature_batch_size
            with watchdog.operation('close_target_cache', update=True):
                edge_cache = FrozenEdgeFeatures(tuple(targets), torch.cat(vectors), torch.cat(flags))
                del vectors, flags, extracted
                cache_state = {'vectors': edge_cache.vectors, 'has_tokens': edge_cache.has_tokens}
                cache_hashes = tensor_hashes(cache_state)
                save_file(cache_state, output / 'target-features.safetensors')
                require((output / 'target-features.safetensors').stat().st_size <= CACHE_BYTES,
                        'target feature cache disk bound exceeded')
                write(output / 'target-features.json', {'schema': SCHEMA, 'targets_sha256': sha(output / 'targets.json'),
                    'tokens_sha256': sha(output / 'tokens.json'), 'tensor_hashes': cache_hashes,
                    'tensor_file_sha256': sha(output / 'target-features.safetensors'),
                    'parent_forward_batches': parent_batches, 'chunk_size': TARGET_CHUNK,
                    'batch_size': model.feature_batch_size})
            base_cache = LogitCache(); edge_rows = OrderedDict()
            def features(index):
                if index not in edge_rows:
                    edge_rows[index] = subset_targets(inventory[index], targets,
                        edge_cache.vectors, edge_cache.has_tokens)
                edge_rows.move_to_end(index)
                while len(edge_rows) > 32: edge_rows.popitem(last=False)
                return base_cache.get(index, data[index]['candidates']), edge_rows[index]
            model.eval()
            for i, row in enumerate(data):
                with watchdog.operation('initial:' + row['id'], update=True), torch.no_grad():
                    base, target = features(i)
                    result = model(row['input'], groups[row['id']], base_features=base, edge_features=target)
                    audit_forward(result, inventory[i])
                    require(result['scores'].tolist() == before[i]['scores']
                            and torch.count_nonzero(result['residuals']).item() == 0, 'initial parent parity differs')
                    append_jsonl(output / 'before.jsonl', {'id': row['id'], 'scores': result['scores'].tolist(),
                        'shared_edges': len(result['edge_ids']), 'all_path_weights_verified': True,
                        'all_shared_contributions_equal': True})
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                lr=RECIPE['learning_rate'], weight_decay=RECIPE['weight_decay'])
            model.train(); updates = 0; epoch_losses = []
            gradient_keys = ('source_embedding.', 'source_encoder.', 'segment_head.0.', 'segment_head.2.')
            maxima = dict.fromkeys(gradient_keys, 0.)
            for epoch in range(RECIPE['epochs']):
                total = 0.
                for i, row in enumerate(data):
                    with watchdog.operation(f'shared_edge:update:{updates + 1}', update=True):
                        base, target = features(i); optimizer.zero_grad(set_to_none=True)
                        result = model(row['input'], groups[row['id']], base_features=base, edge_features=target)
                        loss = train_loss(result['scores'], row['positive_indices'])
                        require(torch.isfinite(loss).item(), 'nonfinite candidate loss'); loss.backward()
                        norms = {k: sum(p.grad.detach().square().sum().item() for n, p in model.named_parameters()
                            if n.startswith(k) and p.grad is not None) ** .5 for k in gradient_keys}
                        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                            RECIPE['clip_norm'], error_if_nonfinite=True)
                        optimizer.step(); updates += 1; total += loss.item()
                        if updates > 1: maxima = {k: max(maxima[k], norms[k]) for k in maxima}
                        append_jsonl(output / 'updates.jsonl', {'update': updates, 'epoch': epoch, 'id': row['id'],
                            'loss': loss.item(), 'gradient_norm': norm.item(), 'gradient_groups': norms})
                epoch_losses.append(total / len(data))
            require(updates == 2966 and all(v > 0 for v in maxima.values()), 'updates or upstream gradients differ')
            require(parent_before == tensor_hashes(model.parent.state_dict())
                    and all(p.grad is None for p in model.parent.parameters()), 'frozen parent changed')
            require(cache_hashes == tensor_hashes(cache_state), 'frozen isolated target cache mutated')
            model.eval(); correct = gains = losses = 0
            for i, row in enumerate(data):
                with watchdog.operation('fit:' + row['id'], update=True), torch.no_grad():
                    base, target = features(i)
                    result = model(row['input'], groups[row['id']], base_features=base, edge_features=target)
                    audit_forward(result, inventory[i])
                    hit = int(result['scores'].argmax().item() in row['positive_indices'])
                    correct += hit; gains += int(hit and not control[i]['correct'])
                    losses += int(control[i]['correct'] and not hit)
                    append_jsonl(output / 'after.jsonl', {'id': row['id'], 'correct': hit,
                        'loss': train_loss(result['scores'], row['positive_indices']).item(), **plain(result)})
            with watchdog.operation('save_reload', update=True):
                state = model.residual_state_dict(); save_file(state, output / 'residual.safetensors')
                fresh = SharedEdgeResidualRanker(parent)
                fresh.load_residual_state_dict(load_file(output / 'residual.safetensors')); fresh.eval()
                reloaded_cache = load_file(output / 'target-features.safetensors')
                require(tensor_hashes(reloaded_cache) == cache_hashes, 'target cache reload differs')
                reload_rows = []
                for i in sorted(set([0, len(data) - 1] + list(range(0, len(data), 100)))):
                    row = data[i]; base, target = features(i)
                    fresh_target = subset_targets(inventory[i], targets,
                        reloaded_cache['vectors'], reloaded_cache['has_tokens'])
                    with torch.no_grad():
                        a = model(row['input'], groups[row['id']], base_features=base, edge_features=target)
                        b = fresh(row['input'], groups[row['id']], base_features=base, edge_features=fresh_target)
                    require(torch.equal(a['scores'], b['scores'])
                            and torch.equal(a['edge_scores'], b['edge_scores']), 'checkpoint reload differs')
                    reload_rows.append({'id': row['id'], 'scores': a['scores'].tolist(),
                        'reloaded': b['scores'].tolist(), 'edge_scores': a['edge_scores'].tolist(),
                        'reloaded_edge_scores': b['edge_scores'].tolist()})
                write(output / 'reload.json', reload_rows)
            final_resources = watchdog._sample('final')
        require(all(sha(p) == h for p, h in pins.items()), 'pinned input changed during training')
        metadata = {'schema': SCHEMA, 'arm': 'shared_edge', 'architecture': model.architecture,
            'recipe': RECIPE, 'pins': pins, 'residual_sha256': sha(output / 'residual.safetensors'),
            'initial_sha256': sha(output / 'initial.safetensors'), 'selection_sha256': sha(output / 'selection.json')}
        write(output / 'checkpoint.json', metadata)
        report = {'schema': SCHEMA, 'completed': True, 'rows': len(data), 'updates': updates,
            'recipe': RECIPE, 'limits': LIMITS, 'pins': pins, 'architecture': model.architecture,
            'fit_correct': correct, 'control_fit_correct': sum(r['correct'] for r in control),
            'fit_gains': gains, 'fit_losses': losses, 'epoch_mean_loss': epoch_losses,
            'upstream_gradient_maxima': maxima, 'initial_hashes': tensor_hashes(initial),
            'final_hashes': tensor_hashes(state), 'parent_before': parent_before,
            'parent_after': tensor_hashes(model.parent.state_dict()),
            'target_cache_tensor_hashes': cache_hashes, 'target_parent_forward_batches': parent_batches,
            'target_cache_bytes': (output / 'target-features.safetensors').stat().st_size,
            'reload_rows': len(reload_rows), 'baseline_swap_bytes': baseline_swap,
            'final_resources': final_resources, 'contracts': {'fit_only': True, 'hold_opened': False,
                'dev_opened': False, 'test_opened': False, 'production_changed': False,
                'candidate_hidden_states_used_in_residual': False,
                'reloaded_target_cache_used_for_inference': True,
                'all_fit_initial_parent_scores_exact': True, 'all_fit_shared_edge_contributions_equal': True,
                'source_codepoint_weight_mass_verified': True},
            'outputs': {p.name: sha(p) for p in output.iterdir() if p.is_file()}}
        report['elapsed_s'] = time.monotonic() - started
        require(report['elapsed_s'] <= LIMITS['whole_seconds'], 'whole-run time limit exceeded')
        write(output / 'report.json', report)
        print(json.dumps({k: report[k] for k in ('completed', 'updates', 'fit_correct',
            'control_fit_correct', 'fit_gains', 'fit_losses', 'elapsed_s')}, sort_keys=True))
    except BaseException as error:
        write(output / 'failure.json', {'schema': SCHEMA, 'completed': False,
            'error_type': type(error).__name__, 'error': str(error), 'elapsed_s': time.monotonic() - started})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--authorization', type=Path)
    parser.add_argument('--prepare-only', action='store_true'); args = parser.parse_args()
    if args.prepare_only:
        data, _, targets, _, _, _, pins = authenticate_inputs()
        require(args.authorization is None, 'prepare-only does not consume authorization')
        value = {'schema': AUTH_SCHEMA, 'rows': len(data), 'targets': len(targets),
                 'recipe': RECIPE, 'pins': pins}
        write(args.output, value); print(json.dumps({'rows': len(data), 'targets': len(targets)}))
    else:
        require(args.authorization is not None, 'prelaunch authorization required')
        run(args.output, args.authorization)
