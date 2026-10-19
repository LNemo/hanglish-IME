"""One fixed FIT run with local auxiliary loss and a verified historical control."""
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
    DATA, SELECTION, PARENT, PARENT_HASHES, SEED, read, require, rows, sha,
    tensor_hashes, train_loss, write,
)
from scripts.train_source_span_pair import FeatureCache, training_data

SCHEMA = 'local-span-objective-training-v1'
OLD = ROOT / '.omx/artifacts/source-span-paired-training-v1'
PREFERENCES = ROOT / '.omx/artifacts/conditional-span-preferences-fit-v1'
PLAN = ROOT / '.omx/plans/local-span-objective-v1.md'
LIMITS = {'whole_seconds': 1800., 'update_seconds': 60.,
          'rss_bytes': 3 * 1024**3, 'swap_growth_bytes': 512 * 1024**2}
RECIPE = {'seed': SEED, 'epochs': 2, 'learning_rate': .001, 'weight_decay': .01,
          'clip_norm': 1., 'threads': 2, 'dtype': 'float32', 'optimizer': 'AdamW',
          'accumulation': 1, 'local_loss_weight': .25}


class VerifiedFeatureCache(FeatureCache):
    """Read the old immutable cache using its hash-bound manifest."""
    def __init__(self):
        self.directory = OLD / 'features'; self.limit = 32
        self.entries = {int(k): v for k, v in read(OLD / 'feature-manifest.json').items()}
        self.memory = OrderedDict()

    def put(self, index, feature):
        raise RuntimeError('historical features are read-only')


def authenticated_inputs():
    report = read(OLD / 'report.json'); proof = read(OLD / 'independent-verification.json')
    require(report['completed'] is True and proof['verified'] is True
            and proof['run_report_sha256'] == sha(OLD / 'report.json'), 'control proof differs')
    require(report['schema'] == 'source-span-paired-training-v1'
            and report['rows'] == 1483, 'control training identity differs')
    require(all(report['recipe'][k] == v for k, v in RECIPE.items() if k != 'local_loss_weight'),
            'historical control recipe differs')
    require(all(sha(p) == h for p, h in report['pins'].items()), 'control code/data changed')
    require(all(sha(OLD / p) == h for p, h in report['outputs'].items()),
            'control artifacts changed')
    require(all(sha(PARENT / p) == h for p, h in PARENT_HASHES.items()), 'parent changed')
    preparation = read(PREFERENCES / 'report.json')
    evidence = read(PREFERENCES / 'independent-verification.json')
    require(evidence['schema'] == 'conditional-span-preference-independent-verification-v1'
            and evidence['verified'] is True
            and evidence['report_sha256'] == sha(PREFERENCES / 'report.json'),
            'local label proof differs')
    require(preparation['schema'] == 'conditional-span-preference-v1'
            and preparation['scope'] == 'authorized_fit_selected_candidates_only'
            and evidence['preferences_sha256'] == preparation['preferences_sha256']
            and evidence['selection_sha256'] == sha(SELECTION)
            and evidence['fit_paths_sha256'] == sha(DATA / 'fit-paths.jsonl')
            and evidence['counts'] == preparation['counts']
            and preparation['counts']['rows'] == 1483
            and preparation['counts']['distinct_preferences'] == 6889
            and preparation['contextual_pairs_with_both_preferences'] == 0
            and evidence['contracts'] == {
                'accepted_labels_reconstructed_from_frozen_selection': True,
                'all_selected_candidate_paths_expanded_from_compact_edges': True,
                'common_source_boundaries_recomputed': True,
                'exact_preference_pair_set_and_witnesses_recomputed': True,
                'models_or_torch_loaded': False,
                'multi_change_outside_segmentation_whole_source_and_ambiguous_excluded': True,
                'row_and_report_aggregates_recomputed': True,
                'test_hold_and_canonical_labels_accessed': False},
            'local label scope or proof contract differs')
    require(sha(PREFERENCES / 'preferences.jsonl') == preparation['preferences_sha256']
            and all(sha(p) == h for p, h in preparation['pins'].items()), 'local label input changed')
    data, groups = training_data()
    labels = list(rows(PREFERENCES / 'preferences.jsonl'))
    require(len(labels) == len(data) == 1483, 'FIT label row count differs')
    for row, label in zip(data, labels):
        require(all(row[k] == label[k] for k in row), 'FIT local label order/domain differs')
    require(sum(len(r['preferences']) for r in labels) == 6889,
            'frozen local contrast count differs')
    control_scores = list(rows(OLD / 'source_local/after.jsonl'))
    require([r['id'] for r in control_scores] == [r['id'] for r in data],
            'control score order differs')
    files = [PLAN, Path(__file__).resolve(), ROOT / 'hanglish/models/local_span_objective.py',
             ROOT / 'hanglish/models/span_supervision.py', PREFERENCES / 'report.json',
             PREFERENCES / 'independent-verification.json', PREFERENCES / 'preferences.jsonl',
             OLD / 'report.json', OLD / 'independent-verification.json',
             OLD / 'initial.safetensors', OLD / 'feature-manifest.json',
             OLD / 'source_local/residual.safetensors', OLD / 'source_local/after.jsonl']
    pins = dict(report['pins']); pins.update({str(p): sha(p) for p in files})
    return data, groups, labels, control_scores, pins


def combined_loss(scores, positives, local_loss):
    candidate_loss = train_loss(scores, positives)
    return candidate_loss + RECIPE['local_loss_weight'] * local_loss, candidate_loss


def run(output):
    started = time.monotonic()
    # Authentication remains model-free and finishes before model loading.
    data, groups, labels, control_scores, pins = authenticated_inputs()
    import torch
    from safetensors.torch import load_file, save_file
    from hanglish.models.sentence_ranker import SentenceRanker
    from hanglish.models.source_span_residual import SourceSpanResidualRanker
    from hanglish.models.local_span_objective import forward_with_local_preferences
    from scripts.preflight_pko_seq2seq import append_jsonl, system_swap_bytes
    from scripts.source_span_resources import AtomicResourceWatchdog

    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    write(output / 'pins.json', pins)
    write(output / 'selection.json', data)
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True); random.seed(SEED); torch.manual_seed(SEED)
    baseline_swap = system_swap_bytes()
    watchdog = AtomicResourceWatchdog(started, baseline_swap, output / 'resources.jsonl',
                                     interval_s=2., limits=LIMITS)
    try:
        with watchdog:
            with watchdog.operation('load_parent_control', update=True):
                parent = SentenceRanker.load(PARENT)
                torch.manual_seed(SEED); model = SourceSpanResidualRanker(parent)
                parent_before = tensor_hashes(model.parent.state_dict())
                initial = load_file(OLD / 'initial.safetensors')
                model.load_residual_state_dict(load_file(OLD / 'source_local/residual.safetensors'))
                model.eval(); cache = VerifiedFeatureCache()
            # Prove the new forward preserves the actual nonzero historical model.
            for i, row in enumerate(data):
                with watchdog.operation('control_parity:' + row['id'], update=True), torch.no_grad():
                    feature = cache.get(i)
                    old = model(row['input'], groups[row['id']], features=feature)
                    new = forward_with_local_preferences(model, row['input'], groups[row['id']],
                                                         feature, labels[i]['preferences'])
                    saved = feature.logits.new_tensor(control_scores[i]['scores'])
                    require(torch.equal(old['scores'], new['scores'])
                            and torch.equal(new['scores'], saved), 'historical control replay differs')
                    require(torch.equal(old['residuals'], new['residuals']), 'global residual differs')
                    append_jsonl(output / 'control-parity.jsonl', {'id': row['id'],
                        'scores': new['scores'].tolist(), 'local_loss': new['local_loss'].item(),
                        'local_margins': new['local_margins'].detach().tolist()})
            with watchdog.operation('initialize_training', update=True):
                model.load_residual_state_dict(initial)
                require(tensor_hashes(model.residual_state_dict()) == tensor_hashes(initial),
                        'initial state differs from matched control')
                save_file(model.residual_state_dict(), output / 'initial.safetensors')
                optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                    lr=RECIPE['learning_rate'], weight_decay=RECIPE['weight_decay'])
                model.train(); updates = 0
            gradient_groups = ('source_embedding.', 'source_encoder.', 'segment_head.0.', 'segment_head.2.')
            maxima = dict.fromkeys(gradient_groups, 0.); epoch_losses = []
            for epoch in range(RECIPE['epochs']):
                totals = {'total': 0., 'candidate': 0., 'local': 0.}
                for i, row in enumerate(data):
                    with watchdog.operation(f'local_aux:update:{updates + 1}', update=True):
                        optimizer.zero_grad(set_to_none=True)
                        result = forward_with_local_preferences(model, row['input'], groups[row['id']],
                                                               cache.get(i), labels[i]['preferences'])
                        loss, candidate_loss = combined_loss(result['scores'], row['positive_indices'],
                                                            result['local_loss'])
                        require(torch.isfinite(loss).item(), 'nonfinite training loss'); loss.backward()
                        norms = {k: sum(p.grad.detach().square().sum().item()
                            for n, p in model.named_parameters() if n.startswith(k) and p.grad is not None) ** .5
                            for k in gradient_groups}
                        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                            RECIPE['clip_norm'], error_if_nonfinite=True)
                        optimizer.step(); updates += 1
                        values = {'total': loss.item(), 'candidate': candidate_loss.item(),
                                  'local': result['local_loss'].item()}
                        for k in totals: totals[k] += values[k]
                        if updates > 1: maxima = {k: max(maxima[k], norms[k]) for k in maxima}
                        # Root update file is included in the existing atomic watchdog's failure accounting.
                        append_jsonl(output / 'updates.jsonl', {'update': updates, 'epoch': epoch,
                            'id': row['id'], 'losses': values, 'gradient_norm': norm.item(),
                            'gradient_groups': norms})
                epoch_losses.append({k: v / len(data) for k, v in totals.items()})
            require(updates == 2966 and all(v > 0 for v in maxima.values()), 'training/gradient count differs')
            require(tensor_hashes(model.parent.state_dict()) == parent_before
                    and all(p.grad is None for p in model.parent.parameters()), 'frozen parent changed')
            model.eval(); correct = gains = losses = local_correct = local_count = 0
            for i, row in enumerate(data):
                with watchdog.operation('fit:' + row['id'], update=True), torch.no_grad():
                    feature = cache.get(i)
                    result = forward_with_local_preferences(model, row['input'], groups[row['id']],
                                                           feature, labels[i]['preferences'])
                    original = model(row['input'], groups[row['id']], features=feature)
                    require(torch.equal(original['scores'], result['scores']), 'trained inference parity differs')
                    hit = int(result['scores'].argmax().item() in row['positive_indices'])
                    before = control_scores[i]['correct']; correct += hit
                    gains += int(hit and not before); losses += int(before and not hit)
                    margins = result['local_margins']; local_count += margins.numel()
                    local_correct += int((margins > 0).sum().item())
                    append_jsonl(output / 'after.jsonl', {'id': row['id'], 'correct': hit,
                        'scores': result['scores'].tolist(), 'base_scores': result['base_scores'].tolist(),
                        'residuals': result['residuals'].tolist(), 'local_loss': result['local_loss'].item(),
                        'local_margins': margins.tolist()})
            with watchdog.operation('save_reload', update=True):
                state = model.residual_state_dict(); save_file(state, output / 'residual.safetensors')
                fresh = SourceSpanResidualRanker(parent)
                fresh.load_residual_state_dict(load_file(output / 'residual.safetensors')); fresh.eval()
                reload_rows = []
                for i in sorted(set([0, len(data) - 1] + list(range(0, len(data), 100)))):
                    row = data[i]; feature = cache.get(i)
                    with torch.no_grad():
                        a = model(row['input'], groups[row['id']], features=feature)['scores']
                        b = fresh(row['input'], groups[row['id']], features=feature)['scores']
                    require(torch.equal(a, b), 'reload differs')
                    reload_rows.append({'id': row['id'], 'scores': a.tolist(), 'reloaded': b.tolist()})
                write(output / 'reload.json', reload_rows)
            final_resources = watchdog._sample('final')
        require(all(sha(p) == h for p, h in pins.items()), 'pinned input changed during training')
        metadata = {'schema': SCHEMA, 'arm': 'local_aux', 'architecture': model.architecture,
                    'recipe': RECIPE, 'pins': pins, 'residual_sha256': sha(output / 'residual.safetensors'),
                    'initial_sha256': sha(output / 'initial.safetensors'),
                    'selection_sha256': sha(output / 'selection.json')}
        write(output / 'checkpoint.json', metadata)
        report = {'schema': SCHEMA, 'completed': True, 'recipe': RECIPE, 'limits': LIMITS,
            'rows': len(data), 'updates': updates, 'pins': pins, 'epoch_mean_losses': epoch_losses,
            'fit_correct': correct, 'control_fit_correct': sum(r['correct'] for r in control_scores),
            'fit_gains': gains, 'fit_losses': losses, 'local_positive_margins': local_correct,
            'local_margin_count': local_count, 'upstream_gradient_maxima': maxima,
            'parent_before': parent_before, 'parent_after': tensor_hashes(parent.model.state_dict()),
            'initial_hashes': tensor_hashes(initial), 'final_hashes': tensor_hashes(state),
            'historical_control_all_row_parity': True, 'trained_global_all_row_parity': True,
            'reload_rows': len(reload_rows), 'baseline_swap_bytes': baseline_swap,
            'final_resources': final_resources, 'elapsed_s': time.monotonic() - started,
            'contracts': {'fit_only': True, 'dev_opened': False, 'test_opened': False,
                          'hold_opened': False, 'production_changed': False},
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
    parser.add_argument('--prepare-only', action='store_true'); args = parser.parse_args()
    if args.prepare_only:
        selected, _, labels, _, pins = authenticated_inputs()
        value = {'rows': len(selected), 'preferences': sum(len(r['preferences']) for r in labels),
                 'recipe': RECIPE, 'pins': pins}
        write(args.output, value); print(json.dumps({'rows': value['rows'], 'preferences': value['preferences']}))
    else:
        run(args.output)
