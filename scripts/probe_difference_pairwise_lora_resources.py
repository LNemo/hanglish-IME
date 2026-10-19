"""Measure the approved balanced pairwise LoRA recipe on real prompt shapes.

Only synthetic KEEP/SWITCH labels are optimized. The disposable adapter is
never a training initialization or a production candidate.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import difference_pairwise_lora_model as pairwise
from scripts import long_context_lora_model as shared
from scripts import probe_long_context_lora_resources as system
from scripts.long_context_lora_resources import observe_resources, ResourceObservationError

DATA = ROOT / '.omx/artifacts/difference-pairwise-lora-data-v1'
DATA_PROOF = ROOT / '.omx/artifacts/difference-pairwise-lora-data-v1-proof.json'
MANIFEST_SHA = '32e83b145e71de39912e122f9925c9cf362875a5206e49dddcb7611a464be132'
DATA_PROOF_SHA = '03943e073ec333b4e8a8e8386ace042c0c3dda6370cdc5b284961262bd2dea72'
MODEL = ROOT / 'data/external/qwen3-4b-mlx-4bit'
VERSIONS = {'mlx': '0.31.2', 'mlx-lm': '0.31.3', 'transformers': '5.12.1'}
LIMITS = {'rss_bytes': 6 * 1024**3, 'mlx_peak_bytes': 6 * 1024**3,
          'cache_bytes': 256 * 1024**2, 'whole_seconds': 900., 'update_seconds': 120.,
          'minimum_system_free_percent': 10., 'maximum_swap_growth_bytes': 512 * 1024**2}
DEFAULT_OUTPUT = ROOT / '.omx/artifacts/difference-pairwise-lora-resource-v1'


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    return system.sha256_file(Path(path))


def key(path):
    return str(Path(path).resolve().relative_to(ROOT))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def representatives_and_schedule(token_rows):
    require(len(token_rows) == 1928, 'expected all 1928 qualified pairs')
    require(len({row['pair_id'] for row in token_rows}) == len(token_rows), 'duplicate pairs')
    for row in token_rows:
        require(row['prompt_tokens'] == len(row['tokens'])
                and 0 < row['prompt_tokens'] <= 1024
                and digest(row['tokens']) == row['tokens_sha256']
                and row['digit_continuation_ids'] == {'1': 16, '2': 17}, 'token row differs')
    ordered = sorted(token_rows, key=lambda row: (row['prompt_tokens'], row['pair_id']))
    n = len(ordered)
    ranks = (0, (n - 1) // 2, math.ceil(.95 * n) - 1, n - 1)
    representatives = [{name: ordered[rank][name]
                        for name in ('pair_id', 'prompt_tokens', 'tokens_sha256')}
                       | {'quantile': quantile}
                       for quantile, rank in zip(('min', 'median', 'p95', 'max'), ranks)]
    schedule, cursor = [], 0
    for index in range(20):
        targets = ['SWITCH'] + ['KEEP'] * (2 if index % 5 == 4 else 3)
        weights = pairwise.unit_weights(targets)
        components = []
        for target, weight in zip(targets, weights):
            shape = representatives[cursor % 4]
            components.append({name: shape[name]
                               for name in ('pair_id', 'prompt_tokens', 'tokens_sha256')}
                              | {'target': target, 'weight': weight})
            cursor += 1
        schedule.append({'unit': index + 1, 'update': index // 4 + 1,
                         'components': components})
    return representatives, schedule


def estimate_seconds(counts, timing):
    require(counts['repair_units'] == 59 and counts['updates'] == 15
            and counts['E_mechanism'] == 481 and counts['E_train640'] == 1556,
            'qualified data estimate counts differ')
    required = ('component_gradient_p95', 'unit_gradient_p95', 'optimizer_update_p95',
                'pair_inference_p95', 'save_reload_finalization_seconds')
    require(all(type(timing[name]) in (float, int) and math.isfinite(timing[name])
                and timing[name] >= 0 for name in required), 'invalid measured timings')
    seconds = 1.5 * (counts['repair_units'] * timing['unit_gradient_p95']
                     + counts['updates'] * timing['optimizer_update_p95']
                     + (counts['E_mechanism'] + counts['E_train640']) * timing['pair_inference_p95']
                     + timing['save_reload_finalization_seconds']) + 300
    return {'seconds': seconds, 'within_7200': seconds <= 7200}


def load_qualified_data():
    require(file_sha(DATA / 'manifest.json') == MANIFEST_SHA
            and file_sha(DATA_PROOF) == DATA_PROOF_SHA, 'qualified data/proof pin differs')
    manifest = json.loads((DATA / 'manifest.json').read_bytes())
    proof = json.loads(DATA_PROOF.read_bytes())
    require(proof['digest'] == digest(proof['payload']) and proof['payload']['passed'] is True
            and proof['payload']['stage'] == 'data'
            and proof['payload']['data_manifest_sha256'] == MANIFEST_SHA,
            'independent data proof differs')
    for filename, sha in manifest['file_hashes'].items():
        require(file_sha(DATA / filename) == sha, f'data artifact changed: {filename}')
    for name, sha in manifest['source_hashes_after'].items():
        require(file_sha(ROOT / name) == sha, f'data source changed: {name}')
    tokens = [json.loads(line) for line in (DATA / 'tokens.jsonl').read_bytes().splitlines()]
    return manifest, tokens


def capture_sources(manifest):
    paths = set(manifest['source_hashes_after'])
    paths.update(key(DATA / name) for name in manifest['file_hashes'])
    paths.update((key(DATA / 'manifest.json'), key(DATA_PROOF), key(Path(__file__)),
                  'scripts/difference_pairwise_lora_model.py',
                  'scripts/long_context_lora_model.py', 'scripts/long_context_lora_resources.py',
                  'scripts/probe_long_context_lora_resources.py',
                  'scripts/verify_difference_pairwise_lora_resources.py'))
    return {name: file_sha(ROOT / name) for name in sorted(paths)}


def check_logits(left, right, *, reload=False):
    require(len(left) == len(right) == 2 and all(math.isfinite(x) for x in left + right),
            'nonfinite or malformed digit logits')
    require(left.index(max(left)) == right.index(max(right)), 'digit argmax differs')
    require(all(abs(a - b) <= (1e-5 + 1e-5 * abs(b) if reload else .125)
                for a, b in zip(left, right)), 'digit logits exceed parity tolerance')


def deadline_exceeded(_signum, _frame):
    raise TimeoutError('900s resource deadline')


def run(output):
    require(not output.exists(), 'resource output already exists')
    output.mkdir(parents=True)
    started = time.monotonic()
    report = {'schema': 'difference-pairwise-lora-resource-v1', 'completed': False,
              'resource_feasible': False, 'terminal_status': 'rejected',
              'synthetic_only': True, 'real_training_run': False, 'old_adapter_loaded': False,
              'limits': LIMITS, 'resource_observations': [], 'units': [], 'updates': [],
              'inference': [], 'equivalence': [], 'reload': []}
    report['numerics'] = {'component_losses_finite': 0, 'component_gradients_finite': 0,
                          'unit_losses_finite': 0, 'unit_gradients_finite': 0,
                          'optimizer_states_finite': 0}
    report['recipe'] = {'seed': shared.SEED, 'lora_layers': shared.LORA_LAYERS,
                        'lora_config': shared.LORA_CONFIG, 'learning_rate': shared.LEARNING_RATE,
                        'adam_betas': list(shared.ADAM_BETAS), 'adam_eps': shared.ADAM_EPS,
                        'bias_correction': True, 'weight_decay': 0.,
                        'accumulation_units': 4, 'clip_norm': shared.CLIP_NORM}
    model = tokenizer = optimizer = value_and_grad = accumulated = result = None
    mx = None
    old_alarm = signal.signal(signal.SIGALRM, deadline_exceeded)
    signal.setitimer(signal.ITIMER_REAL, LIMITS['whole_seconds'])
    try:
        manifest, token_rows = load_qualified_data()
        representatives, schedule = representatives_and_schedule(token_rows)
        tokens = {row['pair_id']: row['tokens'] for row in token_rows}
        before = capture_sources(manifest)
        versions = {name: importlib.metadata.version(name) for name in VERSIONS}
        require(versions == VERSIONS, 'resource package versions differ')
        baseline = system.system_resources()
        report.update({'data': {'manifest': {'path': key(DATA / 'manifest.json'), 'sha256': MANIFEST_SHA},
                                 'proof': {'path': key(DATA_PROOF), 'sha256': DATA_PROOF_SHA}},
                       'source_hashes_before': before, 'versions': versions,
                       'baseline': baseline, 'representatives': representatives, 'schedule': schedule})
        import mlx.core as mx
        from mlx_lm.utils import load
        from mlx_lm.tuner.utils import load_adapters
        mx.set_memory_limit(LIMITS['mlx_peak_bytes'])
        mx.set_cache_limit(LIMITS['cache_bytes'])

        def checkpoint(stage, update_started=None):
            try:
                _, observation = observe_resources(stage=stage, mx=mx, started=started,
                    baseline=baseline, limits=LIMITS, system_reader=system.system_resources,
                    rss_reader=system.rss_bytes)
            except ResourceObservationError as error:
                report['resource_observations'].append(error.record)
                raise
            report['resource_observations'].append(observation)
            system.write_progress(output / 'progress.jsonl', 'resource', observation=observation)
            if update_started is not None:
                require(time.monotonic() - update_started <= 120, 'complete update exceeds120s')

        checkpoint('before_model_load')
        model, tokenizer = load(str(MODEL), tokenizer_config={'trust_remote_code': False})
        checkpoint('model_loaded')
        model.eval()
        flat = shared.configure_fresh_lora(mx, model)
        report['initial_trainable_tensor_hashes'] = shared.tensor_hashes(flat)
        report['trainable'] = {'names': sorted(flat), 'scalar_count': sum(x.size for x in flat.values())}
        del flat
        optimizer = shared.create_optimizer()
        value_and_grad = pairwise.make_component_value_and_grad(mx, model)
        model.train()
        checkpoint('fresh_adapter_initialized')
        for update in range(1, 6):
            update_start = time.monotonic()
            accumulated = None
            unit_ids = []
            for unit in schedule[(update - 1) * 4:update * 4]:
                unit_start = time.monotonic()
                components = [{'tokens': tokens[item['pair_id']], 'target': item['target']}
                              for item in unit['components']]
                def observe_component(record):
                    # The primitive calls this only after loss and all gradient checks.
                    checkpoint(f"unit_{unit['unit']}_component_{record['component_index']}", update_start)
                    report['numerics']['component_losses_finite'] += 1
                    report['numerics']['component_gradients_finite'] += 1

                result = pairwise.materialize_unit(mx, model, components, value_and_grad,
                                                  observe=observe_component)
                require(math.isfinite(result['loss']), 'unit loss is nonfinite')
                pairwise.require_finite_tree(mx, result['gradient'], 'completed unit gradient')
                report['numerics']['unit_losses_finite'] += 1
                report['numerics']['unit_gradients_finite'] += 1
                accumulated = shared.add_materialized_gradients(mx, accumulated, result['gradient'])
                pairwise.require_finite_tree(mx, accumulated, 'update unit accumulator')
                checkpoint(f"unit_{unit['unit']}_complete", update_start)
                unit_record = {'unit': unit['unit'], 'update': update, 'loss': result['loss'],
                               'components': result['components'],
                               'unit_gradient_seconds': time.monotonic() - unit_start}
                report['units'].append(unit_record)
                unit_ids.append(unit['unit'])
                result = None
            optimizer_start = time.monotonic()
            norm = pairwise.apply_unit_update(mx, model, optimizer, accumulated, len(unit_ids))
            report['numerics']['optimizer_states_finite'] += 1
            checkpoint(f'update_{update}_complete', update_start)
            record = {'update': update, 'unit_ids': unit_ids, 'gradient_norm_before_clip': norm,
                      'optimizer_seconds': time.monotonic() - optimizer_start,
                      'complete_update_seconds': time.monotonic() - update_start}
            report['updates'].append(record)
            system.write_progress(output / 'progress.jsonl', 'update', **record)
            accumulated = None
        model.eval()
        final = shared.require_trainable_contract(model)
        report['final_trainable_tensor_hashes'] = shared.tensor_hashes(final)
        require(report['initial_trainable_tensor_hashes'] != report['final_trainable_tensor_hashes'],
                'synthetic adapter did not change')
        del final
        for shape in representatives:
            selected = shared.score_choices(mx, model, tokens[shape['pair_id']], (16, 17))
            ordinary = shared.ordinary_forward_choice_scores(mx, model, tokens[shape['pair_id']], (16, 17))
            mx.eval(selected, ordinary)
            left, right = selected.tolist(), ordinary.tolist()
            check_logits(left, right)
            report['equivalence'].append({'pair_id': shape['pair_id'],
                                           'selected_logits': left, 'ordinary_logits': right})
            del selected, ordinary
            checkpoint(f"equivalence_{shape['quantile']}")
        for shape in representatives:
            for repetition in range(1, 11):
                inference_start = time.monotonic()
                scores = shared.score_choices(mx, model, tokens[shape['pair_id']], (16, 17))
                mx.eval(scores)
                values = scores.tolist()
                require(all(math.isfinite(x) for x in values), 'nonfinite inference')
                checkpoint(f"inference_{shape['quantile']}_{repetition}")
                report['inference'].append({'pair_id': shape['pair_id'], 'repetition': repetition,
                                            'seconds': time.monotonic() - inference_start, 'logits': values})
                del scores
        finalization_started = time.monotonic()
        adapter_dir = output / 'adapter'
        adapter_dir.mkdir()
        adapter_path = adapter_dir / 'adapters.safetensors'
        report['saved_trainable_tensor_hashes'] = shared.save_adapter(mx, model, adapter_path)
        config_path = adapter_dir / 'adapter_config.json'
        config_path.write_text(json.dumps({'fine_tune_type': 'lora', 'num_layers': shared.LORA_LAYERS,
                                          'lora_parameters': shared.LORA_CONFIG}, sort_keys=True) + '\n')
        report['adapter'] = {'path': key(adapter_path), 'sha256': file_sha(adapter_path),
                             'config_path': key(config_path), 'config_sha256': file_sha(config_path)}
        checkpoint('adapter_saved')
        model = tokenizer = optimizer = value_and_grad = accumulated = result = None
        gc.collect()
        mx.clear_cache()
        checkpoint('first_model_released')
        model, tokenizer = load(str(MODEL), tokenizer_config={'trust_remote_code': False})
        model.freeze()
        load_adapters(model, str(adapter_dir))
        model.eval()
        report['reloaded_trainable_tensor_hashes'] = shared.tensor_hashes(shared.require_trainable_contract(model))
        require(report['final_trainable_tensor_hashes'] == report['saved_trainable_tensor_hashes']
                == report['reloaded_trainable_tensor_hashes'], 'saved/reloaded tensor hashes differ')
        for shape, reference in zip(representatives, report['equivalence']):
            actual = shared.score_choices(mx, model, tokens[shape['pair_id']], (16, 17))
            mx.eval(actual)
            values = actual.tolist()
            check_logits(values, reference['selected_logits'], reload=True)
            report['reload'].append({'pair_id': shape['pair_id'],
                                     'expected_logits': reference['selected_logits'], 'actual_logits': values})
            del actual
            checkpoint(f"reload_{shape['quantile']}")
        model = tokenizer = None
        gc.collect()
        mx.clear_cache()
        checkpoint('final_model_released')
        after = capture_sources(manifest)
        require(before == after, 'resource sources changed during probe')
        report['source_hashes_after'] = after
        checkpoint('terminal')
        finalization_end = time.monotonic()
        report['finalization_interval'] = {'start_elapsed': finalization_started - started,
                                           'end_elapsed': finalization_end - started}
        timing = {'component_gradient_p95': system.p95([component['gradient_seconds']
                    for unit in report['units'] for component in unit['components']]),
                  'unit_gradient_p95': system.p95([unit['unit_gradient_seconds'] for unit in report['units']]),
                  'optimizer_update_p95': system.p95([item['optimizer_seconds'] for item in report['updates']]),
                  'pair_inference_p95': system.p95([item['seconds'] for item in report['inference']]),
                  'save_reload_finalization_seconds': finalization_end - finalization_started}
        report.update({'timing_summary': timing, 'runtime_estimate': estimate_seconds(manifest['counts'], timing),
                       'completed': True, 'resource_feasible': True, 'terminal_status': 'success'})
    except BaseException as error:
        report.update({'failure_type': type(error).__name__, 'failure': str(error)})
    finally:
        model = tokenizer = optimizer = value_and_grad = accumulated = result = None
        gc.collect()
        if mx is not None:
            try:
                mx.synchronize()
                mx.clear_cache()
            except Exception as error:
                report['cleanup_error'] = str(error)
                report.update({'completed': False, 'resource_feasible': False, 'terminal_status': 'rejected'})
        report['elapsed_seconds'] = time.monotonic() - started
        if report['elapsed_seconds'] > 900:
            report.update({'completed': False, 'resource_feasible': False, 'terminal_status': 'rejected',
                           'failure': 'whole resource probe exceeds900s'})
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_alarm)
        (output / 'report.json').write_bytes(canonical(report) + b'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run(args.output)
    print(json.dumps({name: report.get(name) for name in
                      ('completed', 'resource_feasible', 'failure', 'runtime_estimate', 'elapsed_seconds')}))
    return 0 if report['completed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
