from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import struct

import pytest

from scripts import verify_difference_pairwise_lora_training as v
from scripts import difference_pairwise_lora_evaluation as producer


@pytest.fixture(scope='module')
def data():
    return v.load_data()


def predictions(data, improve):
    report = {'predictions': {}, 'pair_inference': {}, 'quality': {}}
    for scope in ('mechanism', 'train640'):
        outputs, evidence = [], []
        items = data['items'][scope]
        for item in items:
            role, row = item['role'], item['row']
            pairs = sorted((p for p in data['pairs'] if p['role'] == role
                and p['row_id'] == row['id'] and scope in p['scopes']),
                key=lambda p: p['alternative_source_index'])
            scores = {}
            for pair in pairs:
                alternative = row['candidates'][pair['alternative_source_index']]['text']
                repair = improve and row['result'] not in row['expected'] and alternative in row['expected']
                logits = [0.0, 1.0 if repair else -1.0]
                scores[pair['pair_id']] = logits
                evidence.append({'scope': scope, 'role': role, 'row_id': row['id'],
                    'alternative_source_index': pair['alternative_source_index'],
                    'pair_id': pair['pair_id'], 'logits': logits, 'seconds': .01})
            outputs.append(producer.resolve_selection(producer.prepare_selection(role, row, pairs), scores))
        report['quality'][scope] = producer.score_predictions(items, outputs,
            expected_rows=175 if scope == 'mechanism' else 640,
            minimum_net_gain=-175 if scope == 'mechanism' else 5)
        report['predictions'][scope] = outputs
        report['pair_inference'][scope] = evidence
    # Exercise saved JSON, where original Python object identity cannot survive.
    return json.loads(json.dumps(report, ensure_ascii=False))


@pytest.mark.parametrize('improve', [False, True])
def test_all_2037_raw_scores_replay_and_metrics_match_independently(data, improve):
    report = predictions(data, improve)
    quality = v.verify_evaluations(report, data)
    total = quality['train640']['metrics']['all']
    assert total['before_top1'] == 447
    assert total['top1'] == (467 if improve else 447)
    assert quality['train640']['gate']['passed'] is improve
    assert total['candidate_absent'] == 166
    assert total['candidate_present'] - total['before_top1'] == 27
    assert total['recoverable_errors'] == 20


@pytest.mark.parametrize('mutation', ['missing_pair', 'nan', 'overflow', 'wrong_scope',
    'wrong_identity', 'chosen', 'candidate', 'raw', 'metrics', 'context'])
def test_prediction_and_raw_evidence_corruption_rejected(data, mutation):
    report = predictions(data, False)
    scores = report['pair_inference']['train640']
    prediction = next(p for p in report['predictions']['train640'] if p['pair_scores'])
    if mutation == 'missing_pair': scores.pop()
    elif mutation == 'nan': scores[0]['logits'][0] = math.nan
    elif mutation == 'overflow': scores[0]['logits'] = [-1.7976931348623157e308, 1.7976931348623157e308]
    elif mutation == 'wrong_scope': scores[0]['scope'] = 'mechanism'
    elif mutation == 'wrong_identity': scores[0]['row_id'] = 'bad'
    elif mutation == 'chosen': prediction['chosen_index'] = 1
    elif mutation == 'candidate': prediction['candidates'][0]['score'] += .01
    elif mutation == 'raw': prediction['raw_bypass_response']['result'] = 'changed'
    elif mutation == 'metrics': report['quality']['train640']['metrics']['all']['top1'] += 1
    elif mutation == 'context': prediction['context'] += ' changed'
    with pytest.raises(v.budget.VerificationError):
        v.verify_evaluations(report, data)


def unit_report(data):
    units, updates = [], []
    by_id = {x['unit_id']: x for x in data['units']}
    for scheduled in data['schedule']:
        unit = by_id[scheduled['unit_id']]
        components = [{'component_index': i, 'target': ref['target'], 'weight': ref['weight'],
            'loss': .5, 'gradient_seconds': .1, 'pair_id': ref['pair_id']}
            for i, ref in enumerate(unit['references'])]
        units.append({'order': unit['order'], 'unit_id': unit['unit_id'],
            'update': scheduled['update'], 'loss': .5, 'unit_gradient_seconds': .5, 'components': components})
    for i in range(1, 16):
        ids = [x['unit_id'] for x in units if x['update'] == i]
        updates.append({'update': i, 'unit_ids': ids, 'actual_unit_count': len(ids),
            'gradient_norm_before_clip': 2., 'optimizer_seconds': .1, 'complete_update_seconds': 3.})
    return {'units': units, 'updates': updates}


def test_balanced59_units232_components15_updates_and_tail_verified(data):
    report = unit_report(data)
    v.verify_units(report, data)
    assert report['updates'][-1]['actual_unit_count'] == 3


@pytest.mark.parametrize('mutation', ['duplicate', 'weight', 'loss', 'tail', 'deadline', 'interval', 'nonfinite'])
def test_training_schedule_and_numeric_corruption_rejected(data, mutation):
    report = unit_report(data)
    if mutation == 'duplicate': report['units'][1] = deepcopy(report['units'][0])
    elif mutation == 'weight': report['units'][0]['components'][0]['weight'] = .4
    elif mutation == 'loss': report['units'][0]['loss'] = .8
    elif mutation == 'tail': report['updates'][-1]['actual_unit_count'] = 4
    elif mutation == 'deadline': report['updates'][0]['complete_update_seconds'] = 120.01
    elif mutation == 'interval': report['units'][0]['unit_gradient_seconds'] = .1
    elif mutation == 'nonfinite': report['updates'][0]['gradient_norm_before_clip'] = math.nan
    with pytest.raises(v.budget.VerificationError): v.verify_units(report, data)


def resource_report(data):
    stages, _ = v.expected_resource_stages(data)
    records = []
    for i, stage in enumerate(stages):
        observed = {'rss_peak_bytes': 1000, 'mlx_active_bytes': 1000, 'mlx_cache_bytes': 0,
                    'mlx_peak_bytes': 1000, 'free_percent': 40, 'swap_used_bytes': 100,
                    'swap_growth_bytes': 0, 'elapsed_seconds': i * .1}
        records.append({'stage': stage, 'action': 'none', 'pre': observed, 'post': None,
                        'errors': [], 'cleanup_seconds': 0, 'checkpoint_seconds': .01})
    return {'baseline': {'free_percent': 40, 'swap_used_bytes': 100},
            'resource_observations': records, 'worker_fit_elapsed_seconds': 1000.}, stages


def test_resource_stages_complete_and_terminal_cleanup_bounded(data):
    report, stages = resource_report(data)
    result = v.verify_resources(report, stages)
    assert result['observations'] == len(stages)
    assert stages[-1] == 'terminal_cleanup'


@pytest.mark.parametrize('mutation', ['missing', 'reorder', 'rss', 'swap', 'cache', 'time', 'terminal', 'prebreach'])
def test_resource_limits_and_missing_stage_rejected(data, mutation):
    report, stages = resource_report(data)
    rows = report['resource_observations']
    if mutation == 'missing': rows.pop()
    elif mutation == 'reorder': rows[0], rows[1] = rows[1], rows[0]
    elif mutation == 'rss': rows[0]['pre']['rss_peak_bytes'] = 7 * 1024**3
    elif mutation == 'swap': rows[0]['pre']['swap_growth_bytes'] = 1
    elif mutation == 'cache': rows[0]['pre']['mlx_cache_bytes'] = 300 * 1024**2
    elif mutation == 'time': rows[-1]['pre']['elapsed_seconds'] = 21601
    elif mutation == 'terminal': report['worker_fit_elapsed_seconds'] = 1
    elif mutation == 'prebreach':
        rows[0]['action'] = 'clear_unused_mlx_cache'
        rows[0]['post'] = deepcopy(rows[0]['pre'])
        rows[0]['pre']['free_percent'] = 9
    with pytest.raises(v.budget.VerificationError): v.verify_resources(report, stages)


def test_missing_terminal_rejected_before_prelaunch_or_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(v, 'OUTPUT', tmp_path)
    def forbidden(): raise AssertionError('must not reach labels or prelaunch')
    monkeypatch.setattr(v, 'load_data', forbidden)
    monkeypatch.setattr(v, 'validate_prelaunch', forbidden)
    with pytest.raises(v.budget.VerificationError, match='terminal execution evidence missing'):
        v.verify_training(tmp_path)


def test_verifier_has_no_producer_or_mlx_import():
    import ast
    tree = ast.parse(Path(v.__file__).read_text())
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any(name and ('train_difference' in name or 'difference_pairwise_lora_evaluation' in name
                            or name.startswith('mlx')) for name in imports)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(v.canonical(value) + b'\n')


def disk_fixture(tmp_path, monkeypatch, data, improve):
    output = tmp_path / v.SOLE_OUTPUT
    output.mkdir(parents=True)
    monkeypatch.setattr(v, 'ROOT', tmp_path)
    monkeypatch.setattr(v, 'OUTPUT', output)
    monkeypatch.setattr(v, 'PRELAUNCH', tmp_path / 'prelaunch.json')
    monkeypatch.setattr(v, 'BUDGET_PROOF', tmp_path / 'budget.json')
    monkeypatch.setattr(v, 'RECOVERY_PROOF', tmp_path / 'recovery.json')
    write_json(v.BUDGET_PROOF, {'synthetic': True})
    write_json(v.RECOVERY_PROOF, {'synthetic': True})
    (tmp_path / 'bound.txt').write_text('bound')
    inventory = {'bound.txt': v.sha(tmp_path / 'bound.txt')}
    prelaunch = {'inventory': inventory,
        'budget_proof': {'path': 'budget.json', 'sha256': v.sha(v.BUDGET_PROOF)},
        'recovery_proof': {'path': 'recovery.json', 'sha256': v.sha(v.RECOVERY_PROOF)}}
    write_json(v.PRELAUNCH, prelaunch)
    monkeypatch.setattr(v, 'validate_prelaunch', lambda: prelaunch)
    monkeypatch.setattr(v, 'load_data', lambda: data)
    launch = {'schema': 'difference-pairwise-lora-launch-v2', 'logical_fit_id': v.LOGICAL_FIT_ID,
              'sole_output': v.SOLE_OUTPUT, 'consumed': True, 'released': False,
              'prelaunch': {'path': 'prelaunch.json', 'sha256': v.sha(v.PRELAUNCH)},
              'budget_proof_sha256': v.sha(v.BUDGET_PROOF),
              'recovery_proof_sha256': v.sha(v.RECOVERY_PROOF), 'worker_claim': 'synthetic'}
    write_json(output / 'launch.json', launch)
    worker = {'schema': 'difference-pairwise-lora-worker-start-v2', 'logical_fit_id': v.LOGICAL_FIT_ID,
              'sole_output': v.SOLE_OUTPUT, 'consumed': True, 'released': False,
              'launch_sha256': v.sha(output / 'launch.json'), 'worker_claim': 'synthetic', 'worker_pid': 123}
    write_json(output / 'worker-start.json', worker)
    report = predictions(data, improve)
    report.update(unit_report(data))
    resource, stages = resource_report(data)
    report.update(resource)
    report.update({'schema': 'difference-pairwise-lora-training-v2', 'logical_fit_id': v.LOGICAL_FIT_ID,
        'sole_output': v.SOLE_OUTPUT, 'completed': True, 'execution_valid': True,
        'quality_passed': improve, 'terminal_status': 'quality_passed' if improve else 'quality_rejected',
        'worker_claim': 'synthetic', 'worker_start': worker, 'prelaunch_sha256': v.sha(v.PRELAUNCH),
        'launch_sha256': v.sha(output / 'launch.json'), 'fit_start_monotonic': 100.,
        'real_training_run': True, 'fresh_base_initialized': True, 'old_adapter_loaded': False,
        'disposable_adapter_loaded': False, 'one_epoch': True,
        'eligibility': {'evaluation': False, 'resume': False, 'deploy': False},
        'limits': v.LIMITS, 'recipe': v.RECIPE, 'counts_expected': v.COUNTS,
        'package_versions': v.budget.VERSIONS,
        'data': {'manifest_sha256': v.budget.MANIFEST_SHA256, 'proof_sha256': v.budget.DATA_PROOF_SHA256},
        'budget_proof': prelaunch['budget_proof'], 'recovery_proof': prelaunch['recovery_proof'],
        'source_hashes_before': inventory,
        'source_hashes_after': inventory, 'independent_training_verification_required': True,
        'external_execution_receipt_required': True})
    adir = output / 'adapter'; adir.mkdir()
    offset, header = 0, {}
    for name in v.budget.EXPECTED_NAMES:
        shape = v.budget.EXPECTED_SHAPES[name]
        end = offset + math.prod(shape) * 4
        header[name] = {'dtype': 'F32', 'shape': shape, 'data_offsets': [offset, end]}
        offset = end
    h = json.dumps(header).encode()
    ap, cp = adir / 'adapters.safetensors', adir / 'adapter_config.json'
    ap.write_bytes(struct.pack('<Q', len(h)) + h + bytes(offset))
    write_json(cp, {'fine_tune_type': 'lora', 'num_layers': 4, 'lora_parameters': v.budget.RECIPE['lora_config']})
    hashes = v.budget.read_safetensors(ap)['hashes']
    report['initial_trainable_tensor_hashes'] = dict.fromkeys(hashes, 'a' * 64)
    for name in ('final_trainable_tensor_hashes', 'saved_trainable_tensor_hashes', 'reloaded_trainable_tensor_hashes'):
        report[name] = hashes
    report['adapter'] = {'path': str(ap.relative_to(tmp_path)), 'sha256': v.sha(ap),
        'config_path': str(cp.relative_to(tmp_path)), 'config_sha256': v.sha(cp)}
    representatives = v.expected_resource_stages(data)[1]
    report['reload'] = [{'pair_id': p, 'expected_logits': [0., 1.], 'actual_logits': [0., 1.],
                         'close': True, 'same_argmax': True} for p in representatives]
    events = []
    for update in report['updates']:
        events.extend({'event': 'unit_complete', **u} for u in report['units'] if u['update'] == update['update'])
        events.append({'event': 'update_complete', **update})
    for scope in ('mechanism', 'train640'):
        for i, item in enumerate(data['items'][scope], 1):
            count = sum(p['role'] == item['role'] and p['row_id'] == item['row']['id'] and scope in p['scopes'] for p in data['pairs'])
            events.append({'event': 'evaluation_row', 'scope': scope, 'ordinal': i,
                'role': item['role'], 'row_id': item['row']['id'], 'pair_count': count})
    events.append({'event': 'terminal', 'terminal_status': report['terminal_status'],
        'exit_code': 0 if improve else 2, 'elapsed_seconds': report['worker_fit_elapsed_seconds']})
    (output / 'progress.jsonl').write_bytes(b''.join(v.canonical(x) + b'\n' for x in events))
    write_json(output / 'report.json', report)
    external = {'schema': 'difference-pairwise-lora-external-execution-v2',
        'logical_fit_id': v.LOGICAL_FIT_ID, 'sole_output': v.SOLE_OUTPUT,
        'worker_claim': 'synthetic', 'worker_pid': 123,
        'prelaunch_sha256': v.sha(v.PRELAUNCH), 'launch_sha256': v.sha(output / 'launch.json'),
        'budget_proof_sha256': v.sha(v.BUDGET_PROOF),
        'recovery_proof_sha256': v.sha(v.RECOVERY_PROOF),
        'terminal_report_sha256': v.sha(output / 'report.json'),
        'worker_interval_valid': True, 'deadline_terminated': False, 'spawn_error': None,
        'additional_model_work_authorized': False, 'fit_limit_seconds': 21600,
        'fit_start_monotonic': 100., 'fit_end_monotonic': 1101., 'fit_elapsed_seconds': 1001.,
        'worker_exit_status': 0 if improve else 2, 'authoritative_success': improve, 'quality_rejected': not improve}
    write_json(output / 'external-execution.json', external)
    return output


@pytest.mark.parametrize('improve', [False, True])
def test_full_disk_evidence_valid_quality_pass_or_rejection(tmp_path, monkeypatch, data, improve):
    output = disk_fixture(tmp_path, monkeypatch, data, improve)
    proof = v.verify_training(output)
    assert proof['verification_passed'] is True
    assert proof['quality_gate_passed'] is improve
    assert proof['downstream_evaluation_allowed'] is improve
    assert proof['promotion_allowed'] is False
    assert proof['resume_allowed'] is False
    assert proof['adapter']['scalars'] == 327680


@pytest.mark.parametrize('mutation', ['late_exit', 'worker_identity', 'progress', 'adapter', 'report_pin', 'source'])
def test_complete_disk_corruption_rejected(tmp_path, monkeypatch, data, mutation):
    output = disk_fixture(tmp_path, monkeypatch, data, False)
    if mutation == 'late_exit':
        path = output / 'external-execution.json'; x = v.read_json(path)
        x.update(fit_end_monotonic=21701., fit_elapsed_seconds=21601.)
        write_json(path, x)
    elif mutation == 'worker_identity':
        path = output / 'worker-start.json'; x = v.read_json(path); x['worker_pid'] += 1; write_json(path, x)
    elif mutation == 'progress':
        path = output / 'progress.jsonl'; path.write_bytes(path.read_bytes().split(b'\n', 1)[1])
    elif mutation == 'adapter':
        path = output / 'adapter/adapters.safetensors'; path.write_bytes(path.read_bytes()[:-4] + struct.pack('<f', math.nan))
    elif mutation == 'report_pin':
        path = output / 'report.json'; x = v.read_json(path); x['quality_passed'] = True; write_json(path, x)
    elif mutation == 'source': (tmp_path / 'bound.txt').write_text('changed')
    with pytest.raises(v.budget.VerificationError): v.verify_training(output)


def test_actual_budget_and_source_closure_prelaunch(tmp_path, monkeypatch):
    monkeypatch.setattr(v, 'ROOT', tmp_path)
    monkeypatch.setattr(v, 'PRELAUNCH', tmp_path / 'prelaunch.json')
    monkeypatch.setattr(v, 'BUDGET_PROOF', tmp_path / 'budget.json')
    monkeypatch.setattr(v, 'RECOVERY_PROOF', tmp_path / 'recovery.json')
    monkeypatch.setattr(v.recovery, 'RECOVERY_PROOF_PATH', 'recovery.json')
    sources = {}
    for index in range(48):
        name = f'source/{index}.txt'; path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(str(index))
        sources[name] = v.sha(path)
    actual = {'source_inventory': {'hashes': sources}}
    budget_receipt = {'schema': v.budget.SCHEMA, 'payload': actual,
                      'digest': v.budget.digest_value(actual)}
    write_json(v.BUDGET_PROOF, budget_receipt)
    monkeypatch.setattr(v.budget, 'verify_budget', lambda: actual)
    recovery_payload = {'preflight': {'path': 'preflight.json', 'interpreter': {'synthetic': True}},
        'checks': {'path': 'checks.json'}, 'execution_review': {'path': 'review.md'},
        'execution_sources': {}}
    for name in ('preflight.json', 'checks.json', 'review.md'):
        (tmp_path / name).write_text(name)
    write_json(v.RECOVERY_PROOF, {'synthetic': True})
    monkeypatch.setattr(v.recovery, 'validate_recovery_proof', lambda path: recovery_payload)
    inventory = dict(sources)
    for name in v.ADDITIONS:
        path = tmp_path / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(name)
        inventory[name] = v.sha(path)
    bp = {'path': str(v.BUDGET_PROOF.relative_to(v.ROOT)), 'sha256': v.sha(v.BUDGET_PROOF)}
    inventory[bp['path']] = bp['sha256']
    rp = {'path': 'recovery.json', 'sha256': v.sha(v.RECOVERY_PROOF)}
    inventory[rp['path']] = rp['sha256']
    for name in ('preflight.json', 'checks.json', 'review.md'):
        inventory[name] = v.sha(tmp_path / name)
    payload = {'schema': v.recovery.PRELAUNCH_SCHEMA, 'passed': True,
        'launch_allowed': True, 'logical_fit_id': v.LOGICAL_FIT_ID,
        'sole_output': v.SOLE_OUTPUT, 'package_versions': v.budget.VERSIONS,
        'budget_proof': bp, 'recovery_proof': rp,
        'worker_python': recovery_payload['preflight']['interpreter'], 'inventory': inventory}
    write_json(v.PRELAUNCH, {'payload': payload, 'digest': v.budget.digest_value(payload)})
    assert v.validate_prelaunch() == payload
    del payload['inventory']['scripts/verify_difference_pairwise_lora_training.py']
    write_json(v.PRELAUNCH, {'payload': payload, 'digest': v.budget.digest_value(payload)})
    with pytest.raises(v.budget.VerificationError, match='inventory is incomplete'): v.validate_prelaunch()
