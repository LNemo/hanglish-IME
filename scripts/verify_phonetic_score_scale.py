"""Independent artifact replay; no trainer/core imports or neural inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('fixed_unscaled', 'learned_unscaled', 'fixed_scaled', 'learned_scaled')
ANCHOR = np.array([1., 0., 0., 0., 0.])
DATA = ROOT / '.omx/artifacts/phonetic-choice-features-v1/rows.jsonl'
DATA_SHA = '20071a1874e057f346b4d3f131767c3a13182806576c98ac773c09f389ebccec'


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=2e-8)


def fold(row):
    return int(hashlib.sha256(('phonetic-score-scale-v1:' + row['component_id']).encode()).hexdigest(), 16) % 4


def scale_for(rows, scaled):
    if not scaled:
        return np.ones(4)
    result = np.sqrt(np.mean([np.var(row['features'], axis=0) for row in rows], axis=0))
    return np.where(result > 1e-12, result, 1.)


def row_values(row, parameters, scale):
    matrix = np.column_stack((row['base_scores'], np.asarray(row['features']) / scale))
    scores = matrix @ parameters
    positive = row['positive_indices']
    all_exp = np.exp(scores - scores.max())
    positive_scores = scores[positive]
    positive_exp = np.exp(positive_scores - positive_scores.max())
    loss = scores.max() + np.log(all_exp.sum()) - positive_scores.max() - np.log(positive_exp.sum())
    delta = all_exp / all_exp.sum()
    delta[positive] -= positive_exp / positive_exp.sum()
    return float(loss), scores, delta @ matrix


def replay_updates(records, rows, scale, learned, steps=200):
    need(len(records) == steps, 'update count differs')
    parameters, first, second = ANCHOR.copy(), np.zeros(5), np.zeros(5)
    for step, record in enumerate(records, 1):
        need(record['step'] == step, 'update sequence differs')
        gradient = np.asarray(record['gradient'])
        need(gradient.shape == (5,) and np.isfinite(gradient).all(), 'invalid gradient')
        need(np.isfinite(record['loss_before_update']), 'invalid loss')
        if step in (1, steps):
            values = [row_values(row, parameters, scale) for row in rows]
            loss = np.mean([v[0] for v in values]) + .01 * np.sum((parameters - ANCHOR)**2)
            expected = np.mean([v[2] for v in values], axis=0) + .02 * (parameters - ANCHOR)
            if not learned:
                expected[0] = 0.
            close(record['loss_before_update'], loss)
            close(gradient, expected)
        if not learned:
            need(gradient[0] == 0, 'fixed alpha gradient differs')
        norm = np.linalg.norm(gradient)
        close(record['gradient_norm_before_clip'], norm)
        gradient = gradient / max(1., norm)
        first = .9 * first + .1 * gradient
        second = .999 * second + .001 * gradient**2
        parameters -= .05 * (first / (1 - .9**step)) / (np.sqrt(second / (1 - .999**step)) + 1e-8)
        parameters[0] = max(0., parameters[0]) if learned else 1.
        close(record['parameters_after_update'], parameters)
    return parameters


def check_resources(records, expected_operations):
    need(records and records[-1]['stage'] == 'after_completion', 'missing terminal resource sample')
    opened, completed, previous = None, [], -1.
    for record in records:
        elapsed = record['elapsed_s']
        need(np.isfinite(elapsed) and previous <= elapsed <= 600, 'whole resource limit')
        previous = elapsed
        need(0 < record['rss_bytes'] <= 2 * 1024**3, 'RSS limit')
        need(record['swap_growth_bytes'] <= 512 * 1024**2, 'swap limit')
        duration = record['operation_elapsed_s']
        need(duration is None or np.isfinite(duration) and 0 <= duration <= 60, 'operation limit')
        if record['stage'] == 'operation_start':
            need(opened is None and record['operation'] is not None, 'nested operation')
            opened = record['operation']
        elif record['stage'] == 'operation_end':
            need(opened is not None and opened == record['operation'], 'unmatched operation')
            completed.append(opened)
            opened = None
    need(opened is None and completed == expected_operations, 'resource operation coverage differs')
    return {'elapsed_s': previous, 'operations': len(completed),
            'peak_rss_bytes': max(r['rss_bytes'] for r in records),
            'peak_swap_growth_bytes': max(r['swap_growth_bytes'] for r in records)}


def verify(output):
    output = Path(output)
    need(not any((output / name).exists() for name in ('failure.json', 'resource-watchdog-failure.json')), 'failed run')
    report, completion = read(output / 'report.json'), read(output / 'completion.json')
    need(report['completed'] is True and completion['completed'] is True, 'incomplete run')
    need(completion['report_sha256'] == sha(output / 'report.json'), 'report closure differs')
    need(read(output / 'pins.json') == report['pins'], 'pin manifests differ')
    for name, digest in report['pins'].items():
        need(sha(ROOT / name) == digest, 'runtime pin differs: ' + name)
    for name, digest in report['artifacts'].items():
        need(sha(output / name) == digest, 'artifact differs: ' + name)
    need(sha(DATA) == DATA_SHA, 'TRAIN identity differs')
    rows = lines(DATA)
    need(len(rows) == 1483 and sum(len(r['candidates']) for r in rows) == 202603, 'TRAIN denominator differs')
    need(read(output / 'folds.json') == [{'id': r['id'], 'component_id': r['component_id'], 'fold': fold(r)} for r in rows], 'fold assignment differs')
    need(report['fold_rows'] == {str(f): sum(fold(r) == f for r in rows) for f in range(4)}, 'fold counts differ')
    models = read(output / 'fold-models.json')
    need([(m['fold'], m['arm']) for m in models] == [(f, a) for f in range(4) for a in ARMS], 'model coverage differs')
    updates, predicted = lines(output / 'updates.jsonl'), lines(output / 'predictions.jsonl')
    need(len(updates) == 3200 and len(predicted) == 4 * len(rows), 'record coverage differs')
    summaries = {}
    for model in models:
        f, arm = model['fold'], model['arm']
        training, testing = [r for r in rows if fold(r) != f], [r for r in rows if fold(r) == f]
        need(not {r['component_id'] for r in training} & {r['component_id'] for r in testing}, 'component leakage')
        need((model['training_rows'], model['held_rows'], model['steps']) == (len(training), len(testing), 200), 'model training counts differ')
        scale = scale_for(training, arm in ('fixed_scaled', 'learned_scaled'))
        close(model['scale'], scale)
        selected = [r for r in updates if (r['fold'], r['arm']) == (f, arm)]
        parameters = replay_updates(selected, training, scale, arm.startswith('learned'))
        close(model['parameters'], parameters)
        recorded = [r for r in predicted if (r['fold'], r['arm']) == (f, arm)]
        need([r['id'] for r in recorded] == [r['id'] for r in testing], 'held coverage differs')
        for row, record in zip(testing, recorded):
            loss, scores, _ = row_values(row, parameters, scale)
            choose = lambda values: min(range(len(values)), key=lambda i: (-values[i], row['candidates'][i]))
            winner, base = choose(scores), choose(row['base_scores'])
            need(record['component_id'] == row['component_id'], 'prediction component differs')
            need((record['selected_index'], record['base_selected_index'], record['correct'], record['base_correct']) ==
                 (winner, base, winner in row['positive_indices'], base in row['positive_indices']), 'prediction differs')
            close(record['cross_entropy'], loss)
            close(record['gold_score_gap'], scores[winner] - max(scores[row['positive_indices']]))
    for arm in ARMS:
        records = [r for r in predicted if r['arm'] == arm]
        need(len({r['id'] for r in records}) == len(rows), 'OOF uniqueness differs')
        result = {'rows': len(records), 'top1': sum(r['correct'] for r in records),
                  'base_top1': sum(r['base_correct'] for r in records),
                  'corrections': sum(r['correct'] and not r['base_correct'] for r in records),
                  'regressions': sum(not r['correct'] and r['base_correct'] for r in records),
                  'cross_entropy': float(np.mean([r['cross_entropy'] for r in records])),
                  'mean_gold_score_gap': float(np.mean([r['gold_score_gap'] for r in records]))}
        need(set(report['summaries'][arm]) == set(result), 'summary fields differ')
        for key, value in result.items():
            close(report['summaries'][arm][key], value)
        summaries[arm] = result
    before, after = summaries['fixed_unscaled'], summaries['learned_scaled']
    passed = after['top1'] > before['top1'] and after['cross_entropy'] < before['cross_entropy']
    decision = read(output / 'oof-decision.json')
    need(report['preselected_arm'] == decision['preselected_arm'] == 'learned_scaled', 'preselection differs')
    need(report['qualification_passed'] == report['full_fit_performed'] == decision['qualification_passed'] == passed, 'gate differs')
    need(decision['summaries'] == report['summaries'], 'decision summaries differ')
    replay_updates(lines(output / 'probe-updates.jsonl'), rows, scale_for(rows, True), True, 3)
    qualification = read(output / 'qualification.json')
    durations = qualification['probe_durations_s']
    need(len(durations) == 3 and all(np.isfinite(d) and d >= 0 for d in durations), 'probe timing differs')
    need(qualification['qualified'] is True and 60 + 1.3 * max(durations) * 3400 <= qualification['projected_seconds'] <= 600, 'qualification differs')
    if passed:
        model = read(output / 'model.json')
        need(model == report['model'] and model['arm'] == 'learned_scaled' and model['steps'] == 200, 'full model differs')
        scale = scale_for(rows, True)
        close(model['scale'], scale)
        close(model['parameters'], replay_updates(lines(output / 'full-fit-updates.jsonl'), rows, scale, True))
    else:
        need(report['model'] is None and not (output / 'model.json').exists(), 'unexpected full fit')
    need(report['completed_updates'] == 3200 + 200 * passed, 'completed update count differs')
    operations = ['authenticate_train', 'prepare_probe', 'probe:1', 'probe:2', 'probe:3', 'qualify_resources']
    for f in range(4):
        for arm in ARMS:
            operations += [f'prepare:{f}:{arm}'] + [f'fit:{f}:{arm}:{s}' for s in range(1, 201)] + [f'evaluate:{f}:{arm}']
    operations += ['close_fold_comparison']
    if passed:
        operations += ['prepare_full_fit'] + [f'full_fit:{s}' for s in range(1, 201)] + ['save_full_fit']
    operations += ['verify_pins', 'write_report', 'write_completion']
    resources = check_resources(lines(output / 'resources.jsonl'), operations)
    proof = {'verified': True, 'report_sha256': sha(output / 'report.json'), 'verifier_sha256': sha(__file__),
             'summaries': summaries, 'qualification_passed': passed, 'resources': resources,
             'scope': 'All fold scales, predictions, summaries, saved parameters and Adam recurrences; objective/gradient only first and last update of each fit and probe.',
             'limits': 'Component isolation verified. No independent raw-document join, upstream-model generalization, DEV/TEST or runtime conversion claim.'}
    with (output / 'independent-verification.json').open('x') as stream:
        json.dump(proof, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    result = verify(parser.parse_args().output)
    print(json.dumps({'verified': result['verified'], 'qualification_passed': result['qualification_passed'], 'resources': result['resources']}))
