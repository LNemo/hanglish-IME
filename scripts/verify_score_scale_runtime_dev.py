"""Verify actual score-scale DEV captures without running models or opening labels."""
from __future__ import annotations

import argparse
import hashlib
from itertools import zip_longest
import json
import math
import os
from pathlib import Path

from scripts.verify_production_phonetic_choice_dev import (
    need, sha, read, jsonl, render_public, verify_candidate, verify_score_records,
    verify_choice_row, verify_resources, probe_groups,
)

ROOT = Path(__file__).resolve().parents[1]
SCREEN = ROOT / '.omx/artifacts/phonetic-score-scale-dev-screen-v1'
SOURCE = ROOT / '.omx/artifacts/production-phonetic-choice-dev-v1'
TRAIN = ROOT / '.omx/artifacts/phonetic-score-scale-training-v1'
DEV = ROOT / 'data/challenge/sentence-v3/dev.jsonl'
PRODUCTION = 'experimental-ctc-union-1-488fc89732af'


def expected_version(parameters, scale, checkpoint, artifacts, identity):
    parent_payload = dict(parent=PRODUCTION, artifacts=artifacts,
                          weights=[w / s for w, s in zip(parameters[1:], scale)], source='override')
    parent = 'experimental-production-phonetic-choice-1-' + hashlib.sha256(
        json.dumps(parent_payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:12]
    payload = dict(parent=parent, parameters=parameters, scale=scale,
                   checkpoint=checkpoint, identity_control=identity)
    return 'experimental-score-scale-1-' + hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def audit_response(source, response, paths, model, choice_artifacts, runtime_artifacts, identity=False):
    need(response['input'] == source and response['context'] == '' and response['external_context_used'] is False, 'source-only response differs')
    parameters = [1., 0., 0., 0., 0.] if identity else model['parameters']
    config, choice = response['phonetic_score_scale'], response['phonetic_choice']
    need(config['parameters'] == parameters and config['scale'] == model['scale']
         and config['checkpoint_sha256'] == sha(TRAIN / 'model.json') and config['identity_control'] == identity
         and config['version'] == response['version'] == choice['version']
         and choice['parent_version'] == PRODUCTION
         and response['version'].startswith('experimental-score-scale-1-')
         and 'phonetic_word_union' not in response, 'runtime model contract differs')
    need(choice['artifacts'] == choice_artifacts and response['model_artifacts'] == runtime_artifacts,
         'complete runtime artifact mapping differs')
    need(choice['effective_weights'] == [w / s for w, s in zip(parameters[1:], model['scale'])]
         and choice['calibration_source'] == 'override', 'effective coefficients differ')
    need(response['version'] == expected_version(parameters, model['scale'], config['checkpoint_sha256'],
                                                 choice_artifacts, identity), 'derived version differs')
    for candidate in response['candidates']:
        verify_candidate(source, candidate)
    need(response['result'] == response['candidates'][0]['text'], 'first choice differs')
    features = choice['rows']
    if not response['sentence_ranking_applied']:
        need(not paths and not features and not choice['score_records'], 'unranked trace differs')
        return
    need(paths and len(paths) == len(features) and len({p[0] for p in paths}) == len(paths), 'pool domain differs')
    score_records = verify_score_records(choice['score_records'])
    learned_paths = []
    for path, row in zip(paths, features):
        vector = verify_choice_row(source, path, row, score_records)
        residual = sum(w * (f / s) for w, f, s in zip(parameters[1:], vector, model['scale']))
        score = parameters[0] * path[1] + residual
        need(row['residual'] == residual and row['adjusted_score'] == score and math.isfinite(score), 'recomputed score differs')
        learned_paths.append((path[0], score, path[2]))
    learned_paths.sort(key=lambda p: (-p[1], p[0]))
    need(render_public(source, learned_paths) == response['candidates'], 'public ranking replay differs')
    need(choice['public'] == [c['text'] for c in response['candidates']], 'trace public differs')


def verify(output):
    output = Path(output).resolve()
    need(not (output / 'independent-verification.json').exists(), 'proof already exists')
    need(not any((output / n).exists() for n in ('failure.json', 'resource-watchdog-failure.json')), 'runtime failed')
    launch = read(output.with_suffix('.launch.json'))
    try:
        os.kill(launch['pid'], 0)
    except ProcessLookupError:
        pass
    else:
        raise ValueError('runtime PID is still present')
    runner = ROOT / 'scripts/evaluate_score_scale_runtime_dev.py'
    need(launch['command'] == [str(ROOT / '.venv/bin/python'), '-m', 'scripts.evaluate_score_scale_runtime_dev', '--output', str(output)]
         and launch['runner_sha256'] == sha(runner) and launch['output'] == str(output), 'launch identity differs')
    report, completion, closure = (read(output / n) for n in ('report.json', 'completion.json', 'closure.json'))
    need(report['completed'] is True and completion['completed'] is True
         and completion['report_sha256'] == sha(output / 'report.json'), 'terminal report differs')
    pins = read(output / 'pins.json')
    need(pins == report['pins'] and pins[str(runner)] == sha(runner), 'pin manifest differs')
    for name, digest in pins.items():
        if Path(name).resolve() != DEV.resolve():
            need(sha(name) == digest, 'runtime pin differs: ' + name)
    for name, digest in report['artifacts'].items():
        need(sha(output / name) == digest, 'closed runtime artifact differs')
    need(closure == dict(rows=316, capture_sha256=sha(output / 'capture.jsonl'), labels_opened=False), 'capture closure differs')
    need(sha(SCREEN / 'independent-verification.json') == 'e1b23b2fc23fbacba83c56358207eaa810c4b80addab59481c3b04a47a8ef8f4', 'screen proof identity differs')
    screen, screen_proof = read(SCREEN / 'report.json'), read(SCREEN / 'independent-verification.json')
    need(screen_proof['verified'] is True and screen_proof['quality_gate'] is True
         and screen_proof['report_sha256'] == sha(SCREEN / 'report.json')
         and screen['predictions_sha256'] == sha(SCREEN / 'predictions.jsonl'), 'screen closure differs')
    need(sha(TRAIN / 'model.json') == 'e07462d50da0b67304a82a9a261068ccc84b8055a99b5d70af56db530d245838', 'coefficients differ')
    model = read(TRAIN / 'model.json')
    with (SOURCE / 'capture.jsonl').open() as stream:
        historical = json.loads(next(stream))
    parent_artifacts = historical['parent']['model_artifacts']
    choice_artifacts = dict(historical['proposed']['phonetic_choice']['artifacts'])
    choice_artifacts.pop('.omx/artifacts/phonetic-choice-calibration-training-v1/weights.json')
    for name in ('hanglish/core/score_scale_converter.py', '.omx/artifacts/phonetic-score-scale-training-v1/model.json'):
        choice_artifacts[name] = pins[str(ROOT / name)]
    runtime_artifacts = parent_artifacts | choice_artifacts
    fixed = list(jsonl(SCREEN / 'predictions.jsonl'))
    need(len(fixed) == 316 and len({r['id'] for r in fixed}) == 316, 'screen domain differs')
    groups = probe_groups([(r['id'], r['request']['input'], None) for r in fixed])
    need(read(output / 'probe-schedule.json') == dict(groups=groups), 'probe schedule differs')
    probes = list(jsonl(output / 'probe.jsonl'))
    need(len(probes) == 16, 'probe count differs')
    by_id = {r['id']: r for r in fixed}
    for group, probe in zip(groups, probes):
        identifier = group['probe_id']
        need(probe['id'] == identifier and probe['request'] == by_id[identifier]['request'], 'probe source differs')
        source = probe['request']['input']
        before, after = probe['parent'], probe['identity']
        need(before['version'] == PRODUCTION and before['input'] == source and before['context'] == ''
             and before['external_context_used'] is False and before['model_artifacts'] == parent_artifacts, 'probe baseline contract differs')
        audit_response(source, after, probe['identity_paths'], model, choice_artifacts, runtime_artifacts, identity=True)
        need(probe['parent_paths'] == probe['identity_paths'] and before['candidates'] == after['candidates']
             and before['result'] == after['result']
             and before['candidates'] == by_id[identifier]['parent']['candidates'], 'identity control parity differs')
    qualification = read(output / 'qualification.json')
    timings = qualification['timings']
    need(len(timings) == 16, 'timing count differs')
    for group, timing in zip(groups, timings):
        need(timing['id'] == group['probe_id'] and timing['count'] == group['count']
             and math.isfinite(timing['identity_seconds']) and timing['identity_seconds'] >= 0, 'timing correspondence differs')
    projected = 1.3 * (qualification['elapsed_seconds'] + sum(t['count']*t['identity_seconds'] for t in timings)) + 60.
    need(qualification['qualified'] is True and qualification['projected_seconds'] == projected <= 1800, 'projection differs')
    latencies, count, path_count = [], 0, 0
    for original, expected, actual in zip_longest(jsonl(SOURCE / 'capture.jsonl'), fixed, jsonl(output / 'capture.jsonl')):
        need(original is not None and expected is not None and actual is not None, 'capture coverage differs')
        need(original['id'] == expected['id'] == actual['id']
             and original['request'] == expected['request'] == actual['request'], 'capture source/order differs')
        need(actual['base_paths'] == original['base_paths'], 'full production paths changed')
        need(original['parent']['model_artifacts'] == parent_artifacts, 'historical parent artifacts changed')
        response = actual['response']
        need(response['version'] == report['version'], 'response version changed')
        audit_response(actual['request']['input'], response, actual['base_paths'], model, choice_artifacts, runtime_artifacts)
        need({k: response[k] for k in ('result', 'candidates', 'sentence_ranking_applied')} == expected['proposed'], 'actual/cached result differs')
        need(response['phonetic_choice']['score_records'] == original['proposed']['phonetic_choice']['score_records'], 'frozen likelihood scores differ')
        for row, frozen in zip(response['phonetic_choice']['rows'], expected['score_rows']):
            need(row['features'] == frozen['features'] and row['base_score'] == frozen['base_score']
                 and row['adjusted_score'] == frozen['score'], 'actual/cached score differs')
        for name, digest in response['model_artifacts'].items():
            path = Path(name) if Path(name).is_absolute() else ROOT / name
            need(pins.get(str(path)) == digest, 'response model artifact unbound')
        need(math.isfinite(response['engine_ms']) and response['engine_ms'] >= 0, 'latency invalid')
        latencies.append(response['engine_ms'])
        path_count += len(actual['base_paths'])
        count += 1
    need(count == report['rows'] == report['actual_runtime_parity_rows'] == 316 and report['identity_probe_rows'] == 16, 'runtime count differs')
    need(report['metrics'] == screen['metrics'] and report['quality_gate'] == screen['quality_gate'] is True, 'linked quality differs')
    operations = ['authenticate', 'load_models']
    for group in groups:
        operations += ['probe_parent:' + group['probe_id'], 'probe_identity:' + group['probe_id']]
    operations += ['qualify'] + ['predict:' + r['id'] for r in fixed] + ['close_capture', 'write_report', 'write_completion']
    resources = verify_resources(list(jsonl(output / 'resources.jsonl')), operations)
    ordered = sorted(latencies)
    need(report['latency_ms'] == dict(p50=ordered[len(ordered)//2], p95=ordered[int(.95*(len(ordered)-1))], maximum=max(ordered)), 'latency summary differs')
    need(report['labels_opened'] is False and report['test_accessed'] is False and report['production_changed'] is False, 'runtime scope differs')
    proof = dict(verified=True, report_sha256=sha(output / 'report.json'), verifier_sha256=sha(__file__),
                 helper_sha256=sha(ROOT / 'scripts/verify_production_phonetic_choice_dev.py'),
                 launch_sha256=sha(output.with_suffix('.launch.json')), capture_sha256=sha(output / 'capture.jsonl'),
                 counts=dict(rows=count, paths=path_count, probes=16), resources=resources,
                 metrics=report['metrics'], quality_gate=True, latency_ms=report['latency_ms'],
                 labels_opened=False, scope='Actual316 source-only responses and full paths,16 identity probes, all feature/score/public parity, immutable model and resource evidence.',
                 limitations='Observed DEV. No TEST1000 or web promotion claim. Likelihood model outputs are matched to previous verified captures, not inferred a second time by the verifier.')
    with (output / 'independent-verification.json').open('x') as stream:
        json.dump(proof, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    result = verify(parser.parse_args().output)
    print(json.dumps({k: result[k] for k in ('verified', 'counts', 'quality_gate', 'latency_ms')}))
