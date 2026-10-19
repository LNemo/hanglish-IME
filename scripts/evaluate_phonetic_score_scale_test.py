"""Fixed TEST1000 screen of the score-scale model; no training or inference."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from scripts.evaluate_phonetic_score_scale_dev import (
    ROOT, TRAIN, need, sha, read, rows, write, project, validate_model,
)
from scripts.verify_production_phonetic_choice_test import aggregate
from scripts.source_span_resources import AtomicResourceWatchdog, system_swap_bytes

SOURCE = ROOT / '.omx/artifacts/production-phonetic-choice-test-v1'
DEV_RUN = ROOT / '.omx/artifacts/score-scale-runtime-dev-v1'
TEST = ROOT / 'data/challenge/sentence-v3/test.jsonl'
PLAN = ROOT / '.omx/plans/phonetic-score-scale-test-v1.md'
LIMITS = dict(whole_seconds=120., update_seconds=60., rss_bytes=1024**3, swap_growth_bytes=512*1024**2)
FROZEN = {
    DEV_RUN / 'report.json': '39b76bd5f59fbbe90c8ebbd65dfd15f9973b327746230eb997a215fd65aae959',
    DEV_RUN / 'independent-verification.json': '60b8c4c16e00d36274e2f71760888ea5c12d519669ab4ad38cad5745dd0fd9c0',
    SOURCE / 'report.json': 'e8592ca1435c8ea1da37e7ec63a6ee3ad71ecc3bf2951541b5b791158de8360b',
    SOURCE / 'capture.jsonl': 'fa93215001ea47a9b36796932ec965ee4cdfaf52593a4fab956b51a74961faf0',
    SOURCE / 'independent-verification.json': '0ce8bc781d55ce2ac159447ad3e17082c331f17527ad3aa819cba7ab0865212c',
    TRAIN / 'model.json': 'e07462d50da0b67304a82a9a261068ccc84b8055a99b5d70af56db530d245838',
}
LABEL_SHA = '1dcf804460b40c445abf688dc47b452937d8911ac156b8520141fad25b973d06'


def project_test(record, model):
    need(record['actual_base_pool'] == [p[0] for p in record['base_paths']], 'actual TEST pool differs')
    projected = project(record | {'parent_pool': record['actual_base_pool']}, model)
    projected['actual_base_pool'] = projected.pop('parent_pool')
    return projected


def preservation(metrics):
    english = metrics['english_ascii_signature']
    return not any((metrics['regressions'], metrics['top5_losses'],
                    english.get('signature_regressions', 0), english.get('literal_signature_regressions', 0)))


def decision(metrics):
    return metrics['parent']['corrections'] > 0 and all(preservation(value) for value in metrics.values())


def authenticate():
    need(all(sha(path) == digest for path, digest in FROZEN.items()), 'frozen authority differs')
    dev, proof = read(DEV_RUN / 'report.json'), read(DEV_RUN / 'independent-verification.json')
    need(dev['completed'] is True and dev['quality_gate'] is True and proof['verified'] is True
         and proof['quality_gate'] is True and proof['report_sha256'] == sha(DEV_RUN / 'report.json')
         and dev['actual_runtime_parity_rows'] == 316
         and dev['version'] == 'experimental-score-scale-1-f6976e1b269d', 'actual DEV gate differs')
    old, old_proof = read(SOURCE / 'report.json'), read(SOURCE / 'independent-verification.json')
    need(old['completed'] is True and old_proof['verified'] is True
         and old_proof['report_sha256'] == sha(SOURCE / 'report.json')
         and old_proof['capture_sha256'] == sha(SOURCE / 'capture.jsonl'), 'old TEST evidence differs')
    pins = {str(path): digest for path, digest in FROZEN.items()}
    for report in (dev, old):
        for name, digest in report['pins'].items():
            path = Path(name) if Path(name).is_absolute() else ROOT / name
            need(str(path) not in pins or pins[str(path)] == digest, 'inherited pin collision')
            pins[str(path)] = digest
    for path in (Path(__file__), PLAN, ROOT / 'scripts/evaluate_phonetic_score_scale_dev.py',
                 ROOT / 'scripts/verify_production_phonetic_choice_test.py',
                 ROOT / 'scripts/verify_production_phonetic_choice_dev.py',
                 ROOT / 'scripts/source_span_resources.py'):
        pins[str(path)] = sha(path)
    need(pins[str(TEST)] == LABEL_SHA, 'label authority differs')
    for name, digest in pins.items():
        if Path(name).resolve() != TEST.resolve():
            need(sha(name) == digest, 'runtime pin differs: ' + name)
    model = read(TRAIN / 'model.json')
    validate_model(model)
    return model, pins


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        with AtomicResourceWatchdog(started, system_swap_bytes(), output / 'resources.jsonl', limits=LIMITS, interval_s=.5) as guard:
            with guard.operation('authenticate_dev_then_test_sources', update=True):
                model, pins = authenticate()
                write(output / 'pins.json', pins)
            predictions = []
            with guard.operation('project_all', update=True):
                with (output / 'predictions.jsonl').open('x') as stream:
                    for record in rows(SOURCE / 'capture.jsonl'):
                        value = project_test(record, model)
                        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
                        predictions.append({k: v for k, v in value.items() if k != 'score_rows'})
                    stream.flush(); os.fsync(stream.fileno())
                need(len(predictions) == len({r['id'] for r in predictions}) == 1000, 'TEST coverage differs')
                closed = sha(output / 'predictions.jsonl')
                write(output / 'closure.json', dict(rows=1000, predictions_sha256=closed, labels_opened=False))
            with guard.operation('late_label_join', update=True):
                need(sha(TEST) == LABEL_SHA, 'deferred TEST labels differ')
                labels = list(rows(TEST))
                metrics = {branch: aggregate([r | {'parent': r[branch]} for r in predictions], labels)
                           for branch in ('parent', 'old_calibration')}
                need(metrics['parent']['parent']['counts'] == dict(rows=1000, top1=953, top5=972, errors=0), 'operating TEST counts differ')
                need(metrics['old_calibration']['parent']['counts'] == dict(rows=1000, top1=954, top5=974, errors=0), 'old calibrated TEST counts differ')
                passed = decision(metrics)
                need(sha(output / 'predictions.jsonl') == closed, 'closed predictions changed')
            with guard.operation('write_report', update=True):
                report = dict(schema='phonetic-score-scale-test-screen-v1', completed=True, rows=1000,
                    metrics=metrics, quality_gate=passed, actual_runtime_authorized=passed,
                    predictions_sha256=closed, closure_sha256=sha(output / 'closure.json'), pins=pins, limits=LIMITS,
                    model_sha256=sha(TRAIN / 'model.json'), model_inference_performed=False,
                    training_performed=False, production_changed=False, labels_opened_after_predictions=True,
                    all_first_choices_correct=metrics['parent']['proposed']['counts']['top1'] == 1000,
                    limitations=['Cached actual candidate replay, not actual new runtime inference or latency.',
                                 'Repeatedly observed TEST regression set, not blind generalization.',
                                 'Signature preservation does not prove every COPY span is correct.'])
                write(output / 'report.json', report)
            guard.sample_and_append('after_report')
            with guard.operation('write_completion', update=True):
                need(all(sha(name) == digest for name, digest in pins.items()), 'terminal pin changed')
                write(output / 'completion.json', dict(completed=True, report_sha256=sha(output / 'report.json')))
            guard.sample_and_append('after_completion')
        return report
    except BaseException as error:
        write(output / 'failure.json', dict(completed=False, error=f'{type(error).__name__}: {error}'))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    result = run(parser.parse_args().output)
    print(json.dumps(dict(completed=True, quality_gate=result['quality_gate'],
        counts=result['metrics']['parent']['proposed']['counts'],
        changes={b: {k: result['metrics'][b][k] for k in ('corrections', 'regressions', 'top5_losses')} for b in result['metrics']})))
