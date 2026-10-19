"""Screen frozen score-scale coefficients on closed actual DEV candidate paths."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

from scripts.source_span_resources import AtomicResourceWatchdog, system_swap_bytes

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / '.omx/artifacts/production-phonetic-choice-dev-v1'
TRAIN = ROOT / '.omx/artifacts/phonetic-score-scale-training-v1'
DEV = ROOT / 'data/challenge/sentence-v3/dev.jsonl'
PLAN = ROOT / '.omx/plans/phonetic-score-scale-dev-v1.md'
LIMITS = dict(whole_seconds=120., update_seconds=60., rss_bytes=1024**3, swap_growth_bytes=512*1024**2)
FROZEN = {
    SOURCE / 'report.json': '479ad53464e3c2ef978002fa5616ddaf9afadac936941a859ae5e01f3422b5cc',
    SOURCE / 'capture.jsonl': 'f562b5e813484fa49e2d7bb7d231bde63dd40201120eb0bae2d1043f7f4dc432',
    SOURCE / 'independent-verification.json': 'a7c64c74f7fd756ef53834be326cb96e41927d472a7131020181bd811462977a',
    TRAIN / 'model.json': 'e07462d50da0b67304a82a9a261068ccc84b8055a99b5d70af56db530d245838',
    TRAIN / 'report.json': 'adb25f7f952ce59d0e01187d8b842d2d039bb27d88cc1adc26a729971669df18',
    TRAIN / 'independent-verification.json': '4b9b96bb2a8574df16aa1f34244c800e511ce7c590f975389ec5368b2fb574e1',
}


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def render(source, paths):
    ordered = sorted(paths, key=lambda p: (-p[1], p[0]))
    need(ordered, 'empty pool')
    result = [dict(text=text, score=round(score, 4), segments=segments) for text, score, segments in ordered[:5]]
    if source not in [c['text'] for c in result]:
        result = result[:4] + [dict(text=source, score=None, segments=[dict(input_start=0, input_end=len(source), source=source, text=source, kind='raw')])]
    return result


def validate_model(model):
    need(model['schema'] == 'phonetic-score-scale-model-v1' and model['arm'] == 'learned_scaled', 'model contract differs')
    parameters, scale = model['parameters'], model['scale']
    need(len(parameters) == 5 and len(scale) == 4 and parameters[0] >= 0
         and all(type(x) in (int, float) and math.isfinite(x) for x in parameters + scale)
         and all(x > 0 for x in scale), 'invalid coefficients')
    return parameters, scale


def project(record, model):
    parameters, scale = validate_model(model)
    source = record['request']['input']
    need(record['request'] == dict(input=source, context=''), 'source-only contract differs')
    previous = record['proposed']
    paths, features = record['base_paths'], previous['phonetic_choice']['rows']
    scores = []
    if previous['sentence_ranking_applied']:
        need(paths and len(paths) == len(features) and len({p[0] for p in paths}) == len(paths), 'pool coverage differs')
        need([p[0] for p in paths] == record['parent_pool']
             and render(source, paths) == record['parent']['candidates'], 'production replay differs')
        for (text, base, segments), feature in zip(paths, features):
            need(feature['text'] == text and feature['base_score'] == base and feature['segments'] == segments, 'feature alignment differs')
            values = feature['features']
            need(len(values) == 4 and all(math.isfinite(x) for x in values), 'invalid features')
            score = parameters[0] * base + sum(w * (v / s) for w, v, s in zip(parameters[1:], values, scale))
            need(math.isfinite(score), 'invalid new score')
            scores.append(dict(text=text, base_score=base, features=values, score=score))
        public = render(source, [(p[0], score['score'], p[2]) for p, score in zip(paths, scores)])
    else:
        need(not paths and not features and not record['parent_pool']
             and previous['candidates'] == record['parent']['candidates'], 'unranked contract differs')
        public = previous['candidates']
    compact = lambda response: {k: response[k] for k in ('result', 'candidates', 'sentence_ranking_applied')}
    return dict(id=record['id'], request=record['request'], parent=compact(record['parent']),
                old_calibration=compact(previous), parent_pool=record['parent_pool'], score_rows=scores,
                proposed=dict(result=public[0]['text'], candidates=public,
                              sentence_ranking_applied=previous['sentence_ranking_applied']),
                kind='cached_score_projection')


def compare(records, labels, baseline):
    counts = {branch: dict(rows=len(records), top1=0, top5=0) for branch in (baseline, 'proposed')}
    result = dict(corrections=0, regressions=0, top5_losses=0, ascii_signature_regressions=0,
                  literal_signature_regressions=0, ranked_rows=0, pool_correct=0)
    literal = lambda text: ''.join(c for c in text if not ('a' <= c <= 'z' or re.fullmatch('[가-힣ᄀ-ᇿ㄰-㆏]', c)))
    ascii_words = lambda text: tuple(re.findall('[A-Za-z][A-Za-z0-9_]*', text))
    need(len(records) == len(labels) and len({r['id'] for r in records}) == len(records), 'evaluation coverage differs')
    for row, label in zip(records, labels):
        need(row['id'] == label['id'] and row['request'] == dict(input=label['input'], context='')
             and label.get('context', '') == '' and label['split'] == 'dev', 'late label identity differs')
        gold = set(label['expected'])
        outcomes = {}
        for branch in (baseline, 'proposed'):
            texts = [c['text'] for c in row[branch]['candidates']]
            need(row[branch]['result'] == texts[0], 'first choice differs')
            outcomes[branch] = (texts[0] in gold, bool(gold.intersection(texts)))
            counts[branch]['top1'] += outcomes[branch][0]
            counts[branch]['top5'] += outcomes[branch][1]
        old, new = outcomes[baseline], outcomes['proposed']
        result['corrections'] += not old[0] and new[0]
        result['regressions'] += old[0] and not new[0]
        result['top5_losses'] += old[1] and not new[1]
        signatures = {ascii_words(text) for text in gold}
        if any(signatures):
            result['ascii_signature_regressions'] += ascii_words(row[baseline]['result']) in signatures and ascii_words(row['proposed']['result']) not in signatures
        source = literal(row['request']['input'])
        result['literal_signature_regressions'] += literal(row[baseline]['result']) == source and literal(row['proposed']['result']) != source
        if row['proposed']['sentence_ranking_applied']:
            result['ranked_rows'] += 1
            result['pool_correct'] += bool(gold.intersection(row['parent_pool']))
    result['counts'] = counts
    result['preservation_gate'] = not any(result[k] for k in ('regressions', 'top5_losses', 'ascii_signature_regressions', 'literal_signature_regressions'))
    return result


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        with AtomicResourceWatchdog(started, system_swap_bytes(), output / 'resources.jsonl', limits=LIMITS, interval_s=.5) as guard:
            with guard.operation('authenticate', update=True):
                need(all(sha(p) == h for p, h in FROZEN.items()), 'frozen input differs')
                source_report, train_report = read(SOURCE / 'report.json'), read(TRAIN / 'report.json')
                for folder, report in ((SOURCE, source_report), (TRAIN, train_report)):
                    proof = read(folder / 'independent-verification.json')
                    need(proof['verified'] is True and proof['report_sha256'] == sha(folder / 'report.json') and report['completed'] is True, 'proof differs')
                need(train_report['qualification_passed'] is True and train_report['full_fit_performed'] is True, 'TRAIN gate not passed')
                pins = {str(p): h for p, h in FROZEN.items()}
                for report in (source_report, train_report):
                    for name, digest in report['pins'].items():
                        p = Path(name) if Path(name).is_absolute() else ROOT / name
                        if p.resolve() != DEV.resolve():
                            need(sha(p) == digest, 'inherited pin differs: ' + name)
                        need(str(p) not in pins or pins[str(p)] == digest, 'pin collision')
                        pins[str(p)] = digest
                for p in (Path(__file__), PLAN, ROOT / 'scripts/source_span_resources.py'):
                    pins[str(p)] = sha(p)
                write(output / 'pins.json', pins)
                model = read(TRAIN / 'model.json')
                need(model == train_report['model'], 'saved model differs')
                validate_model(model)
            projected = []
            with guard.operation('project_all', update=True):
                with (output / 'predictions.jsonl').open('x') as stream:
                    for row in rows(SOURCE / 'capture.jsonl'):
                        value = project(row, model)
                        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
                        projected.append({k: v for k, v in value.items() if k != 'score_rows'})
                    stream.flush()
                    os.fsync(stream.fileno())
                need(len(projected) == 316, 'DEV denominator differs')
                prediction_sha = sha(output / 'predictions.jsonl')
                write(output / 'closure.json', dict(rows=316, predictions_sha256=prediction_sha, labels_opened=False))
            with guard.operation('late_label_join', update=True):
                need(sha(DEV) == pins[str(DEV)], 'DEV pin differs')
                labels = list(rows(DEV))
                metrics = {branch: compare(projected, labels, branch) for branch in ('parent', 'old_calibration')}
                need(metrics['parent']['counts']['parent'] == dict(rows=316, top1=262, top5=282), 'production metrics differ')
                need(metrics['old_calibration']['counts']['old_calibration'] == dict(rows=316, top1=263, top5=283), 'old calibration metrics differ')
                passed = metrics['parent']['corrections'] > 0 and all(m['preservation_gate'] for m in metrics.values())
                need(sha(output / 'predictions.jsonl') == prediction_sha, 'closed predictions changed')
            with guard.operation('write_report', update=True):
                report = dict(schema='phonetic-score-scale-dev-screen-v1', completed=True, rows=316,
                    metrics=metrics, quality_gate=passed, actual_runtime_authorized=passed,
                    model_sha256=sha(TRAIN / 'model.json'), pins=pins, limits=LIMITS,
                    predictions_sha256=prediction_sha, closure_sha256=sha(output / 'closure.json'),
                    model_inference_performed=False, training_performed=False, test_accessed=False,
                    production_changed=False, labels_opened_after_predictions=True,
                    limitations=['Cached actual candidate replay, not a fresh runtime or latency measurement.',
                                 'DEV is previously observed; no fresh generalization claim.',
                                 'ASCII and literal signatures do not prove complete span-level copying.'])
                write(output / 'report.json', report)
            guard.sample_and_append('after_report')
            with guard.operation('write_completion', update=True):
                need(all(sha(Path(p)) == digest for p, digest in pins.items()), 'final pin changed')
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
    print(json.dumps({k: result[k] for k in ('completed', 'metrics', 'quality_gate')}, ensure_ascii=False))
