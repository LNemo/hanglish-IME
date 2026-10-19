"""Compare a qualified expanded word selector through the actual sentence pipeline."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hanglish.core.neural_converter import NeuralConverter
from hanglish.models.semantic_ranker import SemanticRanker
from scripts.evaluate_ensemble_regression import file_sha256, rss_bytes, verify_pins, validate_response, protected_segments_preserved
from scripts.probe_natural_homophone_contexts import project

TRAIN = ROOT / '.omx/artifacts/runtime-collision-ranker-training-v1'
PARENT_CAPTURE = ROOT / '.omx/artifacts/productive-endings-runtime-v1'
SOURCE = ROOT / 'data/natural_context/runtime-collision-source-v5'
PLAN = ROOT / '.omx/plans/expanded-word-split-v1.json'
REVIEW = ROOT / '.omx/artifacts/runtime-collision-source-review-v2.json'
OUTPUT = ROOT / '.omx/artifacts/runtime-collision-ranker-runtime-v1'


def held_sample():
    assignments = json.loads(PLAN.read_text())['assignments']
    quarantined = set(json.loads(REVIEW.read_text())['quarantined_ids'])
    rows = [r for r in map(json.loads, (SOURCE / 'selected_train.jsonl').read_text().splitlines())
            if assignments[r['id']]['role'] == 'hold' and r['id'] not in quarantined]
    # Frozen ID-only selection, never conditional on candidate presence or accuracy.
    return sorted(rows, key=lambda r: hashlib.sha256(('expanded-word-runtime-v1\0' + r['id']).encode()).hexdigest())[:64]


def run(report_sha):
    if file_sha256(TRAIN / 'report.json') != report_sha:
        raise ValueError('training report changed')
    training = json.loads((TRAIN / 'report.json').read_text())
    proof = json.loads((TRAIN / 'independent-verification.json').read_text())
    if not training['advance_to_broader_evaluation'] or not proof['verified'] or proof['report_sha256'] != report_sha:
        raise ValueError('training gate/proof missing')
    verify_pins(training['pins'])
    previous = json.loads((PARENT_CAPTURE / 'report.json').read_text())
    if file_sha256(PARENT_CAPTURE / 'report.json') != '3d3fa20abd373069b59a9170ec58a5bad0011e9bb5e544f7c9a0d07f14f5d6e3':
        raise ValueError('production baseline changed')
    verify_pins(previous['pins'])
    if file_sha256(PARENT_CAPTURE / 'capture.jsonl') != previous['capture_sha256']:
        raise ValueError('baseline capture changed')
    OUTPUT.mkdir(exist_ok=False)
    started = time.monotonic()

    def guard():
        if time.monotonic() - started > 1200 or rss_bytes() > 3 * 1024 ** 3:
            raise RuntimeError('runtime comparison exceeds1200seconds/3GiB')

    pins = {**training['pins'], **previous['pins']}
    for path in [Path(__file__), PLAN, REVIEW, SOURCE / 'selected_train.jsonl', TRAIN / 'report.json',
                 TRAIN / 'independent-verification.json', PARENT_CAPTURE / 'report.json',
                 PARENT_CAPTURE / 'capture.jsonl', PARENT_CAPTURE / 'boundary-probes.json',
                 *list((TRAIN / 'checkpoint').iterdir())]:
        if path.is_file():
            pins[str(path.relative_to(ROOT))] = file_sha256(path)
    verify_pins(pins)
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    engine = NeuralConverter()
    engine.load_models()
    if engine.version != previous['version']:
        raise ValueError('current production identity differs')
    baseline_artifacts = dict(engine.model_artifacts)
    parent_ranker = engine.ranker
    trained = SemanticRanker.load(TRAIN / 'checkpoint')
    experimental_version = 'experimental-expanded-word-' + report_sha[:12]
    experimental_artifacts = {k: v for k, v in baseline_artifacts.items() if not k.startswith('checkpoints/semantic_ranker/')}
    experimental_artifacts.update({str(p.relative_to(ROOT)): file_sha256(p) for p in (TRAIN / 'checkpoint').iterdir() if p.is_file()})
    targets = {(suite, r['id']): r for suite in ('dev', 'test')
               for r in map(json.loads, (ROOT / f'data/challenge/v2.2/{suite}.jsonl').read_text().splitlines())}
    baseline_rows = list(map(json.loads, (PARENT_CAPTURE / 'capture.jsonl').read_text().splitlines()))
    cases = []
    for row in baseline_rows:
        target = targets[(row['suite'], row['id'])]
        assert row['input'] == target['input'] and row['context'] == target['context']
        assert row['response']['model_artifacts'] == baseline_artifacts
        cases.append(dict(id=row['id'], suite=row['suite'], input=row['input'], context=row['context'], expected=target['expected'], parent=row['response']))
    for target in held_sample():
        projected = project(target)
        cases.append(dict(id=target['id'], suite='source_hold_sample', input=projected['input'], context=projected['context'], expected=[target['text']]))
    for index, probe in enumerate(json.loads((PARENT_CAPTURE / 'boundary-probes.json').read_text())):
        cases.append(dict(id='productive-' + str(index), suite='productive_probes', input=probe['input'], context='', expected=[probe['word']], parent=probe['response']))
    (OUTPUT / 'selection.json').write_text(json.dumps(cases, ensure_ascii=False) + '\n')
    counts, changes = defaultdict_counter(), {}
    with (OUTPUT / 'capture.jsonl').open('x') as stream:
        for index, case in enumerate(cases):
            if 'parent' not in case:
                engine.ranker = parent_ranker
                engine.version, engine.model_artifacts = previous['version'], baseline_artifacts
                case['parent'] = engine.convert(case['input'], context=case['context'])
            engine.ranker = trained
            engine.version, engine.model_artifacts = experimental_version, experimental_artifacts
            after = engine.convert(case['input'], context=case['context'])
            status = {}
            for name, response in [('before', case['parent']), ('after', after)]:
                validate_response(response, case['input'], case['context'])
                assert protected_segments_preserved(response)
                assert any(c['text'] == case['input'] for c in response['candidates'])
                status[name + '_top1'] = response['result'] in case['expected']
                status[name + '_top5'] = any(c['text'] in case['expected'] for c in response['candidates'])
            suite = case['suite']
            counts[suite].update(dict(rows=1, **status))
            changes.setdefault(suite, dict(fixes=[], losses=[], top5_losses=[]))
            if not status['before_top1'] and status['after_top1']:
                changes[suite]['fixes'].append(case['id'])
            if status['before_top1'] and not status['after_top1']:
                changes[suite]['losses'].append(case['id'])
            if status['before_top5'] and not status['after_top5']:
                changes[suite]['top5_losses'].append(case['id'])
            stream.write(json.dumps({**case, 'experiment': after}, ensure_ascii=False) + '\n')
            stream.flush()
            guard()
            if (index + 1) % 128 == 0:
                print(json.dumps(dict(captured=index + 1, total=len(cases), elapsed_s=time.monotonic() - started)), flush=True)
    verify_pins(pins)
    passed = all(not c['losses'] and not c['top5_losses'] for c in changes.values()) and bool(changes['source_hold_sample']['fixes'])
    report = dict(completed=True, production_changed=False, promotion_gate_passed=passed,
                  scope='exposed regression1140,15 productive diagnostics,64 ID-selected held source sentences',
                  parent_version=previous['version'], version=experimental_version, counts=dict(counts), changes=changes,
                  pins=pins, capture_sha256=file_sha256(OUTPUT / 'capture.jsonl'), selection_sha256=file_sha256(OUTPUT / 'selection.json'),
                  elapsed_s=time.monotonic() - started, peak_rss_bytes=rss_bytes())
    (OUTPUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'pins'}, ensure_ascii=False), flush=True)


def defaultdict_counter():
    from collections import defaultdict
    return defaultdict(Counter)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report-sha256', required=True)
    run(parser.parse_args().report_sha256)
