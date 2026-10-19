"""Replay DEV screen with independent matrix scoring and frozen metric helpers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from scripts.verify_production_phonetic_choice_dev import (
    aggregate, need, sha, read, jsonl, render_public, verify_resources,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / '.omx/artifacts/production-phonetic-choice-dev-v1'
TRAIN = ROOT / '.omx/artifacts/phonetic-score-scale-training-v1'
DEV = ROOT / 'data/challenge/sentence-v3/dev.jsonl'


def verify(output):
    output = Path(output)
    need(not (output / 'independent-verification.json').exists(), 'proof already exists')
    need(not (output / 'failure.json').exists() and not (output / 'resource-watchdog-failure.json').exists(), 'failed screen')
    report, completion, closure = (read(output / n) for n in ('report.json', 'completion.json', 'closure.json'))
    need(report['completed'] is True and completion['completed'] is True
         and completion['report_sha256'] == sha(output / 'report.json'), 'completion differs')
    need(closure == dict(rows=316, predictions_sha256=sha(output / 'predictions.jsonl'), labels_opened=False)
         and report['predictions_sha256'] == closure['predictions_sha256']
         and report['closure_sha256'] == sha(output / 'closure.json'), 'prediction closure differs')
    pins = read(output / 'pins.json')
    need(pins == report['pins'], 'pin manifest differs')
    for name, digest in pins.items():
        if Path(name).resolve() != DEV.resolve():
            need(sha(name) == digest, 'pin differs: ' + name)
    for directory in (SOURCE, TRAIN):
        proof = read(directory / 'independent-verification.json')
        need(proof['verified'] is True and proof['report_sha256'] == sha(directory / 'report.json'), 'input proof differs')
    need(sha(SOURCE / 'capture.jsonl') == 'f562b5e813484fa49e2d7bb7d231bde63dd40201120eb0bae2d1043f7f4dc432', 'source identity differs')
    need(sha(TRAIN / 'model.json') == 'e07462d50da0b67304a82a9a261068ccc84b8055a99b5d70af56db530d245838', 'model identity differs')
    model = read(TRAIN / 'model.json')
    parameters, scale = np.array(model['parameters']), np.array(model['scale'])
    predictions = list(jsonl(output / 'predictions.jsonl'))
    need(len(predictions) == 316 and len({r['id'] for r in predictions}) == 316, 'coverage differs')
    original_count = 0
    for index, original in enumerate(jsonl(SOURCE / 'capture.jsonl')):
        need(index < len(predictions), 'extra original row')
        result = predictions[index]
        original_count += 1
        need(result['id'] == original['id'] and result['request'] == original['request']
             and result['request']['context'] == '' and result['kind'] == 'cached_score_projection', 'source contract differs')
        need(result['parent_pool'] == original['parent_pool'], 'pool changed')
        for branch, old in (('parent', 'parent'), ('old_calibration', 'proposed')):
            need(result[branch] == {k: original[old][k] for k in ('result', 'candidates', 'sentence_ranking_applied')}, 'baseline differs')
        feature_rows, paths = original['proposed']['phonetic_choice']['rows'], original['base_paths']
        if original['proposed']['sentence_ranking_applied']:
            need(len(feature_rows) == len(paths) == len(result['score_rows']) and paths, 'path denominator differs')
            matrix = np.array([[p[1], *(np.array(f['features']) / scale)] for p, f in zip(paths, feature_rows)])
            scores = matrix @ parameters
            for path, feature, score, recorded in zip(paths, feature_rows, scores, result['score_rows']):
                need(recorded['text'] == path[0] and recorded['base_score'] == path[1]
                     and recorded['features'] == feature['features'], 'score source differs')
                np.testing.assert_allclose(recorded['score'], score, rtol=1e-12, atol=1e-12)
            recomputed = sorted([(p[0], float(s), p[2]) for p, s in zip(paths, scores)], key=lambda p: (-p[1], p[0]))
            public = render_public(original['request']['input'], recomputed)
        else:
            need(not paths and not feature_rows and not result['score_rows'], 'unranked projection differs')
            public = original['parent']['candidates']
        need(result['proposed'] == dict(result=public[0]['text'], candidates=public,
             sentence_ranking_applied=original['proposed']['sentence_ranking_applied']), 'replayed public differs')
    need(original_count == 316, 'original denominator differs')
    # Only after all source-derived outputs and coefficients have been replayed.
    need(sha(DEV) == pins[str(DEV)], 'deferred labels differ')
    labels = list(jsonl(DEV))
    need(len(labels) == 316, 'label denominator differs')
    comparisons = {}
    for branch in ('parent', 'old_calibration'):
        result = aggregate([r | {'parent': r[branch]} for r in predictions], labels)
        recorded = report['metrics'][branch]
        for field in ('corrections', 'regressions', 'top5_losses'):
            need(recorded[field] == result[field], 'metric differs: ' + field)
        for field in ('ranked_rows', 'pool_correct'):
            need(recorded[field] == result['stage_counts'][field], 'stage metric differs')
        english = result['english_ascii_signature']
        need(recorded['ascii_signature_regressions'] == english['signature_regressions']
             and recorded['literal_signature_regressions'] == english['literal_signature_regressions'], 'preservation metric differs')
        for field, counterpart in ((branch, 'parent'), ('proposed', 'proposed')):
            need(recorded['counts'][field] == {k: result[counterpart]['counts'][k] for k in ('rows', 'top1', 'top5')}, 'counts differ')
        preserve = not any((result['regressions'], result['top5_losses'], english['signature_regressions'], english['literal_signature_regressions']))
        need(recorded['preservation_gate'] == preserve, 'preservation gate differs')
        comparisons[branch] = result
    passed = comparisons['parent']['corrections'] > 0 and all(report['metrics'][branch]['preservation_gate'] for branch in comparisons)
    need(report['quality_gate'] == report['actual_runtime_authorized'] == passed, 'quality gate differs')
    trace = list(jsonl(output / 'resources.jsonl'))
    need(all(r['elapsed_s'] <= 120 and r['rss_bytes'] <= 1024**3 for r in trace), 'screen-specific resource limit')
    resources = verify_resources(trace, ['authenticate', 'project_all', 'late_label_join', 'write_report', 'write_completion'])
    need(report['model_inference_performed'] is False and report['training_performed'] is False
         and report['test_accessed'] is False and report['production_changed'] is False, 'scope differs')
    proof = dict(verified=True, report_sha256=sha(output / 'report.json'), verifier_sha256=sha(__file__),
                 helper_sha256=sha(ROOT / 'scripts/verify_production_phonetic_choice_dev.py'),
                 comparisons=comparisons, quality_gate=passed, resources=resources,
                 scope='All316 cached rows, matrix scores, public candidate ordering, both baseline metrics and resource trace.',
                 limitations='Not actual runtime inference, unseen generalization, complete COPY-span correctness or TEST1000 success.')
    with (output / 'independent-verification.json').open('x') as stream:
        json.dump(proof, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    result = verify(parser.parse_args().output)
    print(json.dumps({k: result[k] for k in ('verified', 'quality_gate', 'resources')}))
