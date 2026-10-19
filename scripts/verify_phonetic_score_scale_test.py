"""Independent score and metric replay of the frozen TEST1000 screen."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import zip_longest
import json
import os
from pathlib import Path
import re
import numpy as np

from scripts.verify_production_phonetic_choice_dev import need, sha, read, jsonl, render_public, verify_resources

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / '.omx/artifacts/production-phonetic-choice-test-v1'
DEV_RUN = ROOT / '.omx/artifacts/score-scale-runtime-dev-v1'
TRAIN = ROOT / '.omx/artifacts/phonetic-score-scale-training-v1'
TEST = ROOT / 'data/challenge/sentence-v3/test.jsonl'


def metrics(records, labels, baseline):
    need(len(records) == len(labels) == 1000 and len({r['id'] for r in records}) == 1000, 'TEST coverage differs')
    totals = {b: Counter(rows=0, top1=0, top5=0, errors=0) for b in ('parent', 'proposed')}
    categories = {b: defaultdict(lambda: Counter(rows=0, top1=0, top5=0, errors=0)) for b in totals}
    changes, stages, english = Counter(), Counter(), Counter()
    ascii_words = lambda text: tuple(re.findall(r'[A-Za-z][A-Za-z0-9_]*', text))
    literal = lambda text: ''.join(c for c in text if not ('a' <= c <= 'z' or re.fullmatch('[가-힣ᄀ-ᇿ㄰-㆏]', c)))
    for row, label in zip(records, labels):
        need(row['id'] == label['id'] and row['request'] == dict(input=label['input'], context='')
             and label.get('context', '') == '' and label['split'] == 'test', 'label/source identity differs')
        gold = set(label['expected'])
        flags = {}
        for branch, selected in (('parent', baseline), ('proposed', 'proposed')):
            texts = [c['text'] for c in row[selected]['candidates']]
            need(texts and row[selected]['result'] == texts[0] and len(texts) <= 5, 'public contract differs')
            first, any_public = texts[0] in gold, bool(gold.intersection(texts))
            flags[branch] = first, any_public
            observation = dict(rows=1, top1=int(first), top5=int(any_public), errors=0)
            totals[branch].update(observation)
            categories[branch][label.get('category', 'uncategorized')].update(observation)
        old, new = flags['parent'], flags['proposed']
        changes['corrections'] += not old[0] and new[0]
        changes['regressions'] += old[0] and not new[0]
        changes['top5_losses'] += old[1] and not new[1]
        if row['proposed']['sentence_ranking_applied']:
            pool = bool(gold.intersection(row['actual_base_pool']))
            stages.update(dict(ranked_rows=1, actual_base_pool_gold=int(pool), public_top5_gold=int(new[1]),
                selected_gold=int(new[0]), actual_pool_to_public_loss=int(pool and not new[1]), actual_pool_to_first_loss=int(pool and not new[0])))
        source_literal = literal(row['request']['input'])
        english['literal_signature_regressions'] += literal(row[baseline]['result']) == source_literal and literal(row['proposed']['result']) != source_literal
        signatures = {ascii_words(text) for text in gold}
        if any(signatures):
            before, after = ascii_words(row[baseline]['result']) in signatures, ascii_words(row['proposed']['result']) in signatures
            english.update(dict(rows_with_reference_ascii_words=1, parent_exact_signature=int(before),
                                proposed_exact_signature=int(after), signature_regressions=int(before and not after)))
    improvement = changes['corrections'] > 0 and changes['regressions'] == changes['top5_losses'] == 0
    result = {b: dict(counts=dict(totals[b]), categories={k: dict(v) for k, v in categories[b].items()}, passed=totals[b]['top1'] == 1000) for b in totals}
    result.update(dict(changes), improvement_gate=bool(improvement), quality_gate=bool(improvement and not english['signature_regressions'] and not english['literal_signature_regressions']),
                  english_ascii_signature=dict(english), actual_base_pool_stage_counts=dict(stages))
    return result


def verify(output):
    output = Path(output).resolve()
    need(not (output / 'independent-verification.json').exists(), 'proof already exists')
    need(not any((output / n).exists() for n in ('failure.json', 'resource-watchdog-failure.json')), 'failed screen')
    launch = read(output.with_suffix('.launch.json'))
    try:
        os.kill(launch['pid'], 0)
    except ProcessLookupError:
        pass
    else:
        raise ValueError('screen process still present')
    runner = ROOT / 'scripts/evaluate_phonetic_score_scale_test.py'
    need(launch['command'] == [str(ROOT / '.venv/bin/python'), '-m', 'scripts.evaluate_phonetic_score_scale_test', '--output', str(output)]
         and launch['runner_sha256'] == sha(runner), 'launch identity differs')
    report, closure, completion = [read(output / n) for n in ('report.json', 'closure.json', 'completion.json')]
    need(report['completed'] is True and completion['completed'] is True and completion['report_sha256'] == sha(output / 'report.json'), 'completion differs')
    need(closure == dict(rows=1000, predictions_sha256=sha(output / 'predictions.jsonl'), labels_opened=False)
         and report['predictions_sha256'] == closure['predictions_sha256'] and report['closure_sha256'] == sha(output / 'closure.json'), 'prediction closure differs')
    pins = read(output / 'pins.json')
    need(pins == report['pins'] and pins[str(runner)] == sha(runner), 'runtime pins differ')
    for name, digest in pins.items():
        if Path(name).resolve() != TEST.resolve():
            need(sha(name) == digest, 'pin changed: ' + name)
    need(sha(DEV_RUN / 'independent-verification.json') == '60b8c4c16e00d36274e2f71760888ea5c12d519669ab4ad38cad5745dd0fd9c0', 'actual DEV proof differs')
    dev_proof = read(DEV_RUN / 'independent-verification.json')
    need(dev_proof['verified'] is True and dev_proof['quality_gate'] is True
         and dev_proof['report_sha256'] == sha(DEV_RUN / 'report.json'), 'actual DEV qualification differs')
    need(sha(SOURCE / 'capture.jsonl') == 'fa93215001ea47a9b36796932ec965ee4cdfaf52593a4fab956b51a74961faf0', 'original TEST capture differs')
    need(sha(SOURCE / 'independent-verification.json') == '0ce8bc781d55ce2ac159447ad3e17082c331f17527ad3aa819cba7ab0865212c', 'original TEST proof differs')
    need(sha(TRAIN / 'model.json') == 'e07462d50da0b67304a82a9a261068ccc84b8055a99b5d70af56db530d245838', 'model differs')
    model = read(TRAIN / 'model.json')
    scale, coefficients = np.array(model['scale']), np.array(model['parameters'])
    records, count_paths = [], 0
    for original, result in zip_longest(jsonl(SOURCE / 'capture.jsonl'), jsonl(output / 'predictions.jsonl')):
        need(original is not None and result is not None, 'prediction/source count differs')
        need(result['id'] == original['id'] and result['request'] == original['request']
             and result['request']['context'] == '' and result['kind'] == 'cached_score_projection', 'source identity differs')
        paths = original['base_paths']
        need(result['actual_base_pool'] == original['actual_base_pool'] == [p[0] for p in paths], 'candidate domain differs')
        for name, branch in (('parent', 'parent'), ('old_calibration', 'proposed')):
            need(result[name] == {k: original[branch][k] for k in ('result', 'candidates', 'sentence_ranking_applied')}, 'baseline changed')
        feature_rows = original['proposed']['phonetic_choice']['rows']
        if original['proposed']['sentence_ranking_applied']:
            need(paths and len(paths) == len(feature_rows) == len(result['score_rows']), 'score coverage differs')
            matrix = np.column_stack(([p[1] for p in paths], np.array([r['features'] for r in feature_rows]) / scale))
            scores = matrix @ coefficients
            for path, feature, score, saved in zip(paths, feature_rows, scores, result['score_rows']):
                need(saved['text'] == feature['text'] == path[0] and saved['base_score'] == feature['base_score'] == path[1]
                     and saved['features'] == feature['features'], 'feature alignment differs')
                np.testing.assert_allclose(saved['score'], score, atol=1e-12, rtol=1e-12)
            sorted_paths = sorted([(p[0], float(s), p[2]) for p, s in zip(paths, scores)], key=lambda p: (-p[1], p[0]))
            public = render_public(original['request']['input'], sorted_paths)
        else:
            need(not paths and not feature_rows and not result['score_rows'], 'unranked projection differs')
            public = original['parent']['candidates']
        need(result['proposed'] == dict(result=public[0]['text'], candidates=public, sentence_ranking_applied=original['proposed']['sentence_ranking_applied']), 'matrix/public replay differs')
        count_paths += len(paths)
        records.append({k: v for k, v in result.items() if k != 'score_rows'})
    need(len(records) == 1000, 'TEST count differs')
    # Labels become available only after the full immutable numerical replay.
    need(sha(TEST) == pins[str(TEST)] == '1dcf804460b40c445abf688dc47b452937d8911ac156b8520141fad25b973d06', 'label identity differs')
    labels = list(jsonl(TEST))
    comparisons = {branch: metrics(records, labels, branch) for branch in ('parent', 'old_calibration')}
    need(comparisons == report['metrics'], 'independent complete metrics differ')
    preserved = lambda m: not any((m['regressions'], m['top5_losses'], m['english_ascii_signature']['signature_regressions'], m['english_ascii_signature']['literal_signature_regressions']))
    passed = comparisons['parent']['corrections'] > 0 and all(preserved(m) for m in comparisons.values())
    need(report['quality_gate'] == report['actual_runtime_authorized'] == passed, 'decision differs')
    need(report['all_first_choices_correct'] == (comparisons['parent']['proposed']['counts']['top1'] == 1000), 'all-correct claim differs')
    trace = list(jsonl(output / 'resources.jsonl'))
    need(all(r['elapsed_s'] <= 120 and r['rss_bytes'] <= 1024**3 for r in trace), 'screen resource bound exceeded')
    resources = verify_resources(trace, ['authenticate_dev_then_test_sources', 'project_all', 'late_label_join', 'write_report', 'write_completion'])
    need(report['training_performed'] is False and report['model_inference_performed'] is False
         and report['production_changed'] is False and report['labels_opened_after_predictions'] is True, 'screen scope differs')
    proof = dict(verified=True, report_sha256=sha(output / 'report.json'), verifier_sha256=sha(__file__),
        helper_sha256=sha(ROOT / 'scripts/verify_production_phonetic_choice_dev.py'),
        launch_sha256=sha(output.with_suffix('.launch.json')), quality_gate=passed,
        counts=dict(rows=1000, paths=count_paths), metrics=comparisons, resources=resources,
        scope='All1000 matrix scores, public candidates, baseline identities, complete category/stage/preservation metrics, resource and closure records.',
        limitations='Cached regression replay; no actual new TEST inference, unseen generalization, full COPY-span correctness or web deployment claim.')
    with (output / 'independent-verification.json').open('x') as stream:
        json.dump(proof, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    result = verify(parser.parse_args().output)
    print(json.dumps(dict(verified=result['verified'], quality_gate=result['quality_gate'], counts=result['counts'])))
