"""Final bounded DEV comparison after the mixed runtime capture is closed."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.capture_ungated_joint_dev import COMPARATOR, PARENT_VERSION, need, read, sha, verify_pins, write
from scripts.evaluate_ungated_joint_runtime_dev import LABELS, LABEL_SHA, row_metrics
from scripts.evaluate_ungated_mixed_union_fit import edit_distance, transitions

PRIOR = ROOT / '.omx/artifacts/ungated-joint-runtime-dev-v1'
BRANCHES = ('production', 'joint', 'mixed')


def measure(source, candidates, expected):
    result = row_metrics(source, candidates, expected)
    result['edit_distance'] = min(edit_distance(candidates[0]['text'], text) for text in expected)
    return result


def public_texts(trace, baseline):
    pool = trace['pool']
    supported = sorted((p for p in pool if p['supported']), key=lambda p: (-p['score'], p['text']))
    return [p['text'] for p in supported[:4]] + [baseline['input']] if supported else [p['text'] for p in baseline['candidates']]


def authenticate(directory):
    report, closure = read(directory/'report.json'), read(directory/'capture.complete.json')
    need(report.get('completed') is True and report.get('rows') == 316
         and report.get('selected_rows') == 4 and report.get('labels_opened') is False
         and report.get('production_changed') is False
         and closure.get('completed') is True and closure.get('rows') == 316
         and closure.get('selected_rows') == 4
         and closure.get('report_sha256') == sha(directory/'report.json')
         and closure.get('capture_sha256') == report.get('capture_sha256') == sha(directory/'capture.jsonl'),
         'closed source-only mixed DEV capture required')
    verify_pins(report['pins'])
    prior_report, prior_proof = read(PRIOR/'report.json'), read(PRIOR/'independent-evaluation.json')
    need(prior_proof.get('verified') is True and prior_proof.get('rows') == 316
         and prior_proof['hashes']['report.json'] == sha(PRIOR/'report.json')
         and prior_proof['hashes']['capture.jsonl'] == prior_report['capture_sha256'] == sha(PRIOR/'capture.jsonl'),
         'verified prior joint comparison required')
    references = {r['id']: r for r in map(json.loads, (COMPARATOR/'capture.jsonl').open())}
    prior = {r['id']: r for r in map(json.loads, (PRIOR/'capture.jsonl').open())}
    rows = [json.loads(line) for line in (directory/'capture.jsonl').open()]
    need(len(rows) == len(references) == len(prior) == 316
         and len({r['id'] for r in rows}) == 316
         and {r['id'] for r in rows} == set(references) == set(prior), 'DEV source coverage differs')
    selected = []
    for row in rows:
        fixed, old = references[row['id']], prior[row['id']]
        baseline = fixed['response']
        need(row['request'] == fixed['request'] == old['request']
             and baseline['version'] == PARENT_VERSION, 'DEV source identity differs')
        chosen = not baseline['sentence_ranking_applied']
        need(row['selected_unranked'] == old['selected_unranked'] == chosen, 'DEV source selection differs')
        if not chosen:
            digest = hashlib.sha256(json.dumps(baseline, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            need(row['passthrough_exact'] is True and row['cached_response_sha256'] == digest,
                 'ranked control changed')
            continue
        selected.append(row['id'])
        trace = row['trace']
        need(row['fresh_baseline']['candidates'] == baseline['candidates']
             and trace['baseline'] == row['fresh_baseline'], 'fresh baseline differs')
        need(public_texts(trace['joint_trace'], row['fresh_baseline'])
             == [p['text'] for p in old['response']['candidates']], 'fresh joint public parity differs')
        pool = trace['union_pool']
        need(all((isinstance(p['score'], (int,float)) if p['supported'] else p['score'] is None)
                 for p in pool), 'union support/score differs')
        expected_public = public_texts({'pool': pool}, row['fresh_baseline'])
        need([p['text'] for p in row['response']['candidates']] == expected_public,
             'mixed public ordering differs')
        need(row['response']['candidates'][-1] == baseline['candidates'][-1], 'raw reserve changed')
    need(len(selected) == 4 and selected == report['selected_ids'], 'selected DEV denominator differs')
    return report, rows, references, prior


def evaluate(directory):
    report, rows, references, prior = authenticate(directory)
    need(sha(LABELS) == LABEL_SHA, 'DEV labels changed')
    labels = {r['id']: r for r in map(json.loads, LABELS.open())}
    need(set(labels) == set(references), 'DEV labels differ')
    totals = {name: Counter(rows=0) for name in BRANCHES}
    selected_totals = {name: Counter(rows=0) for name in BRANCHES}
    changes = {name: Counter() for name in ('production_to_mixed', 'joint_to_mixed')}
    outcomes = []
    for row in rows:
        label = labels[row['id']]
        need(row['request'] == {'input': label['input'], 'context': ''}, 'DEV label input differs')
        base = references[row['id']]['response']['candidates']
        selected = row['selected_unranked']
        candidates = {'production': base,
                      'joint': prior[row['id']]['response']['candidates'] if selected else base,
                      'mixed': row['response']['candidates'] if selected else base}
        values = {k: measure(label['input'], v, label['expected']) for k,v in candidates.items()}
        for name, value in values.items():
            count = {'rows': 1, **{k: int(v) for k,v in value.items()}}
            totals[name].update(count)
            if selected: selected_totals[name].update(count)
        delta = {name+'_to_mixed': transitions(values[name], values['mixed']) for name in ('production', 'joint')}
        for name, value in delta.items(): changes[name].update(value)
        outcome = {'id': row['id'], 'selected_unranked': selected, 'metrics': values, 'transitions': delta}
        if selected:
            gold = set(label['expected'])
            outcome['coverage'] = {name: bool({p['text'] for p in pool} & gold) for name,pool in
                [('joint', row['trace']['joint_trace']['pool']), ('mixed', row['trace']['union_pool'])]}
        outcomes.append(outcome)
    need(totals['production']['first'] == 267 and totals['production']['top5'] == 284
         and totals['production']['ascii_exact'] == 55 and totals['production']['ascii_reference'] == 59,
         'frozen production DEV baseline differs')
    need(authenticate(directory)[0] == report and sha(LABELS) == LABEL_SHA, 'evaluation sources changed')
    with (directory/'outcomes.jsonl').open('x') as stream:
        for row in outcomes: stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    regression_fields = ('regressions','top5_losses','ascii_regressions','literal_regressions','partial_regressions')
    quality = totals['mixed']['first'] > totals['production']['first'] and not any(changes['production_to_mixed'][k] for k in regression_fields)
    result = {'completed': True, 'rows': 316, 'selected_rows': 4,
              'branches': {k: dict(v) for k,v in totals.items()},
              'selected_branches': {k: dict(v) for k,v in selected_totals.items()},
              'transitions': {k: dict(v) for k,v in changes.items()}, 'quality_gate': bool(quality),
              'selection': 'experimental_eligible_for_final_checks' if quality else 'retain_verified_production',
              'capture_sha256': sha(directory/'capture.jsonl'), 'report_sha256': sha(directory/'report.json'),
              'prior_capture_sha256': sha(PRIOR/'capture.jsonl'), 'labels_sha256': LABEL_SHA,
              'outcomes_sha256': sha(directory/'outcomes.jsonl'), 'evaluator_sha256': sha(Path(__file__)),
              'labels_opened_after_source_closure': True, 'test_opened': False, 'production_changed': False,
              'scope': 'Previously exposed DEV;312cached production controls and4fresh mixed-runtime rows. Final bounded phase-one comparison; no followup research.'}
    write(directory/'evaluation.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    evaluate(parser.parse_args().directory)
