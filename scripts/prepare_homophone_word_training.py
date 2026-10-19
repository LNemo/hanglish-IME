"""Build a document-split selector dataset from frozen real runtime groups."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_ensemble_regression import file_sha256, verify_pins
from scripts.probe_natural_homophone_contexts import DATA, project

SALT = 'natural-homophone-selector-doc-v1'


def partition(group_id):
    value = hashlib.sha256((SALT+'\0'+group_id).encode()).hexdigest()
    return 'hold' if int(value,16)%5 == 0 else 'fit'


def assemble(captures, targets, quarantined):
    pools = {'fit':[], 'hold':[]}
    outcomes = []
    seen = set()
    for capture in captures:
        identity = capture['id']
        if identity in seen:
            raise ValueError('duplicate capture')
        seen.add(identity)
        target = targets[identity]
        if target['split'] != 'train':
            raise ValueError('non-TRAIN source supplied')
        if any(capture[k] != v for k,v in project(target).items()):
            raise ValueError('runtime source projection differs')
        groups = [g for g in capture['groups'] if (g['start'],g['end']) ==
                  (capture['focus_start'],capture['focus_end'])]
        if len(groups)>1:
            raise ValueError('duplicate target group')
        role = partition(target['group_id'])
        status = 'quarantined' if identity in quarantined else 'unscored'
        if identity not in quarantined and groups:
            g = groups[0]
            candidates = g['candidates']
            if (len(set(candidates)) != len(candidates)
                    or not candidates
                    or any(len(g[key]) != len(candidates) for key in ('base_scores','teacher_scores','kinds'))):
                raise ValueError('malformed candidate group')
            present = target['word'] in candidates
            status = 'informative' if present and len(candidates)>1 else ('single' if present else 'missing_gold')
            record = {**g, 'id':identity, 'group_id':target['group_id'],
                      'expected':[target['word']], 'role':role,
                      'spelling_anchors':target['spelling_anchors'], 'match_kind':target['match_kind'],
                      'source_ids':target['source_ids'], 'kind':'source'}
            # Missing-gold and singleton cases stay in held diagnostic denominators.
            if role == 'hold' or status == 'informative':
                pools[role].append(record)
        outcomes.append(dict(id=identity, group_id=target['group_id'], role=role, status=status))
    if seen != set(targets):
        raise ValueError('capture does not cover all source rows')
    if {r['group_id'] for r in pools['fit']} & {r['group_id'] for r in pools['hold']}:
        raise ValueError('document leakage')
    if {s for r in pools['fit'] for s in r['source_ids']} & {s for r in pools['hold'] for s in r['source_ids']}:
        raise ValueError('source identity leakage')
    return pools, outcomes


def run(capture_dir, report_sha, review_path, output):
    output.mkdir(parents=True,exist_ok=False)
    report_path = capture_dir/'report.json'
    if file_sha256(report_path) != report_sha:
        raise ValueError('capture report differs')
    report = json.loads(report_path.read_text())
    if not report['completed'] or report['counts']['rows'] != 853:
        raise ValueError('capture incomplete')
    pins = dict(report['pins'])
    pins[str(report_path.relative_to(ROOT))] = report_sha
    capture_path = capture_dir/'capture.jsonl'
    if file_sha256(capture_path) != report['capture_sha256']:
        raise ValueError('capture changed')
    pins[str(capture_path.relative_to(ROOT))] = report['capture_sha256']
    review = json.loads(review_path.read_text())
    source_path = DATA/'selected_train.jsonl'
    if file_sha256(source_path) != review['source_sha256']:
        raise ValueError('review source differs')
    for path in (review_path, Path(__file__), ROOT/'scripts/probe_natural_homophone_contexts.py'):
        pins[str(path.relative_to(ROOT))] = file_sha256(path)
    verify_pins(pins)
    targets = {r['id']:r for r in map(json.loads,source_path.read_text().splitlines())}
    captures = list(map(json.loads,capture_path.read_text().splitlines()))
    pools, outcomes = assemble(captures,targets,{r['id'] for r in review['quarantined']})
    outputs = {}
    for role, rows in pools.items():
        path = output/(role+'.jsonl')
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        outputs[role] = dict(rows=len(rows), documents=len({r['group_id'] for r in rows}),
                             sha256=file_sha256(path), anchor_counts=dict(Counter(a for r in rows for a in r['spelling_anchors'])))
    counts = {role:dict(Counter(r['status'] for r in outcomes if r['role']==role)) for role in pools}
    manifest = dict(schema='homophone-word-training-v1', completed=True, training_ready=True,
                    supervision='source spelling with limited review; not expert sense labels',
                    production_changed=False, held_source_1583_read=False, split_salt=SALT,
                    outputs=outputs, counts=counts, outcomes=outcomes, pins=pins,
                    limitations=review['scope_limitations'])
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(outputs=outputs,counts=counts),ensure_ascii=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--capture-dir',type=Path,required=True)
    parser.add_argument('--report-sha256',required=True)
    parser.add_argument('--review',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.capture_dir.resolve(),args.report_sha256,args.review.resolve(),args.output.resolve())
