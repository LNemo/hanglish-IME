"""Freeze document components, then join source labels to actual runtime captures."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_ensemble_regression import file_sha256, verify_pins
from scripts.probe_natural_homophone_contexts import project

DATA = ROOT / 'data/natural_context/runtime-collision-source-v5'
SOURCE_SHA = '293c3992878a9f7bdc532669ef2bf233233dd6ab9a4fddd7af1209e6b593de89'
PLAN = ROOT / '.omx/plans/expanded-word-split-v1.json'
CAPTURE = ROOT / '.omx/artifacts/runtime-collision-training-capture-v1'
REVIEW = ROOT / '.omx/artifacts/runtime-collision-source-review-v2.json'
OUTPUT = ROOT / 'data/natural_context/runtime-collision-word-training-v1'
SALT = 'runtime-collision-selector-doc-v1'


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def identities(rows):
    return ({r['group_id'] for r in rows}, {s for r in rows for s in r['source_ids']})


def partition(rows, exposed, protected):
    """Close shared document/source identities before deterministic assignment."""
    groups, sources = identities(rows)
    if groups & protected[0] or sources & protected[1]:
        raise ValueError('previous holdout source supplied')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('duplicate source ID')
    parent = {g: g for g in groups}

    def find(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    owners = {}
    for row in rows:
        for source in row['source_ids']:
            g = find(row['group_id'])
            if source in owners:
                other = find(owners[source])
                parent[max(g, other)] = min(g, other)
            owners[source] = row['group_id']
    components = defaultdict(list)
    for row in rows:
        components[find(row['group_id'])].append(row)
    result = {}
    for key, members in sorted(components.items()):
        member_groups, member_sources = identities(members)
        seen = bool(member_groups & exposed[0] or member_sources & exposed[1])
        hold = int(hashlib.sha256((SALT + '\0' + key).encode()).hexdigest(), 16) % 5 == 0
        role = 'fit' if seen or not hold else 'hold'
        for row in members:
            result[row['id']] = dict(role=role, component=key,
                                    reason='previously_exposed_document' if seen else 'salted_document_split')
    return dict(sorted(result.items()))


def freeze():
    if file_sha256(DATA / 'manifest.json') != SOURCE_SHA:
        raise ValueError('source manifest changed')
    pins = {}

    def pinned(path):
        pins[str(path.relative_to(ROOT))] = file_sha256(path)
        return read_rows(path)

    selected = pinned(DATA / 'selected_train.jsonl')
    exposed = pinned(ROOT / 'data/natural_context/homophone-source-v1/selected_train.jsonl')
    pilot_ids = {r['id'] for r in pinned(ROOT / '.omx/artifacts/runtime-collision-pilot-v1/inputs.jsonl')}
    pilot = pinned(ROOT / 'data/natural_context/runtime-collision-source-v1/selected_train.jsonl')
    exposed += [r for r in pilot if r['id'] in pilot_ids]
    if not pilot_ids <= {r['id'] for r in pilot}:
        raise ValueError('pilot source identities missing')
    natural_path = ROOT / '.omx/artifacts/productive-endings-natural-v1/targets.json'
    pins[str(natural_path.relative_to(ROOT))] = file_sha256(natural_path)
    natural_ids = {r['id'] for r in json.loads(natural_path.read_text())}
    train = pinned(DATA / 'train.jsonl')
    exposed += [r for r in train if r['id'] in natural_ids]
    if not natural_ids <= {r['id'] for r in train}:
        raise ValueError('natural diagnostic source identities missing')
    protected = pinned(ROOT / 'data/natural_context/homophone-source-v1/hold.jsonl')
    protected += pinned(ROOT / 'data/natural_context/homophone-word-training-v1/hold.jsonl')
    assignments = partition(selected, identities(exposed), identities(protected))
    if len(assignments) != 2048 or any(r['split'] != 'train' for r in selected):
        raise ValueError('wrong source selection')
    pins[str((DATA / 'manifest.json').relative_to(ROOT))] = SOURCE_SHA
    payload = dict(schema='expanded-word-split-v1', salt=SALT,
                   created_at=datetime.now(timezone.utc).isoformat(), pins=pins,
                   assignments=assignments, counts=dict(Counter(r['role'] for r in assignments.values())),
                   policy='fixed before examining current capture outcomes; previously exposed documents forced fit',
                   holdout_scope='new document split relative to listed task experiments; not foundation pretraining audit')
    PLAN.parent.mkdir(parents=True, exist_ok=True)
    with PLAN.open('x') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(dict(counts=payload['counts'], split_sha256=file_sha256(PLAN))))


def validate_capture(capture, target):
    if any(capture.get(k) != v for k, v in project(target).items()):
        raise ValueError('source projection differs')
    start, end = capture['focus_start'], capture['focus_end']
    groups, pieces = capture['groups'], capture['ranked_pieces']
    if len(groups) > 1 or len(pieces) > 1:
        raise ValueError('duplicate focus span')
    for obj in groups + pieces:
        if (obj['start'], obj['end'], obj['source']) != (start, end, capture['input'][start:end]):
            raise ValueError('focus span differs')
    preliminary = capture['preliminary']
    cursor = 0
    for piece in preliminary:
        if piece['start'] != cursor or piece['source'] != capture['input'][cursor:piece['end']]:
            raise ValueError('preliminary span differs')
        cursor = piece['end']
    if cursor != len(capture['input']):
        raise ValueError('incomplete preliminary coverage')
    if not groups:
        return
    group = groups[0]
    candidates = group['candidates']
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError('invalid candidates')
    if any(len(group[k]) != len(candidates) for k in ('base_scores', 'teacher_scores', 'kinds')):
        raise ValueError('candidate feature mismatch')
    if any(not math.isfinite(v) for k in ('base_scores', 'teacher_scores') for v in group[k]):
        raise ValueError('nonfinite scores')
    indices = [i for i, p in enumerate(preliminary) if (p['start'], p['end']) == (start, end)]
    if len(indices) != 1 or not pieces:
        raise ValueError('scored focus is unaligned')
    i = indices[0]
    context = capture['context'] + ' ' + ''.join(p['text'] for p in preliminary[:i])
    context += ' [MASK] ' + ''.join(p['text'] for p in preliminary[i + 1:])
    if group['context'] != context:
        raise ValueError('scored context differs')
    scores = dict(zip(candidates, [t + .15 * b for t, b in zip(group['teacher_scores'], group['base_scores'])]))
    kinds = dict(zip(candidates, group['kinds']))
    options = pieces[0]['options']
    if len({o['text'] for o in options}) != len(options) or not set(candidates) <= {o['text'] for o in options}:
        raise ValueError('ranked options differ')
    for option in options:
        text = option['text']
        if text in scores:
            if abs(option['score'] - scores[text]) > 1e-6 or option['kind'] != kinds[text]:
                raise ValueError('runtime adjusted score differs')
        elif option['kind'] != 'raw' or option['score'] != -10000:
            raise ValueError('unexplained unscored option')
    if options != sorted(options, key=lambda o: (-o['score'], o['text'])):
        raise ValueError('ranked order differs')


def assemble(captures, targets, assignments, quarantined):
    if not quarantined <= set(targets):
        raise ValueError('quarantine ID absent from selected source')
    pools = {'fit': [], 'hold': []}
    outcomes = []
    seen = set()
    for capture in captures:
        identity = capture['id']
        if identity in seen:
            raise ValueError('duplicate capture')
        seen.add(identity)
        target = targets[identity]
        if target['split'] != 'train':
            raise ValueError('non-TRAIN source')
        validate_capture(capture, target)
        role = assignments[identity]['role']
        record = dict(id=identity, group_id=target['group_id'], source_ids=target['source_ids'],
                      role=role, expected=[target['word']], family=target['word_input'])
        groups, pieces = capture['groups'], capture['ranked_pieces']
        if groups:
            record.update(groups[0], kind='source')
            present = target['word'] in record['candidates']
            status = 'informative' if present and len(record['candidates']) > 1 else 'single' if present else 'missing_gold'
        else:
            record.update(kind='fixed', fixed_result=pieces[0]['options'][0]['text'] if pieces else None)
            status = 'unscored' if pieces else 'unaligned'
        if identity in quarantined:
            status = 'quarantined'
        elif role == 'hold' or status == 'informative':
            pools[role].append(record)
        outcomes.append(dict(id=identity, group_id=target['group_id'], role=role, status=status))
    if seen != set(targets) or set(assignments) != seen:
        raise ValueError('capture/partition coverage differs')
    for left, right in zip(identities(pools['fit']), identities(pools['hold'])):
        if left & right:
            raise ValueError('document/source leakage')
    return pools, outcomes


def prepare():
    plan = json.loads(PLAN.read_text())
    verify_pins(plan['pins'])
    report_path = CAPTURE / 'report.json'
    report = json.loads(report_path.read_text())
    if not report['completed'] or report['rows'] != 2048 or report['source_manifest_sha256'] != SOURCE_SHA:
        raise ValueError('capture incomplete or wrong source')
    if report['model_labels_supplied'] or report['sentence_inference']:
        raise ValueError('unexpected capture procedure')
    pins = {**plan['pins'], **report['pins']}
    for filename, key in [('capture.jsonl', 'capture_sha256'), ('inputs.jsonl', 'inputs_sha256')]:
        if file_sha256(CAPTURE / filename) != report[key]:
            raise ValueError('capture bytes changed')
    review = json.loads(REVIEW.read_text())
    if len(review['quarantined_ids']) != len(set(review['quarantined_ids'])) or len(review['quarantined_ids']) != 5:
        raise ValueError('quarantine selection differs')
    if review['selected_sha256'] != file_sha256(DATA / 'selected_train.jsonl'):
        raise ValueError('review does not cover selected bytes')
    inherited = ROOT / review['inherited_review']
    if file_sha256(inherited) != review['inherited_review_sha256']:
        raise ValueError('inherited review changed')
    for path in (PLAN, report_path, CAPTURE / 'capture.jsonl', CAPTURE / 'inputs.jsonl', REVIEW, inherited, Path(__file__)):
        pins[str(path.relative_to(ROOT))] = file_sha256(path)
    verify_pins(pins)
    targets = {r['id']: r for r in read_rows(DATA / 'selected_train.jsonl')}
    captures = read_rows(CAPTURE / 'capture.jsonl')
    if [{k: row[k] for k in ('id', 'input', 'context', 'focus_start', 'focus_end')} for row in captures] != read_rows(CAPTURE / 'inputs.jsonl'):
        raise ValueError('captured inference inputs changed')
    pools, outcomes = assemble(captures, targets, plan['assignments'], set(review['quarantined_ids']))
    OUTPUT.mkdir(parents=True, exist_ok=False)
    outputs = {}
    for role, rows in pools.items():
        path = OUTPUT / (role + '.jsonl')
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        outputs[role] = dict(rows=len(rows), documents=len(identities(rows)[0]),
                             families=len({r['family'] for r in rows}), sha256=file_sha256(path))
    manifest = dict(schema='runtime-collision-word-training-v1', completed=True, training_ready=True,
                    training_requires_independent_verification=True, production_changed=False,
                    pins=pins, outputs=outputs, outcomes=outcomes, split_salt=SALT,
                    counts={role: dict(Counter(r['status'] for r in outcomes if r['role'] == role)) for role in pools},
                    limitations=['Source spellings have limited review, not expert sense annotation.',
                                 'Word selection diagnostic; full runtime evaluation required for promotion.',
                                 'Quarantined rows excluded explicitly; unscored/missing-gold HOLD rows retained.'])
    (OUTPUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(outputs=outputs, counts=manifest['counts'], manifest_sha256=file_sha256(OUTPUT / 'manifest.json'))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'prepare'])
    args = parser.parse_args()
    freeze() if args.action == 'freeze' else prepare()
