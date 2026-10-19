"""Freeze v3 evaluation document IDs from only the untouched v2 reserve."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / '.omx/artifacts/finite-final-choice-partition-v3/manifest.json'
SALT = 'finite-final-choice-v3-2026-09-15'
TARGET = 1024
PLAN = '.omx/artifacts/finite-final-choice-plan-v3.md'
V1 = '.omx/artifacts/final-choice-partition-v1/manifest.json'
V2 = '.omx/artifacts/balanced-final-choice-partition-v2/manifest.json'
RAW = 'data/natural_context/klue-v1.1-pilot-v2/train.jsonl'
LINEAGE = '.omx/artifacts/finite-final-choice-warm-lineage-v3/report.json'
PINS = {
 PLAN: 'a9728be30841efa13501c1b04e64adc717d140c9f11e4caac32655f1615d2958',
 V1: 'e997f06558bdb5633446171d35912140fb34b55066c13fa13b668939d6347846',
 V2: '57170914d4f14d5cd725ea0fd52a96b68b0859ba83041b93f1ecc09daa7cc8ca',
 RAW: '904aab04033ac758d5ccda7d8cdd75d9bd14872ad31017a14d884633e5de4d3b',
 LINEAGE: '16b045d2ac8176336b09df9a4026da04fb1adc5d89157b8180f7e4cbf29e4bec',
}

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def require(condition, message):
    if not condition:
        raise ValueError(message)

def unique_strings(values):
    require(isinstance(values,list) and values and all(isinstance(x,str) and x for x in values), 'invalid identity list')
    require(len(set(values)) == len(values), 'duplicated identity list')
    return set(values)

def select_ids(metadata, natural, previous, *, target=TARGET):
    require(type(target) is int and target > 0, 'invalid target')
    ids = unique_strings(previous['remaining_ids'])
    groups = unique_strings(previous['remaining_group_ids'])
    old_ids = unique_strings(previous['eval_ids'])
    old_groups = unique_strings(previous['eval_group_ids'])
    require(not ids & old_ids and not groups & old_groups, 'v2 reserve overlaps exposed evaluation')
    require(ids | old_ids == unique_strings(natural['reserve_ids']), 'v2 does not partition v1 reserve IDs')
    require(groups | old_groups == unique_strings(natural['reserve_group_ids']), 'v2 does not partition v1 reserve groups')
    protected_ids = set().union(*(set(natural[k]) for k in (
        'fit_ids','calibration_ids','holdout_ids','excluded_historical_group_sibling_ids')))
    protected_groups = set().union(*(set(natural[k]) for k in (
        'excluded_historical_group_ids','calibration_group_ids','holdout_group_ids')))
    require(not ids & protected_ids and not groups & protected_groups, 'reserve overlaps historical data')
    by_group = defaultdict(list)
    seen = set()
    for row in metadata:
        identity, group = row['id'], row['group_id']
        require(isinstance(identity,str) and identity and isinstance(group,str) and group, 'invalid source identity')
        require(identity not in seen, 'duplicate source identity')
        seen.add(identity)
        require((identity in ids) == (group in groups), 'reserve contains a partial document')
        if identity in ids:
            by_group[group].append(identity)
    require(set(by_group) == groups and {v for values in by_group.values() for v in values} == ids, 'incomplete reserve metadata')
    ordered = sorted(groups, key=lambda group: (sha((SALT+'\0'+group).encode()),group))
    chosen, chosen_ids = [], []
    for group in ordered:
        if len(chosen_ids) >= target:
            break
        chosen.append(group)
        chosen_ids.extend(by_group[group])
    require(len(chosen_ids) >= target, 'insufficient untouched reserve')
    rest = ordered[len(chosen):]
    return {'eval_ids':chosen_ids,'eval_group_ids':chosen,
            'remaining_ids':[v for g in rest for v in by_group[g]],'remaining_group_ids':rest}

def run(output):
    if output.exists():
        raise FileExistsError(output)
    source = Path(__file__).read_bytes()
    snapshots = {name:(ROOT/name).read_bytes() for name in PINS}
    for name,digest in PINS.items():
        require(sha(snapshots[name]) == digest, f'partition input changed: {name}')
    lineage = json.loads(snapshots[LINEAGE])
    require(lineage['passed'] is True and lineage['warm_natural_subset_of_fit'] is True
            and lineage['warm_optimizer_reserve_overlap_ids'] == 0
            and lineage['warm_optimizer_reserve_overlap_groups'] == 0, 'warm initializer overlaps reserve')
    require(lineage['input_hashes_before'] == lineage['input_hashes_after'], 'lineage inputs changed')
    hashes = dict(PINS)
    for name,digest in lineage['input_hashes_before'].items():
        require(name not in hashes or hashes[name] == digest, 'lineage hash conflict')
        hashes[name] = digest
        require(sha((ROOT/name).read_bytes()) == digest, f'lineage input changed: {name}')
    metadata = [{'id':r['id'],'group_id':r['group_id']}
                for line in snapshots[RAW].splitlines() if line for r in [json.loads(line)]]
    require(len(metadata) == 9002, 'raw TRAIN row count differs')
    natural, previous = json.loads(snapshots[V1])['natural'], json.loads(snapshots[V2])
    require(len(previous['remaining_ids']) == 3101 and len(previous['remaining_group_ids']) == 2392, 'v2 reserve size differs')
    selected = select_ids(metadata,natural,previous)
    for name,digest in hashes.items():
        require(sha((ROOT/name).read_bytes()) == digest, 'partition input changed during selection')
    require(Path(__file__).read_bytes() == source, 'partition builder changed')
    report = {'schema':'finite-final-choice-fresh-evaluation-v3',
              'created_at':datetime.now(timezone.utc).isoformat(),
              'salt':SALT,'target_rows':TARGET,**selected,
              'rows':len(selected['eval_ids']),'groups':len(selected['eval_group_ids']),
              'source_hashes':dict(sorted(hashes.items())),
              'selector_plan_sha256':PINS[PLAN],'builder_sha256':sha(source),
              'warm_lineage_sha256':PINS[LINEAGE],
              'selection_uses_target_text':False,'model_inference_run':False,
              'training_run':False,'old_splits_modified':False,
              'upstream_pretraining_overlap':'unknown'}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:
        stream.write(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'rows':report['rows'],'groups':report['groups'],
                      'remaining_rows':len(selected['remaining_ids']),'manifest_sha256':sha(output.read_bytes())}))

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=OUTPUT)
    run(parser.parse_args().output)
