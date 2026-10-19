"""Capture frozen fresh parent cases, then evaluate the frozen finite v3 selector."""
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch

from hanglish.core.neural_converter import NeuralConverter, file_sha256
from hanglish.core.phonology import romanize_word
from hanglish.models.sentence_ranker import SentenceRanker
from scripts.final_choice_capture import capture_case
from scripts.finite_final_choice_evaluation import evaluate_case, summarize, evaluation_gate
from scripts.evaluate_final_choice_holdout import load_evaluation_capture, _validate_row
from scripts.prepare_final_choice_data import read_runtime_evidence
from scripts.partition_finite_selector_holdout import select_ids, PINS, RAW, V1, V2
from scripts.train_finite_final_choice_selector import (
    validate_evaluation_partition, EVALUATION_PARTITION_SHA256, PLAN_SHA256,
    RUNTIME_VERSION, rss_bytes, verify_hashes,
)
from scripts.load_finite_selector_artifact import load_selector

OUTPUT = ROOT/'.omx/artifacts/finite-final-choice-evaluation-v3'
RUNTIME_PATH = ROOT/'.omx/artifacts/numeric-structure-integration-v1/report.json'
RUNTIME_SHA = '77c04334af817e15bc4c5ccc9d27b1d6ab9f9f6a759b4207e752762ae49d3aab'
PARTITION = ROOT/'.omx/artifacts/finite-final-choice-partition-v3/manifest.json'
PRESERVATION = ROOT/'data/natural_context/final-choice-v1/challenge_preservation'
PRESERVATION_SHA = 'c4f587d42ac0838ce7eb2f31881bc182d8f1e0615bab47965f35da032cbaa133'
FROZEN = {
 'scripts/train_finite_final_choice_selector.py':'c6bf9046d70b4b6f488bbb64a28e0f2409a0e25549f2d78342366e4a5132b591',
 'scripts/finite_final_choice_policy.py':'80b1baff0c65981809a03ee3158ec6402d1a627803a95e70e390a25f07a17b05',
 'scripts/finite_final_choice_evaluation.py':'6ad0fb19f3a334f69a4ffc80bb7b13dbe67f12165b039b1ef44a5b54734ba764',
 'scripts/partition_finite_selector_holdout.py':'7c1851c7dcf9f65783f80e5925075b078ba45d50d6b59d81fd31f44772bced2b',
 'scripts/final_choice_capture.py':'80bad648b1c9d46a9483fafabd61f3b82736853735189697b3d12c6c500926fa',
 'scripts/evaluate_final_choice_holdout.py':'77b6743184cdc7a4ea23cee4a18e1c86f6899267954d9d34892a502163d13a33',
}

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def path_key(path):
    return str(path.resolve().relative_to(ROOT))

def merge(target, values):
    for name,value in values.items():
        if name in target and target[name] != value:
            raise ValueError(f'finite evaluation source conflict: {name}')
        target[name]=value

def require_resources(started):
    if rss_bytes()>3*1024**3 or time.monotonic()-started>1800:
        raise RuntimeError('finite evaluation exceeded3GiB/1800second resource limit')

def read_sources(partition):
    snapshots={name:(ROOT/name).read_bytes() for name in (RAW,V1,V2)}
    if any(digest(raw)!=PINS[name] for name,raw in snapshots.items()):
        raise ValueError('fresh evaluation source bytes changed')
    rows=[json.loads(line) for line in snapshots[RAW].splitlines() if line]
    metadata=[{'id':r['id'],'group_id':r['group_id']} for r in rows]
    selected=select_ids(metadata,json.loads(snapshots[V1])['natural'],json.loads(snapshots[V2]))
    if any(partition[k]!=v for k,v in selected.items()):
        raise ValueError('fresh partition cannot be reproduced')
    by_id={r['id']:r for r in rows}
    if len(by_id)!=len(rows):
        raise ValueError('duplicate fresh source identity')
    sources=[]
    for identity in partition['eval_ids']:
        row=by_id[identity]
        if (row.get('split')!='train' or not isinstance(row.get('text'),str) or not row['text']
                or not isinstance(row.get('left_context'),str)):
            raise ValueError('invalid fresh source row')
        sources.append({'id':identity,'group_id':row['group_id'],'origin':'natural_eval_v3',
                        'input':romanize_word(row['text']),'context':row['left_context'],
                        'expected':[row['text']]})
    return sources,{name:digest(raw) for name,raw in snapshots.items()}

def validate_fresh_row(row, selected, runtime):
    if row.get('origin')!='natural_eval_v3' or selected.get('origin')!='natural_eval_v3':
        raise ValueError('fresh row origin differs')
    _validate_row(dict(row,origin='natural'),dict(selected,origin='natural'),
                  'natural_holdout',RUNTIME_VERSION,runtime['accepted_weight_hashes'])

def load_fresh_parent_capture(directory,sources,manifest_sha,runtime,memory_rows):
    manifest_path=directory/'capture-manifest.json'
    manifest_raw=manifest_path.read_bytes()
    if digest(manifest_raw)!=manifest_sha:
        raise ValueError('fresh capture manifest changed')
    manifest=json.loads(manifest_raw)
    cases_raw=(directory/'parent-capture.jsonl').read_bytes()
    selection_raw=(directory/'selection.jsonl').read_bytes()
    if (digest(cases_raw)!=manifest['cases_sha256']
            or digest(selection_raw)!=manifest['selection_sha256']):
        raise ValueError('fresh capture or selection bytes differ from manifest')
    rows=[json.loads(line) for line in cases_raw.splitlines() if line]
    selected=[json.loads(line) for line in selection_raw.splitlines() if line]
    if (manifest['rows']!=len(sources) or len(rows)!=len(sources)
            or selected!=sources or len({r['id'] for r in rows})!=len(rows)):
        raise ValueError('fresh capture selection or row identity differs')
    for row,source in zip(rows,selected):
        validate_fresh_row(row,source,runtime)
    if rows!=memory_rows:
        raise ValueError('fresh capture disk rows differ from observed parent rows')
    return rows,{path_key(directory/'parent-capture.jsonl'):manifest['cases_sha256'],
                 path_key(directory/'selection.jsonl'):manifest['selection_sha256'],
                 path_key(manifest_path):manifest_sha}

def run(args):
    started=time.monotonic()
    if args.output.exists():
        raise FileExistsError(args.output)
    source=Path(__file__).read_bytes()
    verify_hashes(FROZEN)
    runtime_raw,runtime=read_runtime_evidence(RUNTIME_PATH,RUNTIME_SHA)
    partition_raw,partition=validate_evaluation_partition(PARTITION,EVALUATION_PARTITION_SHA256,[])
    # Freeze validation before target-bearing source reads or actual model loads.
    selector=load_selector(args.selector_report,args.selector_report_sha256,
                           args.selector_checkpoint_sha256,runtime)
    if (not isinstance(args.selector_metadata_sha256,str)
            or len(args.selector_metadata_sha256)!=64
            or set(args.selector_metadata_sha256)-set('0123456789abcdef')
            or digest(selector['metadata_raw'])!=args.selector_metadata_sha256):
        raise ValueError('explicit frozen selector metadata hash differs')
    sources,source_hashes=read_sources(partition)
    if len(sources)!=1024 or len({r['group_id'] for r in sources})!=785:
        raise ValueError('fresh evaluation source count differs')
    preservation=load_evaluation_capture(PRESERVATION,PRESERVATION_SHA,'challenge_preservation')
    if (preservation['manifest']['runtime_version']!=RUNTIME_VERSION
            or preservation['manifest']['model_artifact_hashes']!=runtime['accepted_weight_hashes']):
        raise ValueError('preservation parent differs')
    hashes=dict(selector['report']['source_hashes'])
    merge(hashes,{path_key(selector['checkpoint']/'metadata.json'):args.selector_metadata_sha256})
    for more in (FROZEN,source_hashes,partition['source_hashes'],
                 runtime['integrated_runtime_source_hashes'],runtime['accepted_weight_hashes'],
                 preservation['manifest']['source_hashes']):
        merge(hashes,more)
    paths=[Path(__file__),ROOT/'scripts/load_finite_selector_artifact.py',args.selector_report,
           args.selector_report.parent/'schedule.jsonl',selector['checkpoint']/'metadata.json',
           PARTITION,RUNTIME_PATH]
    paths.extend(selector['checkpoint']/name for name in selector['metadata']['artifact_hashes'])
    paths.extend(PRESERVATION/name for name in ('manifest.json','selection.jsonl','cases.jsonl'))
    merge(hashes,{path_key(path):file_sha256(path) for path in paths})
    verify_hashes(hashes)
    require_resources(started)
    args.output.mkdir(parents=True)
    selection_raw=b''.join((json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n').encode() for r in sources)
    (args.output/'selection.jsonl').write_bytes(selection_raw)
    torch.set_num_threads(2)
    os.environ['OMP_NUM_THREADS']='2'
    converter=NeuralConverter(ranker_kind='hybrid',external_lexicon=True)
    converter.load_models()
    if converter.version!=RUNTIME_VERSION or converter.model_artifacts!=runtime['accepted_weight_hashes']:
        raise ValueError('loaded production parent differs')
    captured=[]
    capture_path=args.output/'parent-capture.jsonl'
    with capture_path.open('x') as stream:
        for index,selected in enumerate(sources,1):
            row=capture_case(converter,selected,expected_runtime_version=RUNTIME_VERSION,
                             expected_model_artifacts=runtime['accepted_weight_hashes'])
            validate_fresh_row(row,selected,runtime)
            captured.append(row)
            stream.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')
            stream.flush()
            require_resources(started)
            if index%32==0:print(json.dumps({'event':'finite_v3_parent_capture','rows':index}),flush=True)
    verify_hashes(hashes)
    if file_sha256(args.output/'selection.jsonl')!=digest(selection_raw):
        raise RuntimeError('fresh selection changed during capture')
    captured_raw=b''.join((json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n').encode() for r in captured)
    capture_hash=digest(captured_raw)
    if file_sha256(capture_path)!=capture_hash:
        raise RuntimeError('fresh capture disk bytes differ from observed parent rows')
    capture_report={'completed':True,'rows':1024,'origin':'natural_eval_v3',
                    'runtime_version':RUNTIME_VERSION,'partition_sha256':EVALUATION_PARTITION_SHA256,
                    'selector_report_sha256':args.selector_report_sha256,
                    'selector_checkpoint_sha256':args.selector_checkpoint_sha256,
                    'selector_metadata_sha256':args.selector_metadata_sha256,
                    'cases_sha256':capture_hash,'selection_sha256':digest(selection_raw),
                    'source_hashes':dict(hashes),'source_hashes_after':dict(hashes),
                    'new_selector_inference_run':False,'deployed':False}
    manifest_raw=(json.dumps(capture_report,indent=2)+'\n').encode()
    (args.output/'capture-manifest.json').write_bytes(manifest_raw)
    captured,capture_hashes=load_fresh_parent_capture(
        args.output,sources,digest(manifest_raw),runtime,captured)
    del converter
    gc.collect()
    merge(hashes,capture_hashes)
    verify_hashes(hashes)
    ranker=SentenceRanker.load(selector['checkpoint'])
    require_resources(started)
    outcomes=[]
    cases_path=args.output/'cases.jsonl'
    with cases_path.open('x') as stream:
        for role,rows in (('natural_evaluation_v3',captured),('challenge_preservation',preservation['rows'])):
            for row in rows:
                outcome=evaluate_case(ranker,row,role)
                outcomes.append(outcome)
                stream.write(json.dumps(outcome,ensure_ascii=False,sort_keys=True)+'\n')
                stream.flush()
                require_resources(started)
                if len(outcomes)%32==0:print(json.dumps({'event':'finite_v3_selection','rows':len(outcomes)}),flush=True)
    if len(outcomes)!=1400:
        raise RuntimeError('finite evaluation cases incomplete')
    suites={role:summarize([r for r in outcomes if r['suite']==role])
            for role in ('natural_evaluation_v3','challenge_preservation')}
    gate=evaluation_gate(suites)
    verify_hashes(hashes)
    if Path(__file__).read_bytes()!=source:
        raise RuntimeError('evaluation source changed')
    (args.output/'source.py').write_bytes(source)
    require_resources(started)
    report={'schema':'finite-final-choice-fresh-evaluation-result-v3',
            'created_at':datetime.now(timezone.utc).isoformat(),'rows':1400,'suites':suites,'gate':gate,
            'passed':gate['passed'],'eligible_for_development':gate['passed'],
            'eligible_for_deployment':False,'deployed':False,'training_run':False,
            'development_data_read':False,'final_data_read':False,
            'selector_report_sha256':args.selector_report_sha256,
            'selector_checkpoint_sha256':args.selector_checkpoint_sha256,
            'selector_metadata_sha256':args.selector_metadata_sha256,
            'plan_sha256':PLAN_SHA256,'evaluation_partition_sha256':EVALUATION_PARTITION_SHA256,
            'runtime_version':RUNTIME_VERSION,'model_artifact_hashes':runtime['accepted_weight_hashes'],
            'source_hashes':hashes,'source_hashes_after':dict(hashes),
            'cases_sha256':file_sha256(cases_path),'parent_capture_sha256':capture_hash,
            'candidate_objects_only_reordered':True,'raw_bypass_preserved':True,
            'preservation_is_new_independent_evaluation':False,
            'elapsed_seconds':time.monotonic()-started,'peak_rss_bytes':rss_bytes()}
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'gate':gate,'suites':suites},ensure_ascii=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--selector-report',type=Path,required=True)
    parser.add_argument('--selector-report-sha256',required=True)
    parser.add_argument('--selector-checkpoint-sha256',required=True)
    parser.add_argument('--selector-metadata-sha256',required=True)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    run(parser.parse_args())
