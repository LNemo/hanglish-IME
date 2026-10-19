"""Score one closed FIT recombination shard after a verified fixed-model probe."""
import argparse
import hashlib
from itertools import zip_longest
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import current_fit_recombination_census as census
from scripts import evaluate_current_fit_recombination_coverage as intake_auth
from scripts.source_span_resources import AtomicResourceWatchdog,system_swap_bytes

ART=ROOT/'.omx/artifacts'
INTAKE=ART/'current-fit-recombination-coverage-intake-v1.json'
PROBE=ART/'current-fit-recombination-scoring-probe-v3'
CORE=ROOT/'scripts/current_recombination_scoring.py'
MODEL=ROOT/'checkpoints/sentence_ranker'
LIMITS=dict(whole_seconds=900.,update_seconds=30.,rss_bytes=2*1024**3,swap_growth_bytes=512*1024**2)


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())


def run(shard,runner_sha,core_sha,probe_report_sha,probe_proof_sha):
    require(type(shard)is int and 0<=shard<6,'shard differs')
    require(sha(__file__)==runner_sha and sha(CORE)==core_sha,'scorer identity differs')
    output=ART/f'current-fit-recombination-scoring-shard-{shard}-v1'
    output.mkdir(exist_ok=False);started=time.monotonic()
    try:
        with AtomicResourceWatchdog(started,system_swap_bytes(),output/'resources.jsonl',limits=LIMITS,interval_s=.5) as guard:
            with guard.operation('authenticate_sources',update=False):
                require(sha(PROBE/'report.json')==probe_report_sha
                        and sha(PROBE/'independent-verification.json')==probe_proof_sha,'probe identity differs')
                probe,proof=read(PROBE/'report.json'),read(PROBE/'independent-verification.json')
                require(probe['completed'] is True and probe['qualified'] is True
                        and proof['verified'] is True and proof['qualified'] is True
                        and proof['report_sha256']==probe_report_sha
                        and probe['core_sha256']==core_sha,'scoring probe did not qualify')
                require(probe['limits']==LIMITS and all(t<=900 for t in probe['projected_shard_seconds']),
                        'probe schedule is not qualified')
                pins=dict(probe['pins'])
                for path,digest in pins.items():require(sha(path)==digest,'probe input changed: '+path)
                auth=intake_auth.authenticate_manifest(INTAKE)
                prepared,row_map,groups,shards,original_pins=census.authenticate_inputs()
                for inventory in (auth['pins'],original_pins):
                    for path,digest in inventory.items():
                        require(path not in pins or pins[path]==digest,'source hashes disagree')
                        pins[path]=digest
                for path in (Path(__file__),PROBE/'report.json',PROBE/'independent-verification.json',
                             PROBE/'completion.json',PROBE/'capture.jsonl',PROBE/'capture.complete.json',
                             PROBE/'resources.jsonl'):
                    pins[str(path.resolve())]=sha(path)
                ids=shards[shard]
                directory=auth['directories'][shard]
                write(output/'pins.json',pins)
            with guard.operation('load_frozen_sentence_ranker',update=False):
                import torch
                torch.set_num_threads(2);torch.set_num_interop_threads(1)
                torch.use_deterministic_algorithms(True)
                from hanglish.models.sentence_ranker import SentenceRanker
                from scripts.current_recombination_scoring import score_row
                ranker=SentenceRanker.load(MODEL);ranker.model.to('cpu')
                require(ranker.max_length==96 and ranker.batch_size==8,'inference bounds differ')
            counts={};timings={}
            with (output/'capture.jsonl').open('x') as stream:
                original_rows=census.iter_jsonl(census.SHARDS[shard]/'capture.jsonl')
                generated_rows=census.iter_jsonl(directory/'capture.jsonl')
                for position,(old,new) in enumerate(zip_longest(original_rows,generated_rows)):
                    require(old is not None and new is not None and position<len(ids)
                            and old['id']==new['id']==ids[position],'source order differs')
                    identifier=old['id'];tick=time.monotonic()
                    with guard.operation('score:'+identifier,update=True):
                        result=score_row(old,new,ranker,check_original_parity=False)
                        require(result['original_score_parity'] is None,'unperformed parity claimed')
                        record=dict(id=identifier,component_id=old['component_id'],request=old['request'],
                                    source_shard=shard,source_row_index=position,scoring=result)
                        stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
                        stream.flush();os.fsync(stream.fileno())
                        counts[identifier]=dict(original=len(result['original']),novel=len(result['novel']),
                            scored_novel=sum(r['status']=='scored' for r in result['novel']),
                            unsupported_novel=sum(r['status']=='unsupported' for r in result['novel']))
                    timings[identifier]=time.monotonic()-tick
                    if (position+1)%25==0:print(json.dumps(dict(shard=shard,scored=position+1,total=len(ids))),flush=True)
            require(list(counts)==ids,'full shard row coverage differs')
            capture_sha=sha(output/'capture.jsonl')
            with guard.operation('write_closure',update=True):
                write(output/'capture.complete.json',dict(schema='current-fit-recombination-scoring-closure-v1',
                    mode='capture',shard=shard,completed=True,rows=len(ids),ids=ids,capture_sha256=capture_sha,
                    runner_sha256=runner_sha,core_sha256=core_sha,labels_opened=False,
                    model_inference=True,phonetic_model_inference=False))
            with guard.operation('write_report',update=True):
                write(output/'report.json',dict(schema='current-fit-recombination-scoring-shard-v1',
                    completed=True,mode='capture',shard=shard,rows=len(ids),ids=ids,row_counts=counts,
                    timing_seconds=timings,runner_sha256=runner_sha,core_sha256=core_sha,
                    capture_sha256=capture_sha,capture_output_bytes=(output/'capture.jsonl').stat().st_size,
                    closure_sha256=sha(output/'capture.complete.json'),pins=pins,limits=LIMITS,
                    probe_report_sha256=probe_report_sha,probe_proof_sha256=probe_proof_sha,
                    model_metadata=dict(max_length=96,batch_size=8,device='cpu',threads=2,interop_threads=1),
                    source_only=True,fit_labels_opened=False,dev_opened=False,test_opened=False,
                    model_inference=True,phonetic_model_inference=False,neural_training=False,production_changed=False))
            with guard.operation('final_rehash',update=False):
                require(all(sha(p)==h for p,h in pins.items()),'input changed during scoring')
            terminal=guard.sample_and_append('after_final_rehash')
            with guard.operation('write_completion',update=True):
                write(output/'completion.json',dict(completed=True,report_sha256=sha(output/'report.json'),
                                                   resource_after_final_rehash=terminal))
            guard.sample_and_append('after_completion')
        print(json.dumps(dict(completed=True,shard=shard,rows=len(ids))),flush=True)
    except BaseException as error:
        write(output/'failure.json',dict(completed=False,error=f'{type(error).__name__}: {error}'))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shard',type=int,required=True)
    for name in ('runner-sha256','core-sha256','probe-report-sha256','probe-proof-sha256'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    run(args.shard,args.runner_sha256,args.core_sha256,args.probe_report_sha256,args.probe_proof_sha256)
