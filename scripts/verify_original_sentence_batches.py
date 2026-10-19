#!/usr/bin/env python3
"""Verify closed original sentence-ranker call-boundary evidence without inference."""
from __future__ import annotations
import argparse, hashlib, json, math, os, re
from datetime import datetime
from itertools import zip_longest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PREPARED=ROOT/'.omx/artifacts/current-phonetic-train-inputs-v1/requests.json'
SOURCE_DIRS=tuple(ROOT/f'.omx/artifacts/current-phonetic-train-shard-{i}-v1' for i in range(6))
SOURCE_HASHES=('0053b278869aec714bd48a2d5e5adc504292c75a78cfd5fa2ba8205ef2e24d4d','909accff83c0cc783049a5c75ed40e64e895e6b159082f60b7bb1264701adb29','de7d0da7228c3ad44f28d80bb69e64bfe0f73d0c9b55d9fa749ac4f498580882','317346be3b2efc2c04a2e849d49adf752ea6dd775ef751c41cd79a7243a54019','9ced02dfa254bd29f3094be38a96307dc79a7285edd18bb7b56a838064f9aaaf','8f1a260362660346fc8b809b0e2369db6b7ec3b34c05235b13ddc9e66c9771ac')
PREPARED_SHA='97b416a805b9da8dc15d4ff9178b3a1db2a98f57f961a0e369c51bc771e1762d'
LIMITS={'whole_seconds':300.,'update_seconds':30.,'rss_bytes':2*1024**3,'swap_growth_bytes':512*1024**2}

def need(v,m):
    if not v: raise ValueError(m)
def finite(v): return type(v) in (int,float) and math.isfinite(v)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def read(path):
    v=json.loads(Path(path).read_text());need(isinstance(v,dict),'expected object');return v
def jsonl(path):
    with Path(path).open() as f:
        for line in f:
            need(line.strip(),'blank JSONL');v=json.loads(line);need(isinstance(v,dict),'row object');yield v
def canonical(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def validate_paths(paths,request):
    need(isinstance(paths,list),'original paths missing')
    seen=set()
    for path in paths:
        need(isinstance(path,list) and len(path)==3 and isinstance(path[0],str) and path[0]
             and finite(path[1]) and isinstance(path[2],list),'original path shape differs')
        need(path[0] not in seen,'duplicate original text');seen.add(path[0])
        source_parts=[];text_parts=[];cursor=0
        for segment in path[2]:
            need(isinstance(segment,dict) and set(segment)=={
                 'input_start','input_end','source','text','kind'},'segment schema differs')
            start,end=segment['input_start'],segment['input_end']
            need(type(start) is int and type(end) is int and start==cursor and start<end
                 and end<=len(request['input'])
                 and isinstance(segment['source'],str) and isinstance(segment['text'],str)
                 and isinstance(segment['kind'],str) and segment['kind'],'segment contract differs')
            need(segment['source']==request['input'][start:end],'segment source slice differs')
            source_parts.append(segment['source']);text_parts.append(segment['text']);cursor=end
        need(cursor==len(request['input']) and ''.join(source_parts)==request['input']
             and ''.join(text_parts)==path[0],
             'segment reconstruction differs')

def validate_trace(original,record,shard,index,capture_sha):
    need(record.get('id')==original.get('id') and record.get('component_id')==original.get('component_id')
         and record.get('request')==original.get('request') and record.get('source_shard')==shard
         and record.get('source_row_index')==index and record.get('original_capture_sha256')==capture_sha,
         'trace source identity differs')
    paths=original.get('base_paths');validate_paths(paths,original['request'])
    texts=[p[0] for p in paths];scores=[p[1] for p in paths]
    need(record.get('original_order')==texts and record.get('original_scores')==scores
         and record.get('original_paths_sha256')==canonical(paths),'original pool/order differs')
    calls=record.get('calls');need(isinstance(calls,list),'calls missing')
    observed=[]; mapped={}
    for call in calls:
        need(isinstance(call,dict) and set(call)=={'contexts','texts','scores','token_lengths','batch_size'},
             'call schema differs')
        contexts,items,values,lengths=call['contexts'],call['texts'],call['scores'],call['token_lengths']
        need(isinstance(contexts,list) and isinstance(items,list) and isinstance(values,list)
             and isinstance(lengths,list) and items and len(contexts)==len(items)==len(values)==len(lengths)
             and all(c=='' for c in contexts) and all(isinstance(t,str) and t for t in items)
             and type(call['batch_size']) is int and call['batch_size']==8
             and all(finite(v) for v in values) and all(type(v) is int and v>0 for v in lengths),
             'call shape/context/numeric contract differs')
        for text,value in zip(items,values):
            need(text not in mapped,'duplicate call text');mapped[text]=float(value);observed.append(text)
    need(len(observed)==len(texts) and set(observed)==set(texts),'call coverage differs')
    delta=max((abs(mapped[t]-float(s)) for t,s in zip(texts,scores)),default=0.)
    need(record.get('tolerance')==1e-5 and finite(record.get('replay_max_absolute_difference'))
         and record['replay_max_absolute_difference']==delta and delta<=1e-5,
         'recorded runtime parity differs')
    return {'originals':len(texts),'calls':len(calls),'max_absolute_difference':delta}

def verify_resources(rows,operations):
    need(rows,'empty resources');starts=[];ends=[];current=None;previous=-1.;baseline=None;terminal=None
    for row in rows:
        need(finite(row.get('elapsed_s')) and previous<=row['elapsed_s']<=300
             and type(row.get('rss_bytes')) is int and 0<=row['rss_bytes']<=LIMITS['rss_bytes']
             and type(row.get('swap_used_bytes')) is int and type(row.get('swap_growth_bytes')) is int
             and row['swap_growth_bytes']<=LIMITS['swap_growth_bytes'],'resource row differs')
        datetime.fromisoformat(row['at']);previous=row['elapsed_s'];op=row.get('operation');elapsed=row.get('operation_elapsed_s')
        is_update=bool(op and op.startswith('replay:'))
        need((finite(elapsed) and elapsed>=0) if is_update else elapsed is None,
             'resource attribution differs')
        if is_update: need(elapsed<=30,'replay exceeded 30 seconds')
        observed=row['swap_used_bytes']-row['swap_growth_bytes'];baseline=observed if baseline is None else baseline
        need(observed==baseline and observed>=0,'swap baseline differs')
        stage=row['stage']
        if stage=='operation_start': need(current is None,'nested op');current=op;starts.append(op)
        elif stage=='operation_end': need(current==op,'end differs');current=None;ends.append(op)
        elif stage=='watchdog': need(op==current,'watchdog differs')
        elif stage=='after_final_rehash': need(current is None and starts==operations,'terminal sample differs');terminal=row
        else: raise ValueError('unexpected resource stage')
    need(starts==operations==ends and terminal,'resource sequence differs')
    return {'rows':len(rows),'operations':len(starts),'after_final_rehash':terminal,
            'peak_rss_bytes':max(r['rss_bytes'] for r in rows),'peak_swap_growth_bytes':max(r['swap_growth_bytes'] for r in rows)}

def selected_sources():
    need(sha(PREPARED)==PREPARED_SHA,'prepared requests differ');prepared=read(PREPARED)
    rows={row['id']:row for row in prepared['rows']}
    ids=[g['probe_id'] for g in prepared['groups']];found={}
    for shard,directory in enumerate(SOURCE_DIRS):
        need(sha(directory/'capture.jsonl')==SOURCE_HASHES[shard],'source capture differs')
        for index,row in enumerate(jsonl(directory/'capture.jsonl')):
            if row['id'] in ids:
                expected=rows[row['id']]
                need(row.get('component_id')==expected.get('component_id')
                     and row.get('request')=={'input':expected.get('input'),'context':''},
                     'source request differs from prepared')
                need(row['id'] not in found,'duplicate probe source')
                found[row['id']]=(row,shard,index)
    need(set(found)==set(ids),'probe sources missing');return ids,found

def verify(output,runner_sha256):
    need(isinstance(runner_sha256,str) and re.fullmatch(r'[0-9a-f]{64}',runner_sha256),
         'runner hash differs')
    output=Path(output);need(output.is_absolute(),'output must be absolute');output=output.resolve()
    proof_path=output/'independent-verification.json';need(output.is_dir() and not proof_path.exists(),'proof state invalid')
    need(not (output/'failure.json').exists() and not (output/'resource-watchdog-failure.json').exists(),
         'failure artifact exists')
    report,completion,pins=read(output/'report.json'),read(output/'completion.json'),read(output/'pins.json')
    launch_path=output.with_suffix('.launch.json');launch=read(launch_path);command=launch.get('command')
    need(isinstance(command,list) and len(command)==3
         and Path(command[0]).resolve()==(ROOT/'.venv/bin/python').resolve()
         and Path(command[1]).is_absolute()
         and command[2]==runner_sha256 and sha(command[1])==runner_sha256
         and launch.get('runner_sha256')==runner_sha256 and launch.get('output')==str(output),'launch differs')
    try: os.kill(launch['pid'],0)
    except ProcessLookupError: pass
    else: raise ValueError('runner PID present')
    capture_sha=sha(output/'capture.jsonl')
    need(report.get('schema')=='original-sentence-batches-probe-v1' and report.get('completed') is True
         and report.get('rows')==16 and report.get('runner_sha256')==runner_sha256
         and report.get('capture_sha256')==capture_sha and report.get('limits')==LIMITS
         and report.get('tolerance')==1e-5 and report.get('source_only') is True
         and report.get('labels_opened') is False and report.get('phonetic_inference') is True
         and report.get('training') is False and report.get('production_changed') is False
         and report.get('pins')==pins,'report differs')
    need(pins.get(str(PREPARED.resolve()))==PREPARED_SHA,'prepared pin differs')
    for directory,digest in zip(SOURCE_DIRS,SOURCE_HASHES):
        need(pins.get(str((directory/'capture.jsonl').resolve()))==digest,'source capture pin differs')
    need(pins.get(str(Path(command[1]).resolve()))==runner_sha256,'runner pin differs')
    for path,digest in pins.items(): need(Path(path).is_file() and sha(path)==digest,'pin differs')
    ids,sources=selected_sources();need(report.get('ids')==ids,'ID order differs')
    summaries=[]
    for identifier,row in zip_longest(ids,jsonl(output/'capture.jsonl')):
        need(identifier is not None and row is not None and row.get('id')==identifier,'capture order differs')
        original,shard,index=sources[identifier]
        summaries.append(validate_trace(original,row,shard,index,SOURCE_HASHES[shard]))
    maximum=max(item['max_absolute_difference'] for item in summaries)
    need(report.get('max_original_score_difference')==maximum and maximum<=1e-5,'report parity differs')
    operations=['authenticate_sources','load_original_engine']+['replay:'+i for i in ids]+['final_rehash']
    resources=verify_resources(list(jsonl(output/'resources.jsonl')),operations)
    need(report.get('terminal')==resources['after_final_rehash']
         and completion=={'completed':True,'report_sha256':sha(output/'report.json')},'completion differs')
    proof={'schema':'original-sentence-batches-independent-verification-v1','verified':True,'rows':16,
           'report_sha256':sha(output/'report.json'),'capture_sha256':capture_sha,
           'launch_sha256':sha(launch_path),'runner_sha256':runner_sha256,'verifier_sha256':sha(__file__),
           'resources':resources,'max_original_score_difference':maximum,
           'scope':'Source IDs/order, original pools, captured call grouping, empty contexts, finite recorded runtime scores, recorded parity, pins, launch and resources.',
           'limitations':'Token lengths are validated as recorded but not retokenized; neural logits are not independently reexecuted.',
           'labels_opened':False,'model_inference_replayed':False}
    with proof_path.open('x') as f: json.dump(proof,f,ensure_ascii=False,sort_keys=True,indent=2);f.write('\n')
    return proof

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--runner-sha256',required=True)
    r=p.parse_args();print(json.dumps(verify(r.output,r.runner_sha256)))
