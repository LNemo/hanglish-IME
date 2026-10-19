"""Join original TRAIN labels to actual full pools and freeze retention families."""
import argparse
from collections import Counter,defaultdict,deque
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hanglish.models.sentence_ranker import SentenceRanker
from scripts.evaluate_ensemble_regression import file_sha256,rss_bytes,verify_pins

DATA=ROOT/'data/natural_context/current-context-v1'
DATA_SHA='b280bd12081c90fdd6bf53a3356b0a5f0ff0ccf83b905ca26b027c9d09027989'
PARENT=ROOT/'checkpoints/sentence_ranker'


def digest_key(value):
    return hashlib.sha256(('sentence-hard-retention-v1\0'+value).encode()).hexdigest()


def retention_split(group_id):
    return 'hold' if int(digest_key(group_id),16)%5==0 else 'fit'


def balanced_select(rows,limit):
    queues=defaultdict(list)
    for row in rows:queues[row['group_id'].split(':')[0]].append(row)
    queues={key:deque(sorted(values,key=lambda r:(digest_key(r['id']),r['id']))) for key,values in queues.items()}
    selected=[]
    while len(selected)<limit:
        changed=False
        for key in sorted(queues):
            if queues[key] and len(selected)<limit:
                selected.append(queues[key].popleft());changed=True
        if not changed:raise ValueError('not enough retention rows')
    return selected


def make_example(row,limit=8):
    candidates=row['candidates'];scores=row['teacher_scores']
    if not row['sentence_eligible']:return None
    if (not candidates or len(set(candidates))!=len(candidates) or len(scores)!=len(candidates)
            or any(not math.isfinite(v) for v in scores)):
        raise ValueError('malformed sentence pool')
    positives=[i for i,c in enumerate(candidates) if c in row['expected']]
    if not positives or len(positives)==len(candidates):return None
    if len(positives)>limit:raise ValueError('positive set exceeds training cap')
    order=sorted(range(len(candidates)),key=lambda i:(-scores[i],candidates[i]))
    selected=positives+[i for i in order if i not in positives][:limit-len(positives)]
    correct=order[0] in positives
    return dict(id=row['id'],group_id=row['group_id'],category=row['category'],context=row['context'],
                candidates=[candidates[i] for i in selected],positives=list(range(len(positives))),
                teacher_scores=[scores[i] for i in selected],teacher_correct=correct,
                loss_weight=1. if correct else 4.,distill=correct,
                expected=row['expected'],source_pool_size=len(candidates))


def aligned_row(capture,target):
    if any(capture[k]!=target[k] for k in ('id','input','context')):
        raise ValueError('natural source identity differs')
    expected_split='train' if capture['suite']=='natural_train' else 'dev'
    if target['split']!=expected_split:raise ValueError('natural source split differs')
    if len(capture['calls'])>1:raise ValueError('multiple sentence-score calls')
    scored=bool(capture['calls'])
    call=capture['calls'][0] if scored else None
    if call and any(c!=capture['context'] for c in call['contexts']):
        raise ValueError('sentence context differs')
    return dict(id=target['id'],group_id=target['group_id'],input=target['input'],context=target['context'],
                split=expected_split,category='natural',expected=target['expected'],sentence_eligible=scored,
                candidates=call['candidates'] if scored else [c['text'] for c in capture['response']['candidates']],
                teacher_scores=call['scores'] if scored else [],fallback_result=capture['response']['result'])


def run(capture_dir,capture_sha,output):
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    try:
        report_path=capture_dir/'report.json'
        if file_sha256(report_path)!=capture_sha:raise ValueError('capture report changed')
        report=json.loads(report_path.read_text())
        if not report['completed'] or report['counts']['rows']!=640:raise ValueError('capture incomplete')
        if file_sha256(capture_dir/'capture.jsonl')!=report['capture_sha256']:raise ValueError('capture changed')
        if file_sha256(DATA/'manifest.json')!=DATA_SHA:raise ValueError('retention provenance changed')
        source_manifest=json.loads((DATA/'manifest.json').read_text())
        pins=dict(report['pins'])
        for path in (report_path,capture_dir/'capture.jsonl',DATA/'manifest.json',Path(__file__)):
            pins[str(path.relative_to(ROOT))]=file_sha256(path)
        targets={};retention=[]
        for role in ('natural_train','natural_dev','retention_train'):
            path=DATA/(role+'.jsonl')
            if file_sha256(path)!=source_manifest['outputs'][role]['sha256']:
                raise ValueError('original source rows changed')
            pins[str(path.relative_to(ROOT))]=file_sha256(path)
            for row in map(json.loads,path.read_text().splitlines()):
                if role=='retention_train':
                    if row['split']!='train':raise ValueError('non-TRAIN retention source')
                    if row['sentence_eligible'] and len(row['candidates'])>1 and row['positive_indices']:
                        retention.append({key:row[key] for key in ('id','group_id','input','context','expected','candidates','teacher_scores','sentence_eligible')}|{'category':row['group_id'].split(':')[0]})
                else:targets[row['id']]={key:row[key] for key in ('id','group_id','input','context','expected','split')}
        captures=list(map(json.loads,(capture_dir/'capture.jsonl').read_text().splitlines()))
        natural={'fit':[],'hold':[]}
        for capture in captures:
            role='fit' if capture['suite']=='natural_train' else 'hold'
            natural[role].append(aligned_row(capture,targets[capture['id']]))
        selected={role:balanced_select([r for r in retention if retention_split(r['group_id'])==role],limit)
                  for role,limit in [('fit',512),('hold',128)]}
        if {r['group_id'] for r in natural['fit']+selected['fit']}&{r['group_id'] for r in natural['hold']+selected['hold']}:
            raise ValueError('training/held source groups overlap')
        import torch
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        ranker=SentenceRanker.load(PARENT)
        for path in PARENT.iterdir():
            if path.is_file():pins[str(path.relative_to(ROOT))]=file_sha256(path)
        verify_pins(pins)
        maximum_delta=0.
        for role,rows in selected.items():
            for index,row in enumerate(rows):
                scores=ranker.score_pairs([row['context']]*len(row['candidates']),row['candidates'])
                delta=max(abs(a-b) for a,b in zip(scores,row['teacher_scores']))
                maximum_delta=max(maximum_delta,delta)
                if delta>1e-4:raise ValueError('retention teacher no longer matches current model')
                row['teacher_scores']=scores
                row['fallback_result']=row['candidates'][min(range(len(scores)),key=lambda i:(-scores[i],row['candidates'][i]))]
                if time.monotonic()-started>600 or rss_bytes()>3*1024**3:
                    raise RuntimeError('preparation exceeds600seconds/3GiB')
            print(json.dumps(dict(retention_role=role,rows=len(rows),elapsed_s=time.monotonic()-started)),flush=True)
        fit=[example for row in natural['fit']+selected['fit'] if (example:=make_example(row)) is not None]
        pools={'natural_hold':natural['hold'],'retention_hold':selected['hold']}
        outputs={}
        for name,rows in [('fit',fit),*pools.items(),('natural_fit_pools',natural['fit'])]:
            path=output/(name+'.jsonl')
            path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
            outputs[name]=dict(rows=len(rows),sha256=file_sha256(path))
        verify_pins(pins)
        manifest=dict(schema='sentence-hard-negative-data-v1',completed=True,pins=pins,outputs=outputs,
            production_changed=False,test_labels_used_for_training=False,full_pool_positive_selection=True,
            natural_fit_all_rows=len(natural['fit']),natural_fit_trainable=sum(r['category']=='natural' for r in fit),
            hard_counts=dict(Counter(r['category'] for r in fit if not r['teacher_correct'])),
            category_counts=dict(Counter(r['category'] for r in fit)),retention_teacher_max_delta=maximum_delta,
            elapsed_s=time.monotonic()-started,peak_rss_bytes=rss_bytes())
        (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({k:manifest[k] for k in ('outputs','hard_counts','category_counts','elapsed_s','peak_rss_bytes')},ensure_ascii=False),flush=True)
    except BaseException as exc:
        (output/'failure.json').write_text(json.dumps(dict(completed=False,error=repr(exc))))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--capture-dir',type=Path,required=True);parser.add_argument('--capture-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.capture_dir.resolve(),args.capture_sha256,args.output.resolve())
