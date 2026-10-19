"""Source-only mixed-span proposals added to the unchanged parent sentence pool."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_sentence_only_stages import source_inputs,baseline_evidence,compact_trace,parity_check
from scripts.trace_conversion_candidates import TraceConverter,trace_conversion,displayed_candidates,path_records
from scripts.evaluate_sentence_only import sha
from scripts.evaluate_ensemble_regression import rss_bytes

CORPUS=ROOT/'data/challenge/sentence-v3'
BASELINE=ROOT/'.omx/artifacts/sentence-only-dev-baseline-v1'


def merge_scored(parent,extras,scores):
    """Preserve each parent path and score; score only genuinely new paths."""
    if len(extras)!=len(scores):raise ValueError('extra/score count mismatch')
    if any(not isinstance(s,(int,float)) or not math.isfinite(s) for s in scores):
        raise ValueError('nonfinite extra score')
    unique={path[0]:path for path in parent}
    if len(unique)!=len(parent):raise ValueError('duplicate parent paths')
    for path,score in zip(extras,scores):
        if path[0] in unique:raise ValueError('extra duplicates a preserved path')
        unique[path[0]]=(path[0],float(score),path[2])
    return sorted(unique.values(),key=lambda p:(-p[1],p[0]))


def assert_segments(source,path):
    cursor=0;parts=[]
    for segment in path[2]:
        start,end=segment['input_start'],segment['input_end']
        if start!=cursor or end<=start or end>len(source) or source[start:end]!=segment['source']:
            raise ValueError('extra path does not cover input offsets exactly')
        if segment['kind']=='raw' and segment['text']!=segment['source']:
            raise ValueError('copied literal was modified')
        parts.append(segment['text']);cursor=end
    if cursor!=len(source) or ''.join(parts)!=path[0]:raise ValueError('extra output/coverage mismatch')


def summarize(captures,labels):
    if [r['id'] for r in captures]!=[r['id'] for r in labels]:raise ValueError('ID order mismatch')
    counts=Counter();outcomes=[];categories={}
    for record,row in zip(captures,labels):
        expected=row['expected']
        before=record['parent_result'] in expected
        after=record['result'] in expected
        parent_pool=[p['text'] for p in record['parent_pool']]
        active_pool=[p['text'] for p in record['scored_pool']]
        generated=parent_pool+[p['text'] for p in record['generated_extras']]
        before5=any(c['text'] in expected for c in record['parent_displayed'])
        after5=any(c['text'] in expected for c in record['displayed'])
        current=dict(rows=1,parent_top1=int(before),top1=int(after),parent_top5=int(before5),top5=int(after5),
                     corrections=int(after and not before),regressions=int(before and not after),
                     top5_losses=int(before5 and not after5),
                     parent_pool=int(any(c in expected for c in parent_pool)),
                     generated_pool=int(any(c in expected for c in generated)),
                     scored_pool=int(any(c in expected for c in active_pool)),
                     rows_with_extras=int(bool(record['generated_extras'])))
        counts.update(current);categories.setdefault(row['category'],Counter()).update(current)
        outcomes.append(dict(id=row['id'],input=row['input'],expected=expected,parent=record['parent_result'],
                             result=record['result'],correct=after,correction=after and not before,regression=before and not after))
    return dict(counts=dict(counts),categories={k:dict(v) for k,v in categories.items()},outcomes=outcomes,
                gate_passed=counts['top1']>counts['parent_top1'] and counts['regressions']==0 and counts['top5_losses']==0)


def run(output):
    import torch
    from hanglish.core.mixed_span_proposals import sentence_extras
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    inputs,identity=source_inputs(CORPUS/'dev.jsonl',CORPUS/'manifest.json')
    references,models=baseline_evidence(BASELINE,inputs,identity)
    files=[Path(__file__),ROOT/'hanglish/core/mixed_span_proposals.py',
           ROOT/'scripts/trace_conversion_candidates.py',ROOT/'hanglish/core/neural_converter.py',
           CORPUS/'dev.jsonl',CORPUS/'manifest.json']
    pins={str(p):sha(p) for p in files}
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();converter=TraceConverter(ranker_kind='hybrid');converter.load_models()
    if converter.model_artifacts!=models:raise ValueError('model identity differs')
    captures=[]
    with (output/'capture.jsonl').open('x') as stream:
        for item in inputs:
            start=time.monotonic()
            trace=trace_conversion(converter,item['input'],'')
            parity_check(compact_trace(item['id'],trace),references[item['id']])
            parent=[(p['text'],p['score'],p['segments']) for p in trace['final_pre_display_paths']]
            parent_snapshot=json.dumps(parent,ensure_ascii=False)
            extras=sentence_extras(item['input'],parent,converter.word_options,limit=96)
            if json.dumps(parent,ensure_ascii=False)!=parent_snapshot:raise ValueError('parent paths mutated')
            if len(extras)>96:raise ValueError('extra candidate budget exceeded')
            for path in extras:assert_segments(item['input'],path)
            if set(p[0] for p in parent)&set(p[0] for p in extras):raise ValueError('parent duplicated')
            sentence_ranked=bool(trace['sentence_score_pairs'])
            accepted=[p for p in extras if converter.sentence_ranker.can_score(p[0])] if sentence_ranked else []
            scores=converter.sentence_ranker.score_pairs(['']*len(accepted),[p[0] for p in accepted]) if accepted else []
            merged=merge_scored(parent,accepted,scores) if accepted else parent
            displayed=displayed_candidates(merged,item['input'])
            record=dict(id=item['id'],input=item['input'],context='',parent_result=trace['result'],
                        parent_displayed=trace['displayed_candidates'],parent_pool=path_records(parent),
                        generated_extras=path_records(extras),accepted_extras=path_records([(p[0],s,p[2]) for p,s in zip(accepted,scores)]),
                        scored_pool=path_records(merged),displayed=displayed,result=displayed[0]['text'],
                        parent_sentence_ranked=sentence_ranked,elapsed_ms=(time.monotonic()-start)*1000)
            if not extras and displayed!=trace['displayed_candidates']:
                raise ValueError('no-extra conversion changed')
            captures.append(record);stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush()
            if len(captures)%25==0:print(json.dumps(dict(captured=len(captures),total=len(inputs),elapsed_s=time.monotonic()-started)),flush=True)
            if time.monotonic()-started>1800 or rss_bytes()>3*1024**3:raise RuntimeError('mixed-span resource limit exceeded')
    labels=[json.loads(line) for line in (CORPUS/'dev.jsonl').read_text().splitlines() if line]
    metrics=summarize(captures,labels)
    for name,digest in {**pins,**models}.items():
        path=Path(name);path=path if path.is_absolute() else ROOT/path
        if sha(path)!=digest:raise ValueError('source/model changed during experiment')
    report=dict(completed=True,input_contract='roman_sentence_only',**identity,**metrics,
                source_hashes=pins,model_artifacts=models,capture_sha256=sha(output/'capture.jsonl'),
                elapsed_s=time.monotonic()-started,peak_rss_bytes=rss_bytes(),
                parent_pool_preserved=True,production_changed=False,training_run=False,test_accessed=False)
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('completed','counts','gate_passed','elapsed_s')},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
