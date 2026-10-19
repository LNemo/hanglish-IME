"""Evaluate complete Roman sentences with an enforced empty external context."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.evaluate_ensemble_regression import rss_bytes


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_inputs(path,manifest):
    identity=json.loads(manifest.read_text())
    if (identity.get('input_contract')!='roman_sentence_only'
            or identity.get('model_input_fields',['input'])!=['input']):
        raise ValueError('manifest must declare Roman-sentence-only model input')
    source_sha=sha(path)
    if identity.get('sha256',{}).get(path.name)!=source_sha:
        raise ValueError('sentence corpus hash differs from manifest')
    cases=[json.loads(line) for line in path.read_text().splitlines() if line]
    if not cases or identity.get('counts',{}).get(path.stem)!=len(cases):
        raise ValueError('sentence corpus count differs from manifest')
    inputs=[];seen=set()
    for case in cases:
        if (case.get('context','')!='' or case.get('left_context')
                or case.get('context_role','none')!='none'):
            raise ValueError('external Korean context is forbidden in sentence-only evaluation')
        text=case.get('input')
        if not isinstance(text,str) or len(text.split())<2 or re.search('[가-힣ᄀ-ᇿ㄰-㆏]',text):
            raise ValueError('input must be a nonempty Roman sentence without supplied Hangul')
        if case.get('split')!=path.stem or not isinstance(case.get('id'),str) or case['id'] in seen:
            raise ValueError('invalid sentence identity or split')
        answers=case.get('expected')
        if not isinstance(answers,list) or not answers or any(not isinstance(a,str) or len(a.split())<2 for a in answers):
            raise ValueError('complete sentence expected answers required')
        seen.add(case['id']);inputs.append(dict(id=case['id'],input=text))
    return inputs,dict(source_sha256=source_sha,manifest_sha256=sha(manifest),corpus_version=identity.get('corpus_version'))


def predict_sentence(converter,text):
    """Only the Roman input string crosses the predictor boundary."""
    result=converter.convert(text,context='',engine='neural',mode='batch',raw=False)
    if result.get('input')!=text or result.get('context')!='' or result.get('external_context_used',False):
        raise ValueError('converter violated sentence-only input contract')
    return result


def summarize(captures,cases):
    if [r['id'] for r in captures]!=[r['id'] for r in cases]:
        raise ValueError('sentence capture/label identities differ')
    totals=Counter();categories=defaultdict(Counter);outcomes=[]
    for capture,case in zip(captures,cases):
        texts=[c['text'] for c in capture.get('response',{}).get('candidates',[])][:5]
        top1=bool(texts) and texts[0] in case['expected']
        top5=any(text in case['expected'] for text in texts)
        counts=dict(rows=1,top1=int(top1),top5=int(top5),errors=int(capture.get('error') is not None))
        totals.update(counts);categories[case.get('category','uncategorized')].update(counts)
        outcomes.append(dict(id=case['id'],input=case['input'],expected=case['expected'],actual=texts,top1=top1,top5=top5,error=capture.get('error')))
    return dict(counts=dict(totals),categories={k:dict(v) for k,v in categories.items()},passed=bool(cases) and totals['top1']==len(cases) and totals['errors']==0,outcomes=outcomes)


def run(path,manifest,output,converter=None):
    inputs,identity=read_inputs(path,manifest)
    output.mkdir(parents=True,exist_ok=False)
    if converter is None:
        import torch
        from hanglish.core.neural_converter import NeuralConverter
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        converter=NeuralConverter();converter.load_models()
    model_pins=getattr(converter,'model_artifacts',{})
    def verify_model():
        for name,expected in model_pins.items():
            p=Path(name);p=p if p.is_absolute() else ROOT/p
            if sha(p)!=expected:raise ValueError('model artifact changed')
    verify_model()
    captures=[];started=time.monotonic()
    with (output/'capture.jsonl').open('x') as stream:
        for item in inputs:
            start=time.monotonic()
            record=dict(id=item['id'],request=dict(input=item['input'],context='',engine='neural',mode='batch',raw=False))
            try:record['response']=predict_sentence(converter,item['input'])
            except Exception as error:record['error']=repr(error)
            record['elapsed_ms']=(time.monotonic()-start)*1000
            captures.append(record);stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush()
            if time.monotonic()-started>1800 or rss_bytes()>3*1024**3:
                raise RuntimeError('sentence evaluation exceeds1800seconds/3GiB')
            if len(captures)%25==0:print(json.dumps(dict(captured=len(captures),total=len(inputs),elapsed_s=time.monotonic()-started)),flush=True)
    if sha(path)!=identity['source_sha256'] or sha(manifest)!=identity['manifest_sha256']:
        raise ValueError('sentence source changed during inference')
    verify_model()
    # Join whole-sentence answers only after all predictions have been captured.
    cases=[json.loads(line) for line in path.read_text().splitlines() if line]
    metrics=summarize(captures,cases)
    report=dict(completed=True,input_contract='roman_sentence_only',external_context_allowed=False,
                source=str(path),manifest=str(manifest),**identity,**metrics,
                capture_sha256=sha(output/'capture.jsonl'),model_artifacts=model_pins,
                evaluator_sha256=sha(Path(__file__)),production_changed=False,training_run=False,
                elapsed_s=time.monotonic()-started,peak_rss_bytes=rss_bytes())
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('completed','input_contract','counts','passed','elapsed_s')},ensure_ascii=False),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--require-pass',action='store_true')
    args=parser.parse_args();result=run(args.data,args.manifest,args.output)
    if args.require_pass and not result['passed']:sys.exit(1)
