"""Compare the integrated converter with its parent on a declared whole-sentence split."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.sentence_data import read_split,DEFAULT_CORPUS
from scripts.evaluate_sentence_only import sha,summarize
from scripts.evaluate_ensemble_regression import rss_bytes


def compare(captures,labels):
    parents=[dict(id=r['id'],response=r['parent']) for r in captures]
    children=[dict(id=r['id'],response=r['mixed']) for r in captures]
    before=summarize(parents,labels);after=summarize(children,labels)
    improvements=sum(not a['top1'] and b['top1'] for a,b in zip(before['outcomes'],after['outcomes']))
    regressions=sum(a['top1'] and not b['top1'] for a,b in zip(before['outcomes'],after['outcomes']))
    losses=sum(a['top5'] and not b['top5'] for a,b in zip(before['outcomes'],after['outcomes']))
    return dict(parent=before,mixed=after,corrections=improvements,regressions=regressions,
                top5_losses=losses,improvement_gate=improvements>0 and regressions==0 and losses==0)


def latency(captures):
    """Engine timings after models were explicitly loaded; nearest-rank quantiles."""
    result={}
    for key in ('parent','mixed'):
        values=sorted(float(row[key]['engine_ms']) for row in captures)
        if not values or any(not math.isfinite(value) or value<0 for value in values):
            raise ValueError('invalid engine timings')
        result[key]={name:values[max(0,math.ceil(fraction*len(values))-1)]
                     for name,fraction in [('p50_ms',.5),('p95_ms',.95),('max_ms',1.)]}
    return result


def run(split,output,allow_test=False):
    import torch
    from hanglish.core.neural_converter import NeuralConverter
    from hanglish.core.mixed_span_converter import MixedSpanConverter
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    labels=read_split(DEFAULT_CORPUS,split,allow_test=allow_test)
    inputs=[dict(id=r['id'],input=r['input']) for r in labels]
    pins={str(p):sha(p) for p in (DEFAULT_CORPUS/(split+'.jsonl'),DEFAULT_CORPUS/'manifest.json',
          Path(__file__),ROOT/'hanglish/core/neural_converter.py',ROOT/'hanglish/core/mixed_span_converter.py',
          ROOT/'hanglish/core/mixed_span_proposals.py')}
    references=None
    if split=='dev':
        reference=ROOT/'.omx/artifacts/mixed-span-sentence-dev-v1'
        report=json.loads((reference/'report.json').read_text())
        if sha(reference/'capture.jsonl')!=report['capture_sha256']:raise ValueError('dev reference hash mismatch')
        references={r['id']:r for r in map(json.loads,(reference/'capture.jsonl').read_text().splitlines())}
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();parent=NeuralConverter();parent.load_models()
    # Share immutable model objects and dictionaries, but retain independent
    # conversion state/locks/caches. This changes no candidate-generation policy.
    mixed=MixedSpanConverter(external_lexicon=False)
    mixed.generator=parent.generator;mixed.numeric_units=parent.numeric_units
    mixed.ranker=parent.ranker;mixed.sentence_ranker=parent.sentence_ranker
    mixed.transducer=parent.transducer;mixed.load_models()
    if mixed.model_artifacts!=parent.model_artifacts:raise ValueError('model identity mismatch')
    pins.update({str(ROOT/p):value for p,value in parent.model_artifacts.items()})
    captures=[]
    with (output/'capture.jsonl').open('x') as stream:
        for item in inputs:
            before=parent.convert(item['input'],context='')
            after=mixed.convert(item['input'],context='')
            for response in (before,after):
                if response['context'] or response['external_context_used']:raise ValueError('external context used')
            if references is not None:
                ref=references[item['id']]
                if (before['candidates']!=ref['parent_displayed'] or before['result']!=ref['parent_result']
                        or after['candidates']!=ref['displayed'] or after['result']!=ref['result']):
                    raise ValueError('integrated runtime differs from the verified dev experiment')
            record=dict(id=item['id'],request=dict(input=item['input'],context=''),parent=before,mixed=after)
            captures.append(record);stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush()
            if len(captures)%50==0:print(json.dumps(dict(captured=len(captures),total=len(inputs),elapsed_s=time.monotonic()-started)),flush=True)
            if time.monotonic()-started>1800 or rss_bytes()>3*1024**3:raise RuntimeError('runtime evaluation resource limit exceeded')
    # Labels are used for scoring only after both engines complete every input.
    metrics=compare(captures,labels)
    for name,digest in pins.items():
        if sha(Path(name))!=digest:raise ValueError('pinned input/model changed')
    report=dict(completed=True,split=split,input_contract='roman_sentence_only',**metrics,
                pins=pins,capture_sha256=sha(output/'capture.jsonl'),
                parent_version=parent.version,mixed_version=mixed.version,
                dev_experiment_parity_checked=references is not None,
                latency=latency(captures),
                elapsed_s=time.monotonic()-started,peak_rss_bytes=rss_bytes(),
                deployed=False,training_run=False,test_accessed=split=='test',
                fully_blind_generalization=False)
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(completed=True,split=split,parent=report['parent']['counts'],mixed=report['mixed']['counts'],
                         corrections=report['corrections'],regressions=report['regressions'],top5_losses=report['top5_losses'],
                         improvement_gate=report['improvement_gate'],elapsed_s=report['elapsed_s'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--split',choices=['dev','test'],required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--final',action='store_true')
    args=parser.parse_args();run(args.split,args.output,args.final)
