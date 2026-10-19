"""Capture complete sentence-score calls from the qualified experimental word model."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hanglish.core.neural_converter import NeuralConverter,file_sha256
from hanglish.core.numeric_parentheses import NumericParenthesesConverter
from hanglish.models.semantic_ranker import SemanticRanker
from scripts.evaluate_ensemble_regression import rss_bytes,verify_pins

REFERENCE=ROOT/'.omx/artifacts/natural-homophone-runtime-v1'
REFERENCE_SHA='ea44f45e47918b2af77ff0c80b847cd3939a1fac429bbb73045ef67069bd4db2'
TRAIN=ROOT/'.omx/artifacts/natural-homophone-ranker-training-v1'
TRAIN_SHA='cf08afc5bba28931b154755e03899a9b7b26b56deca0db5cfa01fa783d3e3279'


class Recorder:
    def __init__(self,delegate):
        self.delegate=delegate
        self.calls=[]

    def __getattr__(self,name):
        return getattr(self.delegate,name)

    def score_pairs(self,contexts,candidates):
        scores=self.delegate.score_pairs(contexts,candidates)
        self.calls.append(dict(contexts=list(contexts),candidates=list(candidates),scores=list(scores)))
        return scores


def load_converter(reference):
    baseline=NeuralConverter();baseline.load_models()
    converter=NumericParenthesesConverter(baseline,experimental=True)
    converter.load_models()
    converter.ranker=SemanticRanker.load(TRAIN/'checkpoint')
    converter.model_artifacts={k:v for k,v in converter.model_artifacts.items()
                               if not k.startswith('checkpoints/semantic_ranker/')}
    converter.model_artifacts.update({str(p.relative_to(ROOT)):file_sha256(p)
        for p in (TRAIN/'checkpoint').iterdir() if p.is_file()})
    identity=dict(boundary_version=converter.version,training_report=TRAIN_SHA,
                  evaluator=file_sha256(ROOT/'scripts/evaluate_natural_homophone_runtime.py'),
                  artifacts=converter.model_artifacts)
    converter.version='experimental-natural-context-1-'+hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:12]
    if converter.version!=reference['version'] or converter.model_artifacts!=reference['model_artifacts']:
        raise ValueError('experimental converter identity differs')
    return converter


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    try:
        if file_sha256(REFERENCE/'report.json')!=REFERENCE_SHA:raise ValueError('reference report changed')
        report=json.loads((REFERENCE/'report.json').read_text())
        if file_sha256(REFERENCE/'capture.jsonl')!=report['capture_sha256']:
            raise ValueError('reference predictions changed')
        if file_sha256(TRAIN/'report.json')!=TRAIN_SHA:raise ValueError('word training changed')
        pins=dict(report['pins'])
        for path in (REFERENCE/'report.json',REFERENCE/'capture.jsonl',Path(__file__)):
            pins[str(path.relative_to(ROOT))]=file_sha256(path)
        verify_pins(pins)
        rows=[r for r in map(json.loads,(REFERENCE/'capture.jsonl').read_text().splitlines())
              if r['suite'] in ('natural_train','natural_dev')]
        if {s:sum(r['suite']==s for r in rows) for s in ('natural_train','natural_dev')}!={'natural_train':512,'natural_dev':128}:
            raise ValueError('source denominator differs')
        import torch
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        converter=load_converter(report)
        recorder=Recorder(converter.sentence_ranker);converter.sentence_ranker=recorder
        counts=dict(rows=0,scored=0,fallback=0,pairs=0)
        with (output/'capture.jsonl').open('x') as stream:
            for index,row in enumerate(rows):
                recorder.calls=[]
                response=converter.convert(row['input'],context=row['context'])
                if response['candidates']!=row['experiment']['candidates']:
                    raise ValueError('full runtime candidate replay differs')
                if len(recorder.calls)!=int(response['sentence_ranking_applied']):
                    raise ValueError('sentence call count differs')
                value={k:row[k] for k in ('id','suite','input','context')}
                value.update(calls=recorder.calls,response=response)
                stream.write(json.dumps(value,ensure_ascii=False)+'\n');stream.flush()
                counts['rows']+=1
                counts['scored']+=bool(recorder.calls)
                counts['fallback']+=not recorder.calls
                counts['pairs']+=sum(len(c['candidates']) for c in recorder.calls)
                if time.monotonic()-started>1200 or rss_bytes()>3*1024**3:
                    raise RuntimeError('sentence capture exceeds1200seconds/3GiB')
                if (index+1)%32==0:
                    print(json.dumps(dict(captured=index+1,total=640,elapsed_s=time.monotonic()-started)),flush=True)
        verify_pins(pins)
        result=dict(schema='current-sentence-pool-capture-v1',completed=True,pins=pins,counts=counts,
                    capture_sha256=file_sha256(output/'capture.jsonl'),runtime_version=converter.version,
                    production_changed=False,training_run=False,labels_joined=False,
                    scope='TRAIN512/dev128; all sentence score candidates, not just displayed top5',
                    reference_projection_parses_all1880_prediction_rows=True,
                    elapsed_s=time.monotonic()-started,peak_rss_bytes=rss_bytes())
        (output/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({k:result[k] for k in ('counts','elapsed_s','peak_rss_bytes')},ensure_ascii=False),flush=True)
    except BaseException as exc:
        (output/'failure.json').write_text(json.dumps(dict(completed=False,error=repr(exc))))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
