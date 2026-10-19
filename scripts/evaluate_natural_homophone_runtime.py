"""Fresh full conversions against pinned production captures for all1880 cases."""
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
from scripts.evaluate_ensemble_regression import (
    load_targets,protected_segments_preserved,rss_bytes,validate_response,verify_pins,
)
from scripts.evaluate_ensemble_natural_regression import read_cases
from scripts.probe_numeric_parentheses import summarize

TRAIN=ROOT/'.omx/artifacts/natural-homophone-ranker-training-v1'
TRAIN_SHA='cf08afc5bba28931b154755e03899a9b7b26b56deca0db5cfa01fa783d3e3279'
REFERENCES=(
    ('.omx/artifacts/ensemble-regression-v2','6f2b8317c497604d83d4046cac1209606996b4bbf4f4f7c185cb21a0cc5d00d7'),
    ('.omx/artifacts/ensemble-natural-regression-v2','58071aa88b6ec9b71cbec74625064aca369b3681c602054b82130572141991a8'),
)
COUNTS=dict(dev=140,test=1000,natural_train=512,natural_dev=128,selection_dev=60,reasoning_dev=40)


def load_references():
    rows=[];pins={}
    for directory,digest in REFERENCES:
        path=ROOT/directory
        if file_sha256(path/'report.json')!=digest:raise ValueError('reference report changed')
        report=json.loads((path/'report.json').read_text())
        if file_sha256(path/'capture.jsonl')!=report['capture_sha256']:
            raise ValueError('reference capture changed')
        verify_pins(report['source_hashes'])
        pins.update(report['source_hashes'])
        pins[directory+'/report.json']=digest
        pins[directory+'/capture.jsonl']=report['capture_sha256']
        for row in map(json.loads,(path/'capture.jsonl').read_text().splitlines()):
            rows.append({k:row[k] for k in ('id','input','context','suite')}|{'parent':row['baseline']})
    if {s:sum(r['suite']==s for r in rows) for s in COUNTS}!=COUNTS:
        raise ValueError('reference suite denominator changed')
    if len({(r['suite'],r['id']) for r in rows})!=1880:
        raise ValueError('duplicate reference cases')
    return rows,pins


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    try:
        if file_sha256(TRAIN/'report.json')!=TRAIN_SHA:raise ValueError('training report changed')
        training=json.loads((TRAIN/'report.json').read_text())
        proof=json.loads((TRAIN/'independent-verification.json').read_text())
        if not training['advance_to_broader_evaluation'] or not proof['verified'] or proof['report_sha256']!=TRAIN_SHA:
            raise ValueError('training qualification missing')
        verify_pins(training['pins'])
        rows,pins=load_references()
        pins.update(training['pins'])
        for path in (TRAIN/'report.json',TRAIN/'independent-verification.json',Path(__file__),
                     ROOT/'hanglish/core/numeric_parentheses.py',ROOT/'scripts/probe_numeric_parentheses.py',
                     ROOT/'scripts/evaluate_ensemble_natural_regression.py'):
            pins[str(path.relative_to(ROOT))]=file_sha256(path)
        checkpoint=TRAIN/'checkpoint'
        if file_sha256(checkpoint/'model.safetensors')!=training['checkpoint_sha256'] or file_sha256(checkpoint/'metadata.json')!=training['metadata_sha256']:
            raise ValueError('trained bundle changed')
        import torch
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        baseline=NeuralConverter();baseline.load_models()
        for row in rows:
            if row['parent']['version']!=baseline.version or row['parent']['model_artifacts']!=baseline.model_artifacts:
                raise ValueError('reference parent identity differs')
        converter=NumericParenthesesConverter(baseline,experimental=True)
        converter.load_models()
        converter.ranker=SemanticRanker.load(checkpoint)
        converter.model_artifacts={k:v for k,v in converter.model_artifacts.items()
                                   if not k.startswith('checkpoints/semantic_ranker/')}
        converter.model_artifacts.update({str(p.relative_to(ROOT)):file_sha256(p) for p in checkpoint.iterdir() if p.is_file()})
        identity={'boundary_version':converter.version,'training_report':TRAIN_SHA,'evaluator':file_sha256(Path(__file__)),
                  'artifacts':converter.model_artifacts}
        converter.version='experimental-natural-context-1-'+hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:12]
        pins.update({k:v for k,v in converter.model_artifacts.items() if not k.startswith('experimental/')})
        verify_pins(pins)
        captures=[]
        with (output/'capture.jsonl').open('x') as stream:
            for index,row in enumerate(rows):
                response=converter.convert(row['input'],context=row['context'])
                for value in (row['parent'],response):
                    validate_response(value,row['input'],row['context'])
                    if not protected_segments_preserved(value):raise ValueError('protected segment changed')
                    if row['input'] not in [c['text'] for c in value['candidates']]:raise ValueError('raw choice missing')
                value={**row,'experiment':response}
                captures.append(value)
                stream.write(json.dumps(value,ensure_ascii=False)+'\n');stream.flush()
                if time.monotonic()-started>1200 or rss_bytes()>3*1024**3:
                    raise RuntimeError('full conversion exceeded1200seconds/3GiB')
                if (index+1)%64==0:
                    print(json.dumps(dict(captured=index+1,total=1880,suite=row['suite'],elapsed_s=time.monotonic()-started)),flush=True)
        verify_pins(pins)
        targets=load_targets()
        targets.update({(suite,row['id']):row['expected'] for suite,row in read_cases()})
        counts,changes=summarize(captures,targets)
        report=dict(schema='natural-homophone-runtime-v1',completed=True,rows=1880,
                    production_changed=False,fresh_experiment_inference=True,cached_pinned_parent=True,
                    scope='numeric annotations plus newly trained word ranker; exposed regression, not fresh generalization',
                    counts=counts,changes=changes,pins=pins,capture_sha256=file_sha256(output/'capture.jsonl'),
                    version=converter.version,model_artifacts=converter.model_artifacts,
                    elapsed_s=time.monotonic()-started,peak_rss_bytes=rss_bytes())
        (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({k:report[k] for k in ('counts','elapsed_s','peak_rss_bytes')},ensure_ascii=False),flush=True)
    except BaseException as exc:
        (output/'failure.json').write_text(json.dumps(dict(completed=False,error=repr(exc))))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
