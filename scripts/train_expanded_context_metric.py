"""Hold the context metric recipe fixed while adding independent source examples."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from hanglish.models.learned_context_metric import LearnedContextMetric,prototype_inputs
from scripts.train_learned_context_metric import fit,predict,save_values,RECIPE
from scripts.train_neural_context_retrieval import ContextEncoder
from scripts.train_structured_candidate_ranker import parent_prediction
from scripts.calibrate_context_evidence import compare
from scripts.evaluate_ensemble_regression import file_sha256,rss_bytes,verify_pins
PRIOR=ROOT/'.omx/artifacts/learned-context-metric-v1'
PRIOR_SHA='9dda92732cb9f28c3dca92972f97bc38ca3288112a10d55a7f13575e4edab242'
EMBEDDINGS=ROOT/'.omx/artifacts/neural-context-retrieval-v1'
BANK=ROOT/'data/natural_context/expanded-context-bank-v1'
BANK_SHA='b1996e6386261883f961ff062db7d785fdfb087f7d1066ecf01a20e2657bf2d2'


def run(output):
    output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    def guard():
        if time.monotonic()-started>600 or rss_bytes()>3*1024**3:raise RuntimeError('expanded context metric resource cap')
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    pins={}
    for folder,expected in [(PRIOR,PRIOR_SHA),(BANK,BANK_SHA)]:
        assert file_sha256(folder/'report.json')==expected
        report=json.loads((folder/'report.json').read_text());proof=json.loads((folder/'independent-verification.json').read_text())
        assert proof['verified'] and proof['report_sha256']==expected
        verify_pins(report['pins']);pins.update(report['pins'])
        for name,sha in report['hashes'].items():assert file_sha256(folder/name)==sha
        for name in ['report.json','independent-verification.json']:
            pins[str((folder/name).relative_to(ROOT))]=file_sha256(folder/name)
    for path in [Path(__file__),ROOT/'.omx/plans/expanded-context-bank-v1.md',BANK/'bank.jsonl',PRIOR/'selection.json',EMBEDDINGS/'embeddings.npz',ROOT/'scripts/train_learned_context_metric.py',ROOT/'scripts/train_neural_context_retrieval.py']:
        pins[str(path.relative_to(ROOT))]=file_sha256(path)
    selection=json.loads((PRIOR/'selection.json').read_text());rows,folds=selection['fit'],selection['folds']
    extras=[json.loads(line) for line in (BANK/'bank.jsonl').read_text().splitlines()]
    assert not {r['id'] for r in rows}&{r['id'] for r in extras}
    assert not {r['group_id'] for r in rows}&{r['group_id'] for r in extras}
    assert not {s for r in rows for s in r.get('source_ids',[])}&{s for r in extras for s in r['source_ids']}
    (output/'selection.json').write_text(json.dumps(dict(fit=rows,folds=folds,extra_bank=extras),ensure_ascii=False)+'\n')
    (output/'configuration.json').write_text(json.dumps(dict(recipe=RECIPE,pins=pins,scope='same metric recipe and queries; independently sourced extra prototypes'),ensure_ascii=False)+'\n')
    with np.load(EMBEDDINGS/'embeddings.npz',allow_pickle=False) as arrays:
        embeddings={id:v for id,v in zip(arrays['ids'].tolist(),arrays['vectors'])}
    assert set(embeddings)=={r['id'] for r in rows}
    encoder=ContextEncoder();guard();encoder.encode(extras,guard,output,pilot=True);encoder.save(output)
    embeddings.update(encoder.embeddings)
    oof={};bundles={}
    for fold in range(4):
        train=[r for r in rows if folds[r['id']]!=fold];valid=[r for r in rows if folds[r['id']]==fold];bank=train+extras
        assert not {r['group_id'] for r in bank}&{r['group_id'] for r in valid}
        assert not {s for r in bank for s in r.get('source_ids',[])}&{s for r in valid for s in r.get('source_ids',[])}
        values=prototype_inputs(bank,train,embeddings);held=prototype_inputs(bank,valid,embeddings)
        save_values(output/f'features-fold{fold}-train.npz',values);save_values(output/f'features-fold{fold}-valid.npz',held)
        model,losses=fit(train,values,guard,output/'training-resource-pilot.json' if fold==0 else None)
        predictions=predict(model,valid,held)
        assert predict(LearnedContextMetric.restore(model.export()),valid,held)==predictions
        oof.update({r['id']:r for r in predictions})
        bundles[str(fold)]=dict(bank_ids=[r['id'] for r in bank],training_ids=[r['id'] for r in train],validation_ids=[r['id'] for r in valid],state=model.export(),losses=losses)
        print(json.dumps(dict(fold=fold,elapsed_s=time.monotonic()-started)),flush=True)
    metrics={kind:compare([parent_prediction(r) for r in rows if r['kind']==kind],[oof[r['id']] for r in rows if r['kind']==kind]) for kind in ('source','retention')}
    gate=metrics['source']['after']>metrics['source']['before'] and metrics['retention']['after']>=metrics['retention']['before']
    if gate:
        bank=rows+extras;values=prototype_inputs(bank,rows,embeddings);save_values(output/'features-all-fit.npz',values)
        model,losses=fit(rows,values,guard);predictions=predict(model,rows,values)
        assert predict(LearnedContextMetric.restore(model.export()),rows,values)==predictions
        bundles['all']=dict(bank_ids=[r['id'] for r in bank],training_ids=[r['id'] for r in rows],state=model.export(),losses=losses)
    (output/'head.json').write_text(json.dumps(dict(recipe=RECIPE,bundles=bundles))+'\n')
    (output/'predictions.json').write_text(json.dumps(dict(out_of_fold=oof),ensure_ascii=False)+'\n')
    verify_pins(pins)
    result=dict(completed=True,training_run=True,production_changed=False,extra_bank_rows=len(extras),encoder_new_rows=len(encoder.embeddings),encoder_trainable_parameters=0,trainable_parameters=sum(p.numel() for p in model.parameters()),recipe=RECIPE,oof_metrics=metrics,crossfit_gate=gate,diagnostic_evaluation_ready=gate,evaluation_run=False,pins=pins,hashes={p.name:file_sha256(p) for p in output.iterdir() if p.is_file()},elapsed_s=time.monotonic()-started,peak_rss_bytes=rss_bytes())
    (output/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('completed','extra_bank_rows','oof_metrics','crossfit_gate','elapsed_s','peak_rss_bytes')},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output.resolve())
