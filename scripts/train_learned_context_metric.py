"""Learn candidate context metric using frozen embeddings and document cross-fit."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from hanglish.models.learned_context_metric import LearnedContextMetric,prototype_inputs
from hanglish.models.candidate_evidence import objective
from scripts.train_retrieved_context_evidence import pack
from scripts.train_structured_candidate_ranker import parent_prediction
from scripts.calibrate_context_evidence import compare
from scripts.evaluate_ensemble_regression import file_sha256,rss_bytes,verify_pins
SOURCE=ROOT/'.omx/artifacts/neural-context-retrieval-v1'
SOURCE_SHA='41aa32e8b1220078e86a8aae3cf58d6e558914a11f654930092b691dcbd5d302'
RECIPE=dict(seed=20260920,steps=300,lr=.003,weight_decay=.01,projection_size=64,residual_bound=4.,folds=4,max_seconds=600,max_rss_bytes=3*1024**3)


def fit(rows,values,guard,pilot_path=None):
    torch.manual_seed(RECIPE['seed']);model=LearnedContextMetric(values['query'].shape[1])
    placeholders={r['id']:dict(values=[[0.]*10 for c in r['candidates']]) for r in rows}
    targets=pack(rows,placeholders);packed=(values,*targets[1:])
    optimizer=torch.optim.Adam(model.parameters(),lr=RECIPE['lr'],weight_decay=RECIPE['weight_decay'])
    losses=[];started=time.monotonic()
    for step in range(RECIPE['steps']):
        optimizer.zero_grad(set_to_none=True);loss=objective(model,packed)
        if not torch.isfinite(loss):raise ValueError('nonfinite metric loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise ValueError('nonfinite metric gradient')
        optimizer.step();losses.append(float(loss.detach()));guard()
        if step==2 and pilot_path:
            estimate=(time.monotonic()-started)/3*300*4
            pilot_path.write_text(json.dumps(dict(updates=3,estimated_four_fold_training_s=estimate,peak_rss_bytes=rss_bytes()))+'\n')
            if estimate>450:raise RuntimeError('metric training projection exceeds450seconds')
    return model.eval(),losses


@torch.inference_mode()
def predict(model,rows,values):
    delta=model(values).double().tolist();out=[]
    for row,ds in zip(rows,delta):
        ds=ds[:len(row['candidates'])]
        scores=[v+.15*b+d for v,b,d in zip(row['teacher_scores'],row['base_scores'],ds)]
        i=min(range(len(scores)),key=lambda i:(-scores[i],row['candidates'][i]))
        result=row['candidates'][i]
        out.append(dict(id=row['id'],result=result,correct=result in row['expected'],delta=ds))
    return out


def save_values(path,values):
    np.savez_compressed(path,**{k:v.numpy() if isinstance(v,torch.Tensor) else np.array(v) for k,v in values.items()})


def run(output):
    output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    def guard():
        if time.monotonic()-started>600 or rss_bytes()>3*1024**3:raise RuntimeError('metric resource cap')
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    assert file_sha256(SOURCE/'report.json')==SOURCE_SHA
    report=json.loads((SOURCE/'report.json').read_text());proof=json.loads((SOURCE/'independent-verification.json').read_text())
    assert proof['verified'] and proof['report_sha256']==SOURCE_SHA
    verify_pins(report['pins'])
    for name,sha in report['hashes'].items():assert file_sha256(SOURCE/name)==sha
    selection=json.loads((SOURCE/'selection.json').read_text());rows,folds=selection['fit'],selection['folds']
    with np.load(SOURCE/'embeddings.npz',allow_pickle=False) as arrays:
        embeddings={id:v for id,v in zip(arrays['ids'].tolist(),arrays['vectors'])}
    assert set(embeddings)=={r['id'] for r in rows}
    pins=dict(report['pins'])
    for path in [Path(__file__),ROOT/'hanglish/models/learned_context_metric.py',ROOT/'.omx/plans/learned-context-metric-v1.md',SOURCE/'report.json',SOURCE/'independent-verification.json',SOURCE/'selection.json',SOURCE/'embeddings.npz']:
        pins[str(path.relative_to(ROOT))]=file_sha256(path)
    (output/'selection.json').write_text(json.dumps(dict(fit=rows,folds=folds),ensure_ascii=False)+'\n')
    oof={};bundles={}
    for fold in range(4):
        train=[r for r in rows if folds[r['id']]!=fold];valid=[r for r in rows if folds[r['id']]==fold]
        assert not {r['group_id'] for r in train}&{r['group_id'] for r in valid}
        assert not {x for r in train for x in r.get('source_ids',[])}&{x for r in valid for x in r.get('source_ids',[])}
        values=prototype_inputs(train,train,embeddings);held=prototype_inputs(train,valid,embeddings)
        save_values(output/f'features-fold{fold}-train.npz',values);save_values(output/f'features-fold{fold}-valid.npz',held)
        model,losses=fit(train,values,guard,output/'resource-pilot.json' if fold==0 else None)
        predictions=predict(model,valid,held)
        assert predict(LearnedContextMetric.restore(model.export()),valid,held)==predictions
        oof.update({r['id']:r for r in predictions})
        bundles[str(fold)]=dict(bank_ids=[r['id'] for r in train],validation_ids=[r['id'] for r in valid],state=model.export(),losses=losses)
        print(json.dumps(dict(fold=fold,elapsed_s=time.monotonic()-started)),flush=True)
    metrics={kind:compare([parent_prediction(r) for r in rows if r['kind']==kind],[oof[r['id']] for r in rows if r['kind']==kind]) for kind in ('source','retention')}
    gate=metrics['source']['after']>metrics['source']['before'] and metrics['retention']['after']>=metrics['retention']['before']
    if gate:
        values=prototype_inputs(rows,rows,embeddings);save_values(output/'features-all-fit.npz',values)
        model,losses=fit(rows,values,guard);predictions=predict(model,rows,values)
        assert predict(LearnedContextMetric.restore(model.export()),rows,values)==predictions
        bundles['all']=dict(bank_ids=[r['id'] for r in rows],state=model.export(),losses=losses)
    (output/'head.json').write_text(json.dumps(dict(recipe=RECIPE,bundles=bundles))+'\n')
    (output/'predictions.json').write_text(json.dumps(dict(out_of_fold=oof),ensure_ascii=False)+'\n')
    verify_pins(pins)
    result=dict(completed=True,training_run=True,production_changed=False,encoder_inference_run=False,encoder_trainable_parameters=0,trainable_parameters=sum(p.numel() for p in model.parameters()),recipe=RECIPE,oof_metrics=metrics,crossfit_gate=gate,diagnostic_evaluation_ready=gate,evaluation_run=False,pins=pins,hashes={p.name:file_sha256(p) for p in output.iterdir() if p.is_file()},elapsed_s=time.monotonic()-started,peak_rss_bytes=rss_bytes())
    (output/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('completed','oof_metrics','crossfit_gate','elapsed_s','peak_rss_bytes')},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output.resolve())
