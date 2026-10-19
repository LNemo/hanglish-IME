"""Document cross-fit neural residual using retrieved TRAIN context evidence."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from hanglish.models.retrieved_context_evidence import ContextBank,RetrievedContextHead,FEATURES
from hanglish.models.candidate_evidence import objective
from scripts.calibrate_context_evidence import compare
from scripts.train_structured_candidate_ranker import parent_prediction
from scripts.evaluate_ensemble_regression import file_sha256,rss_bytes,verify_pins
SOURCE=ROOT/'.omx/artifacts/candidate-evidence-head-v1'
SOURCE_SHA='ef500d83501d3e72310a98d1241ecdc736e16f7b88f530be01c7deddd76c1998'
RECIPE=dict(seed=20260920,steps=300,lr=.01,weight_decay=.01,folds=4,max_seconds=600,max_rss_bytes=3*1024**3)


def capture(bank,rows,guard):
    output={}
    for row in rows:
        if row.get('kind')!='fixed':
            values,evidence=bank.features(row)
            output[row['id']]=dict(values=values,evidence=evidence)
        guard()
    return output


def pack(rows,features):
    width=max(len(r['candidates']) for r in rows)
    values=torch.zeros((len(rows),width,len(FEATURES)),dtype=torch.float64)
    baseline=torch.full((len(rows),width),-1e9,dtype=torch.float64)
    valid=torch.zeros_like(baseline,dtype=torch.bool)
    positive=torch.zeros_like(valid)
    retention=torch.zeros(len(rows),dtype=torch.float64)
    for i,r in enumerate(rows):
        n=len(r['candidates']);values[i,:n]=torch.tensor(features[r['id']]['values'],dtype=torch.float64)
        baseline[i,:n]=torch.tensor([s+.15*b for s,b in zip(r['teacher_scores'],r['base_scores'])],dtype=torch.float64)
        valid[i,:n]=True
        positives=[j for j,c in enumerate(r['candidates']) if c in r['expected']]
        if r['kind']=='source' and not 0<len(positives)<n:raise ValueError('uninformative source')
        if r['kind'] not in ('source','retention'):raise ValueError('unknown role')
        positive[i,positives or list(range(n))]=True
        retention[i]=r['kind']=='retention'
    return values,baseline,valid,positive,retention


def fit(rows,features,guard):
    torch.manual_seed(RECIPE['seed']);model=RetrievedContextHead();packed=pack(rows,features)
    optimizer=torch.optim.Adam(model.parameters(),lr=RECIPE['lr'],weight_decay=RECIPE['weight_decay'])
    losses=[]
    for step in range(RECIPE['steps']):
        optimizer.zero_grad(set_to_none=True);loss=objective(model,packed)
        if not torch.isfinite(loss):raise ValueError('nonfinite loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise ValueError('nonfinite gradient')
        optimizer.step();losses.append(float(loss.detach()));guard()
    return model.eval(),losses


@torch.inference_mode()
def predict(model,rows,features):
    results=[]
    for row in rows:
        if row.get('kind')=='fixed':
            results.append(dict(parent_prediction(row),delta=[]));continue
        delta=model(torch.tensor(features[row['id']]['values'],dtype=torch.float64)).tolist()
        scores=[s+.15*b+d for s,b,d in zip(row['teacher_scores'],row['base_scores'],delta)]
        index=min(range(len(scores)),key=lambda i:(-scores[i],row['candidates'][i]))
        result=row['candidates'][index]
        results.append(dict(id=row['id'],result=result,correct=result in row['expected'],delta=delta))
    return results


def run(output):
    output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    def guard():
        if time.monotonic()-started>600 or rss_bytes()>3*1024**3:raise RuntimeError('retrieval experiment resource limit')
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    assert file_sha256(SOURCE/'report.json')==SOURCE_SHA
    source=json.loads((SOURCE/'report.json').read_text());verify_pins(source['pins'])
    assert file_sha256(SOURCE/'selection.json')==source['hashes']['selection.json']
    selection=json.loads((SOURCE/'selection.json').read_text())
    pins=dict(source['pins'])
    for path in [SOURCE/'report.json',SOURCE/'selection.json',Path(__file__),ROOT/'hanglish/models/retrieved_context_evidence.py',ROOT/'hanglish/models/candidate_evidence.py',ROOT/'.omx/plans/retrieved-context-evidence-v1.md']:
        pins[str(path.relative_to(ROOT))]=file_sha256(path)
    rows,folds=selection['fit'],selection['folds']
    (output/'selection.json').write_text(json.dumps(selection,ensure_ascii=False)+'\n')
    oof={};bundles={}
    for fold in range(4):
        train=[r for r in rows if folds[r['id']]!=fold];validation=[r for r in rows if folds[r['id']]==fold]
        assert not {r['group_id'] for r in train}&{r['group_id'] for r in validation}
        assert not {s for r in train for s in r.get('source_ids',[])}&{s for r in validation for s in r.get('source_ids',[])}
        bank=ContextBank(train);features=capture(bank,train+validation,guard)
        model,losses=fit(train,features,guard)
        predictions=predict(model,validation,features)
        assert predict(RetrievedContextHead.restore(model.export()),validation,features)==predictions
        oof.update({r['id']:r for r in predictions})
        bundles[str(fold)]=dict(bank_ids=[r['id'] for r in train],validation_ids=[r['id'] for r in validation],state=model.export(),losses=losses)
        (output/f'features-fold{fold}.json').write_text(json.dumps(features,ensure_ascii=False)+'\n')
        print(json.dumps(dict(fold=fold,elapsed_s=time.monotonic()-started)),flush=True)
    oof_metrics={kind:compare([parent_prediction(r) for r in rows if r['kind']==kind],[oof[r['id']] for r in rows if r['kind']==kind]) for kind in ('source','retention')}
    crossfit_gate=oof_metrics['source']['after']>oof_metrics['source']['before'] and oof_metrics['retention']['after']>=oof_metrics['retention']['before']
    metrics={};evaluation={}
    if crossfit_gate:
        bank=ContextBank(rows);features=capture(bank,rows,guard);model,losses=fit(rows,features,guard)
        bundles['all']=dict(bank_ids=[r['id'] for r in rows],state=model.export(),losses=losses)
        (output/'features-all-fit.json').write_text(json.dumps(features,ensure_ascii=False)+'\n')
        for role,held in selection['evaluation'].items():
            features=capture(bank,held,guard)
            before=[parent_prediction(r) for r in held];after=predict(model,held,features)
            assert predict(RetrievedContextHead.restore(model.export()),held,features)==after
            evaluation[role]=dict(before=before,after=after)
            metrics[role]=compare(before,after)
            (output/f'features-{role}.json').write_text(json.dumps(features,ensure_ascii=False)+'\n')
    advance=crossfit_gate and metrics['source_hold']['after']>metrics['source_hold']['before'] and all(m['after']>=m['before'] for role,m in metrics.items() if role!='source_hold')
    (output/'head.json').write_text(json.dumps(dict(recipe=RECIPE,features=FEATURES,bundles=bundles))+'\n')
    (output/'predictions.json').write_text(json.dumps(dict(out_of_fold=oof,evaluation=evaluation),ensure_ascii=False)+'\n')
    verify_pins(pins)
    report=dict(completed=True,training_run=True,production_changed=False,neural_encoder_weights_changed=False,trainable_parameters=sum(p.numel() for p in model.parameters()),recipe=RECIPE,oof_metrics=oof_metrics,crossfit_gate=crossfit_gate,metrics=metrics,advance_to_runtime=advance,pins=pins,hashes={p.name:file_sha256(p) for p in output.iterdir() if p.is_file()},elapsed_s=time.monotonic()-started,peak_rss_bytes=rss_bytes())
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('completed','oof_metrics','metrics','advance_to_runtime','elapsed_s','peak_rss_bytes')},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output.resolve())
