"""Compare frozen neural context retrieval against the same document-fold baseline."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from transformers import AutoModel,AutoTokenizer
from hanglish.models.neural_context_retrieval import NeuralContextBank,mean_content
from hanglish.models.retrieved_context_evidence import RetrievedContextHead,context_text,FEATURES
from scripts.train_retrieved_context_evidence import capture,fit,predict,RECIPE
from scripts.train_structured_candidate_ranker import parent_prediction
from scripts.calibrate_context_evidence import compare
from scripts.evaluate_ensemble_regression import file_sha256,rss_bytes,verify_pins
SOURCE=ROOT/'.omx/artifacts/retrieved-context-evidence-v1'
SOURCE_SHA='2f69f477154dfb9debd2dce49381c6478590ca208b7bbbe501d531cfcd266a88'
MODEL=ROOT/'data/external/klue-roberta-small'


class ContextEncoder:
    def __init__(self):
        self.tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
        self.model,self.loading=AutoModel.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,add_pooling_layer=False,output_loading_info=True)
        if self.loading['missing_keys'] or self.loading.get('mismatched_keys'):
            raise ValueError('neural retrieval encoder not fully pretrained')
        self.model.eval();self.model.requires_grad_(False)
        self.embeddings={};self.inputs=[]

    @torch.inference_mode()
    def encode(self,rows,guard,output,pilot=False):
        started=time.monotonic()
        for start in range(0,len(rows),8):
            batch=rows[start:start+8];texts=[context_text(r['context']) for r in batch]
            encoded=self.tokenizer(texts,padding=True,truncation=True,max_length=128,return_tensors='pt',return_special_tokens_mask=True,return_token_type_ids=False)
            special=encoded.pop('special_tokens_mask').bool()
            special |= torch.isin(encoded['input_ids'],torch.tensor(self.tokenizer.all_special_ids))
            hidden=self.model(**encoded,return_dict=True).last_hidden_state
            vectors=mean_content(hidden,encoded['attention_mask'],special).float().cpu().numpy()
            if not np.isfinite(vectors).all():raise ValueError('nonfinite context embedding')
            for i,row in enumerate(batch):
                if row['id'] in self.embeddings:raise ValueError('duplicate encoded row ID')
                self.embeddings[row['id']]=vectors[i]
                n=int(encoded['attention_mask'][i].sum())
                self.inputs.append(dict(id=row['id'],text=texts[i],input_ids=encoded['input_ids'][i,:n].tolist(),special_tokens_mask=special[i,:n].tolist()))
            guard()
            if pilot and start==16:
                estimate=(time.monotonic()-started)/24*len(rows)
                (output/'resource-pilot.json').write_text(json.dumps(dict(encoded_rows=24,estimated_fit_encoding_s=estimate,peak_rss_bytes=rss_bytes()))+'\n')
                if estimate>450:raise RuntimeError('projected FIT encoding exceeds450seconds')
            if start%256==0:print(json.dumps(dict(encoded=min(start+8,len(rows)),rows=len(rows),elapsed_s=time.monotonic()-started)),flush=True)

    def save(self,output):
        ids=list(self.embeddings)
        np.savez(output/'embeddings.npz',ids=np.array(ids),vectors=np.array([self.embeddings[id] for id in ids],dtype=np.float32))
        (output/'encoding-inputs.json').write_text(json.dumps(self.inputs,ensure_ascii=False)+'\n')


def run(output):
    output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    def guard():
        if time.monotonic()-started>600 or rss_bytes()>3*1024**3:raise RuntimeError('neural retrieval resource limit')
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    assert file_sha256(SOURCE/'report.json')==SOURCE_SHA
    old=json.loads((SOURCE/'report.json').read_text());verify_pins(old['pins'])
    proof=json.loads((SOURCE/'independent-verification.json').read_text())
    assert proof['verified'] and proof['report_sha256']==SOURCE_SHA
    assert file_sha256(SOURCE/'selection.json')==old['hashes']['selection.json']
    selection=json.loads((SOURCE/'selection.json').read_text());rows,folds=selection['fit'],selection['folds']
    pins=dict(old['pins'])
    for path in [Path(__file__),ROOT/'hanglish/models/neural_context_retrieval.py',ROOT/'.omx/plans/neural-context-retrieval-v1.md',SOURCE/'report.json',SOURCE/'selection.json',SOURCE/'independent-verification.json',MODEL/'SOURCE.json']:
        pins[str(path.relative_to(ROOT))]=file_sha256(path)
    provenance=json.loads((MODEL/'SOURCE.json').read_text())
    for name,sha in provenance['files'].items():
        assert file_sha256(MODEL/name)==sha
        pins[str((MODEL/name).relative_to(ROOT))]=sha
    (output/'configuration.json').write_text(json.dumps(dict(recipe=RECIPE,pins=pins,model_source=provenance,encoding='mean non-special tokens; TRAIN-centered cosine',production_changed=False),ensure_ascii=False)+'\n')
    (output/'selection.json').write_text(json.dumps(selection,ensure_ascii=False)+'\n')
    encoder=ContextEncoder();guard();encoder.encode(rows,guard,output,pilot=True);encoder.save(output)
    bundles={};oof={}
    for fold in range(4):
        train=[r for r in rows if folds[r['id']]!=fold];valid=[r for r in rows if folds[r['id']]==fold]
        assert not {r['group_id'] for r in train}&{r['group_id'] for r in valid}
        assert not {x for r in train for x in r.get('source_ids',[])}&{x for r in valid for x in r.get('source_ids',[])}
        bank=NeuralContextBank(train,encoder.embeddings);features=capture(bank,train+valid,guard)
        model,losses=fit(train,features,guard);predictions=predict(model,valid,features)
        assert predict(RetrievedContextHead.restore(model.export()),valid,features)==predictions
        oof.update({r['id']:r for r in predictions})
        bundles[str(fold)]=dict(bank_ids=[r['id'] for r in train],validation_ids=[r['id'] for r in valid],center=bank.center.tolist(),state=model.export(),losses=losses)
        (output/f'features-fold{fold}.json').write_text(json.dumps(features,ensure_ascii=False)+'\n')
        print(json.dumps(dict(fold=fold,elapsed_s=time.monotonic()-started)),flush=True)
    metrics_oof={kind:compare([parent_prediction(r) for r in rows if r['kind']==kind],[oof[r['id']] for r in rows if r['kind']==kind]) for kind in ('source','retention')}
    gate=metrics_oof['source']['after']>metrics_oof['source']['before'] and metrics_oof['retention']['after']>=metrics_oof['retention']['before']
    metrics={};evaluation={}
    if gate:
        bank=NeuralContextBank(rows,encoder.embeddings);features=capture(bank,rows,guard);model,losses=fit(rows,features,guard)
        bundles['all']=dict(bank_ids=[r['id'] for r in rows],center=bank.center.tolist(),state=model.export(),losses=losses)
        (output/'features-all-fit.json').write_text(json.dumps(features,ensure_ascii=False)+'\n')
        for role,held in selection['evaluation'].items():
            encode_rows=[r for r in held if r.get('kind')!='fixed']
            assert not {r['group_id'] for r in rows}&{r['group_id'] for r in encode_rows if 'group_id' in r}
            encoder.encode(encode_rows,guard,output)
            features=capture(bank,held,guard)
            before=[parent_prediction(r) for r in held];after=predict(model,held,features)
            assert predict(RetrievedContextHead.restore(model.export()),held,features)==after
            evaluation[role]=dict(before=before,after=after);metrics[role]=compare(before,after)
            (output/f'features-{role}.json').write_text(json.dumps(features,ensure_ascii=False)+'\n')
        encoder.save(output)
    advance=gate and metrics['source_hold']['after']>metrics['source_hold']['before'] and all(m['after']>=m['before'] for role,m in metrics.items() if role!='source_hold')
    (output/'head.json').write_text(json.dumps(dict(recipe=RECIPE,features=FEATURES,bundles=bundles))+'\n')
    (output/'predictions.json').write_text(json.dumps(dict(out_of_fold=oof,evaluation=evaluation),ensure_ascii=False)+'\n')
    verify_pins(pins)
    report=dict(completed=True,training_run=True,production_changed=False,encoder_trainable_parameters=0,head_trainable_parameters=sum(p.numel() for p in model.parameters()),encoder_loading=encoder.loading,recipe=RECIPE,oof_metrics=metrics_oof,crossfit_gate=gate,metrics=metrics,advance_to_runtime=advance,pins=pins,hashes={p.name:file_sha256(p) for p in output.iterdir() if p.is_file()},elapsed_s=time.monotonic()-started,peak_rss_bytes=rss_bytes())
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('completed','oof_metrics','metrics','advance_to_runtime','elapsed_s','peak_rss_bytes')},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output.resolve())
