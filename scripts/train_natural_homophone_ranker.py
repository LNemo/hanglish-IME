"""Adapt the existing word selector to newly captured natural spelling contexts."""
import argparse
import json
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from hanglish.models.semantic_ranker import SemanticRanker, encode_pairs
from scripts.train_semantic_ranker import _optimizer, _set_trainable_layers
from scripts.train_counterfactual_context_ranker import (
    prepare as old_pools, flatten, evaluate, objective as retention_objective,
)
from scripts.evaluate_ensemble_regression import file_sha256, rss_bytes, verify_pins

PARENT=ROOT/'checkpoints/semantic_ranker'
RECIPE=dict(seed=20260920,epochs=2,top_layers=2,batch_groups=4,
            head_lr=5e-5,encoder_lr=1e-5,weight_decay=.01,base_weight=.15,
            retention_temperature=2.,retention_kl_weight=1.)


def objective(logits,units):
    losses=[]
    cursor=0
    for unit in units:
        count=len(unit['candidates'])
        raw=logits[cursor:cursor+count]
        if len(raw)!=count:
            raise ValueError('too few logits')
        if unit['kind']=='source':
            positive=[i for i,c in enumerate(unit['candidates']) if c in unit['expected']]
            if not 0<len(positive)<count:
                raise ValueError('source training group is not informative')
            scores=raw+logits.new_tensor(unit['base_scores'])*RECIPE['base_weight']
            losses.append(torch.logsumexp(scores,0)-torch.logsumexp(scores[positive],0))
        elif unit['kind']=='retention':
            losses.append(retention_objective(raw,[unit]))
        else:
            raise ValueError('unknown training unit')
        cursor+=count
    if cursor!=len(logits) or not losses:
        raise ValueError('unused logits or empty training batch')
    return torch.stack(losses).mean()


def prepare(data,manifest_sha):
    if file_sha256(data/'manifest.json')!=manifest_sha:
        raise ValueError('training manifest changed')
    manifest=json.loads((data/'manifest.json').read_text())
    if not manifest['completed'] or not manifest['training_ready']:
        raise ValueError('training data not ready')
    verification_path=data/'verification.json'
    verification=json.loads(verification_path.read_text())
    if verification.get('verified') is not True or verification.get('manifest_sha256')!=manifest_sha:
        raise ValueError('independent training data verification missing')
    pins=dict(manifest['pins'])
    pins[str((data/'manifest.json').relative_to(ROOT))]=manifest_sha
    pins[str(verification_path.relative_to(ROOT))]=file_sha256(verification_path)
    selected={}
    for role in ('fit','hold'):
        path=data/(role+'.jsonl')
        expected=manifest['outputs'][role]['sha256']
        if file_sha256(path)!=expected:
            raise ValueError('training pool differs')
        pins[str(path.relative_to(ROOT))]=expected
        selected[role]=list(map(json.loads,path.read_text().splitlines()))
        if len(selected[role])!=manifest['outputs'][role]['rows']:
            raise ValueError('pool row count differs')
    if not all(selected.values()):
        raise ValueError('empty source split')
    if {r['group_id'] for r in selected['fit']}&{r['group_id'] for r in selected['hold']}:
        raise ValueError('source document leakage')
    existing,evaluation,existing_pins=old_pools()
    regular=[r for r in existing if r['kind']=='retention']
    if len(regular)!=768:
        raise ValueError('retention selection differs')
    pins.update(existing_pins)
    evaluation={'source_hold':selected['hold'],**evaluation}
    return selected['fit']+regular,evaluation,pins,manifest['counts']


def run(data,manifest_sha,output):
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    def guard():
        if time.monotonic()-started>1200 or rss_bytes()>3*1024**3:
            raise RuntimeError('training exceeds1200seconds/3GiB')
    try:
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        torch.manual_seed(RECIPE['seed']);random.seed(RECIPE['seed'])
        units,pools,pins,source_counts=prepare(data,manifest_sha)
        files=[Path(__file__),ROOT/'scripts/train_counterfactual_context_ranker.py',
               ROOT/'scripts/train_full_context_word_ranker.py',ROOT/'scripts/train_semantic_ranker.py',
               ROOT/'hanglish/models/semantic_ranker.py',ROOT/'scripts/evaluate_ensemble_regression.py',
               *PARENT.iterdir()]
        pins.update({str(p.relative_to(ROOT)):file_sha256(p) for p in files if p.is_file()})
        verify_pins(pins)
        (output/'selection.json').write_text(json.dumps(dict(fit=units,evaluation=pools),ensure_ascii=False)+'\n')
        ranker=SemanticRanker.load(PARENT)
        original={name:value.detach().clone() for name,value in ranker.model.state_dict().items()}
        before=evaluate(ranker,pools,guard)
        (output/'before.json').write_text(json.dumps(before,ensure_ascii=False)+'\n')
        configuration=_set_trainable_layers(ranker.model,RECIPE['top_layers'])
        trainable={name for name,p in ranker.model.named_parameters() if p.requires_grad}
        optimizer=_optimizer(ranker.model,SimpleNamespace(**RECIPE),head_only=False)
        updates=0
        losses=[]
        train_started=time.monotonic()
        with (output/'updates.jsonl').open('x') as stream:
            for epoch in range(RECIPE['epochs']):
                order=list(range(len(units)))
                random.Random(RECIPE['seed']+epoch).shuffle(order)
                ranker.model.train()
                for offset in range(0,len(order),RECIPE['batch_groups']):
                    batch=[units[i] for i in order[offset:offset+RECIPE['batch_groups']]]
                    contexts,candidates=flatten(batch)
                    encoded=encode_pairs(ranker.tokenizer,contexts,candidates,ranker.max_length)
                    optimizer.zero_grad(set_to_none=True)
                    loss=objective(ranker.model(**encoded).logits.reshape(-1),batch)
                    if not torch.isfinite(loss):raise ValueError('nonfinite loss')
                    loss.backward()
                    norm=torch.nn.utils.clip_grad_norm_(ranker.model.parameters(),1.)
                    if not torch.isfinite(norm):raise ValueError('nonfinite gradient')
                    optimizer.step()
                    updates+=1
                    losses.append(float(loss.detach()))
                    stream.write(json.dumps(dict(epoch=epoch,update=updates,ids=[u['id'] for u in batch],loss=losses[-1]))+'\n')
                    stream.flush()
                    guard()
                    if updates==3:
                        projected=(time.monotonic()-train_started)/3*RECIPE['epochs']*((len(units)+3)//4)
                        (output/'resource-pilot.json').write_text(json.dumps(dict(updates=3,estimated_training_s=projected,peak_rss_bytes=rss_bytes())))
                        if projected>900:raise RuntimeError('projected training exceeds900seconds')
                print(json.dumps(dict(epoch=epoch+1,updates=updates,elapsed_s=time.monotonic()-started)),flush=True)
        ranker.model.eval()
        changed=[name for name,value in ranker.model.state_dict().items() if not torch.equal(original[name],value)]
        if not changed or any(name not in trainable for name in changed):
            raise ValueError('weights changed outside intended scope')
        checkpoint=output/'checkpoint'
        ranker.model.save_pretrained(checkpoint,safe_serialization=True)
        ranker.tokenizer.save_pretrained(checkpoint)
        hashes={p.name:file_sha256(p) for p in checkpoint.iterdir() if p.is_file()}
        (checkpoint/'metadata.json').write_text(json.dumps(dict(format_version=1,max_length=ranker.max_length,
            inference_batch_size=ranker.batch_size,artifact_hashes=hashes,experimental=True,recipe=RECIPE,source_pins=pins),indent=2)+'\n')
        after=evaluate(ranker,pools,guard)
        loaded=SemanticRanker.load(checkpoint)
        if evaluate(loaded,pools,guard)!=after:
            raise ValueError('checkpoint reload does not reproduce predictions')
        verify_pins(pins)
        metrics={}
        for role in pools:
            prior,new=before[role]['predictions'],after[role]['predictions']
            metrics[role]=dict(rows=len(prior),before=before[role]['correct'],after=after[role]['correct'],
                fix_ids=[b['id'] for a,b in zip(prior,new) if not a['correct'] and b['correct']],
                loss_ids=[b['id'] for a,b in zip(prior,new) if a['correct'] and not b['correct']])
        advance=(metrics['source_hold']['after']>metrics['source_hold']['before']
                 and all(metrics[role]['after']>=metrics[role]['before'] for role in pools if role!='source_hold'))
        (output/'predictions.json').write_text(json.dumps(dict(before=before,after=after),ensure_ascii=False)+'\n')
        report=dict(schema='natural-homophone-ranker-training-v1',completed=True,training_run=True,
            production_changed=False,advance_to_broader_evaluation=advance,
            source_hold_scope='document split within selected853 TRAIN; candidate-group, not full sentence accuracy',
            held_source_1583_read=False,source_counts=source_counts,recipe=RECIPE,
            configuration=configuration,updates=updates,fit_units=len(units),
            changed_parameter_names=changed,unchanged_tensor_count=len(original)-len(changed),
            mean_loss=sum(losses)/len(losses),metrics=metrics,reload_replay_equal=True,pins=pins,
            selection_sha256=file_sha256(output/'selection.json'),predictions_sha256=file_sha256(output/'predictions.json'),
            updates_sha256=file_sha256(output/'updates.jsonl'),checkpoint_sha256=file_sha256(checkpoint/'model.safetensors'),
            metadata_sha256=file_sha256(checkpoint/'metadata.json'),elapsed_s=time.monotonic()-started,peak_rss_bytes=rss_bytes())
        (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({k:report[k] for k in ('completed','advance_to_broader_evaluation','metrics','updates','elapsed_s','peak_rss_bytes')},ensure_ascii=False),flush=True)
    except BaseException as exc:
        (output/'failure.json').write_text(json.dumps(dict(completed=False,error=repr(exc))))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.data.resolve(),args.manifest_sha256,args.output.resolve())
