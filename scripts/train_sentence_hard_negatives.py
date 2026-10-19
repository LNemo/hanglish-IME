"""Fine-tune sentence ranking on the new runtime's high-scoring TRAIN errors."""
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
from hanglish.models.sentence_ranker import SentenceRanker,encode_sentence_pairs
from scripts.train_sentence_ranker import _optimizer,_set_trainable_layers
from scripts.evaluate_ensemble_regression import file_sha256,rss_bytes,verify_pins

PARENT=ROOT/'checkpoints/sentence_ranker'
RECIPE=dict(seed=20260920,epochs=2,top_layers=2,group_batch=4,head_lr=5e-5,encoder_lr=1e-5,
            weight_decay=.01,temperature=2.,hard_weight=4.,distillation_weight=1.,candidate_cap=8)


def objective(logits,examples):
    cursor=0;losses=[]
    for example in examples:
        count=len(example['candidates']);group=logits[cursor:cursor+count];cursor+=count
        positive=example['positives']
        if len(group)!=count or not positive or len(positive)>=count or any(i<0 or i>=count for i in positive):
            raise ValueError('invalid candidate group or positives')
        correct=example['teacher_correct']
        if example['distill']!=correct or example['loss_weight']!=(1. if correct else RECIPE['hard_weight']):
            raise ValueError('hard-negative policy changed')
        loss=example['loss_weight']*(torch.logsumexp(group,0)-torch.logsumexp(group[positive],0))
        if correct:
            teacher=logits.new_tensor(example['teacher_scores']);temperature=RECIPE['temperature']
            if teacher.shape!=group.shape:raise ValueError('teacher score count differs')
            loss=loss+RECIPE['distillation_weight']*temperature**2*torch.nn.functional.kl_div(
                torch.log_softmax(group/temperature,0),torch.softmax(teacher/temperature,0),reduction='sum')
        losses.append(loss)
    if cursor!=len(logits) or not losses:raise ValueError('unconsumed logits or empty batch')
    return torch.stack(losses).mean()


def evaluate(ranker,pools,guard):
    result={}
    ranker.model.eval()
    for role,rows in pools.items():
        predictions=[]
        for row in rows:
            scores=ranker.score_pairs([row['context']]*len(row['candidates']),row['candidates']) if row['sentence_eligible'] else None
            if scores is None:
                text=row['fallback_result']
            else:
                choice=min(range(len(scores)),key=lambda i:(-scores[i],row['candidates'][i]))
                text=row['candidates'][choice]
            predictions.append(dict(id=row['id'],result=text,correct=text in row['expected'],scores=scores,
                                    covered=any(c in row['expected'] for c in row['candidates']),
                                    sentence_eligible=row['sentence_eligible']))
            guard()
        result[role]=dict(rows=len(rows),correct=sum(p['correct'] for p in predictions),
                          covered=sum(p['covered'] for p in predictions),predictions=predictions)
    return result


def run(data,manifest_sha,output):
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    def guard():
        if time.monotonic()-started>1200 or rss_bytes()>3*1024**3:
            raise RuntimeError('sentence training exceeds1200seconds/3GiB')
    try:
        if file_sha256(data/'manifest.json')!=manifest_sha:raise ValueError('training data changed')
        manifest=json.loads((data/'manifest.json').read_text())
        proof=json.loads((data/'verification.json').read_text())
        if not manifest['completed'] or proof.get('verified') is not True or proof['manifest_sha256']!=manifest_sha:
            raise ValueError('independent data verification missing')
        pins=dict(manifest['pins'])
        paths=[data/'manifest.json',data/'verification.json',Path(__file__),ROOT/'scripts/train_sentence_ranker.py',
               ROOT/'hanglish/models/sentence_ranker.py',ROOT/'scripts/evaluate_ensemble_regression.py',*PARENT.iterdir()]
        for path in paths:
            if path.is_file():pins[str(path.relative_to(ROOT))]=file_sha256(path)
        selected={}
        for role in ('fit','natural_hold','retention_hold'):
            path=data/(role+'.jsonl');digest=manifest['outputs'][role]['sha256']
            if file_sha256(path)!=digest:raise ValueError('prepared rows changed')
            pins[str(path.relative_to(ROOT))]=digest
            selected[role]=list(map(json.loads,path.read_text().splitlines()))
        fit=selected.pop('fit');pools=selected
        if not fit or any(not rows for rows in pools.values()):raise ValueError('empty training or evaluation pool')
        if {r['group_id'] for r in fit}&{r['group_id'] for rows in pools.values() for r in rows}:
            raise ValueError('source groups cross fit/hold')
        verify_pins(pins)
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        torch.manual_seed(RECIPE['seed']);random.seed(RECIPE['seed'])
        (output/'selection.json').write_text(json.dumps(dict(fit=fit,evaluation=pools),ensure_ascii=False)+'\n')
        ranker=SentenceRanker.load(PARENT)
        original={name:value.detach().clone() for name,value in ranker.model.state_dict().items()}
        before=evaluate(ranker,pools,guard)
        (output/'before.json').write_text(json.dumps(before,ensure_ascii=False)+'\n')
        layers=_set_trainable_layers(ranker.model,RECIPE['top_layers'])
        trainable={name for name,value in ranker.model.named_parameters() if value.requires_grad}
        optimizer=_optimizer(ranker.model,SimpleNamespace(**RECIPE),head_only=False)
        losses=[];updates=0;train_started=time.monotonic()
        with (output/'updates.jsonl').open('x') as stream:
            for epoch in range(RECIPE['epochs']):
                order=list(range(len(fit)));random.Random(RECIPE['seed']+epoch).shuffle(order)
                ranker.model.train()
                for start in range(0,len(order),RECIPE['group_batch']):
                    examples=[fit[i] for i in order[start:start+RECIPE['group_batch']]]
                    candidates=[c for r in examples for c in r['candidates']]
                    contexts=[r['context'] for r in examples for _ in r['candidates']]
                    encoded=encode_sentence_pairs(ranker.tokenizer,candidates,contexts,ranker.max_length)
                    optimizer.zero_grad(set_to_none=True)
                    loss=objective(ranker.model(**encoded).logits.reshape(-1),examples)
                    if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(ranker.model.parameters(),1.)
                    if not torch.isfinite(norm):raise ValueError('nonfinite gradients')
                    optimizer.step();updates+=1;losses.append(float(loss.detach()))
                    stream.write(json.dumps(dict(epoch=epoch,update=updates,ids=[r['id'] for r in examples],loss=losses[-1]))+'\n');stream.flush()
                    guard()
                    if updates==3:
                        projected=(time.monotonic()-train_started)/3*RECIPE['epochs']*((len(fit)+3)//4)
                        (output/'resource-pilot.json').write_text(json.dumps(dict(updates=3,estimated_training_s=projected,peak_rss_bytes=rss_bytes())))
                        if projected>900:raise RuntimeError('projected training exceeds900seconds')
                print(json.dumps(dict(epoch=epoch+1,updates=updates,elapsed_s=time.monotonic()-started)),flush=True)
        ranker.model.eval()
        changed=[name for name,value in ranker.model.state_dict().items() if not torch.equal(original[name],value)]
        if not changed or any(name not in trainable for name in changed):raise ValueError('unexpected weight update scope')
        checkpoint=output/'checkpoint';ranker.model.save_pretrained(checkpoint,safe_serialization=True)
        ranker.tokenizer.save_pretrained(checkpoint)
        hashes={p.name:file_sha256(p) for p in checkpoint.iterdir() if p.is_file()}
        (checkpoint/'metadata.json').write_text(json.dumps(dict(format_version=1,max_length=ranker.max_length,
            inference_batch_size=ranker.batch_size,artifact_hashes=hashes,experimental=True,recipe=RECIPE,source_pins=pins),indent=2)+'\n')
        after=evaluate(ranker,pools,guard)
        loaded=SentenceRanker.load(checkpoint)
        if evaluate(loaded,pools,guard)!=after:raise ValueError('saved score replay differs')
        verify_pins(pins)
        metrics={}
        for role in pools:
            a=before[role];b=after[role]
            metrics[role]=dict(rows=a['rows'],before=a['correct'],after=b['correct'],covered=b['covered'],
                fix_ids=[y['id'] for x,y in zip(a['predictions'],b['predictions']) if not x['correct'] and y['correct']],
                loss_ids=[y['id'] for x,y in zip(a['predictions'],b['predictions']) if x['correct'] and not y['correct']])
        advance=metrics['natural_hold']['after']>metrics['natural_hold']['before'] and all(not m['loss_ids'] for m in metrics.values())
        (output/'predictions.json').write_text(json.dumps(dict(before=before,after=after),ensure_ascii=False)+'\n')
        report=dict(schema='sentence-hard-negative-training-v1',completed=True,training_run=True,production_changed=False,
            scope='fixed captured full sentence pools; fallback/uncovered dev rows included; not fresh runtime',
            recipe=RECIPE,layers=layers,updates=updates,fit_units=len(fit),mean_loss=sum(losses)/len(losses),
            metrics=metrics,advance_to_full_runtime=advance,changed_parameter_names=changed,
            unchanged_tensor_count=len(original)-len(changed),reload_replay_equal=True,pins=pins,
            selection_sha256=file_sha256(output/'selection.json'),predictions_sha256=file_sha256(output/'predictions.json'),
            updates_sha256=file_sha256(output/'updates.jsonl'),checkpoint_sha256=file_sha256(checkpoint/'model.safetensors'),
            metadata_sha256=file_sha256(checkpoint/'metadata.json'),elapsed_s=time.monotonic()-started,peak_rss_bytes=rss_bytes())
        (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({k:report[k] for k in ('metrics','advance_to_full_runtime','updates','elapsed_s','peak_rss_bytes')},ensure_ascii=False),flush=True)
    except BaseException as exc:
        (output/'failure.json').write_text(json.dumps(dict(completed=False,error=repr(exc))))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--manifest-sha256',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.data.resolve(),args.manifest_sha256,args.output.resolve())
