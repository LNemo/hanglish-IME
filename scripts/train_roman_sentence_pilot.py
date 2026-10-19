"""One bounded source-aware reranking experiment on TRAIN runtime proposals."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from hanglish.models.roman_sentence_ranker import RomanSentenceRanker, INPUT_CONTRACT
from scripts.train_sentence_ranker import _set_trainable_layers, _optimizer
from scripts.evaluate_sentence_only import sha
from scripts.evaluate_ensemble_regression import rss_bytes
from scripts.sentence_data import read_split, DEFAULT_CORPUS
from scripts.capture_roman_sentence_training import select_train

PARENT=ROOT/'checkpoints/sentence_ranker'
RECIPE=dict(seed=20260920,top_layers=2,head_lr=5e-5,encoder_lr=1e-5,weight_decay=.01,
            epochs=1,candidates=8,max_length=384,distillation_weight=.5,temperature=2.,
            threads=2,rss_limit=3*1024**3,seconds_limit=1800)


def read_rows(path):
    report=json.loads((path.parent/'report.json').read_text())
    if not report.get('completed') or report.get('sha256',{}).get(path.name)!=sha(path):
        raise ValueError('prepared pool file differs from completed capture/adapter report')
    rows=[json.loads(line) for line in path.read_text().splitlines() if line]
    if not rows or len({r['id'] for r in rows})!=len(rows):
        raise ValueError('empty/duplicate pool rows')
    for r in rows:
        if r.get('context') or r.get('left_context'):
            raise ValueError('external Korean context is forbidden')
        if len(r['candidates'])!=len(r['teacher_scores']) or not r['candidates']:
            raise ValueError('candidate/score mismatch')
        if len(set(r['candidates']))!=len(r['candidates']):
            raise ValueError('duplicate pool candidates')
        if any(not isinstance(x,(float,int)) or not math.isfinite(x) for x in r['teacher_scores']):
            raise ValueError('nonfinite teacher score')
    return rows


def validate_corpus_rows(rows,split):
    originals={r['id']:r for r in read_split(DEFAULT_CORPUS,split)}
    if split=='dev' and {r['id'] for r in rows}!=set(originals):
        raise ValueError('development denominator differs from corpus')
    if split=='train' and [r['id'] for r in rows]!=[r['id'] for r in select_train(list(originals.values()))]:
        raise ValueError('training rows differ from the fixed 512 source-only selection')
    for row in rows:
        original=originals.get(row['id'])
        if original is None or any(row[key]!=original[key] for key in ('input','expected')):
            raise ValueError('pool source/target differs from declared corpus split')


def training_group(row):
    positives=[i for i,c in enumerate(row['candidates']) if c in row['expected']]
    negatives=[i for i,c in enumerate(row['candidates']) if c not in row['expected']]
    if not positives or not negatives:return None
    if len(positives)>=RECIPE['candidates']:return None
    negatives.sort(key=lambda i:(-row['teacher_scores'][i],row['candidates'][i]))
    selected=positives+negatives[:RECIPE['candidates']-len(positives)]
    return dict(id=row['id'],input=row['input'],
                candidates=[row['candidates'][i] for i in selected],
                teacher_scores=[row['teacher_scores'][i] for i in selected],
                positive_indices=list(range(len(positives))),
                distill=row['result'] in row['expected'] and row['sentence_ranked'])


def loss_from_scores(scores, group):
    positives=torch.tensor(group['positive_indices'],device=scores.device)
    loss=torch.logsumexp(scores,0)-torch.logsumexp(scores[positives],0)
    if group['distill']:
        temperature=RECIPE['temperature']
        teacher=scores.new_tensor(group['teacher_scores'])
        loss=loss+RECIPE['distillation_weight']*temperature**2*torch.nn.functional.kl_div(
            torch.log_softmax(scores/temperature,0),torch.softmax(teacher/temperature,0),reduction='sum')
    return loss


def optimizer_step(ranker, optimizer, group):
    ranker.model.train();optimizer.zero_grad(set_to_none=True)
    encoded=ranker.encode([group['input']]*len(group['candidates']),group['candidates'])
    scores=ranker.model(**encoded).logits.reshape(-1)
    loss=loss_from_scores(scores,group)
    if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(ranker.model.parameters(),1.)
    if not torch.isfinite(norm):raise RuntimeError('nonfinite gradient')
    optimizer.step()
    return float(loss.detach()),float(norm)


def input_pins(train,dev):
    files=[train,dev,train.parent/'report.json',dev.parent/'report.json',Path(__file__),
           ROOT/'scripts/capture_roman_sentence_training.py',ROOT/'hanglish/models/roman_sentence_ranker.py',
           ROOT/'scripts/train_sentence_ranker.py',ROOT/'data/challenge/sentence-v3/manifest.json']
    files.extend(p for p in PARENT.iterdir() if p.is_file())
    return {str(p):sha(p) for p in files}


def validate_preflight(proof,pins,eligible):
    if (not proof.get('completed') or proof.get('phase')!='preflight'
            or proof.get('pins')!=pins or proof.get('recipe')!=RECIPE
            or proof.get('eligible')!=eligible):
        raise ValueError('preflight does not match this experiment')
    steps=proof.get('updates',[])
    if len(steps)!=3 or [s.get('step') for s in steps]!=[1,2,3]:
        raise ValueError('three actual optimizer updates required')
    for step in steps:
        if any(not isinstance(step.get(key),(int,float)) or not math.isfinite(step[key])
               or step[key]<0 for key in ('loss','gradient_norm','elapsed_s')):
            raise ValueError('invalid preflight optimizer evidence')
    for key,limit in (('peak_rss_bytes',RECIPE['rss_limit']),('elapsed_s',RECIPE['seconds_limit']),
                      ('max_tokens',RECIPE['max_length'])):
        value=proof.get(key)
        if not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<value<=limit:
            raise ValueError('preflight resource qualification failed')


def check_pins(pins):
    for name,expected in pins.items():
        if sha(Path(name))!=expected:raise ValueError('pinned source changed')


def evaluate(rows,ranker,output,guard):
    predictions=[]
    with output.open('x') as stream:
        for row in rows:
            try:
                scores=ranker.score_pairs([row['input']]*len(row['candidates']),row['candidates'])
                order=sorted(range(len(scores)),key=lambda i:(-scores[i],row['candidates'][i]))
                top5=[row['candidates'][i] for i in order[:5]]
                if row['input'] not in top5:top5=top5[:4]+[row['input']]
                record=dict(id=row['id'],result=row['candidates'][order[0]],scores=scores,
                            top5=top5,fallback=False)
            except ValueError as error:
                # Oversized complete pairs retain the actual parent's result;
                # they remain in the denominator and are explicitly counted.
                if 'token budget' not in str(error):raise
                record=dict(id=row['id'],result=row['result'],scores=None,
                            top5=row.get('displayed',[row['result']]),fallback=True)
            stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush()
            predictions.append(record);guard()
    # No target is consulted by scoring or fallback selection.
    return dict(rows=len(rows),correct=sum(p['result'] in r['expected'] for p,r in zip(predictions,rows)),
                top5=sum(any(c in r['expected'] for c in p['top5']) for p,r in zip(predictions,rows)),
                corrections=sum(r['result'] not in r['expected'] and p['result'] in r['expected'] for p,r in zip(predictions,rows)),
                regressions=sum(r['result'] in r['expected'] and p['result'] not in r['expected'] for p,r in zip(predictions,rows)),
                fallbacks=sum(p['fallback'] for p in predictions))


def run(args):
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    random.seed(RECIPE['seed']);torch.manual_seed(RECIPE['seed'])
    train=read_rows(args.train);dev=read_rows(args.dev)
    validate_corpus_rows(train,'train');validate_corpus_rows(dev,'dev')
    if {r['id'] for r in train}&{r['id'] for r in dev}:raise ValueError('split id overlap')
    pins=input_pins(args.train,args.dev)
    groups=[g for r in train if (g:=training_group(r)) is not None]
    if not groups:raise ValueError('no naturally covered competing training candidates')
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    def guard():
        if rss_bytes()>RECIPE['rss_limit'] or time.monotonic()-started>RECIPE['seconds_limit']:
            raise RuntimeError('source-aware training resource limit exceeded')
    ranker=RomanSentenceRanker.from_parent(PARENT)
    layer_info=_set_trainable_layers(ranker.model,RECIPE['top_layers'])
    optimizer=_optimizer(ranker.model,SimpleNamespace(**RECIPE),head_only=False)
    # Validate every selected training pair before making any update.
    lengths=[int(ranker.encode([g['input']]*len(g['candidates']),g['candidates'])['input_ids'].shape[1]) for g in groups]
    if args.phase=='preflight':
        worst=max(range(len(groups)),key=lambda i:(len(groups[i]['candidates'])*lengths[i]**2,groups[i]['id']))
        log=[]
        for step in range(3):
            start=time.monotonic();loss,norm=optimizer_step(ranker,optimizer,groups[worst]);guard()
            log.append(dict(step=step+1,loss=loss,gradient_norm=norm,elapsed_s=time.monotonic()-start))
        report=dict(completed=True,phase='preflight',recipe=RECIPE,input_contract=INPUT_CONTRACT,
                    pins=pins,eligible=len(groups),selected_group=groups[worst]['id'],max_tokens=max(lengths),
                    layers=layer_info,updates=log,peak_rss_bytes=rss_bytes(),elapsed_s=time.monotonic()-started,
                    production_changed=False,quality_evaluated=False)
    else:
        if args.preflight is None:raise ValueError('preflight report required')
        proof=json.loads(args.preflight.read_text())
        validate_preflight(proof,pins,len(groups))
        before=evaluate(dev,ranker,args.output/'untrained-dev.jsonl',guard)
        order=list(groups);random.Random(RECIPE['seed']).shuffle(order)
        losses=[]
        with (args.output/'updates.jsonl').open('x') as stream:
            for step,group in enumerate(order,1):
                loss,norm=optimizer_step(ranker,optimizer,group);guard();losses.append(loss)
                event=dict(step=step,id=group['id'],loss=loss,gradient_norm=norm)
                stream.write(json.dumps(event)+'\n');stream.flush()
                if step%25==0:print(json.dumps(dict(updates=step,total=len(order),loss=loss,elapsed_s=time.monotonic()-started)),flush=True)
        after=evaluate(dev,ranker,args.output/'trained-dev.jsonl',guard)
        checkpoint=args.output/'checkpoint';ranker.model.save_pretrained(checkpoint,safe_serialization=True)
        ranker.tokenizer.save_pretrained(checkpoint)
        artifact_hashes={p.name:sha(p) for p in checkpoint.iterdir() if p.is_file()}
        (checkpoint/'experiment.json').write_text(json.dumps(dict(input_contract=INPUT_CONTRACT,
            max_length=RECIPE['max_length'],artifact_hashes=artifact_hashes,pins=pins),indent=2)+'\n')
        guard()
        report=dict(completed=True,phase='train',recipe=RECIPE,input_contract=INPUT_CONTRACT,pins=pins,
                    preflight_sha256=sha(args.preflight),layers=layer_info,train_rows=len(train),eligible=len(groups),
                    updates=len(losses),mean_loss=sum(losses)/len(losses),untrained=before,trained=after,
                    parent_correct=sum(r['result'] in r['expected'] for r in dev),
                    checkpoint_hashes=artifact_hashes,peak_rss_bytes=rss_bytes(),elapsed_s=time.monotonic()-started,
                    production_changed=False,gate_passed=after['correct']>sum(r['result'] in r['expected'] for r in dev) and after['regressions']==0,
                    test_accessed=False,scope='fixed candidate pool pilot; no candidate-generation improvement claimed')
    check_pins(pins)
    guard()
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('pins','checkpoint_hashes')},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['preflight','train'])
    for name in ('train','dev','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--preflight',type=Path)
    run(parser.parse_args())
