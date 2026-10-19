"""Train the semantic word ranker on captured runtime collision pools."""
from __future__ import annotations

import argparse, json, random, sys, time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from hanglish.models.semantic_ranker import SemanticRanker, encode_pairs
from scripts.train_semantic_ranker import _optimizer, _set_trainable_layers
from scripts.train_counterfactual_context_ranker import prepare as old_pools, flatten
from scripts.train_natural_homophone_ranker import objective, RECIPE
from scripts.train_counterfactual_context_ranker import evaluate as evaluate_scored
from scripts.evaluate_ensemble_regression import file_sha256, rss_bytes, verify_pins

PARENT = ROOT / 'checkpoints/semantic_ranker'
DATA = ROOT / 'data/natural_context/runtime-collision-word-training-v1'

def fixed_accuracy(rows):
    checked = [r for r in rows if r.get('kind') == 'fixed' and ('fixed_result' in r)]
    return {'rows': len(checked), 'correct': sum(isinstance(r['fixed_result'], str) and r['fixed_result'] in r.get('expected', []) for r in checked)}


def prepare(data, manifest_sha):
    if data.resolve() != DATA.resolve():
        raise ValueError('runtime collision word data path is fixed')
    manifest = json.loads((data / 'manifest.json').read_text())
    if file_sha256(data / 'manifest.json') != manifest_sha or not manifest.get('completed') or not manifest.get('training_ready'):
        raise ValueError('training manifest is not verified')
    verification = json.loads((data / 'verification.json').read_text())
    if verification.get('verified') is not True or verification.get('manifest_sha256') != manifest_sha:
        raise ValueError('independent training data verification missing')
    selected = {}
    pins = dict(manifest.get('pins', {}))
    pins[str((data / 'manifest.json').relative_to(ROOT))] = manifest_sha
    pins[str((data / 'verification.json').relative_to(ROOT))] = file_sha256(data / 'verification.json')
    for role in ('fit', 'hold'):
        path = data / (role + '.jsonl')
        if file_sha256(path) != manifest['outputs'][role]['sha256']:
            raise ValueError('training pool differs')
        selected[role] = [json.loads(line) for line in path.read_text().splitlines()]
        if len(selected[role]) != manifest['outputs'][role]['rows']:
            raise ValueError('pool row count differs')
        pins[str(path.relative_to(ROOT))] = manifest['outputs'][role]['sha256']
    fit_ids = {r.get('group_id', r.get('id')) for r in selected['fit']}
    if fit_ids & {r.get('group_id', r.get('id')) for r in selected['hold']}: raise ValueError('source document leakage')
    existing, evaluation, old_pins = old_pools()
    retention = [r for r in existing if r['kind'] == 'retention']
    if len(retention) != 768: raise ValueError('retention selection differs')
    pins.update(old_pins)
    if not all(selected.values()) or any(r.get('kind') != 'source' for r in selected['fit']):
        raise ValueError('empty or invalid training split')
    if {s for r in selected['fit'] for s in r['source_ids']} & {s for r in selected['hold'] for s in r['source_ids']}:
        raise ValueError('source identity leakage')
    units = selected['fit'] + retention
    return units, {'source_hold': selected['hold'], **evaluation}, pins, manifest.get('counts', {})



def evaluate(ranker, pools, guard):
    scored = {role: [r for r in rows if r.get('kind') != 'fixed'] for role, rows in pools.items()}
    results = evaluate_scored(ranker, scored, guard)
    for role, rows in pools.items():
        predictions = {p['id']: p for p in results[role]['predictions']}
        if len(predictions) != len(results[role]['predictions']):
            raise ValueError('duplicate evaluation identity')
        for row in rows:
            if row.get('kind') == 'fixed':
                value = row['fixed_result']
                predictions[row['id']] = dict(id=row['id'], scores=[], result=value,
                    correct=isinstance(value, str) and value in row['expected'])
        ordered = [predictions[row['id']] for row in rows]
        results[role] = dict(rows=len(rows), correct=sum(p['correct'] for p in ordered), predictions=ordered)
    return results

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
        files=[Path(__file__),ROOT/'scripts/train_natural_homophone_ranker.py',ROOT/'scripts/train_counterfactual_context_ranker.py',
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
        report=dict(schema='runtime-collision-ranker-training-v1',completed=True,training_run=True,
            production_changed=False,advance_to_broader_evaluation=advance,
            source_hold_scope='fixed document split within selected2048 TRAIN; all nonquarantined held rows including missing/unscored; word diagnostic, not sentence accuracy',
            held_source_1583_labels_used=False,source_counts=source_counts,recipe=RECIPE,
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
    parser.add_argument('--data',type=Path,default=DATA)
    parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.data.resolve(),args.manifest_sha256,args.output.resolve())
