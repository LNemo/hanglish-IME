"""Measure generation, pruning and final selection on source-selected TRAIN."""
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hanglish.core.neural_converter import NeuralConverter, file_sha256
from scripts.build_natural_context_corpus import stable_hash
from scripts.capture_current_sentence_pools import Recorder as SentenceRecorder
from scripts.prepare_natural_word_ranker import Recorder as WordRecorder, capture_groups
from scripts.probe_natural_homophone_contexts import project
from scripts.evaluate_ensemble_regression import rss_bytes, verify_pins

DATA=ROOT/'data/natural_context/runtime-collision-source-v1'
OUTPUT=ROOT/'.omx/artifacts/runtime-collision-pilot-v1'
MANIFEST_SHA='e2b882f684bbba2cd5a292c188fb4a10f9768df39f4cc426dc6919e0b00e6a28'


def select_pilot(rows, families=64):
    groups=defaultdict(lambda:defaultdict(list))
    for row in rows:
        if row['split']!='train':
            raise ValueError('pilot must use TRAIN only')
        groups[row['word_input']][row['word']].append(row)
    selected=[]
    for source in sorted(groups,key=lambda s:stable_hash('rcc-pilot-family-v1',s))[:families]:
        forms=groups[source]
        if len(forms)<2:
            raise ValueError('pilot family requires two observed source forms')
        for word in sorted(forms,key=lambda w:(len(forms[w]),stable_hash('rcc-pilot-form-v1',w)))[:2]:
            selected.append(min(forms[word],key=lambda r:stable_hash('rcc-pilot-row-v1',r['id'])))
    return selected


def measure(capture, target):
    a,b=capture['focus_start'],capture['focus_end']
    groups=[g for g in capture['groups'] if (g['start'],g['end'])==(a,b)]
    pieces=[p for p in capture['pieces'] if (p['start'],p['end'])==(a,b)]
    group=groups[0] if len(groups)==1 else None
    piece=pieces[0] if len(pieces)==1 else None
    word=target['word']
    full_pool={s for call in capture['sentence_calls'] for s in call['candidates']}
    final=capture['response']['candidates']
    return dict(id=target['id'], family=target['word_input'],word=word,
        aligned_piece=piece is not None, scored_group=group is not None,
        word_gold_in_pool=bool(group and word in group['candidates']),
        contrast_in_pool=bool(group and (set(target['contrast_candidates'])-{word}).intersection(group['candidates'])),
        word_first_correct=bool(piece and piece['options'][0]['text']==word),
        sentence_scored=bool(capture['sentence_calls']),sentence_gold_in_pool=target['text'] in full_pool,
        sentence_first_correct=bool(final and final[0]['text']==target['text']),
        sentence_top5_correct=any(c['text']==target['text'] for c in final))


def run():
    started=time.monotonic()
    if file_sha256(DATA/'manifest.json')!=MANIFEST_SHA:
        raise ValueError('source manifest changed')
    manifest=json.loads((DATA/'manifest.json').read_text())
    selected=DATA/'selected_train.jsonl'
    if file_sha256(selected)!=manifest['outputs']['selected_train']['sha256']:
        raise ValueError('selected source changed')
    rows=select_pilot([json.loads(line) for line in selected.read_text().splitlines()])
    ids=[r['id'] for r in rows]
    sources=[project(r) for r in rows]
    del rows
    pins=dict(manifest['pins'])
    files=[DATA/'manifest.json',selected,Path(__file__),ROOT/'scripts/prepare_natural_word_ranker.py',
           ROOT/'scripts/probe_natural_homophone_contexts.py',ROOT/'scripts/capture_current_sentence_pools.py']
    pins.update({str(p.relative_to(ROOT)):file_sha256(p) for p in files})
    verify_pins(pins)
    (OUTPUT/'inputs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in sources))
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    engine=NeuralConverter();engine.load_models()
    pins.update(engine.model_artifacts)
    word_recorder=WordRecorder(engine.ranker);engine.ranker=word_recorder
    sentence_recorder=SentenceRecorder(engine.sentence_ranker);engine.sentence_ranker=sentence_recorder
    original_prepare=engine._prepare_candidates
    prepared={}
    def record_prepare(*args,**kwargs):
        value=original_prepare(*args,**kwargs)
        prepared['value']=value
        return value
    engine._prepare_candidates=record_prepare
    with (OUTPUT/'capture.jsonl').open('x') as stream:
        for index,source in enumerate(sources):
            word_recorder.records=[];sentence_recorder.calls=[]
            start=time.monotonic()
            response=engine.convert(source['input'],context=source['context'])
            pieces,static,*_=prepared['value']
            groups=capture_groups(static,word_recorder.records,source['context'])
            record={**source,'groups':deepcopy(groups),'pieces':[
                dict(source=text,start=offset,end=offset+len(text),options=[vars(o) for o in options])
                for text,offset,options in pieces], 'sentence_calls':deepcopy(sentence_recorder.calls),
                'response':response,'elapsed_ms':(time.monotonic()-start)*1000}
            stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush()
            if time.monotonic()-started>600 or rss_bytes()>3*1024**3:
                raise RuntimeError('runtime pilot exceeded600seconds/3GiB')
            if (index+1)%16==0:
                print(json.dumps(dict(captured=index+1,total=len(sources),elapsed_s=time.monotonic()-started)),flush=True)
    capture_sha=file_sha256(OUTPUT/'capture.jsonl')
    verify_pins(pins)
    targets={r['id']:r for r in map(json.loads,selected.read_text().splitlines())}
    captured=[json.loads(line) for line in (OUTPUT/'capture.jsonl').read_text().splitlines()]
    if [r['id'] for r in captured]!=ids:
        raise ValueError('capture denominator/order differs')
    outcomes=[measure(c,targets[c['id']]) for c in captured]
    keys=[k for k,v in outcomes[0].items() if isinstance(v,bool)]
    counts={'rows':len(outcomes),**{k:sum(r[k] for r in outcomes) for k in keys}}
    report=dict(schema='runtime-collision-pilot-v1',completed=True,training_run=False,production_changed=False,
        scope='128 TRAIN source tokens in64 families; full production conversion; source spelling not expert-certified',
        held_inputs_used=False,pins=pins,version=engine.version,
        selection_policy='first64 hashed families,2 least frequent distinct selected source forms each,hashed row per form',
        inputs_sha256=file_sha256(OUTPUT/'inputs.jsonl'),capture_sha256=capture_sha,
        counts=counts,outcomes=outcomes,elapsed_s=time.monotonic()-started,peak_rss_bytes=rss_bytes())
    (OUTPUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('counts','elapsed_s','peak_rss_bytes')},ensure_ascii=False),flush=True)


if __name__=='__main__':
    run()
