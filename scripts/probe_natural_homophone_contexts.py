"""Capture real TRAIN candidate groups before joining mined spelling labels."""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hanglish.core.neural_converter import NeuralConverter, file_sha256
from hanglish.core.phonology import romanize_word
from scripts.prepare_natural_word_ranker import Recorder, capture_groups
from scripts.evaluate_ensemble_regression import rss_bytes, verify_pins

DATA=ROOT/'data/natural_context/homophone-source-v1'


def project(row):
    source=romanize_word(row['text'])
    start=len(romanize_word(row['text'][:row['word_start']]))
    end=start+len(row['word_input'])
    if source[start:end]!=row['word_input']:
        raise ValueError('mined token cannot be aligned to whole Roman input')
    return {'id':row['id'],'input':source,'context':row['left_context'],
            'focus_start':start,'focus_end':end}


def summarize(captures,targets):
    counts={'rows':len(captures),'aligned_runtime_piece':0,'scored_group':0,
            'gold_in_scored_candidates':0,'multiple_scored_candidates':0,
            'informative_with_gold':0,'mined_alternative_in_runtime':0,'word_first_correct':0}
    outcomes=[]
    for row in captures:
        target=targets[row['id']]
        groups=[g for g in row['groups'] if g['start']==row['focus_start'] and g['end']==row['focus_end']]
        pieces=[p for p in row['ranked_pieces'] if p['start']==row['focus_start'] and p['end']==row['focus_end']]
        group=groups[0] if len(groups)==1 else None
        piece=pieces[0] if len(pieces)==1 else None
        gold=bool(group and target['word'] in group['candidates'])
        multiple=bool(group and len(group['candidates'])>1)
        alternative=bool(group and (set(target['contrast_candidates'])-{target['word']}).intersection(group['candidates']))
        outcome={'id':row['id'],'aligned_runtime_piece':piece is not None,'scored_group':group is not None,
                 'gold_in_scored_candidates':gold,'multiple_scored_candidates':multiple,
                 'informative_with_gold':gold and multiple,'mined_alternative_in_runtime':alternative,
                 'word_first_correct':bool(piece and piece['options'][0]['text']==target['word']),
                 'spelling_anchors':target['spelling_anchors'],'match_kind':target['match_kind']}
        for key in counts:
            if key!='rows':counts[key]+=outcome[key]
        outcomes.append(outcome)
    return counts,outcomes


def run(output,manifest_sha):
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    def guard():
        if time.monotonic()-started>300 or rss_bytes()>3*1024**3:
            raise RuntimeError('candidate pilot exceeded300seconds/3GiB')
    try:
        if file_sha256(DATA/'manifest.json')!=manifest_sha:
            raise ValueError('source manifest differs')
        manifest=json.loads((DATA/'manifest.json').read_text())
        source_path=DATA/'selected_train.jsonl'
        if file_sha256(source_path)!=manifest['outputs']['selected_train']['sha256']:
            raise ValueError('selected TRAIN differs')
        raw_rows=[json.loads(line) for line in source_path.read_text().splitlines()][:64]
        if len(raw_rows)!=64 or any(r['split']!='train' for r in raw_rows):
            raise ValueError('expected64 TRAIN rows')
        sources=[project(r) for r in raw_rows]
        del raw_rows
        pins=dict(manifest['pins'])
        for path in (DATA/'manifest.json',source_path,Path(__file__),ROOT/'scripts/prepare_natural_word_ranker.py',
                     ROOT/'hanglish/core/neural_converter.py',ROOT/'hanglish/models/semantic_ranker.py',
                     ROOT/'hanglish/models/phonetic_transducer.py'):
            pins[str(path.relative_to(ROOT))]=file_sha256(path)
        verify_pins(pins)
        (output/'inputs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in sources))
        import torch
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        engine=NeuralConverter()
        engine.load_models()
        pins.update(engine.model_artifacts)
        recorder=Recorder(engine.ranker);engine.ranker=recorder
        captures=[]
        with (output/'capture.jsonl').open('x') as stream:
            for index,row in enumerate(sources):
                recorder.records=[]
                start=time.perf_counter()
                pieces,static,warnings,context_used,ranking=engine._prepare_candidates(
                    row['input'],row['context'],learned=True,raw=False)
                groups=capture_groups(static,recorder.records,row['context'])
                value={**row,'groups':deepcopy(groups),'ranked_pieces':[
                    {'source':source,'start':offset,'end':offset+len(source),
                     'options':[vars(option) for option in options]} for source,offset,options in pieces],
                    'prepare_ms':(time.perf_counter()-start)*1000,'warnings':warnings,
                    'ranking_applied':ranking,'version':engine.version,'model_artifacts':engine.model_artifacts}
                captures.append(value)
                stream.write(json.dumps(value,ensure_ascii=False)+'\n');stream.flush();guard()
                if (index+1)%16==0:
                    print(json.dumps({'captured':index+1,'elapsed_s':time.monotonic()-started}),flush=True)
        capture_sha=file_sha256(output/'capture.jsonl')
        verify_pins(pins)
        # The positive spelling and synthetic alternatives are joined only now.
        targets={r['id']:r for r in map(json.loads,source_path.read_text().splitlines())}
        counts,outcomes=summarize(captures,targets)
        report={'schema':'natural-homophone-runtime-pilot-v1','completed':True,'training_run':False,
                'hold_rows_read':False,'production_changed':False,'new_generalization':False,
                'scope':'first64 frozen selected TRAIN rows; actual word preparation, not complete sentence conversion',
                'pins':pins,'capture_sha256':capture_sha,'inputs_sha256':file_sha256(output/'inputs.jsonl'),
                'counts':counts,'outcomes':outcomes,'elapsed_s':time.monotonic()-started,'peak_rss_bytes':rss_bytes()}
        (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({k:report[k] for k in ('completed','counts','elapsed_s','peak_rss_bytes')},ensure_ascii=False),flush=True)
    except BaseException as exc:
        (output/'failure.json').write_text(json.dumps({'completed':False,'error':repr(exc)},indent=2))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--manifest-sha256',required=True)
    args=parser.parse_args();run(args.output,args.manifest_sha256)
