"""Capture a fixed TRAIN-only sentence pilot from actual runtime candidate pools."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.sentence_data import read_split, DEFAULT_CORPUS
from scripts.trace_conversion_candidates import TraceConverter, trace_conversion
from scripts.evaluate_sentence_only import sha
from scripts.evaluate_ensemble_regression import rss_bytes


def select_train(rows):
    """Fixed source/identity selection, independent of labels and performance."""
    key = lambda r: hashlib.sha256(('roman-sentence-pilot-v1:'+r['id']).encode()).hexdigest()
    natural = sorted((r for r in rows if r['category']=='natural_sentence_context'), key=key)
    other = sorted((r for r in rows if r['category']!='natural_sentence_context'), key=key)
    return sorted(natural[:384]+other[:128], key=key)


def compact_trace(trace):
    return dict(input=trace['source'], result=trace['result'],
                candidates=trace['final_pre_display_paths'],
                displayed_candidates=trace['displayed_candidates'],
                sentence_ranked=bool(trace['sentence_score_pairs']),
                runtime=trace['runtime'])


def run(output, corpus=DEFAULT_CORPUS):
    started = time.monotonic()
    import torch
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    rows = read_split(corpus, 'train')
    selected = select_train(rows)
    assert len(selected)==512
    pins = {str(p):sha(p) for p in (corpus/'manifest.json',corpus/'train.jsonl',Path(__file__),
                                   ROOT/'scripts/trace_conversion_candidates.py')}
    output.mkdir(parents=True, exist_ok=False)
    (output/'selection.json').write_text(json.dumps([r['id'] for r in selected],indent=2)+'\n')
    inputs = [dict(id=r['id'],input=r['input']) for r in selected]
    converter = TraceConverter(); converter.load_models()
    model_pins = dict(converter.model_artifacts)
    captures=[]
    with (output/'capture.jsonl').open('x') as stream:
        for row in inputs:
            trace = trace_conversion(converter,row['input'],'')
            if trace['context'] or trace['runtime']['external_context_used']:
                raise ValueError('external context entered capture')
            record = dict(id=row['id'],**compact_trace(trace))
            captures.append(record)
            stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush()
            if len(captures)%32==0:
                print(json.dumps(dict(captured=len(captures),total=len(inputs),elapsed_s=time.monotonic()-started)),flush=True)
            if rss_bytes()>3*1024**3 or time.monotonic()-started>1800:
                raise RuntimeError('capture resource limit exceeded')
    # Labels join only after source-only generation finishes for every row.
    prepared=[]
    for record,row in zip(captures,selected):
        assert record['id']==row['id']
        prepared.append(dict(id=row['id'],input=row['input'],category=row['category'],
                             expected=row['expected'],result=record['result'],
                             candidates=[c['text'] for c in record['candidates']],
                             teacher_scores=[c['score'] for c in record['candidates']],
                             sentence_ranked=record['sentence_ranked']))
    for name,expected in {**pins,**model_pins}.items():
        p=Path(name);p=p if p.is_absolute() else ROOT/p
        if sha(p)!=expected:raise ValueError('source/model changed during capture')
    target=output/'train.jsonl'
    target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in prepared))
    if rss_bytes()>3*1024**3 or time.monotonic()-started>1800:
        raise RuntimeError('capture resource limit exceeded before report')
    report=dict(completed=True,input_contract='roman_sentence_only',split='train',rows=len(prepared),
                counts=dict(Counter(r['category'] for r in prepared)),
                covered=sum(any(c in r['expected'] for c in r['candidates']) for r in prepared),
                parent_correct=sum(r['result'] in r['expected'] for r in prepared),
                sha256={p.name:sha(p) for p in output.iterdir() if p.is_file()},
                source_hashes=pins,model_artifacts=model_pins,elapsed_s=time.monotonic()-started,
                peak_rss_bytes=rss_bytes(),test_accessed=False,production_changed=False)
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('completed','rows','covered','parent_correct','elapsed_s')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.output)
