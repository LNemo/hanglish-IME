"""Compare official NNBC counter grammar and lexical candidates on training only."""
from __future__ import annotations
import argparse
import copy
import csv
from functools import lru_cache
import io
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import hanglish.core.lexical_data as lexical
import hanglish.core.neural_converter as runtime
from hanglish.core.lattice import CandidateGenerator
from scripts.prepare_natural_ranker import selected_sources, source_fields
from scripts.prepare_natural_word_ranker import source_hashes
from scripts.train_natural_sentence_ranker import rss_bytes

SOURCE = ROOT/'.omx/artifacts/numeric-counter-source-v1'
COUNTER_SHA = 'e230971ad18d2162dfe9b46286b9b969f8eeab5d37006cd989959a335ec6b0b2'
ARCHIVE_SHA = 'fd62d3d6d8fa85145528065fabad4d7cb20f6b2201e71be4081a4e9701a5b330'
VARIANTS = ('baseline', 'counter_grammar', 'counter_grammar_and_dictionary')
TRAIN_CASES = 64
PROTECTED = ('10kg','60fps','3d','v1.2.3','192.168.0.1','0xff','1e10',
             'https://example.com/v2','user2@example.com','main.py:42','`12il`',
             '12abc','3nyeon2gaewol','12hours','5test','4API',
             '3gaeeul','3gaeeun','3myeongneun','3myeongga','3gaeeuro','30yakjeom')


def verified_units():
    metadata = json.loads((SOURCE/'SOURCE.json').read_text())
    if metadata['archive_sha256'] != ARCHIVE_SHA or runtime.file_sha256(SOURCE/'NNBC.csv') != COUNTER_SHA:
        raise ValueError('counter source is not the pinned official CSV')
    for name, digest in metadata['files'].items():
        if runtime.file_sha256(SOURCE/name) != digest:
            raise ValueError('counter source artifact differs')
    rows = list(csv.reader(io.StringIO((SOURCE/'NNBC.csv').read_text())))
    return tuple(sorted({r[0] for r in rows if len(r)>=12 and r[4]=='NNBC' and lexical.HANGUL.fullmatch(r[0])},
                        key=lambda s:(-len(s),s)))


def stage_data(output):
    data = output/'data'; data.mkdir()
    for name in ('morphology_lexicon.json','workbench_lexicon.json','common_english.txt','it_terms.txt'):
        (data/name).symlink_to(ROOT/'data'/name)
    (data/'challenge').mkdir()
    (data/'challenge/train_lexicon.json').symlink_to(ROOT/'data/challenge/train_lexicon.json')
    dictionary = data/'external/mecab-ko-dic'; dictionary.mkdir(parents=True)
    for name in lexical.FILES:
        (dictionary/(name+'.csv')).symlink_to(ROOT/'data/external/mecab-ko-dic'/(name+'.csv'))
    (dictionary/'NNBC.csv').write_bytes((SOURCE/'NNBC.csv').read_bytes())
    (dictionary/'COPYING').write_bytes((SOURCE/'COPYING').read_bytes())
    return data


def converter_with_dictionary(base, data):
    files = lexical.FILES
    try:
        lexical.FILES = files + ('NNBC',)
        generator = CandidateGenerator(data=data)
    finally:
        lexical.FILES = files
    converter = copy.copy(base)
    converter.generator = generator
    converter.word_options = lru_cache(maxsize=1024)(converter._word_options)
    return converter


def sentence_pool(converter, text, context):
    pieces, static, *_ = converter._prepare_candidates(text,context,learned=True,raw=False)
    combined = {path[0]:path for path in runtime.beam_paths(pieces)}
    for path in runtime.beam_paths(static): combined.setdefault(path[0],path)
    return list(combined)


def probe(output):
    if output.exists(): raise FileExistsError(output)
    units = verified_units()
    source, _ = selected_sources()
    rows = [source_fields(row,'natural_train') for row in source['natural_train'][:TRAIN_CASES]]
    before = source_hashes(); before['scripts/probe_numeric_counters.py'] = runtime.file_sha256(Path(__file__))
    output.mkdir(parents=True)
    baseline_units = runtime.NUMERIC_UNITS
    expanded_units = tuple(sorted(set(baseline_units)|set(units), key=lambda s:(-len(s),s)))
    selection = {'scope':'natural training only','cases':[r['id'] for r in rows],
                 'counter_sha256':COUNTER_SHA,'archive_sha256':ARCHIVE_SHA,
                 'variants':VARIANTS,'counter_units':len(units),'expanded_units':len(expanded_units),
                 'source_hashes':before,'test_data_read':False,'dev_rows_used':False,
                 'selection_rule':'No training first-choice or protection regressions; require improvement in first choice or sentence coverage; prefer more correct, then more covered, then fewer changes.'}
    (output/'selection.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2)+'\n')
    base = runtime.NeuralConverter(ranker_kind='hybrid'); base.load_models()
    if base.version != 'neural-1-7dfdf27691dd':
        raise ValueError('this archived probe requires its original runtime before counter integration')
    candidate = converter_with_dictionary(base,stage_data(output))
    models = {'baseline':base,'counter_grammar':base,'counter_grammar_and_dictionary':candidate}
    outcomes = {name:[] for name in VARIANTS}; protected = {}; started = time.monotonic()
    try:
        with (output/'cases.jsonl').open('x') as stream:
            for index,row in enumerate(rows):
                record = {k:row[k] for k in ('id','input','context','expected')}; record['variants'] = {}
                for name in VARIANTS:
                    runtime.NUMERIC_UNITS = baseline_units if name=='baseline' else expanded_units
                    converter = models[name]
                    actual = converter.convert(row['input'],row['context'])
                    pool = sentence_pool(converter,row['input'],row['context'])
                    item = {'id':row['id'],'actual':[c['text'] for c in actual['candidates']],
                            'top1':actual['result'] in row['expected'],'top5':any(c['text'] in row['expected'] for c in actual['candidates']),
                            'covered':any(c in row['expected'] for c in pool),'engine_ms':actual['engine_ms'],
                            'paths':pool}
                    outcomes[name].append(item); record['variants'][name] = item
                stream.write(json.dumps(record,ensure_ascii=False)+'\n'); stream.flush()
                if rss_bytes()>3*1024**3 or time.monotonic()-started>1800:
                    raise RuntimeError('numeric probe resource cap exceeded')
                if (index+1)%16==0:
                    print(json.dumps({'cases':index+1,'elapsed_seconds':time.monotonic()-started}),flush=True)
        for name in VARIANTS:
            runtime.NUMERIC_UNITS = baseline_units if name=='baseline' else expanded_units
            converter = models[name]; checks=[]
            for text in PROTECTED:
                actual=converter.convert(text)
                checks.append({'input':text,'result':actual['result'],'preserved':actual['result']==text})
            protected[name]=checks
    finally:
        runtime.NUMERIC_UNITS = baseline_units
    after = source_hashes(); after['scripts/probe_numeric_counters.py'] = runtime.file_sha256(Path(__file__))
    if before != after: raise ValueError('numeric experiment sources changed')
    old={r['id']:r for r in outcomes['baseline']}; summaries={}
    for name,items in outcomes.items():
        summaries[name]={'cases':len(items),'top1':sum(r['top1'] for r in items),'top5':sum(r['top5'] for r in items),
                         'covered':sum(r['covered'] for r in items),'mean_engine_ms':sum(r['engine_ms'] for r in items)/len(items),
                         'corrections':[r['id'] for r in items if r['top1'] and not old[r['id']]['top1']],
                         'regressions':[r['id'] for r in items if not r['top1'] and old[r['id']]['top1']],
                         'protection_failures':[r['input'] for r in protected[name] if not r['preserved']]}
    report={'summaries':summaries,'protection':protected,'parent_runtime_version':base.version,
            'dictionary_fingerprints':{'baseline':base.generator.dictionary_fingerprint,'candidate':candidate.generator.dictionary_fingerprint},
            'source_hashes':before,'selection_sha256':runtime.file_sha256(output/'selection.json'),
            'cases_sha256':runtime.file_sha256(output/'cases.jsonl'),'elapsed_seconds':time.monotonic()-started,
            'peak_rss_bytes':rss_bytes(),'deployed':False,'test_data_read':False}
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (output/'probe-source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'summaries':summaries,'peak_rss_bytes':rss_bytes()}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'.omx/artifacts/numeric-counter-probe-v1')
    probe(parser.parse_args().output)
