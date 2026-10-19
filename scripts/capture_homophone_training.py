"""Freeze actual word proposals for all source-selected natural TRAIN rows."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hanglish.core.neural_converter import NeuralConverter, file_sha256
from hanglish.core.numeric_parentheses import NumericParenthesesConverter
from scripts.prepare_natural_word_ranker import Recorder, capture_groups
from scripts.probe_natural_homophone_contexts import DATA, project, summarize
from scripts.evaluate_ensemble_regression import rss_bytes, verify_pins

MANIFEST_SHA = '5c49a872942c126818ce5cbcc42cad434e3b60cdf7098dfe1097b535d35d25f3'
ANNOTATION_REPORT_SHA = '523af9a2a834b3e4667156ea457f51fac9b050247d565670d62d5608e6b078ad'


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    def guard():
        if time.monotonic()-started > 1200 or rss_bytes() > 3*1024**3:
            raise RuntimeError('candidate capture exceeded1200seconds/3GiB')
    try:
        if file_sha256(DATA/'manifest.json') != MANIFEST_SHA:
            raise ValueError('source manifest differs')
        manifest = json.loads((DATA/'manifest.json').read_text())
        source_path = DATA/'selected_train.jsonl'
        if file_sha256(source_path) != manifest['outputs']['selected_train']['sha256']:
            raise ValueError('selected TRAIN differs')
        annotation_report = ROOT/'.omx/artifacts/numeric-parentheses-v1/report.json'
        if file_sha256(annotation_report) != ANNOTATION_REPORT_SHA:
            raise ValueError('boundary regression evidence differs')
        raw_rows = list(map(json.loads, source_path.read_text().splitlines()))
        if len(raw_rows) != 853 or any(r['split'] != 'train' for r in raw_rows):
            raise ValueError('expected all853 selected TRAIN rows')
        sources = [project(r) for r in raw_rows]
        del raw_rows
        pins = dict(manifest['pins'])
        for path in (DATA/'manifest.json', source_path, annotation_report, Path(__file__),
                     ROOT/'scripts/prepare_natural_word_ranker.py',
                     ROOT/'scripts/probe_natural_homophone_contexts.py',
                     ROOT/'hanglish/core/numeric_parentheses.py'):
            pins[str(path.relative_to(ROOT))] = file_sha256(path)
        verify_pins(pins)
        (output/'inputs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in sources))
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        engine = NumericParenthesesConverter(NeuralConverter(), experimental=True)
        engine.load_models()
        pins.update(engine.model_artifacts)
        pins.pop('experimental/numeric_parentheses.py', None)
        recorder = Recorder(engine.ranker)
        engine.ranker = recorder
        captures = []
        with (output/'capture.jsonl').open('x') as stream:
            for index, row in enumerate(sources):
                recorder.records = []
                start = time.perf_counter()
                pieces, static, warnings, _, ranking = engine._prepare_candidates(
                    row['input'], row['context'], learned=True, raw=False)
                groups = capture_groups(static, recorder.records, row['context'])
                value = {**row, 'groups':deepcopy(groups), 'ranked_pieces':[
                    {'source':source,'start':offset,'end':offset+len(source),
                     'options':[vars(option) for option in options]} for source,offset,options in pieces],
                    'prepare_ms':(time.perf_counter()-start)*1000, 'warnings':warnings,
                    'ranking_applied':ranking, 'version':engine.version}
                captures.append(value)
                stream.write(json.dumps(value,ensure_ascii=False)+'\n')
                stream.flush()
                guard()
                if (index+1)%32 == 0:
                    print(json.dumps({'captured':index+1,'total':853,'elapsed_s':time.monotonic()-started}),flush=True)
        verify_pins(pins)
        # Only source-projected model inputs were used above. Join labels now.
        targets = {r['id']:r for r in map(json.loads,source_path.read_text().splitlines())}
        counts, outcomes = summarize(captures,targets)
        report = dict(schema='homophone-training-capture-v1', completed=True, training_ready=False,
                      training_run=False, production_changed=False, hold_rows_read=False,
                      scope='all853 source-selected TRAIN; actual word preparation, not sentence accuracy',
                      pins=pins, capture_sha256=file_sha256(output/'capture.jsonl'),
                      inputs_sha256=file_sha256(output/'inputs.jsonl'), version=engine.version,
                      counts=counts, outcomes=outcomes, elapsed_s=time.monotonic()-started,
                      peak_rss_bytes=rss_bytes())
        (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({k:report[k] for k in ('completed','counts','elapsed_s','peak_rss_bytes')},ensure_ascii=False),flush=True)
    except BaseException as exc:
        (output/'failure.json').write_text(json.dumps(dict(completed=False,error=repr(exc))))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
