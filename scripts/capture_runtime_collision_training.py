"""Freeze real word-selector inputs from the expanded source TRAIN corpus."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hanglish.core.neural_converter import NeuralConverter, file_sha256
from scripts.prepare_natural_word_ranker import Recorder, capture_groups
from scripts.probe_natural_homophone_contexts import project
from scripts.evaluate_ensemble_regression import rss_bytes, verify_pins

DATA = ROOT / 'data/natural_context/runtime-collision-source-v5'
SOURCE_SHA = '293c3992878a9f7bdc532669ef2bf233233dd6ab9a4fddd7af1209e6b593de89'
OUTPUT = ROOT / '.omx/artifacts/runtime-collision-training-capture-v1'


def run():
    OUTPUT.mkdir(exist_ok=False)
    started = time.monotonic()
    try:
        assert file_sha256(DATA / 'manifest.json') == SOURCE_SHA
        manifest = json.loads((DATA / 'manifest.json').read_text())
        path = DATA / 'selected_train.jsonl'
        assert file_sha256(path) == manifest['outputs']['selected_train']['sha256']
        raw_rows = list(map(json.loads, path.read_text().splitlines()))
        assert len(raw_rows) == 2048 and all(r['split'] == 'train' for r in raw_rows)
        sources = [project(row) for row in raw_rows]
        del raw_rows
        pins = dict(manifest['pins'])
        for p in [DATA / 'manifest.json', path, Path(__file__),
                  ROOT / 'scripts/prepare_natural_word_ranker.py',
                  ROOT / 'scripts/probe_natural_homophone_contexts.py',
                  ROOT / 'hanglish/core/neural_converter.py']:
            pins[str(p.relative_to(ROOT))] = file_sha256(p)
        verify_pins(pins)
        (OUTPUT / 'inputs.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in sources))
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        engine = NeuralConverter()
        engine.load_models()
        pins.update(engine.model_artifacts)
        recorder = Recorder(engine.ranker)
        engine.ranker = recorder
        with (OUTPUT / 'capture.jsonl').open('x') as stream:
            for number, row in enumerate(sources):
                recorder.records = []
                before = time.monotonic()
                pieces, static, warnings, _, ranking = engine._prepare_candidates(
                    row['input'], row['context'], learned=True, raw=False)
                groups = capture_groups(static, recorder.records, row['context'])
                span = (row['focus_start'], row['focus_end'])
                target_groups = [g for g in groups if (g['start'], g['end']) == span]
                target_pieces = [(s, o, choices) for s, o, choices in pieces if (o, o + len(s)) == span]
                assert len(target_groups) <= 1 and len(target_pieces) <= 1
                record = {**row, 'groups': deepcopy(target_groups),
                    'ranked_pieces': [dict(source=s, start=o, end=o + len(s),
                                           options=[vars(c) for c in choices]) for s, o, choices in target_pieces],
                    'preliminary': [dict(source=s, start=o, end=o + len(s), text=choices[0].text)
                                    for s, o, choices in static],
                    'all_scored_groups': len(groups), 'ranking_applied': ranking,
                    'warnings': warnings, 'version': engine.version,
                    'prepare_ms': (time.monotonic() - before) * 1000}
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                stream.flush()
                if rss_bytes() > 3 * 1024 ** 3 or time.monotonic() - started > 2400:
                    raise RuntimeError('capture exceeded 2400seconds/3GiB')
                if (number + 1) % 32 == 0:
                    print(json.dumps(dict(captured=number + 1, total=len(sources),
                                          elapsed_s=time.monotonic() - started, peak_rss_bytes=rss_bytes())), flush=True)
        verify_pins(pins)
        report = dict(completed=True, training_run=False, training_ready=False,
            source_manifest_sha256=SOURCE_SHA, rows=len(sources), version=engine.version,
            scope='all 2048 selected source TRAIN rows, actual full word preparation; only focus groups saved',
            model_labels_supplied=False, sentence_inference=False, pins=pins,
            inputs_sha256=file_sha256(OUTPUT / 'inputs.jsonl'),
            capture_sha256=file_sha256(OUTPUT / 'capture.jsonl'),
            elapsed_s=time.monotonic() - started, peak_rss_bytes=rss_bytes())
        (OUTPUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({k: v for k, v in report.items() if k != 'pins'}), flush=True)
    except BaseException as exc:
        (OUTPUT / 'failure.json').write_text(json.dumps(dict(completed=False, error=repr(exc))))
        raise


if __name__ == '__main__':
    run()
