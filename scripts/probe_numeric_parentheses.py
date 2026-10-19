"""Paired full conversions on frozen TRAIN64 and exposed challenge regression."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hanglish.core.neural_converter import NeuralConverter, file_sha256
from hanglish.core.numeric_parentheses import NumericParenthesesConverter
from scripts.evaluate_ensemble_regression import (
    load_source_rows, load_targets, protected_segments_preserved,
    rss_bytes, validate_response, verify_pins,
)
from scripts.probe_natural_homophone_contexts import DATA, project

MANIFEST_SHA = '5c49a872942c126818ce5cbcc42cad434e3b60cdf7098dfe1097b535d35d25f3'


def summarize(captures, targets):
    counts = {}
    changes = []
    for row in captures:
        suite = row['suite']
        count = counts.setdefault(suite, dict(rows=0, parent_top1=0, parent_top5=0,
                                             experiment_top1=0, experiment_top5=0,
                                             fixes=0, losses=0, top5_losses=0,
                                             candidate_changes=0))
        expected = targets[(suite, row['id'])]
        choices = {name: [c['text'] for c in row[name]['candidates']]
                   for name in ('parent', 'experiment')}
        correct = {name: texts[0] in expected for name, texts in choices.items()}
        present = {name: bool(set(texts).intersection(expected)) for name, texts in choices.items()}
        count['rows'] += 1
        for name in choices:
            count[name+'_top1'] += correct[name]
            count[name+'_top5'] += present[name]
        count['fixes'] += correct['experiment'] and not correct['parent']
        count['losses'] += correct['parent'] and not correct['experiment']
        count['top5_losses'] += present['parent'] and not present['experiment']
        if row['parent']['candidates'] != row['experiment']['candidates']:
            count['candidate_changes'] += 1
            changes.append(dict(id=row['id'], suite=suite, expected=expected, choices=choices))
    return counts, changes


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        if file_sha256(DATA/'manifest.json') != MANIFEST_SHA:
            raise ValueError('natural source manifest changed')
        manifest = json.loads((DATA/'manifest.json').read_text())
        selected = DATA/'selected_train.jsonl'
        if file_sha256(selected) != manifest['outputs']['selected_train']['sha256']:
            raise ValueError('selected TRAIN changed')
        natural = [dict(project(row), suite='natural_train')
                   for row in list(map(json.loads, selected.read_text().splitlines()))[:64]]
        challenge, pins = load_source_rows()
        sources = natural + challenge
        for path in (DATA/'manifest.json', selected, Path(__file__),
                     ROOT/'hanglish/core/numeric_parentheses.py',
                     ROOT/'hanglish/core/neural_converter.py',
                     ROOT/'scripts/probe_natural_homophone_contexts.py'):
            pins[str(path.relative_to(ROOT))] = file_sha256(path)
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        parent = NeuralConverter()
        parent.load_models()
        experiment = NumericParenthesesConverter(parent, experimental=True)
        experiment.load_models()
        pins.update(experiment.model_artifacts)
        # Experimental source identity is metadata, not a repository path.
        pins.pop('experimental/numeric_parentheses.py', None)
        verify_pins(pins)
        captures = []
        with (output/'capture.jsonl').open('x') as stream:
            for index, row in enumerate(sources):
                captured = dict(row)
                for name, engine in (('parent', parent), ('experiment', experiment)):
                    response = engine.convert(row['input'], context=row['context'])
                    validate_response(response, row['input'], row['context'])
                    if not protected_segments_preserved(response):
                        raise ValueError('protected span changed')
                    if row['input'] not in [c['text'] for c in response['candidates']]:
                        raise ValueError('raw choice missing')
                    captured[name] = response
                captures.append(captured)
                stream.write(json.dumps(captured, ensure_ascii=False)+'\n')
                stream.flush()
                if time.monotonic()-started > 900 or rss_bytes() > 3*1024**3:
                    raise RuntimeError('900second/3GiB budget exceeded')
                if (index+1) % 32 == 0:
                    print(json.dumps(dict(captured=index+1, elapsed_s=time.monotonic()-started)), flush=True)
        verify_pins(pins)
        targets = load_targets()
        for row in map(json.loads, selected.read_text().splitlines()):
            targets[('natural_train', row['id'])] = [row['text']]
        counts, changes = summarize(captures, targets)
        report = dict(schema='numeric-parentheses-v1', completed=True, pins=pins,
                      counts=counts, changes=changes, capture_sha256=file_sha256(output/'capture.jsonl'),
                      elapsed_s=time.monotonic()-started, peak_rss_bytes=rss_bytes(),
                      training_run=False, production_changed=False, fresh_generalization=False,
                      scope='TRAIN64 full sentence conversion plus exposed dev140/test1000 regression',
                      versions=dict(parent=parent.version, experiment=experiment.version))
        (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        print(json.dumps({k: report[k] for k in ('counts','elapsed_s','peak_rss_bytes')}, ensure_ascii=False), flush=True)
    except BaseException as exc:
        (output/'failure.json').write_text(json.dumps(dict(completed=False, error=repr(exc))))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
