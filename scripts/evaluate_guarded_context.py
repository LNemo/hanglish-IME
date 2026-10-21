"""Development-only calibration and fresh runtime checks for guarded ranking."""
from __future__ import annotations

import argparse
from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import evaluate_rr_retraining as rr

ARTIFACTS = ROOT / '.omx/artifacts'
STUDENT = ARTIFACTS / 'context-ranking-improvement-20261021/training/checkpoint'
THRESHOLDS = ((.5, 1.), (1., 1.), (2., 1.), (.5, 2.), (1., 2.),
              (2., 2.), (.5, 4.), (1., 4.), (2., 4.))
LIMITS = dict(whole_seconds=5400., update_seconds=60.,
              rss_bytes=3 * 1024**3, swap_growth_bytes=1536 * 1024**2)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def development_cases():
    sources = {
        'natural': ROOT / 'data/natural_context/word-ranker-pilot-v1/natural_dev.jsonl',
        'original': ROOT / 'data/challenge/dev.jsonl',
        'selection': ROOT / 'data/challenge/selection-dev-v1.1/cases.jsonl',
        'reasoning': ROOT / 'data/challenge/reasoning-dev-v1.1/cases.jsonl',
    }
    cases = []
    for suite, path in sources.items():
        for row in load_rows(path):
            cases.append({k: row.get(k, '') for k in ('input', 'context')} | {
                'id': suite + ':' + row['id'], 'expected': row['expected'],
                'suite': suite, 'raw': False})
    return cases, {str(p): digest(p) for p in sources.values()}


class CachedScores:
    """Bounded memoization for DEV sweeps only; final runs never use it."""
    def __init__(self, scorer, limit=40000):
        self.scorer, self.limit = scorer, limit
        self.cache = OrderedDict()

    def __getattr__(self, name):
        return getattr(self.scorer, name)

    def score_pairs(self, contexts, candidates):
        if len(contexts) != len(candidates):
            raise ValueError('contexts and candidates must have equal length')
        keys = list(zip(contexts, candidates))
        missing = list(dict.fromkeys(key for key in keys if key not in self.cache))
        values = (self.scorer.score_pairs([k[0] for k in missing], [k[1] for k in missing])
                  if missing else [])
        if len(values) != len(missing):
            raise ValueError('cached scorer returned the wrong number of scores')
        for key, value in zip(missing, values):
            self.cache[key] = value
        result = [self.cache[key] for key in keys]
        while len(self.cache) > self.limit:
            self.cache.popitem(last=False)
        return result


class CachedGeneration:
    """Cache exact source batches only during DEV sweeps."""
    def __init__(self, model, limit=512):
        self.model, self.limit, self.cache = model, limit, OrderedDict()

    def generate(self, sources):
        key = tuple(sources)
        if key not in self.cache:
            self.cache[key] = deepcopy(self.model.generate(sources))
        result = deepcopy(self.cache[key])
        while len(self.cache) > self.limit:
            self.cache.popitem(last=False)
        return result


def make_converter():
    from hanglish.core.score_scale_converter import ScoreScaleConverter
    converter = ScoreScaleConverter(
        score_scale_checkpoint=rr.SCORE_SCALE_CHECKPOINT,
        phonetic_checkpoint=ARTIFACTS / 'rr-retraining-20261020/training-cpu1/phonetic_transducer.pt',
        word_checkpoint=ARTIFACTS / 'ctc-rr-retraining-20261020/training-word/model.pt',
        growth_checkpoint=ARTIFACTS / 'ctc-rr-retraining-20261020/training-growth/model.pt')
    converter.load_models()
    converter.load_phonetic_model()
    converter.load_word_model()
    converter.load_growth_model()
    return converter


def capture(converter, cases, path, guard, name):
    inputs = [{k: row[k] for k in ('id', 'input', 'context', 'raw')} for row in cases]
    result = []
    with path.open('x') as stream:
        for i, item in enumerate(inputs):
            row = rr.capture_request(converter, item, guard)
            result.append(row)
            rr.append_jsonl(stream, row)
            if (i + 1) % 50 == 0 or i + 1 == len(inputs):
                print(json.dumps(dict(variant=name, completed=i + 1, total=len(inputs))), flush=True)
    return result


def metrics(before, after, cases):
    report = rr.compare_captures(before, after, [dict(c, category=c['suite']) for c in cases])
    outcomes = report.pop('outcomes')
    return report, outcomes


def select_variant(reports):
    eligible = [(key, row) for key, row in reports.items()
                if row['comparison']['corrections'] and not row['comparison']['regressions']
                and not row['comparison']['top5_regressions']
                and row['comparison'].get('baseline', {}).get('counts', {}).get('errors') == 0
                and row['comparison']['proposed']['counts']['errors'] == 0]
    if not eligible:
        return None
    return min(eligible, key=lambda pair: (
        -len(pair[1]['comparison']['corrections']), pair[1]['changed_results'],
        pair[1]['thresholds'][0], -pair[1]['thresholds'][1]))[0]


def validate_configuration(config, fresh=None, configuration_hash=None):
    grid = {f't{a:g}-s{b:g}': [a, b] for a, b in THRESHOLDS}
    if ({name: row.get('thresholds') for name, row in config.get('variants', {}).items()}
            != grid):
        raise ValueError('development configuration differs from declared grid')
    if (config.get('mode') != 'development' or config.get('completed') is not True
            or config.get('test_tuning') is not False or not config.get('selected')
            or config['selected'] != select_variant(config['variants'])):
        raise ValueError('no qualified development configuration')
    if fresh is not None:
        name = config['selected']
        if (fresh.get('mode') != 'development-fresh' or fresh.get('completed') is not True
                or fresh.get('memoization_used') is not False
                or fresh.get('test_tuning') is not False
                or fresh.get('configuration_sha256') != configuration_hash
                or fresh.get('selected') != name or set(fresh.get('variants', {})) != {name}
                or fresh['variants'][name]['thresholds'] != config['variants'][name]['thresholds']
                or select_variant(fresh['variants']) != name):
            raise ValueError('fresh development verification failed')


def run(output, mode, configuration=None, natural_holdout=None, fresh_validation=None):
    import torch
    from hanglish.models.semantic_ranker import SemanticRanker
    from hanglish.models.guarded_semantic_ranker import GuardedSemanticRanker
    from scripts.source_span_resources import AtomicResourceWatchdog, system_swap_bytes

    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    if mode == 'development':
        cases, source_pins = development_cases()
        variants = {f't{a:g}-s{b:g}': (a, b) for a, b in THRESHOLDS}
    else:
        config = json.loads(configuration.read_text())
        validate_configuration(config)
        if mode != 'development-fresh':
            if fresh_validation is None:
                raise ValueError('final evaluation requires fresh development verification')
            fresh = json.loads(fresh_validation.read_text())
            validate_configuration(config, fresh, digest(configuration))
        for path, value in config['source_pins'].items():
            if digest(path) != value:
                raise ValueError('frozen development source changed: ' + path)
        variants = {config['selected']: tuple(config['variants'][config['selected']]['thresholds'])}
        source_pins = {str(configuration): digest(configuration)}
        if fresh_validation is not None:
            source_pins[str(fresh_validation)] = digest(fresh_validation)
        if mode == 'development-fresh':
            cases, pins = development_cases()
            source_pins.update(pins)
        elif mode == 'test':
            rows, identity = rr.load_cases('test')
            cases = [dict(row, suite='test', raw=False) for row in rows]
            source_pins[identity['source']] = identity['source_sha256']
        elif mode == 'diagnostics':
            rows, identity = rr.load_cases('diagnostics')
            cases = [dict(row, suite=row['category']) for row in rows]
            source_pins[identity['source']] = identity['source_sha256']
        else:
            cases = [dict(row, id='natural-held:' + row['id'], suite='natural_holdout', raw=False)
                     for row in load_rows(natural_holdout)]
            source_pins[str(natural_holdout)] = digest(natural_holdout)
    source_pins.update(rr._source_pins())
    for path in [Path(__file__), ROOT / 'hanglish/models/guarded_semantic_ranker.py',
                 *STUDENT.iterdir()]:
        if path.is_file():
            source_pins[str(path)] = digest(path)
    rr.write_json(output / 'inputs.json', [{k: c.get(k, '') for k in ('id', 'input', 'context', 'raw')} for c in cases])
    rr.write_json(output / 'frozen.json', dict(source_pins=source_pins, variants=variants,
                                              mode=mode, test_tuning=False))
    with AtomicResourceWatchdog(started, system_swap_bytes(), output / 'resources.jsonl',
                               limits=LIMITS, interval_s=.5) as guard:
        with guard.operation('load_models', update=True):
            converter = make_converter()
            student = SemanticRanker.load(STUDENT)
            teacher = converter.ranker
            parent_artifacts = dict(converter.model_artifacts)
            if mode == 'development':
                teacher, student = CachedScores(teacher), CachedScores(student)
                converter.sentence_ranker = CachedScores(converter.sentence_ranker)
                converter.word_model = CachedGeneration(converter.word_model)
                converter.growth_model = CachedGeneration(converter.growth_model)
            converter.ranker = teacher
        baseline = capture(converter, cases, output / 'baseline.jsonl', guard, 'baseline')
        reports = {}
        for name, (a, b) in variants.items():
            converter.ranker = GuardedSemanticRanker(teacher, student,
                max_teacher_margin=a, min_student_margin=b)
            current = capture(converter, cases, output / (name + '.jsonl'), guard, name)
            comparison, outcomes = metrics(baseline, current, cases)
            rr.write_json(output / (name + '-outcomes.json'), outcomes)
            reports[name] = dict(thresholds=[a, b], comparison=comparison,
                latency=dict(baseline=rr._latency(baseline), proposed=rr._latency(current)),
                changed_results=sum(x['response']['result'] != y['response']['result']
                                    for x, y in zip(baseline, current)
                                    if 'response' in x and 'response' in y),
                capture_sha256=digest(output / (name + '.jsonl')))
            print(json.dumps(dict(variant=name, counts=comparison['proposed']['counts'],
                                  fixes=len(comparison['corrections']),
                                  regressions=len(comparison['regressions']))), flush=True)
        for path, value in source_pins.items():
            if digest(path) != value:
                raise ValueError('source changed during evaluation: ' + path)
        for path, value in parent_artifacts.items():
            if digest(ROOT / path) != value:
                raise ValueError('baseline model/source changed: ' + path)
        selected = select_variant(reports) if mode == 'development' else config['selected']
        report = dict(completed=True, mode=mode, selected=selected,
                      configuration_sha256=digest(configuration) if configuration else None,
                      variants=reports, source_pins=source_pins, baseline_artifacts=parent_artifacts,
                      baseline_capture_sha256=digest(output / 'baseline.jsonl'),
                      installed_models_changed=False, test_tuning=False,
                      generalization_claim_allowed=False,
                      limitation='Known regression and previously used natural corpora; no blind generalization claim.',
                      memoization_used=mode == 'development', elapsed_seconds=time.monotonic() - started)
        rr.write_json(output / 'report.json', report)
        print(json.dumps(dict(completed=True, selected=report['selected'])), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('development', 'development-fresh', 'test', 'diagnostics', 'natural-holdout'), required=True)
    parser.add_argument('--configuration', type=Path)
    parser.add_argument('--natural-holdout', type=Path)
    parser.add_argument('--fresh-validation', type=Path)
    args = parser.parse_args()
    output_existed = args.output.exists()
    try:
        run(args.output, args.mode, args.configuration, args.natural_holdout, args.fresh_validation)
    except BaseException as error:
        if not output_existed and args.output.is_dir() and not (args.output / 'report.json').exists():
            rr.write_json(args.output / 'failure.json', dict(completed=False,
                error_type=type(error).__name__, error=str(error)))
        raise
