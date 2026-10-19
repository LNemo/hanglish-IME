"""Actual source-only DEV runtime parity against the accepted frozen screen."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import time

from scripts.evaluate_phonetic_score_scale_dev import ROOT, SOURCE, TRAIN, DEV, read, rows, write, sha, need
from scripts.evaluate_phonetic_choice_dev import CHECKPOINT, probe_groups
from scripts.evaluate_sentence_only import predict_sentence
from scripts.source_span_resources import AtomicResourceWatchdog, system_swap_bytes

SCREEN = ROOT / '.omx/artifacts/phonetic-score-scale-dev-screen-v1'
SCREEN_PROOF = 'e1b23b2fc23fbacba83c56358207eaa810c4b80addab59481c3b04a47a8ef8f4'
PLAN = ROOT / '.omx/plans/phonetic-score-scale-runtime-dev-v1.md'
LIMITS = dict(whole_seconds=1800., update_seconds=60., rss_bytes=3*1024**3, swap_growth_bytes=512*1024**2)
PRODUCTION_VERSION = 'experimental-ctc-union-1-488fc89732af'


def authorize():
    need(sha(SCREEN / 'independent-verification.json') == SCREEN_PROOF, 'screen proof differs')
    report, proof = read(SCREEN / 'report.json'), read(SCREEN / 'independent-verification.json')
    need(proof['verified'] is True and proof['quality_gate'] is True
         and proof['report_sha256'] == sha(SCREEN / 'report.json')
         and report['quality_gate'] is True and report['actual_runtime_authorized'] is True, 'screen did not qualify')
    need(sha(SCREEN / 'predictions.jsonl') == report['predictions_sha256'], 'closed predictions differ')
    pins = dict(report['pins'])
    for p in (Path(__file__), PLAN, ROOT / 'hanglish/core/score_scale_converter.py',
              ROOT / 'scripts/evaluate_phonetic_choice_dev.py', ROOT / 'scripts/evaluate_sentence_only.py',
              SCREEN / 'report.json', SCREEN / 'independent-verification.json', SCREEN / 'predictions.jsonl'):
        pins[str(p)] = sha(p)
    for name, digest in pins.items():
        if Path(name).resolve() != DEV.resolve():
            need(sha(name) == digest, 'pin differs: ' + name)
    expected = list(rows(SCREEN / 'predictions.jsonl'))
    need(len(expected) == 316 and len({r['id'] for r in expected}) == 316, 'request coverage differs')
    return report, expected, pins


def load_engines():
    import torch
    from hanglish.core.ctc_union_converter import CTCUnionConverter
    from hanglish.core.score_scale_converter import ScoreScaleConverter
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    class TracedProduction(CTCUnionConverter):
        def _extend_sentence_paths(self, text, context, paths):
            result = super()._extend_sentence_paths(text, context, paths)
            self.full_paths = deepcopy(result)
            return result
    parent = TracedProduction()
    parent.load_models(); parent.load_word_model(); parent.load_growth_model()
    need(parent.version == PRODUCTION_VERSION, 'production identity differs')
    def make(identity, phonetic=None):
        value = ScoreScaleConverter(phonetic_checkpoint=CHECKPOINT, phonetic_model=phonetic,
            identity_control=identity, external_lexicon=False,
            word_model=parent.word_model, growth_model=parent.growth_model)
        shared = ('generator', 'numeric_units', 'ranker', 'sentence_ranker', 'transducer')
        for name in shared:
            setattr(value, name, getattr(parent, name))
        value.load_models(); value.load_phonetic_model()
        need(value.choice_parent_version == parent.version and all(
            getattr(value, name) is getattr(parent, name) for name in (*shared, 'word_model', 'growth_model')), 'shared model differs')
        return value
    learned = make(False)
    identity = make(True, learned.phonetic_model)
    return parent, identity, learned


def check_response(original, expected, response, base_paths, model):
    need(original['id'] == expected['id'] and original['request'] == expected['request'], 'source order differs')
    source = expected['request']['input']
    need(response['input'] == source and response['context'] == '' and response['external_context_used'] is False, 'external context')
    need(response['candidates'] == expected['proposed']['candidates']
         and response['result'] == expected['proposed']['result']
         and response['sentence_ranking_applied'] == expected['proposed']['sentence_ranking_applied'], 'actual public differs from screen')
    need([list(p) for p in base_paths] == original['base_paths'], 'actual complete production pool differs')
    config = response['phonetic_score_scale']
    need(config['parameters'] == model['parameters'] and config['scale'] == model['scale']
         and config['checkpoint_sha256'] == sha(TRAIN / 'model.json') and config['identity_control'] is False
         and config['version'] == response['version']
         and response['version'].startswith('experimental-score-scale-1-')
         and response['phonetic_choice']['parent_version'] == PRODUCTION_VERSION
         and 'phonetic_word_union' not in response, 'coefficients or version differ')
    features = response['phonetic_choice']['rows']
    need(len(features) == len(expected['score_rows']), 'feature coverage differs')
    for actual, fixed in zip(features, expected['score_rows']):
        need(actual['text'] == fixed['text'] and actual['base_score'] == fixed['base_score']
             and actual['features'] == fixed['features'] and actual['adjusted_score'] == fixed['score'], 'runtime score/feature differs')


def append(stream, record):
    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
    stream.flush(); os.fsync(stream.fileno())


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        with AtomicResourceWatchdog(started, system_swap_bytes(), output / 'resources.jsonl', limits=LIMITS, interval_s=1.) as guard:
            with guard.operation('authenticate', update=True):
                screen, expected, pins = authorize()
                groups = probe_groups([(r['id'], r['request']['input'], None) for r in expected])
                write(output / 'pins.json', pins)
                write(output / 'probe-schedule.json', dict(groups=groups))
            with guard.operation('load_models', update=True):
                parent, identity, learned = load_engines()
                for engine in (parent, identity, learned):
                    for name, digest in engine.model_artifacts.items():
                        path = Path(name) if Path(name).is_absolute() else ROOT / name
                        need(pins.get(str(path)) == digest, 'unbound runtime model artifact: ' + str(path))
            sources = {r['id']: r['request']['input'] for r in expected}
            timings = []
            with (output / 'probe.jsonl').open('x') as stream:
                for group in groups:
                    identifier = group['probe_id']; source = sources[identifier]
                    with guard.operation('probe_parent:' + identifier, update=True):
                        parent.full_paths = []
                        before = predict_sentence(parent, source)
                    tick = time.monotonic()
                    with guard.operation('probe_identity:' + identifier, update=True):
                        after = predict_sentence(identity, source)
                        need(before['candidates'] == after['candidates'] and before['result'] == after['result']
                             and parent.full_paths == identity.choice_base_paths, 'actual identity control differs')
                        append(stream, dict(id=identifier, request=dict(input=source, context=''), parent=before,
                            identity=after, parent_paths=parent.full_paths, identity_paths=identity.choice_base_paths))
                    timings.append(dict(id=identifier, count=group['count'], identity_seconds=time.monotonic()-tick))
            elapsed = time.monotonic()-started
            projected = 1.3 * (elapsed + sum(r['count']*r['identity_seconds'] for r in timings)) + 60.
            with guard.operation('qualify', update=True):
                write(output / 'qualification.json', dict(qualified=projected <= 1800, projected_seconds=projected, elapsed_seconds=elapsed, timings=timings))
                need(projected <= 1800, 'runtime resource projection failed')
            model = read(TRAIN / 'model.json')
            duration_ms = []
            with (output / 'capture.jsonl').open('x') as stream:
                original_rows = iter(rows(SOURCE / 'capture.jsonl'))
                for index, target in enumerate(expected):
                    original = next(original_rows)
                    with guard.operation('predict:' + target['id'], update=True):
                        response = predict_sentence(learned, target['request']['input'])
                        check_response(original, target, response, learned.choice_base_paths, model)
                        append(stream, dict(id=target['id'], request=target['request'], response=response, base_paths=learned.choice_base_paths))
                        duration_ms.append(response['engine_ms'])
                    if (index+1) % 25 == 0:
                        print(json.dumps(dict(captured=index+1, total=316, elapsed_s=time.monotonic()-started)), flush=True)
                need(next(original_rows, None) is None, 'extra source capture')
            with guard.operation('close_capture', update=True):
                write(output / 'closure.json', dict(rows=316, capture_sha256=sha(output / 'capture.jsonl'), labels_opened=False))
            with guard.operation('write_report', update=True):
                for name, digest in pins.items():
                    if Path(name).resolve() != DEV.resolve():
                        need(sha(name) == digest, 'final pin changed')
                ordered = sorted(duration_ms)
                report = dict(schema='score-scale-runtime-dev-v1', completed=True, rows=316, pins=pins, limits=LIMITS,
                    metrics=screen['metrics'], quality_gate=screen['quality_gate'], version=learned.version,
                    actual_runtime_parity_rows=316, identity_probe_rows=16, labels_opened=False, test_accessed=False,
                    production_changed=False, artifacts={name: sha(output / name) for name in
                        ('capture.jsonl', 'closure.json', 'probe.jsonl', 'probe-schedule.json', 'qualification.json')},
                    latency_ms=dict(p50=ordered[len(ordered)//2], p95=ordered[int(.95*(len(ordered)-1))], maximum=max(ordered)),
                    limitations=['Previously observed DEV; no fresh generalization claim.', 'Metrics linked by exact public parity to independently verified screen.'])
                write(output / 'report.json', report)
            guard.sample_and_append('after_report')
            with guard.operation('write_completion', update=True):
                write(output / 'completion.json', dict(completed=True, report_sha256=sha(output / 'report.json')))
            guard.sample_and_append('after_completion')
        return report
    except BaseException as error:
        write(output / 'failure.json', dict(completed=False, error=f'{type(error).__name__}: {error}'))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    result = run(parser.parse_args().output)
    print(json.dumps({k: result[k] for k in ('completed', 'rows', 'quality_gate', 'latency_ms')}))
