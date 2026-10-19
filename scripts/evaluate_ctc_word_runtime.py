"""Evaluate actual single-request CTC mixed-word augmentation against MixedSpan."""
import argparse
from collections import Counter
import json
import math
import re
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_sentence_only import read_inputs, predict_sentence, sha, summarize
from scripts.evaluate_ensemble_regression import rss_bytes

REFERENCE = {
    'dev': ('7e1830fee4f0bf7ecc2fd796ee1ba3cff8451070ef270330cbb90aa9150ca34f',
            '8e501ea1050e70fa9395b3b6ab47c28ce238685cf55c7af92b0f7474e85d1a93', 316, 255, 275),
    'test': ('737c35d135b283e50778d170d56d7e702a506a6c349aa68ce7bc847451e4a8e4',
             '95ec02d40f4bdc4483bc8efd6c8623a312d5b4b045285aee909a392f8ec7cae7', 1000, 946, 965),
}


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def load_reference(split, directory, inputs):
    report_sha, capture_sha, count, top1, top5 = REFERENCE[split]
    if sha(directory / 'report.json') != report_sha or sha(directory / 'capture.jsonl') != capture_sha:
        raise ValueError('frozen parent reference identity differs')
    report = json.loads((directory / 'report.json').read_text())
    if (report.get('completed') is not True or report.get('split') != split
            or report.get('input_contract') != 'roman_sentence_only'
            or report.get('capture_sha256') != capture_sha
            or report.get('mixed', {}).get('counts') != dict(rows=count, top1=top1, top5=top5, errors=0)):
        raise ValueError('frozen parent reference contract differs')
    references = read_rows(directory / 'capture.jsonl')
    if len(references) != count or len(inputs) != count:
        raise ValueError('parent reference denominator differs')
    for reference, item in zip(references, inputs):
        if (reference.get('id') != item['id'] or reference.get('request') != dict(input=item['input'], context='')):
            raise ValueError('parent reference source or identity differs')
    return references


def verify_response_identity(before, after, parent, proposed):
    info = after.get('ctc_word', {})
    if (before.get('version') != parent.version or after.get('version') != proposed.version
            or info.get('parent_version') != parent.version
            or info.get('artifacts') != proposed.ctc_word_artifacts):
        raise ValueError('runtime response model identity differs')


def verify_runtime_contract(source, before, after):
    candidates = after.get('candidates')
    if (not isinstance(candidates, list) or not 1 <= len(candidates) <= 5
            or after.get('result') != candidates[0].get('text')
            or source not in [candidate.get('text') for candidate in candidates]):
        raise ValueError('runtime candidates omit the original source or selected result')
    for candidate in candidates:
        cursor = 0
        pieces = []
        for segment in candidate.get('segments', []):
            end = segment.get('input_end')
            if (segment.get('input_start') != cursor or not isinstance(end, int)
                    or not cursor < end <= len(source)
                    or segment.get('source') != source[cursor:end]
                    or not isinstance(segment.get('text'), str)):
                raise ValueError('runtime candidate source coverage differs')
            pieces.append(segment['text'])
            cursor = end
        if cursor != len(source) or ''.join(pieces) != candidate.get('text'):
            raise ValueError('runtime candidate text/source coverage differs')
        if candidate.get('score') is None:
            expected = dict(input_start=0, input_end=len(source), source=source,
                            text=source, kind='raw')
            if candidate.get('text') != source or candidate.get('segments') != [expected]:
                raise ValueError('runtime original raw candidate differs')
    if source not in [candidate['text'] for candidate in candidates[:4]]:
        if candidates[-1]['text'] != source:
            raise ValueError('runtime original candidate must occupy the final slot')
    info = after.get('ctc_word', {})
    names = ('generation_requests', 'completed_generations',
             'generated_proposals', 'accepted_proposals')
    values = [info.get(name) for name in names]
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in values):
        raise ValueError('invalid CTC runtime counters')
    requested, complete, generated, accepted = values
    ranked = before.get('sentence_ranking_applied') is True
    if after.get('sentence_ranking_applied') != before.get('sentence_ranking_applied'):
        raise ValueError('CTC runtime parent ranking gate differs')
    reason = ('unsupported_source' if len(source.split()) < 2 or re.search(r'[가-힣]', source)
              else 'source_token_limit' if len(source.encode('utf-8')) + 2 > 512 else None)
    eligible = ranked and reason is None
    prediction = info.get('prediction')
    if requested != int(eligible) or not 0 <= accepted <= generated <= 96:
        raise ValueError('CTC runtime generation count or proposal bounds differ')
    if eligible:
        if not isinstance(prediction, dict) or prediction.get('reason') not in (
                'complete', 'empty_output', 'invalid_utf8', 'invalid_jamo', 'no_valid_path'):
            raise ValueError('CTC runtime prediction is absent or invalid')
        if complete != int(prediction['reason'] == 'complete'):
            raise ValueError('CTC runtime completion count differs')
    elif complete or generated or accepted or (
            ranked and (not isinstance(prediction, dict) or prediction.get('reason') != reason)) or (
            not ranked and prediction is not None):
        raise ValueError('CTC runtime ineligible generation differs')
    if not complete and (generated or accepted):
        raise ValueError('incomplete CTC output supplied proposals')
    elapsed = info.get('generation_ms')
    if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError('invalid CTC generation time')


def compare(captures, labels):
    summaries = {
        name: summarize([dict(id=row['id'], response=row[name]) for row in captures], labels)
        for name in ('parent', 'proposed')
    }
    changes = {name: [] for name in ('corrections', 'regressions', 'top5_losses')}
    for before, after in zip(summaries['parent']['outcomes'], summaries['proposed']['outcomes']):
        for name, condition in (
            ('corrections', not before['top1'] and after['top1']),
            ('regressions', before['top1'] and not after['top1']),
            ('top5_losses', before['top5'] and not after['top5']),
        ):
            if condition:
                changes[name].append(before['id'])
    return dict(**summaries, **changes,
                improvement_gate=bool(changes['corrections']) and not changes['regressions']
                and not changes['top5_losses'])


def latency(captures):
    result = {}
    for name in ('parent', 'proposed'):
        values = sorted(row[name]['engine_ms'] for row in captures)
        if not values or any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError('invalid measured timings')
        result[name] = {key: values[max(0, math.ceil(q * len(values)) - 1)]
                        for key, q in [('p50_ms', .5), ('p95_ms', .95), ('max_ms', 1.)]}
    return result


def verify_dev_qualification(directory):
    if directory is None:
        raise ValueError('test evaluation requires a completed DEV qualification')
    report_path = directory / 'report.json'
    report = json.loads(report_path.read_text())
    if (report.get('completed') is not True or report.get('split') != 'dev'
            or report.get('input_contract') != 'roman_sentence_only'
            or report.get('improvement_gate') is not True
            or report.get('capture_sha256') != sha(directory / 'capture.jsonl')
            or report.get('gold_not_passed_to_predictor') is not True
            or report.get('proposed', {}).get('counts', {}).get('rows') != 316):
        raise ValueError('DEV runtime qualification failed')
    if not report.get('pins'):
        raise ValueError('DEV runtime qualification has no model/source pins')
    for path, digest in report['pins'].items():
        if sha(Path(path)) != digest:
            raise ValueError('DEV qualification model/source identity changed')
    return dict(report=str(report_path), report_sha256=sha(report_path),
                capture_sha256=report['capture_sha256'])


def run(split, output, final=False, qualification=None):
    if split == 'test' and not final:
        raise ValueError('test evaluation requires a frozen configuration and --final')
    qualification_identity = verify_dev_qualification(qualification) if split == 'test' else None
    import torch
    from hanglish.core.mixed_span_converter import MixedSpanConverter
    from hanglish.core.ctc_word_converter import CTCWordConverter
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    corpus = ROOT / 'data/challenge/sentence-v3'
    inputs, identity = read_inputs(corpus / (split + '.jsonl'), corpus / 'manifest.json')
    output.mkdir(parents=True, exist_ok=False)
    reference_dir = ROOT / ('.omx/artifacts/mixed-span-runtime-' + split + '-v1')
    references = load_reference(split, reference_dir, inputs)
    tracked = [Path(__file__), corpus / (split + '.jsonl'), corpus / 'manifest.json',
               reference_dir / 'report.json', reference_dir / 'capture.jsonl']
    tracked += [ROOT / p for p in (
        'hanglish/core/ctc_word_converter.py', 'hanglish/core/ctc_word_proposals.py',
        'hanglish/core/neural_converter.py', 'hanglish/core/mixed_span_converter.py',
        'hanglish/core/mixed_span_proposals.py', 'hanglish/models/sentence_transducer.py', 'hanglish/models/sentence_ctc.py',
        'scripts/evaluate_sentence_only.py', 'scripts/evaluate_ensemble_regression.py',
    )]
    started = time.monotonic()
    parent = MixedSpanConverter()
    parent.load_models()
    proposed = CTCWordConverter(external_lexicon=False)
    for name in ('generator', 'numeric_units', 'ranker', 'sentence_ranker', 'transducer'):
        setattr(proposed, name, getattr(parent, name))
    proposed.load_models()
    proposed.load_word_model()
    tracked += [Path(p) if Path(p).is_absolute() else ROOT / p
                for p in proposed.model_artifacts]
    tracked += [ROOT / p for p in proposed.ctc_word_artifacts]
    pins = {str(p): sha(p) for p in tracked}
    load_elapsed_s = time.monotonic() - started
    captures = []
    reasons = Counter()
    with (output / 'capture.jsonl').open('x') as stream:
        for item, reference in zip(inputs, references):
            before = predict_sentence(parent, item['input'])
            after = predict_sentence(proposed, item['input'])
            verify_response_identity(before, after, parent, proposed)
            verify_runtime_contract(item['input'], before, after)
            if (before['result'] != reference['mixed']['result']
                    or before['candidates'] != reference['mixed']['candidates']):
                raise ValueError('parent runtime has drifted from frozen evaluation')
            record = dict(id=item['id'], request=dict(input=item['input'], context=''),
                          parent=before, proposed=after)
            captures.append(record)
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            stream.flush()
            info = after.get('ctc_word', {})
            prediction = info.get('prediction') or {}
            reasons[str(prediction.get('reason', info.get('reason', 'not_requested')))] += 1
            elapsed = time.monotonic() - started
            if elapsed > 1800 or rss_bytes() > 3 * 1024**3:
                raise RuntimeError('runtime evaluation exceeded 1800seconds/3GiB')
            if len(captures) % 25 == 0:
                print(json.dumps(dict(captured=len(captures), total=len(inputs),
                                      elapsed_s=elapsed, peak_rss_bytes=rss_bytes())), flush=True)
    capture_sha = sha(output / 'capture.jsonl')
    # read_inputs validated the corpus (including label schema) before inference;
    # only source strings crossed the predictor boundary. Compute label metrics
    # after capture, including runtime's mandatory original-text option.
    labels = read_rows(corpus / (split + '.jsonl'))
    metrics = compare(captures, labels)
    for path, digest in pins.items():
        if sha(Path(path)) != digest:
            raise ValueError('a pinned model, corpus, or implementation changed')
    report = dict(completed=True, split=split, input_contract='roman_sentence_only',
                  dev_qualification=qualification_identity,
                  **identity, **metrics, pins=pins, capture_sha256=capture_sha,
                  parent_version=parent.version, proposed_version=proposed.version,
                  parent_frozen_parity_checked=True, single_request_generation=True,
                  gold_not_passed_to_predictor=True, labels_opened_after_capture=False,
                  timing_scope='paired warm parent/child; immutable models and generator shared; child includes fresh whole-sentence generation',
                  latency=latency(captures), generation_reasons=dict(reasons),
                  load_elapsed_s=load_elapsed_s, elapsed_s=time.monotonic() - started,
                  peak_rss_bytes=rss_bytes(), resource_limit_s=1800,
                  resource_limit_bytes=3 * 1024**3, deployed=False,
                  test_accessed=split == 'test', fully_blind_generalization=False)
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in (
        'completed', 'split', 'corrections', 'regressions', 'top5_losses',
        'improvement_gate', 'latency', 'elapsed_s', 'peak_rss_bytes')}, ensure_ascii=False), flush=True)
    print(json.dumps({name: report[name]['counts'] for name in ('parent', 'proposed')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['dev', 'test'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--final', action='store_true')
    parser.add_argument('--qualification', type=Path)
    args = parser.parse_args()
    try:
        run(args.split, args.output, args.final, args.qualification)
    except Exception as error:
        # Preserve partial captures rather than treating interrupted work as a
        # completed evaluation or silently reusing its output directory.
        if args.output.is_dir() and not (args.output / 'report.json').exists():
            (args.output / 'failure.json').write_text(json.dumps(dict(
                completed=False, error=repr(error)), indent=2) + '\n')
        raise
