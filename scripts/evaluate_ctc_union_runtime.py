"""Paired whole-source evaluation of CTC1 plus growth candidates."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_ctc_word_runtime import (
    compare, latency, read_rows, verify_dev_qualification, verify_runtime_contract,
)
from scripts.evaluate_sentence_only import read_inputs, predict_sentence, sha
from scripts.evaluate_ensemble_regression import rss_bytes

REFERENCE = {
    'dev': ('f59c35b8829154961d41a45057651d674814e3155da5bcf68b3a307882664c9b',
            'c764896e0da4006f61310293738c89ed2a4ea591579d24bd13c5ec0a0d971d64',
            316, 260, 280),
    'test': ('120e4c0186a6e62a217e2b7394f9c40aafc09c8cf439472a596a754a7ede2077',
             '93b6ce723097f2fb8a2cd65eb8629ae366030fa684d03dd481256db7bf0c57f4',
             1000, 951, 970),
}


def load_reference(split, directory, inputs):
    report_sha, capture_sha, count, top1, top5 = REFERENCE[split]
    if (sha(directory / 'report.json') != report_sha
            or sha(directory / 'capture.jsonl') != capture_sha):
        raise ValueError('frozen CTC1 reference identity differs')
    report = json.loads((directory / 'report.json').read_text())
    if (report.get('completed') is not True or report.get('split') != split
            or report.get('input_contract') != 'roman_sentence_only'
            or report.get('capture_sha256') != capture_sha
            or report.get('proposed', {}).get('counts') != dict(
                rows=count, top1=top1, top5=top5, errors=0)):
        raise ValueError('frozen CTC1 reference contract differs')
    references = read_rows(directory / 'capture.jsonl')
    if len(references) != count or len(inputs) != count:
        raise ValueError('CTC1 reference denominator differs')
    for reference, item in zip(references, inputs):
        if (reference.get('id') != item['id']
                or reference.get('request') != dict(input=item['input'], context='')):
            raise ValueError('CTC1 reference source identity differs')
    return references


def verify_identity(before, after, parent, proposed):
    info = after.get('ctc_union', {})
    required = {'experimental', 'generation_requests', 'completed_generations',
                'generated_proposals', 'accepted_proposals', 'prediction',
                'generation_ms', 'parent_version', 'artifacts'}
    if (set(info) != required or info.get('experimental') is not True
            or before.get('version') != parent.version or after.get('version') != proposed.version
            or info.get('parent_version') != parent.version
            or info.get('artifacts') != proposed.growth_artifacts):
        raise ValueError('union runtime identity differs')
    # CTC1 still runs once, and its output/provenance must remain unchanged.
    original = before.get('ctc_word', {})
    retained = after.get('ctc_word', {})
    for key in set(original) | set(retained):
        if key != 'generation_ms' and original.get(key) != retained.get(key):
            raise ValueError('union changed the original CTC1 stage')


def verify_union_contract(source, before, after):
    verify_runtime_contract(source, before, after)
    growth = dict(after, ctc_word=after.get('ctc_union', {}))
    verify_runtime_contract(source, before, growth)


def qualify(directory):
    identity = verify_dev_qualification(directory)
    report = json.loads((directory / 'report.json').read_text())
    if (report.get('experiment') != 'ctc_growth_union_v1'
            or report.get('parent', {}).get('counts') != dict(
                rows=316, top1=260, top5=280, errors=0)
            or report['pins'].get(str(Path(__file__).resolve())) != sha(Path(__file__))):
        raise ValueError('DEV qualification is not the current union runtime')
    return identity


def run(split, output, final=False, qualification=None):
    if split == 'test' and not final:
        raise ValueError('test requires --final and a frozen DEV-qualified configuration')
    qualification_identity = qualify(qualification) if split == 'test' else None
    import torch
    from hanglish.core.ctc_word_converter import CTCWordConverter
    from hanglish.core.ctc_union_converter import CTCUnionConverter
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    corpus = ROOT / 'data/challenge/sentence-v3'
    inputs, identity = read_inputs(corpus / (split + '.jsonl'), corpus / 'manifest.json')
    reference_dir = ROOT / ('.omx/artifacts/ctc-word-runtime-' + split + '-v1')
    references = load_reference(split, reference_dir, inputs)
    output.mkdir(parents=True, exist_ok=False)
    tracked = [Path(__file__), corpus / (split + '.jsonl'), corpus / 'manifest.json',
               reference_dir / 'report.json', reference_dir / 'capture.jsonl']
    tracked += [ROOT / p for p in (
        'scripts/evaluate_ctc_word_runtime.py', 'scripts/evaluate_sentence_only.py',
        'scripts/evaluate_ensemble_regression.py', 'hanglish/core/ctc_union_converter.py',
        'hanglish/core/ctc_word_converter.py', 'hanglish/core/ctc_word_proposals.py',
        'hanglish/core/neural_converter.py', 'hanglish/core/mixed_span_converter.py',
        'hanglish/core/mixed_span_proposals.py', 'hanglish/models/sentence_transducer.py',
        'hanglish/models/sentence_ctc.py',
    )]
    started = time.monotonic()
    parent = CTCWordConverter()
    parent.load_models()
    parent.load_word_model()
    proposed = CTCUnionConverter(external_lexicon=False, word_model=parent.word_model)
    for name in ('generator', 'numeric_units', 'ranker', 'sentence_ranker', 'transducer'):
        setattr(proposed, name, getattr(parent, name))
    proposed.load_models()
    proposed.load_growth_model()
    for artifacts in (parent.model_artifacts, proposed.model_artifacts,
                      proposed.ctc_word_artifacts, proposed.growth_artifacts):
        tracked += [Path(p) if Path(p).is_absolute() else ROOT / p for p in artifacts]
    pins = {str(p): sha(p) for p in tracked}
    load_elapsed_s = time.monotonic() - started
    captures = []
    with (output / 'capture.jsonl').open('x') as stream:
        for item, reference in zip(inputs, references):
            before = predict_sentence(parent, item['input'])
            after = predict_sentence(proposed, item['input'])
            verify_identity(before, after, parent, proposed)
            verify_union_contract(item['input'], before, after)
            if (before['result'] != reference['proposed']['result']
                    or before['candidates'] != reference['proposed']['candidates']):
                raise ValueError('parent CTC1 differs from the frozen runtime')
            record = dict(id=item['id'], request=dict(input=item['input'], context=''),
                          parent=before, proposed=after)
            captures.append(record)
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            stream.flush()
            elapsed = time.monotonic() - started
            if elapsed > 1800 or rss_bytes() > 3 * 1024**3:
                raise RuntimeError('union runtime exceeded 1800 seconds/3 GiB')
            if len(captures) % 25 == 0:
                print(json.dumps(dict(captured=len(captures), total=len(inputs),
                                      elapsed_s=elapsed, peak_rss_bytes=rss_bytes())), flush=True)
        os.fsync(stream.fileno())
    capture_sha = sha(output / 'capture.jsonl')
    labels = read_rows(corpus / (split + '.jsonl'))
    metrics = compare(captures, labels)
    for path, digest in pins.items():
        if sha(Path(path)) != digest:
            raise ValueError('a pinned implementation/model/corpus changed')
    report = dict(completed=True, experiment='ctc_growth_union_v1', split=split,
                  input_contract='roman_sentence_only', dev_qualification=qualification_identity,
                  **identity, **metrics, pins=pins, capture_sha256=capture_sha,
                  parent_version=parent.version, proposed_version=proposed.version,
                  parent_frozen_parity_checked=True, single_request_generation=True,
                  gold_not_passed_to_predictor=True, labels_opened_after_capture=False,
                  timing_scope='paired warm engines sharing immutable models; both CTC outputs generated afresh',
                  latency=latency(captures), load_elapsed_s=load_elapsed_s,
                  elapsed_s=time.monotonic() - started, peak_rss_bytes=rss_bytes(),
                  resource_limit_s=1800, resource_limit_bytes=3 * 1024**3,
                  deployed=False, test_accessed=split == 'test', fully_blind_generalization=False)
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
    run(args.split, args.output, args.final, args.qualification)
