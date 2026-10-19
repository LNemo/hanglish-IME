"""Source-only training probe for bounded sentence proposal search; never deploys."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hanglish.core.neural_converter import NeuralConverter, beam_paths, file_sha256
from scripts.prepare_natural_ranker import OUTPUT, validate_prepared
from scripts.prepare_natural_word_ranker import source_hashes
from scripts.train_natural_sentence_ranker import rss_bytes

MANIFEST = '92a37bbc699af4a4981bbe9a29f60c48d71c9901b5113a230791aa7e9f769503'
TRAIN_CASES = 64
VARIANTS = ('baseline48', 'wider96', 'diverse48', 'augment96')


def wider_beam(pieces, width):
    paths = [('', 0., [])]
    for source, offset, options in pieces:
        expanded = []
        for prefix, score, segments in paths:
            for option in options[:8]:
                segment = {'input_start': offset, 'input_end': offset+len(source),
                           'source': source, 'text': option.text, 'kind': option.kind}
                expanded.append((prefix+option.text, score+option.score, segments+[segment]))
        unique = {}
        for path in sorted(expanded, key=lambda p: (-p[1], p[0])):
            unique.setdefault(path[0], path)
        paths = list(unique.values())[:width]
    return paths


def single_alternatives(pieces):
    """One-position alternatives to the first path, ordered without labels."""
    if not pieces:
        return []
    alternatives = []
    for position, (_, _, options) in enumerate(pieces):
        for rank, option in enumerate(options[1:8], 1):
            if option.kind == 'raw' and option.score < 6:
                continue
            selected = [choices[0] for _, _, choices in pieces]
            selected[position] = option
            segments = [{'input_start': offset, 'input_end': offset+len(source),
                         'source': source, 'text': choice.text, 'kind': choice.kind}
                        for (source, offset, _), choice in zip(pieces, selected)]
            path = (''.join(choice.text for choice in selected), sum(choice.score for choice in selected), segments)
            alternatives.append((rank, options[0].score-option.score, path[0], path))
    return [item[-1] for item in sorted(alternatives, key=lambda item: item[:3])]


def unique_paths(paths, limit):
    unique = {}
    for path in paths:
        unique.setdefault(path[0], path)
        if len(unique) == limit:
            break
    return list(unique.values())


def proposal_variants(pieces, static):
    words, lexical = beam_paths(pieces), beam_paths(static)
    baseline = unique_paths(words+lexical, 48)
    alternatives = single_alternatives(pieces)+single_alternatives(static)
    return {'baseline48': baseline,
            'wider96': unique_paths(wider_beam(pieces, 48)+wider_beam(static, 48), 96),
            'diverse48': unique_paths(words+lexical[:12]+alternatives, 48),
            'augment96': unique_paths(baseline+alternatives, 96)}


def can_compose(target, pieces, option_limit=None):
    """Exact finite-lattice reachability, with no target used during generation."""
    positions = {0}
    for _, _, options in pieces:
        options = options if option_limit is None else options[:option_limit]
        positions = {position+len(option.text) for position in positions for option in options
                     if target.startswith(option.text, position)}
        if not positions:
            return False
    return len(target) in positions


def classify(targets, pieces, static, baseline):
    if any(p[0] in targets for p in baseline):
        return 'present'
    if not any(can_compose(target, pieces) for target in targets):
        return 'missing_piece_candidate'
    if not any(can_compose(target, pieces, 8) or can_compose(target, static, 8) for target in targets):
        return 'option_cap'
    return 'beam_pruning'


def summarize(outcomes):
    return {'cases': len(outcomes), 'covered': sum(r['covered'] for r in outcomes),
            'top1': sum(r['top1'] for r in outcomes), 'fallbacks': sum(r['fallback'] for r in outcomes),
            'errors': sum(r['error'] is not None for r in outcomes),
            'mean_candidates': sum(r['candidate_count'] for r in outcomes)/len(outcomes),
            'mean_ranking_ms': sum(r['ranking_ms'] for r in outcomes)/len(outcomes)}


def probe(output):
    if output.exists(): raise FileExistsError(output)
    _, prepared = validate_prepared(OUTPUT, MANIFEST)
    # Existing source-only hash order, selected before any new inference/results.
    sources = prepared['natural_train'][:TRAIN_CASES]
    before = source_hashes(); before[str(Path(__file__).relative_to(ROOT))] = file_sha256(Path(__file__))
    converter = NeuralConverter(ranker_kind='hybrid'); converter.load_models()
    output.mkdir(parents=True)
    selection = {'scope': 'natural training only', 'source_manifest_sha256': MANIFEST,
                 'source_ids': [r['id'] for r in sources], 'variants': VARIANTS,
                 'runtime_version': converter.version, 'source_hashes': before,
                 'dev_rows_used': False, 'test_data_read': False}
    (output / 'selection.json').write_text(json.dumps(selection, ensure_ascii=False, indent=2)+'\n')
    outcomes = {name: [] for name in VARIANTS}; diagnoses = []; started = time.monotonic()
    with (output / 'cases.jsonl').open('x') as stream:
        for index, row in enumerate(sources):
            pieces, static, *_ = converter._prepare_candidates(row['input'], row['context'], learned=True, raw=False)
            pools = proposal_variants(pieces, static)
            # Full runtime pool parity is checked before target-based diagnostics.
            if [p[0] for p in pools['baseline48']] != row['candidates']:
                raise ValueError('baseline differs from previously pinned runtime candidates')
            diagnosis = classify(row['expected'], pieces, static, pools['baseline48'])
            diagnoses.append({'id': row['id'], 'reason': diagnosis})
            record = {key:row[key] for key in ('id', 'input', 'context', 'expected')}
            record.update(diagnosis=diagnosis, pools={})
            for name, paths in pools.items():
                can_rank = len(row['input'].split()) > 1 and len(paths) > 1 and all(converter.sentence_ranker.can_score(p[0]) for p in paths)
                rank_start = time.monotonic()
                error = None
                try:
                    if can_rank:
                        scores = converter.sentence_ranker.score_pairs([row['context']]*len(paths), [p[0] for p in paths])
                        chosen = min(range(len(paths)), key=lambda i: (-scores[i], paths[i][0]))
                        result = paths[chosen][0]
                    else:
                        result = beam_paths(pieces)[0][0]
                except Exception as exc:
                    # An inference error is a failed case, never a raw-input success.
                    result = None
                    error = type(exc).__name__ + ': ' + str(exc)
                item = {'id': row['id'], 'covered': any(p[0] in row['expected'] for p in paths),
                        'top1': result in row['expected'], 'result': result,
                        'fallback': not can_rank, 'error': error, 'candidate_count': len(paths),
                        'ranking_ms': 1000*(time.monotonic()-rank_start)}
                outcomes[name].append(item)
                record['pools'][name] = {'paths': [p[0] for p in paths], **item}
            record['pieces'] = [[source, offset, [asdict(o) for o in options]] for source,offset,options in pieces]
            record['static_pieces'] = [[source, offset, [asdict(o) for o in options]] for source,offset,options in static]
            stream.write(json.dumps(record, ensure_ascii=False)+'\n'); stream.flush()
            if rss_bytes() > 3*1024**3 or time.monotonic()-started > 1800:
                raise RuntimeError('proposal probe resource cap exceeded')
            if (index+1) % 16 == 0:
                print(json.dumps({'cases': index+1, 'elapsed_seconds': time.monotonic()-started}), flush=True)
    after = source_hashes(); after[str(Path(__file__).relative_to(ROOT))] = file_sha256(Path(__file__))
    if before != after: raise ValueError('proposal sources changed')
    summaries = {name:summarize(rows) for name,rows in outcomes.items()}
    old = {r['id']:r for r in outcomes['baseline48']}
    for name in summaries:
        summaries[name].update(corrections=[r['id'] for r in outcomes[name] if r['top1'] and not old[r['id']]['top1']],
                               regressions=[r['id'] for r in outcomes[name] if not r['top1'] and old[r['id']]['top1']])
    report = {'selection_sha256': file_sha256(output/'selection.json'),
              'cases_sha256': file_sha256(output/'cases.jsonl'), 'summaries': summaries,
              'diagnoses': {key:sum(d['reason']==key for d in diagnoses) for key in ('present','missing_piece_candidate','option_cap','beam_pruning')},
              'case_diagnoses': diagnoses, 'peak_rss_bytes': rss_bytes(),
              'elapsed_seconds': time.monotonic()-started, 'deployed': False,
              'next_gate': 'Choose on training evidence, then natural dev128 and original dev140/60/40 before final regression.'}
    (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'summaries': summaries, 'diagnoses': report['diagnoses']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'.omx/artifacts/proposal-search-v1')
    probe(parser.parse_args().output)
