"""Train-only diagnosis of missing spellings in the current complete piece lattice."""
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hanglish.core.neural_converter import NeuralConverter, file_sha256
from hanglish.core.phonology import romanize_word
from scripts.prepare_context_training import OUTPUT, source_hashes, validate_context_data
from scripts.prepare_natural_word_ranker import SPAN, capture_groups
from scripts.probe_proposal_search import can_compose
from scripts.train_natural_sentence_ranker import rss_bytes

MANIFEST = '7c1c03205dbc9c8983c2b3dee37e4a6bcabb04d45a265e9e1699460595275b10'
DESTINATION = ROOT / '.omx/artifacts/missing-candidate-audit-v1'


def aligned_target(expected, start, end):
    offset, segments = 0, []
    for match in SPAN.finditer(expected):
        roman = romanize_word(match[0])
        segments.append((offset, offset + len(roman), match[0]))
        offset += len(roman)
    selected = [s for s in segments if start <= s[0] and s[1] <= end]
    if selected and selected[0][0] == start and selected[-1][1] == end:
        return ''.join(s[2] for s in selected)
    return None


def hashes():
    values = source_hashes()
    for name in ('scripts/audit_missing_candidates.py', 'scripts/probe_proposal_search.py',
                 'scripts/prepare_natural_word_ranker.py', 'scripts/train_natural_sentence_ranker.py'):
        values[name] = file_sha256(ROOT / name)
    return values


def main():
    if DESTINATION.exists():
        raise FileExistsError(DESTINATION)
    _, data = validate_context_data(OUTPUT, MANIFEST)
    sources = data['natural_train']
    before = hashes()
    converter = NeuralConverter(context_ranking=False, ranker_kind='hybrid')
    converter.load_models()
    DESTINATION.mkdir(parents=True)
    counts, missing_counts, outcomes = Counter(), Counter(), []
    started = time.monotonic()
    with (DESTINATION / 'cases.jsonl').open('x') as stream:
        for index, row in enumerate(sources):
            # All candidate options are produced from source alone. Disabling word
            # scoring changes only ordering/scores, not static candidate membership.
            _, pieces, *_ = converter._prepare_candidates(row['input'], row['context'], learned=True, raw=False)
            records = [(g['context'], c, s) for g in row['groups']
                       for c, s in zip(g['candidates'], g['teacher_scores'])]
            captured = capture_groups(pieces, records, row['context'])
            keys = ('source', 'start', 'end', 'context', 'candidates', 'base_scores', 'kinds', 'teacher_scores')
            if [{k:g[k] for k in keys} for g in captured] != [{k:g[k] for k in keys} for g in row['groups']]:
                raise ValueError('static pieces do not reproduce frozen word features')
            covered = any(c in row['expected'] for c in row['candidates'])
            composable = any(can_compose(target, pieces) for target in row['expected'])
            reason = 'covered' if covered else ('combination_lost' if composable else 'missing_piece')
            counts[reason] += 1
            missing = []
            if not composable:
                for source, offset, options in pieces:
                    targets = {aligned_target(t, offset, offset + len(source)) for t in row['expected']}
                    targets.discard(None)
                    if not targets:
                        if any(c.isalpha() for c in source):
                            missing.append({'source':source,'offset':offset,'expected':[],
                                            'reason':'unaligned_boundary','options':[asdict(o) for o in options]})
                        continue
                    if any(o.text in targets for o in options):
                        continue
                    raw_only = all(o.kind == 'raw' for o in options)
                    generator = list(converter.generator.options(source)) if source.isascii() and source.isalpha() else []
                    base_has = any(o.text in targets for o in generator)
                    kind = ('runtime_filter' if base_has else 'protected_raw' if raw_only else
                            'missing_with_lexical' if any(o.kind == 'lexicon' for o in options) else
                            'missing_with_morphology' if any(o.kind == 'morphology' for o in options) else
                            'missing_generated' if any(o.kind == 'neural' for o in options) else 'missing_other')
                    missing.append({'source':source,'offset':offset,'expected':sorted(targets),'reason':kind,
                                    'options':[asdict(o) for o in options],
                                    'generator_options':[asdict(o) for o in generator]})
                missing_counts.update(m['reason'] for m in missing)
            result = {k:row[k] for k in ('id','input','context','expected')}
            result.update(reason=reason, missing=missing)
            outcomes.append(result)
            stream.write(json.dumps(result, ensure_ascii=False) + '\n')
            stream.flush()
            if rss_bytes() > 3*1024**3 or time.monotonic()-started > 1800:
                raise RuntimeError('audit resource limit exceeded')
            if (index+1)%64 == 0:
                print(json.dumps({'cases':index+1,'seconds':round(time.monotonic()-started,2)}),flush=True)
    validate_context_data(OUTPUT, MANIFEST)
    if before != hashes():
        raise ValueError('audit sources changed')
    report = {'scope':'natural training512 only; no dev/final inference', 'test_data_read':False,
              'counts':dict(counts),'missing_piece_counts':dict(missing_counts),
              'source_manifest_sha256':MANIFEST,'source_hashes':before,
              'source_ids':[r['id'] for r in sources],
              'cases_sha256':file_sha256(DESTINATION/'cases.jsonl'),
              'seconds':time.monotonic()-started,'peak_rss_bytes':rss_bytes(),
              'runtime_version':converter.version,'deployed':False}
    (DESTINATION/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (DESTINATION/'audit-source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report | {'source_hashes':'omitted','source_ids':'omitted'},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
