"""Fresh train-only paired runtime probe; never reuse historical word features."""
import json
from functools import lru_cache
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from hanglish.core.lattice import WordOption
from hanglish.core.neural_converter import NeuralConverter, file_sha256
from hanglish.core.phonology import romanize_word
from scripts.constrained_phonetic_search import generate_constrained
from scripts.prepare_context_training import load_sources, source_hashes, mixed_split_guard
from scripts.prepare_natural_ranker import source_fields
from scripts.evaluate_roberta_context_ranker import SentencePoolRecorder
from scripts.train_natural_sentence_ranker import rss_bytes

VERSION = 'neural-1-f97b831f0fdb'
OUTPUT = ROOT / '.omx/artifacts/constrained-runtime-v1'
POLICY = {'additional_candidates': 4, 'beam_size': 8, 'threads': 2,
          'scope': 'all fixed natural train512; source-only proposals',
          'strong_option_threshold': 5.5, 'preserve_parent_prefix': True}


def extend_options(converter, source, learned, baseline, generate=generate_constrained):
    if (not learned or converter.transducer is None or not source.isascii()
            or not source.isalpha() or not source.islower() or not 2 <= len(source) <= 40
            or any(o.score >= POLICY['strong_option_threshold'] for o in baseline)):
        return baseline
    with converter.lock:
        generated = generate(converter.transducer, source, beam_size=POLICY['beam_size'])
    seen = {option.text for option in baseline}
    additions = []
    for text, score in generated:
        if text in seen:
            continue
        if not text or romanize_word(text) != source:
            raise ValueError('constrained generator emitted an invalid spelling')
        seen.add(text)
        additions.append(WordOption(text, 4. + score / len(text), 'neural'))
        if len(additions) == POLICY['additional_candidates']:
            break
    return (*baseline, *additions)


def install(converter):
    audit = {}
    def options(source, learned):
        baseline = converter._word_options(source, learned)
        result = extend_options(converter, source, learned, baseline)
        assert tuple(result[:len(baseline)]) == tuple(baseline)
        audit[(source, learned)] = len(result) - len(baseline)
        return result
    converter.word_options = lru_cache(maxsize=1024)(options)
    return audit


def hashes():
    result = source_hashes()
    for relative in ('scripts/probe_constrained_runtime.py',
                     'scripts/constrained_phonetic_search.py',
                     'scripts/evaluate_roberta_context_ranker.py',
                     'scripts/train_natural_sentence_ranker.py'):
        result[relative] = file_sha256(ROOT / relative)
    return result


def metrics(cases):
    result = {'rows': len(cases)}
    for side in ('before', 'new'):
        for key in ('top1', 'top5', 'covered'):
            result[f'{side}_{key}'] = sum(c[f'{side}_{key}'] is True for c in cases)
    result['corrections'] = [c['id'] for c in cases if c['new_top1'] and not c['before_top1']]
    result['regressions'] = [c['id'] for c in cases if c['before_top1'] and not c['new_top1']]
    return result


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    raw, guard = load_sources()
    sources = {role: [source_fields(row, role) for row in rows] for role, rows in raw.items()}
    split_guard = mixed_split_guard(sources)
    rows = sources['natural_train']
    assert len(rows) == 512
    before = hashes()
    torch.set_num_threads(2)
    converter = NeuralConverter(ranker_kind='hybrid')
    converter.load_models()
    assert converter.version == VERSION
    parent = converter.word_options
    audit = install(converter)
    changed = converter.word_options
    recorder = SentencePoolRecorder(converter.sentence_ranker)
    converter.sentence_ranker = recorder
    OUTPUT.mkdir(parents=True)
    (OUTPUT/'selection.json').write_text(json.dumps(rows, ensure_ascii=False)+'\n')
    (OUTPUT/'source.py').write_bytes(Path(__file__).read_bytes())
    cases, started = [], time.monotonic()
    with (OUTPUT/'cases.jsonl').open('x') as stream:
        for index, row in enumerate(rows):
            case = {k: row[k] for k in ('id', 'input', 'context', 'expected')}
            try:
                for side, options in (('before', parent), ('new', changed)):
                    converter.word_options = options
                    recorder.candidates = None
                    response = converter.convert(row['input'], row['context'])
                    candidates = [c['text'] for c in response['candidates']]
                    pool = recorder.candidates
                    case.update({f'{side}_candidates': candidates,
                                 f'{side}_pool': pool,
                                 f'{side}_top1': candidates[0] in row['expected'],
                                 f'{side}_top5': any(c in row['expected'] for c in candidates),
                                 f'{side}_covered': None if pool is None else any(c in row['expected'] for c in pool),
                                 f'{side}_engine_ms': response['engine_ms']})
            finally:
                converter.word_options = parent
            cases.append(case)
            stream.write(json.dumps(case, ensure_ascii=False)+'\n')
            stream.flush()
            if rss_bytes() > 3*1024**3 or time.monotonic()-started > 1800:
                raise RuntimeError('probe resource bound exceeded')
            if (index+1) % 32 == 0:
                print(json.dumps({'rows': index+1, 'seconds': round(time.monotonic()-started, 1)}), flush=True)
    if before != hashes() or converter.version != VERSION:
        raise ValueError('runtime/source/model changed during probe')
    report = {'policy': POLICY, 'parent_version': VERSION, 'source_hashes': before,
              'challenge_guard': guard, 'split_guard': split_guard,
              'selection_sha256': file_sha256(OUTPUT/'selection.json'),
              'cases_sha256': file_sha256(OUTPUT/'cases.jsonl'), 'metrics': metrics(cases),
              'audit': {'calls': len(audit), 'changed': sum(n > 0 for n in audit.values()),
                        'added': sum(audit.values()), 'max_added': max(audit.values(), default=0)},
              'seconds': time.monotonic()-started, 'peak_rss_bytes': rss_bytes(),
              'test_data_read': False, 'development_inference_run': False, 'deployed': False,
              'timing_policy': 'Shared model/generator caches; descriptive, not an isolated speed comparison'}
    (OUTPUT/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'metrics': report['metrics'], 'audit': report['audit']}), flush=True)


if __name__ == '__main__':
    main()
