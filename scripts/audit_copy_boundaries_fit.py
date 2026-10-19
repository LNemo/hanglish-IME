"""Close a source-only COPY derivation inventory over the current FIT pool."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hanglish.core.source_correspondence import align
from scripts.audit_recombination_tokenization import digest, require
from scripts.train_character_edge_full_fit import ResourceGuard

INTAKE = ROOT / '.omx/artifacts/current-fit-recombination-selection-intake-v1.json'
INTAKE_SHA = '70e5c963c2f78ae15f36e2bededb67f65c12b8430b6f9eb359b7cf44648d17aa'
PLAN = ROOT / '.omx/plans/copy-boundary-fit-v1.md'
MAX_OUTPUT_BYTES = 256 * 1024**2


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def mixed(segment):
    return (segment['source'] != segment['text'] and any(
        'a' <= c <= 'z' or 'A' <= c <= 'Z' or '0' <= c <= '9' for c in segment['text']))


def segment_key(segment):
    return (segment['source'], segment['text'], segment['kind'])


def local_segment(key):
    source, text, kind = key
    return dict(input_start=0, input_end=len(source), source=source, text=text, kind=kind)


def shifted(children, start):
    return [dict(child, input_start=child['input_start'] + start,
                 input_end=child['input_end'] + start) for child in children]


def project_candidate(source, candidate, inventory, generator):
    from hanglish.core.copy_boundaries import refine_path
    before = align(source, candidate['text'], candidate['segments'])
    refined = []
    keys = []
    for segment in candidate['segments']:
        if mixed(segment):
            key = segment_key(segment)
            if key not in inventory:
                inventory[key] = refine_path(key[0], key[1], [local_segment(key)], generator=generator)
            refined.extend(shifted(inventory[key]['segments'], segment['input_start']))
            keys.append(key)
        else:
            refined.append(dict(segment))
    after = align(source, candidate['text'], refined)
    require(before.source == after.source and before.text == after.text,
            'refinement changed a candidate source or output')
    return refined, keys, {
        'candidates': 1,
        'refined_candidates': int(refined != candidate['segments']),
        'coarse_spans': len(before.spans), 'refined_spans': len(after.spans),
        'mixed_before': sum(s.unresolved_ascii_output for s in before.spans),
        'mixed_after': sum(s.unresolved_ascii_output for s in after.spans),
        'copy_before': sum(s.action == 'copy' for s in before.spans),
        'copy_after': sum(s.action == 'copy' for s in after.spans),
    }


def authenticate():
    require(digest(INTAKE) == INTAKE_SHA, 'FIT intake changed')
    intake = json.loads(INTAKE.read_text())
    pins = {str(INTAKE): INTAKE_SHA}
    for shard in intake['shards']:
        directory = Path(shard['directory'])
        for name, sha in shard['hashes'].items():
            path = directory / name
            require(digest(path) == sha, 'closed FIT artifact changed')
            pins[str(path)] = sha
        proof = json.loads((directory / 'independent-verification.json').read_text())
        require(proof['verified'] is True, 'FIT scoring proof missing')
    paths = [Path(__file__), PLAN, ROOT/'hanglish/core/copy_boundaries.py',
             ROOT/'hanglish/core/source_correspondence.py',
             ROOT/'scripts/train_character_edge_full_fit.py']
    paths.extend(sorted((ROOT/'hanglish/core').glob('*.py')))
    paths.extend(sorted(p for p in (ROOT/'hanglish/core/data').rglob('*') if p.is_file()))
    paths.extend(ROOT/'data'/name for name in ('morphology_lexicon.json',
        'workbench_lexicon.json', 'common_english.txt', 'it_terms.txt', 'challenge/train_lexicon.json'))
    paths.extend(sorted((ROOT/'data/external/mecab-ko-dic').glob('*.csv')))
    for path in paths:
        require(path.is_file(), 'runtime source missing: '+str(path))
        pins[str(path.resolve())] = digest(path)
    return intake, pins


def write_json(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    guard = ResourceGuard(output)
    try:
        intake, pins = authenticate()
        guard.observe('authenticated')
        from hanglish.core.lattice import CandidateGenerator
        generator = CandidateGenerator(external_lexicon=True)
        guard.observe('generator-loaded')
        inventory = {}
        occurrences = Counter()
        total = Counter(rows=0, no_pool_rows=0, old_candidates=0, novel_candidates=0,
                        unsupported_candidates=0)
        ordered_ids = []
        old_identity, new_identity = hashlib.sha256(), hashlib.sha256()
        bytes_written = 0
        with (output/'rows.jsonl').open('x') as target:
            for shard in intake['shards']:
                for line in (Path(shard['directory'])/'capture.jsonl').open():
                    row = json.loads(line)
                    require(row['id'].startswith('s3-train-') and row['request']['context'] == '',
                            'non-FIT or external context in source')
                    ordered_ids.append(row['id'])
                    source, scoring = row['request']['input'], row['scoring']
                    candidates = scoring['original'] + scoring['novel']
                    stats = Counter(rows=1, no_pool_rows=int(not scoring['ranked']),
                                    old_candidates=len(scoring['original']), novel_candidates=len(scoring['novel']),
                                    unsupported_candidates=sum(c['status']=='unsupported' for c in scoring['novel']))
                    for candidate in candidates:
                        frozen = canonical(candidate)
                        children, keys, counts = project_candidate(source, candidate, inventory, generator)
                        require(canonical(candidate) == frozen, 'input candidate mutated')
                        old_payload={k:v for k,v in candidate.items() if k!='segments'}
                        projected=dict(candidate,segments=children)
                        new_payload={k:v for k,v in projected.items() if k!='segments'}
                        old_identity.update((row['id']+'\0'+canonical(old_payload)+'\n').encode())
                        new_identity.update((row['id']+'\0'+canonical(new_payload)+'\n').encode())
                        require(''.join(s['text'] for s in children) == candidate['text'],
                                'refined text differs')
                        occurrences.update(keys); stats.update(counts)
                    total.update(stats)
                    encoded = canonical({'id':row['id'], 'counts':dict(stats)})+'\n'
                    bytes_written += len(encoded.encode())
                    require(bytes_written < MAX_OUTPUT_BYTES, 'audit output size exceeded')
                    target.write(encoded)
                    guard.observe('row-'+str(total['rows']))
                    if total['rows'] % 256 == 0:
                        print(json.dumps({'rows':total['rows'],'unique_mixed':len(inventory)}), flush=True)
        require(len(ordered_ids)==len(set(ordered_ids))==1642
                and total['old_candidates']==249177 and total['novel_candidates']==149885
                and total['unsupported_candidates']==31 and total['no_pool_rows']==13,
                'complete FIT denominator differs')
        require(old_identity.hexdigest() == new_identity.hexdigest(), 'candidate data changed')
        with (output/'inventory.jsonl').open('x') as target:
            for key in sorted(inventory):
                encoded = canonical({'key':list(key),'occurrences':occurrences[key],
                                     'refinement':inventory[key]})+'\n'
                bytes_written += len(encoded.encode())
                require(bytes_written < MAX_OUTPUT_BYTES, 'audit output size exceeded')
                target.write(encoded)
        for path,sha in pins.items(): require(digest(Path(path))==sha,'source changed during audit')
        guard.observe('closed')
        report = {'completed':True,'rows':1642,'counts':dict(total),'unique_mixed_segments':len(inventory),
                  'ordered_ids':ordered_ids,'pins':pins,
                  'candidate_nonalignment_identity_sha256':old_identity.hexdigest(),
                  'candidate_records_unmutated':True,
                  'generator':{'external_lexicon':True,'external_version':generator.external_version,
                    'external_fingerprint':generator.external_fingerprint,
                    'dictionary_fingerprint':generator.dictionary_fingerprint,
                    'english_suffixes':[list(v) for v in generator.english_suffixes]},
                  'hashes':{name:digest(output/name) for name in ('inventory.jsonl','rows.jsonl','resources.jsonl')},
                  'resources':{'seconds':time.monotonic()-guard.started, 'max_rss_bytes':guard.peak_rss,
                               'max_swap_growth_bytes':guard.peak_swap_growth},
                  'labels_opened':False,'dev_opened':False,'test_opened':False,
                  'model_inference':False,'neural_training':False,'production_changed':False,
                  'scope':'Explicit candidate-specific COPY derivations; no accuracy or global English-recognition claim.'}
        write_json(output/'report.json',report)
        print(json.dumps({'completed':True,'counts':dict(total),'resources':report['resources']},ensure_ascii=False),flush=True)
    except Exception as error:
        write_json(output/'failure.json',{'completed':False,'error':str(error),'error_type':type(error).__name__})
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    run(parser.parse_args().output)
