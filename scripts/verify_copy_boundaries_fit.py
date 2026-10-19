"""Independently reconstruct admitted particle children and FIT audit counts."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hanglish.core.lattice import CandidateGenerator
from hanglish.core.source_correspondence import align
from scripts.audit_recombination_tokenization import digest,require


def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)


def expected_children(key,generator):
    source,text,kind=key
    coarse=dict(input_start=0,input_end=len(source),source=source,text=text,kind=kind)
    matches=set()
    if kind=='mixed' and any(o.text==text and o.kind=='mixed' for o in generator.options(source)):
        for roman,hangul in generator.english_suffixes:
            if (roman and source.lower().endswith(roman) and len(source)>len(roman)+1):
                boundary=len(source)-len(roman)
                if text==source[:boundary]+hangul:
                    matches.add((boundary,roman,hangul))
    if len(matches)!=1:return [coarse]
    boundary,roman,hangul=next(iter(matches))
    return [dict(input_start=0,input_end=boundary,source=source[:boundary],text=source[:boundary],kind='raw'),
            dict(input_start=boundary,input_end=len(source),source=source[boundary:],text=hangul,kind='mixed')]


def verify(directory):
    started=time.monotonic()
    report=json.loads((directory/'report.json').read_text())
    require(report['completed'] and report['rows']==1642 and report['labels_opened'] is False
            and report['production_changed'] is False,'incomplete source-only audit')
    for path,sha in report['pins'].items():require(digest(Path(path))==sha,'source changed')
    for name,sha in report['hashes'].items():require(digest(directory/name)==sha,'audit artifact changed')
    generator=CandidateGenerator(external_lexicon=True)
    require(report['generator']==dict(external_lexicon=True,external_version=generator.external_version,
        external_fingerprint=generator.external_fingerprint,dictionary_fingerprint=generator.dictionary_fingerprint,
        english_suffixes=[list(v) for v in generator.english_suffixes]),'runtime generator differs')
    inventory={};expected={};occurrences=Counter()
    for line in (directory/'inventory.jsonl').open():
        row=json.loads(line);key=tuple(row['key']);value=row['refinement']
        require(key not in inventory,'duplicate derivation key')
        source,text,kind=key
        children=expected_children(key,generator)
        require(value['source']==source and value['text']==text and value['segments']==children,
                'independent particle-boundary replay differs')
        require(value['coarse_segments']==[dict(input_start=0,input_end=len(source),source=source,text=text,kind=kind)],
                'coarse parent not retained')
        require(value['refined']==(len(children)==2),'refinement flag differs')
        aligned=align(source,text,children)
        require(''.join(s.source for s in aligned.spans)==source,'child source coverage differs')
        inventory[key]=row;expected[key]=children
    total=Counter(rows=0,no_pool_rows=0,old_candidates=0,novel_candidates=0,unsupported_candidates=0)
    ordered=[];identity=hashlib.sha256()
    intake=json.loads((ROOT/'.omx/artifacts/current-fit-recombination-selection-intake-v1.json').read_text())
    with (directory/'rows.jsonl').open() as recorded:
        for shard in intake['shards']:
            for line in (Path(shard['directory'])/'capture.jsonl').open():
                row=json.loads(line);scoring=row['scoring'];source=row['request']['input'];ordered.append(row['id'])
                stats=Counter(rows=1,no_pool_rows=int(not scoring['ranked']),old_candidates=len(scoring['original']),
                    novel_candidates=len(scoring['novel']),unsupported_candidates=sum(c['status']=='unsupported' for c in scoring['novel']))
                for candidate in scoring['original']+scoring['novel']:
                    old=align(source,candidate['text'],candidate['segments'])
                    new_segments=[]
                    for s in candidate['segments']:
                        unresolved=s['source']!=s['text'] and any('a'<=c<='z' or 'A'<=c<='Z' or '0'<=c<='9' for c in s['text'])
                        if unresolved:
                            key=(s['source'],s['text'],s['kind']);require(key in expected,'mixed segment absent from inventory')
                            occurrences[key]+=1
                            for child in expected[key]:
                                new_segments.append(dict(child,input_start=child['input_start']+s['input_start'],input_end=child['input_end']+s['input_start']))
                        else:new_segments.append(dict(s))
                    new=align(source,candidate['text'],new_segments)
                    stats.update(candidates=1,refined_candidates=int(candidate['segments']!=new_segments),coarse_spans=len(old.spans),refined_spans=len(new.spans),
                        mixed_before=sum(s.unresolved_ascii_output for s in old.spans),mixed_after=sum(s.unresolved_ascii_output for s in new.spans),
                        copy_before=sum(s.action=='copy' for s in old.spans),copy_after=sum(s.action=='copy' for s in new.spans))
                    payload={k:v for k,v in candidate.items() if k!='segments'}
                    identity.update((row['id']+'\0'+canonical(payload)+'\n').encode())
                require(json.loads(next(recorded))=={'id':row['id'],'counts':dict(stats)},'row counts differ')
                total.update(stats)
                rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
                require(rss<=2*1024**3 and time.monotonic()-started<=1800,'verification resource bound exceeded')
        require(next(recorded,None) is None,'extra audit rows')
    require(len(ordered)==len(set(ordered))==1642 and report['ordered_ids']==ordered
            and dict(total)==report['counts'] and total['old_candidates']==249177 and total['novel_candidates']==149885
            and total['no_pool_rows']==13 and total['unsupported_candidates']==31,'FIT denominator or counts differ')
    require(len(inventory)==report['unique_mixed_segments'] and set(occurrences)==set(inventory)
            and all(occurrences[k]==v['occurrences'] for k,v in inventory.items()),'inventory usage differs')
    require(identity.hexdigest()==report['candidate_nonalignment_identity_sha256'],'candidate scores/text/order differ')
    resources=[json.loads(line) for line in (directory/'resources.jsonl').open()]
    require(resources and all(r['seconds']<=1800 and r['rss_bytes']<=2*1024**3
            and r['swap_growth_bytes']<=512*1024**2 for r in resources),'recorded resource bound exceeded')
    require(report['resources']['max_rss_bytes']==max(r['rss_bytes'] for r in resources)
            and report['resources']['max_swap_growth_bytes']==max(0,max(r['swap_growth_bytes'] for r in resources)),
            'recorded resource summary differs')
    for path,sha in report['pins'].items():require(digest(Path(path))==sha,'source changed during verification')
    for name,sha in report['hashes'].items():require(digest(directory/name)==sha,'audit changed during verification')
    result={'verified':True,'rows':len(ordered),'counts':dict(total),'unique_mixed_segments':len(inventory),
        'report_sha256':digest(directory/'report.json'),'verifier_sha256':digest(Path(__file__)),
        'elapsed_seconds':time.monotonic()-started,'max_rss_bytes':rss,
        'labels_opened':False,'scope':'Independent inventory-rule replay and full FIT source/count reconstruction; no new adapter import, model inference, or accuracy claim.'}
    with (directory/'independent-verification.json').open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    verify(parser.parse_args().directory)
