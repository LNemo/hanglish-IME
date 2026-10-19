"""Replay fixed neural beams and enumerate COPY-preserving proposals independently."""
import argparse
from copy import deepcopy
import json
import hashlib
from pathlib import Path
import re
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hanglish.core.copy_boundaries import refine_path
from hanglish.core.source_correspondence import align
from scripts.audit_recombination_tokenization import digest,require

MODEL=ROOT/'.omx/artifacts/observed-phonetic-augmented-training-v1/phonetic_transducer.pt'
MODEL_SHA='f06b0a7cd6c5ccbdfaa376cc29f59e90b3fcf0ae3a157fd2c4f7efaa73b4180d'


def enumerate_proposals(source,parents,generator,beams,parent_pools=None):
    existing={p['text'] for p in parents};seen=set(existing);result=[]
    for index,parent in enumerate(parents):
        if not any(s['kind']=='mixed' and s['source']!=s['text']
                   and re.search('[A-Za-z0-9]',s['text']) for s in parent['segments']):continue
        refinement=refine_path(source,parent['text'],parent['segments'],generator=generator)
        cursor=0
        for derivation in refinement['derivations']:
            children=derivation['children']
            if derivation['status']=='refined':
                for offset,child in enumerate(children):
                    if child.get('action')!='convert' or not re.fullmatch('[A-Za-z]{1,40}',child['source']):continue
                    child_index=cursor+offset;roman=child['source'].lower()
                    require(roman in beams,'child beam missing')
                    for target,probability in beams[roman]:
                        segments=deepcopy(refinement['segments'])
                        old=segments[child_index]['text']
                        segments[child_index]['text']=target;segments[child_index]['kind']='copy_child_phonetic'
                        text=''.join(s['text'] for s in segments)
                        if text in seen:continue
                        seen.add(text)
                        # Source positions, exact COPY text, and fresh output offsets are checked on the actual path.
                        aligned=align(source,text,segments)
                        for copied in refinement['copy_spans']:
                            start,end=copied['source_boundary']
                            require(any(s.input_start==start and s.input_end==end and s.source==s.text==copied['source']
                                        for s in aligned.spans),'COPY child changed in proposal')
                        result.append({'parent_candidate_index':index,'text':text,'segments':segments,
                            'source_span':[child['input_start'],child['input_end']], 'model_source':roman,
                            'replacement':{'from':old,'to':target},'phonetic_log_probability':float(probability),
                            'parent_evidence':{'parent_text':parent['text'],'parent_index':derivation['parent_index'],
                                'segment_index':child_index,'derivation':deepcopy(derivation)}})
                        if parent_pools is not None:result[-1]['parent_pool']=parent_pools[index]
                        if len(result)==96:return result
            cursor+=len(children)
    return result


def verify(directory):
    started=time.monotonic()
    report=json.loads((directory/'report.json').read_text());closure=json.loads((directory/'capture.complete.json').read_text())
    capture_sha=digest(directory/'capture.jsonl')
    require(report['completed'] and report['rows']==1642 and report['labels_opened'] is False
            and closure['capture_sha256']==capture_sha and closure['rows']==1642
            and closure['report_sha256']==digest(directory/'report.json')
            and closure['beams_sha256']==digest(directory/'beams.json')
            and closure['resources_sha256']==digest(directory/'resources.jsonl'),'capture not closed')
    for path,sha in report['pins'].items():require(digest(Path(path))==sha,'frozen input changed')
    for name,sha in report['hashes'].items():require(digest(directory/name)==sha,'capture artifact changed')
    require(digest(MODEL)==MODEL_SHA,'phonetic checkpoint changed')
    from hanglish.core.lattice import CandidateGenerator
    generator=CandidateGenerator(external_lexicon=True)
    beams=json.loads((directory/'beams.json').read_text())
    expected_sources=sorted({roman for roman,_ in generator.english_suffixes})
    require(sorted(beams)==expected_sources and len(beams)==66,'particle inference sources differ')
    import torch
    from hanglish.models.phonetic_transducer import PhoneticTransducer
    from hanglish.models.batched_phonetic_decode import generate_batched
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    model=PhoneticTransducer.load(MODEL,device='cpu')
    for roman in expected_sources:
        decoded=[[text,float(score)] for text,score in generate_batched(model,roman,beam_size=8)]
        require(decoded==beams[roman],'neural beam replay differs: '+roman)
    del model
    intake=json.loads((ROOT/'.omx/artifacts/current-fit-recombination-selection-intake-v1.json').read_text())
    seen=[];total_new=0;new_rows=0;fallback_parents=0;no_pool=0;old_count=novel_count=unsupported=0
    with (directory/'capture.jsonl').open() as captures:
        for shard in intake['shards']:
            for line in (Path(shard['directory'])/'capture.jsonl').open():
                source=json.loads(line);row=json.loads(next(captures));scoring=source['scoring']
                require(row['id']==source['id'] and row['request']==source['request']
                        and row['source_shard']==source['source_shard']
                        and row['source_row_index']==source['source_row_index'],'FIT row identity differs')
                if scoring['ranked']:
                    parents=scoring['original']+scoring['novel'];kind='scored_pool'
                    pools=['original']*len(scoring['original'])+['novel']*len(scoring['novel'])
                else:
                    parents=scoring['fallback_public'];kind='fallback_public';no_pool+=1;fallback_parents+=len(parents)
                    pools=['fallback_public']*len(parents)
                require(row['parent_source']==kind,'fallback source policy differs')
                proposed=enumerate_proposals(source['request']['input'],parents,generator,beams,pools)
                require(row['proposals']==proposed,'independent child proposal enumeration differs')
                seen.append(row['id']);total_new+=len(proposed);new_rows+=bool(proposed)
                old_count+=len(scoring['original']);novel_count+=len(scoring['novel'])
                unsupported+=sum(c['status']=='unsupported' for c in scoring['novel'])
                rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
                require(rss<=2*1024**3 and time.monotonic()-started<=1800,'verification resource bound exceeded')
        require(next(captures,None) is None and len(seen)==len(set(seen))==1642,'capture denominator differs')
    require((no_pool,old_count,novel_count,unsupported)==(13,249177,149885,31),'parent denominator differs')
    require(report['counts']=={'rows':1642,'no_pool_rows':no_pool,'old_candidates':old_count,
            'novel_candidates':novel_count,'unsupported_candidates':unsupported,'fallback_parents':fallback_parents,
            'total_new':total_new,'new_rows':new_rows},'reported generation counts differ')
    require(report['ordered_id_sha256']==hashlib.sha256(''.join(i+'\n' for i in seen).encode()).hexdigest(),
            'ordered IDs differ')
    for path,sha in report['pins'].items():require(digest(Path(path))==sha,'inputs changed during replay')
    for name,sha in report['hashes'].items():require(digest(directory/name)==sha,'capture changed during replay')
    resources=[json.loads(line) for line in (directory/'resources.jsonl').open()]
    require(resources and all(r['seconds']<=1800 and r['rss_bytes']<=2*1024**3
            and r['swap_growth_bytes']<=512*1024**2 for r in resources),'capture resource bound exceeded')
    result={'verified':True,'rows':1642,'capture_sha256':capture_sha,'report_sha256':digest(directory/'report.json'),
            'new_proposals':total_new,'new_rows':new_rows,'fallback_parents':fallback_parents,
            'beam_sources_replayed':len(beams),'verifier_sha256':digest(Path(__file__)),
            'elapsed_seconds':time.monotonic()-started,'max_rss_bytes':rss,'labels_opened':False,
            'scope':'Exact frozen neural-beam replay; independent child proposal enumeration using previously verified boundary adapter. No new proposal helper import, labels or sentence ranking.'}
    with (directory/'independent-verification.json').open('x') as target:json.dump(result,target,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    verify(parser.parse_args().directory)
