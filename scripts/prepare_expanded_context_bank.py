"""Select independently sourced observed words as extra context prototypes."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hanglish.core.phonology import romanize_word
from hanglish.models.retrieved_context_evidence import context_text
from scripts.evaluate_ensemble_regression import file_sha256,verify_pins

SOURCE=ROOT/'data/natural_context/runtime-collision-source-v5'
SOURCE_SHA='293c3992878a9f7bdc532669ef2bf233233dd6ab9a4fddd7af1209e6b593de89'
QUERY=ROOT/'.omx/artifacts/neural-context-retrieval-v1'
QUERY_SHA='41aa32e8b1220078e86a8aae3cf58d6e558914a11f654930092b691dcbd5d302'


def priority(id):
    return hashlib.sha256(('expanded-context-bank-v1\0'+id).encode()).hexdigest()


def excluded_closure(raw, exclusions):
    groups={r['group_id'] for r in exclusions if r.get('group_id')}
    sources={s for r in exclusions for s in r.get('source_ids',[]) or []}
    while True:
        size=len(groups)+len(sources)
        for row in raw:
            if row['group_id'] in groups or sources.intersection(row['source_ids']):
                groups.add(row['group_id']);sources.update(row['source_ids'])
        if len(groups)+len(sources)==size:return groups,sources


def document_components(raw):
    parents={r['group_id']:r['group_id'] for r in raw};owners={}
    def root(group):
        while parents[group]!=group:
            parents[group]=parents[parents[group]];group=parents[group]
        return group
    for row in raw:
        for source in row['source_ids']:
            if source in owners:
                a,b=root(row['group_id']),root(owners[source])
                parents[max(a,b)]=min(a,b)
            else:owners[source]=row['group_id']
    return {group:root(group) for group in parents}


def select(raw,fit,exclusions,quarantine,cap=24):
    vocabulary={c for row in fit for c in row['candidates']}
    groups,sources=excluded_closure(raw,exclusions)
    excluded_ids={r['id'] for r in exclusions if 'id' in r}
    components=document_components(raw)
    selected=[];counts=Counter();used_groups=defaultdict(set);used_sources=defaultdict(set);used_text=defaultdict(set);forms=Counter()
    for row in sorted(raw,key=lambda r:priority(r['id'])):
        word=row['word']
        if row['id'] in excluded_ids or row['group_id'] in groups or sources.intersection(row['source_ids']):
            counts['excluded_document']+=1;continue
        if word not in vocabulary:counts['outside_fit_vocabulary']+=1;continue
        if word in quarantine:counts['quarantined_form']+=1;continue
        if (row['split']!='train' or not row['group_id'] or not row['source_ids']
                or row['source_partition']!='upstream_train_and_existing_component_train'):
            raise ValueError('bank requires TRAIN source provenance')
        a,b=row['word_start'],row['word_end'];text=row['text']
        if not 0<=a<b<=len(text) or text[a:b]!=word:raise ValueError('source span mismatch')
        if romanize_word(word)!=row['word_input']:raise ValueError('source romanization changed')
        if '[MASK]' in text or '[MASK]' in row['left_context']:
            counts['literal_mask']+=1;continue
        context=row['left_context']+'\n'+text[:a]+' [MASK] '+text[b:]
        normalized=context_text(context)
        if not normalized:counts['empty_context']+=1;continue
        if forms[word]>=cap:counts['per_word_cap']+=1;continue
        component=components[row['group_id']]
        if component in used_groups[word] or used_sources[word].intersection(row['source_ids']):
            counts['duplicate_document_per_word']+=1;continue
        if normalized in used_text[word]:counts['duplicate_context_per_word']+=1;continue
        selected.append(dict(id='extra:'+row['id'],original_id=row['id'],group_id=row['group_id'],document_component=component,source_ids=row['source_ids'],kind='bank',expected=[word],candidates=[word],context=context,source=row['word_input'],word_start=a,word_end=b))
        forms[word]+=1;used_groups[word].add(component);used_sources[word].update(row['source_ids']);used_text[word].add(normalized)
    return selected,dict(counts),dict(sorted(forms.items())),groups,sources


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    assert file_sha256(SOURCE/'manifest.json')==SOURCE_SHA
    source=json.loads((SOURCE/'manifest.json').read_text());proof=json.loads((SOURCE/'independent-verification.json').read_text())
    assert proof['verified'] and proof['manifest_sha256']==SOURCE_SHA
    assert file_sha256(QUERY/'report.json')==QUERY_SHA
    query_report=json.loads((QUERY/'report.json').read_text());verify_pins(query_report['pins'])
    assert file_sha256(QUERY/'selection.json')==query_report['hashes']['selection.json']
    data=json.loads((QUERY/'selection.json').read_text())
    raw=[];selected=[]
    for role,target in [('train',raw),('selected_train',selected)]:
        path=SOURCE/source['outputs'][role]['file'];assert file_sha256(path)==source['outputs'][role]['sha256']
        target.extend(json.loads(line) for line in path.read_text().splitlines())
    review_path=ROOT/'.omx/artifacts/runtime-collision-source-review-v2.json';review=json.loads(review_path.read_text())
    assert review['selected_sha256']==source['outputs']['selected_train']['sha256']
    exclusions=selected+data['fit']+[r for rows in data['evaluation'].values() for r in rows]
    bank,counts,forms,groups,sources=select(raw,data['fit'],exclusions,set(review['quarantined_forms']))
    pins=dict(query_report['pins'])
    for path in [Path(__file__),SOURCE/'manifest.json',SOURCE/'independent-verification.json',SOURCE/'train.jsonl',SOURCE/'selected_train.jsonl',QUERY/'selection.json',QUERY/'report.json',review_path,ROOT/'.omx/plans/expanded-context-bank-v1.md']:
        pins[str(path.relative_to(ROOT))]=file_sha256(path)
    with (output/'bank.jsonl').open('x') as stream:
        for row in bank:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    (output/'exclusions.json').write_text(json.dumps(dict(groups=sorted(groups),source_ids=sorted(sources)),ensure_ascii=False)+'\n')
    report=dict(completed=True,training_ready=False,purpose='extra context bank; not runtime candidate supervision',source_rows=len(raw),rows=len(bank),words=len(forms),documents=len({r['group_id'] for r in bank}),forms=forms,exclusions=counts,all_spellings_expert_verified=False,labels_corrected=False,model_inference_performed=False,pins=pins,hashes={name:file_sha256(output/name) for name in ['bank.jsonl','exclusions.json']})
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['completed','source_rows','rows','words','documents','exclusions']},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output.resolve())
