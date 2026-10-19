"""Late FIT-only coverage comparison for independently closed child proposals."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_recombination_tokenization import digest,require

LABELS=ROOT/'.omx/artifacts/current-phonetic-train-features-v1/rows.jsonl'
LABEL_SHA='4c13063656d9a108316b9ce1ceec394a61e2c45c2ba86e7a8c8361f7480d6cb4'


def outcomes(source,capture,label):
    require(source['id']==capture['id']==label['id']
            and source['request']==capture['request']=={'input':label['input'],'context':''},
            'FIT identity differs')
    scoring=source['scoring'];gold=set(label['expected'])
    require(gold and all(isinstance(t,str) for t in gold),'FIT gold missing')
    scorable={c['text'] for c in scoring['original']}|{c['text'] for c in scoring['novel'] if c['status']=='scored'}
    available={c['text'] for c in scoring['original']+scoring['novel']}
    if not scoring['ranked']:available.update(c['text'] for c in scoring['fallback_public'])
    new={c['text'] for c in capture['proposals']}
    require(not new&available and len(new)==len(capture['proposals']),'new proposal duplicates old pool')
    before=bool(gold&available);after=bool(gold&(available|new))
    return {'id':source['id'],'old_scorable_covered':bool(gold&scorable),
            'old_available_covered':before,'expanded_available_covered':after,
            'coverage_gain':not before and after,'coverage_loss':before and not after,
            'new_gold_proposals':len(gold&new),'new_proposals':len(new),
            'no_pool':not scoring['ranked']}


def run(capture,output):
    report=json.loads((capture/'report.json').read_text())
    proof=json.loads((capture/'independent-verification.json').read_text())
    require(report['completed'] and proof['verified'] and report['rows']==proof['rows']==1642
            and proof['report_sha256']==digest(capture/'report.json')
            and proof['capture_sha256']==digest(capture/'capture.jsonl')
            and report['labels_opened'] is False and proof['labels_opened'] is False,
            'independently closed source-only capture required before labels')
    for path,sha in report['pins'].items():require(digest(Path(path))==sha,'source changed before labels')
    # First label read only after source closure and independent replay.
    require(digest(LABELS)==LABEL_SHA,'FIT labels changed')
    labels={row['id']:row for row in (json.loads(line) for line in LABELS.open())}
    require(len(labels)==1642,'FIT label denominator differs')
    intake=json.loads((ROOT/'.omx/artifacts/current-fit-recombination-selection-intake-v1.json').read_text())
    output.mkdir(parents=True,exist_ok=False)
    totals=Counter(rows=0,old_scorable_covered=0,old_available_covered=0,expanded_available_covered=0,
                   coverage_gain=0,coverage_loss=0,new_gold_proposals=0,new_proposals=0,no_pool=0)
    seen=set()
    with (capture/'capture.jsonl').open() as predictions,(output/'outcomes.jsonl').open('x') as target:
        for shard in intake['shards']:
            for line in (Path(shard['directory'])/'capture.jsonl').open():
                source=json.loads(line);proposed=json.loads(next(predictions))
                require(source['id'] not in seen,'duplicate FIT row');seen.add(source['id'])
                result=outcomes(source,proposed,labels[source['id']]);totals['rows']+=1
                for key,value in result.items():
                    if key!='id':totals[key]+=int(value)
                target.write(json.dumps(result,ensure_ascii=False)+'\n')
        require(next(predictions,None) is None and len(seen)==1642,'FIT capture denominator differs')
    require(totals['old_scorable_covered']==1571 and totals['no_pool']==13 and not totals['coverage_loss'],
            'frozen baseline or union coverage differs')
    result={'completed':True,'rows':1642,'counts':dict(totals),'capture_directory':str(capture.resolve()),
            'capture_sha256':digest(capture/'capture.jsonl'),'capture_report_sha256':digest(capture/'report.json'),
            'capture_proof_sha256':digest(capture/'independent-verification.json'),'labels_sha256':LABEL_SHA,
            'outcomes_sha256':digest(output/'outcomes.jsonl'),'evaluator_sha256':digest(Path(__file__)),
            'labels_opened_after_verified_capture':True,'dev_opened':False,'test_opened':False,
            'neural_training':False,'production_changed':False,
            'scope':'FIT generated-candidate coverage only; new proposals have no sentence scores or first/top5 decisions.'}
    (output/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.capture,args.output)
