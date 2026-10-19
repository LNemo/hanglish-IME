"""Complete the exposed 1880-case ensemble evaluation with its remaining740."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import evaluate_ensemble_regression as base

SOURCES = (
    ('train', '.omx/artifacts/numeric-structure-runtime-v1/cases.jsonl',
     '50550395a81a82635b4f0cf0ff5d1840e28f221d86426716cd6ff70e0bc50557'),
    ('development', '.omx/artifacts/numeric-structure-development-v1/cases.jsonl',
     '2baaccfbca4d7f4370a9c0b61c548fa9d4e33c50440c38c6dabdcd67d2ca920c'),
)
COUNTS = {'natural_train':512,'natural_dev':128,'selection_dev':60,'reasoning_dev':40}
BASELINE = {'natural_train':(318,339),'natural_dev':(81,86),
            'selection_dev':(60,60),'reasoning_dev':(34,40)}
PREVIOUS_REPORT = '.omx/artifacts/ensemble-regression-v2/report.json'
PREVIOUS_SHA = '6f2b8317c497604d83d4046cac1209606996b4bbf4f4f7c185cb21a0cc5d00d7'


def read_cases(root=ROOT, sources=SOURCES, counts=COUNTS):
    result=[]
    for role,path,sha in sources:
        rows=base.parse_jsonl(base.checked_bytes(root/path,sha))
        for row in rows:
            suite='natural_train' if role=='train' else row['suite']
            if suite not in counts:
                continue
            result.append((suite,row))
    if {s:sum(suite==s for suite,_ in result) for s in counts}!=counts:
        raise ValueError('remaining suite denominator differs')
    if len({(s,r['id']) for s,r in result})!=len(result):
        raise ValueError('duplicate suite identifiers')
    return result


def source_rows(cases):
    return [{**{k:r[k] for k in ('id','input','context')},'suite':s,'category':s}
            for s,r in cases]


def compare_baselines(captures, cases):
    references={(s,r['id']):r for s,r in cases}
    if len(references)!=len(captures):
        raise ValueError('reference denominator differs')
    for row in captures:
        reference=references[(row['suite'],row['id'])]
        if any(row[k]!=reference[k] for k in ('input','context')):
            raise ValueError('source identity differs')
        candidates=reference['child_candidates'] if row['suite']=='natural_train' else reference['candidates']
        if row['baseline']['candidates']!=candidates:
            raise ValueError(f"baseline candidate objects differ: {row['id']}")


def run(output):
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    try:
        proof=base.validate_independent_training(base.CHECKPOINT,base.VERIFICATION,base.VERIFICATION_SHA)
        training=proof['training_report']
        rows=source_rows(read_cases())
        previous=json.loads(base.checked_bytes(ROOT/PREVIOUS_REPORT,PREVIOUS_SHA))
        pins=base.merge_pins(proof['pins'],{path:sha for _,path,sha in SOURCES},
            {name:base.file_sha256(ROOT/name) for name in base.RUNTIME_FILES},
            {str(Path(__file__).relative_to(ROOT)):base.file_sha256(Path(__file__)),PREVIOUS_REPORT:PREVIOUS_SHA})
        base.verify_pins(pins)
        import torch
        torch.set_num_threads(2); torch.set_num_interop_threads(1)
        tick=time.perf_counter()
        converter=base.default_converter_factory(checkpoint=base.CHECKPOINT,
            checkpoint_sha256=training['checkpoint_sha256'],metadata_sha256=training['metadata_sha256'],
            experimental=True,context_ranking=True,external_lexicon=True,ranker_kind='auto')
        converter.load_models()
        initialization_ms=(time.perf_counter()-tick)*1000
        artifacts=base.artifact_file_pins(converter)
        pins=base.merge_pins(pins,artifacts)
        base.verify_pins(pins)
        base.check_resources(started)
        captures=base.capture_rows(converter,rows,output/'capture.jsonl',guard=lambda:base.check_resources(started))
        capture_sha=base.file_sha256(output/'capture.jsonl')
        cases=read_cases()
        compare_baselines(captures,cases)
        targets={(s,r['id']):r['expected'] for s,r in cases}
        metrics,outcomes=base.score_captures(captures,targets)
        for suite,values in metrics.items():
            first,five=BASELINE[suite]
            reference={'rows':COUNTS[suite],'top1':first,'top5':five}
            values['historical_reference']=reference
            values['baseline_matches_historical_reference']=values['arms']['baseline']==reference
            if not values['baseline_matches_historical_reference']:
                raise ValueError('baseline metrics differ from accepted reference')
        identities=base.runtime_identities(captures)
        if identities!=previous['runtime_identities']:
            raise ValueError('runtime/model identities differ from earlier1140')
        for name,sha in previous['source_hashes'].items():
            if name in pins and pins[name]!=sha:
                raise ValueError('shared source changed since earlier1140')
        base.verify_pins(pins)
        if base.artifact_file_pins(converter)!=artifacts:
            raise ValueError('model artifacts changed')
        elapsed,peak=base.check_resources(started)
        report=dict(schema='ensemble-natural-regression-v1',completed=True,rows=len(captures),
            suites=metrics,capture_sha256=capture_sha,source_hashes=pins,runtime_identities=identities,
            baseline_candidate_objects_match_approved_reference=True,
            previous1140_report_sha256=PREVIOUS_SHA,combined_regression_rows=previous['rows']+len(captures),
            combined_suites={**previous['suites'],**metrics},
            timings={'initialization_ms':initialization_ms,**base.timing_summary(captures)},
            raw_preserved_all=True,protected_segments_preserved_all=True,
            labels_passed_to_converter=False,source_projection_parsed_expected_bytes=True,
            expected_joined_after_closed_capture=True,generalization_claim_allowed=False,
            production_changed=False,training_run=False,elapsed_seconds=elapsed,peak_rss_bytes=peak)
        (output/'outcomes.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in outcomes))
        (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        return report
    except Exception as error:
        (output/'report.json').unlink(missing_ok=True)
        (output/'failure.json').write_text(json.dumps(dict(error_type=type(error).__name__,error=str(error),elapsed_seconds=time.monotonic()-started),indent=2)+'\n')
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    r=run(parser.parse_args().output)
    print(json.dumps({k:r[k] for k in ('completed','rows','combined_regression_rows','elapsed_seconds','peak_rss_bytes')}),flush=True)
