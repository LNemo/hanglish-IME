"""Compare unconstrained and prefix-constrained generation on train-only missing groups."""
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from hanglish.core.neural_converter import file_sha256, TRANSDUCER
from hanglish.core.phonology import romanize_word
from hanglish.models.phonetic_transducer import PhoneticTransducer
from scripts.constrained_phonetic_search import generate_constrained
from scripts.prepare_context_training import OUTPUT, source_hashes, validate_context_data
from scripts.train_natural_sentence_ranker import rss_bytes

MANIFEST = '7c1c03205dbc9c8983c2b3dee37e4a6bcabb04d45a265e9e1699460595275b10'
DESTINATION = ROOT / '.omx/artifacts/constrained-phonetic-probe-v1'


def hashes():
    result = source_hashes()
    for name in ('scripts/probe_constrained_phonetic_search.py', 'scripts/constrained_phonetic_search.py',
                 'scripts/train_natural_sentence_ranker.py'):
        result[name] = file_sha256(ROOT / name)
    return result


def main():
    if DESTINATION.exists():
        raise FileExistsError(DESTINATION)
    _, data = validate_context_data(OUTPUT, MANIFEST)
    groups = [g for row in data['natural_train'] for g in row['groups']
              if not g['positive_indices'] and len(g['source']) <= 40
              and g['source'].isascii() and g['source'].isalpha() and g['source'].islower()
              and all(all('가'<=c<='힣' for c in t) for t in g['expected'])]
    # This is an explicitly failure-selected TRAIN diagnostic, not a held-out accuracy estimate.
    selected = [{k:g[k] for k in ('id','source','expected')} for g in groups]
    before = hashes()
    torch.set_num_threads(2)
    model = PhoneticTransducer.load(TRANSDUCER)
    DESTINATION.mkdir(parents=True)
    (DESTINATION/'selection.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2)+'\n')
    started, outcomes = time.monotonic(), []
    with (DESTINATION/'cases.jsonl').open('x') as stream:
        for index, row in enumerate(selected):
            begin=time.monotonic()
            baseline=model.generate(row['source'],beam_size=8)
            baseline_ms=1000*(time.monotonic()-begin)
            stats={};begin=time.monotonic()
            constrained=generate_constrained(model,row['source'],beam_size=8,stats=stats)
            constrained_ms=1000*(time.monotonic()-begin)
            if any(romanize_word(text)!=row['source'] for text,_ in baseline+constrained):
                raise ValueError('non-forward-valid generated candidate')
            case={**row,'baseline':baseline,'constrained':constrained,
                  'baseline_covered':any(t in row['expected'] for t,_ in baseline),
                  'constrained_covered':any(t in row['expected'] for t,_ in constrained),
                  'baseline_top1':bool(baseline) and baseline[0][0] in row['expected'],
                  'constrained_top1':bool(constrained) and constrained[0][0] in row['expected'],
                  'baseline_ms':baseline_ms,'constrained_ms':constrained_ms,'stats':stats}
            outcomes.append(case);stream.write(json.dumps(case,ensure_ascii=False)+'\n');stream.flush()
            if rss_bytes()>3*1024**3 or time.monotonic()-started>1800:
                raise RuntimeError('generation probe resource bound exceeded')
            if (index+1)%16==0:
                print(json.dumps({'cases':index+1,'seconds':round(time.monotonic()-started,2)}),flush=True)
    validate_context_data(OUTPUT,MANIFEST)
    if before!=hashes():raise ValueError('generation probe sources changed')
    report={'scope':'failure-selected natural TRAIN groups only; no dev/final inference',
            'rows':len(outcomes),'source_manifest_sha256':MANIFEST,'source_hashes':before,
            'selection_sha256':file_sha256(DESTINATION/'selection.json'),
            'cases_sha256':file_sha256(DESTINATION/'cases.jsonl'),
            'metrics':{key:sum(row[key] for row in outcomes) for key in
                       ('baseline_covered','constrained_covered','baseline_top1','constrained_top1')},
            'coverage_gains':[r['id'] for r in outcomes if r['constrained_covered'] and not r['baseline_covered']],
            'coverage_losses':[r['id'] for r in outcomes if r['baseline_covered'] and not r['constrained_covered']],
            'median_ms':{key:statistics.median(r[key] for r in outcomes) for key in ('baseline_ms','constrained_ms')},
            'timing_policy':'Sequential paired diagnostic; descriptive timing, not an isolated performance benchmark.',
            'seconds':time.monotonic()-started,'peak_rss_bytes':rss_bytes(),
            'test_data_read':False,'deployed':False}
    (DESTINATION/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (DESTINATION/'source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:v for k,v in report.items() if k!='source_hashes'},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
