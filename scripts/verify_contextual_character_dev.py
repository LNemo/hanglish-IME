"""Source-only independent replay of contextual predictions on frozen DEV pools."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PARENT = ROOT / '.omx/artifacts/character-edge-dev-full-v1'
PARENT_CAPTURE_SHA = '8985ec4f50b0151987e69cad63a3033203d1407af68cbc9c250fe82c7a7858f1'
PARENT_REPORT_SHA = 'ea29e58810b9543205d4ec94b244e5aa9a1138f74a4acb95182e3c4f13a69614'
CALIBRATION = ROOT / '.omx/artifacts/phonetic-score-scale-training-v1/model.json'
CALIBRATION_SHA = 'e07462d50da0b67304a82a9a261068ccc84b8055a99b5d70af56db530d245838'
EXPECTED_COUNTS = dict(no_pool_rows=4, novel_candidates=27831, old_candidates=47694,
                       ranked_rows=312, scored_novel=27831, unsupported_novel=0)
ARMS = ('old_contextual', 'expanded_contextual')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def reference_base(candidate, calibration):
    p, scale = calibration['parameters'], calibration['scale']
    features = candidate['features']
    require(len(p) == 5 and len(scale) == len(features) == 4,
            'calibration dimensions differ')
    value = p[0]*candidate['sentence_score']
    value += sum(p[i+1]*(features[i]/scale[i]) for i in range(4))
    require(math.isfinite(value), 'nonfinite calibrated score')
    return value


def reference_public(source, candidates, scores):
    require(len(candidates) == len(scores) and candidates, 'public domain differs')
    ranked = sorted(zip(candidates, scores), key=lambda item: (-item[1], item[0]['text']))
    result = [{'text':c['text'], 'score':round(float(score),4),
               'segments':deepcopy(c['segments'])} for c,score in ranked[:5]]
    if source not in {c['text'] for c in result}:
        result = result[:4]+[{'text':source,'score':None,'segments':[
            {'input_start':0,'input_end':len(source),'source':source,'text':source,'kind':'raw'}]}]
    return result


def replay_row(parent, model, calibration):
    import torch
    scoring = parent['scoring']
    if not scoring['ranked']:
        require(not scoring['original'] and not scoring['novel']
                and scoring['fallback_public'], 'fallback domain differs')
        return {arm:[] for arm in ARMS}, {
            arm:deepcopy(scoring['fallback_public']) for arm in ARMS}
    original = scoring['original']
    novel = [c for c in scoring['novel'] if c['status'] == 'scored']
    pool = original+novel
    candidates = [{'text':c['text'],'segments':c['segments']} for c in pool]
    base = [reference_base(c,calibration) for c in pool]
    source = parent['request']['input']
    require(reference_public(source,candidates[:len(original)],base[:len(original)])
            == parent['predictions']['old_calibrated']
            and reference_public(source,candidates,base)
            == parent['predictions']['expanded_calibrated'], 'fixed calibrated comparator differs')
    scores = []
    with torch.no_grad():
        for start in range(0,len(candidates),32):
            prepared = model.prepare(source,candidates[start:start+32],base[start:start+32])
            scores.extend(model(prepared)['scores'].detach().cpu().tolist())
    require(len(scores) == len(candidates) and all(math.isfinite(v) for v in scores),
            'replayed scores differ')
    old_scores = scores[:len(original)]
    return {'old_contextual':old_scores,'expanded_contextual':scores}, {
        'old_contextual':reference_public(source,candidates[:len(original)],old_scores),
        'expanded_contextual':reference_public(source,candidates,scores)}


def authenticate(directory):
    require(not (directory/'independent-verification.json').exists(), 'proof already exists')
    report_path = directory/'report.json'
    report = read(report_path)
    require(report.get('schema') == 'contextual-character-dev-capture-v1'
            and report.get('candidate_chunk') == 32
            and report.get('completed') is True and report.get('rows') == 316
            and report.get('labels_opened') is False and report.get('test_opened') is False
            and report.get('production_changed') is False
            and Path(report['parent_directory']).resolve() == PARENT,
            'source-only DEV capture required')
    require(report.get('limits') == {'whole_seconds':600.,'row_seconds':30.,
            'rss_bytes':2*1024**3,'swap_growth_bytes':512*1024**2,'output_bytes':128*1024**2}
            and sum((directory/name).stat().st_size for name in
                    ('report.json','capture.jsonl','resources.jsonl')) <= 128*1024**2,
            'capture limits or output size differ')
    require(digest(PARENT/'capture.jsonl') == PARENT_CAPTURE_SHA
            and digest(PARENT/'report.json') == PARENT_REPORT_SHA
            and digest(CALIBRATION) == CALIBRATION_SHA, 'fixed source evidence changed')
    parent_report = read(PARENT/'report.json')
    parent_proof = read(PARENT/'independent-verification.json')
    require(parent_report['completed'] and parent_report['labels_opened'] is False
            and parent_proof['verified'] and parent_proof['rows'] == 316
            and parent_proof['labels_opened'] is False
            and parent_proof['report_sha256'] == PARENT_REPORT_SHA
            and parent_proof['capture_sha256'] == PARENT_CAPTURE_SHA,
            'verified parent capture required')
    training = Path(report['training_directory'])
    trained = read(training/'report.json')
    proof = read(training/'independent-verification.json')
    require(trained['completed'] and trained['scope'] == 'full' and trained['rows'] == 1642
            and trained['updates'] == 6284 and proof['verified'] and proof['rows'] == 1642
            and proof['report_sha256'] == digest(training/'report.json')
            and proof['checkpoint_sha256'] == digest(training/'model.pt')
            and trained['hashes']['model.pt'] == proof['checkpoint_sha256'],
            'independently verified full FIT checkpoint required')
    model_source = ROOT/'hanglish/models/contextual_character_residual.py'
    require(trained['model_source_sha256'] == digest(model_source), 'model source changed')
    pins = dict(report['pins'])
    require(pins and all(digest(path) == sha for path,sha in pins.items()), 'capture pin changed')
    for dependency in (parent_report,trained):
        for path,sha in dependency['pins'].items():
            require(digest(path) == sha, 'parent or training pin changed')
            require(path not in pins or pins[path] == sha, 'inconsistent dependency pin')
            pins[path] = sha
    for name in ('capture.jsonl','resources.jsonl'):
        require(digest(directory/name) == report['hashes'][name], 'capture output changed')
        pins[str(directory/name)] = report['hashes'][name]
    for path in (report_path,training/'report.json',training/'independent-verification.json',
                 training/'model.pt',PARENT/'report.json',PARENT/'independent-verification.json',
                 PARENT/'capture.jsonl',CALIBRATION,Path(__file__).resolve()):
        pins[str(path)] = digest(path)
    require(all(digest(path) == sha for path,sha in pins.items()), 'capture pin changed')
    resources = [json.loads(line) for line in (directory/'resources.jsonl').open()]
    require(resources and all(0 <= r['seconds'] <= 600 and 0 <= r['rss_bytes'] <= 2*1024**3
                              and r['swap_growth_bytes'] <= 512*1024**2 for r in resources)
            and all(a['seconds'] <= b['seconds'] for a,b in zip(resources,resources[1:]))
            and report['resources']['max_rss_bytes'] == max(r['rss_bytes'] for r in resources)
            and report['resources']['max_swap_growth_bytes'] == max(0,max(r['swap_growth_bytes'] for r in resources))
            and resources[-1]['seconds'] <= report['resources']['seconds'] <= 600,
            'capture resource bounds differ')
    return report, training, pins


def verify(directory):
    directory = directory.resolve()
    started = time.monotonic()
    report,training,pins = authenticate(directory)
    import torch
    from hanglish.models.contextual_character_residual import ContextualCharacterResidualRanker
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    model = ContextualCharacterResidualRanker()
    require(sum(p.numel() for p in model.parameters()) == 31681, 'model size differs')
    model.load_state_dict(torch.load(training/'model.pt',map_location='cpu',weights_only=True))
    model.eval()
    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'nonfinite checkpoint')
    calibration = read(CALIBRATION)
    counts = {key:0 for key in EXPECTED_COUNTS}
    seen = set()
    with (PARENT/'capture.jsonl').open() as parents, (directory/'capture.jsonl').open() as captures:
        for index,line in enumerate(parents):
            row_started = time.monotonic()
            parent = json.loads(line)
            saved_line = next(captures,None)
            require(saved_line is not None,'missing capture row')
            saved = json.loads(saved_line)
            require(saved['id'] == parent['id'] and saved['request'] == parent['request']
                    and saved['source_row_index'] == index and saved['dataset_split'] == 'dev'
                    and saved['id'] not in seen and saved['request']['context'] == '',
                    'DEV source identity differs')
            seen.add(saved['id'])
            scores,predictions = replay_row(parent,model,calibration)
            require(saved['contextual_scores'] == scores and saved['predictions'] == predictions,
                    'fresh checkpoint scores or public predictions differ')
            scoring = parent['scoring']
            counts['ranked_rows' if scoring['ranked'] else 'no_pool_rows'] += 1
            counts['old_candidates'] += len(scoring['original'])
            counts['novel_candidates'] += len(scoring['novel'])
            counts['scored_novel'] += sum(c['status']=='scored' for c in scoring['novel'])
            counts['unsupported_novel'] += sum(c['status']=='unsupported' for c in scoring['novel'])
            require(time.monotonic()-row_started <= 30 and time.monotonic()-started <= 600,
                    'replay time bound exceeded')
        require(next(captures,None) is None and len(seen)==316,'DEV denominator differs')
    require(counts == report['counts'] == EXPECTED_COUNTS, 'DEV counts differ')
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    require(rss <= 2*1024**3 and all(digest(path)==sha for path,sha in pins.items())
            and time.monotonic()-started <= 600,
            'replay resources exceeded or evidence changed')
    result = {'verified':True,'rows':316,'counts':counts,
              'report_sha256':digest(directory/'report.json'),
              'capture_sha256':digest(directory/'capture.jsonl'),
              'checkpoint_sha256':digest(training/'model.pt'),
              'verifier_sha256':digest(Path(__file__)),'elapsed_seconds':time.monotonic()-started,
              'max_rss_bytes':rss,'labels_opened':False,'test_opened':False,
              'scope':'Exact contextual checkpoint replay on fixed DEV pools; independent calibration and public rendering. Prior sentence logits reused, not replayed. No labels.'}
    with (directory/'independent-verification.json').open('x') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    verify(parser.parse_args().directory)
