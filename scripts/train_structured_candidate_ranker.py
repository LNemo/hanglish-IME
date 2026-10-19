"""Train an explicit candidate/source/origin and masked-context cross encoder."""
import argparse
import json
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from hanglish.models.structured_candidate_ranker import StructuredCandidateRanker, ENCODING
from scripts.train_natural_homophone_ranker import objective
from scripts.train_semantic_ranker import _optimizer, _set_trainable_layers
from scripts.calibrate_context_evidence import compare
from scripts.evaluate_ensemble_regression import file_sha256, rss_bytes, verify_pins

SOURCE = ROOT / '.omx/artifacts/candidate-evidence-head-v1'
SOURCE_SHA = 'ef500d83501d3e72310a98d1241ecdc736e16f7b88f530be01c7deddd76c1998'
PARENT = ROOT / 'checkpoints/semantic_ranker'
RECIPE = dict(seed=20260920, epochs=2, top_layers=2, batch_groups=4,
              head_lr=5e-5, encoder_lr=1e-5, weight_decay=.01, max_length=128)


def parent_prediction(row):
    if row.get('kind') == 'fixed':
        result = row['fixed_result']
        scores = []
    else:
        scores = row['teacher_scores']
        adjusted = [s + .15 * b for s, b in zip(scores, row['base_scores'])]
        result = row['candidates'][min(range(len(scores)), key=lambda i: (-adjusted[i], row['candidates'][i]))]
    return dict(id=row['id'], result=result, correct=isinstance(result, str) and result in row['expected'], scores=scores)


def evaluate(ranker, pools, guard):
    ranker.model.eval()
    results = {}
    for role, rows in pools.items():
        predictions = []
        for row in rows:
            if row.get('kind') == 'fixed' or not ranker.can_score(row):
                prediction = dict(parent_prediction(row), fallback=True)
            else:
                scores = ranker.score_group(row)
                adjusted = [s + .15 * b for s, b in zip(scores, row['base_scores'])]
                result = row['candidates'][min(range(len(scores)), key=lambda i: (-adjusted[i], row['candidates'][i]))]
                prediction = dict(id=row['id'], result=result, correct=result in row['expected'], scores=scores, fallback=False)
            predictions.append(prediction)
            guard()
        results[role] = predictions
    return results


def prepare():
    if file_sha256(SOURCE / 'report.json') != SOURCE_SHA:
        raise ValueError('evidence source report changed')
    report = json.loads((SOURCE / 'report.json').read_text())
    proof = json.loads((SOURCE / 'independent-verification.json').read_text())
    if not proof['verified'] or proof['report_sha256'] != SOURCE_SHA:
        raise ValueError('source proof missing')
    verify_pins(report['pins'])
    if file_sha256(SOURCE / 'selection.json') != report['hashes']['selection.json']:
        raise ValueError('source selection changed')
    selection = json.loads((SOURCE / 'selection.json').read_text())
    semantic = {r['id']: r for r in selection['evaluation']['semantic_hold']}
    audit_path = ROOT / '.omx/artifacts/semantic-context-learning-audit-v1/capture.jsonl'
    with audit_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['arm'] == 'parent' and row['id'] in semantic:
                target = semantic[row['id']]
                if target['context'] != row['own']['calls'][0]['contexts'][0]:
                    raise ValueError('semantic context differs from source capture')
                target['source'] = row['input']
    if any('source' not in r for r in semantic.values()):
        raise ValueError('missing original Roman source')
    fit_ids = {r['group_id'] for r in selection['fit']}
    held_ids = {r['group_id'] for rows in selection['evaluation'].values() for r in rows if 'group_id' in r}
    if fit_ids & held_ids:
        raise ValueError('training and held documents overlap')
    pins = dict(report['pins'])
    for path in (Path(__file__), ROOT / 'hanglish/models/structured_candidate_ranker.py',
                 ROOT / '.omx/plans/structured-candidate-ranker-v1.md', SOURCE / 'report.json',
                 SOURCE / 'independent-verification.json', SOURCE / 'selection.json', audit_path,
                 ROOT / 'scripts/train_natural_homophone_ranker.py', ROOT / 'scripts/train_semantic_ranker.py'):
        pins[str(path.relative_to(ROOT))] = file_sha256(path)
    return selection['fit'], selection['evaluation'], pins


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    def guard():
        if time.monotonic() - started > 1200 or rss_bytes() > 3 * 1024 ** 3:
            raise RuntimeError('structured candidate training exceeds1200seconds/3GiB')

    try:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.manual_seed(RECIPE['seed'])
        random.seed(RECIPE['seed'])
        units, pools, pins = prepare()
        verify_pins(pins)
        ranker = StructuredCandidateRanker.from_parent(PARENT)
        original = {name: value.detach().clone() for name, value in ranker.model.state_dict().items()}
        source_units = units
        units = [r for r in units if ranker.can_score(r)]
        excluded = [r['id'] for r in source_units if not ranker.can_score(r)]
        if not units:
            raise ValueError('no eligible structured groups')
        (output / 'selection.json').write_text(json.dumps(dict(fit=units, excluded_fit=excluded, evaluation=pools), ensure_ascii=False) + '\n')
        parent = {role: [parent_prediction(r) for r in rows] for role, rows in pools.items()}
        before = evaluate(ranker, pools, guard)
        (output / 'before.json').write_text(json.dumps(dict(parent=parent, structured_untrained=before), ensure_ascii=False) + '\n')
        configuration = _set_trainable_layers(ranker.model, RECIPE['top_layers'])
        trainable = {name for name, p in ranker.model.named_parameters() if p.requires_grad}
        optimizer = _optimizer(ranker.model, SimpleNamespace(**RECIPE), head_only=False)
        updates, losses = 0, []
        fit_started = time.monotonic()
        with (output / 'updates.jsonl').open('x') as stream:
            for epoch in range(RECIPE['epochs']):
                order = list(range(len(units)))
                random.Random(RECIPE['seed'] + epoch).shuffle(order)
                ranker.model.train()
                for start in range(0, len(order), RECIPE['batch_groups']):
                    batch = [units[i] for i in order[start:start + RECIPE['batch_groups']]]
                    encoded = ranker.encode_groups(batch)
                    optimizer.zero_grad(set_to_none=True)
                    loss = objective(ranker.model(**encoded).logits.reshape(-1), batch)
                    if not torch.isfinite(loss):
                        raise ValueError('nonfinite structured loss')
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(ranker.model.parameters(), 1.)
                    if not torch.isfinite(norm):
                        raise ValueError('nonfinite structured gradient')
                    optimizer.step()
                    updates += 1
                    losses.append(float(loss.detach()))
                    stream.write(json.dumps(dict(epoch=epoch, update=updates, ids=[r['id'] for r in batch], loss=losses[-1])) + '\n')
                    stream.flush()
                    guard()
                    if updates == 3:
                        estimated = (time.monotonic() - fit_started) / 3 * RECIPE['epochs'] * ((len(units) + 3) // 4)
                        (output / 'resource-pilot.json').write_text(json.dumps(dict(updates=3, estimated_training_s=estimated, peak_rss_bytes=rss_bytes())) + '\n')
                        if estimated > 900:
                            raise RuntimeError('structured training projected over900seconds')
                print(json.dumps(dict(epoch=epoch + 1, updates=updates, elapsed_s=time.monotonic() - started)), flush=True)
        ranker.model.eval()
        changed = [name for name, value in ranker.model.state_dict().items() if not torch.equal(original[name], value)]
        if not changed or any(name not in trainable for name in changed):
            raise ValueError('unexpected changed tensor scope')
        checkpoint = output / 'checkpoint'
        ranker.model.save_pretrained(checkpoint, safe_serialization=True)
        ranker.tokenizer.save_pretrained(checkpoint)
        hashes = {p.name: file_sha256(p) for p in checkpoint.iterdir() if p.is_file()}
        (checkpoint / 'metadata.json').write_text(json.dumps(dict(format_version=1, encoding=ENCODING,
            max_length=ranker.max_length, inference_batch_size=ranker.batch_size, artifact_hashes=hashes,
            experimental=True, recipe=RECIPE, source_pins=pins), indent=2) + '\n')
        after = evaluate(ranker, pools, guard)
        loaded = StructuredCandidateRanker.load(checkpoint)
        if evaluate(loaded, pools, guard) != after:
            raise ValueError('structured checkpoint reload predictions differ')
        metrics = {role: compare(parent[role], after[role]) for role in pools}
        untrained_metrics = {role: compare(parent[role], before[role]) for role in pools}
        advance = metrics['source_hold']['after'] > metrics['source_hold']['before'] and all(m['after'] >= m['before'] for role, m in metrics.items() if role != 'source_hold')
        (output / 'predictions.json').write_text(json.dumps(dict(parent=parent, structured_untrained=before, after=after), ensure_ascii=False) + '\n')
        verify_pins(pins)
        result = dict(completed=True, training_run=True, production_changed=False, recipe=RECIPE,
                      encoding=ENCODING, configuration=configuration, updates=updates, fit_rows=len(units),
                      excluded_fit=excluded, metrics=metrics, untrained_metrics=untrained_metrics,
                      advance_to_runtime=advance, mean_loss=sum(losses) / len(losses),
                      changed_parameter_names=changed, unchanged_tensor_count=len(original) - len(changed),
                      reload_predictions_equal=True, pins=pins,
                      hashes={name: file_sha256(output / name) for name in ('selection.json', 'before.json', 'updates.jsonl', 'predictions.json')},
                      checkpoint_sha256=file_sha256(checkpoint / 'model.safetensors'),
                      metadata_sha256=file_sha256(checkpoint / 'metadata.json'),
                      elapsed_s=time.monotonic() - started, peak_rss_bytes=rss_bytes(),
                      scope='one fixed training recipe; exposed diagnostic word pools, not full runtime or fresh generalization')
        (output / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({k: result[k] for k in ('completed', 'fit_rows', 'excluded_fit', 'updates', 'metrics', 'advance_to_runtime', 'elapsed_s', 'peak_rss_bytes')}, ensure_ascii=False), flush=True)
    except BaseException as exc:
        (output / 'failure.json').write_text(json.dumps(dict(completed=False, error=repr(exc))))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output.resolve())
