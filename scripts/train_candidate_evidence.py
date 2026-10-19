"""Train a candidate-origin-aware residual head using document cross-fitting."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from hanglish.models.candidate_evidence import CandidateEvidenceHead, FEATURES, pack, objective, predict
from scripts.calibrate_context_evidence import compare
from scripts.evaluate_ensemble_regression import file_sha256, rss_bytes, verify_pins

SOURCE = ROOT / '.omx/artifacts/context-evidence-calibration-v1'
SOURCE_SHA = '53ddbb9fb110bc4172ec41458276676cb43c4b6d4dd6a9332e8ccc93ebfd7df3'
RECIPE = dict(seed=20260920, steps=300, learning_rate=.01, weight_decay=.01,
              folds=4, hidden_units=8, residual_bound=2., residual_l2=.03,
              temperature=2., base_weight=.15, max_seconds=600, max_rss_bytes=3 * 1024 ** 3)


def add_origins(selection):
    rows = selection['fit'] + [r for pool in selection['evaluation'].values() for r in pool]
    needed = {r['id']: r for r in rows if r.get('kind') != 'fixed' and 'kinds' not in r}
    seen, paths = set(), []
    for role in ('natural_train', 'retention_train'):
        path = ROOT / f'data/natural_context/current-context-v1/{role}.jsonl'
        paths.append(path)
        with path.open() as stream:
            for line in stream:
                for group in json.loads(line)['groups']:
                    if group['id'] not in needed:
                        continue
                    row = needed[group['id']]
                    indices = [group['candidates'].index(c) for c in row['candidates']]
                    if (group['context'] != row['context']
                            or [group['base_scores'][i] for i in indices] != row['base_scores']):
                        raise ValueError('original group features differ')
                    row['kinds'] = [group['kinds'][i] for i in indices]
                    seen.add(row['id'])
    path = ROOT / '.omx/artifacts/semantic-context-learning-audit-v1/capture.jsonl'
    paths.append(path)
    with path.open() as stream:
        for line in stream:
            original = json.loads(line)
            if original['arm'] != 'parent' or original['id'] not in needed:
                continue
            row = needed[original['id']]
            options = {o['text']: o for o in original['options']}
            if [options[c]['score'] for c in row['candidates']] != row['base_scores']:
                raise ValueError('semantic group features differ')
            row['kinds'] = [options[c]['kind'] for c in row['candidates']]
            seen.add(row['id'])
    if seen != set(needed):
        raise ValueError('candidate origins missing from pinned captures')
    return paths


def fit(rows, neutral, guard=lambda: None):
    torch.manual_seed(RECIPE['seed'])
    model = CandidateEvidenceHead()
    packed = pack(rows, neutral)
    optimizer = torch.optim.Adam(model.parameters(), lr=RECIPE['learning_rate'], weight_decay=RECIPE['weight_decay'])
    losses = []
    for step in range(RECIPE['steps']):
        optimizer.zero_grad(set_to_none=True)
        loss = objective(model, packed)
        if not torch.isfinite(loss):
            raise ValueError('nonfinite evidence-head loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(norm):
            raise ValueError('nonfinite evidence-head gradient')
        optimizer.step()
        if step % 50 == 0 or step + 1 == RECIPE['steps']:
            losses.append(dict(step=step + 1, loss=float(loss.detach())))
        guard()
    return model.eval(), losses


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    def guard():
        if time.monotonic() - started > RECIPE['max_seconds'] or rss_bytes() > RECIPE['max_rss_bytes']:
            raise RuntimeError('evidence head exceeds600seconds/3GiB')

    try:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        if file_sha256(SOURCE / 'report.json') != SOURCE_SHA:
            raise ValueError('calibration source report changed')
        report = json.loads((SOURCE / 'report.json').read_text())
        proof = json.loads((SOURCE / 'independent-verification.json').read_text())
        if not proof['verified'] or proof['report_sha256'] != SOURCE_SHA:
            raise ValueError('verified source evidence missing')
        verify_pins(report['pins'])
        for filename, expected in report['hashes'].items():
            if file_sha256(SOURCE / filename) != expected:
                raise ValueError('frozen evidence changed')
        selection = json.loads((SOURCE / 'selection.json').read_text())
        metadata_paths = add_origins(selection)
        neutral = {r['candidate']: r['score'] for r in map(json.loads, (SOURCE / 'neutral-capture.jsonl').read_text().splitlines())}
        pins = dict(report['pins'])
        for path in (Path(__file__), ROOT / 'hanglish/models/candidate_evidence.py',
                     ROOT / '.omx/plans/candidate-evidence-head-v1.md', SOURCE / 'report.json',
                     SOURCE / 'independent-verification.json', SOURCE / 'selection.json', SOURCE / 'neutral-capture.jsonl', *metadata_paths):
            pins[str(path.relative_to(ROOT))] = file_sha256(path)
        verify_pins(pins)
        (output / 'selection.json').write_text(json.dumps(selection, ensure_ascii=False) + '\n')
        units, pools, folds = selection['fit'], selection['evaluation'], selection['folds']
        initial = CandidateEvidenceHead().eval()
        fold_states, fold_losses, oof = {}, {}, {}
        for fold in range(RECIPE['folds']):
            training = [r for r in units if folds[r['id']] != fold]
            validation = [r for r in units if folds[r['id']] == fold]
            model, losses = fit(training, neutral, guard)
            fold_states[str(fold)], fold_losses[str(fold)] = model.export(), losses
            oof.update({r['id']: r for r in predict(model, validation, neutral)})
            print(json.dumps(dict(fold=fold, trained_rows=len(training), held_rows=len(validation), elapsed_s=time.monotonic() - started)), flush=True)
        oof_metrics = {kind: compare(predict(initial, [r for r in units if r['kind'] == kind], neutral),
                                     [oof[r['id']] for r in units if r['kind'] == kind]) for kind in ('source', 'retention')}
        model, losses = fit(units, neutral, guard)
        bundle = dict(features=FEATURES, recipe=RECIPE, state=model.export(), fold_states=fold_states,
                      losses=losses, fold_losses=fold_losses, parent_weights_sha256=file_sha256(ROOT / 'checkpoints/semantic_ranker/model.safetensors'))
        (output / 'head.json').write_text(json.dumps(bundle, indent=2) + '\n')
        reloaded = CandidateEvidenceHead.restore(json.loads((output / 'head.json').read_text())['state'])
        metrics, predictions = {}, {}
        crossfit_gate = oof_metrics['source']['after'] > oof_metrics['source']['before'] and oof_metrics['retention']['after'] >= oof_metrics['retention']['before']
        if crossfit_gate:
            for role, rows in pools.items():
                before, after = predict(initial, rows, neutral), predict(model, rows, neutral)
                if predict(reloaded, rows, neutral) != after:
                    raise ValueError('saved evidence head does not reproduce predictions')
                predictions[role] = dict(before=before, after=after)
                metrics[role] = compare(before, after)
        else:
            if predict(reloaded, units, neutral) != predict(model, units, neutral):
                raise ValueError('saved evidence head does not reproduce TRAIN predictions')
        (output / 'predictions.json').write_text(json.dumps(dict(out_of_fold=oof, evaluation=predictions), ensure_ascii=False) + '\n')
        advance = (crossfit_gate and metrics['source_hold']['after'] > metrics['source_hold']['before']
                   and all(m['after'] >= m['before'] for role, m in metrics.items() if role != 'source_hold'))
        verify_pins(pins)
        result = dict(completed=True, production_changed=False, neural_encoder_weights_changed=False,
                      trainable_parameters=sum(p.numel() for p in model.parameters()), fit_rows=len(units),
                      updates_per_fit=RECIPE['steps'], fits=5, features=FEATURES, recipe=RECIPE,
                      oof_metrics=oof_metrics, crossfit_gate=crossfit_gate, evaluation_run=crossfit_gate,
                      metrics=metrics, advance_to_runtime=advance, reload_predictions_equal=True,
                      scope='TRAIN component cross-fit; existing diagnostic holds only after crossfit gate; not fresh generalization',
                      pins=pins, hashes={name: file_sha256(output / name) for name in ('selection.json', 'head.json', 'predictions.json')},
                      elapsed_s=time.monotonic() - started, peak_rss_bytes=rss_bytes())
        (output / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({k: result[k] for k in ('completed', 'oof_metrics', 'metrics', 'advance_to_runtime', 'elapsed_s', 'peak_rss_bytes')}, ensure_ascii=False), flush=True)
    except BaseException as exc:
        (output / 'failure.json').write_text(json.dumps(dict(completed=False, error=repr(exc))))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output.resolve())
