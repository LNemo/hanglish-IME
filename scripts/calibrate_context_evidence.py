"""Learn one context-free preference correction on frozen TRAIN candidates."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from hanglish.models.semantic_ranker import SemanticRanker
from scripts.evaluate_ensemble_regression import file_sha256, rss_bytes, verify_pins

SOURCE = ROOT / '.omx/artifacts/runtime-collision-ranker-training-v1'
SOURCE_SHA = '163466c149944fa2c032b731cd01a343b4783b9a34bdf5de5bab04a804a850d6'
PARENT = ROOT / 'checkpoints/semantic_ranker'
RECIPE = dict(folds=4, salt='context-evidence-doc-v1', steps=300, learning_rate=.03,
              l2=.01, minimum_alpha=0., maximum_alpha=1., temperature=2., base_weight=.15,
              max_seconds=600, max_rss_bytes=3 * 1024 ** 3)


def has_context(context):
    if '[MASK]' in context:
        left, _, right = context.rpartition('[MASK]')
        return bool((left + right).strip())
    return bool(context.strip())


def corrected_scores(row, neutral, alpha):
    if not 0 <= alpha <= 1:
        raise ValueError('correction coefficient outside bounds')
    strength = alpha if has_context(row['context']) else 0.
    return [s + .15 * b - strength * neutral[c]
            for s, b, c in zip(row['teacher_scores'], row['base_scores'], row['candidates'])]


def pack(rows, neutral):
    if not rows:
        raise ValueError('empty training partition')
    width = max(len(r['candidates']) for r in rows)
    base = torch.full((len(rows), width), -1e9, dtype=torch.float64)
    prior = torch.zeros_like(base)
    positive = torch.zeros_like(base, dtype=torch.bool)
    retention = torch.zeros(len(rows), dtype=torch.float64)
    for i, row in enumerate(rows):
        count = len(row['candidates'])
        if not count or len(row['teacher_scores']) != count or len(row['base_scores']) != count:
            raise ValueError('invalid training features')
        values = corrected_scores(row, neutral, 0.)
        base[i, :count] = torch.tensor(values, dtype=torch.float64)
        if has_context(row['context']):
            prior[i, :count] = torch.tensor([neutral[c] for c in row['candidates']], dtype=torch.float64)
        positives = [j for j, c in enumerate(row['candidates']) if c in row['expected']]
        if row['kind'] == 'source' and not 0 < len(positives) < count:
            raise ValueError('source group is not informative')
        if row['kind'] not in ('source', 'retention'):
            raise ValueError('unknown training role')
        positive[i, positives or list(range(count))] = True
        retention[i] = row['kind'] == 'retention'
    if not torch.isfinite(base).all() or not torch.isfinite(prior).all():
        raise ValueError('nonfinite training features')
    return base, prior, positive, retention


def objective(alpha, packed):
    base, prior, positive, retention = packed
    scores = base - alpha * prior
    loss = torch.logsumexp(scores, 1) - torch.logsumexp(scores.masked_fill(~positive, -1e9), 1)
    temperature = RECIPE['temperature']
    teacher = torch.softmax(base / temperature, 1)
    kl = (teacher * (torch.log_softmax(base / temperature, 1) - torch.log_softmax(scores / temperature, 1))).sum(1)
    return (loss + retention * temperature ** 2 * kl).mean() + RECIPE['l2'] * alpha.square()


def fit(rows, neutral, guard=lambda: None):
    packed = pack(rows, neutral)
    alpha = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([alpha], lr=RECIPE['learning_rate'])
    initial = float(objective(alpha, packed).detach())
    for _ in range(RECIPE['steps']):
        optimizer.zero_grad(set_to_none=True)
        loss = objective(alpha, packed)
        if not torch.isfinite(loss):
            raise ValueError('nonfinite calibration loss')
        loss.backward()
        if not torch.isfinite(alpha.grad):
            raise ValueError('nonfinite calibration gradient')
        optimizer.step()
        with torch.no_grad():
            alpha.clamp_(0., 1.)
        guard()
    return dict(alpha=float(alpha.detach()), initial_loss=initial,
                final_loss=float(objective(alpha, packed).detach()), steps=RECIPE['steps'])


def predict(rows, neutral, alpha):
    results = []
    for row in rows:
        if row.get('kind') == 'fixed':
            result = row['fixed_result']
        else:
            scores = corrected_scores(row, neutral, alpha)
            index = min(range(len(scores)), key=lambda i: (-scores[i], row['candidates'][i]))
            result = row['candidates'][index]
        results.append(dict(id=row['id'], result=result,
                            correct=isinstance(result, str) and result in row['expected']))
    return results


def compare(before, after):
    if [r['id'] for r in before] != [r['id'] for r in after]:
        raise ValueError('prediction identity mismatch')
    return dict(rows=len(before), before=sum(r['correct'] for r in before), after=sum(r['correct'] for r in after),
                fix_ids=[b['id'] for a, b in zip(before, after) if not a['correct'] and b['correct']],
                loss_ids=[b['id'] for a, b in zip(before, after) if a['correct'] and not b['correct']])


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    def guard():
        if time.monotonic() - started > RECIPE['max_seconds'] or rss_bytes() > RECIPE['max_rss_bytes']:
            raise RuntimeError('context evidence calibration resource cap exceeded')

    try:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        if file_sha256(SOURCE / 'report.json') != SOURCE_SHA:
            raise ValueError('source training report changed')
        source = json.loads((SOURCE / 'report.json').read_text())
        verify_pins(source['pins'])
        for name in ('selection', 'predictions'):
            if file_sha256(SOURCE / (name + '.json')) != source[name + '_sha256']:
                raise ValueError('source saved features changed')
        selection = json.loads((SOURCE / 'selection.json').read_text())
        # Only unchanged production scores, never the failed trained model's scores.
        before = json.loads((SOURCE / 'predictions.json').read_text())['before']
        fit_rows = selection['fit']
        pools = selection['evaluation']
        for role, rows in pools.items():
            for row, prediction in zip(rows, before[role]['predictions']):
                if row['id'] != prediction['id']:
                    raise ValueError('baseline evaluation alignment differs')
                if row.get('kind') != 'fixed':
                    row['teacher_scores'] = prediction['scores']
        plan_path = ROOT / '.omx/plans/expanded-word-split-v1.json'
        plan = json.loads(plan_path.read_text())
        folds = {}
        for row in fit_rows:
            component = plan['assignments'][row['id']]['component'] if row['kind'] == 'source' else row['group_id']
            folds[row['id']] = int(hashlib.sha256((RECIPE['salt'] + '\0' + component).encode()).hexdigest(), 16) % RECIPE['folds']
        if set(folds.values()) != set(range(RECIPE['folds'])):
            raise ValueError('empty cross-fit fold')
        pins = dict(source['pins'])
        for path in (Path(__file__), SOURCE / 'report.json', SOURCE / 'selection.json', SOURCE / 'predictions.json', plan_path,
                     ROOT / '.omx/plans/context-evidence-calibration-v1.md'):
            pins[str(path.relative_to(ROOT))] = file_sha256(path)
        verify_pins(pins)
        (output / 'selection.json').write_text(json.dumps(dict(fit=fit_rows, evaluation=pools, folds=folds), ensure_ascii=False) + '\n')
        candidates = sorted({c for row in fit_rows for c in row['candidates']} |
                            {c for rows in pools.values() for row in rows if row.get('kind') != 'fixed' for c in row['candidates']})
        ranker = SemanticRanker.load(PARENT)
        neutral = {}
        with (output / 'neutral-capture.jsonl').open('x') as stream:
            for start in range(0, len(candidates), 128):
                batch = candidates[start:start + 128]
                values = ranker.score_pairs(['[MASK]'] * len(batch), batch)
                for candidate, score in zip(batch, values):
                    neutral[candidate] = score
                    stream.write(json.dumps(dict(candidate=candidate, score=score), ensure_ascii=False) + '\n')
                stream.flush()
                guard()
        del ranker
        print(json.dumps(dict(neutral_candidates=len(neutral), elapsed_s=time.monotonic() - started)), flush=True)
        fold_fits, out_of_fold = {}, {}
        for fold in range(RECIPE['folds']):
            training = [r for r in fit_rows if folds[r['id']] != fold]
            validation = [r for r in fit_rows if folds[r['id']] == fold]
            fitted = fit(training, neutral, guard)
            fold_fits[str(fold)] = {**fitted, 'fit_rows': len(training), 'held_rows': len(validation)}
            out_of_fold.update({r['id']: r for r in predict(validation, neutral, fitted['alpha'])})
        oof = {kind: compare(predict([r for r in fit_rows if r['kind'] == kind], neutral, 0.),
                             [out_of_fold[r['id']] for r in fit_rows if r['kind'] == kind]) for kind in ('source', 'retention')}
        fitted = fit(fit_rows, neutral, guard)
        (output / 'coefficient.json').write_text(json.dumps(dict(recipe=RECIPE, fitted=fitted, fold_fits=fold_fits), indent=2) + '\n')
        metrics, predictions = {}, {}
        for role, rows in pools.items():
            parent = predict(rows, neutral, 0.)
            if parent != [{k: r[k] for k in ('id', 'result', 'correct')} for r in before[role]['predictions']]:
                raise ValueError('zero coefficient does not reproduce production baseline')
            after = predict(rows, neutral, fitted['alpha'])
            predictions[role] = dict(before=parent, after=after)
            metrics[role] = compare(parent, after)
        advance = (fitted['alpha'] > 0 and oof['source']['after'] > oof['source']['before']
                   and oof['retention']['after'] >= oof['retention']['before']
                   and metrics['source_hold']['after'] > metrics['source_hold']['before']
                   and all(m['after'] >= m['before'] for role, m in metrics.items() if role != 'source_hold'))
        (output / 'predictions.json').write_text(json.dumps(dict(evaluation=predictions, out_of_fold=out_of_fold), ensure_ascii=False) + '\n')
        verify_pins(pins)
        report = dict(completed=True, production_changed=False, neural_weights_changed=False,
                      calibrated_parameter_count=1, training_rows=len(fit_rows), neutral_candidates=len(neutral),
                      alpha=fitted['alpha'], fit=fitted, fold_fits=fold_fits, oof=oof, metrics=metrics,
                      advance_to_runtime=advance, pins=pins, recipe=RECIPE,
                      scope='TRAIN document cross-fit and existing diagnostic holdouts; not fresh generalization',
                      hashes={name: file_sha256(output / name) for name in ('selection.json', 'neutral-capture.jsonl', 'coefficient.json', 'predictions.json')},
                      elapsed_s=time.monotonic() - started, peak_rss_bytes=rss_bytes())
        (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({k: report[k] for k in ('completed', 'alpha', 'oof', 'metrics', 'advance_to_runtime', 'elapsed_s', 'peak_rss_bytes')}, ensure_ascii=False), flush=True)
    except BaseException as exc:
        (output / 'failure.json').write_text(json.dumps(dict(completed=False, error=repr(exc))))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output.resolve())
