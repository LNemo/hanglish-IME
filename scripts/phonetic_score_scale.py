"""Numerical helpers for the frozen TRAIN-only score-scale comparison."""
from __future__ import annotations

import hashlib
import numpy as np

ARMS = ("fixed_unscaled", "learned_unscaled", "fixed_scaled", "learned_scaled")
ANCHOR = np.array([1., 0., 0., 0., 0.], dtype=np.float64)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def fold_for(component):
    need(isinstance(component, str) and component, "component ID required")
    digest = hashlib.sha256(("phonetic-score-scale-v1:" + component).encode()).hexdigest()
    return int(digest, 16) % 4


def arrays(rows):
    need(rows and len({r["id"] for r in rows}) == len(rows), "empty/duplicate row IDs")
    width = max(len(row["candidates"]) for row in rows)
    base = np.zeros((len(rows), width), dtype=np.float64)
    features = np.zeros((len(rows), width, 4), dtype=np.float64)
    valid = np.zeros(base.shape, dtype=bool)
    positives = np.zeros(base.shape, dtype=bool)
    for index, row in enumerate(rows):
        size = len(row["candidates"])
        need(size and size == len(row["base_scores"]) == len(row["features"])
             and len(set(row["candidates"])) == size, "candidate domain differs")
        selected = row["positive_indices"]
        need(selected and all(type(i) is int and 0 <= i < size for i in selected),
             "positive domain differs")
        base[index, :size] = row["base_scores"]
        features[index, :size] = row["features"]
        valid[index, :size] = True
        positives[index, selected] = True
    need(np.isfinite(base).all() and np.isfinite(features).all(), "nonfinite input")
    folds = np.array([fold_for(r["component_id"]) for r in rows])
    need(set(folds) == set(range(4)), "all four component folds required")
    return base, features, valid, positives, folds


def feature_scale(features, valid, train_indices, scaled):
    if not scaled:
        return np.ones(4, dtype=np.float64)
    need(len(train_indices) > 0, "training indices empty")
    selected = features[train_indices]
    mask = valid[train_indices]
    counts = mask.sum(axis=1)
    need((counts > 0).all(), "empty candidate set")
    means = (selected * mask[:, :, None]).sum(axis=1) / counts[:, None]
    variances = (((selected - means[:, None, :]) ** 2) * mask[:, :, None]).sum(axis=1) / counts[:, None]
    scale = np.sqrt(variances.mean(axis=0))
    need(np.isfinite(scale).all(), "nonfinite feature scale")
    return np.where(scale > 1e-12, scale, 1.)


def design(base, features, scale):
    need(np.asarray(scale).shape == (4,) and np.isfinite(scale).all()
         and (np.asarray(scale) > 0).all(), "invalid feature scale")
    return np.concatenate((base[:, :, None], features / scale), axis=2)


def losses_and_scores(parameters, matrix, valid, positives):
    need(np.asarray(parameters).shape == (5,) and np.isfinite(parameters).all()
         and parameters[0] >= 0, "invalid parameters")
    need(matrix.shape[:2] == valid.shape == positives.shape
         and matrix.shape[2] == 5 and len(matrix) > 0
         and valid.any(axis=1).all() and positives.any(axis=1).all()
         and not (positives & ~valid).any(), "invalid loss masks")
    scores = np.where(valid, np.einsum("rcf,f->rc", matrix, parameters), -np.inf)
    maximum = scores.max(axis=1, keepdims=True)
    all_exp = np.exp(scores - maximum)
    all_sum = all_exp.sum(axis=1, keepdims=True)
    pos_scores = np.where(positives, scores, -np.inf)
    pos_maximum = pos_scores.max(axis=1, keepdims=True)
    pos_exp = np.exp(pos_scores - pos_maximum)
    pos_sum = pos_exp.sum(axis=1, keepdims=True)
    losses = (maximum + np.log(all_sum) - pos_maximum - np.log(pos_sum)).ravel()
    delta = all_exp / all_sum - pos_exp / pos_sum
    need(np.isfinite(losses).all() and np.isfinite(delta).all(), "nonfinite loss")
    return losses, scores, delta


def objective(parameters, matrix, valid, positives, learn_alpha):
    losses, _, delta = losses_and_scores(parameters, matrix, valid, positives)
    difference = parameters - ANCHOR
    loss = float(losses.mean() + .01 * np.dot(difference, difference))
    gradient = np.einsum("rc,rcf->f", delta, matrix) / len(matrix) + .02 * difference
    if not learn_alpha:
        gradient[0] = 0.
    need(np.isfinite(gradient).all(), "nonfinite gradient")
    return loss, gradient


def adam_step(parameters, first, second, step, matrix, valid, positives, learn_alpha):
    loss, gradient = objective(parameters, matrix, valid, positives, learn_alpha)
    norm = float(np.linalg.norm(gradient))
    clipped = gradient / norm if norm > 1 else gradient
    first = .9 * first + .1 * clipped
    second = .999 * second + .001 * clipped ** 2
    updated = parameters - .05 * (first / (1 - .9 ** step)) / (
        np.sqrt(second / (1 - .999 ** step)) + 1e-8)
    updated[0] = max(0., updated[0]) if learn_alpha else 1.
    need(np.isfinite(updated).all(), "nonfinite update")
    return updated, first, second, {"step": step, "loss_before_update": loss,
        "gradient": gradient.tolist(), "gradient_norm_before_clip": norm,
        "parameters_after_update": updated.tolist()}


def predictions(rows, indices, parameters, matrix, valid, positives):
    losses, scores, _ = losses_and_scores(parameters, matrix, valid, positives)
    result = []
    for local, index in enumerate(indices):
        row = rows[index]
        choose = lambda values: min(range(len(row["candidates"])),
                                    key=lambda i: (-values[i], row["candidates"][i]))
        selected = choose(scores[local])
        baseline = choose(row["base_scores"])
        positive = row["positive_indices"]
        best_gold = max(scores[local, i] for i in positive)
        result.append({"id": row["id"], "component_id": row["component_id"],
            "selected_index": selected, "base_selected_index": baseline,
            "correct": selected in positive, "base_correct": baseline in positive,
            "cross_entropy": float(losses[local]),
            "gold_score_gap": float(scores[local, selected] - best_gold)})
    return result


def summary(predicted):
    need(predicted and len({r["id"] for r in predicted}) == len(predicted), "prediction coverage differs")
    return {"rows": len(predicted), "top1": sum(r["correct"] for r in predicted),
            "base_top1": sum(r["base_correct"] for r in predicted),
            "corrections": sum(r["correct"] and not r["base_correct"] for r in predicted),
            "regressions": sum(not r["correct"] and r["base_correct"] for r in predicted),
            "cross_entropy": sum(r["cross_entropy"] for r in predicted) / len(predicted),
            "mean_gold_score_gap": sum(r["gold_score_gap"] for r in predicted) / len(predicted)}
