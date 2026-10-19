"""Pure calibration and prediction for the experimental final-choice gate.

The gate is fitted only from role-tagged natural calibration disagreements.
Candidate generation, selector scoring, and evaluation remain outside this
module.
"""
from __future__ import annotations

import json
import math

import torch
import torch.nn.functional as F

from scripts.final_choice_policy import FEATURE_NAMES


CALIBRATION_ROLE = "natural_calibration"
DIMENSIONS = len(FEATURE_NAMES)
SEED = 23
STEPS = 500
LEARNING_RATE = 0.05
ACCEPT_WEIGHT = 1.0
STAY_WEIGHT = 5.0
L2_WEIGHT = 0.01
CLIP = 8.0
THRESHOLD = 0.5
MIN_CLASS_GROUPS = 8


class GateLabelsInsufficient(ValueError):
    """Raised when calibration labels cannot support the frozen gate fit."""


def _finite_vector(values, name):
    if (not isinstance(values, (list, tuple)) or len(values) != DIMENSIONS
            or any(type(value) not in (int, float) or not math.isfinite(value)
                   for value in values)):
        raise ValueError(f"{name} must contain {DIMENSIONS} finite numbers")
    return [float(value) for value in values]


def _calibration_rows(rows):
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError("gate calibration rows must be a non-empty sequence")
    parsed = []
    for row in rows:
        if not isinstance(row, dict) or row.get("role") != CALIBRATION_ROLE:
            raise ValueError("gate fitting accepts only natural_calibration rows")
        if row.get("reason") != "eligible_disagreement":
            raise ValueError("gate fitting accepts only eligible disagreements")
        group_id = row.get("group_id")
        parent_accepted = row.get("parent_accepted")
        proposal_accepted = row.get("proposal_accepted")
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("gate calibration row requires a group_id")
        if type(parent_accepted) is not bool or type(proposal_accepted) is not bool:
            raise ValueError("gate calibration acceptance flags must be booleans")
        features = _finite_vector(row.get("features"), "gate features")
        # Accept only when switching repairs a parent error. Every other
        # eligible disagreement is a stay label.
        label = int(not parent_accepted and proposal_accepted)
        parsed.append((features, label, group_id))
    return parsed


def _scaler(matrix):
    mean = matrix.mean(dim=0)
    scale = matrix.std(dim=0, unbiased=False)
    constant_indices = (scale == 0).nonzero(as_tuple=False).flatten().tolist()
    if constant_indices:
        scale = scale.clone()
        scale[constant_indices] = 1.0
    transformed = torch.clamp((matrix - mean) / scale, -CLIP, CLIP)
    if not torch.isfinite(transformed).all():
        raise ValueError("gate feature scaling produced a non-finite value")
    return mean, scale, transformed, constant_indices


def _settings():
    return {
        "role": CALIBRATION_ROLE,
        "dimensions": DIMENSIONS,
        "feature_names": list(FEATURE_NAMES),
        "seed": SEED,
        "initialization": "zeros",
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "steps": STEPS,
        "full_batch": True,
        "accept_weight": ACCEPT_WEIGHT,
        "stay_weight": STAY_WEIGHT,
        "l2_weight": L2_WEIGHT,
        "l2_includes_bias": False,
        "clip": [-CLIP, CLIP],
        "threshold": THRESHOLD,
        "minimum_distinct_groups_per_class": MIN_CLASS_GROUPS,
        "standard_deviation": "population",
    }


def fit_gate(rows):
    """Fit the frozen Linear(6, 1) gate from eligible CAL disagreements."""
    parsed = _calibration_rows(rows)
    accept_groups = {group for _, label, group in parsed if label == 1}
    stay_groups = {group for _, label, group in parsed if label == 0}
    if min(len(accept_groups), len(stay_groups)) < MIN_CLASS_GROUPS:
        raise GateLabelsInsufficient(
            "gate labels insufficient: accept and stay each require at least "
            f"{MIN_CLASS_GROUPS} distinct natural_calibration groups"
        )

    matrix = torch.tensor([features for features, _, _ in parsed], dtype=torch.float64)
    labels = torch.tensor([label for _, label, _ in parsed], dtype=torch.float64)
    mean, scale, transformed, constant_indices = _scaler(matrix)

    row_weights = torch.where(labels == 1, ACCEPT_WEIGHT, STAY_WEIGHT)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        model = torch.nn.Linear(DIMENSIONS, 1, bias=True, dtype=torch.float64)
        torch.nn.init.zeros_(model.weight)
        torch.nn.init.zeros_(model.bias)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        for _ in range(STEPS):
            optimizer.zero_grad()
            logits = model(transformed).squeeze(1)
            loss = (F.binary_cross_entropy_with_logits(
                logits, labels, reduction="none"
            ) * row_weights).mean() + L2_WEIGHT * model.weight.square().sum()
            if not torch.isfinite(loss):
                raise ValueError("gate training produced a non-finite loss")
            loss.backward()
            optimizer.step()
            if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
                raise ValueError("gate training produced non-finite coefficients")

    counts = {
        "rows": len(parsed),
        "groups": len({group for _, _, group in parsed}),
        "accept_rows": int(labels.sum().item()),
        "stay_rows": int((labels == 0).sum().item()),
        "accept_groups": len(accept_groups),
        "stay_groups": len(stay_groups),
    }
    artifact = {
        "schema": "final-choice-linear-gate-v1",
        "weights": model.weight.detach()[0].tolist(),
        "bias": float(model.bias.detach()[0]),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "constant_features": [FEATURE_NAMES[index] for index in constant_indices],
        "counts": counts,
        "settings": _settings(),
    }
    return validate_gate(artifact)


def validate_gate(artifact):
    """Validate and return a JSON-serializable gate artifact copy."""
    if not isinstance(artifact, dict) or artifact.get("schema") != "final-choice-linear-gate-v1":
        raise ValueError("unknown final-choice gate schema")
    weights = _finite_vector(artifact.get("weights"), "gate weights")
    mean = _finite_vector(artifact.get("mean"), "gate mean")
    scale = _finite_vector(artifact.get("scale"), "gate scale")
    bias = artifact.get("bias")
    if type(bias) not in (int, float) or not math.isfinite(bias):
        raise ValueError("gate bias must be finite")
    if any(value <= 0 for value in scale):
        raise ValueError("gate scales must be positive")
    constants = artifact.get("constant_features")
    if (not isinstance(constants, list) or len(constants) != len(set(constants))
            or any(name not in FEATURE_NAMES for name in constants)):
        raise ValueError("gate constant feature names differ")
    counts = artifact.get("counts")
    settings = artifact.get("settings")
    if not isinstance(counts, dict) or settings != _settings():
        raise ValueError("gate counts or frozen settings differ")
    for key in ("rows", "groups", "accept_rows", "stay_rows",
                "accept_groups", "stay_groups"):
        if type(counts.get(key)) is not int or counts[key] < 0:
            raise ValueError("gate counts are malformed")
    if (counts["rows"] != counts["accept_rows"] + counts["stay_rows"]
            or counts["groups"] > counts["rows"]
            or counts["accept_groups"] > counts["accept_rows"]
            or counts["stay_groups"] > counts["stay_rows"]
            or not max(counts["accept_groups"], counts["stay_groups"])
            <= counts["groups"] <= counts["accept_groups"] + counts["stay_groups"]
            or min(counts["accept_groups"], counts["stay_groups"]) < MIN_CLASS_GROUPS):
        raise ValueError("gate label counts do not satisfy the fit contract")
    result = {
        "schema": artifact["schema"], "weights": weights, "bias": float(bias),
        "mean": mean, "scale": scale, "constant_features": list(constants),
        "counts": dict(counts), "settings": _settings(),
    }
    # This is an explicit part of the persistence contract.
    json.dumps(result, allow_nan=False)
    return result


def transform_features(artifact, features):
    gate = validate_gate(artifact)
    values = _finite_vector(features, "gate features")
    standardized = [(value - center) / scale
                    for value, center, scale in zip(values, gate["mean"], gate["scale"])]
    if any(not math.isfinite(value) for value in standardized):
        raise ValueError("gate feature transform produced a non-finite value")
    return [max(-CLIP, min(CLIP, value)) for value in standardized]


def predict_probability(artifact, features):
    gate = validate_gate(artifact)
    transformed = transform_features(gate, features)
    score = sum(weight * value for weight, value in zip(gate["weights"], transformed)) + gate["bias"]
    if not math.isfinite(score):
        raise ValueError("gate score is non-finite")
    if score >= 0:
        probability = 1.0 / (1.0 + math.exp(-score))
    else:
        exponent = math.exp(score)
        probability = exponent / (1.0 + exponent)
    if not math.isfinite(probability):
        raise ValueError("gate probability is non-finite")
    return probability


def predict_accept(artifact, features):
    return predict_probability(artifact, features) >= THRESHOLD


def dumps_gate(artifact):
    return json.dumps(validate_gate(artifact), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def loads_gate(value):
    if not isinstance(value, str):
        raise ValueError("serialized gate must be JSON text")
    return validate_gate(json.loads(value))
