"""Shared MLX scorer and optimizer primitives for long-context finite-choice LoRA."""
from __future__ import annotations

import hashlib
import math
from typing import Iterable, Sequence


SEED = 20260915
LORA_LAYERS = 4
LORA_KEYS = ("self_attn.q_proj", "self_attn.v_proj")
LORA_CONFIG = {
    "rank": 8,
    "scale": 20.0,
    "dropout": 0.0,
    "keys": list(LORA_KEYS),
}
EXPECTED_TRAINABLE_TENSORS = 16
EXPECTED_TRAINABLE_SCALARS = 327680
LEARNING_RATE = 1e-5
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
CLIP_NORM = 1.0


def validate_gold_positions(candidate_count: int, positions: Iterable[int]) -> tuple[int, ...]:
    """Return a canonical finite nonempty gold set for a 2..5 choice row."""
    if type(candidate_count) is not int or not 2 <= candidate_count <= 5:
        raise ValueError("candidate_count must be an integer from 2 through 5")
    values = tuple(positions)
    if not values or any(type(value) is not int for value in values):
        raise ValueError("allowed gold positions must be nonempty integers")
    if len(set(values)) != len(values) or any(not 0 <= value < candidate_count for value in values):
        raise ValueError("allowed gold positions must be unique and inside the candidate set")
    return tuple(sorted(values))


def reference_log_mass_loss(scores: Sequence[float], allowed_gold_positions: Iterable[int]) -> float:
    """High-precision Python reference for the finite candidate log-mass loss."""
    values = tuple(float(value) for value in scores)
    if not 2 <= len(values) <= 5 or not all(math.isfinite(value) for value in values):
        raise ValueError("scores must contain two through five finite values")
    gold = validate_gold_positions(len(values), allowed_gold_positions)

    def logsumexp(items: Sequence[float]) -> float:
        maximum = max(items)
        return maximum + math.log(sum(math.exp(item - maximum) for item in items))

    return logsumexp(values) - logsumexp(tuple(values[index] for index in gold))


def reference_average_and_clip(microbatch_gradients: Sequence[Sequence[float]],
                               max_norm: float = CLIP_NORM) -> tuple[tuple[float, ...], float]:
    """Pure reference for actual-count averaging followed by one global clip."""
    rows = tuple(tuple(float(value) for value in row) for row in microbatch_gradients)
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("microbatch gradients must be a nonempty rectangular matrix")
    if not all(math.isfinite(value) for row in rows for value in row):
        raise ValueError("microbatch gradients must be finite")
    averaged = tuple(sum(row[index] for row in rows) / len(rows)
                     for index in range(len(rows[0])))
    norm = math.sqrt(sum(value * value for value in averaged))
    if not math.isfinite(norm):
        raise FloatingPointError("averaged gradient norm is nonfinite")
    scale = min(1.0, max_norm / max(norm, 1e-12))
    return tuple(value * scale for value in averaged), norm


def mlx_log_mass_loss(mx, scores, allowed_gold_positions: Iterable[int]):
    """Compute float32 finite-N loss without materializing vocabulary-wide positions."""
    if getattr(scores, "ndim", None) != 1 or not 2 <= scores.shape[0] <= 5:
        raise ValueError("choice scores must be a rank-one tensor of length 2 through 5")
    gold = validate_gold_positions(scores.shape[0], allowed_gold_positions)
    values = scores.astype(mx.float32)
    return mx.logsumexp(values) - mx.logsumexp(values[mx.array(gold)])


def score_choices(mx, model, prompt_tokens: Sequence[int], digit_token_ids: Sequence[int]):
    """Score digit continuations from the last real prompt position, with no cache/padding."""
    tokens = tuple(prompt_tokens)
    digits = tuple(digit_token_ids)
    if not tokens or any(type(token) is not int for token in tokens):
        raise ValueError("prompt_tokens must be nonempty integers")
    if not 2 <= len(digits) <= 5 or any(type(token) is not int for token in digits):
        raise ValueError("digit_token_ids must contain two through five integers")
    inputs = mx.array([tokens])
    hidden = model.model(inputs)[:, -1:, :]
    if bool(model.args.tie_word_embeddings):
        logits = model.model.embed_tokens.as_linear(hidden)
    else:
        if not hasattr(model, "lm_head"):
            raise RuntimeError("untied model has no lm_head")
        logits = model.lm_head(hidden)
    return logits[0, 0, mx.array(digits)].astype(mx.float32)


def ordinary_forward_choice_scores(mx, model, prompt_tokens, digit_token_ids):
    """Diagnostic-only ordinary all-position forward for scorer equivalence checks."""
    logits = model(mx.array([tuple(prompt_tokens)]))
    return logits[0, -1, mx.array(tuple(digit_token_ids))].astype(mx.float32)


def expected_trainable_names() -> tuple[str, ...]:
    return tuple(
        f"model.layers.{layer}.self_attn.{projection}.{factor}"
        for layer in range(32, 36)
        for projection in ("q_proj", "v_proj")
        for factor in ("lora_a", "lora_b")
    )


def flat_tree(tree) -> dict:
    from mlx.utils import tree_flatten

    return dict(tree_flatten(tree))


def require_trainable_contract(model) -> dict:
    flat = flat_tree(model.trainable_parameters())
    expected = set(expected_trainable_names())
    if set(flat) != expected:
        missing, extra = sorted(expected - set(flat)), sorted(set(flat) - expected)
        raise RuntimeError(f"trainable LoRA keys differ; missing={missing}, extra={extra}")
    scalars = sum(array.size for array in flat.values())
    if len(flat) != EXPECTED_TRAINABLE_TENSORS or scalars != EXPECTED_TRAINABLE_SCALARS:
        raise RuntimeError(f"trainable LoRA shape differs: {len(flat)} tensors/{scalars} scalars")
    return flat


def configure_fresh_lora(mx, model):
    """Freeze a fresh base and seed immediately before converting its final q/v layers."""
    from mlx_lm.tuner.utils import linear_to_lora_layers

    model.freeze()
    mx.random.seed(SEED)
    linear_to_lora_layers(model, LORA_LAYERS, LORA_CONFIG)
    flat = require_trainable_contract(model)
    mx.eval(flat)
    return flat


def create_optimizer():
    import mlx.optimizers as optim

    return optim.Adam(
        learning_rate=LEARNING_RATE,
        betas=list(ADAM_BETAS),
        eps=ADAM_EPS,
        bias_correction=True,
    )


def require_matching_gradient_tree(model, gradients) -> dict:
    expected = set(require_trainable_contract(model))
    flat = flat_tree(gradients)
    if set(flat) != expected:
        raise RuntimeError("gradient tree does not exactly match the trainable adapter tree")
    return flat


def materialize_microbatch(mx, loss, gradients) -> None:
    mx.eval(loss, gradients)


def add_materialized_gradients(mx, accumulated, gradients):
    from mlx.utils import tree_map

    result = gradients if accumulated is None else tree_map(lambda left, right: left + right,
                                                             accumulated, gradients)
    mx.eval(result)
    return result


def average_and_clip_gradients(mx, gradients, microbatch_count: int,
                               max_norm: float = CLIP_NORM):
    """Average by the actual count and apply one global L2-norm clip."""
    from mlx.utils import tree_map

    if type(microbatch_count) is not int or microbatch_count < 1:
        raise ValueError("microbatch_count must be positive")
    if not math.isfinite(max_norm) or max_norm <= 0:
        raise ValueError("max_norm must be finite and positive")
    averaged = tree_map(lambda value: value / microbatch_count, gradients)
    squared = sum((mx.sum(value.astype(mx.float32) ** 2)
                   for value in flat_tree(averaged).values()), mx.array(0.0, mx.float32))
    norm = mx.sqrt(squared)
    scale = mx.minimum(mx.array(1.0, mx.float32), max_norm / mx.maximum(norm, 1e-12))
    clipped = tree_map(lambda value: value * scale.astype(value.dtype), averaged)
    mx.eval(norm, clipped)
    if (not bool(mx.isfinite(norm).item()) or any(
            not bool(mx.all(mx.isfinite(value)).item())
            for value in flat_tree(clipped).values())):
        raise FloatingPointError("averaged/clipped gradients are nonfinite")
    return clipped, norm


def apply_materialized_update(mx, model, optimizer, gradients) -> None:
    optimizer.update(model, gradients)
    mx.eval(model.state, optimizer.state)
    if any(not bool(mx.all(mx.isfinite(value)).item())
           for value in require_trainable_contract(model).values()):
        raise FloatingPointError("updated adapter parameters are nonfinite")


def tensor_hashes(arrays: dict) -> dict[str, str]:
    """Hash materialized tensor bytes with dtype/shape domain separation."""
    import numpy as np

    result = {}
    for name, value in sorted(arrays.items()):
        raw = np.asarray(value).tobytes(order="C")
        header = f"{value.dtype}:{tuple(value.shape)}:".encode()
        result[name] = hashlib.sha256(header + raw).hexdigest()
    return result


def save_adapter(mx, model, path) -> dict[str, str]:
    flat = require_trainable_contract(model)
    mx.eval(flat)
    mx.save_safetensors(str(path), flat)
    return tensor_hashes(flat)


__all__ = [
    "ADAM_BETAS", "ADAM_EPS", "CLIP_NORM", "EXPECTED_TRAINABLE_SCALARS",
    "EXPECTED_TRAINABLE_TENSORS", "LEARNING_RATE", "LORA_CONFIG", "LORA_LAYERS",
    "SEED", "add_materialized_gradients", "apply_materialized_update",
    "average_and_clip_gradients", "configure_fresh_lora", "create_optimizer",
    "expected_trainable_names", "materialize_microbatch", "mlx_log_mass_loss",
    "ordinary_forward_choice_scores", "reference_log_mass_loss",
    "reference_average_and_clip",
    "require_matching_gradient_tree", "require_trainable_contract", "save_adapter",
    "score_choices", "tensor_hashes", "validate_gold_positions",
]
