"""Balanced repair-unit gradients for the approved pairwise LoRA recipe.

MLX imports remain lazy. A repair and its preservation controls form one unit;
optimizer accumulation counts units, never their variable component counts.
"""
from __future__ import annotations

import math
import time

from scripts import long_context_lora_model as shared


def unit_weights(targets):
    targets = tuple(targets)
    if len(targets) not in (3, 4) or targets != ('SWITCH', *('KEEP',) * (len(targets) - 1)):
        raise ValueError('a unit requires SWITCH followed by two or three KEEP components')
    return (0.5, *(0.5 / (len(targets) - 1) for _ in targets[1:]))


def reference_unit_loss(component_losses, targets):
    weights = unit_weights(targets)
    losses = tuple(component_losses)
    if len(losses) != len(weights) or not all(math.isfinite(value) for value in losses):
        raise ValueError('unit losses must be finite and aligned with components')
    return math.fsum(weight * value for weight, value in zip(weights, losses))


def make_component_value_and_grad(mx, model):
    from mlx import nn

    def loss(current_model, tokens, target):
        if target not in ('KEEP', 'SWITCH'):
            raise ValueError('unknown pair target')
        scores = shared.score_choices(mx, current_model, tokens, (16, 17))
        return shared.mlx_log_mass_loss(mx, scores, (0 if target == 'KEEP' else 1,))

    return nn.value_and_grad(model, loss)


def require_finite_tree(mx, tree, label):
    arrays = shared.flat_tree(tree)
    if not arrays or any(not bool(mx.all(mx.isfinite(value)).item())
                         for value in arrays.values()):
        raise FloatingPointError(f'{label} is empty or nonfinite')


def materialize_unit(mx, model, components, value_and_grad, observe=None):
    """Return balanced loss/gradient after materializing each component in order.

    Components contain source-qualified tokens and a separate target. The
    optional observer runs after materialization so resource observations refer
    to executed work. No optimizer step or clipping occurs inside a unit.
    """
    from mlx.utils import tree_map

    components = tuple(components)
    weights = unit_weights(component['target'] for component in components)
    gradient = None
    losses, records = [], []
    started = time.monotonic()
    for index, (component, weight) in enumerate(zip(components, weights)):
        component_start = time.monotonic()
        loss, unscaled = value_and_grad(model, component['tokens'], component['target'])
        shared.materialize_microbatch(mx, loss, unscaled)
        shared.require_matching_gradient_tree(model, unscaled)
        value = float(loss.item())
        if not math.isfinite(value):
            raise FloatingPointError('component loss is nonfinite')
        require_finite_tree(mx, unscaled, 'component gradient')
        scaled = tree_map(lambda array: array * weight, unscaled)
        mx.eval(scaled)
        require_finite_tree(mx, scaled, 'weighted component gradient')
        gradient = shared.add_materialized_gradients(mx, gradient, scaled)
        require_finite_tree(mx, gradient, 'unit gradient accumulator')
        losses.append(value)
        record = {'component_index': index, 'target': component['target'],
                  'weight': weight, 'loss': value,
                  'gradient_seconds': time.monotonic() - component_start}
        records.append(record)
        if observe is not None:
            observe(record)
        del loss, unscaled, scaled
    mx.eval(gradient)
    return {'loss': reference_unit_loss(losses, [item['target'] for item in components]),
            'gradient': gradient, 'components': records,
            'unit_gradient_seconds': time.monotonic() - started}


def apply_unit_update(mx, model, optimizer, accumulated_gradient, unit_count):
    """Average the actual full/tail unit count, clip once, then update Adam."""
    if type(unit_count) is not int or unit_count not in (3, 4):
        raise ValueError('approved updates contain four units or the final three-unit tail')
    gradient, norm = shared.average_and_clip_gradients(mx, accumulated_gradient, unit_count)
    shared.apply_materialized_update(mx, model, optimizer, gradient)
    require_finite_tree(mx, optimizer.state, 'updated optimizer state')
    return float(norm.item())
