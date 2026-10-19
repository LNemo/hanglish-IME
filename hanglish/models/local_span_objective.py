"""Local conditional-preference loss for ``SourceSpanResidualRanker``.

The existing ranker remains the owner of every trainable tensor.  This module
replays its forward calculation once so the same per-span logits can support a
strictly local auxiliary objective without loading a parent model twice or
changing any checkpoint key.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from hanglish.models.source_span_residual import (
    FrozenCandidates,
    SourceSpanResidualRanker,
    source_byte_offsets,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


_SIDE_FIELDS = {
    'path_id', 'span_indices', 'output_start', 'output_end', 'text',
    'actions', 'unresolved_ascii_output',
}
_PREFERENCE_FIELDS = {
    'preference_id', 'input_start', 'input_end', 'source',
    'preferred_index', 'dispreferred_index', 'prefix', 'suffix',
    'preferred_text', 'dispreferred_text', 'witnesses',
}


def _validate_features(model, groups, candidates, features):
    require(isinstance(features, FrozenCandidates)
            and features.candidates == candidates
            and features.logits.shape == (len(groups),)
            and features.hidden.ndim == 3
            and features.hidden.shape[0] == len(groups)
            and features.hidden.shape[2] == model.target_hidden
            and len(features.token_spans) == len(groups)
            and all(len(spans) == features.hidden.shape[1]
                    for spans in features.token_spans)
            and not features.logits.requires_grad
            and not features.hidden.requires_grad
            and torch.isfinite(features.logits).all().item()
            and torch.isfinite(features.hidden).all().item(),
            'cached frozen features differ')


def _side_expected(path, indices):
    spans = [path.spans[index] for index in indices]
    return {
        'path_id': path.path_id,
        'span_indices': indices,
        'output_start': spans[0].output_start,
        'output_end': spans[-1].output_end,
        'text': path.text[spans[0].output_start:spans[-1].output_end],
        'actions': [span.action for span in spans],
        'unresolved_ascii_output': any(
            span.unresolved_ascii_output for span in spans),
    }


def _validate_side(side, path, start, end):
    require(isinstance(side, dict) and set(side) == _SIDE_FIELDS,
            'preference side fields differ')
    indices = side.get('span_indices')
    require(isinstance(indices, list) and bool(indices)
            and all(type(index) is int and 0 <= index < len(path.spans)
                    for index in indices)
            and indices == sorted(set(indices)),
            'preference span indices differ')
    expected_indices = [index for index, span in enumerate(path.spans)
                        if start <= span.input_start and span.input_end <= end]
    require(indices == expected_indices
            and path.spans[indices[0]].input_start == start
            and path.spans[indices[-1]].input_end == end,
            'preference source interval differs')
    require(side == _side_expected(path, indices),
            'preference side evidence differs')
    return indices


def _outside_signature(path, indices):
    selected = set(indices)
    return tuple((span.input_start, span.input_end, span.source, span.text)
                 for index, span in enumerate(path.spans)
                 if index not in selected)


def _validated_preferences(source, candidates, groups, preferences):
    require(isinstance(preferences, (list, tuple)),
            'preferences must be a sequence')
    path_maps = [{path.path_id: path for path in group} for group in groups]
    require(all(len(paths) == len(group)
                for paths, group in zip(path_maps, groups)),
            'duplicate candidate path ids differ')
    preference_ids = set()
    semantic_keys = set()
    validated = []
    for preference in preferences:
        require(isinstance(preference, dict)
                and set(preference) == _PREFERENCE_FIELDS,
                'preference fields differ')
        preference_id = preference.get('preference_id')
        require(isinstance(preference_id, str) and bool(preference_id)
                and preference_id not in preference_ids,
                'duplicate or malformed preference id')
        preference_ids.add(preference_id)
        start, end = preference.get('input_start'), preference.get('input_end')
        require(type(start) is int and type(end) is int
                and 0 <= start < end <= len(source)
                and not (start == 0 and end == len(source))
                and preference.get('source') == source[start:end],
                'preference source span differs')
        preferred_index = preference.get('preferred_index')
        dispreferred_index = preference.get('dispreferred_index')
        require(type(preferred_index) is int and type(dispreferred_index) is int
                and 0 <= preferred_index < len(candidates)
                and 0 <= dispreferred_index < len(candidates)
                and preferred_index != dispreferred_index,
                'preference candidate indices differ')
        prefix, suffix = preference.get('prefix'), preference.get('suffix')
        preferred_text = preference.get('preferred_text')
        dispreferred_text = preference.get('dispreferred_text')
        require(all(isinstance(value, str) for value in (
            prefix, suffix, preferred_text, dispreferred_text)),
            'preference text fields differ')
        semantic_key = (start, end, preferred_index, dispreferred_index,
                        prefix, suffix, preferred_text, dispreferred_text)
        require(semantic_key not in semantic_keys,
                'duplicate semantic preference')
        semantic_keys.add(semantic_key)
        witnesses = preference.get('witnesses')
        require(isinstance(witnesses, list) and bool(witnesses),
                'preference witnesses differ')
        witness_keys = set()
        checked = []
        for witness in witnesses:
            require(isinstance(witness, dict)
                    and set(witness) == {'preferred', 'dispreferred'},
                    'preference witness fields differ')
            left, right = witness['preferred'], witness['dispreferred']
            require(isinstance(left, dict) and isinstance(right, dict),
                    'preference witness sides differ')
            left_path = path_maps[preferred_index].get(left.get('path_id'))
            right_path = path_maps[dispreferred_index].get(right.get('path_id'))
            require(left_path is not None and right_path is not None,
                    'preference witness path differs')
            left_indices = _validate_side(left, left_path, start, end)
            right_indices = _validate_side(right, right_path, start, end)
            require(_outside_signature(left_path, left_indices)
                    == _outside_signature(right_path, right_indices),
                    'outside producer spans differ')
            require(left_path.text[:left['output_start']] == prefix
                    and left_path.text[left['output_end']:] == suffix
                    and right_path.text[:right['output_start']] == prefix
                    and right_path.text[right['output_end']:] == suffix
                    and left['text'] == preferred_text
                    and right['text'] == dispreferred_text
                    and left_path.text == candidates[preferred_index]
                    and right_path.text == candidates[dispreferred_index],
                    'preference output context differs')
            witness_key = (left_path.path_id, tuple(left_indices),
                           right_path.path_id, tuple(right_indices))
            require(witness_key not in witness_keys,
                    'duplicate preference witness')
            witness_keys.add(witness_key)
            checked.append((preferred_index, left_path.path_id, left_indices,
                            dispreferred_index, right_path.path_id, right_indices))
        validated.append(checked)
    return validated


def forward_with_local_preferences(model, source, groups, features, preferences):
    """Return the unchanged global ranking plus a local preference objective.

    ``local_loss`` is the row mean of preference losses; each preference loss is
    the mean ``softplus(-margin)`` over its distinct structural witnesses.
    ``local_margins`` contains the mean witness margin for each preference.
    Neither quantity includes the frozen parent logit or a raw candidate score.
    """
    require(isinstance(model, SourceSpanResidualRanker),
            'source-span residual model required')
    candidates = model._validate_paths(source, groups)
    _validate_features(model, groups, candidates, features)
    validated = _validated_preferences(
        source, candidates, groups, preferences)

    from hanglish.models.span_token_bridge import pooling_weights

    offsets = source_byte_offsets(source)
    source_ids = torch.tensor([byte + 1 for byte in source.encode('utf-8')],
                              dtype=torch.long, device=model.device).unsqueeze(0)
    states, _ = model.source_encoder(model.source_embedding(source_ids))
    states = states[0]
    hidden = features.hidden.to(model.device)
    residuals = []
    span_logits = {}
    for candidate_index, group in enumerate(groups):
        paths = {path.path_id: path for path in group}
        path_scores = []
        for path in (paths[path_id] for path_id in sorted(paths)):
            vectors = []
            for span in path.spans:
                source_vector = states[
                    offsets[span.input_start]:offsets[span.input_end]].mean(dim=0)
                weights = hidden.new_tensor(pooling_weights(
                    list(features.token_spans[candidate_index]),
                    span.output_start, span.output_end))
                mass = weights.sum()
                target_vector = ((hidden[candidate_index] * weights[:, None]).sum(dim=0)
                                 / mass.clamp_min(1))
                extra = hidden.new_tensor([
                    float(span.action == 'copy'), float(span.action == 'convert'),
                    float(span.unresolved_ascii_output), float(mass.item() == 0),
                    (span.input_end - span.input_start) / len(source),
                    (span.output_end - span.output_start) / max(1, len(path.text)),
                    span.input_start / len(source),
                    span.output_start / max(1, len(path.text)),
                ])
                vectors.append(torch.cat((source_vector, target_vector, extra)))
            values = model.segment_head(torch.stack(vectors)).reshape(-1)
            span_logits[(candidate_index, path.path_id)] = values
            path_scores.append(values.mean())
        residuals.append(torch.stack(path_scores).max())
    residuals = torch.stack(residuals)
    base_scores = features.logits.to(model.device)

    preference_losses = []
    preference_margins = []
    for witnesses in validated:
        witness_margins = []
        for (preferred_index, preferred_path_id, preferred_spans,
             dispreferred_index, dispreferred_path_id, dispreferred_spans) in witnesses:
            preferred = span_logits[(preferred_index, preferred_path_id)][preferred_spans].mean()
            dispreferred = span_logits[(dispreferred_index, dispreferred_path_id)][
                dispreferred_spans].mean()
            witness_margins.append(preferred - dispreferred)
        margins = torch.stack(witness_margins)
        preference_margins.append(margins.mean())
        preference_losses.append(F.softplus(-margins).mean())
    if preference_losses:
        local_loss = torch.stack(preference_losses).mean()
        local_margins = torch.stack(preference_margins)
    else:
        local_loss = residuals.sum() * 0.0
        local_margins = residuals.new_empty((0,))
    return {
        'scores': base_scores + residuals,
        'base_scores': base_scores,
        'residuals': residuals,
        'local_loss': local_loss,
        'local_margins': local_margins,
    }
