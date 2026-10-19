"""Candidate-independent source-edge residual ranking.

Every structural edge is scored once from its source span and an output text
encoded in isolation.  Paths only look up those shared scores, so an identical
``(input_start, input_end, text)`` edge has an identical contribution weight
and score wherever it occurs.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from hanglish.models.source_span_residual import (
    FrozenCandidates,
    SourceSpanResidualRanker,
    require,
    source_byte_offsets,
)


SCHEMA = 'shared-source-edge-residual-v1'


@dataclass(frozen=True)
class FrozenEdgeFeatures:
    """Frozen isolated-output vectors indexed by exact text."""

    texts: tuple[str, ...]
    vectors: torch.Tensor
    has_tokens: torch.Tensor


class SharedEdgeResidualRanker(SourceSpanResidualRanker):
    """Score shared source/output edges without candidate sentence context.

    ``path_audit`` in the forward result is a tuple per candidate.  Each entry
    is a tuple of dictionaries sorted by ``path_id`` with ``edge_ids`` (indices
    into the returned ``edge_ids`` tuple), ``weights``, ``contributions``, and
    the weighted ``residual``.  Duplicate path IDs are collapsed before max.
    """

    def __init__(self, parent_ranker, *, max_length=256, byte_dim=32,
                 source_hidden=64, segment_hidden=64, max_source_bytes=2048):
        super().__init__(
            parent_ranker, max_length=max_length, byte_dim=byte_dim,
            source_hidden=source_hidden, segment_hidden=segment_hidden,
            max_source_bytes=max_source_bytes)
        inherited = {key: value for key, value in self.architecture.items()
                     if key != 'span_reduction'}
        self.architecture = {
            **inherited,
            'schema': SCHEMA,
            'path_reduction': 'distinct_path_weighted_sum_max',
            'edge_reduction': 'source_codepoint_fraction_sum',
            'target_context': 'isolated_exact_output_text',
        }

    def edge_inventory(self, source, groups):
        """Return the canonical edge domain sorted by source interval and text."""
        self._validate_paths(source, groups)
        return tuple(sorted({
            (span.input_start, span.input_end, span.text)
            for group in groups for path in group for span in path.spans
        }))

    @torch.no_grad()
    def extract_edge_features(self, texts):
        """Encode ordered, unique exact edge texts in isolation.

        Callers building a global cache choose the stable text order.  The
        canonical per-row order can be derived from :meth:`edge_inventory`.
        Token limits are checked by ``encode_aligned_candidates`` without
        truncating the output text.
        """
        from hanglish.models.span_token_bridge import (
            encode_aligned_candidates,
            pooling_weights,
        )

        require(isinstance(texts, (list, tuple)) and bool(texts)
                and all(isinstance(text, str) for text in texts)
                and len(set(texts)) == len(texts),
                'ordered unique edge texts required')
        texts = tuple(texts)
        encoded, token_spans = encode_aligned_candidates(
            self.tokenizer, list(texts), self.max_length)
        self.parent.eval()
        vectors = []
        token_flags = []
        for start in range(0, len(texts), self.feature_batch_size):
            stop = min(start + self.feature_batch_size, len(texts))
            result = self.parent(
                **{key: value[start:stop].to(self.device)
                   for key, value in encoded.items()},
                output_hidden_states=True, return_dict=True)
            hidden = result.hidden_states[-1].detach()
            require(hidden.ndim == 3 and hidden.shape[0] == stop - start
                    and hidden.shape[2] == self.target_hidden,
                    'frozen edge feature shape differs')
            for local_index, text in enumerate(texts[start:stop]):
                spans = list(token_spans[start + local_index])
                require(len(spans) == hidden.shape[1],
                        'frozen edge token metadata differs')
                weights = hidden.new_tensor(pooling_weights(spans, 0, len(text)))
                mass = weights.sum()
                has_token = bool(mass.item() > 0)
                if has_token:
                    vector = (hidden[local_index] * weights[:, None]).sum(0) / mass
                else:
                    vector = hidden.new_zeros((self.target_hidden,))
                vectors.append(vector.detach())
                token_flags.append(has_token)
        result = FrozenEdgeFeatures(
            texts,
            torch.stack(vectors),
            torch.tensor(token_flags, dtype=torch.bool, device=self.device),
        )
        self._validate_edge_features(result, set(texts))
        return result

    def _validate_base_features(self, candidates, groups, features):
        dtype = self.source_embedding.weight.dtype
        require(isinstance(features, FrozenCandidates)
                and features.candidates == candidates
                and features.logits.shape == (len(groups),)
                and features.logits.dtype == dtype
                and not features.logits.requires_grad
                and torch.isfinite(features.logits).all().item(),
                'cached frozen base features differ')

    def _validate_edge_features(self, features, needed_texts):
        dtype = self.source_embedding.weight.dtype
        require(isinstance(features, FrozenEdgeFeatures)
                and isinstance(features.texts, tuple) and bool(features.texts)
                and all(isinstance(text, str) for text in features.texts)
                and len(set(features.texts)) == len(features.texts)
                and features.vectors.shape == (len(features.texts), self.target_hidden)
                and features.vectors.dtype == dtype
                and features.has_tokens.shape == (len(features.texts),)
                and features.has_tokens.dtype == torch.bool
                and not features.vectors.requires_grad
                and not features.has_tokens.requires_grad
                and torch.isfinite(features.vectors).all().item()
                and needed_texts.issubset(set(features.texts)),
                'cached frozen edge features differ')
        for index, text in enumerate(features.texts):
            has_tokens = bool(features.has_tokens[index].item())
            require((has_tokens or torch.count_nonzero(features.vectors[index]).item() == 0)
                    and (text != '' or not has_tokens),
                    'cached frozen edge token evidence differs')

    def forward(self, source, groups, *, base_features: FrozenCandidates,
                edge_features: FrozenEdgeFeatures):
        candidates = self._validate_paths(source, groups)
        self._validate_base_features(candidates, groups, base_features)
        edge_ids = self.edge_inventory(source, groups)
        needed_texts = {edge[2] for edge in edge_ids}
        self._validate_edge_features(edge_features, needed_texts)

        offsets = source_byte_offsets(source)
        source_ids = torch.tensor(
            [byte + 1 for byte in source.encode('utf-8')],
            dtype=torch.long, device=self.device).unsqueeze(0)
        states, _ = self.source_encoder(self.source_embedding(source_ids))
        states = states[0]
        text_indices = {text: index for index, text in enumerate(edge_features.texts)}
        target_vectors = edge_features.vectors.to(self.device)
        has_tokens = edge_features.has_tokens.to(self.device)

        edge_vectors = []
        for start, end, text in edge_ids:
            source_vector = states[offsets[start]:offsets[end]].mean(dim=0)
            target_index = text_indices[text]
            target_vector = target_vectors[target_index]
            source_text = source[start:end]
            action_copy = source_text == text
            unresolved = (not action_copy and any(
                'a' <= char <= 'z' or 'A' <= char <= 'Z' or '0' <= char <= '9'
                for char in text))
            extra = source_vector.new_tensor([
                float(action_copy), float(not action_copy), float(unresolved),
                float(not bool(has_tokens[target_index].item())),
                (end - start) / len(source), len(text) / len(source),
                start / len(source), end / len(source),
            ])
            edge_vectors.append(torch.cat((source_vector, target_vector, extra)))
        edge_scores = self.segment_head(torch.stack(edge_vectors)).reshape(-1)
        edge_lookup = {edge: index for index, edge in enumerate(edge_ids)}

        residuals = []
        path_audit = []
        winning_path_ids = []
        for group in groups:
            paths = {path.path_id: path for path in group}
            candidate_paths = []
            path_scores = []
            for path_id in sorted(paths):
                path = paths[path_id]
                ids = tuple(edge_lookup[(span.input_start, span.input_end, span.text)]
                            for span in path.spans)
                weights = edge_scores.new_tensor([
                    (span.input_end - span.input_start) / len(source)
                    for span in path.spans])
                selected = edge_scores[list(ids)]
                contributions = selected * weights
                residual = contributions.sum()
                path_scores.append(residual)
                candidate_paths.append({
                    'path_id': path_id,
                    'edge_ids': ids,
                    'weights': weights,
                    'contributions': contributions,
                    'residual': residual,
                })
            stacked_path_scores = torch.stack(path_scores)
            maximum = stacked_path_scores.max()
            residuals.append(maximum)
            winning_path_ids.append(tuple(
                record['path_id'] for record, score in zip(candidate_paths, path_scores)
                if bool((score.detach() == maximum.detach()).item())))
            path_audit.append(tuple(candidate_paths))
        residuals = torch.stack(residuals)
        base_scores = base_features.logits.to(self.device)
        return {
            'scores': base_scores + residuals,
            'base_scores': base_scores,
            'residuals': residuals,
            'edge_ids': edge_ids,
            'edge_scores': edge_scores,
            'path_audit': tuple(path_audit),
            'winning_path_ids': tuple(winning_path_ids),
        }


__all__ = ['FrozenEdgeFeatures', 'SharedEdgeResidualRanker']
