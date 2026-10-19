"""Full-candidate contextual character-edge residual ranker.

The source side and scoring head are intentionally identical to
``CharacterEdgeResidualRanker``.  The target GRU instead reads each complete
candidate output once, and every path occurrence selects the bidirectional
states at its own byte boundaries.  This lets an otherwise identical edge
receive different evidence in different output contexts without adding any
parameters.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from hanglish.models.character_edge_residual import (
    CharacterEdgeResidualRanker,
    PreparedCharacterEdges,
    _require,
)


SCHEMA = 'contextual-character-residual-v1'


@dataclass(frozen=True)
class PreparedContextualCharacterEdges:
    """Parent validation plus complete candidate and occurrence byte bounds."""

    base: PreparedCharacterEdges
    candidate_target_bytes: torch.Tensor
    candidate_target_byte_lengths: torch.Tensor
    occurrence_target_byte_starts: torch.Tensor
    occurrence_target_byte_ends: torch.Tensor

    @property
    def source(self) -> str:
        return self.base.source

    @property
    def candidate_texts(self) -> tuple[str, ...]:
        return self.base.candidate_texts

    @property
    def edge_ids(self) -> tuple[tuple[int, int, str], ...]:
        return self.base.edge_ids

    @property
    def path_edge_indices(self) -> torch.Tensor:
        return self.base.path_edge_indices

    @property
    def path_candidate_indices(self) -> torch.Tensor:
        return self.base.path_candidate_indices

    @property
    def path_weights(self) -> torch.Tensor:
        return self.base.path_weights

    @property
    def base_scores(self) -> torch.Tensor:
        return self.base.base_scores


class ContextualCharacterResidualRanker(CharacterEdgeResidualRanker):
    """Score aligned occurrences using their complete candidate output context."""

    def __init__(self, byte_dim: int = 16, hidden: int = 32,
                 head_hidden: int = 32, max_source_bytes: int = 2048,
                 max_target_bytes: int = 2048):
        super().__init__(
            byte_dim=byte_dim,
            hidden=hidden,
            head_hidden=head_hidden,
            max_source_bytes=max_source_bytes,
            max_target_bytes=max_target_bytes,
        )
        self.architecture = {
            **self.architecture,
            'schema': SCHEMA,
            'target_encoding': (
                'full_candidate_complete_utf8_bytes_bidirectional_gru_'
                'occurrence_endpoints'
            ),
        }

    def prepare(self, source: str, candidates: list[dict],
                base_scores: list[float]) -> PreparedContextualCharacterEdges:
        """Reuse parent validation, then retain complete-output byte boundaries."""
        base = super().prepare(source, candidates, base_scores)

        encoded_candidates = []
        occurrence_starts = []
        occurrence_ends = []
        for candidate in candidates:
            encoded = candidate['text'].encode('utf-8')
            _require(len(encoded) <= self.max_target_bytes,
                     'complete candidate target exceeds byte bound')
            encoded_candidates.append(encoded)
            cursor = 0
            for segment in candidate['segments']:
                occurrence_starts.append(cursor)
                cursor += len(segment['text'].encode('utf-8'))
                occurrence_ends.append(cursor)
            _require(cursor == len(encoded),
                     'candidate occurrence byte coverage differs')

        _require(len(occurrence_starts) == base.path_edge_indices.numel(),
                 'candidate occurrence order differs from parent path')
        max_length = max((len(value) for value in encoded_candidates), default=0)
        candidate_target_bytes = torch.zeros(
            (len(encoded_candidates), max_length), dtype=torch.long)
        for index, value in enumerate(encoded_candidates):
            if value:
                candidate_target_bytes[index, :len(value)] = torch.tensor(
                    [byte + 1 for byte in value], dtype=torch.long)

        result = PreparedContextualCharacterEdges(
            base=base,
            candidate_target_bytes=candidate_target_bytes,
            candidate_target_byte_lengths=torch.tensor(
                [len(value) for value in encoded_candidates], dtype=torch.long),
            occurrence_target_byte_starts=torch.tensor(
                occurrence_starts, dtype=torch.long),
            occurrence_target_byte_ends=torch.tensor(
                occurrence_ends, dtype=torch.long),
        )
        _require(all(not tensor.requires_grad and tensor.device.type == 'cpu'
                     for tensor in self._prepared_tensors(result)),
                 'prepared tensors must be fixed on CPU')
        return result

    @staticmethod
    def _prepared_tensors(
            prepared: PreparedContextualCharacterEdges | PreparedCharacterEdges,
    ) -> tuple[torch.Tensor, ...]:
        if isinstance(prepared, PreparedContextualCharacterEdges):
            return (
                *CharacterEdgeResidualRanker._prepared_tensors(prepared.base),
                prepared.candidate_target_bytes,
                prepared.candidate_target_byte_lengths,
                prepared.occurrence_target_byte_starts,
                prepared.occurrence_target_byte_ends,
            )
        return CharacterEdgeResidualRanker._prepared_tensors(prepared)

    def _encode_candidate_states(self, byte_ids: torch.Tensor,
                                 lengths: torch.Tensor) -> torch.Tensor:
        """Encode complete outputs, leaving all states exactly zero when empty."""
        candidate_count, padded_length = byte_ids.shape
        states = self.target_embedding.weight.new_zeros((
            candidate_count,
            padded_length,
            self.target_encoder.hidden_size * 2,
        ))
        nonempty = torch.nonzero(lengths > 0, as_tuple=False).reshape(-1)
        if nonempty.numel() == 0:
            return states
        selected_lengths = lengths.index_select(0, nonempty)
        embedded = self.target_embedding(byte_ids.index_select(0, nonempty))
        packed = pack_padded_sequence(
            embedded,
            selected_lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_states, _ = self.target_encoder(packed)
        selected_states, _ = pad_packed_sequence(
            packed_states,
            batch_first=True,
            total_length=padded_length,
        )
        return states.index_copy(0, nonempty, selected_states)

    def forward(
            self, prepared: PreparedContextualCharacterEdges,
    ) -> dict[str, torch.Tensor | tuple]:
        _require(isinstance(prepared, PreparedContextualCharacterEdges),
                 'prepared contextual character edges required')
        base = prepared.base
        device = self.device

        source_states, _ = self.source_encoder(
            self.source_embedding(base.source_bytes.to(device).unsqueeze(0)))
        source_states = source_states.squeeze(0)
        edge_count = len(base.edge_ids)
        source_vectors = source_states.new_zeros(
            (edge_count, source_states.shape[1]))
        source_vectors.index_add_(
            0,
            base.edge_source_pool_ids.to(device),
            source_states.index_select(
                0, base.edge_source_byte_indices.to(device)),
        )
        source_vectors = source_vectors / base.edge_source_byte_counts.to(
            device=device,
            dtype=source_vectors.dtype,
        ).unsqueeze(1)

        candidate_states = self._encode_candidate_states(
            prepared.candidate_target_bytes.to(device),
            prepared.candidate_target_byte_lengths.to(device),
        )
        occurrence_count = base.path_edge_indices.numel()
        occurrence_target_vectors = candidate_states.new_zeros((
            occurrence_count,
            self.target_encoder.hidden_size * 2,
        ))
        starts = prepared.occurrence_target_byte_starts.to(device)
        ends = prepared.occurrence_target_byte_ends.to(device)
        nonempty_occurrences = torch.nonzero(
            ends > starts, as_tuple=False).reshape(-1)
        if nonempty_occurrences.numel():
            candidate_indices = base.path_candidate_indices.to(device).index_select(
                0, nonempty_occurrences)
            selected_starts = starts.index_select(0, nonempty_occurrences)
            selected_ends = ends.index_select(0, nonempty_occurrences)
            hidden = self.target_encoder.hidden_size
            contextual = torch.cat((
                candidate_states[
                    candidate_indices, selected_ends - 1, :hidden],
                candidate_states[
                    candidate_indices, selected_starts, hidden:],
            ), dim=1)
            occurrence_target_vectors = occurrence_target_vectors.index_copy(
                0, nonempty_occurrences, contextual)

        path_edge_indices = base.path_edge_indices.to(device)
        occurrence_source_vectors = source_vectors.index_select(
            0, path_edge_indices)
        occurrence_metadata = base.edge_metadata.to(
            device=device,
            dtype=source_vectors.dtype,
        ).index_select(0, path_edge_indices)
        occurrence_inputs = torch.cat((
            occurrence_source_vectors,
            occurrence_target_vectors,
            occurrence_metadata,
        ), dim=1)
        occurrence_scores = self.edge_head(occurrence_inputs).reshape(-1)
        occurrence_contributions = occurrence_scores * base.path_weights.to(
            device=device,
            dtype=occurrence_scores.dtype,
        )
        residuals = occurrence_scores.new_zeros(len(base.candidate_texts))
        residuals.index_add_(
            0,
            base.path_candidate_indices.to(device),
            occurrence_contributions,
        )
        base_scores = base.base_scores.to(device)
        return {
            'scores': base_scores + residuals.to(base_scores.dtype),
            'base_scores': base_scores,
            'residuals': residuals,
            'occurrence_scores': occurrence_scores,
            'occurrence_contributions': occurrence_contributions,
            'occurrence_source_vectors': occurrence_source_vectors,
            'occurrence_target_vectors': occurrence_target_vectors,
            'occurrence_metadata': occurrence_metadata,
            'edge_source_vectors': source_vectors,
            'candidate_target_states': candidate_states,
            'path_edge_indices': base.path_edge_indices,
            'path_candidate_indices': base.path_candidate_indices,
            'edge_ids': base.edge_ids,
            'candidate_texts': base.candidate_texts,
        }


__all__ = [
    'ContextualCharacterResidualRanker',
    'PreparedContextualCharacterEdges',
]
