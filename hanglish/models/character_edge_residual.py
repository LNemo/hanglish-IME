"""Character-preserving shared-edge residual ranker.

The experimental ranker learns only from complete UTF-8 byte sequences.  Each
canonical source-span/output-text edge is encoded once, then candidate scores
are formed by a source-length-weighted sum of those shared edge scores.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from hanglish.core.source_correspondence import align


SCHEMA = 'character-edge-residual-v1'


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class PreparedCharacterEdges:
    """Validated immutable metadata and CPU tensors for one candidate group."""

    source: str
    candidate_texts: tuple[str, ...]
    edge_ids: tuple[tuple[int, int, str], ...]
    source_bytes: torch.Tensor
    edge_source_byte_indices: torch.Tensor
    edge_source_pool_ids: torch.Tensor
    edge_source_byte_counts: torch.Tensor
    target_bytes: torch.Tensor
    target_byte_lengths: torch.Tensor
    edge_metadata: torch.Tensor
    path_edge_indices: torch.Tensor
    path_candidate_indices: torch.Tensor
    path_weights: torch.Tensor
    base_scores: torch.Tensor


class CharacterEdgeResidualRanker(nn.Module):
    """Rank aligned outputs with trainable source and exact-target byte encoders."""

    def __init__(self, byte_dim: int = 16, hidden: int = 32,
                 head_hidden: int = 32, max_source_bytes: int = 2048,
                 max_target_bytes: int = 2048):
        super().__init__()
        for name, value, upper in (
            ('byte_dim', byte_dim, 1024), ('hidden', hidden, 2048),
            ('head_hidden', head_hidden, 2048),
            ('max_source_bytes', max_source_bytes, 1_000_000),
            ('max_target_bytes', max_target_bytes, 1_000_000),
        ):
            _require(type(value) is int and 1 <= value <= upper,
                     f'invalid {name}')
        self.max_source_bytes = max_source_bytes
        self.max_target_bytes = max_target_bytes
        self.source_embedding = nn.Embedding(257, byte_dim, padding_idx=0)
        self.target_embedding = nn.Embedding(257, byte_dim, padding_idx=0)
        self.source_encoder = nn.GRU(
            byte_dim, hidden, batch_first=True, bidirectional=True)
        self.target_encoder = nn.GRU(
            byte_dim, hidden, batch_first=True, bidirectional=True)
        # copy flag, source codepoint fraction, target/source codepoint fraction
        self.edge_head = nn.Sequential(
            nn.Linear(hidden * 4 + 3, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 1),
        )
        nn.init.zeros_(self.edge_head[-1].weight)
        nn.init.zeros_(self.edge_head[-1].bias)
        self.architecture = {
            'schema': SCHEMA,
            'byte_vocabulary': 257,
            'byte_dim': byte_dim,
            'hidden_per_direction': hidden,
            'head_hidden': head_hidden,
            'max_source_bytes': max_source_bytes,
            'max_target_bytes': max_target_bytes,
            'source_encoding': 'complete_utf8_bytes_bidirectional_gru',
            'target_encoding': 'isolated_complete_utf8_bytes_bidirectional_gru',
            'edge_identity': '(input_start,input_end,target_text)',
            'path_reduction': 'source_codepoint_fraction_sum',
            'metadata': ('copy', 'source_codepoint_fraction',
                         'target_to_source_codepoint_fraction'),
            'parameter_count': sum(parameter.numel() for parameter in self.parameters()),
            'trainable_parameter_count': sum(
                parameter.numel() for parameter in self.parameters()
                if parameter.requires_grad),
        }

    @property
    def device(self) -> torch.device:
        return self.source_embedding.weight.device

    def prepare(self, source: str, candidates: list[dict],
                base_scores: list[float]) -> PreparedCharacterEdges:
        """Validate alignments and build complete byte/edge/path tensors on CPU."""
        _require(isinstance(source, str) and bool(source),
                 'nonempty source required')
        source_raw = source.encode('utf-8')
        _require(len(source_raw) <= self.max_source_bytes,
                 'complete source exceeds byte bound')
        _require(isinstance(candidates, list) and bool(candidates),
                 'candidate list required')
        _require(isinstance(base_scores, list)
                 and len(base_scores) == len(candidates),
                 'one base score per candidate required')
        _require(all(type(score) in (int, float) and math.isfinite(score)
                     for score in base_scores), 'finite base scores required')

        paths = []
        alignment_fields = {'input_start', 'input_end', 'source', 'text'}
        for candidate in candidates:
            _require(isinstance(candidate, dict)
                     and set(candidate) == {'text', 'segments'}
                     and isinstance(candidate['text'], str)
                     and isinstance(candidate['segments'], list),
                     'candidate must contain only text and segments')
            for segment in candidate['segments']:
                _require(isinstance(segment, dict)
                         and set(segment) in (alignment_fields,
                                              alignment_fields | {'kind'})
                         and ('kind' not in segment
                              or isinstance(segment['kind'], str)),
                         'segment fields differ')
            paths.append(align(source, candidate['text'], [
                {key: segment[key] for key in alignment_fields}
                for segment in candidate['segments']
            ]))
        candidate_texts = tuple(path.text for path in paths)
        _require(len(set(candidate_texts)) == len(candidate_texts),
                 'candidate output texts must be distinct')

        edge_ids = tuple(sorted({
            (span.input_start, span.input_end, span.text)
            for path in paths for span in path.spans
        }))
        edge_lookup = {edge: index for index, edge in enumerate(edge_ids)}
        offsets = [0]
        for character in source:
            offsets.append(offsets[-1] + len(character.encode('utf-8')))

        byte_indices, pool_ids, byte_counts = [], [], []
        target_raw, metadata = [], []
        source_length = len(source)
        for edge_index, (start, end, text) in enumerate(edge_ids):
            begin_byte, end_byte = offsets[start], offsets[end]
            indices = list(range(begin_byte, end_byte))
            byte_indices.extend(indices)
            pool_ids.extend([edge_index] * len(indices))
            byte_counts.append(len(indices))
            encoded_target = text.encode('utf-8')
            _require(len(encoded_target) <= self.max_target_bytes,
                     'complete target edge exceeds byte bound')
            target_raw.append(encoded_target)
            source_fraction = (end - start) / source_length
            metadata.append((
                float(source[start:end] == text),
                source_fraction,
                len(text) / source_length,
            ))

        max_target_length = max((len(value) for value in target_raw), default=0)
        target_bytes = torch.zeros(
            (len(edge_ids), max_target_length), dtype=torch.long)
        for index, value in enumerate(target_raw):
            if value:
                target_bytes[index, :len(value)] = torch.tensor(
                    [byte + 1 for byte in value], dtype=torch.long)

        path_edge_indices, path_candidate_indices, path_weights = [], [], []
        for candidate_index, path in enumerate(paths):
            for span in path.spans:
                path_edge_indices.append(edge_lookup[
                    (span.input_start, span.input_end, span.text)])
                path_candidate_indices.append(candidate_index)
                path_weights.append(
                    (span.input_end - span.input_start) / source_length)

        result = PreparedCharacterEdges(
            source=source,
            candidate_texts=candidate_texts,
            edge_ids=edge_ids,
            source_bytes=torch.tensor(
                [byte + 1 for byte in source_raw], dtype=torch.long),
            edge_source_byte_indices=torch.tensor(byte_indices, dtype=torch.long),
            edge_source_pool_ids=torch.tensor(pool_ids, dtype=torch.long),
            edge_source_byte_counts=torch.tensor(byte_counts, dtype=torch.long),
            target_bytes=target_bytes,
            target_byte_lengths=torch.tensor(
                [len(value) for value in target_raw], dtype=torch.long),
            edge_metadata=torch.tensor(metadata, dtype=torch.float32),
            path_edge_indices=torch.tensor(path_edge_indices, dtype=torch.long),
            path_candidate_indices=torch.tensor(
                path_candidate_indices, dtype=torch.long),
            path_weights=torch.tensor(path_weights, dtype=torch.float32),
            base_scores=torch.tensor(base_scores, dtype=torch.float64),
        )
        _require(all(not tensor.requires_grad and tensor.device.type == 'cpu'
                     for tensor in self._prepared_tensors(result)),
                 'prepared tensors must be fixed on CPU')
        return result

    @staticmethod
    def _prepared_tensors(prepared: PreparedCharacterEdges) -> tuple[torch.Tensor, ...]:
        return tuple(value for value in prepared.__dict__.values()
                     if isinstance(value, torch.Tensor))

    def _encode_targets(self, byte_ids: torch.Tensor,
                        lengths: torch.Tensor) -> torch.Tensor:
        """Encode padded targets while assigning an exact zero to empty text."""
        edge_count = lengths.numel()
        vectors = self.target_embedding.weight.new_zeros(
            (edge_count, self.target_encoder.hidden_size * 2))
        nonempty = torch.nonzero(lengths > 0, as_tuple=False).reshape(-1)
        if nonempty.numel() == 0:
            return vectors
        selected_lengths = lengths.index_select(0, nonempty)
        selected = byte_ids.index_select(0, nonempty)
        embedded = self.target_embedding(selected)
        packed = pack_padded_sequence(
            embedded, selected_lengths.detach().cpu(), batch_first=True,
            enforce_sorted=False)
        _, hidden = self.target_encoder(packed)
        encoded = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return vectors.index_copy(0, nonempty, encoded)

    def forward(self, prepared: PreparedCharacterEdges) -> dict[str, torch.Tensor | tuple]:
        _require(isinstance(prepared, PreparedCharacterEdges),
                 'prepared character edges required')
        device = self.device
        source_bytes = prepared.source_bytes.to(device)
        source_states, _ = self.source_encoder(
            self.source_embedding(source_bytes.unsqueeze(0)))
        source_states = source_states.squeeze(0)

        edge_count = len(prepared.edge_ids)
        source_vectors = source_states.new_zeros(
            (edge_count, source_states.shape[1]))
        source_vectors.index_add_(
            0, prepared.edge_source_pool_ids.to(device),
            source_states.index_select(
                0, prepared.edge_source_byte_indices.to(device)))
        source_vectors = source_vectors / prepared.edge_source_byte_counts.to(
            device=device, dtype=source_vectors.dtype).unsqueeze(1)
        target_vectors = self._encode_targets(
            prepared.target_bytes.to(device),
            prepared.target_byte_lengths.to(device))
        edge_inputs = torch.cat((
            source_vectors, target_vectors,
            prepared.edge_metadata.to(device=device, dtype=source_vectors.dtype),
        ), dim=1)
        edge_scores = self.edge_head(edge_inputs).reshape(-1)

        selected_scores = edge_scores.index_select(
            0, prepared.path_edge_indices.to(device))
        edge_contributions = selected_scores * prepared.path_weights.to(
            device=device, dtype=selected_scores.dtype)
        residuals = edge_scores.new_zeros(len(prepared.candidate_texts))
        residuals.index_add_(
            0, prepared.path_candidate_indices.to(device), edge_contributions)
        base_scores = prepared.base_scores.to(device)
        return {
            'scores': base_scores + residuals.to(base_scores.dtype),
            'base_scores': base_scores,
            'residuals': residuals,
            'edge_scores': edge_scores,
            'edge_contributions': edge_contributions,
            'edge_source_vectors': source_vectors,
            'edge_target_vectors': target_vectors,
            'edge_ids': prepared.edge_ids,
            'candidate_texts': prepared.candidate_texts,
        }


__all__ = ['CharacterEdgeResidualRanker', 'PreparedCharacterEdges']
