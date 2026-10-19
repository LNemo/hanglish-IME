"""Experimental source-aligned residual over a frozen sentence ranker.

The frozen candidate encoder supplies both its original score and contextual
token vectors. A small UTF-8 encoder supplies ordered source-span vectors. No
gold context is accepted. Multiple structural paths for one output use a max,
not a sum that rewards the number of paths. This module is not production-wired.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from hanglish.core.source_correspondence import Path, align


SCHEMA = 'source-span-residual-v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class FrozenCandidates:
    candidates: tuple[str, ...]
    logits: torch.Tensor
    hidden: torch.Tensor
    token_spans: tuple[tuple[tuple[int, int] | None, ...], ...]


def source_byte_offsets(source):
    require(isinstance(source, str) and bool(source), 'nonempty source required')
    offsets = [0]
    for char in source:
        offsets.append(offsets[-1] + len(char.encode('utf-8')))
    return offsets


class SourceSpanResidualRanker(nn.Module):
    def __init__(self, parent_ranker, *, max_length=256, byte_dim=32,
                 source_hidden=64, segment_hidden=64, max_source_bytes=2048):
        super().__init__()
        require(type(max_length) is int and 4 <= max_length <= 512,
                'invalid candidate token bound')
        self.parent = parent_ranker.model
        self.tokenizer = parent_ranker.tokenizer
        self.max_length = max_length
        require(type(max_source_bytes) is int and max_source_bytes > 0,
                'invalid source byte bound')
        self.max_source_bytes = max_source_bytes
        self.target_hidden = int(self.parent.config.hidden_size)
        self.feature_batch_size = getattr(parent_ranker, 'batch_size', 8)
        require(type(self.feature_batch_size) is int and 1 <= self.feature_batch_size <= 64,
                'invalid parent feature batch size')
        self.architecture = {'schema': SCHEMA, 'max_length': max_length,
                             'max_source_bytes': max_source_bytes, 'byte_dim': byte_dim,
                             'source_hidden': source_hidden, 'segment_hidden': segment_hidden,
                             'target_hidden': self.target_hidden,
                             'feature_batch_size': self.feature_batch_size,
                             'path_reduction': 'distinct_path_scalar_max',
                             'span_reduction': 'mean'}
        for parameter in self.parent.parameters():
            parameter.requires_grad_(False)
        self.parent.eval()
        self.source_embedding = nn.Embedding(257, byte_dim, padding_idx=0)
        self.source_encoder = nn.GRU(byte_dim, source_hidden, batch_first=True,
                                     bidirectional=True)
        # COPY, CONVERT, unresolved ASCII, target-without-token, lengths, positions.
        self.segment_head = nn.Sequential(
            nn.Linear(source_hidden * 2 + self.target_hidden + 8, segment_hidden),
            nn.GELU(), nn.Linear(segment_hidden, 1))
        nn.init.zeros_(self.segment_head[-1].weight)
        nn.init.zeros_(self.segment_head[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        # Frozen dropout must stay disabled even while the residual is learning.
        self.parent.eval()
        return self

    @property
    def device(self):
        return self.source_embedding.weight.device

    def residual_state_dict(self):
        return {name: value.detach().cpu().contiguous().clone()
                for name, value in self.state_dict().items() if not name.startswith('parent.')}

    def load_residual_state_dict(self, state):
        expected = {name for name in self.state_dict() if not name.startswith('parent.')}
        require(set(state) == expected, 'residual checkpoint tensor names differ')
        own = self.state_dict()
        for name, value in state.items():
            require(value.shape == own[name].shape and value.dtype == own[name].dtype
                    and torch.isfinite(value).all().item(), 'residual checkpoint tensor differs')
        result = self.load_state_dict(state, strict=False)
        require(not result.unexpected_keys
                and set(result.missing_keys) == {n for n in own if n.startswith('parent.')},
                'residual checkpoint loading differs')

    def _validate_paths(self, source, groups):
        require(isinstance(source, str) and bool(source) and isinstance(groups, list)
                and bool(groups), 'source and candidate paths required')
        require(len(source.encode('utf-8')) <= self.max_source_bytes,
                'complete source exceeds byte bound')
        texts = []
        for group in groups:
            require(isinstance(group, list) and bool(group), 'empty candidate path group')
            require(all(isinstance(p, Path) and p.source == source for p in group),
                    'candidate path source differs')
            require(all(p.text == group[0].text for p in group), 'candidate path outputs differ')
            for path in group:
                reconstructed = align(source, path.text, [dict(
                    input_start=s.input_start, input_end=s.input_end,
                    source=s.source, text=s.text) for s in path.spans])
                require(reconstructed == path, 'candidate alignment differs')
            texts.append(group[0].text)
        require(len(set(texts)) == len(texts), 'duplicate candidate outputs')
        return tuple(texts)

    @torch.no_grad()
    def extract_features(self, candidates):
        from hanglish.models.span_token_bridge import encode_aligned_candidates

        require(isinstance(candidates, (list, tuple)) and bool(candidates)
                and len(set(candidates)) == len(candidates), 'distinct candidates required')
        encoded, token_spans = encode_aligned_candidates(
            self.tokenizer, list(candidates), self.max_length)
        self.parent.eval()
        logits, hidden = [], []
        for start in range(0, len(candidates), self.feature_batch_size):
            stop = start + self.feature_batch_size
            result = self.parent(**{k: v[start:stop].to(self.device) for k, v in encoded.items()},
                                 output_hidden_states=True, return_dict=True)
            logits.append(result.logits.reshape(-1).detach())
            hidden.append(result.hidden_states[-1].detach())
        logits, hidden = torch.cat(logits), torch.cat(hidden)
        require(logits.shape == (len(candidates),)
                and hidden.shape[:2] == encoded['input_ids'].shape
                and hidden.shape[2] == self.target_hidden,
                'frozen model feature shape differs')
        return FrozenCandidates(tuple(candidates), logits, hidden,
                                tuple(tuple(v) for v in token_spans))

    def forward(self, source, groups, *, features=None):
        from hanglish.models.span_token_bridge import pooling_weights

        candidates = self._validate_paths(source, groups)
        features = features if features is not None else self.extract_features(candidates)
        require(isinstance(features, FrozenCandidates) and features.candidates == candidates
                and features.logits.shape == (len(groups),)
                and features.hidden.ndim == 3 and features.hidden.shape[0] == len(groups)
                and features.hidden.shape[2] == self.target_hidden
                and len(features.token_spans) == len(groups)
                and all(len(s) == features.hidden.shape[1] for s in features.token_spans)
                and not features.logits.requires_grad and not features.hidden.requires_grad
                and torch.isfinite(features.logits).all().item()
                and torch.isfinite(features.hidden).all().item(), 'cached frozen features differ')
        offsets = source_byte_offsets(source)
        source_ids = torch.tensor([b + 1 for b in source.encode('utf-8')],
                                  dtype=torch.long, device=self.device).unsqueeze(0)
        states, _ = self.source_encoder(self.source_embedding(source_ids))
        states = states[0]
        hidden = features.hidden.to(self.device)
        deltas = []
        for index, group in enumerate(groups):
            paths = {path.path_id: path for path in group}
            path_scores = []
            for path in (paths[k] for k in sorted(paths)):
                vectors = []
                for span in path.spans:
                    source_vector = states[offsets[span.input_start]:offsets[span.input_end]].mean(dim=0)
                    weights = hidden.new_tensor(pooling_weights(
                        list(features.token_spans[index]), span.output_start, span.output_end))
                    mass = weights.sum()
                    target_vector = (hidden[index] * weights[:, None]).sum(dim=0) / mass.clamp_min(1)
                    extra = hidden.new_tensor([
                        float(span.action == 'copy'), float(span.action == 'convert'),
                        float(span.unresolved_ascii_output), float(mass.item() == 0),
                        (span.input_end - span.input_start) / len(source),
                        (span.output_end - span.output_start) / max(1, len(path.text)),
                        span.input_start / len(source), span.output_start / max(1, len(path.text))])
                    vectors.append(torch.cat((source_vector, target_vector, extra)))
                path_scores.append(self.segment_head(torch.stack(vectors)).mean())
            deltas.append(torch.stack(path_scores).max())
        residuals = torch.stack(deltas)
        return {'scores': features.logits.to(self.device) + residuals,
                'base_scores': features.logits.to(self.device), 'residuals': residuals}
