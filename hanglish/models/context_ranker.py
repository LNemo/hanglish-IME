"""Small deterministic neural ranker for context-sensitive word candidates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import torch
from torch import nn


DEFAULT_CHECKPOINT = Path("checkpoints/context_ranker.pt")
MAX_CONTEXT_CHARS = 500


def _stable_bucket(feature: str, buckets: int) -> int:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % buckets


def _bounded_text(text: str) -> str:
    normalized = " ".join(text.strip().split())
    if len(normalized) <= MAX_CONTEXT_CHARS:
        return normalized
    mask = normalized.rfind("[MASK]")
    if mask < 0:
        return normalized[-MAX_CONTEXT_CHARS:]
    start = max(0, mask - MAX_CONTEXT_CHARS // 2)
    end = min(len(normalized), start + MAX_CONTEXT_CHARS)
    start = max(0, end - MAX_CONTEXT_CHARS)
    return normalized[start:end]


def _features(text: str, namespace: str) -> list[str]:
    """Return bounded n-grams, retaining the masked target at the context tail.

    Callers should place older left context before the sentence containing
    ``[MASK]``.  Keeping the last 500 characters then bounds latency while
    preserving the decision-local text.
    """
    normalized = _bounded_text(text)
    padded = f"^{normalized}$"
    result = [f"{namespace}:bias"]
    for word in normalized.split()[-48:]:
        result.append(f"{namespace}:w:{word}")
    for size in (1, 2, 3):
        result.extend(
            f"{namespace}:c{size}:{padded[index:index + size]}"
            for index in range(max(0, len(padded) - size + 1))
        )
    return result


class _RankerNet(nn.Module):
    def __init__(self, buckets: int = 8192, embedding_dim: int = 64,
                 hidden_dim: int = 96, dropout: float = 0.1) -> None:
        super().__init__()
        self.buckets = buckets
        self.embedding = nn.EmbeddingBag(buckets, embedding_dim, mode="mean")
        combined = embedding_dim * 4
        self.mlp = nn.Sequential(
            nn.Linear(combined, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _encode(self, texts: Iterable[str], namespace: str, device: torch.device) -> torch.Tensor:
        flat: list[int] = []
        offsets: list[int] = []
        for text in texts:
            offsets.append(len(flat))
            flat.extend(_stable_bucket(feature, self.buckets) for feature in _features(text, namespace))
        indices = torch.tensor(flat, dtype=torch.long, device=device)
        starts = torch.tensor(offsets, dtype=torch.long, device=device)
        return self.embedding(indices, starts)

    def forward(self, contexts: list[str], candidates: list[str]) -> torch.Tensor:
        if len(contexts) != len(candidates):
            raise ValueError("contexts and candidates must have equal length")
        if not contexts:
            return torch.empty(0, device=self.embedding.weight.device)
        device = self.embedding.weight.device
        context = self._encode(contexts, "context", device)
        candidate = self._encode(candidates, "candidate", device)
        joined = torch.cat((context, candidate, context * candidate,
                            torch.abs(context - candidate)), dim=1)
        return self.mlp(joined).squeeze(1)


class ContextRanker:
    """Inference wrapper around a trained context/candidate scoring network."""

    def __init__(self, model: _RankerNet, metadata: dict | None = None) -> None:
        self.model = model.eval()
        self.metadata = dict(metadata or {})

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CHECKPOINT) -> "ContextRanker":
        checkpoint_path = Path(path)
        metadata_path = checkpoint_path.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("format_version") != 1:
            raise ValueError("unsupported context-ranker metadata format")
        expected_hash = metadata.get("checkpoint_sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError("context-ranker metadata lacks a valid checkpoint checksum")
        actual_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError("context-ranker checkpoint checksum mismatch")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("format_version") != 1:
            raise ValueError("unsupported context-ranker checkpoint")
        config = payload.get("model_config")
        state = payload.get("state_dict")
        if not isinstance(config, dict) or not isinstance(state, dict):
            raise ValueError("invalid context-ranker checkpoint")
        allowed = {"buckets", "embedding_dim", "hidden_dim", "dropout"}
        if set(config) != allowed:
            raise ValueError("invalid context-ranker model configuration")
        model = _RankerNet(**config)
        model.load_state_dict(state, strict=True)
        return cls(model, metadata)

    @torch.inference_mode()
    def score_pairs(self, contexts: list[str], candidates: list[str]) -> list[float]:
        """Score aligned pairs; a larger value means a better contextual fit."""
        if not isinstance(contexts, list) or not isinstance(candidates, list):
            raise TypeError("contexts and candidates must be lists")
        if len(contexts) != len(candidates):
            raise ValueError("contexts and candidates must have equal length")
        self.model.eval()
        return self.model(contexts, candidates).cpu().tolist()


__all__ = ["ContextRanker"]
