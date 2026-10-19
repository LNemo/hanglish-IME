"""Pretrained Korean semantic cross-encoder for candidate ranking."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


DEFAULT_CHECKPOINT = Path("checkpoints/semantic_ranker")
LEFT_CONTEXT_CHARS = 64
RIGHT_CONTEXT_CHARS = 24


def render_candidate(context: str, candidate: str) -> str:
    """Place a candidate at the last mask, preserving literal earlier masks."""
    normalized = " ".join(context.strip().split())
    if "[MASK]" in normalized:
        left, _, right = normalized.rpartition("[MASK]")
        return left + candidate + right
    return f"{normalized} {candidate}".strip()


def context_neighborhood(context: str, candidate: str) -> str:
    """Bound text around the active (last) mask before WordPiece truncation."""
    normalized = " ".join(context.strip().split())
    if "[MASK]" not in normalized:
        return render_candidate(normalized[-LEFT_CONTEXT_CHARS:], candidate)
    left, _, right = normalized.rpartition("[MASK]")
    return left[-LEFT_CONTEXT_CHARS:] + candidate + right[:RIGHT_CONTEXT_CHARS]


def encode_pairs(tokenizer, contexts: list[str], candidates: list[str], max_length: int):
    """Keep each candidate in sequence A and truncate only bounded context B."""
    neighborhoods = [context_neighborhood(context, candidate)
                     for context, candidate in zip(contexts, candidates)]
    return tokenizer(candidates, neighborhoods, padding=True, truncation="only_second",
                     max_length=max_length, return_tensors="pt")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SemanticRanker:
    """Local-only inference wrapper with the same API as ``ContextRanker``."""

    def __init__(self, model, tokenizer, *, max_length: int = 96,
                 batch_size: int = 8, metadata: dict | None = None) -> None:
        if (isinstance(max_length, bool) or not isinstance(max_length, int)
                or not 4 <= max_length <= 512):
            raise ValueError("semantic-ranker max_length must be an integer from 4 to 512")
        if (isinstance(batch_size, bool) or not isinstance(batch_size, int)
                or not 1 <= batch_size <= 64):
            raise ValueError("semantic-ranker batch_size must be an integer from 1 to 64")
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.metadata = dict(metadata or {})

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CHECKPOINT) -> "SemanticRanker":
        checkpoint = Path(path)
        metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("format_version") != 1:
            raise ValueError("unsupported semantic-ranker checkpoint")
        hashes = metadata.get("artifact_hashes")
        required = {"model.safetensors", "config.json", "tokenizer.json", "vocab.txt",
                    "tokenizer_config.json", "special_tokens_map.json"}
        if not isinstance(hashes, dict) or set(hashes) != required:
            raise ValueError("semantic-ranker requires a complete artifact hash manifest")
        max_length = metadata.get("max_length")
        batch_size = metadata.get("inference_batch_size")
        if (isinstance(max_length, bool) or not isinstance(max_length, int)
                or not 4 <= max_length <= 512
                or isinstance(batch_size, bool) or not isinstance(batch_size, int)
                or not 1 <= batch_size <= 64):
            raise ValueError("semantic-ranker metadata has invalid inference bounds")
        unexpected = {item.name for item in checkpoint.iterdir() if item.is_file()} - required - {"metadata.json"}
        if unexpected:
            raise ValueError("semantic-ranker bundle contains unverified artifacts")
        for relative, expected in hashes.items():
            artifact = checkpoint / relative
            if not artifact.is_file() or _sha256(artifact) != expected:
                raise ValueError(f"semantic-ranker artifact checksum mismatch: {relative}")
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, local_files_only=True, trust_remote_code=False)
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, local_files_only=True, trust_remote_code=False,
            use_safetensors=True)
        return cls(model, tokenizer,
                   max_length=max_length,
                   batch_size=batch_size,
                   metadata=metadata)

    @torch.inference_mode()
    def score_pairs(self, contexts: list[str], candidates: list[str]) -> list[float]:
        if not isinstance(contexts, list) or not isinstance(candidates, list):
            raise TypeError("contexts and candidates must be lists")
        if len(contexts) != len(candidates):
            raise ValueError("contexts and candidates must have equal length")
        if not contexts:
            return []
        device = next(self.model.parameters()).device
        result: list[float] = []
        for start in range(0, len(contexts), self.batch_size):
            end = start + self.batch_size
            current_candidates = candidates[start:end]
            encoded = encode_pairs(
                self.tokenizer, contexts[start:end], current_candidates, self.max_length)
            encoded = {name: value.to(device) for name, value in encoded.items()}
            logits = self.model(**encoded).logits.reshape(-1)
            result.extend(float(value) for value in logits.cpu())
        if len(result) != len(candidates):
            raise RuntimeError("semantic-ranker returned an invalid score count")
        return result


__all__ = ["SemanticRanker", "context_neighborhood", "encode_pairs", "render_candidate"]
