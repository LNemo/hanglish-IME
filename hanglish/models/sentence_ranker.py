"""Safe local full-sentence cross-encoder used after lattice generation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


DEFAULT_CHECKPOINT = Path("checkpoints/sentence_ranker")
REQUIRED_ARTIFACTS = {
    "model.safetensors", "config.json", "tokenizer.json", "vocab.txt",
    "tokenizer_config.json", "special_tokens_map.json",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode_sentence_pairs(tokenizer, sentences: list[str], contexts: list[str],
                          max_length: int):
    """Encode each complete candidate once, with external context as sequence B."""
    normalized_sentences = [" ".join(value.strip().split()) for value in sentences]
    normalized_contexts = [" ".join(value.strip().split()) for value in contexts]
    reserved = tokenizer.num_special_tokens_to_add(pair=True)
    for index, (sentence, context) in enumerate(zip(normalized_sentences, normalized_contexts)):
        if context:
            input_ids = tokenizer(sentence, add_special_tokens=False, truncation=False)["input_ids"]
            # Fast tokenizers cannot truncate sequence B down to zero tokens.
            # Keep the complete candidate when it uses the entire pair budget.
            if len(input_ids) + reserved == max_length:
                normalized_contexts[index] = ""
    return tokenizer(normalized_sentences, normalized_contexts, padding=True,
                     truncation="only_second", max_length=max_length,
                     return_tensors="pt")


class SentenceRanker:
    """Deterministic local-only full-sentence scoring wrapper."""

    def __init__(self, model, tokenizer, *, max_length: int = 96,
                 batch_size: int = 8, metadata: dict | None = None) -> None:
        if (isinstance(max_length, bool) or not isinstance(max_length, int)
                or not 4 <= max_length <= 512):
            raise ValueError("sentence-ranker max_length must be an integer from 4 to 512")
        if (isinstance(batch_size, bool) or not isinstance(batch_size, int)
                or not 1 <= batch_size <= 64):
            raise ValueError("sentence-ranker batch_size must be an integer from 1 to 64")
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.metadata = dict(metadata or {})

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CHECKPOINT) -> "SentenceRanker":
        checkpoint = Path(path)
        metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("format_version") != 1:
            raise ValueError("unsupported sentence-ranker checkpoint")
        hashes = metadata.get("artifact_hashes")
        if not isinstance(hashes, dict) or set(hashes) != REQUIRED_ARTIFACTS:
            raise ValueError("sentence-ranker requires a complete artifact hash manifest")
        max_length = metadata.get("max_length")
        batch_size = metadata.get("inference_batch_size")
        if (isinstance(max_length, bool) or not isinstance(max_length, int)
                or not 4 <= max_length <= 512
                or isinstance(batch_size, bool) or not isinstance(batch_size, int)
                or not 1 <= batch_size <= 64):
            raise ValueError("sentence-ranker metadata has invalid inference bounds")
        unexpected = {item.name for item in checkpoint.iterdir() if item.is_file()}
        unexpected -= REQUIRED_ARTIFACTS | {"metadata.json"}
        if unexpected:
            raise ValueError("sentence-ranker bundle contains unverified artifacts")
        for relative, expected in hashes.items():
            artifact = checkpoint / relative
            if not artifact.is_file() or _sha256(artifact) != expected:
                raise ValueError(f"sentence-ranker artifact checksum mismatch: {relative}")
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, local_files_only=True, trust_remote_code=False)
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, local_files_only=True, trust_remote_code=False,
            use_safetensors=True)
        return cls(model, tokenizer, max_length=max_length,
                   batch_size=batch_size, metadata=metadata)

    @torch.inference_mode()
    def score_pairs(self, contexts: list[str], candidates: list[str]) -> list[float]:
        if not isinstance(contexts, list) or not isinstance(candidates, list):
            raise TypeError("contexts and candidates must be lists")
        if len(contexts) != len(candidates):
            raise ValueError("contexts and candidates must have equal length")
        if not contexts:
            return []
        oversized = [candidate for candidate in candidates if not self.can_score(candidate)]
        if oversized:
            raise ValueError("candidate sentence exceeds the full-sentence token budget")
        device = next(self.model.parameters()).device
        scores: list[float] = []
        for start in range(0, len(contexts), self.batch_size):
            end = start + self.batch_size
            encoded = encode_sentence_pairs(
                self.tokenizer, candidates[start:end], contexts[start:end], self.max_length)
            encoded = {name: value.to(device) for name, value in encoded.items()}
            scores.extend(float(value) for value in self.model(**encoded).logits.reshape(-1).cpu())
        if len(scores) != len(candidates):
            raise RuntimeError("sentence-ranker returned an invalid score count")
        return scores

    def can_score(self, candidate: str) -> bool:
        """Return whether sequence A fits intact beside sequence B special tokens."""
        encoded = self.tokenizer(
            " ".join(candidate.strip().split()), add_special_tokens=False,
            truncation=False)
        input_ids = encoded["input_ids"]
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        reserved = self.tokenizer.num_special_tokens_to_add(pair=True)
        return len(input_ids) + reserved <= self.max_length


__all__ = ["SentenceRanker", "encode_sentence_pairs"]
