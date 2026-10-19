"""Local-only frozen ELECTRA replaced-token plausibility features.

The returned values are mean log probabilities that candidate tokens were
*original* under an ELECTRA discriminator. They are RTD auxiliary features,
not autoregressive or masked-language-model likelihoods.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = ROOT / "data/external/koelectra-small-v3"
SOURCE_REVISION = "7488f8db0f208beff4a1f3f9bb3ed04650a89ed7"
PINNED_HASHES = MappingProxyType({
    "SOURCE.json": "036f9b20cf62b5fcde78d9c72606d1ebbe170dad8344038d14960e8143ce2562",
    "tokenizer_config.json": "cd074f01387ed507bc657bebc91c9653874194ba612ceecffe72e6d5c631988c",
    "config.json": "f243c4994815a04b20a084cd2ec24cc78326842d56a4923a8c6751ca0ea40c25",
    "vocab.txt": "9f50b0a44b0272dcf78bfd7a52a0815588f69918bb87ec60b21ca235c5aa0eb4",
    "pytorch_model.bin": "cb3a73908dd25ea9f5ccac1aafc164b941339f11f3499ea6f1a08a8435bb7522",
})
MAX_BATCH = 8
MAX_CANDIDATES = 10
MAX_CONTEXT_TOKENS = 96
MAX_TOTAL_TOKENS = 384


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path_key(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _validate_bundle(path: Path) -> dict[str, str]:
    # Hugging Face's existing download bookkeeping is not a model sidecar.
    entries = {entry.name for entry in path.iterdir()
               if not (entry.name == ".cache" and entry.is_dir())}
    if entries != set(PINNED_HASHES):
        raise ValueError("KoELECTRA bundle contains unpinned or missing artifacts")
    hashes = {}
    for name, expected in PINNED_HASHES.items():
        artifact = path / name
        if not artifact.is_file() or _sha(artifact) != expected:
            raise ValueError(f"KoELECTRA artifact differs: {name}")
        hashes[_path_key(artifact)] = expected
    source = json.loads((path / "SOURCE.json").read_text(encoding="utf-8"))
    if (source.get("revision") != SOURCE_REVISION
            or source.get("files") != {
                name: digest for name, digest in PINNED_HASHES.items()
                if name != "SOURCE.json"
            }):
        raise ValueError("KoELECTRA source provenance differs")
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if (config.get("architectures") != ["ElectraForPreTraining"]
            or config.get("model_type") != "electra"):
        raise ValueError("KoELECTRA checkpoint is not an RTD discriminator")
    return dict(sorted(hashes.items()))


def _loading_info_valid(info: dict) -> bool:
    allowed = lambda name: isinstance(name, str) and name.endswith("position_ids")
    return (
        isinstance(info, dict)
        and all(allowed(name) for name in info.get("missing_keys", []))
        and all(allowed(name) for name in info.get("unexpected_keys", []))
        and not info.get("mismatched_keys")
        and not info.get("error_msgs")
    )


def _discriminator_head_complete(model) -> bool:
    names = {name for name, _ in model.named_parameters()}
    required = {
        "discriminator_predictions.dense.weight",
        "discriminator_predictions.dense.bias",
        "discriminator_predictions.dense_prediction.weight",
        "discriminator_predictions.dense_prediction.bias",
    }
    return all(any(name.endswith(required_name) for name in names)
               for required_name in required)


def _common_difference_spans(candidates: list[str]) -> list[tuple[int, int, bool]]:
    prefix = 0
    shortest = min(map(len, candidates))
    while prefix < shortest and len({value[prefix] for value in candidates}) == 1:
        prefix += 1
    suffix = 0
    while (suffix < shortest - prefix
           and len({value[len(value) - suffix - 1] for value in candidates}) == 1):
        suffix += 1
    spans = []
    all_identical = len(set(candidates)) == 1
    for value in candidates:
        if all_identical:
            spans.append((0, len(value), True))
            continue
        start, end = prefix, len(value) - suffix
        fallback = False
        while start < end and value[start].isspace():
            start += 1
        while end > start and value[end - 1].isspace():
            end -= 1
        if start == end:
            # A candidate can be the deletion side of an insertion/deletion.
            # Select the adjacent word deterministically (right, then left).
            cursor = start
            while cursor < len(value) and value[cursor].isspace():
                cursor += 1
            if cursor < len(value):
                start, end = cursor, cursor
                while start > 0 and not value[start - 1].isspace():
                    start -= 1
                while end < len(value) and not value[end].isspace():
                    end += 1
            elif start > 0:
                end = start
                start -= 1
                while start > 0 and value[start].isspace():
                    start -= 1
                while start > 0 and not value[start - 1].isspace():
                    start -= 1
            else:
                start, end, fallback = 0, len(value), True
        else:
            while start > 0 and not value[start - 1].isspace():
                start -= 1
            if (0 < end < len(value) and not value[end - 1].isspace()
                    and not value[end].isspace()):
                while end < len(value) and not value[end].isspace():
                    end += 1
        if start == end:
            start, end, fallback = 0, len(value), True
        spans.append((start, end, fallback))
    return spans


def _plain_encoding(tokenizer, text):
    encoded = tokenizer(text, add_special_tokens=False, truncation=False,
                        return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    if ids and isinstance(ids[0], list):
        ids, offsets = ids[0], offsets[0]
    return list(ids), [tuple(value) for value in offsets]


class ElectraPlausibility:
    """Frozen KoELECTRA RTD feature scorer for complete candidate strings."""

    def __init__(self, model, tokenizer, source_hashes: dict[str, str]):
        self.model = model
        self.tokenizer = tokenizer
        self._source_hashes = dict(source_hashes)

    @property
    def source_hashes(self):
        return dict(self._source_hashes)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CHECKPOINT, *,
             _tokenizer_loader=None, _model_loader=None):
        checkpoint = Path(path)
        before = _validate_bundle(checkpoint)
        if _tokenizer_loader is None or _model_loader is None:
            from transformers import AutoTokenizer, ElectraForPreTraining
            _tokenizer_loader = _tokenizer_loader or AutoTokenizer.from_pretrained
            _model_loader = _model_loader or ElectraForPreTraining.from_pretrained
        tokenizer = _tokenizer_loader(
            checkpoint, local_files_only=True, trust_remote_code=False, use_fast=True)
        loaded = _model_loader(
            checkpoint, local_files_only=True, trust_remote_code=False,
            weights_only=True, use_safetensors=False, output_loading_info=True)
        if not isinstance(loaded, tuple) or len(loaded) != 2:
            raise ValueError("KoELECTRA loader did not return loading information")
        model, loading_info = loaded
        if not _loading_info_valid(loading_info) or not _discriminator_head_complete(model):
            raise ValueError("KoELECTRA discriminator head was not completely restored")
        model = model.to("cpu").eval()
        after = _validate_bundle(checkpoint)
        if before != after:
            raise ValueError("KoELECTRA bundle changed while loading")
        return cls(model, tokenizer, before)

    def _prepare(self, context: str, candidates: list[str]):
        reserved = self.tokenizer.num_special_tokens_to_add(pair=False)
        spans = _common_difference_spans(candidates)
        candidate_encodings = [_plain_encoding(self.tokenizer, value)
                               for value in candidates]
        if any(not ids or len(ids) + reserved > MAX_TOTAL_TOKENS
               or any(token_id == self.tokenizer.unk_token_id for token_id in ids)
               for ids, _ in candidate_encodings):
            return None
        max_candidate_tokens = max(len(ids) for ids, _ in candidate_encodings)
        context_ids, context_offsets = _plain_encoding(self.tokenizer, context)
        available = MAX_TOTAL_TOKENS - reserved - max_candidate_tokens
        keep = min(MAX_CONTEXT_TOKENS, available, len(context_ids))
        if keep:
            context_start = context_offsets[len(context_offsets) - keep][0]
            trimmed_context = context[context_start:].strip()
        else:
            trimmed_context = ""
        # A suffix beginning inside a WordPiece can tokenize differently from
        # that suffix in the original context. Count the actual text again;
        # remove whole leading words until every candidate shares a safe tail.
        context_budget = min(MAX_CONTEXT_TOKENS, available)
        while trimmed_context:
            trimmed_ids, _ = _plain_encoding(self.tokenizer, trimmed_context)
            if len(trimmed_ids) <= context_budget:
                break
            words = trimmed_context.split(maxsplit=1)
            trimmed_context = words[1] if len(words) > 1 else ""
        prepared = []
        for candidate, changed_span, (candidate_ids, _) in zip(
                candidates, spans, candidate_encodings):
            joined = trimmed_context + (" " if trimmed_context else "") + candidate
            candidate_start = len(joined) - len(candidate)
            prepared.append({
                "text": joined, "candidate": candidate,
                "candidate_start": candidate_start,
                "candidate_ids": candidate_ids,
                "candidate_token_count": len(candidate_ids),
                "changed_span": changed_span,
            })
        return prepared

    @torch.inference_mode()
    def score_candidates(self, context: str, candidates: list[str]):
        if not isinstance(context, str) or not isinstance(candidates, list):
            raise TypeError("context must be a string and candidates must be a list")
        if (not 1 <= len(candidates) <= MAX_CANDIDATES
                or any(not isinstance(value, str) or not value for value in candidates)):
            raise ValueError("one to ten non-empty candidate strings are required")
        prepared = self._prepare(context, candidates)
        if prepared is None:
            return None
        results = []
        for batch_start in range(0, len(prepared), MAX_BATCH):
            batch = prepared[batch_start:batch_start + MAX_BATCH]
            encoded = self.tokenizer(
                [row["text"] for row in batch], add_special_tokens=True,
                padding=True, truncation=False,
                return_offsets_mapping=True, return_special_tokens_mask=True,
                return_tensors="pt")
            offsets = encoded.pop("offset_mapping")
            special = encoded.pop("special_tokens_mask")
            model_inputs = {name: value.to("cpu") for name, value in encoded.items()}
            if any(int(mask.sum().item()) > MAX_TOTAL_TOKENS
                   for mask in model_inputs["attention_mask"]):
                return None
            logits = self.model(**model_inputs).logits
            if logits.ndim == 3 and logits.shape[-1] == 1:
                logits = logits.squeeze(-1)
            if logits.ndim != 2 or logits.shape != model_inputs["input_ids"].shape:
                raise ValueError("KoELECTRA returned malformed discriminator logits")
            if not torch.isfinite(logits).all():
                raise ValueError("KoELECTRA returned non-finite discriminator logits")
            original_logprob = -F.softplus(logits)
            for index, row in enumerate(batch):
                candidate_start = row["candidate_start"]
                candidate_end = candidate_start + len(row["candidate"])
                changed_start, changed_end, fallback = row["changed_span"]
                candidate_positions, changed_positions, context_positions = [], [], []
                for position, pair in enumerate(offsets[index].tolist()):
                    start, end = pair
                    if special[index, position].item() or start == end:
                        continue
                    if start >= candidate_start and end <= candidate_end:
                        local_start, local_end = start - candidate_start, end - candidate_start
                        candidate_positions.append(position)
                        if local_start < changed_end and local_end > changed_start:
                            changed_positions.append(position)
                    elif end <= candidate_start:
                        context_positions.append(position)
                if len(candidate_positions) != row["candidate_token_count"]:
                    return None
                if [model_inputs["input_ids"][index, position].item()
                        for position in candidate_positions] != row["candidate_ids"]:
                    return None
                if len(context_positions) > MAX_CONTEXT_TOKENS:
                    raise ValueError("KoELECTRA context trimming exceeded its token bound")
                if any(model_inputs["input_ids"][index, position].item()
                       == self.tokenizer.unk_token_id for position in candidate_positions):
                    return None
                if not changed_positions:
                    changed_positions = list(candidate_positions)
                    fallback = True
                candidate_values = original_logprob[index, candidate_positions]
                changed_values = original_logprob[index, changed_positions]
                results.append({
                    "mean_original_logprob": float(candidate_values.mean().cpu()),
                    "changed_original_logprob": float(changed_values.mean().cpu()),
                    "candidate_tokens": len(candidate_positions),
                    "changed_tokens": len(changed_positions),
                    "context_tokens": len(context_positions),
                    "changed_span_fallback": bool(fallback),
                })
        return results


__all__ = ["ElectraPlausibility"]
