"""Explicit experimental two-transducer candidate union.

This module is intentionally not wired into the default converter or web app.
Both branches run the ordinary converter, after which their displayed finite
candidates are rescored once more by the shared production sentence ranker.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import threading
import time

from hanglish.core.neural_converter import NeuralConverter, ROOT, file_sha256
from hanglish.models.phonetic_transducer import PhoneticTransducer


def _path_key(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _valid_sha(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _sentence_artifacts(artifacts):
    if not isinstance(artifacts, dict):
        return {}
    return {name: digest for name, digest in artifacts.items()
            if name.startswith("checkpoints/sentence_ranker/")}


def _is_original_raw(candidate, text):
    segments = candidate.get("segments")
    return (
        candidate.get("text") == text
        and isinstance(segments, list) and bool(segments)
        and all(segment.get("kind") == "raw"
                and segment.get("text") == segment.get("source") for segment in segments)
    )


def _validate_candidate(candidate, source_text):
    if not isinstance(candidate, dict) or not isinstance(candidate.get("text"), str):
        raise ValueError("ensemble branch returned a malformed candidate")
    segments = candidate.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("ensemble branch returned malformed candidate segments")
    source_parts, output_parts = [], []
    cursor = 0
    for segment in segments:
        start, end = segment.get("input_start"), segment.get("input_end")
        source, output, kind = (segment.get("source"), segment.get("text"),
                                segment.get("kind"))
        if (type(start) is not int or type(end) is not int or start != cursor
                or not 0 <= start <= end <= len(source_text)
                or (start == end and source_text)
                or source_text[start:end] != source
                or not isinstance(output, str) or not isinstance(kind, str)
                or (kind in {"raw", "english"} and output != source)):
            raise ValueError("ensemble candidate segment provenance differs")
        source_parts.append(source)
        output_parts.append(output)
        cursor = end
    if (cursor != len(source_text) or "".join(source_parts) != source_text
            or "".join(output_parts) != candidate["text"]):
        raise ValueError("ensemble candidate segments do not reconstruct the candidate")


def _raw_candidate(candidates, text):
    for candidate in candidates:
        _validate_candidate(candidate, text)
        if _is_original_raw(candidate, text):
            value = deepcopy(candidate)
            value["score"] = None
            return value
    return {"text": text, "score": None, "segments": [{
        "input_start": 0, "input_end": len(text), "source": text,
        "text": text, "kind": "raw",
    }]}


def _finite_displayed(candidates, text):
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("ensemble branch returned no candidates")
    values = []
    for candidate in candidates:
        _validate_candidate(candidate, text)
        score = candidate.get("score")
        if score is None:
            if _is_original_raw(candidate, text):
                continue
            raise ValueError("None score is reserved for the original raw candidate")
        if type(score) not in (int, float) or not math.isfinite(score):
            raise ValueError("ensemble branch returned a malformed candidate score")
        values.append(candidate)
    return values


def _union(old_candidates, supported_candidates, text):
    old = _finite_displayed(old_candidates, text)
    supported = _finite_displayed(supported_candidates, text)
    union = []
    seen = set()
    for candidate in [*old, *supported]:
        if candidate["text"] not in seen:
            union.append(deepcopy(candidate))
            seen.add(candidate["text"])
    if [candidate["text"] for candidate in union[:len(old)]] != [
            candidate["text"] for candidate in old]:
        raise ValueError("ensemble union changed the baseline candidate prefix")
    if len(union) > 10:
        raise ValueError("ensemble union exceeded two displayed five-candidate pools")
    return union


class EnsembleConverter:
    """Experimental old/new transducer union using one shared ranker stack."""

    def __init__(self, checkpoint: Path, *, checkpoint_sha256: str,
                 metadata_sha256: str, experimental: bool = False,
                 context_ranking: bool = True, external_lexicon: bool = True,
                 ranker_kind: str = "auto", baseline=None, supported=None,
                 transducer_loader=None):
        if experimental is not True:
            raise ValueError("EnsembleConverter requires experimental=True")
        if not _valid_sha(checkpoint_sha256) or not _valid_sha(metadata_sha256):
            raise ValueError("verified checkpoint and metadata SHA-256 values are required")
        self.checkpoint = Path(checkpoint)
        self.metadata_path = self.checkpoint.with_suffix(".json")
        self.checkpoint_sha256 = checkpoint_sha256
        self.metadata_sha256 = metadata_sha256
        self.baseline = baseline or NeuralConverter(
            context_ranking=context_ranking, external_lexicon=external_lexicon,
            ranker_kind=ranker_kind)
        self.supported = supported or NeuralConverter(
            context_ranking=context_ranking, external_lexicon=external_lexicon,
            ranker_kind=ranker_kind)
        self._owns_supported = supported is None
        self._transducer_loader = transducer_loader or PhoneticTransducer.load
        self.lock = threading.RLock()
        self.models_loaded = False
        self.version = "experimental-ensemble-unloaded"
        self.model_artifacts = {}
        self.last_baseline_response = None
        self.last_supported_response = None
        self.ensemble_metadata = {
            "experimental": True,
            "shared_generator": False,
            "shared_word_ranker": False,
            "shared_sentence_ranker": False,
            "independent_word_caches": True,
            "independent_branch_locks": True,
            "maximum_union_candidates": 10,
            "performance_limit": "two complete branch conversions plus one fresh union rescore",
            "memory_limit": "two transducers; generator and rankers are shared",
        }

    def _verify_checkpoint(self):
        if (not self.checkpoint.is_file() or not self.metadata_path.is_file()
                or file_sha256(self.checkpoint) != self.checkpoint_sha256
                or file_sha256(self.metadata_path) != self.metadata_sha256):
            raise ValueError("verified supported transducer files differ")

    def load_models(self):
        with self.lock:
            if self.models_loaded:
                return
            self._verify_checkpoint()
            self.baseline.load_models()
            if self._owns_supported:
                # These objects are read-only during inference. Branch-local
                # word caches and locks remain distinct because they belong to
                # each NeuralConverter instance.
                self.supported.generator = self.baseline.generator
                self.supported.numeric_units = self.baseline.numeric_units
                self.supported.ranker = self.baseline.ranker
                self.supported.sentence_ranker = self.baseline.sentence_ranker
                self.supported.ranker_kind = self.baseline.ranker_kind
                self.supported.transducer = self._transducer_loader(
                    self.checkpoint, device="cpu")
                artifacts = dict(self.baseline.model_artifacts)
                artifacts.pop("checkpoints/phonetic_transducer.pt", None)
                artifacts.pop("checkpoints/phonetic_transducer.json", None)
                artifacts[_path_key(self.checkpoint)] = self.checkpoint_sha256
                artifacts[_path_key(self.metadata_path)] = self.metadata_sha256
                self.supported.model_artifacts = dict(sorted(artifacts.items()))
                supported_digest = hashlib.sha256(json.dumps({
                    "checkpoint_sha256": self.checkpoint_sha256,
                    "metadata_sha256": self.metadata_sha256,
                    "ensemble_source_sha256": file_sha256(Path(__file__)),
                }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                self.supported.version = "experimental-supported-1-" + supported_digest[:12]
                self.supported.models_loaded = True
            else:
                self.supported.load_models()
            # Close the checkpoint load against replacement between the
            # constructor hash check and the model's adjacent-metadata read.
            self._verify_checkpoint()
            if self.baseline.lock is self.supported.lock:
                raise ValueError("ensemble branches must have independent locks")
            if self.baseline.word_options is self.supported.word_options:
                raise ValueError("ensemble branches must have independent word caches")
            if self.baseline.sentence_ranker is not self.supported.sentence_ranker:
                raise ValueError("ensemble branches must share one sentence ranker")
            if self.baseline.ranker is not self.supported.ranker:
                raise ValueError("ensemble branches must share one word ranker")
            source_sha = file_sha256(Path(__file__))
            identity = {
                "baseline_version": self.baseline.version,
                "checkpoint_sha256": self.checkpoint_sha256,
                "metadata_sha256": self.metadata_sha256,
                "ensemble_source_sha256": source_sha,
            }
            digest = hashlib.sha256(json.dumps(
                identity, sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest()
            self.version = "experimental-ensemble-1-" + digest[:12]
            self.model_artifacts = dict(self.baseline.model_artifacts)
            self.model_artifacts["experimental/ensemble_converter.py"] = source_sha
            self.model_artifacts["experimental/supported_phonetic_transducer.pt"] = \
                self.checkpoint_sha256
            self.model_artifacts["experimental/supported_phonetic_transducer.json"] = \
                self.metadata_sha256
            self.model_artifacts = dict(sorted(self.model_artifacts.items()))
            self.ensemble_metadata.update({
                "shared_generator": self.baseline.generator is self.supported.generator,
                "shared_word_ranker": True,
                "shared_sentence_ranker": True,
            })
            self.models_loaded = True

    def _baseline_only(self, *args, **kwargs):
        with self.lock:
            response = self.baseline.convert(*args, **kwargs)
            self.last_baseline_response = deepcopy(response)
            self.last_supported_response = None
            return response

    def _decorate(self, response, started, *, union_applied, fallback_reason):
        value = deepcopy(response)
        total_ms = round((time.perf_counter() - started) * 1000, 3)
        metadata = deepcopy(self.ensemble_metadata)
        metadata.update({"union_applied": union_applied,
                         "fallback_reason": fallback_reason})
        value.update({
            "version": self.version,
            "model_artifacts": self.model_artifacts,
            "engine_ms": total_ms,
            "ensemble_total_ms": total_ms,
            "union_applied": union_applied,
            "fallback_reason": fallback_reason,
            "ensemble": metadata,
        })
        return value

    def convert(self, text, context="", engine="neural", mode="batch", raw=False):
        started = time.perf_counter()
        with self.lock:
            if raw or engine != "neural":
                return self._baseline_only(
                    text, context, engine=engine, mode=mode, raw=raw)
            self.load_models()
            baseline = self.baseline.convert(text, context, engine=engine, mode=mode, raw=False)
            supported = self.supported.convert(text, context, engine=engine, mode=mode, raw=False)
            self.last_baseline_response = deepcopy(baseline)
            self.last_supported_response = deepcopy(supported)
            same_sentence_ranker = (
                _sentence_artifacts(baseline.get("model_artifacts"))
                == _sentence_artifacts(supported.get("model_artifacts"))
                and bool(_sentence_artifacts(baseline.get("model_artifacts")))
            )
            if (baseline.get("sentence_ranking_applied") is not True
                    or supported.get("sentence_ranking_applied") is not True):
                return self._decorate(
                    baseline, started, union_applied=False,
                    fallback_reason="branch_sentence_ranking_unavailable")
            if not same_sentence_ranker:
                return self._decorate(
                    baseline, started, union_applied=False,
                    fallback_reason="sentence_ranker_artifacts_differ")
            union = _union(baseline["candidates"], supported["candidates"], text)
            if not union:
                value = deepcopy(baseline)
                value["candidates"] = [_raw_candidate(baseline["candidates"], text)]
                value["result"] = value["candidates"][0]["text"]
                return self._decorate(
                    value, started, union_applied=False,
                    fallback_reason="no_finite_candidates")
            sentence_ranker = self.baseline.sentence_ranker
            if not all(sentence_ranker.can_score(candidate["text"]) for candidate in union):
                return self._decorate(
                    baseline, started, union_applied=False,
                    fallback_reason="candidate_exceeds_sentence_budget")
            scores = list(sentence_ranker.score_pairs(
                [context] * len(union), [candidate["text"] for candidate in union]))
            if (len(scores) != len(union)
                    or any(type(score) not in (int, float) or not math.isfinite(score)
                           for score in scores)):
                raise ValueError("ensemble sentence ranker returned malformed scores")
            rescored = []
            for candidate, score in zip(union, scores):
                value = deepcopy(candidate)
                value["score"] = round(float(score), 4)
                rescored.append((value, float(score)))
            ranked = [candidate for candidate, _ in sorted(
                rescored, key=lambda item: (-item[1], item[0]["text"]))]
            candidates = ranked[:5]
            if not any(candidate["text"] == text for candidate in candidates):
                raw_candidate = _raw_candidate(baseline["candidates"], text)
                candidates = ranked[:4] + [raw_candidate]
            response = deepcopy(baseline)
            response.update({
                "candidates": candidates,
                "result": candidates[0]["text"],
                "warnings": list(dict.fromkeys([
                    *baseline.get("warnings", []), *supported.get("warnings", []),
                ])),
            })
            return self._decorate(
                response, started, union_applied=True, fallback_reason=None)


__all__ = ["EnsembleConverter"]
