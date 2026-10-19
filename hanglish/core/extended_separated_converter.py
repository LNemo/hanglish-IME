"""Experimental 256-token ranking of the separated expanded candidate pool."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time

from hanglish.core.neural_converter import file_sha256
from hanglish.core.separated_converter import SeparatedConverter


SCHEMA_VERSION = "extended-separated-ranking-v1"
POLICY = "expanded_pool_256"
TUNED_MAX_LENGTH = 96
RUNTIME_MAX_LENGTH = 256


def _stable_id(prefix, value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"))
    return prefix + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _finite_score(value):
    return (type(value) in (int, float) and math.isfinite(value))


def _contains_hangul(value):
    return any(("\u1100" <= character <= "\u11ff")
               or ("\u3130" <= character <= "\u318f")
               or ("\uac00" <= character <= "\ud7a3")
               for character in value)


class ExtendedSeparatedConverter(SeparatedConverter):
    """Apply the DEV-selected expanded-pool policy without retraining."""

    def __init__(self, **kwargs):
        selected = kwargs.pop("generation_mode", "independent")
        if selected != "independent":
            raise ValueError("ExtendedSeparatedConverter requires independent mode")
        super().__init__(generation_mode="independent", **kwargs)
        self._extended_sentence_ranker = None
        self._extended_source_ranker = None
        self.extended_artifacts = {
            "hanglish/core/extended_separated_converter.py": file_sha256(
                Path(__file__)),
            "hanglish/core/separated_converter.py": file_sha256(
                Path(__file__).with_name("separated_converter.py")),
        }
        self._extended_parent_version = self.version
        self._stamp_extended_version()

    def _stamp_extended_version(self):
        payload = {
            "parent": self._extended_parent_version,
            "artifacts": self.extended_artifacts,
            "policy": POLICY,
            "runtime_max_length": RUNTIME_MAX_LENGTH,
            "tuned_max_length": TUNED_MAX_LENGTH,
        }
        self.version = "experimental-expanded-pool-256-1-" + hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]

    def load_models(self):
        with self.lock:
            super().load_models()
            # The parent chain restamps its own stable CTC-union version on
            # every load call.  Capture that value before applying this layer.
            self._extended_parent_version = self.version
            self.model_artifacts = {
                **getattr(self, "model_artifacts", {}),
                **self.extended_artifacts,
            }
            self._stamp_extended_version()

    def _load_extended_ranker(self):
        original = getattr(self, "sentence_ranker", None)
        if original is None:
            return None
        if self._extended_sentence_ranker is not None:
            if self._extended_source_ranker is not original:
                raise RuntimeError("extended ranker source instance changed")
            self._validate_extended_rankers(original,
                                            self._extended_sentence_ranker)
            return self._extended_sentence_ranker
        self._validate_original_ranker(original)
        from hanglish.models.sentence_ranker import SentenceRanker

        extended = SentenceRanker(
            original.model, original.tokenizer,
            max_length=RUNTIME_MAX_LENGTH,
            batch_size=original.batch_size,
            metadata=original.metadata)
        self._validate_extended_rankers(original, extended)
        self._extended_source_ranker = original
        self._extended_sentence_ranker = extended
        return extended

    @staticmethod
    def _validate_original_ranker(original):
        capacity = getattr(getattr(original, "model", None), "config", None)
        positions = getattr(capacity, "max_position_embeddings", None)
        metadata = getattr(original, "metadata", {})
        if (getattr(original, "max_length", None) != TUNED_MAX_LENGTH
                or not isinstance(metadata, dict)
                or metadata.get("max_length") != TUNED_MAX_LENGTH):
            raise RuntimeError("sentence ranker tuned-length contract differs")
        if (isinstance(positions, bool) or not isinstance(positions, int)
                or positions < RUNTIME_MAX_LENGTH):
            raise RuntimeError("sentence ranker cannot represent 256 positions")

    @classmethod
    def _validate_extended_rankers(cls, original, extended):
        cls._validate_original_ranker(original)
        positions = getattr(getattr(extended.model, "config", None),
                            "max_position_embeddings", None)
        if (extended.model is not original.model
                or extended.tokenizer is not original.tokenizer
                or extended.max_length != RUNTIME_MAX_LENGTH
                or isinstance(positions, bool) or not isinstance(positions, int)
                or positions < RUNTIME_MAX_LENGTH):
            raise RuntimeError(
                "extended ranker must share the 256-capable model and tokenizer")

    def _model_id(self):
        artifacts = dict(sorted(getattr(self, "model_artifacts", {}).items()))
        return _stable_id("sentence-ranker-256-v1-", {
            "artifacts": artifacts,
            "parent_version": self._extended_parent_version,
            "max_length": RUNTIME_MAX_LENGTH,
        })

    def _runtime_identity(self):
        original = getattr(self, "sentence_ranker", None)
        extended = getattr(self, "_extended_sentence_ranker", None)
        positions = getattr(getattr(getattr(original, "model", None),
                                    "config", None),
                            "max_position_embeddings", None)
        shared = bool(original is not None and extended is not None
                      and self._extended_source_ranker is original
                      and getattr(extended, "model", None)
                      is getattr(original, "model", None)
                      and getattr(extended, "tokenizer", None)
                      is getattr(original, "tokenizer", None))
        return {
            "experimental": True,
            "schema_version": SCHEMA_VERSION,
            "policy": POLICY,
            "runtime_version": self.version,
            "parent_version": self._extended_parent_version,
            "artifacts": dict(self.extended_artifacts),
            "model_artifacts": dict(sorted(
                getattr(self, "model_artifacts", {}).items())),
            "model_id": self._model_id(),
            "shared_original_model_and_tokenizer": shared,
            "tuned_max_length": TUNED_MAX_LENGTH,
            "runtime_max_length": RUNTIME_MAX_LENGTH,
            "max_position_embeddings": positions,
            "interpretation": (
                "extrapolation beyond tuned length; no new training"),
            "capture_attributes": {
                "last_separated_capture": "pre_extended_independent_pipeline",
                "last_pipeline": "final_extended_pipeline",
            },
        }

    @staticmethod
    def _canonical_raw(source):
        return {"text": source, "score": None, "segments": [{
            "input_start": 0, "input_end": len(source), "source": source,
            "text": source, "kind": "raw",
        }]}

    def _eligible_reason(self, response, text, context, engine, raw):
        if raw:
            return "raw_request"
        if engine != "neural":
            return "non_neural_engine"
        if not self.context_ranking:
            return "context_ranking_disabled"
        if context.strip():
            return "nonempty_context_not_evaluated"
        if len(text.split()) < 2:
            return "single_word_not_evaluated"
        if _contains_hangul(text):
            return "non_roman_source_not_evaluated"
        if getattr(self, "sentence_ranker", None) is None:
            return "sentence_ranker_unavailable"
        pipeline = response.get("pipeline")
        if not isinstance(pipeline, dict):
            raise RuntimeError("independent response lacks pipeline capture")
        parent = pipeline.get("parent")
        if not isinstance(parent, dict):
            raise RuntimeError("independent pipeline lacks parent metadata")
        parent_ranked = parent.get("sentence_ranking_applied")
        response_ranked = response.get("sentence_ranking_applied")
        if not isinstance(parent_ranked, bool) or not isinstance(response_ranked, bool):
            raise RuntimeError("sentence ranking flags are malformed")
        if parent_ranked != response_ranked:
            raise RuntimeError("parent and response ranking flags differ")
        if parent_ranked:
            return "parent_sentence_ranking_already_applied"
        return None

    def _unique_nonraw_pool(self, response, source):
        pool = response["pipeline"].get("pool")
        if not isinstance(pool, list):
            raise RuntimeError("independent candidate pool is malformed")
        unique = []
        seen = set()
        for row in pool:
            if (not isinstance(row, dict)
                    or not isinstance(row.get("candidate_id"), str)
                    or row.get("candidate_id") != row.get("path_id")
                    or not isinstance(row.get("text"), str)
                    or not isinstance(row.get("segments"), list)):
                raise RuntimeError("independent candidate row is malformed")
            text = row["text"]
            # Validate full contiguous coverage before ranking any policy row.
            self._source_spans(source, text, row["segments"])
            expected_id = self._path_identity(
                source, (text, 0.0, row["segments"]))
            if row["path_id"] != expected_id:
                raise RuntimeError("independent candidate path identity differs")
            if text == source or text in seen:
                continue
            seen.add(text)
            unique.append(row)
        return unique

    def _metadata(self, *, target, applied, reason, candidate_rows=(),
                  score_rows=(),
                  unsupported=(), elapsed_ms=0.0):
        return {
            "schema_version": SCHEMA_VERSION,
            "identity": self._runtime_identity(),
            "policy": POLICY,
            "target": target,
            "applied": applied,
            "reason": reason,
            "source_only_scoring": True,
            "score_context": "",
            "input_truncation": False,
            "unique_nonraw_candidates": len(candidate_rows or score_rows),
            "candidate_refs": [
                {"candidate_id": row["candidate_id"], "text": row["text"]}
                for row in (candidate_rows or score_rows)],
            "unsupported_texts": list(unsupported),
            "scores": [{"candidate_id": row["candidate_id"],
                        "text": row["text"], "score": row["score"]}
                       for row in score_rows],
            "ranking_ms": round(elapsed_ms, 3),
        }

    def _preserve(self, response, *, reason, started, target=False,
                  candidate_rows=(), unsupported=()):
        response["extended_sentence_ranking"] = self._metadata(
            target=target, applied=False, reason=reason,
            candidate_rows=candidate_rows, unsupported=unsupported)
        response["engine_ms"] = round((time.perf_counter() - started) * 1000, 3)
        self.last_pipeline = deepcopy(response["pipeline"])
        return response

    def _apply_ranked_pool(self, response, source, unique, scores,
                           ranking_ms):
        model_id = self._model_id()
        rescored = []
        selected_ids = {row["candidate_id"] for row in unique}
        for row, score in zip(unique, scores):
            value = deepcopy(row)
            value.update({
                "score": score,
                "score_type": "sentence_ranker_256",
                "model_id": model_id,
                "evaluated": True,
                "reason": "scored_at_256",
                "selection_eligible": True,
            })
            value["extended_ranking"] = {
                "evaluated": True, "reason": "scored_at_256",
                "score": score, "score_type": "sentence_ranker_256",
                "model_id": model_id,
            }
            rescored.append(value)
        scored_by_id = {row["candidate_id"]: row for row in rescored}
        pool = response["pipeline"]["pool"]
        first_ids = set()
        for index, row in enumerate(pool):
            replacement = scored_by_id.get(row["candidate_id"])
            if replacement is not None and row["candidate_id"] not in first_ids:
                pool[index] = deepcopy(replacement)
                first_ids.add(row["candidate_id"])
            elif row["text"] == source:
                row["selection_eligible"] = False
                row["extended_ranking"] = {
                    "evaluated": False, "reason": "reserved_canonical_raw"}
            elif row["candidate_id"] not in selected_ids:
                row["selection_eligible"] = False
                row["extended_ranking"] = {
                    "evaluated": False,
                    "reason": "duplicate_text_after_first_structural_path"}

        ranked = sorted(rescored,
                        key=lambda row: (-row["score"], row["text"]))
        raw = self._canonical_raw(source)
        raw_path = (source, 0.0, raw["segments"])
        raw_id = self._path_identity(source, raw_path)
        raw_row = next((row for row in pool if row["path_id"] == raw_id), None)
        if raw_row is None:
            raw_row = self._candidate_row(
                source, raw_path, origin="raw_fallback", score=None,
                score_type=None, model_id=None, evaluated=False,
                reason="appended_for_user", selection_eligible=True)
            raw_row["pool_index"] = len(pool)
            raw_row["extended_ranking"] = {
                "evaluated": False, "reason": "reserved_canonical_raw"}
            pool.append(raw_row)
        raw_row["selection_eligible"] = True
        raw_row["extended_ranking"] = {
            "evaluated": False, "reason": "reserved_canonical_raw"}
        displayed_rows = ranked[:4]
        candidates = [{"text": row["text"], "score": round(row["score"], 4),
                       "segments": deepcopy(row["segments"])}
                      for row in displayed_rows] + [raw]
        response.update({
            "result": candidates[0]["text"],
            "candidates": candidates,
            "sentence_ranking_applied": True,
            "ranking_applied": True,
        })
        old_warning = (
            "문장 모델의 길이 범위를 넘어 단어별 문맥 순위를 사용했습니다.")
        response["warnings"] = [warning for warning in response.get("warnings", [])
                                if warning != old_warning]
        refs = [{"index": index, "candidate_id": row["candidate_id"],
                 "candidate": deepcopy(candidate)}
                for index, (row, candidate) in enumerate(
                    zip(displayed_rows, candidates[:-1]))]
        refs.append({"index": len(candidates) - 1,
                     "candidate_id": raw_row["candidate_id"],
                     "candidate": deepcopy(raw)})
        pipeline = response["pipeline"]
        input_ids = [row["candidate_id"] for row in unique]
        output_ids = [row["candidate_id"] for row in displayed_rows] + [
            raw_row["candidate_id"]]
        request_id = pipeline.get("request_id")
        stage = {
            "stage_id": _stable_id("stage-v1-", {
                "request_id": request_id, "name": POLICY,
                "input_path_ids": input_ids, "output_path_ids": output_ids,
            }),
            "name": POLICY,
            "input_path_ids": input_ids,
            "generated_path_ids": [],
            "output_path_ids": output_ids,
            "evaluated": True,
            "reason": "scored_full_nonraw_pool_at_256",
            "score_type": "sentence_ranker_256",
            "model_id": model_id,
            "counts": {"input": len(input_ids), "scored": len(rescored),
                       "displayed_nonraw": len(displayed_rows),
                       "displayed_raw": 1},
            "source_only": True,
            "source_truncated": False,
            "score_context": "",
            "score_refs": [{"candidate_id": row["candidate_id"],
                            "score": row["score"]} for row in rescored],
            "ranking_ms": round(ranking_ms, 3),
        }
        pipeline["stages"].append(stage)
        pipeline["selection_policy"] = POLICY
        pipeline["response_candidates"] = refs
        pipeline["final"] = {
            "sentence_ranking_applied": True,
            "ranking_applied": True,
            "policy": POLICY,
            "response_candidate_ids": output_ids,
        }
        identity = pipeline.setdefault("identity", {})
        identity.setdefault("extensions", {})["extended_sentence_ranking"] = (
            self._runtime_identity())
        response["extended_sentence_ranking"] = self._metadata(
            target=True, applied=True,
            reason="scored_full_nonraw_pool_at_256",
            candidate_rows=rescored, score_rows=rescored,
            elapsed_ms=ranking_ms)
        self.last_pipeline = deepcopy(pipeline)
        return response

    def convert(self, text, context="", engine="neural", mode="batch",
                raw=False):
        started = time.perf_counter()
        with self.lock:
            response = super().convert(
                text, context, engine, mode, raw,
                generation_mode="independent")
            reason = self._eligible_reason(
                response, text, context, engine, raw)
            if reason is not None:
                return self._preserve(
                    response, reason=reason, started=started,
                    target=False)
            unique = self._unique_nonraw_pool(response, text)
            if not unique:
                return self._preserve(
                    response, reason="empty_nonraw_pool", started=started,
                    target=True)
            ranker = self._load_extended_ranker()
            unsupported = [row["text"] for row in unique
                           if not ranker.can_score(row["text"])]
            if unsupported:
                return self._preserve(
                    response, reason="candidate_exceeds_256", started=started,
                    target=True, candidate_rows=unique,
                    unsupported=unsupported)
            ranking_started = time.perf_counter()
            scores = ranker.score_pairs(
                [""] * len(unique), [row["text"] for row in unique])
            ranking_ms = (time.perf_counter() - ranking_started) * 1000
            if (not isinstance(scores, (list, tuple))
                    or len(scores) != len(unique)
                    or any(not _finite_score(score) for score in scores)):
                raise RuntimeError("extended sentence ranker returned invalid scores")
            result = self._apply_ranked_pool(
                response, text, unique, list(map(float, scores)), ranking_ms)
            result["engine_ms"] = round(
                (time.perf_counter() - started) * 1000, 3)
            return result


__all__ = ["ExtendedSeparatedConverter", "POLICY", "RUNTIME_MAX_LENGTH",
           "SCHEMA_VERSION", "TUNED_MAX_LENGTH"]
