"""Experimental separation of sentence proposal generation and ranking.

The production converter chain remains unchanged.  ``legacy`` delegates to it
without modifying the response.  ``independent`` reproduces its parent lattice
and ranking policy, then runs each proposal generator even when sentence
ranking was gated.  Unranked proposals are captured but cannot affect the
selected result.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time

from hanglish.core import ctc_word_proposals, mixed_span_proposals
from hanglish.core.ctc_union_converter import CTCUnionConverter
from hanglish.core.ctc_word_converter import CTCWordConverter
from hanglish.core.neural_converter import beam_paths, file_sha256


SCHEMA_VERSION = "separated-candidate-pipeline-v1"
GENERATION_MODES = frozenset({"legacy", "independent"})


def _finite_score(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value))


def _stable_id(prefix, value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"))
    return prefix + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


class SeparatedConverter(CTCUnionConverter):
    """Run the frozen mixed/CTC generators independently of ranker gates."""

    def __init__(self, *, generation_mode="legacy", **kwargs):
        super().__init__(**kwargs)
        self.generation_mode = self._validate_mode(generation_mode)
        self.last_separated_capture = None
        self.last_pipeline = None
        self._legacy_capture_active = False
        self._legacy_prepared = None
        self._legacy_full_paths = None

    @staticmethod
    def _validate_mode(value):
        if value not in GENERATION_MODES:
            raise ValueError("generation_mode must be 'legacy' or 'independent'")
        return value

    @staticmethod
    def _ranker_id(ranker):
        if ranker is None:
            return None
        for name in ("model_id", "version"):
            value = getattr(ranker, name, None)
            if isinstance(value, str) and value:
                return value
        return ranker.__class__.__name__

    @staticmethod
    def _artifact_model_id(name, artifacts):
        return _stable_id("model-v1-", {
            "name": name, "artifacts": dict(sorted(artifacts.items())),
        })

    @staticmethod
    def _identity():
        return {
            "experimental": True,
            "adapter": "SeparatedConverter",
            "source": "hanglish/core/separated_converter.py",
            "source_sha256": file_sha256(Path(__file__)),
            "schema_version": SCHEMA_VERSION,
        }

    def _prepare_candidates(self, text, context, *, learned, raw):
        result = super()._prepare_candidates(
            text, context, learned=learned, raw=raw)
        if getattr(self, "_legacy_capture_active", False):
            self._legacy_prepared = result
        return result

    def _extend_sentence_paths(self, text, context, paths):
        result = super()._extend_sentence_paths(text, context, paths)
        if getattr(self, "_legacy_capture_active", False):
            self._legacy_full_paths = deepcopy(result)
        return result

    @staticmethod
    def _path_identity(source, path):
        text, _, segments = path
        return _stable_id("path-v1-", {
            "source": source, "text": text, "segments": segments,
        })

    @staticmethod
    def _source_spans(source, text, segments):
        if not isinstance(segments, list):
            raise RuntimeError("candidate segments must be a list")
        cursor = 0
        output = []
        spans = []
        for segment in segments:
            if (not isinstance(segment, dict)
                    or not {"input_start", "input_end", "source", "text", "kind"}
                    <= set(segment)):
                raise RuntimeError("candidate segment fields are missing")
            start, end = segment["input_start"], segment["input_end"]
            segment_source = segment["source"]
            segment_text = segment["text"]
            if (isinstance(start, bool) or not isinstance(start, int)
                    or isinstance(end, bool) or not isinstance(end, int)
                    or start != cursor or not start < end <= len(source)
                    or not isinstance(segment_source, str)
                    or segment_source != source[start:end]
                    or not isinstance(segment_text, str)):
                raise RuntimeError("candidate does not preserve source coverage")
            copied = segment_source == segment_text
            spans.append({
                "input_start": start,
                "input_end": end,
                "source": segment_source,
                "text": segment_text,
                "kind": segment["kind"],
                "action": "copy" if copied else "convert",
                "copy_exact": True if copied else None,
            })
            cursor = end
            output.append(segment_text)
        if cursor != len(source) or "".join(output) != text:
            raise RuntimeError("candidate path does not cover the full source")
        return spans

    @staticmethod
    def _stage_id(request_id, name, input_ids, generated_ids, output_ids):
        return _stable_id("stage-v1-", {
            "request_id": request_id,
            "name": name,
            "input_path_ids": input_ids,
            "generated_path_ids": generated_ids,
            "output_path_ids": output_ids,
        })

    def _parent_paths(self, text, context, engine, raw):
        learned = engine == "neural"
        if learned and not raw:
            self.load_models()
        pieces, static_pieces, warnings, context_used, ranking_applied = (
            self._prepare_candidates(
                text, context, learned=learned, raw=raw))
        local_paths = beam_paths(pieces)
        paths = local_paths
        sentence_ranker = getattr(self, "sentence_ranker", None)
        use_sentence = self.ranker_kind != "hybrid" or len(text.split()) > 1
        sentence_ranking_applied = False
        sentence_reason = "request_not_eligible"
        if (learned and not raw and self.context_ranking
                and sentence_ranker is not None and len(paths) > 1
                and use_sentence):
            proposals = {path[0]: path for path in paths}
            for path in beam_paths(static_pieces):
                proposals.setdefault(path[0], path)
            complete = list(proposals.values())
            if all(sentence_ranker.can_score(path[0]) for path in complete):
                extras = self._numeric_sentence_extras(
                    pieces, static_pieces, complete)
                if extras and all(sentence_ranker.can_score(path[0])
                                  for path in extras):
                    complete.extend(extras)
                scores = self._score_candidates(
                    sentence_ranker, context, [path[0] for path in complete],
                    "parent sentence ranker")
                paths = sorted([
                    (path[0], score, path[2])
                    for path, score in zip(complete, scores)
                ], key=lambda path: (-path[1], path[0]))
                sentence_ranking_applied = True
                sentence_reason = "scored"
                ranking_applied = True
                context_used = bool(context.strip())
            else:
                sentence_reason = "unsupported_by_ranker"
                warnings.append(
                    "문장 모델의 길이 범위를 넘어 단어별 문맥 순위를 사용했습니다.")
        elif learned and not raw and sentence_ranker is None:
            sentence_reason = "ranker_unavailable"
        elif learned and not raw and len(paths) <= 1:
            sentence_reason = "single_parent_path"
        elif learned and not raw and not use_sentence:
            sentence_reason = "single_word_hybrid_policy"
        return {
            "learned": learned,
            "paths": paths,
            "local_paths": local_paths,
            "warnings": warnings,
            "context_used": context_used,
            "ranking_applied": ranking_applied,
            "sentence_ranking_applied": sentence_ranking_applied,
            "sentence_reason": sentence_reason,
        }

    def _score_candidates(self, ranker, context, candidates, label):
        with self.lock:
            scores = ranker.score_pairs(
                [context] * len(candidates), list(candidates))
        if (not isinstance(scores, (list, tuple))
                or len(scores) != len(candidates)
                or any(not _finite_score(score) for score in scores)):
            raise RuntimeError(f"{label} returned invalid scores")
        return [float(score) for score in scores]

    def _candidate_row(self, source, path, *, origin, score, score_type,
                       model_id, evaluated, reason, selection_eligible):
        text, _, segments = path
        if score is not None and not _finite_score(score):
            raise RuntimeError("candidate score must be finite or null")
        path_id = self._path_identity(source, path)
        spans = self._source_spans(source, text, segments)
        return {
            "candidate_id": path_id,
            "path_id": path_id,
            "text": text,
            "segments": deepcopy(segments),
            "source_spans": spans,
            "source_coverage": {
                "complete": True, "input_start": 0, "input_end": len(source),
            },
            "origin": origin,
            "origins": [origin],
            "origin_score": (float(path[1])
                             if _finite_score(path[1]) else None),
            "origin_score_type": (
                score_type if origin in ("parent", "legacy_final_pool")
                else ("inherited_parent_path_score" if origin == "mixed_span"
                      else ("proposal_placeholder"
                            if origin in ("ctc_word", "ctc_growth")
                            else None))),
            "origin_score_model_id": (
                model_id if origin in ("parent", "legacy_final_pool")
                else None),
            "score": score,
            "score_type": score_type,
            "model_id": model_id,
            "evaluated": evaluated,
            "reason": reason,
            "selection_eligible": selection_eligible,
        }

    @staticmethod
    def _merge_candidate(pool, by_path, row):
        previous = by_path.get(row["path_id"])
        if previous is None:
            row["pool_index"] = len(pool)
            pool.append(row)
            by_path[row["path_id"]] = row
            return row
        if row["origin"] not in previous["origins"]:
            previous["origins"].append(row["origin"])
        return previous

    def _stage_capture(self, request_id, name, input_paths, generated_rows,
                       output_paths, *, evaluated, reason, score_type,
                       model_id, counts, source_only, prediction=None):
        input_ids = [self._path_identity(self._pipeline_source, path)
                     for path in input_paths]
        generated_ids = [row["path_id"] for row in generated_rows]
        output_ids = [self._path_identity(self._pipeline_source, path)
                      for path in output_paths]
        result = {
            "stage_id": self._stage_id(
                request_id, name, input_ids, generated_ids, output_ids),
            "name": name,
            "input_path_ids": input_ids,
            "generated_path_ids": generated_ids,
            "output_path_ids": output_ids,
            "evaluated": evaluated,
            "reason": reason,
            "score_type": score_type,
            "model_id": model_id,
            "counts": dict(counts),
            "source_only": source_only,
            "source_truncated": False,
        }
        if prediction is not None:
            result["prediction"] = deepcopy(prediction)
        return result

    def _score_generated(self, *, name, context, paths, extras, pool, by_path,
                         parent_ranked):
        ranker = getattr(self, "sentence_ranker", None)
        model_id = self._ranker_id(ranker)
        generated = []
        for extra in extras:
            row = self._candidate_row(
                self._pipeline_source, extra, origin=name, score=None,
                score_type=None, model_id=None, evaluated=False,
                reason=("pending_sentence_ranking" if parent_ranked
                        else "parent_sentence_ranking_unavailable"),
                selection_eligible=False)
            generated.append(self._merge_candidate(pool, by_path, row))
        if name == "mixed_span":
            for row in generated:
                row["origin_score_type"] = (
                    "inherited_sentence_ranker_score" if parent_ranked
                    else "inherited_local_path_score")
                row["origin_score_model_id"] = (
                    model_id if parent_ranked else None)
        if not parent_ranked:
            return paths, generated, 0, False, (
                "parent_sentence_ranking_unavailable")
        if ranker is None:
            return paths, generated, 0, False, "ranker_unavailable"
        accepted = []
        accepted_rows = []
        existing = {path[0] for path in paths}
        accepted_texts = set()
        for extra, row in zip(extras, generated):
            if extra[0] in existing or extra[0] in accepted_texts:
                row["reason"] = "duplicate_output_not_selected"
            elif ranker.can_score(extra[0]):
                accepted.append(extra)
                accepted_rows.append(row)
                accepted_texts.add(extra[0])
            else:
                row["reason"] = "unsupported_by_ranker"
        if not accepted:
            return paths, generated, 0, True, "no_scoreable_proposals"
        scores = self._score_candidates(
            ranker, context, [path[0] for path in accepted], f"{name} ranker")
        if len(existing) != len(paths):
            raise RuntimeError(f"{name} pool contains duplicate paths")
        for row, score in zip(accepted_rows, scores):
            row.update({
                "score": score,
                "score_type": "sentence_ranker",
                "model_id": model_id,
                "evaluated": True,
                "reason": "scored",
                "selection_eligible": True,
            })
        additional = [(path[0], score, path[2])
                      for path, score in zip(accepted, scores)]
        output = sorted([*paths, *additional],
                        key=lambda path: (-path[1], path[0]))
        return output, generated, len(accepted), True, "scored"

    def _run_mixed_stage(self, request_id, context, paths, pool, by_path,
                         parent_ranked, eligible):
        if not eligible:
            extras = []
            output = paths
            generated = []
            accepted = 0
            evaluated = False
            reason = "request_not_eligible"
        else:
            extras = mixed_span_proposals.sentence_extras(
                self._pipeline_source, paths, self.word_options, limit=96)
            output, generated, accepted, evaluated, reason = (
                self._score_generated(
                    name="mixed_span", context=context, paths=paths,
                    extras=extras, pool=pool, by_path=by_path,
                    parent_ranked=parent_ranked))
            if not extras:
                evaluated = False
                reason = "no_generated_proposals"
        self.mixed_span_counts = {
            "generated": len(extras), "accepted": accepted,
        }
        stage = self._stage_capture(
            request_id, "mixed_span", paths, generated, output,
            evaluated=evaluated, reason=reason,
            score_type="sentence_ranker" if evaluated else None,
            model_id=(self._ranker_id(getattr(self, "sentence_ranker", None))
                      if evaluated else None),
            counts={"model_calls": 0, "generated": len(extras),
                    "accepted": accepted},
            source_only=True)
        stage["generation_reason"] = (
            "complete" if eligible else "request_not_eligible")
        stage["generator_model_id"] = "rule:mixed_span_proposals"
        return output, stage

    def _predict_ctc(self, name, model, validate):
        started = time.perf_counter()
        predictions = model.generate([self._pipeline_source])
        elapsed = (time.perf_counter() - started) * 1000
        if (not isinstance(predictions, list) or len(predictions) != 1
                or not isinstance(predictions[0], dict)):
            raise RuntimeError(f"{name} model returned an invalid prediction batch")
        prediction = predictions[0]
        validate(prediction)
        return dict(prediction), elapsed

    def _run_ctc_stage(self, request_id, name, context, paths, pool, by_path,
                       parent_ranked, eligible, *, model_loader, validate):
        generation_reason = "request_not_eligible"
        prediction = None
        model_calls = 0
        elapsed = 0.0
        extras = []
        if eligible:
            generation_reason = CTCWordConverter._generation_reason(
                self._pipeline_source)
            if generation_reason is None:
                model = model_loader()
                prediction, elapsed = self._predict_ctc(name, model, validate)
                model_calls = 1
                generation_reason = prediction["reason"]
                if generation_reason == "complete":
                    extras = ctc_word_proposals.sentence_extras(
                        self._pipeline_source, prediction["text"], paths,
                        limit=96)
            else:
                prediction = {"text": None, "reason": generation_reason,
                              "generated_tokens": 0}
        output, generated, accepted, evaluated, ranking_reason = (
            self._score_generated(
                name=name, context=context, paths=paths, extras=extras,
                pool=pool, by_path=by_path, parent_ranked=parent_ranked))
        if not extras:
            evaluated = False
            if not eligible:
                ranking_reason = "request_not_eligible"
            elif generation_reason == "complete":
                ranking_reason = "no_generated_proposals"
            else:
                ranking_reason = "generation_incomplete"
        counts = {
            "model_calls": model_calls,
            "generation_requests": model_calls,
            "completed_generations": int(generation_reason == "complete"),
            "generated_proposals": len(extras),
            "accepted_proposals": accepted,
        }
        stage = self._stage_capture(
            request_id, name, paths, generated, output,
            evaluated=evaluated, reason=ranking_reason,
            score_type="sentence_ranker" if evaluated else None,
            model_id=(self._ranker_id(getattr(self, "sentence_ranker", None))
                      if evaluated else None),
            counts=counts, source_only=True, prediction=prediction)
        stage["generation_reason"] = generation_reason
        stage["generation_ms"] = round(elapsed, 3)
        artifacts = (getattr(self, "ctc_word_artifacts", {})
                     if name == "ctc_word"
                     else getattr(self, "growth_artifacts", {}))
        stage["generator_model_id"] = self._artifact_model_id(name, artifacts)
        return output, stage, prediction, elapsed, counts

    def _legacy_pipeline(self, text, context, engine, mode, raw, response):
        if self._legacy_full_paths is not None:
            paths = self._legacy_full_paths
        elif self._legacy_prepared is not None:
            paths = beam_paths(self._legacy_prepared[0])
        else:
            raise RuntimeError("legacy conversion did not expose its parent lattice")
        request_id = _stable_id("request-v1-", {
            "input": text, "context": context, "engine": engine,
            "mode": mode, "raw": raw,
        })
        sentence_ranked = bool(response.get("sentence_ranking_applied"))
        ranker_id = (self._ranker_id(getattr(self, "sentence_ranker", None))
                     if sentence_ranked else None)
        pool = []
        by_path = {}
        for path in paths:
            row = self._candidate_row(
                text, path, origin="legacy_final_pool", score=float(path[1]),
                score_type=("sentence_ranker" if sentence_ranked
                            else "local_path_score"),
                model_id=ranker_id, evaluated=sentence_ranked,
                reason=("scored" if sentence_ranked
                        else "sentence_ranking_not_applied"),
                selection_eligible=True)
            self._merge_candidate(pool, by_path, row)
        response_candidates = []
        for index, candidate in enumerate(response["candidates"]):
            candidate_path = (candidate["text"], 0.0, candidate["segments"])
            row = by_path.get(self._path_identity(text, candidate_path))
            if row is None:
                fallback = (candidate["text"], 0.0,
                            deepcopy(candidate["segments"]))
                row = self._candidate_row(
                    text, fallback, origin="raw_fallback", score=None,
                    score_type=None, model_id=None, evaluated=False,
                    reason="appended_for_user", selection_eligible=True)
                self._merge_candidate(pool, by_path, row)
            response_candidates.append({
                "index": index, "candidate_id": row["candidate_id"],
                "candidate": deepcopy(candidate),
            })
        output_ids = [row["path_id"] for row in pool]
        counts = {
            "mixed_generated": response.get("mixed_span", {}).get("generated", 0),
            "mixed_accepted": response.get("mixed_span", {}).get("accepted", 0),
            "ctc_word_model_calls": response.get("ctc_word", {}).get(
                "generation_requests", 0),
            "ctc_growth_model_calls": response.get("ctc_union", {}).get(
                "generation_requests", 0),
        }
        return {
            "schema_version": SCHEMA_VERSION,
            "identity": self._identity(),
            "mode": "legacy",
            "request_id": request_id,
            "request": {"input": text, "context": context, "engine": engine,
                        "mode": mode, "raw": raw},
            "source_only_generation": True,
            "source_truncated": False,
            "selection_policy": "legacy_parent_policy",
            "parent": {
                "sentence_ranking_applied": sentence_ranked,
                "sentence_ranking_reason": (
                    "scored" if sentence_ranked else "not_applied"),
                "path_ids": output_ids,
            },
            "stages": [{
                "stage_id": self._stage_id(
                    request_id, "legacy_chain", [], output_ids, output_ids),
                "name": "legacy_chain",
                "input_path_ids": [],
                "generated_path_ids": output_ids,
                "output_path_ids": output_ids,
                "evaluated": sentence_ranked,
                "reason": "complete",
                "score_type": ("sentence_ranker" if sentence_ranked
                               else "local_path_score"),
                "model_id": ranker_id,
                "counts": counts,
                "source_only": True,
                "source_truncated": False,
            }],
            "pool": pool,
            "response_candidates": response_candidates,
        }

    def _independent_convert(self, text, context, engine, mode, raw):
        started = time.perf_counter()
        self._pipeline_source = text
        request_id = _stable_id("request-v1-", {
            "input": text, "context": context, "engine": engine,
            "mode": mode, "raw": raw,
        })
        parent = self._parent_paths(text, context, engine, raw)
        paths = parent["paths"]
        sentence_ranked = parent["sentence_ranking_applied"]
        ranker_id = self._ranker_id(getattr(self, "sentence_ranker", None))
        pool = []
        by_path = {}
        for path in paths:
            row = self._candidate_row(
                text, path, origin="parent", score=float(path[1]),
                score_type=("sentence_ranker" if sentence_ranked
                            else "local_path_score"),
                model_id=ranker_id if sentence_ranked else None,
                evaluated=sentence_ranked,
                reason=("scored" if sentence_ranked
                        else parent["sentence_reason"]),
                selection_eligible=True)
            self._merge_candidate(pool, by_path, row)
        local_ids = [self._path_identity(text, path)
                     for path in parent["local_paths"]]
        path_ids = [self._path_identity(text, path) for path in paths]
        stages = [{
            "stage_id": self._stage_id(
                request_id, "parent_lattice", [], local_ids, local_ids),
            "name": "parent_lattice",
            "input_path_ids": [],
            "generated_path_ids": local_ids,
            "output_path_ids": local_ids,
            "evaluated": False,
            "reason": "complete",
            "score_type": "local_path_score",
            "model_id": None,
            "counts": {"generated": len(local_ids)},
            "source_only": True,
            "source_truncated": False,
        }, {
            "stage_id": self._stage_id(
                request_id, "sentence_ranking", local_ids, [], path_ids),
            "name": "sentence_ranking",
            "input_path_ids": local_ids,
            "generated_path_ids": [],
            "output_path_ids": path_ids,
            "evaluated": sentence_ranked,
            "reason": parent["sentence_reason"],
            "score_type": "sentence_ranker" if sentence_ranked else None,
            "model_id": ranker_id if sentence_ranked else None,
            "counts": {"input": len(local_ids), "output": len(path_ids)},
            "source_only": False,
            "source_truncated": False,
        }]
        eligible = parent["learned"] and not raw
        paths, stage = self._run_mixed_stage(
            request_id, context, paths, pool, by_path, sentence_ranked,
            eligible)
        stages.append(stage)

        paths, stage, prediction, elapsed, counts = self._run_ctc_stage(
            request_id, "ctc_word", context, paths, pool, by_path,
            sentence_ranked, eligible, model_loader=self.load_word_model,
            validate=CTCWordConverter._validate_prediction)
        stages.append(stage)
        self.ctc_word_prediction = prediction
        self.ctc_word_ms = elapsed
        self.ctc_word_counts = {key: counts[key] for key in (
            "generation_requests", "completed_generations",
            "generated_proposals", "accepted_proposals")}

        paths, stage, prediction, elapsed, counts = self._run_ctc_stage(
            request_id, "ctc_growth", context, paths, pool, by_path,
            sentence_ranked, eligible, model_loader=self.load_growth_model,
            validate=CTCWordConverter._validate_prediction)
        stages.append(stage)
        self.ctc_union_prediction = prediction
        self.ctc_union_ms = elapsed
        self.ctc_union_counts = {key: counts[key] for key in (
            "generation_requests", "completed_generations",
            "generated_proposals", "accepted_proposals")}

        candidates = [
            {"text": output, "score": round(score, 4),
             "segments": deepcopy(segments)}
            for output, score, segments in paths[:5]
        ]
        if not any(candidate["text"] == text for candidate in candidates):
            original_path = (text, 0.0, [{
                "input_start": 0, "input_end": len(text), "source": text,
                "text": text, "kind": "raw",
            }])
            original = {"text": text, "score": None,
                        "segments": deepcopy(original_path[2])}
            candidates = candidates[:4] + [original]
            row = self._candidate_row(
                text, original_path, origin="raw_fallback", score=None,
                score_type=None, model_id=None, evaluated=False,
                reason="appended_for_user", selection_eligible=True)
            self._merge_candidate(pool, by_path, row)

        response = {
            "input": text, "context": context, "engine": engine, "mode": mode,
            "raw": raw, "version": self.version, "candidates": candidates,
            "result": candidates[0]["text"],
            "ranker": (self.ranker_kind if parent["learned"] and not raw
                       and self.context_ranking else None),
            "model_artifacts": (getattr(self, "model_artifacts", {})
                                if parent["learned"] and not raw else {}),
            "context_used": parent["context_used"],
            "external_context_used": parent["context_used"],
            "ranking_applied": parent["ranking_applied"],
            "sentence_ranking_applied": sentence_ranked,
            "warnings": list(dict.fromkeys(parent["warnings"])),
        }
        response_candidates = []
        for index, candidate in enumerate(candidates):
            candidate_path = (candidate["text"], 0.0, candidate["segments"])
            row = by_path[self._path_identity(text, candidate_path)]
            response_candidates.append({
                "index": index,
                "candidate_id": row["candidate_id"],
                "candidate": deepcopy(candidate),
            })
        selection_policy = ("sentence_ranker_comparable_scores"
                            if sentence_ranked
                            else "retain_parent_order_capture_unscored")
        pipeline = {
            "schema_version": SCHEMA_VERSION,
            "identity": self._identity(),
            "mode": "independent",
            "request_id": request_id,
            "request": {"input": text, "context": context, "engine": engine,
                        "mode": mode, "raw": raw},
            "source_only_generation": True,
            "source_truncated": False,
            "selection_policy": selection_policy,
            "parent": {
                "sentence_ranking_applied": sentence_ranked,
                "sentence_ranking_reason": parent["sentence_reason"],
                "path_ids": path_ids,
            },
            "stages": stages,
            "pool": pool,
            "response_candidates": response_candidates,
        }
        self.last_separated_capture = deepcopy(pipeline)
        self.last_pipeline = deepcopy(pipeline)
        response["pipeline"] = pipeline
        response["mixed_span"] = dict(self.mixed_span_counts)
        response["ctc_word"] = {
            "experimental": True, **self.ctc_word_counts,
            "prediction": deepcopy(self.ctc_word_prediction),
            "generation_ms": round(self.ctc_word_ms, 3),
            "parent_version": getattr(self, "word_parent_version", None),
            "artifacts": dict(getattr(self, "ctc_word_artifacts", {})),
        }
        response["ctc_union"] = {
            "experimental": True, **self.ctc_union_counts,
            "prediction": deepcopy(self.ctc_union_prediction),
            "generation_ms": round(self.ctc_union_ms, 3),
            "parent_version": getattr(self, "growth_parent_version", None),
            "artifacts": dict(getattr(self, "growth_artifacts", {})),
        }
        response["engine_ms"] = round((time.perf_counter() - started) * 1000, 3)
        return response

    def convert(self, text, context="", engine="neural", mode="batch",
                raw=False, *, generation_mode=None):
        selected = self._validate_mode(
            self.generation_mode if generation_mode is None else generation_mode)
        if selected == "legacy":
            with self.lock:
                self.last_separated_capture = None
                self.last_pipeline = None
                self._legacy_prepared = None
                self._legacy_full_paths = None
                self._legacy_capture_active = True
                try:
                    response = super().convert(
                        text, context, engine, mode, raw)
                finally:
                    self._legacy_capture_active = False
                self.last_pipeline = self._legacy_pipeline(
                    text, context, engine, mode, raw, response)
                return response
        with self.lock:
            self.last_pipeline = None
            self.mixed_span_counts = {"generated": 0, "accepted": 0}
            self.ctc_word_counts = self._empty_counts()
            self.ctc_word_prediction = None
            self.ctc_word_ms = 0.0
            self.ctc_union_counts = self._empty_growth_counts()
            self.ctc_union_prediction = None
            self.ctc_union_ms = 0.0
            return self._independent_convert(
                text, context, engine, mode, raw)


__all__ = ["GENERATION_MODES", "SCHEMA_VERSION", "SeparatedConverter"]
