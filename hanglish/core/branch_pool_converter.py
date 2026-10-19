"""Preserve both bounded sentence pools and score their union once."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import threading
import time

from hanglish.core.ensemble_converter import _validate_candidate
from hanglish.core.neural_converter import beam_paths, file_sha256


@dataclass
class BranchPool:
    paths: list
    warnings: list[str]
    reason: str | None = None


def collect_pool(branch, text, context):
    """Mirror NeuralConverter's admission gates before sentence score_pairs."""
    pieces, static, warnings, _, _ = branch._prepare_candidates(
        text, context, learned=True, raw=False)
    paths = beam_paths(pieces)
    ranker = branch.sentence_ranker
    use_sentence = branch.ranker_kind != "hybrid" or len(text.split()) > 1
    if not branch.context_ranking or ranker is None or len(paths) <= 1 or not use_sentence:
        return BranchPool(paths, warnings, "sentence_ranking_unavailable")
    proposals = {path[0]: path for path in paths}
    for path in beam_paths(static):
        proposals.setdefault(path[0], path)
    complete = list(proposals.values())
    if not all(ranker.can_score(path[0]) for path in complete):
        return BranchPool(complete, warnings, "sentence_budget_exceeded")
    extras = branch._numeric_sentence_extras(pieces, static, complete)
    if extras and all(ranker.can_score(path[0]) for path in extras):
        complete.extend(extras)
    if len(complete) > 96:
        raise ValueError("branch pool exceeds the existing 48+48 path budget")
    return BranchPool(complete, warnings)


def union_paths(baseline, supported):
    unique = {}
    for path in [*baseline, *supported]:
        unique.setdefault(path[0], path)
    return list(unique.values())


class BranchPoolConverter:
    def __init__(self, models, *, experimental=False):
        if experimental is not True:
            raise ValueError("BranchPoolConverter requires experimental=True")
        self.models = models
        self.lock = threading.RLock()
        self.models_loaded = False
        self.last_trace = None

    def load_models(self):
        with self.lock, self.models.lock:
            if self.models_loaded:
                return
            self.models.load_models()
            source_sha = file_sha256(Path(__file__))
            self.model_artifacts = dict(self.models.model_artifacts)
            self.model_artifacts["experimental/branch_pool_converter.py"] = source_sha
            identity = {"models_version": self.models.version, "source_sha256": source_sha}
            digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            self.version = "experimental-branch-pool-1-" + digest[:12]
            self.models_loaded = True

    def convert(self, text, context="", engine="neural", mode="batch", raw=False):
        started = time.perf_counter()
        with self.lock, self.models.lock:
            self.last_trace = None
            if raw or engine != "neural":
                return self.models.baseline.convert(text, context, engine=engine, mode=mode, raw=raw)
            self.load_models()
            baseline = self.models.baseline
            supported = self.models.supported
            phase_started = time.perf_counter()
            old = collect_pool(baseline, text, context)
            old_ms = (time.perf_counter() - phase_started) * 1000
            # A baseline fallback does not need the supported branch.
            phase_started = time.perf_counter()
            new = collect_pool(supported, text, context) if old.reason is None else None
            new_ms = (time.perf_counter() - phase_started) * 1000
            fallback = old.reason or (new.reason if new else None)
            self.last_trace = {"baseline_pool": [p[0] for p in old.paths],
                               "supported_pool": [p[0] for p in new.paths] if new else [],
                               "union_pool": [], "fallback_reason": fallback,
                               "baseline_prepare_ms": old_ms, "supported_prepare_ms": new_ms,
                               "sentence_score_ms": None}
            if fallback:
                response = baseline.convert(text, context, engine=engine, mode=mode)
            else:
                paths = union_paths(old.paths, new.paths)
                if len(paths) > 192:
                    raise ValueError("union pool exceeds two bounded branch pools")
                for output, score, segments in paths:
                    _validate_candidate({"text": output, "score": score, "segments": segments}, text)
                self.last_trace["union_pool"] = [p[0] for p in paths]
                phase_started = time.perf_counter()
                scores = list(baseline.sentence_ranker.score_pairs([context] * len(paths),
                                                                 [p[0] for p in paths]))
                self.last_trace["sentence_score_ms"] = (time.perf_counter() - phase_started) * 1000
                if len(scores) != len(paths) or any(not math.isfinite(s) for s in scores):
                    raise ValueError("sentence ranker returned invalid union scores")
                ranked = sorted(zip(paths, scores), key=lambda pair: (-pair[1], pair[0][0]))
                candidates = [{"text": p[0], "score": round(float(score), 4),
                               "segments": deepcopy(p[2])} for p, score in ranked[:5]]
                if not any(c["text"] == text for c in candidates):
                    candidates = candidates[:4] + [{"text": text, "score": None,
                        "segments": [{"input_start": 0, "input_end": len(text),
                                      "source": text, "text": text, "kind": "raw"}]}]
                response = {"input": text, "context": context, "engine": engine, "mode": mode,
                            "raw": raw, "result": candidates[0]["text"], "candidates": candidates,
                            "ranker": baseline.ranker_kind, "context_used": bool(context.strip()),
                            "external_context_used": bool(context.strip()), "ranking_applied": True,
                            "sentence_ranking_applied": True,
                            "warnings": list(dict.fromkeys(old.warnings + new.warnings))}
            response.update({"version": self.version, "model_artifacts": self.model_artifacts,
                "engine_ms": round((time.perf_counter() - started) * 1000, 3),
                "branch_pool_union": {"experimental": True, "fallback_reason": fallback,
                    "branch_pool_preserved": not bool(fallback),
                    "final_candidate_preservation_guaranteed": False}})
            return response
