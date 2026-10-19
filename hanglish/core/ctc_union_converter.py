"""Experimental union of frozen CTC1 paths and growth-CTC word proposals."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

from hanglish.core import ctc_word_proposals
from hanglish.core.ctc_word_converter import CTCWordConverter
from hanglish.core.neural_converter import ROOT, file_sha256


DEFAULT_GROWTH_CHECKPOINT = (
    ROOT / ".omx/artifacts/sentence-ctc-growth-pilot-v1/model.pt")
EXPECTED_GROWTH_CHECKPOINT_SHA256 = (
    "6aba4c441f3f0cb7cd870be80ccbf39a215c87a76890d5e2a90fdbb14c9618a7")
EXPECTED_GROWTH_METADATA_SHA256 = (
    "954f8022a2a674dfd379002f0713998e508bae844a0ce211e2cf88131c54ed78")


class CTCUnionConverter(CTCWordConverter):
    """Add growth-CTC proposals after the complete frozen CTC1 parent pool."""

    def __init__(self, *, growth_checkpoint=DEFAULT_GROWTH_CHECKPOINT,
                 growth_model=None, **kwargs):
        super().__init__(**kwargs)
        self.growth_checkpoint = Path(growth_checkpoint)
        self.growth_model = growth_model
        self._growth_model_loaded = growth_model is not None
        self.growth_artifacts = self._growth_artifact_identity()
        self._ctc_parent_version = self.version
        self.growth_parent_version = self._ctc_parent_version
        self._stamp_growth_version()
        self.ctc_union_counts = self._empty_growth_counts()
        self.ctc_union_prediction = None
        self.ctc_union_ms = 0.0

    @staticmethod
    def _empty_growth_counts():
        return {
            "generation_requests": 0,
            "completed_generations": 0,
            "generated_proposals": 0,
            "accepted_proposals": 0,
        }

    def _growth_artifact_identity(self):
        artifacts = {
            "hanglish/core/ctc_union_converter.py": file_sha256(Path(__file__)),
        }
        for path in (self.growth_checkpoint,
                     self.growth_checkpoint.with_suffix(".json")):
            if not path.is_file():
                continue
            try:
                name = str(path.relative_to(ROOT))
            except ValueError:
                name = "external/" + path.name
            artifacts[name] = file_sha256(path)
        return artifacts

    def _stamp_growth_version(self):
        self.growth_parent_version = self._ctc_parent_version
        payload = json.dumps({"parent": self.growth_parent_version,
                              "artifacts": self.growth_artifacts}, sort_keys=True)
        self.version = "experimental-ctc-union-1-" + hashlib.sha256(
            payload.encode()).hexdigest()[:12]

    def load_models(self):
        with self.lock:
            was_loaded = self.models_loaded
            super().load_models()
            if not was_loaded:
                self._ctc_parent_version = self.version
            self.model_artifacts = {
                **getattr(self, "model_artifacts", {}),
                **self.growth_artifacts,
            }
            self._stamp_growth_version()

    def load_growth_model(self):
        """Load and return the immutable growth SentenceCTC checkpoint."""
        with self.lock:
            if self._growth_model_loaded:
                return self.growth_model
            current = self._growth_artifact_identity()
            if current != self.growth_artifacts:
                raise ValueError("CTC-union checkpoint or source identity changed")
            if self.growth_checkpoint == DEFAULT_GROWTH_CHECKPOINT:
                checkpoint = str(self.growth_checkpoint.relative_to(ROOT))
                metadata = str(
                    self.growth_checkpoint.with_suffix(".json").relative_to(ROOT))
                if (current.get(checkpoint) != EXPECTED_GROWTH_CHECKPOINT_SHA256
                        or current.get(metadata) != EXPECTED_GROWTH_METADATA_SHA256):
                    raise ValueError("growth SentenceCTC checkpoint identity differs")
            from hanglish.models.sentence_ctc import SentenceCTC

            self.growth_model = SentenceCTC.load(self.growth_checkpoint)
            self._growth_model_loaded = True
            return self.growth_model

    def _extend_sentence_paths(self, text, context, paths):
        # This single parent call runs mixed-span and frozen CTC1 augmentation
        # exactly once. Every returned parent tuple and score is retained.
        paths = super()._extend_sentence_paths(text, context, paths)
        reason = self._generation_reason(text)
        if reason is not None:
            self.ctc_union_prediction = {
                "text": None, "reason": reason, "generated_tokens": 0}
            return paths

        started = time.perf_counter()
        self.ctc_union_counts["generation_requests"] = 1
        with self.lock:
            predictions = self.load_growth_model().generate([text])
        self.ctc_union_ms = (time.perf_counter() - started) * 1000
        if (not isinstance(predictions, list) or len(predictions) != 1
                or not isinstance(predictions[0], dict)):
            raise RuntimeError("CTC-union model returned an invalid prediction batch")
        prediction = predictions[0]
        self._validate_prediction(prediction)
        self.ctc_union_prediction = dict(prediction)
        if prediction["reason"] != "complete":
            return paths
        self.ctc_union_counts["completed_generations"] = 1

        extras = ctc_word_proposals.sentence_extras(
            text, prediction["text"], paths, limit=96)
        self.ctc_union_counts["generated_proposals"] = len(extras)
        accepted = [path for path in extras if self.sentence_ranker.can_score(path[0])]
        self.ctc_union_counts["accepted_proposals"] = len(accepted)
        if not accepted:
            return paths
        scores = self.sentence_ranker.score_pairs(
            [context] * len(accepted), [path[0] for path in accepted])
        if (len(scores) != len(accepted)
                or any(isinstance(score, bool)
                       or not isinstance(score, (int, float))
                       or not math.isfinite(score) for score in scores)):
            raise RuntimeError("CTC-union ranker returned invalid scores")
        existing = {path[0] for path in paths}
        if len(existing) != len(paths) or any(path[0] in existing for path in accepted):
            raise RuntimeError("CTC-union pool contains duplicate paths")
        additional = [(path[0], float(score), path[2])
                      for path, score in zip(accepted, scores)]
        return sorted([*paths, *additional], key=lambda path: (-path[1], path[0]))

    def convert(self, text, context="", engine="neural", mode="batch", raw=False):
        with self.lock:
            self.ctc_union_counts = self._empty_growth_counts()
            self.ctc_union_prediction = None
            self.ctc_union_ms = 0.0
            response = super().convert(text, context, engine, mode, raw)
            response["ctc_union"] = {
                "experimental": True, **self.ctc_union_counts,
                "prediction": (dict(self.ctc_union_prediction)
                               if self.ctc_union_prediction is not None else None),
                "generation_ms": round(self.ctc_union_ms, 3),
                "parent_version": self.growth_parent_version,
                "artifacts": dict(self.growth_artifacts),
            }
            return response


__all__ = ["CTCUnionConverter", "DEFAULT_GROWTH_CHECKPOINT"]
