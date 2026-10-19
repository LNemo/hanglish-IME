"""Experimental runtime for source-aligned words from a sentence transducer."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

from hanglish.core import neural_word_proposals
from hanglish.core.mixed_span_converter import MixedSpanConverter
from hanglish.core.neural_converter import ROOT, file_sha256
from hanglish.models import sentence_transducer
from hanglish.models.sentence_transducer import HANGUL


DEFAULT_WORD_CHECKPOINT = (
    ROOT / ".omx/artifacts/sentence-transducer-balanced-pilot-v1/model.pt")
EXPECTED_BALANCED_CHECKPOINT_SHA256 = (
    "7a4f4ade509663515760b9b396d05a94351b9262cd25e0eb2625cbe2672eb4d7")


class NeuralWordConverter(MixedSpanConverter):
    """Add bounded generated-word paths after all parent path generation."""

    def __init__(self, *, word_checkpoint=DEFAULT_WORD_CHECKPOINT,
                 word_model=None, **kwargs):
        super().__init__(**kwargs)
        self.word_checkpoint = Path(word_checkpoint)
        self.word_model = word_model
        self._word_model_loaded = word_model is not None
        self.neural_word_artifacts = self._artifact_identity()
        self._mixed_version = self.version
        self.word_parent_version = self._mixed_version
        self._stamp_word_version()
        self.neural_word_counts = self._empty_counts()
        self.neural_word_prediction = None
        self.neural_word_ms = 0.0

    @staticmethod
    def _empty_counts():
        return {
            "generation_requests": 0,
            "completed_generations": 0,
            "generated_proposals": 0,
            "accepted_proposals": 0,
        }

    def _artifact_identity(self):
        metadata = self.word_checkpoint.with_suffix(".json")
        artifacts = {
            "hanglish/core/neural_word_converter.py": file_sha256(Path(__file__)),
            "hanglish/core/neural_word_proposals.py": file_sha256(
                Path(neural_word_proposals.__file__)),
            "hanglish/models/sentence_transducer.py": file_sha256(
                Path(sentence_transducer.__file__)),
        }
        for path in (self.word_checkpoint, metadata):
            if path.is_file():
                try:
                    name = str(path.relative_to(ROOT))
                except ValueError:
                    name = "external/" + path.name
                artifacts[name] = file_sha256(path)
        return artifacts

    def _stamp_word_version(self):
        self.word_parent_version = self._mixed_version
        payload = json.dumps({
            "parent": self.word_parent_version,
            "artifacts": self.neural_word_artifacts,
        }, sort_keys=True)
        self.version = "experimental-neural-word-1-" + hashlib.sha256(
            payload.encode()).hexdigest()[:12]

    def load_models(self):
        with self.lock:
            # Parent sentence models remain independently lazy.  The whole-
            # sentence generator is loaded only by load_word_model or when the
            # post-ranking extension actually runs.
            was_loaded = self.models_loaded
            super().load_models()
            if not was_loaded:
                self._mixed_version = self.version
            self.model_artifacts = {
                **getattr(self, "model_artifacts", {}),
                **self.neural_word_artifacts,
            }
            self._stamp_word_version()

    def load_word_model(self):
        """Load and return the immutable balanced sentence transducer."""
        with self.lock:
            if self._word_model_loaded:
                return self.word_model
            current = self._artifact_identity()
            if current != self.neural_word_artifacts:
                raise ValueError("neural-word checkpoint or source identity changed")
            checkpoint_sha = current.get(str(self.word_checkpoint.relative_to(ROOT))) \
                if self.word_checkpoint.is_relative_to(ROOT) else None
            if (self.word_checkpoint == DEFAULT_WORD_CHECKPOINT
                    and checkpoint_sha != EXPECTED_BALANCED_CHECKPOINT_SHA256):
                raise ValueError("balanced neural-word checkpoint identity differs")
            from hanglish.models.sentence_transducer import SentenceTransducer

            self.word_model = SentenceTransducer.load(self.word_checkpoint)
            self._word_model_loaded = True
            return self.word_model

    @staticmethod
    def _generation_reason(text):
        if not isinstance(text, str) or len(text.split()) < 2:
            return "unsupported_source"
        if HANGUL.search(text):
            return "unsupported_source"
        if len(text.encode("utf-8")) + 2 > 512:
            return "source_token_limit"
        return None

    def _extend_sentence_paths(self, text, context, paths):
        paths = super()._extend_sentence_paths(text, context, paths)
        reason = self._generation_reason(text)
        if reason is not None:
            self.neural_word_prediction = {
                "text": None, "reason": reason, "generated_tokens": 0}
            return paths

        started = time.perf_counter()
        self.neural_word_counts["generation_requests"] = 1
        with self.lock:
            predictions = self.load_word_model().generate([text])
        self.neural_word_ms = (time.perf_counter() - started) * 1000
        if (not isinstance(predictions, list) or len(predictions) != 1
                or not isinstance(predictions[0], dict)):
            raise RuntimeError("neural-word model returned an invalid prediction batch")
        prediction = predictions[0]
        if (set(prediction) != {"text", "reason", "generated_tokens"}
                or prediction["reason"] not in ("complete", "token_limit", "invalid_encoding")
                or not isinstance(prediction["generated_tokens"], int)
                or isinstance(prediction["generated_tokens"], bool)
                or not 0 <= prediction["generated_tokens"] <= 512
                or (prediction["reason"] == "complete") != isinstance(prediction["text"], str)):
            raise RuntimeError("neural-word model returned an invalid prediction")
        self.neural_word_prediction = dict(prediction)
        if prediction["reason"] != "complete":
            return paths
        self.neural_word_counts["completed_generations"] = 1

        extras = neural_word_proposals.sentence_extras(
            text, prediction["text"], paths, limit=96)
        self.neural_word_counts["generated_proposals"] = len(extras)
        accepted = [path for path in extras if self.sentence_ranker.can_score(path[0])]
        self.neural_word_counts["accepted_proposals"] = len(accepted)
        if not accepted:
            return paths
        scores = self.sentence_ranker.score_pairs(
            [context] * len(accepted), [path[0] for path in accepted])
        if (len(scores) != len(accepted)
                or any(isinstance(score, bool)
                       or not isinstance(score, (int, float))
                       or not math.isfinite(score) for score in scores)):
            raise RuntimeError("neural-word ranker returned invalid scores")
        existing = {path[0] for path in paths}
        if len(existing) != len(paths) or any(path[0] in existing for path in accepted):
            raise RuntimeError("neural-word pool contains duplicate paths")
        additional = [
            (path[0], float(score), path[2])
            for path, score in zip(accepted, scores)
        ]
        return sorted([*paths, *additional], key=lambda path: (-path[1], path[0]))

    def convert(self, text, context="", engine="neural", mode="batch", raw=False):
        with self.lock:
            self.neural_word_counts = self._empty_counts()
            self.neural_word_prediction = None
            self.neural_word_ms = 0.0
            response = super().convert(text, context, engine, mode, raw)
            response["neural_word"] = {
                "experimental": True,
                **self.neural_word_counts,
                "prediction": (dict(self.neural_word_prediction)
                               if self.neural_word_prediction is not None else None),
                "generation_ms": round(self.neural_word_ms, 3),
                "parent_version": self.word_parent_version,
                "artifacts": dict(self.neural_word_artifacts),
            }
            return response


__all__ = ["DEFAULT_WORD_CHECKPOINT", "NeuralWordConverter"]
