"""Experimental runtime for source-aligned words from SentenceCTC."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import time

from hanglish.core import ctc_word_proposals
from hanglish.core.mixed_span_converter import MixedSpanConverter
from hanglish.core.neural_converter import ROOT, file_sha256
from hanglish.models import sentence_ctc
from hanglish.models import sentence_transducer


DEFAULT_WORD_CHECKPOINT = ROOT / ".omx/artifacts/sentence-ctc-pilot-v1/model.pt"
EXPECTED_CTC_CHECKPOINT_SHA256 = (
    "832e7171a73daad4a02864668447725e7ae790e50ab7458f3a409701f0878663")
EXPECTED_CTC_METADATA_SHA256 = (
    "a5c8b896518209ea8f7de24c3a4d273539378a5ca3a2476cb54c9df527a2209b")
HANGUL = re.compile(r"[가-힣]")
REASONS = {"complete", "empty_output", "invalid_utf8", "invalid_jamo", "no_valid_path"}


class CTCWordConverter(MixedSpanConverter):
    """Add bounded, SentenceRanker-scored paths from one complete CTC output."""

    def __init__(self, *, word_checkpoint=DEFAULT_WORD_CHECKPOINT,
                 word_model=None, **kwargs):
        super().__init__(**kwargs)
        self.word_checkpoint = Path(word_checkpoint)
        self.word_model = word_model
        self._word_model_loaded = word_model is not None
        self.ctc_word_artifacts = self._artifact_identity()
        self._mixed_version = self.version
        self.word_parent_version = self._mixed_version
        self._stamp_word_version()
        self.ctc_word_counts = self._empty_counts()
        self.ctc_word_prediction = None
        self.ctc_word_ms = 0.0

    @staticmethod
    def _empty_counts():
        return {
            "generation_requests": 0,
            "completed_generations": 0,
            "generated_proposals": 0,
            "accepted_proposals": 0,
        }

    def _artifact_identity(self):
        artifacts = {
            "hanglish/core/ctc_word_converter.py": file_sha256(Path(__file__)),
            "hanglish/core/ctc_word_proposals.py": file_sha256(
                Path(ctc_word_proposals.__file__)),
            "hanglish/models/sentence_ctc.py": file_sha256(Path(sentence_ctc.__file__)),
            "hanglish/models/sentence_transducer.py": file_sha256(
                Path(sentence_transducer.__file__)),
        }
        for path in (self.word_checkpoint, self.word_checkpoint.with_suffix(".json")):
            if not path.is_file():
                continue
            try:
                name = str(path.relative_to(ROOT))
            except ValueError:
                name = "external/" + path.name
            artifacts[name] = file_sha256(path)
        return artifacts

    def _stamp_word_version(self):
        self.word_parent_version = self._mixed_version
        payload = json.dumps({"parent": self.word_parent_version,
                              "artifacts": self.ctc_word_artifacts}, sort_keys=True)
        self.version = "experimental-ctc-word-1-" + hashlib.sha256(
            payload.encode()).hexdigest()[:12]

    def load_models(self):
        with self.lock:
            was_loaded = self.models_loaded
            super().load_models()
            if not was_loaded:
                self._mixed_version = self.version
            self.model_artifacts = {
                **getattr(self, "model_artifacts", {}),
                **self.ctc_word_artifacts,
            }
            self._stamp_word_version()

    def load_word_model(self):
        """Load and return the immutable SentenceCTC checkpoint."""
        with self.lock:
            if self._word_model_loaded:
                return self.word_model
            current = self._artifact_identity()
            if current != self.ctc_word_artifacts:
                raise ValueError("CTC-word checkpoint or source identity changed")
            if self.word_checkpoint == DEFAULT_WORD_CHECKPOINT:
                checkpoint = str(self.word_checkpoint.relative_to(ROOT))
                metadata = str(self.word_checkpoint.with_suffix(".json").relative_to(ROOT))
                if (current.get(checkpoint) != EXPECTED_CTC_CHECKPOINT_SHA256
                        or current.get(metadata) != EXPECTED_CTC_METADATA_SHA256):
                    raise ValueError("SentenceCTC checkpoint identity differs")
            from hanglish.models.sentence_ctc import SentenceCTC

            self.word_model = SentenceCTC.load(self.word_checkpoint)
            self._word_model_loaded = True
            return self.word_model

    @staticmethod
    def _generation_reason(text):
        if not isinstance(text, str) or len(text.split()) < 2 or HANGUL.search(text):
            return "unsupported_source"
        if len(text.encode("utf-8")) + 2 > 512:
            return "source_token_limit"
        return None

    @staticmethod
    def _validate_prediction(prediction):
        required = {"text", "reason", "generated_tokens"}
        if (not isinstance(prediction, dict) or not required <= set(prediction)
                or prediction["reason"] not in REASONS
                or isinstance(prediction["generated_tokens"], bool)
                or not isinstance(prediction["generated_tokens"], int)
                or not 0 <= prediction["generated_tokens"] <= 511
                or ((prediction["reason"] == "complete")
                    != isinstance(prediction["text"], str))
                or (prediction["reason"] != "complete" and prediction["text"] is not None)):
            raise RuntimeError("CTC-word model returned an invalid prediction")
        labels = prediction.get("labels")
        if labels is not None and (not isinstance(labels, list)
                                   or len(labels) != prediction["generated_tokens"]
                                   or any(isinstance(label, bool) or not isinstance(label, int)
                                          or label <= 0 for label in labels)):
            raise RuntimeError("CTC-word model returned invalid labels")
        frames = prediction.get("frames")
        if frames is not None and (isinstance(frames, bool)
                                   or not isinstance(frames, int) or frames < 0):
            raise RuntimeError("CTC-word model returned invalid frame metadata")
        if ("decode_mode" in prediction
                and prediction["decode_mode"] != "ctc_prefix_beam"):
            raise RuntimeError("CTC-word model returned an invalid decoder mode")
        score = prediction.get("log_probability")
        if score is not None and (isinstance(score, bool)
                                  or not isinstance(score, (int, float))
                                  or not math.isfinite(score)):
            raise RuntimeError("CTC-word model returned an invalid probability")
        for key in ("beam", "greedy"):
            if key in prediction and not isinstance(prediction[key], dict):
                raise RuntimeError("CTC-word model returned invalid decoder metadata")

    def _extend_sentence_paths(self, text, context, paths):
        paths = super()._extend_sentence_paths(text, context, paths)
        reason = self._generation_reason(text)
        if reason is not None:
            self.ctc_word_prediction = {
                "text": None, "reason": reason, "generated_tokens": 0}
            return paths

        started = time.perf_counter()
        self.ctc_word_counts["generation_requests"] = 1
        with self.lock:
            predictions = self.load_word_model().generate([text])
        self.ctc_word_ms = (time.perf_counter() - started) * 1000
        if (not isinstance(predictions, list) or len(predictions) != 1
                or not isinstance(predictions[0], dict)):
            raise RuntimeError("CTC-word model returned an invalid prediction batch")
        prediction = predictions[0]
        self._validate_prediction(prediction)
        self.ctc_word_prediction = dict(prediction)
        if prediction["reason"] != "complete":
            return paths
        self.ctc_word_counts["completed_generations"] = 1

        extras = ctc_word_proposals.sentence_extras(
            text, prediction["text"], paths, limit=96)
        self.ctc_word_counts["generated_proposals"] = len(extras)
        accepted = [path for path in extras if self.sentence_ranker.can_score(path[0])]
        self.ctc_word_counts["accepted_proposals"] = len(accepted)
        if not accepted:
            return paths
        scores = self.sentence_ranker.score_pairs(
            [context] * len(accepted), [path[0] for path in accepted])
        if (len(scores) != len(accepted)
                or any(isinstance(score, bool)
                       or not isinstance(score, (int, float))
                       or not math.isfinite(score) for score in scores)):
            raise RuntimeError("CTC-word ranker returned invalid scores")
        existing = {path[0] for path in paths}
        if len(existing) != len(paths) or any(path[0] in existing for path in accepted):
            raise RuntimeError("CTC-word pool contains duplicate paths")
        additional = [(path[0], float(score), path[2])
                      for path, score in zip(accepted, scores)]
        return sorted([*paths, *additional], key=lambda path: (-path[1], path[0]))

    def convert(self, text, context="", engine="neural", mode="batch", raw=False):
        with self.lock:
            self.ctc_word_counts = self._empty_counts()
            self.ctc_word_prediction = None
            self.ctc_word_ms = 0.0
            response = super().convert(text, context, engine, mode, raw)
            response["ctc_word"] = {
                "experimental": True, **self.ctc_word_counts,
                "prediction": (dict(self.ctc_word_prediction)
                               if self.ctc_word_prediction is not None else None),
                "generation_ms": round(self.ctc_word_ms, 3),
                "parent_version": self.word_parent_version,
                "artifacts": dict(self.ctc_word_artifacts),
            }
            return response


__all__ = ["CTCWordConverter", "DEFAULT_WORD_CHECKPOINT"]
