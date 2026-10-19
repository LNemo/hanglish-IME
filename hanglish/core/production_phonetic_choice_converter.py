"""Experimental phonetic calibration over the production CTC-union pool."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from hanglish.core.ctc_union_converter import CTCUnionConverter
from hanglish.core.neural_converter import ROOT, file_sha256
from hanglish.core.phonetic_choice_converter import (
    DEFAULT_CALIBRATION_CHECKPOINT,
    PhoneticChoiceConverter,
    _validate_weights,
)


_FEATURE_MODULE = ROOT / "hanglish/models/phonetic_choice_features.py"
_LIKELIHOOD_MODULE = ROOT / "hanglish/models/phonetic_likelihood.py"
_BATCHED_DECODER = ROOT / "hanglish/models/batched_phonetic_decode.py"
_CHOICE_MODULE = ROOT / "hanglish/core/phonetic_choice_converter.py"


def _artifact_name(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return "external/" + path.name


class ProductionPhoneticChoiceConverter(CTCUnionConverter):
    """Apply the frozen phonetic residual to production candidates only.

    The production CTC union remains the sole candidate generator.  The
    phonetic transducer is used only by the teacher-forced likelihood scorer;
    it never proposes an additional path in this adapter.
    """

    _empty_choice_trace = staticmethod(PhoneticChoiceConverter._empty_choice_trace)
    _runtime_edge = staticmethod(PhoneticChoiceConverter._runtime_edge)
    _score_supported_pairs = PhoneticChoiceConverter._score_supported_pairs
    calibrate_paths = PhoneticChoiceConverter.calibrate_paths

    def __init__(self, *, phonetic_checkpoint, phonetic_model=None,
                 calibration_checkpoint=DEFAULT_CALIBRATION_CHECKPOINT,
                 calibration_weights=None, likelihood_scorer=None, **kwargs):
        super().__init__(**kwargs)
        self.phonetic_checkpoint = Path(phonetic_checkpoint)
        self.phonetic_model = phonetic_model
        self._phonetic_model_loaded = phonetic_model is not None
        self.phonetic_likelihood_artifacts = self._phonetic_artifact_identity()
        self.calibration_checkpoint = Path(calibration_checkpoint)
        self._likelihood_scorer = likelihood_scorer
        if calibration_weights is None:
            weights, artifacts = self._load_calibration_checkpoint()
            self.calibration_source = "checkpoint"
        else:
            weights = _validate_weights(calibration_weights)
            artifacts = self._calibration_source_artifacts(
                include_checkpoint=False)
            self.calibration_source = "override"
        self.calibration_weights = weights
        self.calibration_artifacts = artifacts
        self.choice_parent_version = self.version
        self._choice_parent_loaded = bool(self.models_loaded)
        self.choice_version = self._choice_version()
        self.version = self.choice_version
        self.choice_base_paths = []
        self.choice_trace = self._empty_choice_trace()
        self.choice_ms = 0.0

    def _phonetic_artifact_identity(self) -> dict[str, str]:
        paths = [self.phonetic_checkpoint,
                 self.phonetic_checkpoint.with_suffix(".json")]
        return {
            _artifact_name(path): file_sha256(path)
            for path in paths if path.is_file()
        }

    def _calibration_source_artifacts(self, *,
                                      include_checkpoint: bool) -> dict[str, str]:
        paths = [Path(__file__), _CHOICE_MODULE, _FEATURE_MODULE,
                 _LIKELIHOOD_MODULE, _BATCHED_DECODER]
        if include_checkpoint:
            paths.append(self.calibration_checkpoint)
        artifacts = {}
        for path in paths:
            if not path.is_file():
                raise ValueError(
                    f"phonetic calibration artifact missing: {path.name}")
            artifacts[_artifact_name(path)] = file_sha256(path)
        artifacts.update(self.phonetic_likelihood_artifacts)
        return artifacts

    def _load_calibration_checkpoint(self):
        return PhoneticChoiceConverter._load_calibration_checkpoint(self)

    def _choice_version(self) -> str:
        payload = json.dumps({
            "parent": self.choice_parent_version,
            "artifacts": self.calibration_artifacts,
            "weights": self.calibration_weights,
            "source": self.calibration_source,
        }, sort_keys=True, separators=(",", ":"))
        return "experimental-production-phonetic-choice-1-" + hashlib.sha256(
            payload.encode()).hexdigest()[:12]

    def load_models(self):
        """Load the production parent while keeping its identity independent."""
        choice_version = self.choice_version
        self.version = self.choice_parent_version
        try:
            super().load_models()
            loaded_parent_version = self.version
            if self._choice_parent_loaded:
                if loaded_parent_version != self.choice_parent_version:
                    raise ValueError("loaded production parent version changed")
            else:
                self.choice_parent_version = loaded_parent_version
                self._choice_parent_loaded = True
                self.choice_version = self._choice_version()
            current_phonetic = self._phonetic_artifact_identity()
            if current_phonetic != self.phonetic_likelihood_artifacts:
                raise ValueError("phonetic checkpoint identity changed")
            current = self._calibration_source_artifacts(
                include_checkpoint=self.calibration_source == "checkpoint")
            if current != self.calibration_artifacts:
                raise ValueError("phonetic calibration artifact identity changed")
            self.model_artifacts = {
                **getattr(self, "model_artifacts", {}),
                **self.calibration_artifacts,
            }
            self.version = self.choice_version
        except BaseException:
            self.version = choice_version
            raise

    def load_phonetic_model(self):
        """Load the frozen likelihood model without generating candidates."""
        with self.lock:
            if self._phonetic_model_loaded:
                return self.phonetic_model
            if self._phonetic_artifact_identity() != self.phonetic_likelihood_artifacts:
                raise ValueError("phonetic checkpoint identity changed")
            if not self.phonetic_checkpoint.is_file():
                raise ValueError("phonetic checkpoint missing")
            from hanglish.models.batched_phonetic_decode import generate_batched
            from hanglish.models.phonetic_transducer import PhoneticTransducer

            class BatchedPhoneticTransducer(PhoneticTransducer):
                def generate(self, roman, beam_size=4):
                    return generate_batched(self, roman, beam_size)

            self.phonetic_model = BatchedPhoneticTransducer.load(
                self.phonetic_checkpoint)
            self._phonetic_model_loaded = True
            return self.phonetic_model

    @staticmethod
    def _validate_path(source: str, text: str, segments: list[dict]) -> None:
        cursor = 0
        for segment in segments:
            start, end = segment.get("input_start"), segment.get("input_end")
            if (type(start) is not int or type(end) is not int
                    or start != cursor or not start < end <= len(source)
                    or segment.get("source") != source[start:end]
                    or not isinstance(segment.get("text"), str)):
                raise RuntimeError(
                    "production phonetic path does not preserve source offsets")
            cursor = end
        if cursor != len(source) or text != "".join(
                row["text"] for row in segments):
            raise RuntimeError(
                "production phonetic path text differs from its segments")

    def _extend_sentence_paths(self, text, context, paths):
        parent = CTCUnionConverter._extend_sentence_paths(
            self, text, context, paths)
        self.choice_base_paths = list(parent)
        return self.calibrate_paths(text, parent)

    def convert(self, text, context="", engine="neural", mode="batch", raw=False):
        with self.lock:
            self.choice_base_paths = []
            self.choice_trace = self._empty_choice_trace()
            self.choice_trace["effective_weights"] = list(
                self.calibration_weights)
            self.choice_trace["calibration_source"] = self.calibration_source
            self.choice_ms = 0.0
            response = super().convert(text, context, engine, mode, raw)
            public = [row["text"] for row in response.get("candidates", [])]
            self.choice_trace["public"] = public
            response["phonetic_choice"] = {
                "experimental": True,
                "choice_ms": round(self.choice_ms, 3),
                "parent_version": self.choice_parent_version,
                "version": self.choice_version,
                "artifacts": dict(self.calibration_artifacts),
                **deepcopy(self.choice_trace),
            }
            return response


__all__ = ["ProductionPhoneticChoiceConverter"]
