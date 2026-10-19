"""Experimental phonetic-likelihood calibration over the complete union pool."""
from __future__ import annotations

from copy import deepcopy
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

from hanglish.core.neural_converter import ROOT, file_sha256
from hanglish.core.phonetic_word_union_converter import PhoneticWordUnionConverter
from hanglish.core.phonology import romanize_word
from hanglish.models.phonetic_choice_features import (
    FEATURE_NAMES,
    _expected_status,
    _path_features,
    _validate_edge,
)


DEFAULT_CALIBRATION_CHECKPOINT = (
    ROOT / ".omx/artifacts/phonetic-choice-calibration-training-v1/weights.json"
)
DEFAULT_CALIBRATION_SHA256 = (
    "930e3ef942061e6d459d7527c0e073fc6ad5626d59f5170e9b11df8499b39f4b"
)
FEATURE_MANIFEST_SHA256 = (
    "b884847fc4fc5f69d20acb95628135f82ba5fa6596479e540600f29e59517a84"
)
_FEATURE_MODULE = ROOT / "hanglish/models/phonetic_choice_features.py"
_LIKELIHOOD_MODULE = ROOT / "hanglish/models/phonetic_likelihood.py"
_FIT_SCHEMA = "phonetic-likelihood-fit-v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite_number(value: object) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value))


def _pair_id(source: str, target: str) -> str:
    payload = json.dumps([_FIT_SCHEMA, source, target], ensure_ascii=False,
                         separators=(",", ":")).encode()
    return "phonetic-pair-" + hashlib.sha256(payload).hexdigest()


def _validate_weights(values) -> tuple[float, ...]:
    _require(isinstance(values, (list, tuple)) and len(values) == len(FEATURE_NAMES)
             and all(_finite_number(value) for value in values),
             "phonetic calibration requires four finite weights")
    return tuple(float(value) for value in values)


class PhoneticChoiceConverter(PhoneticWordUnionConverter):
    """Add a frozen four-feature residual to every union candidate score.

    Each request recomputes its supported source/target likelihoods.  Runtime
    paths have one alignment each; no alternative alignment or label is
    synthesized here.
    """

    def __init__(self, *, calibration_checkpoint=DEFAULT_CALIBRATION_CHECKPOINT,
                 calibration_weights=None, likelihood_scorer=None, **kwargs):
        super().__init__(**kwargs)
        self.calibration_checkpoint = Path(calibration_checkpoint)
        self._likelihood_scorer = likelihood_scorer
        if calibration_weights is None:
            weights, artifacts = self._load_calibration_checkpoint()
            self.calibration_source = "checkpoint"
        else:
            weights = _validate_weights(calibration_weights)
            artifacts = self._calibration_source_artifacts(include_checkpoint=False)
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

    @staticmethod
    def _empty_choice_trace():
        return {
            "rows": [],
            "score_records": [],
            "edge_status_counts": {},
            "supported_pair_count": 0,
            "effective_weights": [],
            "calibration_source": None,
            "zero_weight_parity": False,
            "zero_weight_score_parity": False,
            "zero_weight_order_parity": False,
            "likelihood_ms": 0.0,
            "calibration_ms": 0.0,
            "public": [],
        }

    def _calibration_source_artifacts(self, *, include_checkpoint: bool) -> dict[str, str]:
        paths = [Path(__file__), _FEATURE_MODULE, _LIKELIHOOD_MODULE]
        if include_checkpoint:
            paths.append(self.calibration_checkpoint)
        artifacts = {}
        for path in paths:
            _require(path.is_file(), f"phonetic calibration artifact missing: {path.name}")
            try:
                name = str(path.relative_to(ROOT))
            except ValueError:
                name = "external/" + path.name
            artifacts[name] = file_sha256(path)
        return artifacts

    def _load_calibration_checkpoint(self) -> tuple[tuple[float, ...], dict[str, str]]:
        before = self._calibration_source_artifacts(include_checkpoint=True)
        _require(file_sha256(self.calibration_checkpoint) == DEFAULT_CALIBRATION_SHA256,
                 "phonetic calibration checkpoint identity differs")
        value = json.loads(self.calibration_checkpoint.read_text())
        _require(isinstance(value, dict)
                 and value.get("schema") == "phonetic-choice-calibration-v1"
                 and value.get("feature_names") == list(FEATURE_NAMES)
                 and value.get("feature_manifest_sha256") == FEATURE_MANIFEST_SHA256
                 and value.get("intercept") == 0.0
                 and value.get("steps") == 200,
                 "phonetic calibration checkpoint contract differs")
        weights = _validate_weights(value.get("weights"))
        after = self._calibration_source_artifacts(include_checkpoint=True)
        _require(before == after, "phonetic calibration artifacts changed while loading")
        return weights, before

    def _choice_version(self) -> str:
        payload = json.dumps({
            "parent": self.choice_parent_version,
            "artifacts": self.calibration_artifacts,
            "weights": self.calibration_weights,
            "source": self.calibration_source,
        }, sort_keys=True, separators=(",", ":"))
        return "experimental-phonetic-choice-1-" + hashlib.sha256(
            payload.encode()).hexdigest()[:12]

    def load_models(self):
        """Let the union load normally without replacing this adapter's stamp."""
        # The parent uses ``self.version`` while constructing its own stamp.
        # Present the frozen union version during that call so it cannot absorb
        # this child adapter's version and create a circular parent identity.
        choice_version = self.choice_version
        self.version = self.choice_parent_version
        try:
            super().load_models()
            loaded_parent_version = self.version
            if self._choice_parent_loaded:
                _require(loaded_parent_version == self.choice_parent_version,
                         "loaded phonetic union version changed")
            else:
                self.choice_parent_version = loaded_parent_version
                self._choice_parent_loaded = True
                self.choice_version = self._choice_version()
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

    @staticmethod
    def _runtime_edge(source: str, segment: dict) -> dict:
        start, end = segment.get("input_start"), segment.get("input_end")
        source_text, target = segment.get("source"), segment.get("text")
        if (type(start) is not int or type(end) is not int
                or not 0 <= start < end <= len(source)
                or source_text != source[start:end]
                or not isinstance(target, str)):
            raise ValueError("phonetic calibration segment identity differs")
        status = _expected_status(source_text, target)
        edge = {"input_start": start, "input_end": end, "source": source_text,
                "target": target, "status": status}
        if status == "supported":
            edge["pair_id"] = _pair_id(source_text, target)
            edge["romanize_match"] = romanize_word(target) == source_text
        _validate_edge(edge, source)
        return edge

    def _score_supported_pairs(self, requests: list[tuple[str, str]]) -> list[dict]:
        if not requests:
            return []
        model = self.load_phonetic_model()
        scorer = self._likelihood_scorer
        if scorer is None:
            from hanglish.models.phonetic_likelihood import score_pairs
            scorer = score_pairs
        with self.lock:
            values = scorer(model, requests, batch_size=64)
        _require(isinstance(values, (list, tuple)) and len(values) == len(requests),
                 "phonetic likelihood result count differs")
        result = []
        for (source, target), value in zip(requests, values):
            _require(isinstance(value, dict) and set(value) == {
                "source", "target", "log_probability", "token_count",
                "mean_log_probability",
            }, "phonetic likelihood result is malformed")
            record = {"pair_id": _pair_id(source, target), **value}
            expected_keys = {"pair_id", "source", "target", "log_probability",
                             "token_count", "mean_log_probability"}
            _require(set(record) == expected_keys
                     and record["source"] == source and record["target"] == target,
                     "phonetic likelihood result identity differs")
            total = record["log_probability"]
            mean = record["mean_log_probability"]
            tokens = record["token_count"]
            _require(_finite_number(total) and total <= 0
                     and _finite_number(mean) and mean <= 0
                     and type(tokens) is int and tokens == 3 * len(target) + 1
                     and mean == total / tokens,
                     "phonetic likelihood result value differs")
            result.append(record)
        return result

    def calibrate_paths(self, source: str, paths):
        """Calibrate a complete request pool while preserving every path."""
        _require(isinstance(source, str) and source, "phonetic calibration source is empty")
        _require(isinstance(paths, list) and paths, "phonetic calibration paths are empty")
        started = time.perf_counter()
        path_edges = []
        requests = {}
        status_counts = Counter()
        texts = set()
        for path in paths:
            _require(isinstance(path, tuple) and len(path) == 3,
                     "phonetic calibration path is malformed")
            text, score, segments = path
            _require(isinstance(text, str) and text and text not in texts
                     and _finite_number(score) and isinstance(segments, list),
                     "phonetic calibration candidate domain differs")
            texts.add(text)
            self._validate_path(source, text, segments)
            edges = [self._runtime_edge(source, segment) for segment in segments]
            path_edges.append(edges)
            for edge in edges:
                status_counts[edge["status"]] += 1
                if edge["status"] == "supported":
                    key = (edge["source"], edge["target"])
                    prior = requests.setdefault(edge["pair_id"], key)
                    _require(prior == key, "phonetic pair identity collision")

        ordered_ids = sorted(requests)
        ordered_pairs = [requests[identifier] for identifier in ordered_ids]
        likelihood_started = time.perf_counter()
        records = self._score_supported_pairs(ordered_pairs)
        likelihood_ms = (time.perf_counter() - likelihood_started) * 1000
        by_pair_id = {record["pair_id"]: record for record in records}
        _require([record["pair_id"] for record in records] == ordered_ids,
                 "phonetic likelihood result ordering differs")

        rows = []
        calibrated = []
        for index, ((text, base_score, segments), edges) in enumerate(zip(paths, path_edges)):
            path = {"path_id": f"runtime-path-{index}",
                    "edge_indices": list(range(len(edges)))}
            features = _path_features(source, text, path, edges, by_pair_id)
            residual = sum(weight * feature for weight, feature
                           in zip(self.calibration_weights, features))
            adjusted = float(base_score) + residual
            _require(math.isfinite(adjusted), "phonetic calibrated score is nonfinite")
            calibrated.append((text, adjusted, segments))
            rows.append({
                "text": text,
                "base_score": float(base_score),
                "features": list(features),
                "residual": residual,
                "adjusted_score": adjusted,
                "segments": deepcopy(segments),
                "edges": deepcopy(edges),
            })
        calibrated.sort(key=lambda path: (-path[1], path[0]))
        zero = all(weight == 0.0 for weight in self.calibration_weights)
        score_parity = all(row["adjusted_score"] == row["base_score"] for row in rows)
        order_parity = [path[0] for path in calibrated] == [path[0] for path in paths]
        shadow_zero = [
            (path[0], float(path[1]) + sum(0.0 * value for value in row["features"]),
             path[2])
            for path, row in zip(paths, rows)
        ]
        shadow_score_parity = all(shadow[1] == base[1]
                                  for shadow, base in zip(shadow_zero, paths))
        shadow_zero.sort(key=lambda path: (-path[1], path[0]))
        shadow_zero_parity = (shadow_score_parity
                              and [path[0] for path in shadow_zero]
                              == [path[0] for path in paths])
        if zero:
            _require(score_parity and order_parity,
                     "zero-weight phonetic calibration changed the parent pool")
            result = paths
        else:
            result = calibrated
        total_ms = (time.perf_counter() - started) * 1000
        calibration_ms = max(0.0, total_ms - likelihood_ms)
        self.choice_ms = total_ms
        self.choice_trace = {
            "rows": rows,
            "score_records": deepcopy(records),
            "edge_status_counts": dict(sorted(status_counts.items())),
            "supported_pair_count": len(records),
            "effective_weights": list(self.calibration_weights),
            "calibration_source": self.calibration_source,
            "zero_weight_parity": shadow_zero_parity,
            "zero_weight_score_parity": score_parity if zero else False,
            "zero_weight_order_parity": order_parity if zero else False,
            "likelihood_ms": likelihood_ms,
            "calibration_ms": calibration_ms,
            "public": [],
        }
        return result

    def _extend_sentence_paths(self, text, context, paths):
        parent = super()._extend_sentence_paths(text, context, paths)
        self.choice_base_paths = list(parent)
        return self.calibrate_paths(text, parent)

    def convert(self, text, context="", engine="neural", mode="batch", raw=False):
        with self.lock:
            self.choice_base_paths = []
            self.choice_trace = self._empty_choice_trace()
            self.choice_trace["effective_weights"] = list(self.calibration_weights)
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


__all__ = [
    "DEFAULT_CALIBRATION_CHECKPOINT",
    "DEFAULT_CALIBRATION_SHA256",
    "PhoneticChoiceConverter",
]
