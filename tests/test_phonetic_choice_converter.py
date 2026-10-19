import json
import math
import threading

import pytest

from hanglish.core.phonetic_word_union_converter import PhoneticWordUnionConverter
from hanglish.core.phonetic_choice_converter import (
    DEFAULT_CALIBRATION_CHECKPOINT,
    DEFAULT_CALIBRATION_SHA256,
    PhoneticChoiceConverter,
)
from hanglish.core.neural_converter import file_sha256
from hanglish.models.phonetic_choice_features import FEATURE_NAMES


def segment(source, start, end, text, kind="test"):
    return {"input_start": start, "input_end": end,
            "source": source[start:end], "text": text, "kind": kind}


class Scores:
    def __init__(self, means=None):
        self.means = means or {}
        self.calls = []

    def __call__(self, model, pairs, batch_size):
        self.calls.append((model, list(pairs), batch_size))
        result = []
        for source, target in pairs:
            tokens = 3 * len(target) + 1
            mean = self.means.get((source, target), -0.5)
            result.append({"source": source, "target": target,
                           "log_probability": mean * tokens,
                           "token_count": tokens,
                           "mean_log_probability": mean})
        return result


def converter(weights, scorer=None):
    value = object.__new__(PhoneticChoiceConverter)
    value.lock = threading.RLock()
    value.calibration_weights = tuple(weights)
    value.calibration_source = "override"
    value._likelihood_scorer = scorer or Scores()
    value.phonetic_model = object()
    value.choice_base_paths = []
    value.choice_trace = value._empty_choice_trace()
    value.choice_ms = 0.0
    value.load_phonetic_model = lambda: value.phonetic_model
    return value


def test_zero_weights_compute_likelihood_but_preserve_parent_objects_scores_and_ties():
    source = "gim han"
    paths = [
        ("김 한", 2.0, [segment(source, 0, 3, "김"), segment(source, 3, 4, " "),
                         segment(source, 4, 7, "한")]),
        ("김한", 2.0, [segment(source, 0, 3, "김"), segment(source, 3, 4, ""),
                        segment(source, 4, 7, "한")]),
    ]
    scorer = Scores()
    value = converter([0, 0, 0, 0], scorer)

    result = value.calibrate_paths(source, paths)

    assert result is paths
    assert [(row[0], row[1], row[2]) for row in result] == paths
    assert len(scorer.calls) == 1
    assert scorer.calls[0][0] is value.phonetic_model and scorer.calls[0][2] == 64
    assert set(scorer.calls[0][1]) == {("gim", "김"), ("han", "한")}
    assert value.choice_trace["zero_weight_score_parity"] is True
    assert value.choice_trace["zero_weight_order_parity"] is True
    assert value.choice_trace["effective_weights"] == [0.0] * 4


def test_supported_scores_join_exact_source_spans_and_use_source_length_weights():
    source = "gim han"
    paths = [("김 한", 3.0, [segment(source, 0, 3, "김"),
                              segment(source, 3, 4, " "),
                              segment(source, 4, 7, "한")])]
    scorer = Scores({("gim", "김"): -1.0, ("han", "한"): -2.0})
    value = converter([1, 0, 0, 0], scorer)

    result = value.calibrate_paths(source, paths)

    expected = 3 / 7 * -1.0 + 3 / 7 * -2.0
    assert result[0][1] == pytest.approx(3.0 + expected)
    assert value.choice_trace["rows"][0]["features"] == pytest.approx(
        [expected, 6 / 7, 0, 0])
    assert {(row["source"], row["target"]) for row in value.choice_trace["score_records"]} == {
        ("gim", "김"), ("han", "한")}
    assert value.choice_trace["supported_pair_count"] == 2
    assert value.choice_trace["zero_weight_parity"] is True


def test_copy_never_requests_likelihood_and_is_reference_category():
    source = "test"
    paths = [(source, 1.25, [segment(source, 0, 4, source)])]
    scorer = Scores()
    value = converter([1, 1, 1, 1], scorer)

    result = value.calibrate_paths(source, paths)

    assert scorer.calls == []
    assert result == paths
    assert value.choice_trace["rows"][0]["features"] == [0.0] * 4
    assert value.choice_trace["edge_status_counts"] == {"copy": 1}


def test_unsupported_conversion_is_retained_and_counted_without_model_call():
    source = "Test!"
    paths = [("테스트", 4.0, [segment(source, 0, 5, "테스트")])]
    scorer = Scores()
    value = converter([0, 0, 2, 0], scorer)

    result = value.calibrate_paths(source, paths)

    assert result[0][0] == "테스트" and result[0][2] is paths[0][2]
    assert result[0][1] == 6.0
    assert scorer.calls == []
    assert value.choice_trace["rows"][0]["features"] == [0.0, 0.0, 1.0, 0.0]
    assert value.choice_trace["edge_status_counts"] == {"unsupported_ascii": 1}


def test_forward_mismatch_is_feature_not_rejection():
    source = "gim"
    paths = [("한", 1.0, [segment(source, 0, 3, "한")])]
    scorer = Scores({("gim", "한"): -0.25})
    value = converter([0, 0, 0, 2], scorer)

    result = value.calibrate_paths(source, paths)

    assert result[0][0] == "한" and result[0][1] == 3.0
    assert value.choice_trace["rows"][0]["features"] == [-0.25, 1.0, 0.0, 1.0]
    assert value.choice_trace["rows"][0]["edges"][0]["romanize_match"] is False


def test_unique_supported_pairs_are_scored_once_per_request():
    source = "gim!"
    paths = [
        ("김!", 2.0, [segment(source, 0, 3, "김"), segment(source, 3, 4, "!")]),
        ("김?", 1.0, [segment(source, 0, 3, "김"), segment(source, 3, 4, "?")]),
    ]
    scorer = Scores()
    value = converter([0, 0, 0, 0], scorer)

    assert value.calibrate_paths(source, paths) is paths
    assert scorer.calls[0][1] == [("gim", "김")]
    assert value.choice_trace["supported_pair_count"] == 1


def test_parent_extension_runs_once_and_base_paths_are_exposed(monkeypatch):
    source = "gim"
    parent = [("김", 1.0, [segment(source, 0, 3, "김")])]
    calls = []
    monkeypatch.setattr(
        PhoneticWordUnionConverter, "_extend_sentence_paths",
        lambda self, text, context, paths: calls.append((text, context, paths)) or parent)
    value = converter([0, 0, 0, 0], Scores())

    result = value._extend_sentence_paths(source, "ctx", [("old", 0.0, [])])

    assert len(calls) == 1
    assert value.choice_base_paths == parent
    assert value.choice_base_paths[0][2] is parent[0][2]
    assert result is parent


def test_first_load_adopts_loaded_parent_version_and_repeated_load_is_idempotent(monkeypatch):
    value = object.__new__(PhoneticChoiceConverter)
    value.choice_parent_version = "constructor-union"
    value._choice_parent_loaded = False
    value.calibration_weights = (0.0, 0.0, 0.0, 0.0)
    value.calibration_source = "override"
    value.calibration_artifacts = {"artifact": "digest"}
    value.choice_version = value._choice_version()
    value.version = value.choice_version
    value.models_loaded = False
    observed = []

    def parent_load(self):
        observed.append(self.version)
        self.version = "loaded-union"
        self.models_loaded = True

    monkeypatch.setattr(PhoneticWordUnionConverter, "load_models", parent_load)
    monkeypatch.setattr(value, "_calibration_source_artifacts",
                        lambda **kwargs: {"artifact": "digest"})

    value.load_models()
    first_choice_version = value.choice_version
    value.load_models()

    assert observed == ["constructor-union", "loaded-union"]
    assert value.choice_parent_version == "loaded-union"
    assert value.choice_version == first_choice_version
    assert value.version == first_choice_version
    assert value.model_artifacts == {"artifact": "digest"}


def test_convert_resets_and_publishes_choice_diagnostics_without_hiding_parent(monkeypatch):
    value = converter([0, 0, 0, 0], Scores())
    value.choice_parent_version = "union-v1"
    value.choice_version = "choice-v1"
    value.calibration_artifacts = {"weights.json": "abc"}
    value.choice_trace["public"] = ["stale"]

    def parent_convert(self, text, context, engine, mode, raw):
        paths = [("김", 2.0, [segment(text, 0, len(text), "김")])]
        calibrated = self._extend_sentence_paths(text, context, paths)
        return {"result": calibrated[0][0],
                "candidates": [{"text": row[0]} for row in calibrated],
                "phonetic_word_union": {"kept": True}}

    monkeypatch.setattr(PhoneticWordUnionConverter, "convert", parent_convert)
    monkeypatch.setattr(PhoneticWordUnionConverter, "_extend_sentence_paths",
                        lambda self, text, context, paths: paths)

    response = value.convert("gim")

    assert response["phonetic_word_union"] == {"kept": True}
    assert response["phonetic_choice"]["public"] == ["김"]
    assert response["phonetic_choice"]["effective_weights"] == [0.0] * 4
    assert response["phonetic_choice"]["parent_version"] == "union-v1"
    assert response["phonetic_choice"]["version"] == "choice-v1"
    assert response["phonetic_choice"]["artifacts"] == {"weights.json": "abc"}


def test_unranked_convert_still_reports_effective_weights_without_paths(monkeypatch):
    weights = (0.1, 0.2, -0.3, 0.4)
    value = converter(weights, Scores())
    value.choice_parent_version = "union-v1"
    value.choice_version = "choice-v1"
    value.calibration_artifacts = {}
    monkeypatch.setattr(
        PhoneticWordUnionConverter, "convert",
        lambda self, *args: {"result": "raw", "candidates": [{"text": "raw"}]})

    response = value.convert("raw", raw=True)

    assert value.choice_base_paths == []
    assert response["phonetic_choice"]["rows"] == []
    assert response["phonetic_choice"]["score_records"] == []
    assert response["phonetic_choice"]["effective_weights"] == list(weights)
    assert response["phonetic_choice"]["calibration_source"] == "override"
    assert response["phonetic_choice"]["public"] == ["raw"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 1.0])
def test_missing_or_nonfinite_likelihood_is_rejected(bad):
    source = "gim"
    paths = [("김", 1.0, [segment(source, 0, 3, "김")])]

    def scorer(model, pairs, batch_size):
        if math.isfinite(bad):
            return []
        return [{"source": "gim", "target": "김", "log_probability": bad,
                 "token_count": 4, "mean_log_probability": bad}]

    value = converter([0, 0, 0, 0], scorer)
    with pytest.raises(ValueError, match="result (count|value)"):
        value.calibrate_paths(source, paths)


def test_source_offsets_and_output_concatenation_are_strict():
    value = converter([0, 0, 0, 0])
    with pytest.raises(RuntimeError, match="offsets"):
        value.calibrate_paths("gim", [("김", 1.0, [
            {"input_start": 1, "input_end": 3, "source": "im", "text": "김", "kind": "x"}
        ])])


def test_default_checkpoint_schema_feature_names_and_hash_are_frozen():
    value = object.__new__(PhoneticChoiceConverter)
    value.calibration_checkpoint = DEFAULT_CALIBRATION_CHECKPOINT
    weights, artifacts = value._load_calibration_checkpoint()
    payload = json.loads(DEFAULT_CALIBRATION_CHECKPOINT.read_text())

    assert file_sha256(DEFAULT_CALIBRATION_CHECKPOINT) == DEFAULT_CALIBRATION_SHA256
    assert payload["feature_names"] == list(FEATURE_NAMES)
    assert weights == tuple(payload["weights"])
    assert artifacts[".omx/artifacts/phonetic-choice-calibration-training-v1/weights.json"] \
        == DEFAULT_CALIBRATION_SHA256


def test_bad_weight_override_is_rejected_without_loading_checkpoint():
    with pytest.raises(ValueError, match="four finite"):
        # Validation is a constructor boundary; invoke the same private loader
        # path without constructing any parent neural model.
        from hanglish.core.phonetic_choice_converter import _validate_weights
        _validate_weights([0.0, 0.0, float("nan"), 0.0])
