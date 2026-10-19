import threading
from pathlib import Path

import pytest

from hanglish.core.ctc_union_converter import CTCUnionConverter
from hanglish.core.neural_converter import file_sha256
from hanglish.core.production_phonetic_choice_converter import (
    ProductionPhoneticChoiceConverter,
)


def segment(source, start, end, text, kind="fixture"):
    return {"input_start": start, "input_end": end,
            "source": source[start:end], "text": text, "kind": kind}


class Scores:
    def __init__(self, means=None):
        self.means = means or {}
        self.calls = []

    def __call__(self, model, pairs, batch_size):
        self.calls.append((model, list(pairs), batch_size))
        rows = []
        for source, target in pairs:
            tokens = 3 * len(target) + 1
            mean = self.means.get((source, target), -0.5)
            rows.append({"source": source, "target": target,
                         "log_probability": mean * tokens,
                         "token_count": tokens,
                         "mean_log_probability": mean})
        return rows


def bare_converter(weights, scorer=None):
    value = object.__new__(ProductionPhoneticChoiceConverter)
    value.lock = threading.RLock()
    value.calibration_weights = tuple(weights)
    value.calibration_source = "override"
    value._likelihood_scorer = scorer or Scores()
    value.phonetic_model = object()
    value._phonetic_model_loaded = True
    value.choice_base_paths = []
    value.choice_trace = value._empty_choice_trace()
    value.choice_ms = 0.0
    return value


def test_production_parent_runs_once_and_zero_weights_preserve_exact_pool(monkeypatch):
    source = "gim han"
    initial = [("초기", 0.0, [])]
    parent = [
        ("김 한", 2.0, [segment(source, 0, 3, "김"),
                         segment(source, 3, 4, " "),
                         segment(source, 4, 7, "한")]),
        ("김한", 1.0, [segment(source, 0, 3, "김"),
                        segment(source, 3, 4, ""),
                        segment(source, 4, 7, "한")]),
    ]
    calls = []
    monkeypatch.setattr(
        CTCUnionConverter, "_extend_sentence_paths",
        lambda self, text, context, paths: (
            calls.append((text, context, paths)) or parent))
    scorer = Scores()
    value = bare_converter([0, 0, 0, 0], scorer)

    result = value._extend_sentence_paths(source, "문맥", initial)

    assert calls == [(source, "문맥", initial)]
    assert result is parent
    assert value.choice_base_paths == parent
    assert value.choice_base_paths[0][2] is parent[0][2]
    assert set(scorer.calls[0][1]) == {("gim", "김"), ("han", "한")}
    assert value.choice_trace["zero_weight_score_parity"] is True
    assert value.choice_trace["zero_weight_order_parity"] is True


def test_frozen_residual_reorders_only_complete_production_paths(monkeypatch):
    source = "gim"
    parent = [
        ("한", 1.1, [segment(source, 0, 3, "한")]),
        ("김", 1.0, [segment(source, 0, 3, "김")]),
    ]
    monkeypatch.setattr(CTCUnionConverter, "_extend_sentence_paths",
                        lambda self, *args: parent)
    scorer = Scores({("gim", "한"): -2.0, ("gim", "김"): -0.1})
    value = bare_converter([1, 0, 0, 0], scorer)

    result = value._extend_sentence_paths(source, "", [])

    assert [row[0] for row in result] == ["김", "한"]
    assert {row["text"] for row in value.choice_trace["rows"]} == {"김", "한"}
    assert value.choice_trace["supported_pair_count"] == 2
    assert value.choice_trace["zero_weight_parity"] is True
    assert all(row[0] in {path[0] for path in parent} for row in result)
    assert all(row[2] is next(path[2] for path in parent if path[0] == row[0])
               for row in result)


def test_constructor_keeps_production_parent_and_pins_scorer_artifacts(tmp_path):
    checkpoint = tmp_path / "phonetic.pt"
    metadata = checkpoint.with_suffix(".json")
    checkpoint.write_bytes(b"frozen-model")
    metadata.write_text('{"schema":"fixture"}')
    phonetic_model = object()

    value = ProductionPhoneticChoiceConverter(
        phonetic_checkpoint=checkpoint,
        phonetic_model=phonetic_model,
        calibration_weights=[0, 0, 0, 0],
        external_lexicon=False,
        word_model=object(),
        growth_model=object(),
    )

    assert value.choice_parent_version.startswith("experimental-ctc-union-1-")
    assert value.version.startswith("experimental-production-phonetic-choice-1-")
    assert value.load_phonetic_model() is phonetic_model
    assert value.calibration_artifacts["external/phonetic.pt"] == file_sha256(
        checkpoint)
    assert value.calibration_artifacts["external/phonetic.json"] == file_sha256(
        metadata)
    source_key = "hanglish/core/production_phonetic_choice_converter.py"
    assert value.calibration_artifacts[source_key] == file_sha256(
        Path(source_key))
    reused_key = "hanglish/core/phonetic_choice_converter.py"
    assert value.calibration_artifacts[reused_key] == file_sha256(
        Path(reused_key))


def test_first_load_adopts_actual_production_version_and_merges_artifacts(monkeypatch):
    value = bare_converter([0, 0, 0, 0])
    value.choice_parent_version = "constructor-production"
    value._choice_parent_loaded = False
    value.phonetic_likelihood_artifacts = {"phonetic.pt": "model"}
    value.calibration_artifacts = {"adapter.py": "source", "phonetic.pt": "model"}
    value.choice_version = value._choice_version()
    value.version = value.choice_version
    value.models_loaded = False
    observed = []

    def parent_load(self):
        observed.append(self.version)
        self.version = "loaded-production"
        self.models_loaded = True
        self.model_artifacts = {"production": "parent"}

    monkeypatch.setattr(CTCUnionConverter, "load_models", parent_load)
    monkeypatch.setattr(value, "_phonetic_artifact_identity",
                        lambda: {"phonetic.pt": "model"})
    monkeypatch.setattr(value, "_calibration_source_artifacts",
                        lambda **kwargs: dict(value.calibration_artifacts))

    value.load_models()
    version = value.choice_version
    value.load_models()

    assert observed == ["constructor-production", "loaded-production"]
    assert value.choice_parent_version == "loaded-production"
    assert value.choice_version == version
    assert value.version == version
    assert value.model_artifacts == {
        "production": "parent", **value.calibration_artifacts}


def test_convert_keeps_ctc_union_metadata_and_publishes_choice_trace(monkeypatch):
    value = bare_converter([0.1, 0.2, -0.3, 0.4], Scores())
    value.choice_parent_version = "production-v1"
    value.choice_version = "choice-v1"
    value.calibration_artifacts = {"adapter.py": "hash"}

    monkeypatch.setattr(
        CTCUnionConverter, "convert",
        lambda self, *args: {
            "result": "raw", "candidates": [{"text": "raw"}],
            "ctc_union": {"preserved": True},
        })

    response = value.convert("raw", raw=True)

    assert response["ctc_union"] == {"preserved": True}
    assert "phonetic_word_union" not in response
    assert response["phonetic_choice"]["rows"] == []
    assert response["phonetic_choice"]["public"] == ["raw"]
    assert response["phonetic_choice"]["effective_weights"] == [
        0.1, 0.2, -0.3, 0.4]
    assert response["phonetic_choice"]["parent_version"] == "production-v1"
    assert response["phonetic_choice"]["version"] == "choice-v1"


def test_invalid_complete_production_path_is_rejected(monkeypatch):
    value = bare_converter([0, 0, 0, 0])
    monkeypatch.setattr(
        CTCUnionConverter, "_extend_sentence_paths",
        lambda self, *args: [("김", 1.0, [segment("gim", 1, 3, "김")])])

    with pytest.raises(RuntimeError, match="source offsets"):
        value._extend_sentence_paths("gim", "", [])
