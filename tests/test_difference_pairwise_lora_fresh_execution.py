from copy import deepcopy
import hashlib
import json

import pytest

from scripts.difference_pairwise_lora_evaluation import canonical
from scripts.difference_pairwise_lora_fresh_execution import (
    DIGIT_IDS,
    PAIR_INFERENCE_SCHEMA,
    execute,
)
from scripts.difference_pairwise_lora_inference import prepare_inference_selection


def _candidate(text, score):
    return {"text": text, "score": score, "segments": [{
        "input_start": 0, "input_end": 6, "source": "imnida", "text": text,
        "kind": "morphology",
    }]}


def _cohort(*, no_alternative=False):
    rows, qualifications, targets = [], {}, []
    for index in range(888):
        role = "fresh_natural" if index < 512 else "challenge_preservation"
        candidates = [_candidate("임니다", 2.0)]
        if not no_alternative:
            candidates.append(_candidate("입니다", 1.0))
        context = f"문서 {index}"
        parent = {"input": "imnida", "context": context, "raw": False,
                  "version": "neural-1-867f13596ed5", "result": "임니다",
                  "candidates": candidates}
        raw = {"input": "imnida", "context": context, "raw": True,
               "version": "neural-1-867f13596ed5", "result": "imnida",
               "candidates": [{"text": "imnida"}]}
        row = {"role": role, "id": f"r{index:04d}",
               "group_id": f"g{index % 387:03d}" if index < 512 else f"p{index:04d}",
               "input": "imnida", "context": context, "parent_response": parent,
               "candidates": candidates, "raw_bypass_response": raw}
        rows.append(row)
        targets.append({"id": row["id"], "group_id": row["group_id"], "role": role,
                        "expected": ["임니다"]})
        selection = prepare_inference_selection(role, row)
        for pair in selection["pairs"]:
            pair_id = pair["pair_id"]
            qualifications[pair_id] = {
                "pair_id": pair_id,
                "pair_sha256": hashlib.sha256(canonical(pair)).hexdigest(),
                "messages": [{"role": "user", "content": pair_id}],
                "tokens": [1, 2, 3],
                "prompt_tokens": 3,
                "digit_continuation_ids": DIGIT_IDS,
                "rendered_prefix": pair_id,
            }
    return rows, qualifications, targets


def _callbacks(targets, events, *, release_error=None):
    def checkpoint(stage):
        events.append(("checkpoint", stage))
        return {"passed": True, "stage": stage, "rss_bytes": 10}

    def validate_sources(stage):
        events.append(("sources", stage))
        return {"passed": True, "stage": stage, "source_sha256": "a" * 64}

    def score_pair(pair, qualification):
        events.append(("score", pair["row_id"]))
        assert qualification["pair_id"] == pair["pair_id"]
        return [1.0, 0.0]

    def release_model():
        events.append(("release", None))
        if release_error:
            raise release_error
        return {"released": True, "model_loaded": True}

    def load_targets():
        events.append(("targets", None))
        return deepcopy(targets)

    return score_pair, release_model, load_targets, checkpoint, validate_sources


def _run(tmp_path, rows, qualifications, targets, events, **overrides):
    callbacks = _callbacks(targets, events, release_error=overrides.pop("release_error", None))
    names = ("score_pair", "release_model", "load_targets", "checkpoint", "validate_sources")
    values = dict(zip(names, callbacks))
    values.update(overrides)
    return execute(rows, qualifications, tmp_path / "output", **values)


def test_complete_888_row_execution_closes_releases_checks_then_loads_targets(tmp_path):
    rows, qualifications, targets = _cohort()
    for target in targets[:512]:
        target["expected"] = ["입니다"]
    events = []
    def score_by_role(pair, qualification):
        events.append(("score", pair["row_id"]))
        assert qualification["pair_id"] == pair["pair_id"]
        return [0.0, 1.0] if pair["role"] == "fresh_natural" else [1.0, 0.0]

    report = _run(tmp_path, rows, qualifications, targets, events,
                  score_pair=score_by_role)
    assert report["completed"] is True
    assert report["terminal_status"] == "quality_passed"
    assert report["pair_inference_count"] == report["qualified_pair_count"] == 888
    assert report["prediction_count"] == 888
    assert report["pair_stream_closed"] is report["prediction_stream_closed"] is True
    assert report["model_release_succeeded"] is True
    assert report["final_source_check_succeeded"] is True
    assert report["final_resource_check_succeeded"] is True
    release_index = events.index(("release", None))
    final_resource = events.index(("checkpoint", "after_model_release"))
    final_sources = events.index(("sources", "after_model_release"))
    target_index = events.index(("targets", None))
    assert release_index < final_resource < final_sources < target_index
    first_score = next(index for index, item in enumerate(events) if item[0] == "score")
    assert events[first_score + 1] == ("checkpoint", "pair_1_scored")
    pair_lines = (tmp_path / "output/pair_inferences.jsonl").read_text().splitlines()
    predictions = (tmp_path / "output/predictions.jsonl").read_text().splitlines()
    assert len(pair_lines) == len(predictions) == 888
    assert json.loads(pair_lines[0])["schema"] == PAIR_INFERENCE_SCHEMA
    assert all(json.loads(line)["complete"] is True for line in predictions)
    assert report["metrics_verification"]["verification_passed"] is True
    assert report["metrics_verification"]["quality_gate_passed"] is True
    assert report["metrics_verification"]["execution_verified"] is False
    assert report["promotion_allowed"] is False


def test_rows_with_no_alternative_serialize_predictions_without_scoring(tmp_path):
    rows, qualifications, targets = _cohort(no_alternative=True)
    events = []
    report = _run(tmp_path, rows, qualifications, targets, events)
    assert report["qualified_pair_count"] == report["pair_inference_count"] == 0
    assert report["prediction_count"] == 888
    assert not any(kind == "score" for kind, _ in events)


@pytest.mark.parametrize("corruption", ["missing", "extra", "hash", "tokens", "overflow", "digits"])
def test_prompt_coverage_and_binding_fail_before_first_score_or_target(tmp_path, corruption):
    rows, qualifications, targets = _cohort()
    qualifications = deepcopy(qualifications)
    first = next(iter(qualifications))
    if corruption == "missing":
        del qualifications[first]
    elif corruption == "extra":
        qualifications["extra"] = deepcopy(qualifications[first])
    elif corruption == "hash":
        qualifications[first]["pair_sha256"] = "0" * 64
    elif corruption == "tokens":
        qualifications[first]["tokens"] = []
        qualifications[first]["prompt_tokens"] = 0
    elif corruption == "overflow":
        qualifications[first]["tokens"] = list(range(1025))
        qualifications[first]["prompt_tokens"] = 1025
    else:
        qualifications[first]["digit_continuation_ids"] = {"1": 1, "2": 2}
    events = []
    with pytest.raises(ValueError):
        _run(tmp_path, rows, qualifications, targets, events)
    assert not any(kind in {"score", "targets"} for kind, _ in events)
    assert ("release", None) in events
    report = json.loads((tmp_path / "output/report.json").read_text())
    assert report["terminal_status"] == "failed"
    assert report["target_accessed"] is False


def test_last_row_scoring_error_closes_and_releases_but_never_loads_targets(tmp_path):
    rows, qualifications, targets = _cohort()
    events = []
    callbacks = _callbacks(targets, events)
    calls = 0

    def fail_last(pair, qualification):
        nonlocal calls
        calls += 1
        events.append(("score", pair["row_id"]))
        if calls == 888:
            raise RuntimeError("last-row failure")
        return [1.0, 0.0]

    with pytest.raises(RuntimeError, match="last-row failure"):
        execute(rows, qualifications, tmp_path / "output", score_pair=fail_last,
                release_model=callbacks[1], load_targets=callbacks[2],
                checkpoint=callbacks[3], validate_sources=callbacks[4])
    assert ("release", None) in events
    assert ("targets", None) not in events
    report = json.loads((tmp_path / "output/report.json").read_text())
    assert report["pair_stream_closed"] is report["prediction_stream_closed"] is True
    assert report["prediction_count"] == 887
    assert report["target_accessed"] is False


def test_last_qualified_prompt_is_prevalidated_before_any_scoring(tmp_path):
    rows, qualifications, targets = _cohort()
    last = next(reversed(qualifications))
    qualifications[last]["pair_sha256"] = "0" * 64
    events = []
    with pytest.raises(ValueError, match="qualified prompt"):
        _run(tmp_path, rows, qualifications, targets, events)
    assert not any(kind in {"score", "targets"} for kind, _ in events)


@pytest.mark.parametrize("logits", [[float("nan"), 0.0], [0.0], [0.0, float("inf")]])
def test_malformed_or_nonfinite_logits_block_targets(tmp_path, logits):
    rows, qualifications, targets = _cohort()
    events = []
    with pytest.raises(ValueError, match="logits"):
        _run(tmp_path, rows, qualifications, targets, events,
             score_pair=lambda _pair, _qualification: logits)
    assert ("targets", None) not in events


def test_release_failure_blocks_targets_and_is_reported(tmp_path):
    rows, qualifications, targets = _cohort()
    events = []
    with pytest.raises(RuntimeError, match="release failed"):
        _run(tmp_path, rows, qualifications, targets, events,
             release_error=RuntimeError("release failed"))
    assert ("targets", None) not in events
    report = json.loads((tmp_path / "output/report.json").read_text())
    assert report["model_release_succeeded"] is False


@pytest.mark.parametrize("release_evidence", [
    None,
    {"released": False},
    [],
    {"released": True, "invalid": float("nan")},
])
def test_missing_false_or_malformed_release_evidence_blocks_targets(
        tmp_path, release_evidence):
    rows, qualifications, targets = _cohort()
    events = []

    def unproven_release():
        events.append(("release", None))
        return release_evidence

    with pytest.raises((TypeError, ValueError)):
        _run(tmp_path, rows, qualifications, targets, events,
             release_model=unproven_release)
    assert ("targets", None) not in events
    report = json.loads((tmp_path / "output/report.json").read_text())
    assert report["model_release_succeeded"] is False
    assert report["target_accessed"] is False


@pytest.mark.parametrize("callback_name,stage", [
    ("checkpoint", "after_model_release"),
    ("validate_sources", "after_model_release"),
])
def test_final_resource_or_source_failure_blocks_targets(tmp_path, callback_name, stage):
    rows, qualifications, targets = _cohort()
    events = []

    def failing(value):
        events.append((callback_name, value))
        if value == stage:
            raise RuntimeError(f"{callback_name} failed")
        return {"passed": True, "stage": value}

    with pytest.raises(RuntimeError, match="failed"):
        _run(tmp_path, rows, qualifications, targets, events, **{callback_name: failing})
    assert ("targets", None) not in events


def test_existing_output_is_never_overwritten_and_no_callback_runs(tmp_path):
    rows, qualifications, targets = _cohort()
    output = tmp_path / "output"
    output.mkdir()
    marker = output / "marker"
    marker.write_text("preserve")
    events = []
    callbacks = _callbacks(targets, events)
    with pytest.raises(FileExistsError):
        execute(rows, qualifications, output, score_pair=callbacks[0],
                release_model=callbacks[1], load_targets=callbacks[2],
                checkpoint=callbacks[3], validate_sources=callbacks[4])
    assert marker.read_text() == "preserve"
    assert events == []


def test_prediction_close_failure_blocks_targets_and_release_still_runs(tmp_path, monkeypatch):
    from scripts import difference_pairwise_lora_fresh_execution as module

    rows, qualifications, targets = _cohort()
    events = []
    original = module._ExclusiveJsonl.close

    def fail_prediction_close(self):
        original(self)
        if self.path.name == "predictions.jsonl":
            raise OSError("prediction close failed")

    monkeypatch.setattr(module._ExclusiveJsonl, "close", fail_prediction_close)
    with pytest.raises(OSError, match="prediction close failed"):
        _run(tmp_path, rows, qualifications, targets, events)
    assert ("release", None) in events
    assert ("targets", None) not in events
    report = json.loads((tmp_path / "output/report.json").read_text())
    assert report["prediction_stream_closed"] is False


def test_prediction_write_failure_blocks_targets_and_release_still_runs(tmp_path, monkeypatch):
    from scripts import difference_pairwise_lora_fresh_execution as module

    rows, qualifications, targets = _cohort()
    events = []
    original = module._ExclusiveJsonl.write

    def fail_first_prediction(self, value):
        if self.path.name == "predictions.jsonl":
            raise OSError("prediction write failed")
        return original(self, value)

    monkeypatch.setattr(module._ExclusiveJsonl, "write", fail_first_prediction)
    with pytest.raises(OSError, match="prediction write failed"):
        _run(tmp_path, rows, qualifications, targets, events)
    assert ("release", None) in events
    assert ("targets", None) not in events
    report = json.loads((tmp_path / "output/report.json").read_text())
    assert report["prediction_count"] == 0
    assert report["target_accessed"] is False


def test_target_access_is_recorded_before_a_loader_reads_then_raises(tmp_path):
    rows, qualifications, targets = _cohort()
    events = []

    def read_then_raise():
        events.append(("targets", None))
        _ = targets[0]["expected"]
        raise RuntimeError("target loader failed after read")

    with pytest.raises(RuntimeError, match="after read"):
        _run(tmp_path, rows, qualifications, targets, events, load_targets=read_then_raise)
    report = json.loads((tmp_path / "output/report.json").read_text())
    assert report["target_accessed"] is True
    assert any(event["stage"] == "target_access_started" for event in report["events"])


@pytest.mark.parametrize("stream_name", ["pair_inferences.jsonl", "predictions.jsonl"])
def test_closed_stream_corruption_during_release_is_rejected_before_targets(
        tmp_path, stream_name):
    rows, qualifications, targets = _cohort()
    events = []
    output = tmp_path / "output"

    def corrupt_during_release():
        events.append(("release", None))
        with (output / stream_name).open("ab") as stream:
            stream.write(b"{}\n")
        return {"released": True, "model_loaded": True}

    with pytest.raises(ValueError, match="integrity differs"):
        _run(tmp_path, rows, qualifications, targets, events,
             release_model=corrupt_during_release)
    assert ("targets", None) not in events
    report = json.loads((output / "report.json").read_text())
    assert report["target_accessed"] is False


def test_closed_stream_corruption_during_final_source_check_blocks_targets(tmp_path):
    rows, qualifications, targets = _cohort()
    events = []
    output = tmp_path / "output"

    def corrupting_source_check(stage):
        events.append(("sources", stage))
        if stage == "after_model_release":
            with (output / "predictions.jsonl").open("ab") as stream:
                stream.write(b"{}\n")
        return {"passed": True, "stage": stage}

    with pytest.raises(ValueError, match="integrity differs"):
        _run(tmp_path, rows, qualifications, targets, events,
             validate_sources=corrupting_source_check)
    assert ("targets", None) not in events
    report = json.loads((output / "report.json").read_text())
    assert report["final_source_check_succeeded"] is True
    assert report["closed_streams_reopened_and_verified"] is False
    assert report["target_accessed"] is False
