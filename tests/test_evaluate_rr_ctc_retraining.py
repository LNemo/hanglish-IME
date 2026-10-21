from contextlib import contextmanager
import inspect

import pytest

from hanglish.core.neural_converter import NeuralConverter
from hanglish.core.workbench import SCORE_SCALE_MODEL
from scripts.evaluate_rr_ctc_retraining import (
    LIMITS,
    OLD_GROWTH_CHECKPOINT,
    OLD_PHONETIC_CHECKPOINT,
    OLD_WORD_CHECKPOINT,
    RESOURCE_RATIONALE,
    SCORE_SCALE_CHECKPOINT,
    capture_request,
    checkpoint_bundle,
    comparison,
    fixed_artifacts,
    make_converter,
    production_converter_kwargs,
    proposal_activity,
    proposal_difference,
    source_pins,
    verify_arm_pair,
)
from scripts.source_span_resources import ResourceExceeded


def capture(identifier, candidates=("gold",), source="roman"):
    return {
        "id": identifier,
        "request": {"id": identifier, "input": source, "context": "", "raw": False},
        "response": {"result": candidates[0],
                     "candidates": [{"text": item} for item in candidates]},
        "elapsed_ms": 1.0,
    }


def audited_capture(identifier, *, word=(1, 1, 3, 2, "complete"),
                    growth=(1, 1, 2, 1, "complete")):
    row = capture(identifier)
    for group, values in (("word_ctc", word), ("growth_ctc", growth)):
        requested, completed, generated, accepted, reason = values
        row["response"][group] = {
            "generation_requests": requested,
            "completed_generations": completed,
            "generated_proposals": generated,
            "accepted_proposals": accepted,
            "prediction": {"reason": reason, "text": "한글", "generated_tokens": 2},
        }
    return row


def identity(bundle, *, capture_hash="capture", artifacts=None):
    return {
        "input_sha256": "inputs", "source_hashes": {"runtime.py": "same"},
        "rows": 1, "capture_sha256": capture_hash,
        "checkpoint_files": bundle,
        "model_artifacts": artifacts or {"fixed/model.bin": "same"},
    }


def bundles(tmp_path):
    result = {}
    for arm in ("before", "after"):
        result[arm] = {}
        for group in ("phonetic", "word_ctc", "growth_ctc"):
            path = tmp_path / arm / group / "model.pt"
            result[arm][group] = {str(path): f"{arm}-{group}"}
    return result


def test_converter_factory_changes_only_three_checkpoints_and_keeps_production_defaults(tmp_path):
    paths = [tmp_path / name for name in ("phonetic.pt", "word.pt", "growth.pt")]
    calls = []

    class Converter:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    value = make_converter(*paths, Converter)
    assert isinstance(value, Converter)
    assert calls == [production_converter_kwargs(*paths)]
    assert calls[0] == {
        "score_scale_checkpoint": SCORE_SCALE_CHECKPOINT,
        "phonetic_checkpoint": paths[0],
        "word_checkpoint": paths[1],
        "growth_checkpoint": paths[2],
    }
    assert SCORE_SCALE_CHECKPOINT == SCORE_SCALE_MODEL
    assert "external_lexicon" not in calls[0]
    assert inspect.signature(NeuralConverter.__init__).parameters[
        "external_lexicon"].default is True


def test_default_arm_checkpoints_are_current_production_files():
    assert OLD_PHONETIC_CHECKPOINT.is_file()
    assert OLD_WORD_CHECKPOINT.is_file()
    assert OLD_GROWTH_CHECKPOINT.is_file()
    bundle = checkpoint_bundle(
        OLD_PHONETIC_CHECKPOINT, OLD_WORD_CHECKPOINT, OLD_GROWTH_CHECKPOINT)
    assert set(bundle) == {"phonetic", "word_ctc", "growth_ctc"}
    assert all(len(files) == 2 for files in bundle.values())


def test_runtime_and_ctc_dependencies_are_source_pinned():
    pins = source_pins()
    for suffix in (
        "scripts/evaluate_rr_ctc_retraining.py",
        "hanglish/core/ctc_word_converter.py",
        "hanglish/core/ctc_union_converter.py",
        "hanglish/core/ctc_word_proposals.py",
        "hanglish/models/sentence_ctc.py",
    ):
        assert any(path.endswith(suffix) for path in pins)


def test_pair_verification_allows_only_declared_checkpoint_group(tmp_path):
    values = bundles(tmp_path)
    before = identity(values["before"])
    after_bundle = {
        "phonetic": values["after"]["phonetic"],
        "word_ctc": values["before"]["word_ctc"],
        "growth_ctc": values["before"]["growth_ctc"],
    }
    after = identity(after_bundle)
    rows = [capture("one")]
    verify_arm_pair(
        "original", "phonetic_only", before, after, rows, rows,
        before_capture_sha256="capture", after_capture_sha256="capture",
        allowed_groups={"phonetic"})


def test_pair_verification_rejects_undeclared_checkpoint_or_fixed_artifact_change(tmp_path):
    values = bundles(tmp_path)
    rows = [capture("one")]
    before = identity(values["before"])
    undeclared = identity({
        "phonetic": values["after"]["phonetic"],
        "word_ctc": values["after"]["word_ctc"],
        "growth_ctc": values["before"]["growth_ctc"],
    })
    with pytest.raises(ValueError, match="unexpected checkpoint change in word_ctc"):
        verify_arm_pair(
            "original", "phonetic_only", before, undeclared, rows, rows,
            before_capture_sha256="capture", after_capture_sha256="capture",
            allowed_groups={"phonetic"})
    changed_fixed = identity({
        "phonetic": values["after"]["phonetic"],
        "word_ctc": values["before"]["word_ctc"],
        "growth_ctc": values["before"]["growth_ctc"],
    }, artifacts={"fixed/model.bin": "changed"})
    with pytest.raises(ValueError, match="non-target runtime model artifacts differ"):
        verify_arm_pair(
            "original", "phonetic_only", before, changed_fixed, rows, rows,
            before_capture_sha256="capture", after_capture_sha256="capture",
            allowed_groups={"phonetic"})


def test_pair_verification_rejects_truncated_or_reordered_capture(tmp_path):
    values = bundles(tmp_path)
    before = identity(values["before"])
    after = identity(values["before"])
    rows = [capture("one")]
    with pytest.raises(ValueError, match="capture row count differs"):
        verify_arm_pair(
            "a", "b", {**before, "rows": 2}, after, rows, rows,
            before_capture_sha256="capture", after_capture_sha256="capture",
            allowed_groups=set())
    with pytest.raises(ValueError, match="cases differ"):
        verify_arm_pair(
            "a", "b", before, after, rows, [capture("two")],
            before_capture_sha256="capture", after_capture_sha256="capture",
            allowed_groups=set())


def test_fixed_artifact_filter_removes_only_explicit_keys():
    value = {"model_artifacts": {
        "external/model.pt": "candidate", "fixed/model.bin": "same"}}
    assert fixed_artifacts(value, {"external/model.pt"}) == {
        "fixed/model.bin": "same"}


def test_comparison_binds_requested_arms_and_reports_changes():
    captures = {
        "original": [capture("one", ("wrong", "gold"))],
        "all_retrained": [capture("one", ("gold",))],
    }
    cases = [{"id": "one", "input": "roman", "expected": ["gold"],
              "category": "fixture"}]
    result = comparison("original", "all_retrained", captures, cases)
    assert result["original"]["counts"]["top1"] == 0
    assert result["all_retrained"]["counts"]["top1"] == 1
    assert result["corrections"] == ["one"]
    assert result["regressions"] == []


def test_resource_failure_remains_arm_fatal():
    class Guard:
        @contextmanager
        def operation(self, name, *, update):
            yield

    class Converter:
        def convert(self, *args, **kwargs):
            raise ResourceExceeded("resource limit")

    item = {"id": "one", "input": "roman", "context": "", "raw": False}
    with pytest.raises(ResourceExceeded, match="resource limit"):
        capture_request(Converter(), item, Guard())


def test_capture_retains_ctc_predictions_and_proposal_counters():
    class Guard:
        @contextmanager
        def operation(self, name, *, update):
            yield

    class Converter:
        def convert(self, *args, **kwargs):
            return {
                "input": "roman words", "context": "", "result": "한글 단어",
                "candidates": [{"text": "한글 단어"}], "engine_ms": 1.5,
                "ctc_word": {
                    "generation_requests": 1, "completed_generations": 1,
                    "generated_proposals": 3, "accepted_proposals": 2,
                    "prediction": {"reason": "complete", "text": "한글 단어",
                                   "generated_tokens": 4},
                },
                "ctc_union": {
                    "generation_requests": 1, "completed_generations": 1,
                    "generated_proposals": 2, "accepted_proposals": 1,
                    "prediction": {"reason": "complete", "text": "한글 단어",
                                   "generated_tokens": 4},
                },
            }

    item = {"id": "one", "input": "roman words", "context": "", "raw": False}
    row = capture_request(Converter(), item, Guard(), timer=lambda: 1.0)
    assert row["response"]["word_ctc"]["accepted_proposals"] == 2
    assert row["response"]["growth_ctc"]["generated_proposals"] == 2
    assert row["response"]["word_ctc"]["prediction"]["text"] == "한글 단어"


def test_capture_rejects_missing_ctc_audit_metadata_as_request_error():
    class Guard:
        @contextmanager
        def operation(self, name, *, update):
            yield

    class Converter:
        def convert(self, *args, **kwargs):
            return {
                "input": "roman", "context": "", "result": "한글",
                "candidates": [{"text": "한글"}], "engine_ms": 1.0,
            }

    item = {"id": "one", "input": "roman", "context": "", "raw": False}
    row = capture_request(Converter(), item, Guard(), timer=lambda: 1.0)
    assert row["error"] == "ValueError: runtime response is missing ctc_word"


def test_proposal_activity_and_differences_are_label_free_and_arm_bound():
    before = [audited_capture("one"), audited_capture(
        "two", word=(1, 1, 1, 0, "complete"),
        growth=(0, 0, 0, 0, "unsupported_source"))]
    after = [audited_capture(
        "one", word=(1, 1, 4, 3, "complete"),
        growth=(1, 1, 4, 2, "complete")), before[1]]
    after[0]["response"]["growth_ctc"]["prediction"]["text"] = "새 한글"
    activity = proposal_activity(before)
    assert activity["word_ctc"]["generated_proposals"] == 4
    assert activity["word_ctc"]["accepted_proposals"] == 2
    assert activity["word_ctc"]["requests_with_accepted_proposals"] == 1
    assert activity["growth_ctc"]["prediction_reasons"] == {
        "complete": 1, "unsupported_source": 1}
    difference = proposal_difference(before, after)
    assert difference["word_ctc"]["deltas"]["generated_proposals"] == 1
    assert difference["word_ctc"]["deltas"]["accepted_proposals"] == 1
    assert difference["growth_ctc"]["requests_with_changed_prediction"] == 1


def test_resource_envelope_is_three_arm_specific():
    assert LIMITS == {
        "whole_seconds": 5400.0,
        "update_seconds": 60.0,
        "rss_bytes": 3 * 1024**3,
        "swap_growth_bytes": 1536 * 1024**2,
    }
    rationale = " ".join(RESOURCE_RATIONALE)
    assert "three full-engine arms" in rationale
    assert "two Torch CPU threads" in rationale
