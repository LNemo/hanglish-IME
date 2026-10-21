from contextlib import contextmanager
import inspect

import pytest

from hanglish.core.neural_converter import NeuralConverter
from hanglish.core.phonology import romanize_word
from hanglish.core.workbench import SCORE_SCALE_MODEL
from scripts.evaluate_rr_retraining import (
    HISTORICAL_RULE_DIAGNOSTICS,
    LIMITS,
    RESOURCE_RATIONALE,
    ROOT,
    SCORE_SCALE_CHECKPOINT,
    _source_pins,
    capture_request,
    compare_captures,
    load_cases,
    make_score_scale_converter,
    project_inputs,
    production_converter_kwargs,
    topk_outcome,
    verify_capture_identity,
)
from scripts.source_span_resources import ResourceExceeded


def capture(identifier, candidates=None, error=None, source="roman"):
    row = {
        "id": identifier,
        "request": {"id": identifier, "input": source, "context": "", "raw": False},
        "elapsed_ms": 1.0,
    }
    if candidates is not None:
        row["response"] = {
            "result": candidates[0],
            "candidates": [{"text": text} for text in candidates],
        }
    if error is not None:
        row["error"] = error
    return row


def case(identifier, expected, category="fixture", source="roman"):
    return {"id": identifier, "input": source, "expected": expected,
            "category": category}


def test_topk_counts_first_candidate_and_first_five_only():
    row = capture("one", ["wrong", "also-wrong", "answer", "x", "y", "answer2"])
    assert topk_outcome(row, ["answer"]) == {
        "top1": False, "top5": True, "error": None,
        "actual": ["wrong", "also-wrong", "answer", "x", "y"],
    }
    failed = capture("one", ["answer"], error="RuntimeError: failed")
    outcome = topk_outcome(failed, ["answer"])
    assert not outcome["top1"] and not outcome["top5"]


def test_comparison_accounts_for_corrections_regressions_top5_and_errors():
    cases = [
        case("corrected", ["gold"], "rr"),
        case("regressed", ["gold"], "rr"),
        case("top5-lost", ["gold"], "raw"),
        case("error", ["gold"], "raw"),
    ]
    baseline = [
        capture("corrected", ["wrong", "gold"]),
        capture("regressed", ["gold"]),
        capture("top5-lost", ["wrong", "gold"]),
        capture("error", ["gold"]),
    ]
    proposed = [
        capture("corrected", ["gold"]),
        capture("regressed", ["wrong", "gold"]),
        capture("top5-lost", ["wrong"]),
        capture("error", error="TimeoutError: request"),
    ]
    result = compare_captures(baseline, proposed, cases)
    assert result["corrections"] == ["corrected"]
    assert result["regressions"] == ["regressed", "error"]
    assert result["top5_gains"] == []
    assert result["top5_regressions"] == ["top5-lost", "error"]
    assert result["baseline"]["counts"] == {
        "rows": 4, "top1": 2, "top5": 4, "errors": 0,
    }
    assert result["proposed"]["counts"] == {
        "rows": 4, "top1": 1, "top5": 2, "errors": 1,
    }
    assert result["proposed"]["categories"]["raw"]["errors"] == 1


def test_capture_identity_rejects_input_hash_mismatch():
    rows = [capture("same", ["gold"])]
    baseline = {"input_sha256": "old", "source_hashes": {"engine.py": "same"},
                "rows": 1, "capture_sha256": "baseline"}
    proposed = {"input_sha256": "new", "source_hashes": {"engine.py": "same"},
                "rows": 1, "capture_sha256": "proposed"}
    with pytest.raises(ValueError, match="input hash mismatch"):
        verify_capture_identity(
            baseline, proposed, rows, rows,
            baseline_capture_sha256="baseline", proposed_capture_sha256="proposed")


def test_capture_identity_rejects_source_or_case_mismatch():
    rows = [capture("same", ["gold"])]
    identity = {"input_sha256": "same", "source_hashes": {"engine.py": "old"},
                "rows": 1, "capture_sha256": "capture"}
    changed = {"input_sha256": "same", "source_hashes": {"engine.py": "new"},
               "rows": 1, "capture_sha256": "capture"}
    with pytest.raises(ValueError, match="source hash mismatch"):
        verify_capture_identity(
            identity, changed, rows, rows,
            baseline_capture_sha256="capture", proposed_capture_sha256="capture")
    with pytest.raises(ValueError, match="cases or rules differ"):
        verify_capture_identity(
            identity, identity, rows, [capture("same", ["gold"], source="different")],
            baseline_capture_sha256="capture", proposed_capture_sha256="capture")


@pytest.mark.parametrize(("field", "message"), [
    ("rows", "row count differs"),
    ("capture_sha256", "capture hash differs"),
])
def test_capture_identity_rejects_truncated_or_modified_capture(field, message):
    rows = [capture("same", ["gold"])]
    identity = {"input_sha256": "same", "source_hashes": {"engine.py": "same"},
                "rows": 1, "capture_sha256": "actual"}
    tampered = dict(identity)
    tampered[field] = 2 if field == "rows" else "stale"
    with pytest.raises(ValueError, match=message):
        verify_capture_identity(
            tampered, identity, rows, rows,
            baseline_capture_sha256="actual", proposed_capture_sha256="actual")


def test_comparison_rejects_label_order_mismatch():
    rows = [capture("one", ["gold"])]
    with pytest.raises(ValueError, match="capture/label identity mismatch"):
        compare_captures(rows, rows, [case("other", ["gold"])])


def test_resource_limit_aborts_arm_instead_of_becoming_completed_row():
    class Guard:
        @contextmanager
        def operation(self, name, *, update):
            assert name == "predict:one" and update is True
            yield

    class Converter:
        def convert(self, *args, **kwargs):
            raise ResourceExceeded("RSS bound exceeded")

    item = {"id": "one", "input": "roman", "context": "", "raw": False}
    with pytest.raises(ResourceExceeded, match="RSS bound exceeded"):
        capture_request(Converter(), item, Guard())


def test_ordinary_request_error_is_recorded_for_comparison_accounting():
    class Guard:
        @contextmanager
        def operation(self, name, *, update):
            yield

    class Converter:
        def convert(self, *args, **kwargs):
            raise ValueError("bad candidate")

    item = {"id": "one", "input": "roman", "context": "", "raw": False}
    row = capture_request(Converter(), item, Guard(), timer=lambda: 1.0)
    assert row["error"] == "ValueError: bad candidate"
    assert row["elapsed_ms"] == 0.0


def test_rr_romanizer_rule_dependencies_are_source_pinned():
    pins = _source_pins()
    for name in (
        "hanglish/core/phonology.py",
        "hanglish/core/predicate_boundaries.py",
        "hanglish/core/data/predicate_boundary_inventory.json",
    ):
        assert str((ROOT / name).resolve()) in pins


def test_diagnostics_keep_official_historical_and_raw_categories_separate():
    cases, _ = load_cases("diagnostics")
    counts = {}
    for row in cases:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    assert counts == {
        "official_rr_normal": 138,
        "historical_rule_diagnostic": 8,
        "raw_ascii_protection": 5,
    }
    historical = [(row["input"], row["expected"][0]) for row in cases
                  if row["category"] == "historical_rule_diagnostic"]
    assert historical == list(HISTORICAL_RULE_DIAGNOSTICS)
    assert all(romanize_word(hangul) == roman for roman, hangul in historical)


def test_diagnostic_projection_excludes_labels_and_preserves_categories_in_cases():
    cases, _ = load_cases("diagnostics")
    projected = project_inputs(cases)
    assert len(projected) == 151
    assert all(set(row) == {"id", "input", "context", "raw"} for row in projected)
    assert all("expected" not in row and "category" not in row for row in projected)
    assert sum(row["category"] == "historical_rule_diagnostic" for row in cases) == 8


def test_historical_diagnostics_are_not_exact_dev_or_test_inputs():
    known_inputs = {row["input"] for split in ("dev", "test")
                    for row in load_cases(split)[0]}
    assert not ({roman for roman, _ in HISTORICAL_RULE_DIAGNOSTICS} & known_inputs)


def test_evaluator_resource_adjustment_changes_only_operational_envelope():
    assert LIMITS == {
        "whole_seconds": 3600.0,
        "update_seconds": 60.0,
        "rss_bytes": 3 * 1024**3,
        "swap_growth_bytes": 1536 * 1024**2,
    }
    rationale = " ".join(RESOURCE_RATIONALE)
    assert "two Torch CPU threads remain unchanged" in rationale
    assert "host-wide background variation" in rationale
    assert "without changing cases, checkpoints, scoring, or model selection" in rationale


def test_converter_factory_matches_production_external_lexicon_and_calibration_defaults(tmp_path):
    checkpoint = tmp_path / "phonetic_transducer.pt"
    calls = []

    class Converter:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    value = make_score_scale_converter(checkpoint, Converter)
    assert isinstance(value, Converter)
    assert calls == [production_converter_kwargs(checkpoint)]
    assert calls[0] == {
        "score_scale_checkpoint": SCORE_SCALE_CHECKPOINT,
        "phonetic_checkpoint": checkpoint,
    }
    assert SCORE_SCALE_CHECKPOINT == SCORE_SCALE_MODEL
    assert "external_lexicon" not in calls[0]
    external = inspect.signature(NeuralConverter.__init__).parameters["external_lexicon"]
    assert external.default is True
