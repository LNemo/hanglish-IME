from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts.capture_copy_child_fit import (
    capture_row,
    coarse_eligible,
    validate_beam,
)


class Generator:
    english_suffixes = [("imnida", "입니다")]

    def options(self, source):
        if source.lower() == "widgetimnida":
            return [SimpleNamespace(text="widget입니다", kind="mixed")]
        return []


def candidate(text="widget입니다", *, status="preserved"):
    return {
        "text": text,
        "segments": [{
            "input_start": 0,
            "input_end": 12,
            "source": "widgetimnida",
            "text": text,
            "kind": "mixed",
        }],
        "status": status,
    }


def scoring_row(*, original=None, novel=None, fallback=None, ranked=True):
    request = {"input": "widgetimnida", "context": ""}
    identity = "s3-train-00001"
    return {
        "id": identity,
        "component_id": None,
        "request": request,
        "source_shard": 0,
        "source_row_index": 17,
        "scoring": {
            "schema": "current-fit-recombination-scoring-row-v1",
            "id": identity,
            "component_id": None,
            "request": request,
            "ranked": ranked,
            "original": [] if original is None else original,
            "novel": [] if novel is None else novel,
            "fallback_public": fallback,
        },
    }


def fake_proposals(source, text, segments, *, generator, generated,
                   existing_texts, limit):
    proposed = text + "!"
    proposals = [] if proposed in existing_texts else [{
        "text": proposed,
        "segments": deepcopy(segments),
        "source_span": [6, 12],
        "model_source": "imnida",
        "replacement": {"from": "입니다", "to": "임니다"},
        "phonetic_log_probability": -1.0,
        "parent_evidence": {"parent_text": text},
    }][:limit]
    request = {"request_index": 0, "parent_index": 0,
               "segment_index": 1, "input_start": 6, "input_end": 12,
               "source": "imnida", "model_source": "imnida",
               "normalization": {"operation": "ascii_lower",
                                 "scope": "convert_child_only",
                                 "input": "imnida", "output": "imnida"}}
    return {
        "refinement": {},
        "requests": [request],
        "proposals": proposals,
        "diagnostics": {
            "generation": [{"status": "generated"}],
            "proposal_count": len(proposals),
        },
    }


def test_validate_beam_accepts_stable_equal_score_order_and_empty_beam():
    assert validate_beam("imnida", [("임니다", -1.0), ("입니다", -1.0)]) == [
        ["임니다", -1.0], ["입니다", -1.0]
    ]
    assert validate_beam("reul", []) == []


def test_validate_beam_rejects_nonfinite_duplicate_and_forward_mismatch():
    with pytest.raises(ValueError, match="finite"):
        validate_beam("reul", [("를", float("nan"))])
    with pytest.raises(ValueError, match="duplicated"):
        validate_beam("reul", [("를", -1.0), ("를", -2.0)])
    with pytest.raises(ValueError, match="forward"):
        validate_beam("reul", [("에", -1.0)])


def test_capture_scored_pool_keeps_stable_parent_order_and_budget():
    first = candidate("widget입니다")
    second = candidate("widget임니다", status="scored")
    unsupported = candidate("widget입니다아", status="unsupported")
    row = scoring_row(original=[first], novel=[second, unsupported])
    before = deepcopy(row)

    captured = capture_row(
        row, generator=Generator(), generated={"imnida": []},
        proposal_fn=fake_proposals, maximum_new=2)

    assert row == before
    assert captured["parent_source"] == "scored_pool"
    assert captured["parent_counts"]["original_parents"] == 1
    assert captured["parent_counts"]["novel_parents"] == 2
    assert captured["parent_counts"]["scorable_parents"] == 2
    assert captured["parent_counts"]["unsupported_parents"] == 1
    assert [value["parent_candidate_index"] for value in captured["proposals"]] == [0, 1]
    assert [value["parent_pool"] for value in captured["proposals"]] == ["original", "novel"]
    assert captured["budgets"] == {
        "maximum_new": 2, "accepted_new": 2,
        "remaining_new": 0, "limit_reached": True,
    }
    assert captured["parent_counts"]["budget_skipped_parents"] == 1


def test_capture_no_pool_uses_fallback_parents_without_scores():
    fallback = candidate()
    fallback["score"] = None
    row = scoring_row(ranked=False, fallback=[fallback])
    captured = capture_row(
        row, generator=Generator(), generated={"imnida": []},
        proposal_fn=fake_proposals)
    assert captured["parent_source"] == "fallback_public"
    assert captured["parent_counts"]["fallback_parents"] == 1
    assert captured["parent_counts"]["scorable_parents"] == 0
    assert captured["proposals"][0]["parent_pool"] == "fallback_public"


def test_cheap_gate_skips_nonmixed_or_identity_ascii_without_helper_call():
    parent = candidate()
    parent["segments"][0]["kind"] = "raw"
    assert coarse_eligible(parent) is False
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("helper should not run")

    row = scoring_row(original=[parent])
    captured = capture_row(
        row, generator=Generator(), generated={"imnida": []},
        proposal_fn=forbidden)
    assert called is False
    assert captured["proposals"] == []
    assert captured["parent_counts"]["cheap_skipped_parents"] == 1


def test_capture_integrates_real_helper_and_records_child_evidence():
    row = scoring_row(original=[candidate()])
    captured = capture_row(
        row, generator=Generator(), generated={"imnida": [("임니다", -0.5)]})
    assert captured["proposals"][0]["text"] == "widget임니다"
    assert captured["proposals"][0]["parent_candidate_index"] == 0
    assert captured["proposals"][0]["replacement"] == {
        "from": "입니다", "to": "임니다"
    }
    assert captured["proposals"][0]["segments"] == [
        {"input_start": 0, "input_end": 6, "source": "widget",
         "text": "widget", "kind": "raw"},
        {"input_start": 6, "input_end": 12, "source": "imnida",
         "text": "임니다", "kind": "copy_child_phonetic"},
    ]
    assert captured["requests"][0]["parent_candidate_index"] == 0
    assert captured["parent_diagnostics"][0]["diagnostics"]["proposal_count"] == 1
