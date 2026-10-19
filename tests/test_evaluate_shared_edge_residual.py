import json

import pytest
import torch
from types import SimpleNamespace

from scripts.evaluate_shared_edge_residual import (
    ARM_NAMES,
    EVAL_LIMITS,
    TARGET_CACHE_BYTES,
    build_prediction,
    common_support_reason,
    consume_authorization,
    decide,
    extend_target_cache,
    isolated_target_texts,
    verify_control_replay,
)
from scripts.evaluate_source_span_pair import collect_candidate_groups


def candidate(source, text, pieces):
    segments = []; cursor = 0
    for width, value in pieces:
        segments.append({"input_start": cursor, "input_end": cursor + width,
                         "source": source[cursor:cursor + width], "text": value,
                         "kind": "mock"})
        cursor += width
    assert cursor == len(source)
    return {"text": text, "segments": segments}


def source_rows():
    source = "abc"
    first = candidate(source, "가나", [(1, "가"), (2, "나")])
    second = candidate(source, "가나", [(2, "가"), (1, "나")])
    response_path = candidate(source, "가나", [(1, ""), (1, "가"), (1, "나")])
    proposal = candidate(source, "라", [(3, "라")]) | {
        "proposal_id": "proposal-1", "edge_ids": ["edge-1"],
        "origin": "source_edge_recombination", "selection_eligible": False,
        "score": None, "score_type": None}
    runtime = {"id": "row-1", "request": {"input": source, "context": ""},
               "elapsed_ms": 4.0,
               "proposed": {"result": "가나", "candidates": [response_path],
                            "pipeline": {"pool": [first, second]}}}
    proposal_row = {"id": "row-1", "request": runtime["request"],
                    "edge_inventory": [{"edge_id": "edge-1",
                        "structural_segment": proposal["segments"][0], "provenance": []}],
                    "proposals": [proposal]}
    return runtime, proposal_row


def domain():
    runtime, proposal = source_rows()
    texts, representatives, groups = collect_candidate_groups(runtime, proposal)
    return runtime, texts, representatives, groups, isolated_target_texts(groups)


def latency():
    return {"captured_runtime": 4.0, "parent_features": 1.0,
            "target_extraction": 2.0, "source_local_control": 3.0,
            "shared_edge": 4.0}


def test_isolated_target_domain_is_exact_sorted_edge_texts():
    _, _, _, groups, targets = domain()
    assert targets == ["", "가", "나", "라"]
    assert all(target in {span.text for group in groups for path in group for span in path.spans}
               for target in targets)


def test_common_support_checks_source_full_candidates_then_isolated_targets():
    calls = []
    assert common_support_reason("abc", ["가"], ["", "가"],
        lambda values: calls.append(list(values)) or True) is None
    assert calls == [["가"], ["", "가"]]
    assert common_support_reason("x" * 2049, ["가"], ["가"],
        lambda _values: pytest.fail("must not tokenize")) == "source_exceeds_2048_bytes"

    calls.clear()
    def reject_second(values):
        calls.append(list(values))
        if len(calls) == 2:
            raise ValueError("too long")
        return True
    assert common_support_reason("abc", ["가"], ["long"], reject_second) \
        == "isolated_target_exceeds_256_tokens"
    assert calls == [["가"], ["long"]]


def test_target_cache_extraction_is_text_only_chunked_and_bounded():
    calls = []
    class FakeModel:
        def extract_edge_features(self, texts):
            calls.append(list(texts))
            return SimpleNamespace(texts=tuple(texts),
                vectors=torch.arange(len(texts) * 4, dtype=torch.float32).reshape(len(texts), 4),
                has_tokens=torch.ones(len(texts), dtype=torch.bool))
    vectors, flags = {}, {}
    size = extend_target_cache(FakeModel(), [f"t{i}" for i in range(5)],
                               vectors, flags, chunk_size=2)
    assert calls == [["t0", "t1"], ["t2", "t3"], ["t4"]]
    assert set(vectors) == set(flags) == {f"t{i}" for i in range(5)}
    assert size == 5 * (4 * 4 + 1)
    with pytest.raises(ValueError, match="cache input"):
        extend_target_cache(FakeModel(), ["t0"], vectors, flags)


def test_supported_prediction_records_scores_cache_audit_and_four_branches():
    runtime, texts, representatives, groups, targets = domain()
    scores = {"parent": [1.0, 2.0], "source_local_control": [4.0, 3.0],
              "shared_edge": [5.0, 5.0]}
    audit = {"edge_ids": [[0, 1, "가"]], "edge_scores": [.25],
             "residuals": [.1, .2],
             "winning_path_ids": [["path-1"], ["path-2"]]}
    row = build_prediction(runtime, texts, representatives, groups, targets,
        scores, latency(), None, audit,
        {"requested": len(targets), "hits": 1, "misses": len(targets) - 1})
    assert tuple(row["responses"]) == ARM_NAMES
    assert row["responses"]["parent"]["result"] == "라"
    assert row["responses"]["source_local_control"]["result"] == "가나"
    assert row["responses"]["shared_edge"]["result"] == "가나"
    assert row["shared_edge_audit"] == audit
    assert row["target_cache"]["hits"] + row["target_cache"]["misses"] == len(targets)
    assert [len(value) for value in row["path_ids"]] == [3, 1]


def test_common_fallback_preserves_exact_captured_response_and_has_no_outputs():
    runtime, texts, representatives, groups, targets = domain()
    zero_latency = {key: (4.0 if key == "captured_runtime" else 0.0)
                    for key in latency()}
    row = build_prediction(runtime, texts, representatives, groups, targets,
        None, zero_latency, "isolated_target_exceeds_256_tokens", None,
        {"requested": len(targets), "hits": 0, "misses": len(targets)})
    assert row["scores"] == {} and row["shared_edge_audit"] is None
    assert all(value == row["responses"]["captured_runtime"]
               for value in row["responses"].values())


def metric(top1, top5, outcomes):
    return {"counts": {"top1": top1, "top5": top5}, "outcomes": outcomes}


def outcome(row_id, top1, top5, protected=True):
    return {"id": row_id, "top1": top1, "top5": top5,
            "protected_exact": protected}


def test_gate_selects_only_shared_edge_against_both_frozen_baselines():
    baseline = [outcome("a", False, True), outcome("b", True, True)]
    improved = [outcome("a", True, True), outcome("b", True, True)]
    metrics = {"captured_runtime": metric(1, 2, baseline),
               "parent": metric(99, 99, baseline),
               "source_local_control": metric(1, 2, baseline),
               "shared_edge": metric(2, 2, improved)}
    decision = decide(metrics)
    assert decision["selected_for_1000_case_regression"] == "shared_edge"
    assert decision["historical_control_eligible"] is False
    metrics["shared_edge"] = metric(2, 1, improved)
    assert decide(metrics)["eligible_for_1000_case_regression"] is False
    metrics["shared_edge"] = metric(2, 2, [outcome("a", True, True, False), improved[1]])
    assert decide(metrics)["eligible_for_1000_case_regression"] is False


def test_control_replay_is_required_only_on_common_supported_rows():
    runtime, texts, representatives, groups, targets = domain()
    scores = {"parent": [1.0, 2.0], "source_local_control": [4.0, 3.0],
              "shared_edge": [5.0, 5.0]}
    audit = {"edge_ids": [], "edge_scores": [], "residuals": [0., 0.],
             "winning_path_ids": [[], []]}
    prediction = build_prediction(runtime, texts, representatives, groups, targets,
        scores, latency(), None, audit,
        {"requested": len(targets), "hits": 0, "misses": len(targets)})
    historical = json.loads(json.dumps(prediction))
    historical["scores"]["source_local"] = historical["scores"].pop("source_local_control")
    historical["responses"]["source_local"] = historical["responses"].pop("source_local_control")
    historical["full_response_sha256"]["source_local"] = (
        historical["full_response_sha256"].pop("source_local_control"))
    verify_control_replay(prediction, historical)
    historical["scores"]["source_local"][0] += .1
    with pytest.raises(ValueError, match="supported score replay"):
        verify_control_replay(prediction, historical)

    fallback = json.loads(json.dumps(prediction)); fallback["support"] = {
        "supported": False, "reason": "isolated_target_exceeds_256_tokens"}
    fallback["scores"] = {}
    verify_control_replay(fallback, historical)


def test_limits_and_arm_names_are_frozen_to_the_plan():
    assert ARM_NAMES == ("captured_runtime", "parent", "source_local_control", "shared_edge")
    assert EVAL_LIMITS == {"whole_seconds": 1800.0, "update_seconds": 60.0,
                           "rss_bytes": 3 * 1024**3,
                           "swap_growth_bytes": 512 * 1024**2}
    assert TARGET_CACHE_BYTES == 256 * 1024**2


def test_prelaunch_authorization_is_exact_and_hash_bound(tmp_path):
    expected = {"schema": "shared-edge-residual-dev-prelaunch-authorization-v1",
                "rows": 316}
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(expected), encoding="utf-8")
    pins = consume_authorization(path, expected)
    assert list(pins) == [str(path.resolve())]
    path.write_text(json.dumps(expected | {"rows": 315}), encoding="utf-8")
    with pytest.raises(ValueError, match="authorization differs"):
        consume_authorization(path, expected)
