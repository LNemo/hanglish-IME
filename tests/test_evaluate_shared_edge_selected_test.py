import json

import pytest

import scripts.evaluate_shared_edge_selected_test as selected
from scripts.evaluate_shared_edge_residual import isolated_target_texts
from scripts.evaluate_source_span_pair import collect_candidate_groups


def segment(source, start, end, text):
    return {"input_start": start, "input_end": end, "source": source[start:end],
            "text": text, "kind": "mock"}


def pool_candidate(source, identifier, parts):
    segments = [segment(source, start, end, text) for start, end, text in parts]
    return {"candidate_id": identifier, "path_id": identifier,
            "text": "".join(row["text"] for row in segments),
            "origin": "mock", "origins": ["mock"], "segments": segments}


def source_row():
    source = "abc"
    pool = [
        pool_candidate(source, "path-a", [(0, 1, "가"), (1, 3, "bc")]),
        pool_candidate(source, "path-b", [(0, 1, "a"), (1, 2, "나"), (2, 3, "c")]),
        pool_candidate(source, "path-c", [(0, 2, "ab"), (2, 3, "다")]),
    ]
    public = [{"text": row["text"], "score": 0.0,
               "segments": row["segments"]} for row in pool]
    response = {"input": source, "context": "", "result": public[0]["text"],
                "candidates": public, "pipeline": {"pool": pool}}
    return {"id": "row-1", "request": {"input": source, "context": ""},
            "elapsed_ms": 2.0, "proposed": response}


def latency():
    return {"captured_runtime": 2.0, "source_edge_proposal": 1.0,
            "parent_features": 3.0, "target_extraction": 2.0,
            selected.SELECTED_ARM: 4.0}


def test_fresh_proposal_and_prediction_capture_source_edge_audit_and_cache():
    runtime = source_row(); proposal = selected.make_proposal_row(runtime)
    texts, representatives, groups = collect_candidate_groups(runtime, proposal)
    targets = isolated_target_texts(groups)
    audit = {"edge_ids": [[0, 1, "가"]], "edge_scores": [.2],
             "residuals": [.1] * len(texts),
             "winning_path_ids": [[group[0].path_id] for group in groups]}
    prediction = selected.build_prediction(runtime, proposal, texts, representatives,
        groups, targets, [float(index) for index in range(len(texts))],
        [float(index) - .1 for index in range(len(texts))], None,
        latency(), audit, {"requested": len(targets), "hits": 1,
                           "misses": len(targets) - 1})
    assert prediction["proposal_ids"] == [row["proposal_id"] for row in proposal["proposals"]]
    assert prediction["candidate_texts"] == texts
    assert prediction["isolated_target_texts"] == targets
    assert prediction["shared_edge_audit"] == audit
    assert len(prediction["base_scores"]) == len(texts)
    assert prediction["responses"][selected.SELECTED_ARM]["candidates"][-1]["text"] == "abc"


def test_unsupported_row_preserves_exact_captured_response_without_scores():
    runtime = source_row(); proposal = selected.make_proposal_row(runtime)
    texts, representatives, groups = collect_candidate_groups(runtime, proposal)
    targets = isolated_target_texts(groups)
    zero = {key: (2.0 if key == "captured_runtime" else 0.0) for key in latency()}
    prediction = selected.build_prediction(runtime, proposal, texts, representatives,
        groups, targets, None, None, "isolated_target_exceeds_256_tokens", zero, None,
        {"requested": len(targets), "hits": 0, "misses": len(targets)})
    assert prediction["scores"] == prediction["base_scores"] == []
    assert prediction["shared_edge_audit"] is None
    assert prediction["responses"][selected.SELECTED_ARM] \
        == prediction["responses"]["captured_runtime"]


def metric(outcomes, top1, top5, protected=1000):
    return {"counts": {"rows": 1000, "errors": 0, "top1": top1, "top5": top5,
                       "protected_exact": protected}, "outcomes": outcomes}


def test_fixed_gate_requires_gain_thresholds_and_zero_paired_losses():
    baseline, improved = [], []
    for index in range(1000):
        top1, top5 = index < 953, index < 972
        baseline.append({"id": str(index), "top1": top1, "top5": top5,
                         "protected_exact": True})
        improved.append({"id": str(index), "top1": top1 or index == 953,
                         "top5": top5, "protected_exact": True})
    metrics = {"captured_runtime": metric(baseline, 953, 972),
               selected.SELECTED_ARM: metric(improved, 954, 972)}
    assert selected.decide(metrics)["offline_1000_case_gate"] is True
    improved[0] = {**improved[0], "top1": False}
    metrics[selected.SELECTED_ARM] = metric(improved, 953, 972)
    decision = selected.decide(metrics)
    assert decision["offline_1000_case_gate"] is False
    assert decision["versus_captured_runtime"]["top1_loss_ids"] == ["0"]


def test_dev_selection_and_checkpoint_are_exactly_bound_without_test_targets():
    dev = selected.validate_dev_selection(selected.DEFAULT_DEV)
    training = selected.validate_selected_training(dev["report"], selected.DEFAULT_TRAINING)
    assert dev["proof"]["verified"] is True
    assert training["report"]["completed"] is True
    assert training["residual_path"].name == "residual.safetensors"


def test_prelaunch_authorization_is_exact_and_hash_bound(tmp_path):
    expected = {"schema": selected.AUTHORIZATION_SCHEMA, "rows": 1000}
    path = tmp_path / "authorization.json"; path.write_text(json.dumps(expected))
    assert list(selected.consume_authorization(path, expected)) == [str(path.resolve())]
    path.write_text(json.dumps(expected | {"rows": 999}))
    with pytest.raises(ValueError, match="authorization differs"):
        selected.consume_authorization(path, expected)


def test_limits_and_frozen_identities_match_predeclared_gate():
    assert selected.SELECTED_ARM == "shared_edge"
    assert selected.BASELINE == {"top1": 953, "top5": 972}
    assert selected.EVAL_LIMITS == {"whole_seconds": 1800.0, "update_seconds": 60.0,
                                    "rss_bytes": 3 * 1024**3,
                                    "swap_growth_bytes": 512 * 1024**2}
    assert selected.DEV_REPORT_SHA256 == "65138dc85c01480cb8d9c8ab67a9a288c77252934f9db3f7841b65b0cf147dad"
    assert selected.DEV_PROOF_SHA256 == "6506d584f276da6311f1c464fa0374c88faa4a0fe123a32565dc75ee2f8a4d55"
