import json
from pathlib import Path

import pytest

from scripts.evaluate_local_span_objective import (
    ARM_NAMES,
    DEFAULT_CONTROL,
    DEFAULT_INPUT,
    EVAL_LIMITS,
    PROOF_SCHEMA,
    RECIPE,
    TRAIN_LIMITS,
    _finite_number,
    build_prediction,
    decide,
    validate_control_dev_capture,
    validate_training_artifacts,
    verify_control_replay,
)
from scripts.evaluate_source_span_pair import canonical_hash, collect_candidate_groups, iter_jsonl
from scripts.score_source_edge_recombination import RUNTIME
from scripts.preflight_source_span_residual import PARENT, PARENT_HASHES, sha


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True))


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


def test_supported_prediction_has_four_fixed_branches_and_path_evidence():
    runtime, proposal = source_rows()
    texts, representatives, groups = collect_candidate_groups(runtime, proposal)
    scores = {"parent": [1.0, 2.0], "source_local_control": [4.0, 3.0],
              "local_aux": [5.0, 5.0]}
    latency = {"captured_runtime": 4.0, "parent_features": 1.0,
               "source_local_control": 2.0, "local_aux": 3.0}
    row = build_prediction(runtime, texts, representatives, groups, scores, latency, None)
    assert tuple(row["responses"]) == ARM_NAMES
    assert row["responses"]["parent"]["result"] == "라"
    assert row["responses"]["source_local_control"]["result"] == "가나"
    assert row["responses"]["local_aux"]["result"] == "가나"
    assert [len(paths) for paths in row["path_ids"]] == [3, 1]
    assert set(row["scores"]) == {"parent", "source_local_control", "local_aux"}
    assert row["protected_paths"]["verified"] is True


def test_common_fallback_is_exact_for_every_branch():
    runtime, proposal = source_rows()
    texts, representatives, groups = collect_candidate_groups(runtime, proposal)
    latency = {"captured_runtime": 4.0, "parent_features": 0.0,
               "source_local_control": 0.0, "local_aux": 0.0}
    row = build_prediction(runtime, texts, representatives, groups, None, latency,
                           "source_exceeds_2048_bytes")
    expected = canonical_hash(runtime["proposed"])
    assert set(row["full_response_sha256"].values()) == {expected}
    assert row["scores"] == {}
    assert all(value == row["responses"]["captured_runtime"]
               for value in row["responses"].values())


def test_historical_dev_source_capture_is_closed_and_replay_is_exact():
    auth = validate_control_dev_capture()
    assert auth["closure"].name == "predictions.complete.json"
    runtime, proposal = source_rows()
    texts, representatives, groups = collect_candidate_groups(runtime, proposal)
    latency = {"captured_runtime": 4.0, "parent_features": 1.0,
               "source_local_control": 2.0, "local_aux": 3.0}
    prediction = build_prediction(runtime, texts, representatives, groups,
        {"parent": [1.0, 2.0], "source_local_control": [4.0, 3.0],
         "local_aux": [5.0, 5.0]}, latency, None)
    historical = json.loads(json.dumps(prediction))
    historical["scores"]["source_local"] = historical["scores"].pop("source_local_control")
    historical["responses"]["source_local"] = historical["responses"].pop("source_local_control")
    historical["full_response_sha256"]["source_local"] = (
        historical["full_response_sha256"].pop("source_local_control"))
    historical["scores"].pop("local_aux")
    historical["responses"].pop("local_aux")
    historical["full_response_sha256"].pop("local_aux")
    verify_control_replay(prediction, historical)
    historical["scores"]["source_local"][0] += .01
    with pytest.raises(ValueError, match="historical source-local"):
        verify_control_replay(prediction, historical)


def test_historical_first_dev_row_rebuilds_with_frozen_policy_without_models():
    auth = validate_control_dev_capture()
    runtime = next(iter_jsonl(RUNTIME / "capture.jsonl"))
    proposal = next(iter_jsonl(DEFAULT_INPUT / "capture.jsonl"))
    historical = next(iter_jsonl(auth["predictions"]))
    texts, representatives, groups = collect_candidate_groups(runtime, proposal)
    scores = historical["scores"]
    prediction = build_prediction(runtime, texts, representatives, groups, {
        "parent": scores["parent"],
        "source_local_control": scores["source_local"],
        "local_aux": scores["source_local"],
    }, {"captured_runtime": runtime["elapsed_ms"], "parent_features": 0.0,
        "source_local_control": 0.0, "local_aux": 0.0},
        historical["support"]["reason"])
    verify_control_replay(prediction, historical)


def metric(top1, top5, outcomes):
    return {"counts": {"top1": top1, "top5": top5}, "outcomes": outcomes}


def outcome(row_id, top1, top5, protected=True):
    return {"id": row_id, "top1": top1, "top5": top5,
            "protected_exact": protected}


def test_gate_selects_only_local_aux_against_control_and_capture():
    baseline = [outcome("a", False, True), outcome("b", True, True)]
    improved = [outcome("a", True, True), outcome("b", True, True)]
    metrics = {
        "captured_runtime": metric(1, 2, baseline),
        "parent": metric(99, 99, baseline),  # parent is reported, never a gate baseline
        "source_local_control": metric(1, 2, baseline),
        "local_aux": metric(2, 2, improved),
    }
    result = decide(metrics)
    assert result["eligible_for_1000_case_regression"] is True
    assert result["selected_for_1000_case_regression"] == "local_aux"
    assert result["historical_control_eligible"] is False

    metrics["local_aux"] = metric(2, 1, improved)
    assert decide(metrics)["eligible_for_1000_case_regression"] is False
    metrics["local_aux"] = metric(2, 2, [outcome("a", True, True, False), improved[1]])
    assert decide(metrics)["eligible_for_1000_case_regression"] is False


def make_training_bundle(directory: Path):
    directory.mkdir()
    control_report = json.loads((DEFAULT_CONTROL / "report.json").read_text())
    initial_source = DEFAULT_CONTROL / "initial.safetensors"
    (directory / "initial.safetensors").write_bytes(initial_source.read_bytes())
    for name, payload in {
        "pins.json": b"{}", "selection.json": b"[]", "control-parity.jsonl": b"{}\n",
        "updates.jsonl": b"{}\n", "after.jsonl": b"{}\n", "residual.safetensors": b"residual",
        "reload.json": b"[]", "resources.jsonl": b"{}\n"}.items():
        (directory / name).write_bytes(payload)
    pins = {str(PARENT / name): digest for name, digest in PARENT_HASHES.items()}
    write_json(directory / "pins.json", pins)
    architecture = json.loads(
        (DEFAULT_CONTROL / "source_local/checkpoint.json").read_text())["architecture"]
    checkpoint = {"schema": "local-span-objective-training-v1", "arm": "local_aux",
                  "architecture": architecture, "recipe": RECIPE, "pins": pins,
                  "residual_sha256": sha(directory / "residual.safetensors"),
                  "initial_sha256": sha(directory / "initial.safetensors"),
                  "selection_sha256": sha(directory / "selection.json")}
    write_json(directory / "checkpoint.json", checkpoint)
    output_names = ["pins.json", "selection.json", "initial.safetensors",
                    "control-parity.jsonl", "updates.jsonl", "after.jsonl",
                    "residual.safetensors", "reload.json", "checkpoint.json",
                    "resources.jsonl"]
    report = {"schema": "local-span-objective-training-v1", "completed": True,
              "rows": 1483, "updates": 2966, "recipe": RECIPE, "limits": TRAIN_LIMITS,
              "contracts": {"fit_only": True, "dev_opened": False,
                  "test_opened": False, "hold_opened": False,
                  "production_changed": False},
              "historical_control_all_row_parity": True,
              "trained_global_all_row_parity": True,
              "parent_before": {"parent": "same"}, "parent_after": {"parent": "same"},
              "pins": pins, "reload_rows": 16,
              "initial_hashes": control_report["arms"]["source_local"]["initial_hashes"],
              "epoch_mean_losses": [
                  {"total": 1.0, "candidate": .8, "local": .8},
                  {"total": .9, "candidate": .7, "local": .8}],
              "upstream_gradient_maxima": {"source_embedding.": 1.0,
                  "source_encoder.": 1.0, "segment_head.0.": 1.0,
                  "segment_head.2.": 1.0},
              "outputs": {name: sha(directory / name) for name in output_names}}
    write_json(directory / "report.json", report)
    proof = {"schema": PROOF_SCHEMA, "verified": True,
             "report_sha256": sha(directory / "report.json"),
             "producer_sha256": sha(Path("scripts/train_local_span_objective.py")),
             "counts": {"selected_fit_rows": 1483, "preferences": 6889,
                        "updates": 2966, "reload_rows": 16},
             "contracts": {"fit_only": True, "hold_opened": False,
                 "dev_opened": False, "test_opened": False,
                 "production_changed": False,
                 "historical_control_all_scores_exact": True,
                 "score_fp32_base_plus_residual_replayed": True,
                 "local_softplus_losses_replayed_from_saved_margins": True,
                 },
             "tensors": {"parent_unchanged": True},
             "resources": {"atomic_operation_attribution_verified": True}}
    write_json(directory / "independent-verification.json", proof)


def test_training_authorization_binds_control_initial_architecture_and_proof(tmp_path):
    training = tmp_path / "training"
    make_training_bundle(training)
    result = validate_training_artifacts(training)
    assert result["architecture"]["schema"] == "source-span-residual-v1"
    assert result["local_aux"]["metadata"]["arm"] == "local_aux"
    assert result["control"]["metadata"]["arm"] == "source_local"
    proof = json.loads((training / "independent-verification.json").read_text())
    proof["counts"]["updates"] = 2965
    write_json(training / "independent-verification.json", proof)
    with pytest.raises(ValueError, match="independent proof"):
        validate_training_artifacts(training)


def test_limits_and_branch_names_are_frozen_to_the_plan():
    assert ARM_NAMES == ("captured_runtime", "parent", "source_local_control", "local_aux")
    assert EVAL_LIMITS == {"whole_seconds": 1800.0, "update_seconds": 60.0,
                           "rss_bytes": 3 * 1024**3,
                           "swap_growth_bytes": 512 * 1024**2}
    assert _finite_number(-4096) is True
    assert _finite_number(float("nan")) is False
