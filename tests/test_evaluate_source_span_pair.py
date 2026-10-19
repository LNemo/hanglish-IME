import json
from pathlib import Path

import pytest

from hanglish.core.source_correspondence import align
from scripts.evaluate_source_span_pair import (
    ARMS,
    LIMITS,
    PARENT,
    PARENT_HASHES,
    RECIPE,
    TRAIN_SCHEMA,
    build_prediction,
    canonical_hash,
    collect_candidate_groups,
    decide,
    path_preserves_fixture,
    protected_fixtures,
    support_reason,
    validate_training_artifacts,
    verify_paths_protect_literals,
)
from scripts.preflight_source_span_residual import sha


def candidate(source, text, pieces):
    segments = []
    cursor = 0
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
        "score": None, "score_type": None,
    }
    runtime = {"id": "row-1", "request": {"input": source, "context": ""},
               "proposed": {"result": "가나", "candidates": [response_path],
                            "pipeline": {"pool": [first, second]}}}
    proposal_row = {"id": "row-1", "request": runtime["request"],
                    "edge_inventory": [{"edge_id": "edge-1",
                        "structural_segment": proposal["segments"][0], "provenance": []}],
                    "proposals": [proposal]}
    return runtime, proposal_row


def test_candidate_domain_matches_policy_and_retains_every_observed_path():
    runtime, proposal = source_rows()
    texts, representatives, groups = collect_candidate_groups(runtime, proposal)
    assert texts == ["가나", "라"]
    assert [item["text"] for item in representatives] == texts
    assert [len(group) for group in groups] == [3, 1]
    assert len({path.path_id for path in groups[0]}) == 3
    assert all(path.source == "abc" and path.text == texts[index]
               for index, group in enumerate(groups) for path in group)


def test_common_fallback_preserves_exact_captured_response_for_every_arm():
    runtime, proposal = source_rows()
    texts, representatives, groups = collect_candidate_groups(runtime, proposal)
    calls = []
    reason = support_reason("a" * 2049, texts, lambda values: calls.append(values))
    assert reason == "source_exceeds_2048_bytes"
    assert calls == []
    latency = {"captured_runtime": 4.0, "parent_features": 0.0,
               "whole_source": 0.0, "source_local": 0.0}
    row = build_prediction(runtime, texts, representatives, groups, None, latency, reason)
    assert row["support"] == {"supported": False, "reason": reason}
    assert set(row["responses"]) == {"captured_runtime", "parent", *ARMS}
    expected = canonical_hash(runtime["proposed"])
    assert set(row["full_response_sha256"].values()) == {expected}
    assert all(value == row["responses"]["captured_runtime"]
               for value in row["responses"].values())


def test_supported_scores_use_complete_domain_tie_break_and_explicit_raw_copy():
    runtime, proposal = source_rows()
    texts, representatives, groups = collect_candidate_groups(runtime, proposal)
    latency = {"captured_runtime": 4.0, "parent_features": 1.0,
               "whole_source": 2.0, "source_local": 3.0}
    scores = {"parent": [1.0, 2.0], "whole_source": [4.0, 3.0],
              "source_local": [5.0, 5.0]}
    row = build_prediction(runtime, texts, representatives, groups, scores, latency, None)
    assert row["responses"]["parent"]["result"] == "라"
    assert row["responses"]["whole_source"]["result"] == "가나"
    assert row["responses"]["source_local"]["result"] == "가나"
    for arm in ("parent", *ARMS):
        candidates = row["responses"][arm]["candidates"]
        assert candidates[-1] == {"text": "abc", "score": None, "segments": [{
            "input_start": 0, "input_end": 3, "source": "abc", "text": "abc",
            "kind": "raw"}]}
    with pytest.raises(ValueError, match="score map"):
        build_prediction(runtime, texts, representatives, groups,
                         {"parent": [1.0, 2.0]}, latency, None)


def test_url_email_and_explicit_copy_fixtures_require_exact_copy_spans():
    source = "`https://a.kr` x user@example.kr"
    fixtures = protected_fixtures(source)
    assert {item["kind"] for item in fixtures} == {"url", "email", "explicit_copy"}
    good = align(source, source, [{"input_start": 0, "input_end": len(source),
                                  "source": source, "text": source}])
    checked = verify_paths_protect_literals(source, [[good]])
    assert checked["verified"] is True
    assert checked["path_fixture_checks"] == 3
    url = next(item for item in fixtures if item["kind"] == "url")
    assert path_preserves_fixture(good, url) is True

    bad_text = source.replace("https://a.kr", "URL")
    bad = align(source, bad_text, [
        {"input_start": 0, "input_end": 1, "source": "`", "text": "`"},
        {"input_start": 1, "input_end": 13, "source": "https://a.kr", "text": "URL"},
        {"input_start": 13, "input_end": len(source), "source": source[13:],
         "text": source[13:]},
    ])
    assert path_preserves_fixture(bad, url) is False
    assert verify_paths_protect_literals(source, [[bad]])["verified"] is False


def fake_metric(top1, top5, outcomes):
    return {"counts": {"top1": top1, "top5": top5}, "outcomes": outcomes}


def outcome(row_id, top1, top5, protected=True):
    return {"id": row_id, "top1": top1, "top5": top5,
            "protected_exact": protected}


def test_gate_compares_both_baselines_and_uses_frozen_tie_break():
    base = [outcome("a", False, True), outcome("b", True, True)]
    improved = [outcome("a", True, True), outcome("b", True, True)]
    metrics = {
        "captured_runtime": fake_metric(1, 2, base),
        "parent": fake_metric(1, 2, base),
        "whole_source": fake_metric(2, 2, improved),
        "source_local": fake_metric(2, 2, improved),
    }
    result = decide(metrics)
    assert result["selected_for_1000_case_regression"] == "whole_source"
    assert result["arms"]["whole_source"]["eligible_for_1000_case_regression"] is True
    assert result["source_localization_claim_supported"] is False
    assert result["production_qualified"] is False

    metrics["source_local"] = fake_metric(3, 2, improved)
    result = decide(metrics)
    assert result["selected_for_1000_case_regression"] == "source_local"
    assert result["source_localization_claim_supported"] is True


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True))


def test_training_validation_binds_report_pins_and_both_checkpoints(tmp_path):
    training = tmp_path / "training"
    training.mkdir()
    (training / "selection.json").write_text("[]")
    (training / "initial.safetensors").write_bytes(b"initial")
    pins = {str(PARENT / name): digest for name, digest in PARENT_HASHES.items()}
    write_json(training / "pins.json", pins)
    (training / "feature-manifest.json").write_text("{}")
    (training / "before.jsonl").write_text("{}\n")
    architecture = {"schema": "source-span-residual-v1", "max_length": 256}
    arms = {}
    initial_hashes = {"tensor": "same"}
    for arm in ARMS:
        directory = training / arm
        directory.mkdir()
        residual = directory / "residual.safetensors"
        residual.write_bytes((arm + " residual").encode())
        checkpoint = {"schema": TRAIN_SCHEMA, "arm": arm, "architecture": architecture,
                      "recipe": RECIPE, "pins": pins,
                      "selection_sha256": sha(training / "selection.json"),
                      "initial_sha256": sha(training / "initial.safetensors"),
                      "residual_sha256": sha(residual)}
        write_json(directory / "checkpoint.json", checkpoint)
        for name in ("updates.jsonl", "after.jsonl", "reload.json"):
            (directory / name).write_text("{}\n")
        arms[arm] = {"updates": 2966,
                     "checkpoint_sha256": sha(directory / "checkpoint.json"),
                     "initial_hashes": initial_hashes,
                     "reload_rows": 16, "epoch_mean_loss": [1.0, .5],
                     "upstream_gradient_maxima": {
                         "source_embedding.": 1.0, "source_encoder.": 1.0,
                         "segment_head.0.": 1.0, "segment_head.2.": 1.0}}
    required = ["pins.json", "selection.json", "initial.safetensors",
                "feature-manifest.json", "before.jsonl"] + [
        f"{arm}/{name}" for arm in ARMS for name in
        ("updates.jsonl", "after.jsonl", "residual.safetensors",
         "checkpoint.json", "reload.json")]
    report = {"schema": TRAIN_SCHEMA, "completed": True, "rows": 1483,
              "recipe": RECIPE, "limits": LIMITS,
              "contracts": {"fit_only": True, "hold_opened": False,
                  "dev_opened": False, "test_opened": False,
                  "production_changed": False,
                  "matched_initialization_data_order_recipe": True},
              "parent_before": {"x": "y"}, "parent_after": {"x": "y"},
              "pins": pins, "arms": arms,
              "outputs": {name: sha(training / name) for name in required}}
    write_json(training / "report.json", report)
    write_json(training / "independent-verification.json", {
        "schema": "source-span-paired-training-independent-verification-v1",
        "verified": True, "run_report_sha256": sha(training / "report.json")})
    result = validate_training_artifacts(training)
    assert result["architecture"] == architecture
    assert set(result["checkpoints"]) == set(ARMS)
    (training / "whole_source" / "residual.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checkpoint identity"):
        validate_training_artifacts(training)
