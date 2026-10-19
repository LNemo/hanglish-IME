import json

import pytest

from scripts import score_fresh_recombination_training as fresh
from scripts import score_recombination_training_pools as base


def prepared(identifier="r1", split="inner_hold", component="c1"):
    return {"id": identifier, "input": "imnida test", "expected": ["입니다 테스트"],
            "group_id": "g1", "source_ids": ["s1"], "component_id": component,
            "split": split, "parent_result": "임니다 테스트",
            "parent_candidates": ["임니다 테스트", "imnida test"],
            "original_candidates": ["임니다 테스트", "imnida test"],
            "recombined_candidates": ["임니다 테스트", "입니다 테스트", "imnida test"],
            "novel_candidates": ["입니다 테스트"]}


def test_reuses_frozen_scoring_helpers():
    assert fresh.base.source_projection is base.source_projection
    assert fresh.base.score_row is base.score_row
    assert fresh.base.summarize is base.summarize
    assert fresh.base.adequacy_gate is base.adequacy_gate
    assert fresh.base.frozen_model_pins is base.frozen_model_pins


def test_reused_fallback_preserves_exact_parent_display():
    row = prepared()
    row.update(parent_result="imnida test",
               parent_candidates=["imnida test", "displayed"],
               original_candidates=["pool-a", "displayed", "imnida test"],
               recombined_candidates=["pool-a", "displayed", "new", "imnida test"],
               novel_candidates=["new"])
    source = base.source_projection(row)
    result = base.apply_scores(source, source["recombined_candidates"], {}, False)
    assert result == {"applied": False, "result": "imnida test",
                      "candidates": ["imnida test", "displayed"]}


def test_reused_support_gate_is_union_only():
    inner = {"counts": {"adequacy_eligible": 64},
             "adequacy_eligible_parent_wrong_ids": [f"r{x}" for x in range(16)]}
    assert base.adequacy_gate(inner)["passed"] is True


def test_pending_capture_hashes_fail_before_model_import(monkeypatch, tmp_path):
    monkeypatch.setattr(fresh, "EXPECTED_CAPTURE_HASHES", {"report.json": "PENDING"})
    with pytest.raises(ValueError, match="not frozen"):
        fresh.validate_inputs(tmp_path, tmp_path / "proof.json")


def test_run_does_not_reach_model_factory_before_qualification(monkeypatch, tmp_path):
    monkeypatch.setattr(fresh, "EXPECTED_CAPTURE_HASHES", {"report.json": "PENDING"})
    called = []
    with pytest.raises(ValueError, match="not frozen"):
        fresh.run(tmp_path, tmp_path / "proof.json", tmp_path / "output",
                  ranker_factory=lambda _: called.append(True))
    assert called == [] and not (tmp_path / "output").exists()


def test_selection_constants_are_frozen():
    assert fresh.SELECTION_HASHES == {
        "manifest.json": "5680aa160f99d9656b02e52e06d4218f0ca386848d7b733fdf03d2570acfa696",
        "selection.jsonl": "15a0617e95d5c2fb0ec7ce895230794a9ac9d27f185b288eb3d138e0f91968d2",
        "independent-verification.json": "5aa1c088cfe84c30581325b934098e41db568fd14824153c56c61d9e0519a492",
    }


def test_global_partition_tamper_is_rejected(tmp_path):
    path = tmp_path / "prepared.jsonl"
    path.write_text(json.dumps(prepared(component="wrong"), ensure_ascii=False) + "\n")
    selection = {
        "paths": {}, "manifest": {}, "proof": {},
        "selection": [{"id": "r1", "input": "imnida test"}],
        "partition": {"r1": ("expected-component", "inner_hold")}}
    with pytest.raises(ValueError, match="global partition"):
        fresh.validate_prepared_sources(
            path, selection, expected_rows=1, expected_partition={"inner_hold": 1})


def test_global_partition_binding_accepts_exact_source_projection(tmp_path):
    path = tmp_path / "prepared.jsonl"
    path.write_text(json.dumps(prepared(), ensure_ascii=False) + "\n")
    selection = {"selection": [{"id": "r1", "input": "imnida test"}],
                 "partition": {"r1": ("c1", "inner_hold")}}
    rows = fresh.validate_prepared_sources(
        path, selection, expected_rows=1, expected_partition={"inner_hold": 1})
    assert rows[0]["component_id"] == "c1" and "expected" not in rows[0]


def test_full_synthetic_authorization_succeeds_without_model(monkeypatch, tmp_path):
    data = tmp_path / "data"; data.mkdir()
    source = prepared()
    (data / "prepared.jsonl").write_text(json.dumps(source, ensure_ascii=False) + "\n")
    for name in ("runtime-capture.jsonl", "source-capture.jsonl"):
        (data / name).write_text(json.dumps({"id": "r1"}) + "\n")
    (data / "capture.complete.json").write_text(json.dumps({"rows": 1}) + "\n")
    (data / "pins.json").write_text("{}\n")
    reference = tmp_path / "reference.json"; reference.write_text("{}\n")
    reference_sha = base.sha256_file(reference)
    monkeypatch.setattr(fresh, "EXPECTED_ROWS", 1)
    monkeypatch.setattr(fresh, "EXPECTED_PARTITION", {"inner_hold": 1})
    monkeypatch.setattr(fresh, "RUNTIME_REFERENCE", reference)
    monkeypatch.setattr(fresh, "RUNTIME_REFERENCE_SHA", reference_sha)
    selection = {"paths": {}, "manifest": {}, "proof": {},
                 "selection": [{"id": "r1", "input": "imnida test"}],
                 "partition": {"r1": ("c1", "inner_hold")}}
    monkeypatch.setattr(fresh, "selection_authorization", lambda: selection)
    model_pins = {base.path_key(base.CHECKPOINT / name): digest
                  for name, digest in base.EXPECTED_MODEL.items()}
    outputs = {name: {"rows": 1, "sha256": base.sha256_file(data / name)}
               for name in ("runtime-capture.jsonl", "source-capture.jsonl",
                            "capture.complete.json", "prepared.jsonl")}
    report = {"schema": fresh.CAPTURE_SCHEMA, "completed": True,
              "input_contract": "roman_sentence_only", "rows": 1,
              "prepared_rows": 1, "partition_counts": {"inner_hold": 1},
              "runtime_version": fresh.EXPECTED_RUNTIME_VERSION,
              "outputs": outputs, "pins": {}, "model_pins": model_pins,
              "runtime_reference": {"path": base.path_key(reference),
                                    "sha256": reference_sha},
              "source_only_capture": True,
              "targets_joined_after_closed_capture": True,
              "labels_opened_after_capture": True, "single_request_conversion": True,
              "typed_pipeline_captured": True, "model_loaded": True,
              "model_inference": True, "dev_accessed": False,
              "test_accessed": False, "production_changed": False}
    (data / "report.json").write_text(json.dumps(report) + "\n")
    report_sha = base.sha256_file(data / "report.json")
    proof = {"schema": fresh.PROOF_SCHEMA, "verified": True,
             "report_sha256": report_sha,
             "runtime_capture_sha256": outputs["runtime-capture.jsonl"]["sha256"],
             "source_capture_sha256": outputs["source-capture.jsonl"]["sha256"],
             "closure_sha256": outputs["capture.complete.json"]["sha256"],
             "prepared_sha256": outputs["prepared.jsonl"]["sha256"],
             "source_selection_manifest_sha256": fresh.SELECTION_HASHES["manifest.json"],
             "source_selection_sha256": fresh.SELECTION_HASHES["selection.jsonl"],
             "source_selection_proof_sha256": fresh.SELECTION_HASHES["independent-verification.json"],
             "reference_report_sha256": reference_sha,
             "producer_sha256": base.sha256_file(fresh.CAPTURE_PRODUCER),
             "verifier_sha256": fresh.CAPTURE_VERIFIER_SHA,
             "partition_counts": {"inner_hold": 1}, "counts": {"rows": 1},
             "model_pins": model_pins,
             "contracts": {"source_only_capture": True,
                 "capture_closed_before_labels": True,
                 "model_loaded_by_verifier": False,
                 "model_inference_by_verifier": False,
                 "dev_opened": False, "test_opened": False,
                 "production_changed": False}}
    proof_path = data / "proof.json"; proof_path.write_text(json.dumps(proof) + "\n")
    hashes = {"report.json": report_sha,
              "independent-verification.json": base.sha256_file(proof_path),
              "pins.json": base.sha256_file(data / "pins.json"),
              **{name: detail["sha256"] for name, detail in outputs.items()}}
    monkeypatch.setattr(fresh, "EXPECTED_CAPTURE_HASHES", hashes)
    authorized = fresh.validate_inputs(data, proof_path)
    assert len(authorized["source_rows"]) == 1


def test_self_test_is_model_free():
    assert fresh.self_test() == {"schema": fresh.SCHEMA, "self_test": True}
