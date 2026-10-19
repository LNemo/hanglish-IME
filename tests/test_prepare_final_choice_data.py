import copy
import hashlib
import json

import pytest

import scripts.prepare_final_choice_data as module


def natural(source_id, group, text):
    return {"id": source_id, "group_id": group, "text": text,
            "left_context": "문맥", "split": "train"}


def challenge(source_id, family):
    return {"id": source_id, "family": family, "input": "roman",
            "context": "문맥", "expected": ["로만"], "split": "train"}


def test_role_selection_preserves_exact_manifest_order(monkeypatch):
    rows = {"natural_raw_train": [natural("a", "g1", "가"),
                                   natural("b", "g2", "나")],
            "challenge_train": [challenge("c", "f1"), challenge("d", "f2")]}
    partition = {"natural": {"fit_ids": ["b", "a"]},
                 "challenge": {"fit_ids": ["d", "c"]}}
    monkeypatch.setitem(module.ROLE_SPECS, "natural_fit", ("natural", "fit_ids", 2))
    monkeypatch.setitem(module.ROLE_SPECS, "challenge_fit", ("challenge", "fit_ids", 2))
    selected = module.select_sources(rows, partition, "natural_fit")
    assert [row["id"] for row in selected] == ["b", "a"]
    assert selected[0] == {"id": "b", "group_id": "g2", "origin": "natural",
                           "input": "na", "context": "문맥", "expected": ["나"]}
    selected = module.select_sources(rows, partition, "challenge_fit")
    assert [row["id"] for row in selected] == ["d", "c"]
    assert selected[0]["group_id"] == "f2" and selected[0]["origin"] == "challenge"


def test_role_selection_rejects_count_and_source_drift(monkeypatch):
    rows = {"natural_raw_train": [natural("a", "g", "가")], "challenge_train": []}
    partition = {"natural": {"fit_ids": ["missing"]}}
    monkeypatch.setitem(module.ROLE_SPECS, "natural_fit", ("natural", "fit_ids", 1))
    with pytest.raises(ValueError, match="absent"):
        module.select_sources(rows, partition, "natural_fit")
    partition["natural"]["fit_ids"] = []
    with pytest.raises(ValueError, match="count"):
        module.select_sources(rows, partition, "natural_fit")


def test_runtime_evidence_requires_explicit_hash_and_complete_common_schema(tmp_path):
    report = {"passed": True, "integrated_runtime_version": "runtime",
              "integrated_runtime_source_hashes": {"source.py": "a" * 64},
              "runtime_source_hashes_after": {"source.py": "a" * 64},
              "accepted_weight_hashes": {f"weight-{i}": "b" * 64 for i in range(16)}}
    raw = json.dumps(report).encode()
    path = tmp_path / "runtime.json"
    path.write_bytes(raw)
    module.read_runtime_evidence(path, hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError, match="hash"):
        module.read_runtime_evidence(path, "0" * 64)
    report["accepted_weight_hashes"].pop("weight-0")
    bad = json.dumps(report).encode()
    path.write_bytes(bad)
    with pytest.raises(ValueError, match="integration schema"):
        module.read_runtime_evidence(path, hashlib.sha256(bad).hexdigest())


def test_partition_loader_rejects_input_snapshot_drift(tmp_path, monkeypatch):
    manifest = {"inputs": {"natural": {"sha256": "a" * 64}},
                "model_inference_run": False, "training_run": False,
                "development_data_read": False, "test_data_read": False,
                "deployed": False}
    raw = json.dumps(manifest).encode()
    path = tmp_path / "partition.json"
    path.write_bytes(raw)
    monkeypatch.setattr(module, "validate_partition", lambda rows, value: None)
    with pytest.raises(ValueError, match="snapshots"):
        module.read_partition(
            path, hashlib.sha256(raw).hexdigest(),
            input_loader=lambda: ({}, {"natural": {"sha256": "b" * 64}}),
        )


def test_partition_loader_propagates_structural_validation_failure(tmp_path, monkeypatch):
    snapshots = {"natural": {"sha256": "a" * 64}}
    manifest = {"inputs": snapshots, "model_inference_run": False,
                "training_run": False, "development_data_read": False,
                "test_data_read": False, "deployed": False}
    raw = json.dumps(manifest).encode()
    path = tmp_path / "partition.json"
    path.write_bytes(raw)
    monkeypatch.setattr(
        module, "validate_partition",
        lambda rows, value: (_ for _ in ()).throw(ValueError("invalid partition")),
    )
    with pytest.raises(ValueError, match="invalid partition"):
        module.read_partition(
            path, hashlib.sha256(raw).hexdigest(),
            input_loader=lambda: ({}, snapshots),
        )


def captured(source):
    candidate = {"text": "로만", "score": 1.0, "segments": []}
    return {**source, "result": "로만", "candidates": [candidate],
            "label_aligned": True, "selector_eligible": True,
            "all_candidates_can_score": True}


def test_failed_capture_leaves_no_completed_manifest(tmp_path):
    sources = [{"id": "one", "input": "roman", "context": "", "expected": ["로만"]},
               {"id": "two", "input": "roman", "context": "", "expected": ["로만"]}]
    calls = []

    def fail_second(converter, source, **kwargs):
        calls.append(source["id"])
        if len(calls) == 2:
            raise RuntimeError("capture failed")
        return captured(source)

    with pytest.raises(RuntimeError, match="capture failed"):
        module.capture_to_directory(
            tmp_path / "role", object(), sources, runtime_version="runtime",
            model_artifacts={}, manifest_base={"role": "fixture"}, capture=fail_second,
        )
    assert calls == ["one", "two"]
    assert (tmp_path / "role/cases.jsonl").exists()
    assert not (tmp_path / "role/manifest.json").exists()


def test_prevalidation_can_fail_before_converter_creation(monkeypatch, tmp_path):
    class Bomb:
        def __init__(self, *args, **kwargs):
            raise AssertionError("converter was created before gates")

    monkeypatch.setattr(module, "NeuralConverter", Bomb)
    monkeypatch.setattr(module, "read_partition",
                        lambda: (_ for _ in ()).throw(ValueError("partition gate")))
    args = type("Args", (), {"output": tmp_path, "role": "natural_fit",
                             "runtime_evidence": tmp_path / "missing",
                             "runtime_evidence_sha256": "0" * 64})
    with pytest.raises(Exception) as error:
        module.run(args)
    assert not isinstance(error.value, AssertionError)
