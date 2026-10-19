from copy import deepcopy
import hashlib
import json

import pytest

from scripts import prepare_difference_pairwise_lora_fresh as bridge


WEIGHTS = {"test-model": "0" * 64}


@pytest.fixture
def cohort():
    sources, captures = [], []
    for index in range(888):
        source = {"id": f"r{index}", "group_id": f"g{index % 387}" if index < 512 else f"p{index}",
                  "origin": "synthetic", "input": "imnida", "context": ""}
        candidate = {"text": "입니다", "score": 1.0, "segments": [{
            "input_start": 0, "input_end": 6, "source": "imnida", "text": "입니다", "kind": "lexicon"}]}
        parent = {"input": source["input"], "context": "", "version": bridge.VERSION,
                  "raw": False, "result": "입니다", "candidates": [candidate], "model_artifacts": WEIGHTS}
        raw = {"input": source["input"], "context": "", "version": bridge.VERSION,
               "raw": True, "result": "imnida", "candidates": [{"text": "imnida"}]}
        sources.append(source)
        captures.append({**source, "parent": parent, "parent_raw": raw,
                         "child": {"result": "must never be copied"}, "expected": ["secret"]})
    selected = [{"id": r["id"], "group_id": r["group_id"]} for r in sources[:512]]
    return sources, captures, selected


def test_complete_projection_preserves_objects_and_roles(cohort):
    sources, captures, selection = cohort
    rows = bridge.project_parent_rows(*cohort, WEIGHTS)
    assert len(rows) == 888
    assert [r["role"] for r in rows] == ["fresh_natural"] * 512 + ["challenge_preservation"] * 376
    for output, source, capture in zip(rows, sources, captures):
        assert all(output[key] == source[key] for key in bridge.SOURCE_FIELDS)
        assert output["parent_response"] is capture["parent"]
        assert output["candidates"] is capture["parent"]["candidates"]
        assert output["raw_bypass_response"] is capture["parent_raw"]
        assert "expected" not in output and "child" not in output


def test_projection_never_reads_targets_or_child(cohort):
    class SourceOnly(dict):
        def __getitem__(self, key):
            assert key in (*bridge.SOURCE_FIELDS, "parent", "parent_raw")
            return super().__getitem__(key)

        def __iter__(self):
            raise AssertionError("must not copy arbitrary input fields")

        def keys(self):
            raise AssertionError("must not copy arbitrary input fields")

    sources, captures, selection = cohort
    rows = bridge.project_parent_rows([SourceOnly(r) for r in sources],
                                     [SourceOnly(r) for r in captures], selection, WEIGHTS)
    assert len(rows) == 888


@pytest.mark.parametrize("change", ["missing", "duplicate", "order", "group", "version", "raw",
                                    "weights", "source", "result", "nan"])
def test_corrupt_or_incomplete_parent_evidence_rejected(cohort, change):
    sources, captures, selection = cohort
    if change == "missing":
        captures.pop()
    elif change == "duplicate":
        sources[-1] = deepcopy(sources[-2]); captures[-1] = deepcopy(captures[-2])
    elif change == "order":
        selection.reverse()
    elif change == "group":
        sources[0]["group_id"] = "g386"; selection[0]["group_id"] = "g386"
    elif change == "version":
        captures[0]["parent"]["version"] = "different"
    elif change == "raw":
        captures[0]["parent_raw"]["raw"] = False
    elif change == "weights":
        captures[0]["parent"]["model_artifacts"] = {}
    elif change == "source":
        captures[0]["context"] = "changed"
    elif change == "result":
        captures[0]["parent"]["result"] = "other"
    elif change == "nan":
        captures[0]["parent"]["candidates"][0]["score"] = float("nan")
    with pytest.raises(ValueError):
        bridge.project_parent_rows(sources, captures, selection, WEIGHTS)


def test_training_gate_precedes_parent_io_and_output_creation(monkeypatch, tmp_path):
    def reject():
        raise ValueError("TRAIN rejected")

    def forbidden(*args):
        raise AssertionError("parent files opened before TRAIN success")

    monkeypatch.setattr(bridge, "require_training_proof", reject)
    monkeypatch.setattr(bridge, "verify_pins", forbidden)
    output = tmp_path / "not-created"
    with pytest.raises(ValueError, match="TRAIN rejected"):
        bridge.prepare(output)
    assert not output.exists()


def test_training_proof_requires_independent_replay(monkeypatch, tmp_path):
    from scripts import verify_difference_pairwise_lora_training as verifier

    path = tmp_path / "proof.json"
    payload = {"verification_passed": True, "quality_gate_passed": True,
               "downstream_evaluation_allowed": True, "adapter": {}, "logical_fit_id": "test"}
    path.write_text(json.dumps({"schema": verifier.SCHEMA, "payload": payload,
                               "digest": hashlib.sha256(bridge.canonical(payload)).hexdigest()}))
    monkeypatch.setattr(bridge, "TRAIN_PROOF", path)
    monkeypatch.setattr(verifier, "verify_training", lambda: {**payload, "quality_gate_passed": False})
    with pytest.raises(ValueError, match="replay differs"):
        bridge.require_training_proof()


def test_rejected_training_proof_stops_before_replay(monkeypatch, tmp_path):
    from scripts import verify_difference_pairwise_lora_training as verifier

    path = tmp_path / "proof.json"
    payload = {"verification_passed": True, "quality_gate_passed": False,
               "downstream_evaluation_allowed": False}
    path.write_text(json.dumps({"schema": verifier.SCHEMA, "payload": payload,
                               "digest": hashlib.sha256(bridge.canonical(payload)).hexdigest()}))
    monkeypatch.setattr(bridge, "TRAIN_PROOF", path)

    def forbidden():
        raise AssertionError("rejected quality proof cannot proceed")

    monkeypatch.setattr(verifier, "verify_training", forbidden)
    with pytest.raises(ValueError, match="accepted pairwise TRAIN"):
        bridge.require_training_proof()


def test_prepare_writes_exact_source_projection_once(monkeypatch, tmp_path, cohort):
    from scripts import verify_difference_pairwise_lora_training as verifier

    sources, captures, selection = cohort
    monkeypatch.setattr(bridge, "ROOT", tmp_path)
    proof_path = tmp_path / "train-proof.json"
    payload = {"verification_passed": True, "quality_gate_passed": True,
               "downstream_evaluation_allowed": True, "adapter": {}, "logical_fit_id": "test"}
    proof_path.write_bytes(bridge.canonical({"schema": verifier.SCHEMA, "payload": payload,
        "digest": hashlib.sha256(bridge.canonical(payload)).hexdigest()}))
    monkeypatch.setattr(bridge, "TRAIN_PROOF", proof_path)
    training = {"path": "train-proof.json", "sha256": bridge.sha(proof_path),
                "adapter": {}, "logical_fit_id": "test"}
    events = []

    def replay():
        events.append("independent_replay")
        return deepcopy(payload)

    original_pins, original_rows = bridge.verify_pins, bridge.read_jsonl

    def verify_after_replay(pins):
        assert events and events[0] == "independent_replay"
        events.append("parent_pins")
        return original_pins(pins)

    def read_after_replay(path):
        assert events and events[0] == "independent_replay"
        events.append("parent_rows")
        return original_rows(path)

    monkeypatch.setattr(verifier, "verify_training", replay)
    monkeypatch.setattr(bridge, "verify_pins", verify_after_replay)
    monkeypatch.setattr(bridge, "read_jsonl", read_after_replay)
    (tmp_path / "test-model").write_bytes(b"synthetic weights")
    weights = {"test-model": bridge.sha(tmp_path / "test-model")}
    for capture in captures:
        capture["parent"]["model_artifacts"] = weights
    proof = {"payload": {"verification_passed": True, "runtime_version": bridge.VERSION,
                         "model_artifacts": weights}}
    runtime = {"integrated_runtime_version": bridge.VERSION, "accepted_weight_hashes": weights,
               "integrated_runtime_source_hashes": {}}
    values = {
        bridge.PARENT + "-proof.json": proof,
        bridge.PARENT + "/report.json": {},
        bridge.PARENT + "/sources.jsonl": sources,
        bridge.PARENT + "/captures.jsonl": captures,
        ".omx/artifacts/long-context-lora-data-v1-review2/fresh_selection.jsonl": selection,
        ".omx/artifacts/numeric-structure-integration-v1/report.json": runtime,
    }
    for name, value in values.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"".join(bridge.canonical(r) + b"\n" for r in value)
                         if name.endswith(".jsonl") else bridge.canonical(value))
    monkeypatch.setattr(bridge, "PINS", {name: bridge.sha(tmp_path / name) for name in values})
    # The source receipt must refer to the actual executing file, even in a
    # synthetic workspace. Copy only this helper; no real evaluation data loads.
    own_path = tmp_path / "scripts/bridge.py"
    own_path.parent.mkdir()
    own_path.write_text("synthetic source receipt")
    monkeypatch.setattr(bridge, "__file__", str(own_path))
    output = tmp_path / "projection"
    manifest = bridge.prepare(output)
    assert events[:2] == ["independent_replay", "parent_pins"]
    assert events.count("parent_rows") == 3
    expected = bridge.project_parent_rows(sources, captures, selection, weights)
    assert bridge.read_jsonl(output / "inputs.jsonl") == expected
    assert manifest["inputs_sha256"] == bridge.sha(output / "inputs.jsonl")
    assert manifest["training"] == training
    assert manifest["inference_authorized"] is False
    assert manifest["target_files_read"] is False
    assert manifest["historical_targets_exposed_in_numeric_lane"] is True
    assert json.loads((output / "manifest.json").read_text()) == manifest
    with pytest.raises(ValueError, match="already exists"):
        bridge.prepare(output)
