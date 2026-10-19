import ast
from copy import deepcopy
from pathlib import Path

import pytest

from scripts import verify_current_phonetic_train_pool as verifier


def _copy_segment(source="abc"):
    return {"input_start": 0, "input_end": len(source), "source": source,
            "text": source, "kind": "raw"}


def _ranked_record():
    source = "abc"
    segment = _copy_segment(source)
    paths = [[source, 1.0, [segment]]]
    artifacts = {"adapter.py": "a" * 64, "features.py": "b" * 64,
                 "likelihood.py": "c" * 64}
    models = {"model.bin": "d" * 64, **artifacts}
    version = verifier.expected_version(artifacts)
    candidate = {"text": source, "score": 1.0, "segments": [segment]}
    response = {
        "input": source, "context": "", "external_context_used": False,
        "version": version, "model_artifacts": models,
        "result": source, "candidates": [candidate],
        "sentence_ranking_applied": True,
        "phonetic_word_union": {"union": [source]},
        "phonetic_choice": {
            "parent_version": verifier.PARENT_VERSION, "version": version,
            "artifacts": artifacts, "effective_weights": [0.0] * 4,
            "calibration_source": "override", "score_records": [],
            "rows": [{"text": source, "base_score": 1.0,
                      "features": [0.0] * 4, "residual": 0,
                      "adjusted_score": 1.0, "segments": [segment],
                      "edges": [{"input_start": 0, "input_end": 3,
                                 "source": source, "target": source,
                                 "status": "copy"}]}],
            "supported_pair_count": 0, "edge_status_counts": {"copy": 1},
            "zero_weight_parity": True,
            "zero_weight_score_parity": True,
            "zero_weight_order_parity": True,
            "public": [source],
        },
    }
    expected = {"id": "s3-train-00001", "input": source,
                "component_id": "component-1"}
    record = {"id": expected["id"], "component_id": expected["component_id"],
              "request": {"input": source, "context": ""},
              "response": response, "base_paths": paths, "elapsed_s": 0.1}
    return expected, record, version, models, artifacts


def test_verifier_has_no_producer_tensor_or_neural_imports():
    tree = ast.parse(Path(verifier.__file__).read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    forbidden = ("current_phonetic_train_pool", "torch", "phonetic_likelihood",
                 "phonetic_choice_converter")
    assert not any(item in name for name in imports for item in forbidden)


def test_current_frozen_artifact_oracle_and_zero_version():
    models, choice = verifier.expected_artifacts()
    assert len(models) == 32
    assert len(choice) == 3
    assert verifier.OLD_CHECKPOINT not in models
    assert verifier.OLD_CHECKPOINT not in choice
    assert verifier.expected_version(choice) == \
        "experimental-phonetic-choice-1-2bbe1463904b"


def test_partition_keeps_components_together_and_is_deterministic():
    rows = [
        {"id": "s3-train-1", "input": "aaaa", "component_id": "c1"},
        {"id": "s3-train-2", "input": "bb", "component_id": "c2"},
        {"id": "s3-train-3", "input": "a", "component_id": "c1"},
        {"id": "s3-train-4", "input": "ccc", "component_id": "c3"},
    ]
    shards = verifier.partition(rows, 2)
    assert shards == verifier.partition(deepcopy(rows), 2)
    assert {next(i for i, shard in enumerate(shards) if row["id"] in shard)
            for row in rows if row["component_id"] == "c1"} == {0}
    assert sorted(item for shard in shards for item in shard) == \
        sorted(row["id"] for row in rows)


def test_probe_groups_cover_every_row_once_and_choose_longest():
    rows = [{"id": f"s3-train-{index:05d}", "input": "a" * (index + 1),
             "component_id": f"c{index}"} for index in range(32)]
    groups = verifier.probe_groups(rows)
    assert len(groups) == 16
    assert [group["count"] for group in groups] == [2] * 16
    assert [group["maximum_source_length"] for group in groups] == \
        list(range(2, 33, 2))
    assert sorted(item for group in groups for item in group["ids"]) == \
        sorted(row["id"] for row in rows)


def test_projection_replays_all_six_shards_without_requiring_qualification():
    rows = [{"id": f"s3-train-{index:05d}", "input": "a" * (index + 1),
             "component_id": f"c{index}"} for index in range(16)]
    groups = verifier.probe_groups(rows)
    shards = [[row["id"]] for row in rows[:6]]
    timings = {group["probe_id"]: 2000.0 for group in groups}
    projected = verifier.projected_seconds(groups, shards, timings, 2.0)
    assert len(projected) == 6
    assert all(value == 1.3 * 2002.0 + 60.0 for value in projected)
    assert not all(value <= verifier.LIMITS["whole_seconds"] for value in projected)


def test_exact_probe_and_capture_launch_commands():
    probe = verifier.PROBE.resolve()
    assert verifier.expected_launch_command(probe, "probe") == [
        str(verifier.ROOT / ".venv/bin/python"), "-m",
        "scripts.current_phonetic_train_pool", "--mode", "probe",
        "--output", str(probe),
    ]
    capture = (verifier.ROOT /
               ".omx/artifacts/current-phonetic-train-shard-3-v1").resolve()
    assert verifier.expected_launch_command(capture, "capture", 3)[-2:] == \
        ["--shard", "3"]


@pytest.mark.parametrize(
    "mode,shard",
    [("prepare", None), ("capture", None), ("probe", 0),
     ("capture", -1), ("capture", 6), ("capture", True)],
)
def test_launch_command_rejects_invalid_mode_or_shard(mode, shard):
    with pytest.raises(ValueError):
        verifier.expected_launch_command(verifier.PROBE, mode, shard)


def test_independent_zero_record_replay_accepts_exact_record():
    expected, record, version, models, artifacts = _ranked_record()
    counts = verifier.verify_record(expected, record, version, models, artifacts)
    assert counts == {"rows": 1, "ranked_rows": 1, "paths": 1,
                      "pairs": 0, "supported_edges": 0}


def test_independent_zero_record_rejects_forged_prefixed_version():
    expected, record, version, models, artifacts = _ranked_record()
    record["response"]["version"] = version + "forged"
    record["response"]["phonetic_choice"]["version"] = version + "forged"
    with pytest.raises(ValueError, match="runtime identity"):
        verifier.verify_record(expected, record, version, models, artifacts)


def test_independent_zero_record_rejects_feature_tampering():
    expected, record, version, models, artifacts = _ranked_record()
    record["response"]["phonetic_choice"]["rows"][0]["features"][1] = 0.25
    with pytest.raises(ValueError, match="feature vector"):
        verifier.verify_record(expected, record, version, models, artifacts)


def test_independent_zero_record_rejects_missing_model_artifact():
    expected, record, version, models, artifacts = _ranked_record()
    expected_models = deepcopy(models)
    record["response"]["model_artifacts"].pop("model.bin")
    with pytest.raises(ValueError, match="runtime identity"):
        verifier.verify_record(expected, record, version, expected_models, artifacts)


def test_prepared_proof_is_hash_linked_and_current():
    proof = verifier.read(verifier.PREPARED / "independent-verification.json")
    assert proof["verified"] is True
    assert proof["manifest_sha256"] == verifier.sha(
        verifier.PREPARED / "manifest.json")
    assert proof["requests_sha256"] == verifier.sha(
        verifier.PREPARED / "requests.json")
    assert proof["verifier_sha256"] == verifier.sha(Path(verifier.__file__))
