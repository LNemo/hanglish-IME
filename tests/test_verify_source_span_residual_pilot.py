import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "source_span_residual_verifier",
    ROOT / ".omx/artifacts/verify-source-span-residual-pilot-v1.py",
)
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def cohort():
    return [{
        "id": f"row-{index}", "input": f"source-{index}",
        "component_id": f"component-{index}", "candidates": ["정답", "오답"],
        "positive_indices": [0], "multiple_paths": False,
        "unresolved_ascii": False, "stratum": "hash",
    } for index in range(16)]


def test_multi_positive_loss_is_stable_and_matches_probability_mass():
    scores = [1001.0, 1000.0, 999.0]
    actual = verifier.listwise_loss(scores, [0, 2])
    expected = -math.log((math.exp(1) + math.exp(-1))
                         / (math.exp(1) + 1 + math.exp(-1)))
    assert actual == pytest.approx(expected)


def test_fixed_cohort_uses_hash_fill_when_multiple_path_stratum_is_empty():
    rows = [{
        "id": f"row-{index:02d}", "input": "x" * (index + 1),
        "component_id": f"component-{index:02d}", "candidates": ["가", "나"],
        "positive_indices": [0], "multiple_paths": False,
        "unresolved_ascii": index < 4,
    } for index in range(24)]
    selected = verifier.select_cohort(rows)
    assert len(selected) == len({row["component_id"] for row in selected}) == 16
    assert sum(row["stratum"] == "multiple_paths" for row in selected) == 0
    assert sum(row["stratum"] == "unresolved_ascii" for row in selected) == 4
    assert sum(row["stratum"] == "hash_fill" for row in selected) == 2


def test_compact_path_identity_and_unresolved_ascii_reject_tampering():
    source = "testreul"
    edges = [{"input_start": 0, "input_end": 4, "text": "test"},
             {"input_start": 4, "input_end": 8, "text": "를"}]
    path_id = verifier.compact_path_id(source, edges)
    record = {
        "schema": "source-correspondence-v1", "id": "one", "input": source,
        "component_id": "component", "edges": edges,
        "paths": [{"path_id": path_id, "text": "test를", "edge_indices": [0, 1]}],
        "reference_path_id": path_id,
    }
    verifier.validate_compact_record(record)
    assert verifier.path_has_unresolved_ascii(record, record["paths"][0]) is False
    tampered = json.loads(json.dumps(record))
    tampered["paths"][0]["text"] = "injected"
    with pytest.raises(ValueError, match="identity"):
        verifier.validate_compact_record(tampered)


def make_scores(run, rows):
    before, after, reloaded = [], [], []
    expected_after_loss = verifier.listwise_loss([3.0, 1.0], [0])
    for row in rows:
        before.append({"id": row["id"], "scores": [2.0, 1.0],
                       "parent_reference_scores": [2.0, 1.0],
                       "token_spans": [[[0, 2]], [[0, 2]]]})
        after.append({"id": row["id"], "scores": [3.0, 1.0],
                      "residuals": [1.0, 0.0], "loss": expected_after_loss})
        reloaded.append({"id": row["id"], "scores": [3.0, 1.0]})
    dump(run / "before.json", before)
    dump(run / "after.json", after)
    dump(run / "reloaded.json", reloaded)
    return before


def test_score_replay_recomputes_loss_and_rejects_residual_corruption(tmp_path):
    rows = cohort()
    make_scores(tmp_path, rows)
    metrics, _ = verifier.verify_scores(tmp_path, rows)
    assert metrics["before_correct"] == metrics["after_correct"] == 16
    damaged = json.loads((tmp_path / "after.json").read_text())
    damaged[3]["residuals"][0] = 9.0
    dump(tmp_path / "after.json", damaged)
    with pytest.raises(ValueError, match="addition"):
        verifier.verify_scores(tmp_path, rows)


def make_updates(run, rows):
    records = []
    for index in range(128):
        value = 0.0 if index == 0 else 0.25
        records.append({
            "update": index + 1, "epoch": index // 16, "id": rows[index % 16]["id"],
            "loss": 0.5, "gradient_norm": value,
            "gradient_group_norms": {name: value for name in verifier.GRADIENT_GROUPS},
        })
    (run / "updates.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    return records


def test_update_replay_checks_exact_order_and_post_first_gradient_flow(tmp_path):
    rows = cohort()
    records = make_updates(tmp_path, rows)
    assert all(verifier.verify_updates(tmp_path, rows).values())
    records[17]["id"] = "wrong-row"
    (tmp_path / "updates.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    with pytest.raises(ValueError, match="order"):
        verifier.verify_updates(tmp_path, rows)


def resource_row(operation, stage, elapsed, operation_elapsed):
    return {"at": "2026-09-21T00:00:00+00:00", "elapsed_s": elapsed,
            "rss_bytes": 1000, "swap_used_bytes": 2000, "swap_growth_bytes": 0,
            "operation_elapsed_s": operation_elapsed, "stage": stage,
            "operation": operation}


def make_resources(run, rows):
    dump(run / "cohort.json", rows)
    names = (["load_parent_and_residual"]
             + ["features:" + row["id"] for row in rows]
             + [f"update:{index}" for index in range(1, 129)]
             + ["after:" + row["id"] for row in rows]
             + ["save_reload"])
    trace, elapsed = [], 0.0
    for name in names:
        trace.append(resource_row(name, "operation_start", elapsed, 0.0))
        elapsed += 0.01
        trace.append(resource_row(name, "operation_end", elapsed, 0.01))
        elapsed += 0.01
    (run / "resources.jsonl").write_text("".join(json.dumps(row) + "\n" for row in trace))
    report = {"elapsed_s": elapsed, "baseline_swap_bytes": 2000,
              "final_resources": resource_row(None, "final", elapsed, None)}
    return trace, report


def test_resource_verification_requires_all_162_operation_pairs_and_bounds(tmp_path):
    trace, report = make_resources(tmp_path, cohort())
    result = verifier.verify_resources(tmp_path, report)
    assert result["operation_pairs"] == 162
    trace[-1]["operation_elapsed_s"] = 61.0
    (tmp_path / "resources.jsonl").write_text("".join(json.dumps(row) + "\n" for row in trace))
    with pytest.raises(ValueError, match="time hard bound"):
        verifier.verify_resources(tmp_path, report)


def test_tensor_inspection_binds_shapes_hashes_features_and_parent(tmp_path, monkeypatch):
    rows = cohort()
    before = make_scores(tmp_path, rows)
    residual = {name: np.ones(shape, dtype=np.float32)
                for name, shape in verifier.RESIDUAL_SHAPES.items()}
    save_file(residual, tmp_path / "residual.safetensors")
    features = {}
    for index in range(16):
        features[f"row{index}.logits"] = np.array([2.0, 1.0], dtype=np.float32)
        features[f"row{index}.hidden"] = np.zeros((2, 1, 256), dtype=np.float32)
    save_file(features, tmp_path / "features.safetensors")
    parent = tmp_path / "parent"
    parent.mkdir()
    parent_tensor = {"weight": np.array([1.0, 2.0], dtype=np.float32)}
    save_file(parent_tensor, parent / "model.safetensors")
    monkeypatch.setattr(verifier, "PARENT", parent)

    after_hashes = {name: verifier.tensor_digest(value) for name, value in residual.items()}
    before_hashes = {name: hashlib_of_zeros(value.size * 4)
                     for name, value in residual.items()}
    parent_hash = {"weight": verifier.tensor_digest(parent_tensor["weight"])}
    report = {
        "trainable_parameters": 71073, "residual_before": before_hashes,
        "residual_after": after_hashes,
        "changed_residual_names": sorted(residual),
        "parent_before": parent_hash, "parent_after": parent_hash,
    }
    result = verifier.verify_safetensors(tmp_path, rows, before, report)
    assert result["residual_tensors"] == 13 and result["parent_tensors"] == 1

    damaged = dict(residual)
    damaged["segment_head.2.weight"] = np.ones((2, 64), dtype=np.float32)
    save_file(damaged, tmp_path / "residual.safetensors")
    with pytest.raises(ValueError, match="shape"):
        verifier.verify_safetensors(tmp_path, rows, before, report)


def test_resource_boundary_race_is_disclosed_but_unrelated_operation_rejected(tmp_path):
    trace, report = make_resources(tmp_path, cohort())
    upcoming = trace[2]["operation"]
    trace.insert(2, resource_row(upcoming, "watchdog", .015, .015))
    path = tmp_path / "resources.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in trace))
    result = verifier.verify_resources(tmp_path, report)
    assert result["non_atomic_boundary_samples"] == [upcoming]
    assert result["operation_timer_attribution_fully_verified"] is False
    trace[2]["operation"] = "update:128"
    path.write_text("".join(json.dumps(row) + "\n" for row in trace))
    with pytest.raises(ValueError, match="outside an operation"):
        verifier.verify_resources(tmp_path, report)


def hashlib_of_zeros(size):
    import hashlib
    return hashlib.sha256(b"\0" * size).hexdigest()
