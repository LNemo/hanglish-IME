import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file


ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = ROOT / ".omx/artifacts/verify-source-span-pair-v1.py"
SPEC = importlib.util.spec_from_file_location("source_span_pair_verifier", VERIFIER_PATH)
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def dump_jsonl(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values))


def rows(count=3):
    return [{"id": f"row-{index}", "input": f"source-{index}",
             "component_id": f"component-{index}",
             "candidates": ["정답", "오답"], "positive_indices": [0]}
            for index in range(count)]


def test_verifier_has_no_torch_tokenizer_or_project_model_imports():
    tree = ast.parse(VERIFIER_PATH.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not any(name == "torch" or name.startswith("transformers")
                   or name.startswith("hanglish") or name.startswith("scripts")
                   for name in imported)


def test_source_path_validation_reconstructs_identity_and_rejects_tampering():
    source = "testreul"
    edges = [{"input_start": 0, "input_end": 4, "text": "test"},
             {"input_start": 4, "input_end": 8, "text": "를"}]
    path_id = verifier.compact_path_id(source, edges)
    record = {"schema": "source-correspondence-v1", "id": "one", "input": source,
              "component_id": "component", "edges": edges,
              "paths": [{"path_id": path_id, "text": "test를", "edge_indices": [0, 1]}],
              "reference_path_id": path_id}
    verifier.validate_source_record(record)
    record["paths"][0]["text"] = "injected"
    with pytest.raises(ValueError, match="identity"):
        verifier.validate_source_record(record)


def test_update_replay_requires_both_epochs_exact_order_and_gradient_flow(tmp_path):
    selected = rows()
    arm = "whole_source"
    updates = []
    for index in range(len(selected) * verifier.EPOCHS):
        value = 0.0 if index == 0 else 0.25
        updates.append({"update": index + 1, "epoch": index // len(selected),
                        "id": selected[index % len(selected)]["id"], "loss": 0.5,
                        "gradient_norm": value,
                        "gradient_groups": {name: value for name in verifier.GRADIENT_GROUPS}})
    dump_jsonl(tmp_path / arm / "updates.jsonl", updates)
    result = verifier.verify_updates(tmp_path, arm, selected)
    assert result["updates"] == 6
    assert result["epoch_mean_loss"] == [0.5, 0.5]
    updates[4]["id"] = "wrong"
    dump_jsonl(tmp_path / arm / "updates.jsonl", updates)
    with pytest.raises(ValueError, match="order"):
        verifier.verify_updates(tmp_path, arm, selected)


def test_before_and_after_replay_loss_argmax_addition_and_reload(tmp_path, monkeypatch):
    selected = rows()
    before = [{"id": row["id"], "scores": [2.0, 1.0],
               "parent_reference_scores": [2.0, 1.0],
               "canonical_token_input_equal": True,
               "loss": verifier.listwise_loss([2.0, 1.0], [0]), "correct": 1}
              for row in selected]
    dump_jsonl(tmp_path / "before.jsonl", before)
    loaded, metrics = verifier.verify_before(tmp_path, selected)
    assert metrics["correct"] == 3
    arm = "source_local"
    trained = [{"id": row["id"], "scores": [3.0, 1.0], "residuals": [1.0, 0.0],
                "loss": verifier.listwise_loss([3.0, 1.0], [0]), "correct": 1}
               for row in selected]
    dump_jsonl(tmp_path / arm / "after.jsonl", trained)
    monkeypatch.setattr(verifier, "RELOAD_INDICES", [0, 2])
    dump(tmp_path / arm / "reload.json", [
        {"id": selected[index]["id"], "scores": [3.0, 1.0], "reloaded": [3.0, 1.0]}
        for index in (0, 2)])
    assert verifier.verify_after(tmp_path, arm, selected, loaded)["reload_rows"] == 2
    trained[1]["residuals"][0] = 9.0
    dump_jsonl(tmp_path / arm / "after.jsonl", trained)
    with pytest.raises(ValueError, match="base plus residual"):
        verifier.verify_after(tmp_path, arm, selected, loaded)


def resource_row(operation, stage, elapsed, baseline=2000):
    active = operation is not None
    return {"at": "2026-09-21T00:00:00+00:00", "elapsed_s": elapsed,
            "rss_bytes": 1000, "swap_used_bytes": baseline,
            "swap_growth_bytes": 0, "operation_elapsed_s": elapsed if active else None,
            "stage": stage, "operation": operation}


def test_atomic_resource_replay_requires_exact_operation_sequence(tmp_path):
    selected = rows(1)
    names = verifier.expected_operations(selected)
    trace, elapsed = [], 0.0
    for name in names:
        trace.append(resource_row(name, "operation_start", elapsed))
        elapsed += 0.001
        trace.append(resource_row(name, "watchdog", elapsed))
        elapsed += 0.001
        trace.append(resource_row(name, "operation_end", elapsed))
        elapsed += 0.001
        trace.append(resource_row(None, "watchdog", elapsed))
        elapsed += 0.001
    dump_jsonl(tmp_path / "resources.jsonl", trace)
    report = {"baseline_swap_bytes": 2000, "elapsed_s": elapsed,
              "final_resources": resource_row(None, "final", elapsed)}
    result = verifier.verify_resources(tmp_path, report, selected)
    assert result["operation_pairs"] == len(names)
    assert result["atomic_operation_attribution_verified"] is True
    trace[2]["operation"] = names[1]
    dump_jsonl(tmp_path / "resources.jsonl", trace)
    with pytest.raises(ValueError, match="pairing"):
        verifier.verify_resources(tmp_path, report, selected)


def test_fp32_replay_handles_large_common_offsets_without_hiding_score_corruption():
    base = verifier.fp32(.3)
    residual = verifier.fp32(80.)
    score = verifier.fp32(base + residual)
    assert score != base + residual
    assert score == float(np.float32(base) + np.float32(residual))
    assert score + 1e-6 != verifier.fp32(base + residual)
    scores = np.array([80., 79.], dtype=np.float32)
    lse = np.float32(scores.max() + np.log(np.exp(scores - scores.max()).sum()))
    loss = float(lse - scores[0])
    exact = verifier.listwise_loss(scores.tolist(), [0])
    assert abs(loss - exact) < verifier.loss_tolerance(scores.tolist())
    assert abs(loss + .001 - exact) > verifier.loss_tolerance(scores.tolist())


def test_residual_tensor_validation_checks_names_shapes_finiteness_and_hashes(tmp_path):
    tensors = {name: np.zeros(shape, dtype=np.float32)
               for name, shape in verifier.RESIDUAL_SHAPES.items()}
    path = tmp_path / "residual.safetensors"
    save_file(tensors, path)
    hashes = verifier.validate_tensor_file(path)
    assert len(hashes) == 13
    damaged = dict(tensors)
    damaged["segment_head.2.weight"] = np.zeros((2, 64), dtype=np.float32)
    save_file(damaged, path)
    with pytest.raises(ValueError, match="shape"):
        verifier.validate_tensor_file(path)


def test_feature_manifest_binds_metadata_tensor_shape_logits_and_bytes(tmp_path, monkeypatch):
    selected = rows(2)
    before = [{"scores": [2.0, 1.0]}, {"scores": [2.0, 1.0]}]
    manifest = {}
    for index, row in enumerate(selected):
        tensor_path = tmp_path / "features" / f"{index:04d}.safetensors"
        metadata_path = tmp_path / "features" / f"{index:04d}.json"
        tensor_path.parent.mkdir(exist_ok=True)
        save_file({"logits": np.array([2.0, 1.0], dtype=np.float32),
                   "hidden": np.zeros((2, 3, 256), dtype=np.float32)}, tensor_path)
        metadata = {"candidates": row["candidates"],
                    "token_spans": [[[None][0], [0, 1], None],
                                    [None, [0, 1], None]],
                    "tensors_sha256": verifier.sha(tensor_path)}
        dump(metadata_path, metadata)
        manifest[str(index)] = {"tensors_sha256": verifier.sha(tensor_path),
                                "metadata_sha256": verifier.sha(metadata_path)}
    dump(tmp_path / "feature-manifest.json", manifest)
    result = verifier.verify_features(tmp_path, selected, before)
    assert result["files"] == 4 and result["rows"] == 2
    manifest["0"]["metadata_sha256"] = "0" * 64
    dump(tmp_path / "feature-manifest.json", manifest)
    with pytest.raises(ValueError, match="manifest hash"):
        verifier.verify_features(tmp_path, selected, before)
