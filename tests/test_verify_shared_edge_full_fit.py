from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_shared_edge_full_fit.py"
SPEC = importlib.util.spec_from_file_location("verify_shared_edge_full_fit", SCRIPT)
verifier = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(verifier)
PURE = verifier.helper()


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True))


def write_jsonl(path, values):
    path.write_text("".join(json.dumps(value, sort_keys=True) + "\n" for value in values))


def fixture():
    row = {"id": "row", "input": "a", "component_id": "component",
        "candidates": ["가", "나"], "positive_indices": [0], "selected_indices": [0, 1],
        "base_scores": [verifier.f32(.8), verifier.f32(.1)],
        "candidate_path_counts": [1, 1], "selected_path_expansion_count": 0}
    inventory = {"edges": [(0, 1, "가"), (0, 1, "나")], "groups": [
        [{"path_id": "path-a", "edge_indices": [0], "weight_numerators": [1],
          "weight_denominator": 1}],
        [{"path_id": "path-b", "edge_indices": [1], "weight_numerators": [1],
          "weight_denominator": 1}]]}
    residuals = [verifier.f32(.2), verifier.f32(.3)]
    scores = [verifier.f32(base + residual)
              for base, residual in zip(row["base_scores"], residuals)]
    historical = {"correct": 1, "scores": scores, "base_scores": row["base_scores"],
                  "residuals": residuals}
    counts = verifier.row_counts(row, scores, {"가", "나"}, inventory)
    counts.update({"historical_selected_correct": 1,
                   "historical_correctness_disagreement": 0})
    saved = {"id": "row", "scores": scores, "residuals": residuals,
        "edge_ids": [[0, 1, "가"], [0, 1, "나"]], "edge_scores": [.2, .3],
        "winning_path_ids": [["path-a"], ["path-b"]], "counts": counts,
        "historical_selected_score_max_abs_drift": 0.,
        "historical_selected_base_max_abs_drift": 0.,
        "historical_selected_residual_max_abs_drift": 0.}
    return saved, row, inventory, historical


def test_prediction_replays_edges_paths_fp32_scores_counts_and_drift():
    values = fixture()
    counts, *drifts = verifier.audit_prediction(*values, {"가", "나"})
    assert counts["restricted_top1"] == counts["full_top1"] == 1
    assert drifts == [0., 0., 0.]


@pytest.mark.parametrize("field,match", [
    ("edge", "edge"), ("residual", "residual"), ("winner", "winners"),
    ("score", "FP32"), ("count", "counts"), ("drift", "drift")])
def test_prediction_rejects_tampered_evidence(field, match):
    saved, row, inventory, historical = fixture()
    if field == "edge": saved["edge_ids"][0][2] = "조작"
    elif field == "residual": saved["residuals"][0] += 1
    elif field == "winner": saved["winning_path_ids"][0] = ["bad"]
    elif field == "score": saved["scores"][0] += 1
    elif field == "count": saved["counts"]["full_top1"] = 0
    else: saved["historical_selected_score_max_abs_drift"] = 1
    with pytest.raises(ValueError, match=match):
        verifier.audit_prediction(saved, row, inventory, historical, {"가", "나"})


def test_same_forward_expansion_can_only_lose_when_all_positives_selected():
    _, row, inventory, _ = fixture()
    row["selected_indices"] = [0]
    counts = verifier.row_counts(row, [.5, .8], {"가", "나"}, inventory)
    assert counts["restricted_top1"] == 1 and counts["full_top1"] == 0
    bad = deepcopy(row); bad["positive_indices"] = [1]
    with pytest.raises(ValueError, match="gained"):
        verifier.row_counts(bad, [.5, .8], {"가", "나"}, inventory)


def test_target_cache_requires_byte_exact_known_reuse_and_metadata(tmp_path, monkeypatch):
    train = tmp_path / "train"; run = tmp_path / "run"; train.mkdir(); run.mkdir()
    old_targets = ["가"]; targets = ["가", "나"]
    old_vectors = np.ones((1, 256), dtype=np.float32); old_flags = np.array([True])
    vectors = np.vstack((old_vectors, np.full((1, 256), 2, dtype=np.float32)))
    flags = np.array([True, True])
    write_json(train / "targets.json", old_targets)
    save_file({"vectors": old_vectors, "has_tokens": old_flags}, train / "target-features.safetensors")
    save_file({"vectors": vectors, "has_tokens": flags}, run / "target-features.safetensors")
    selection = [{"candidates": ["가", "나"]}]
    tokens = {"reserved_tokens": 2, "target_counts": [3, 3],
              "candidate_counts": [[3, 3]], "max_target_tokens": 3,
              "max_candidate_tokens": 3}
    write_json(run / "tokens.json", tokens)
    hashes = {"vectors": PURE.tensor_digest(vectors), "has_tokens": PURE.tensor_digest(flags)}
    old_hashes = {"vectors": PURE.tensor_digest(old_vectors),
                  "has_tokens": PURE.tensor_digest(old_flags)}
    metadata = {"schema": verifier.RUN_SCHEMA,
        "old_target_file_sha256": verifier.sha(train / "targets.json"),
        "old_tensor_file_sha256": verifier.sha(train / "target-features.safetensors"),
        "old_tensor_hashes": old_hashes, "tensor_hashes": hashes,
        "tensor_file_sha256": verifier.sha(run / "target-features.safetensors"),
        "tokens_sha256": verifier.sha(run / "tokens.json"), "reused_targets": 1,
        "new_targets": 1, "known_reuse_exact": True, "chunk_size": 256,
        "tensor_bytes": vectors.nbytes + flags.nbytes}
    write_json(run / "target-features.json", metadata)
    report = {"target_cache_tensor_hashes": hashes, "old_target_cache_unchanged": True}
    monkeypatch.setattr(verifier, "TRAIN", train); monkeypatch.setattr(verifier, "ROWS", 1)
    assert verifier.verify_cache(run, targets, selection, report, PURE)["reused"] == 1
    tampered = vectors.copy(); tampered[0, 0] = 9
    save_file({"vectors": tampered, "has_tokens": flags}, run / "target-features.safetensors")
    with pytest.raises(ValueError, match="reused exactly"):
        verifier.verify_cache(run, targets, selection, report, PURE)


def resource(stage, operation, elapsed, operation_elapsed, swap=0):
    return {"at": "time", "elapsed_s": elapsed, "rss_bytes": 100,
            "swap_used_bytes": 10 + swap, "swap_growth_bytes": swap,
            "operation_elapsed_s": operation_elapsed, "stage": stage,
            "operation": operation}


def test_resource_sequence_and_final_sample(tmp_path):
    selection = [{"id": "a"}]; operations = verifier.expected_operations(selection, 1)
    values = []; elapsed = 0.
    for operation in operations:
        elapsed += .1; values.append(resource("operation_start", operation, elapsed, 0.))
        elapsed += .1; values.append(resource("operation_end", operation, elapsed, .1))
    elapsed += .1; final = resource("final", None, elapsed, None, -1); values.append(final)
    elapsed += .1; values.append(resource("watchdog", None, elapsed, None, -1))
    path = tmp_path / "resources.jsonl"; write_jsonl(path, values)
    report = {"baseline_swap_bytes": 10, "final_resources": final, "elapsed_s": elapsed + .1}
    assert verifier.verify_resources(report, path, operations)["operation_pairs"] == len(operations)
    values[0]["operation"] = "wrong"; write_jsonl(path, values)
    with pytest.raises(ValueError, match="order"):
        verifier.verify_resources(report, path, operations)


def test_verifier_has_no_torch_model_or_producer_imports():
    source = SCRIPT.read_text()
    assert "import torch" not in source
    assert "from hanglish" not in source
    assert "from scripts.audit_shared_edge_full_fit" not in source


def test_alternate_run_is_not_authorized(tmp_path):
    with pytest.raises(ValueError, match="alternate"):
        verifier.verify(tmp_path)
