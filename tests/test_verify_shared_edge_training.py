from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_shared_edge_training.py"
SPEC = importlib.util.spec_from_file_location("verify_shared_edge_training", SCRIPT)
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)
HELPER = verifier.helper_module()


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True))


def write_jsonl(path, values):
    path.write_text("".join(json.dumps(value, sort_keys=True) + "\n" for value in values))


def compact_path(source, spans):
    identity = [source, [[start, end, text] for start, end, text in spans]]
    return "alignment-v1-" + verifier.digest(identity)


def inventory_fixture():
    source = "ab"
    edges = [{"input_start": 0, "input_end": 1, "text": "아"},
             {"input_start": 1, "input_end": 2, "text": "비"},
             {"input_start": 1, "input_end": 2, "text": "바"}]
    spans_a = [(0, 1, "아"), (1, 2, "비")]
    spans_b = [(0, 1, "아"), (1, 2, "바")]
    record = {"id": "row", "component_id": "component", "input": source,
              "schema": "source-correspondence-v1", "edges": edges,
              "paths": [
                  {"path_id": compact_path(source, spans_a), "text": "아비",
                   "edge_indices": [0, 1]},
                  {"path_id": compact_path(source, spans_b), "text": "아바",
                   "edge_indices": [0, 2]}],
              "reference_path_id": compact_path(source, spans_a)}
    selected = {"id": "row", "component_id": "component", "input": source,
                "arms": {"b_union": {"candidates": ["아비", "아바"],
                                      "positive_indices": [0]}}}
    targets, rows, counts = verifier.reconstruct_inventory([selected], [record])
    return targets, rows[0], counts


def test_inventory_reconstructs_targets_edges_paths_and_source_weights():
    targets, row, counts = inventory_fixture()
    assert targets == ["바", "비", "아"]
    assert [(edge["input_start"], edge["input_end"], edge["text"])
            for edge in row["edges"]] == [(0, 1, "아"), (1, 2, "바"), (1, 2, "비")]
    assert row["paths"][0][0]["weight_numerators"] == [1, 1]
    assert row["paths"][0][0]["weight_denominator"] == 2
    assert counts["edges"] == 3 and counts["paths"] == 2


def after_fixture():
    _, row, _ = inventory_fixture()
    edge_scores = [verifier.fp32(value) for value in (1., 3., 2.)]
    path_audit = []
    residuals, winners = [], []
    for paths in row["paths"]:
        actual_paths = []
        for path in paths:
            weights = [verifier.fp32(value / path["weight_denominator"])
                       for value in path["weight_numerators"]]
            contributions = [verifier.fp32(edge_scores[index] * weight)
                             for index, weight in zip(path["edge_indices"], weights)]
            residual = verifier.fp32(sum(contributions))
            actual_paths.append({"path_id": path["path_id"],
                                 "edge_ids": path["edge_indices"], "weights": weights,
                                 "contributions": contributions, "residual": residual})
        maximum = max(value["residual"] for value in actual_paths)
        residuals.append(maximum)
        winners.append([value["path_id"] for value in actual_paths
                        if value["residual"] == maximum])
        path_audit.append(actual_paths)
    base = {"scores": [verifier.fp32(.2), verifier.fp32(.1)]}
    scores = [verifier.fp32(parent + residual)
              for parent, residual in zip(base["scores"], residuals)]
    loss = HELPER.listwise_loss(scores, row["positive_indices"])
    saved = {"id": "row", "correct": 0, "loss": loss,
             "scores": scores, "base_scores": base["scores"], "residuals": residuals,
             "edge_ids": [[edge["input_start"], edge["input_end"], edge["text"]]
                          for edge in row["edges"]],
             "edge_scores": edge_scores, "path_audit": path_audit,
             "winning_path_ids": winners}
    return saved, row, base, {"correct": 1}


def test_after_replays_every_edge_path_weight_contribution_winner_and_score():
    saved, row, base, control = after_fixture()
    result = verifier.audit_after_row(saved, row, base, control, HELPER)
    assert result["correct"] == 0 and result["loss"] == 1
    assert saved["path_audit"][0][0]["contributions"][0] \
        == saved["path_audit"][1][0]["contributions"][0]


@pytest.mark.parametrize("mutation,match", [
    ("edge", "edge domain"), ("weight", "weights"),
    ("contribution", "contribution"), ("winner", "winning"),
    ("residual", "winning"), ("score", "base plus residual"),
    ("loss", "CE loss")])
def test_after_rejects_tampered_edge_path_and_ranking_evidence(mutation, match):
    saved, row, base, control = after_fixture()
    if mutation == "edge": saved["edge_ids"][0][2] = "조작"
    elif mutation == "weight": saved["path_audit"][0][0]["weights"][0] += .1
    elif mutation == "contribution": saved["path_audit"][0][0]["contributions"][0] += .1
    elif mutation == "winner": saved["winning_path_ids"][0] = ["bad"]
    elif mutation == "residual": saved["residuals"][0] += .1
    elif mutation == "score": saved["scores"][0] += .1
    else: saved["loss"] += .1
    with pytest.raises(ValueError, match=match):
        verifier.audit_after_row(saved, row, base, control, HELPER)


def test_target_cache_checks_shape_flags_zero_vectors_metadata_and_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr(verifier, "TARGETS", 2)
    targets = ["", "아"]
    selection = [{"candidates": ["아", "어"]}]
    write_json(tmp_path / "targets.json", targets)
    tokens = {"reserved_tokens": 3, "target_counts": [3, 4],
              "candidate_counts": [[4, 4]], "max_target_tokens": 4,
              "max_candidate_tokens": 4}
    write_json(tmp_path / "tokens.json", tokens)
    vectors = np.zeros((2, 256), dtype=np.float32); vectors[1, 0] = 1.
    flags = np.asarray([False, True], dtype=np.bool_)
    save_file({"vectors": vectors, "has_tokens": flags}, tmp_path / "target-features.safetensors")
    hashes = {"vectors": HELPER.tensor_digest(vectors), "has_tokens": HELPER.tensor_digest(flags)}
    metadata = {"schema": verifier.RUN_SCHEMA,
        "targets_sha256": verifier.sha(tmp_path / "targets.json"),
        "tokens_sha256": verifier.sha(tmp_path / "tokens.json"),
        "tensor_hashes": hashes,
        "tensor_file_sha256": verifier.sha(tmp_path / "target-features.safetensors"),
        "parent_forward_batches": 1, "chunk_size": 256, "batch_size": 8}
    write_json(tmp_path / "target-features.json", metadata)
    report = {"target_cache_tensor_hashes": hashes, "target_parent_forward_batches": 1,
              "target_cache_bytes": (tmp_path / "target-features.safetensors").stat().st_size}
    result = verifier.verify_target_cache(tmp_path, targets, selection, report, HELPER)
    assert result["vector_shape"] == [2, 256] and result["has_tokens"] == 1
    flags[0] = True
    save_file({"vectors": vectors, "has_tokens": flags}, tmp_path / "changed.safetensors")
    (tmp_path / "changed.safetensors").replace(tmp_path / "target-features.safetensors")
    with pytest.raises(ValueError):
        verifier.verify_target_cache(tmp_path, targets, selection, report, HELPER)


def test_updates_require_exact_CE_only_order_losses_and_gradients(tmp_path):
    run = tmp_path; selection = [{"id": "a"}, {"id": "b"}]
    write_json(run / "selection.json", selection)
    values = []
    for epoch in range(2):
        for index, row in enumerate(selection):
            update = epoch * 2 + index + 1
            values.append({"update": update, "epoch": epoch, "id": row["id"],
                "loss": .5, "gradient_norm": 1.,
                "gradient_groups": {name: float(update) for name in verifier.GRADIENT_GROUPS}})
    write_jsonl(run / "updates.jsonl", values)
    result = verifier.verify_updates(run, row_count=2, epochs=2)
    assert result["updates"] == 4 and result["epoch_mean_loss"] == [.5, .5]
    values[1]["id"] = "a"; write_jsonl(run / "updates.jsonl", values)
    with pytest.raises(ValueError, match="order"):
        verifier.verify_updates(run, row_count=2, epochs=2)


def resource(stage, operation, elapsed, operation_elapsed):
    return {"at": "time", "elapsed_s": elapsed, "rss_bytes": 100,
            "swap_used_bytes": 10, "swap_growth_bytes": 0,
            "operation_elapsed_s": operation_elapsed, "stage": stage,
            "operation": operation}


def test_resources_replay_atomic_sequence_and_final_bounds(tmp_path):
    selection = [{"id": "a"}]
    operations = verifier.expected_operations(selection, targets=2, updates=2)
    rows, elapsed = [], 0.
    for operation in operations:
        elapsed += .1; rows.append(resource("operation_start", operation, elapsed, 0.))
        elapsed += .1; rows.append(resource("operation_end", operation, elapsed, .1))
    path = tmp_path / "resources.jsonl"; write_jsonl(path, rows)
    final = resource("final", None, elapsed + .1, None)
    report = {"baseline_swap_bytes": 10, "final_resources": final, "elapsed_s": elapsed + .2}
    result = verifier.verify_resources(report, path, selection, targets=2, updates=2)
    assert result["operation_pairs"] == len(operations)
    rows[2]["operation"] = "wrong"; write_jsonl(path, rows)
    with pytest.raises(ValueError, match="order"):
        verifier.verify_resources(report, path, selection, targets=2, updates=2)


def test_verifier_does_not_import_trainer_model_torch_or_tokenizer():
    source = SCRIPT.read_text()
    assert "import torch" not in source
    assert "from scripts.train_shared_edge_residual" not in source
    assert "from hanglish.models.shared_edge_residual" not in source
    assert "tokenizer(" not in source


def test_alternate_run_is_not_authorized(tmp_path):
    with pytest.raises(ValueError, match="alternate"):
        verifier.verify(tmp_path)
