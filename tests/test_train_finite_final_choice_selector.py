import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

import scripts.train_finite_final_choice_selector as module
from scripts.train_final_choice_selector import SelectorExample


def example(name, correct, positives=(0,)):
    return SelectorExample(name, "group", "natural_fit", "ctx", ("a", "b"),
                           positives, (1., .2), (0, 1), correct, 1.)


def test_schedule_serialization_matches_preflight_contract_shape():
    batches = module.balanced_schedule([
        *(example(f"p{i}", True) for i in range(6)),
        example("r0", False), example("r1", False),
    ])
    records, raw, repeats = module.schedule_records(batches)
    assert [record["step"] for record in records] == [1, 2]
    assert all(record["examples"][0]["parent_correct"] is False for record in records)
    assert sum(repeats.values()) == 2
    assert raw == b"".join((json.dumps(row, sort_keys=True) + "\n").encode()
                           for row in records)


def test_loss_is_multi_positive_and_kd_only_for_parent_correct():
    correct = example("correct", True, (0, 1))
    repair = example("repair", False, (1,))
    scores = torch.tensor([.2, .1, .2, .8], requires_grad=True)
    loss = module.selector_loss(scores, [correct, repair], module.RECIPE)
    no_kd = module.selector_loss(
        scores, [correct, repair], {**module.RECIPE, "distillation_weight": 0.}
    )
    assert loss > no_kd
    loss.backward()
    assert torch.isfinite(scores.grad).all()


def test_preflight_rejects_rehashed_wrong_scope(tmp_path, monkeypatch):
    report = {"schema": "wrong"}
    raw = json.dumps(report).encode()
    path = tmp_path / "report.json"
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(module, "PREFLIGHT_SHA256", digest)
    with pytest.raises(ValueError, match="contract differs"):
        module.validate_preflight(path, digest)


def test_partition_rejects_fit_overlap_and_target_flags(tmp_path):
    manifest = {
        "schema": "finite-final-choice-fresh-evaluation-v3",
        "salt": "finite-final-choice-v3-2026-09-15", "target_rows": 1024,
        "selector_plan_sha256": module.PLAN_SHA256,
        "rows": 1, "groups": 1, "eval_ids": ["fit"], "eval_group_ids": ["g"],
        "remaining_ids": ["other"], "remaining_group_ids": ["h"],
        "source_hashes": {
            ".omx/artifacts/finite-final-choice-plan-v3.md": module.PLAN_SHA256,
            ".omx/artifacts/finite-final-choice-warm-lineage-v3/report.json":
                module.LINEAGE_SHA256,
        },
        "builder_sha256": "a" * 64, "selection_uses_target_text": False,
        "warm_lineage_sha256": module.LINEAGE_SHA256,
        "model_inference_run": False, "training_run": False,
        "old_splits_modified": False, "upstream_pretraining_overlap": "unknown",
    }
    raw = json.dumps(manifest).encode()
    path = tmp_path / "partition.json"
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(module, "EVALUATION_PARTITION_SHA256", digest)
    with pytest.raises(ValueError, match="contract differs"):
        module.validate_evaluation_partition(path, digest, [])
    manifest["rows"] = 1024
    manifest["eval_ids"] = [f"id{i}" for i in range(1024)]
    manifest["groups"] = 785
    manifest["eval_group_ids"] = ["g", *(f"g{i}" for i in range(784))]
    raw = json.dumps(manifest).encode()
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(module, "EVALUATION_PARTITION_SHA256", digest)
    monkeypatch.setattr(module, "EVALUATION_BUILDER_SHA256", "a" * 64)
    with pytest.raises(ValueError, match="overlaps natural FIT"):
        module.validate_evaluation_partition(
            path, digest, [{"id": "id0", "group_id": "g"}],
        )
    monkeypatch.undo()


def test_resource_gate_fails_closed(monkeypatch):
    monkeypatch.setattr(module, "rss_bytes",
                        lambda: module.RECIPE["rss_limit_bytes"] + 1)
    with pytest.raises(RuntimeError, match="resource bound"):
        module.require_resources(0., "test")


def test_captured_parent_summary_is_distinct_from_model_evaluation():
    natural = {"rows": [{"id": "n", "result": "a", "expected": ["a"]}]}
    challenge = {"rows": [{"id": "c", "result": "x", "expected": ["y"]}]}
    examples = [example("n", True)]
    summary = module.captured_production_summary(natural, challenge, examples)
    assert summary == {
        "all_rows": 2, "all_top1": 1, "eligible_rows": 1,
        "eligible_top1": 1,
        "source": "captured production parent responses; not warm RoBERTa scores",
    }


def test_batched_fit_evaluation_preserves_group_boundaries_and_ties():
    examples = [example("first", True), example("second", False, (1,))]

    class Model:
        def eval(self):
            return self

    class Ranker:
        model = Model()

        def score_pairs(self, contexts, candidates):
            assert len(contexts) == len(candidates) == 4
            return [1., 1., .2, .8]

    result = module.evaluate_fit(Ranker(), examples)
    assert result["rows"] == 2
    assert result["correct"] == 2
    assert [row["chosen_source_index"] for row in result["outcomes"]] == [0, 1]


def test_provenance_rejects_partition_or_lineage_hash_conflicts():
    common = {"shared": "a" * 64}
    kwargs = dict(
        preflight={"source_hashes": common},
        preflight_path=module.ROOT / "preflight.json", preflight_raw=b"preflight",
        plan_path=module.ROOT / "plan.md", plan_raw=b"plan",
        partition_path=module.ROOT / "partition.json", partition_raw=b"partition",
        lineage_path=module.ROOT / "lineage.json", lineage_raw=b"lineage",
        runner_sha="e" * 64,
    )
    partition = {"builder_sha256": "b" * 64,
                 "source_hashes": {"shared": "c" * 64}}
    lineage = {"input_hashes_before": common}
    with pytest.raises(ValueError, match="partition source conflict"):
        module.provenance(partition=partition, lineage=lineage, **kwargs)
    partition["source_hashes"] = common
    lineage["input_hashes_before"] = {"shared": "d" * 64}
    with pytest.raises(ValueError, match="lineage source conflict"):
        module.provenance(partition=partition, lineage=lineage, **kwargs)
