from dataclasses import replace
import hashlib
import random

import pytest
import torch

import scripts.train_balanced_final_choice_selector as module
from scripts.train_final_choice_selector import SelectorExample


def example(index, parent_correct):
    return SelectorExample(
        f"id-{index}", f"group-{index}", "natural_fit", "ctx",
        ("가", "카"), (0,) if parent_correct else (1,), (1., 0.),
        (0, 1), parent_correct, 1. if parent_correct else 3.,
    )


def test_balanced_schedule_is_deterministic_preserves_once_and_cycles_errors(monkeypatch):
    monkeypatch.setitem(module.RECIPE, "preserve_per_batch", 3)
    monkeypatch.setitem(module.RECIPE, "group_batch", 4)
    values = [*(example(index, True) for index in range(10)),
              *(example(100 + index, False) for index in range(2))]
    left = module.balanced_schedule(values, seed=23)
    right = module.balanced_schedule(values, seed=23)
    assert [[item.case_id for item in batch] for batch in left] == [
        [item.case_id for item in batch] for batch in right]
    preserved = [item.case_id for batch in left for item in batch if item.parent_correct]
    errors = [item.case_id for batch in left for item in batch if not item.parent_correct]
    assert sorted(preserved) == [f"id-{index}" for index in range(10)]
    assert len(set(preserved)) == len(preserved)
    assert len(errors) == len(left) == 4 and set(errors) == {"id-100", "id-101"}
    assert all(sum(not item.parent_correct for item in batch) == 1 for batch in left)
    assert [len(batch) for batch in left] == [4, 4, 4, 2]
    assert all(not batch[0].parent_correct for batch in left)
    expected_preserve = [f"id-{index}" for index in range(10)]
    random.Random(23).shuffle(expected_preserve)
    assert [item.case_id for batch in left for item in batch[1:]] == expected_preserve
    expected_error_cycle = ["id-100", "id-101"]
    repair_rng = random.Random(23)
    repair_rng.shuffle(expected_error_cycle)
    assert errors[:2] == expected_error_cycle


def test_schedule_forces_all_weights_to_one():
    batches = module.balanced_schedule([example(0, True), example(1, False)])
    assert all(item.loss_weight == 1. for batch in batches for item in batch)


def test_schedule_rejects_missing_parent_class():
    with pytest.raises(ValueError, match="parent-error"):
        module.balanced_schedule([example(0, True)])
    with pytest.raises(ValueError, match="parent-correct"):
        module.balanced_schedule([example(0, False)])


def test_schedule_record_has_stable_hash_and_truthful_error_repeats():
    batches = module.balanced_schedule(
        [*(example(index, True) for index in range(7)), example(100, False)]
    )
    records, raw, repeats = module.schedule_record(batches)
    assert len(records) == 3
    assert repeats == {"natural_fit:id-100": 3}
    assert hashlib.sha256(raw).hexdigest() == module.sha256_bytes(raw)


def test_balanced_loss_equals_unweighted_base_loss():
    correct = replace(example(0, True), loss_weight=1.)
    error = replace(example(1, False), loss_weight=1.)
    scores = torch.tensor([1., 0., .2, .8], requires_grad=True)
    loss = module.selector_loss(scores, [correct, error], module.RECIPE)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(scores.grad).all()


def test_plan_requires_explicit_matching_sha(tmp_path, monkeypatch):
    plan = tmp_path / "plan.md"
    plan.write_bytes(b"fixed plan")
    digest = hashlib.sha256(b"fixed plan").hexdigest()
    monkeypatch.setattr(module, "PLAN_SHA256", digest)
    assert module.read_plan(plan, digest) == b"fixed plan"
    with pytest.raises(ValueError, match="frozen v2"):
        module.read_plan(plan, "0" * 64)
    with pytest.raises(ValueError, match="explicit"):
        module.read_plan(plan, "bad")


def test_provenance_passes_base_helper_sha_not_new_runner(monkeypatch, tmp_path):
    observed = {}

    def fake_base(natural, challenge, helper_sha):
        observed["helper_sha"] = helper_sha
        return {"base": "a" * 64}

    monkeypatch.setattr(module, "base_provenance_hashes", fake_base)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "FROZEN_HELPERS", {})
    plan = tmp_path / "plan.md"
    plan.write_text("plan")
    evaluation = tmp_path / "eval.json"
    evaluation.write_text("{}")
    hashes = module.balanced_provenance_hashes(
        {}, {}, plan, "b" * 64, evaluation, "d" * 64, "c" * 64
    )
    assert observed["helper_sha"] == module.BASE_HELPER_SHA256
    assert hashes["scripts/train_balanced_final_choice_selector.py"] == "c" * 64


def test_evaluation_partition_is_metadata_only_and_disjoint(tmp_path, monkeypatch):
    source_hashes = {
        ".omx/artifacts/final-choice-partition-v1/manifest.json": module.PARTITION_V1_SHA256,
        "data/natural_context/klue-v1.1-pilot-v2/train.jsonl": module.RAW_TRAIN_SHA256,
        ".omx/artifacts/balanced-final-choice-plan-v2.md": module.PLAN_SHA256,
    }
    manifest = {
        "schema": module.EVALUATION_SCHEMA, "salt": module.EVALUATION_SALT,
        "target_rows": 2, "selector_plan_sha256": module.PLAN_SHA256,
        "builder_sha256": module.PARTITION_BUILDER_SHA256,
        "source_hashes": source_hashes, "eval_ids": ["e1", "e2"],
        "eval_group_ids": ["eg"], "remaining_ids": ["r1"],
        "remaining_group_ids": ["rg"], "rows": 2, "groups": 1,
        "selection_uses_target_text": False, "model_inference_run": False,
        "training_run": False, "old_splits_modified": False,
        "upstream_pretraining_overlap": "unknown",
    }
    monkeypatch.setattr(module, "EVALUATION_TARGET", 2)
    raw = __import__("json").dumps(manifest).encode()
    digest = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(module, "EVALUATION_PARTITION_SHA256", digest)
    path = tmp_path / "manifest.json"
    path.write_bytes(raw)
    _, parsed = module.read_evaluation_partition(
        path, digest, [{"id": "fit", "group_id": "fg"}]
    )
    assert parsed["eval_ids"] == ["e1", "e2"]
    manifest["eval_ids"][0] = "fit"
    raw = __import__("json").dumps(manifest).encode(); path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(module, "EVALUATION_PARTITION_SHA256", digest)
    with pytest.raises(ValueError, match="overlaps"):
        module.read_evaluation_partition(
            path, digest, [{"id": "fit", "group_id": "fg"}]
        )


def test_nonfinite_score_parity_and_parameters_fail_closed():
    item = example(0, True)
    with pytest.raises(ValueError, match="finite"):
        module.score_parity_delta([[float("nan"), 0.]], [[0., 0.]], [item])
    model = torch.nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.fill_(float("nan"))
    with pytest.raises(RuntimeError, match="nonfinite"):
        module.require_finite_parameters(model)


def test_resource_bound_fails_closed(monkeypatch):
    monkeypatch.setattr(module, "rss_bytes", lambda: module.RECIPE["rss_limit_bytes"] + 1)
    with pytest.raises(RuntimeError, match="resource bound"):
        module.require_resources(0., "fixture")
