from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest

from scripts.long_context_lora_checkpoint_v2 import (
    ADAM_CONFIG, ADAPTER_CONFIG, CheckpointError, CheckpointStore, LockUnavailable,
    canonical_bytes, canonical_digest, cumulative_active_seconds, envelope, expected_adapter_shapes,
    make_run_identity, resolve_postcommit_failure, resume_preflight, schedule_cursor,
    validate_adam_constructor, validate_adapter_tensors, validate_envelope,
    validate_optimizer_tensors,
)


HASH = "a" * 64


class NumpyCodec:
    def flatten(self, tree):
        return dict(tree)

    def unflatten(self, flat):
        return dict(flat)

    def save(self, path, arrays):
        with path.open("wb") as handle:
            np.savez(handle, **arrays)

    def load(self, path):
        with np.load(path, allow_pickle=False) as values:
            return {name: values[name] for name in values.files}

    def materialize(self, *values):
        return None


@pytest.fixture(scope="module")
def adapter():
    return {name: np.full(shape, index / 1000, dtype=np.float32)
            for index, (name, shape) in enumerate(expected_adapter_shapes().items(), 1)}


def optimizer(adapter, update):
    state = {
        "step": np.asarray(update, dtype=np.uint64),
        "learning_rate": np.asarray(1e-5, dtype=np.float32),
    }
    for name, value in adapter.items():
        state[f"{name}.m"] = np.full(value.shape, update / 100, dtype=np.float32)
        state[f"{name}.v"] = np.full(value.shape, update / 1000, dtype=np.float32)
    return state


def identity(*, updates=3, mode="synthetic", final=None):
    final = (1 if updates == 3 else 4) if final is None else final
    return make_run_identity(
        logical_run_id=f"test-{updates}", mode=mode, baseline_swap_used_bytes=1000,
        schedule_sha256=HASH, microbatches=(updates - 1) * 4 + final,
        updates=updates, final_update_microbatches=final,
        source_hashes={"source": HASH}, config_hashes={"config": HASH},
        model_hashes={"model": HASH}, package_versions={"mlx": "0.31.2"},
        created_at="2026-09-15T00:00:00+00:00")


def publish(segment, run, adapter, update, callback=lambda stage: {"stage": stage},
            failure_hook=None):
    final = update == run["schedule"]["updates"]
    count = min(update * 4, run["schedule"]["microbatches"])
    batch_count = run["schedule"]["final_update_microbatches"] if final else 4
    return segment.publish_checkpoint(
        codec=NumpyCodec(), adapter_tensors=adapter,
        optimizer_state=optimizer(adapter, update), update=update,
        completed_microbatches=count,
        next_schedule=None if final else {"role": "synthetic", "id": f"row-{count + 1}"},
        batch_records=[{"role": "synthetic", "id": f"row-{index}",
                        "microbatch": index, "update": update}
                       for index in range(count - batch_count + 1, count + 1)],
        losses=[float(index) for index in range(batch_count)],
        adapter_config=ADAPTER_CONFIG, optimizer_config=ADAM_CONFIG,
        recipe=run["recipe"], elapsed_seconds=float(update),
        resource_callback=callback, failure_hook=failure_hook)


def test_envelope_digest_is_payload_only_and_canonical():
    first = envelope({"b": 2, "a": "한글"})
    second = envelope({"a": "한글", "b": 2})
    assert first == second
    assert first["digest"] == canonical_digest(first["payload"])
    altered = copy.deepcopy(first)
    altered["digest"] = "0" * 64
    with pytest.raises(CheckpointError):
        validate_envelope(altered)
    with pytest.raises(ValueError):
        envelope({"digest": "self-reference"})


def test_exact_adapter_and_adam_inventory(adapter):
    inventory = validate_adapter_tensors(adapter)
    assert len(inventory) == 16
    assert sum(item["size"] for item in inventory.values()) == 327680
    state = optimizer(adapter, 341)
    optimizer_inventory = validate_optimizer_tensors(state, 341, inventory, ADAM_CONFIG)
    assert len(optimizer_inventory) == 34
    wrong_lr = dict(state, learning_rate=np.asarray(1.0001e-5, dtype=np.float32))
    with pytest.raises(CheckpointError, match="learning rate"):
        validate_optimizer_tensors(wrong_lr, 341, inventory, ADAM_CONFIG)
    wrong_step = dict(state, step=np.asarray(342, dtype=np.uint64))
    with pytest.raises(CheckpointError, match="step"):
        validate_optimizer_tensors(wrong_step, 341, inventory, ADAM_CONFIG)
    nonfinite = dict(adapter)
    first = next(iter(nonfinite))
    nonfinite[first] = nonfinite[first].copy()
    nonfinite[first].flat[0] = np.nan
    with pytest.raises(CheckpointError, match="nonfinite"):
        validate_adapter_tensors(nonfinite)


def test_adam_constructor_learning_rate_is_checked_before_state_restore():
    class FakeAdam:
        learning_rate = np.asarray(2e-5, dtype=np.float32)
        betas = [0.9, 0.999]
        eps = 1e-8
        bias_correction = True

    with pytest.raises(CheckpointError, match="constructor"):
        validate_adam_constructor(FakeAdam())
    FakeAdam.learning_rate = np.asarray(1e-5, dtype=np.float32)
    validate_adam_constructor(FakeAdam())


def test_run_identity_and_original_baseline_accounting():
    run = identity()
    events = [
        {"kind": "segment_start", "segment_id": "segment-0001"},
        {"kind": "segment_terminal", "segment_id": "segment-0001",
         "terminal_segment_seconds": 100.0},
    ]
    at_limit = resume_preflight(run, events, {
        "swap_used_bytes": 1000 + 512 * 1024 * 1024,
        "system_free_percent": 10.0,
    })
    assert at_limit["prior_terminal_seconds"] == 100.0
    with pytest.raises(CheckpointError, match="swap"):
        resume_preflight(run, events, {
            "swap_used_bytes": 1001 + 512 * 1024 * 1024,
            "system_free_percent": 10.0,
        })
    with pytest.raises(CheckpointError, match="free memory"):
        resume_preflight(run, events, {"swap_used_bytes": 900, "system_free_percent": 9.99})
    assert cumulative_active_seconds(events, 25.0) == 125.0
    with pytest.raises(CheckpointError, match="terminal"):
        cumulative_active_seconds(events[:-1])
    almost = copy.deepcopy(events)
    almost[1]["terminal_segment_seconds"] = 899.999
    assert resume_preflight(run, almost, {
        "swap_used_bytes": 1000, "system_free_percent": 10.0,
    })["prior_terminal_seconds"] == 899.999
    exact = copy.deepcopy(events)
    exact[1]["terminal_segment_seconds"] = 900.0
    with pytest.raises(CheckpointError, match="active time"):
        resume_preflight(run, exact, {
            "swap_used_bytes": 1000, "system_free_percent": 10.0,
        })


def test_schedule_cursor_discards_partial_accumulation():
    schedule = [{"microbatch": i, "update": (i - 1) // 4 + 1,
                 "role": "synthetic", "id": f"row-{i}"} for i in range(1, 12)]
    cursor = schedule_cursor(schedule, 2)
    assert cursor == {
        "completed_updates": 2, "completed_microbatches": 8,
        "next_schedule_position": 8,
        "next_expected": {"role": "synthetic", "id": "row-9"},
    }


def test_atomic_publish_roundtrip_and_resume(tmp_path, adapter):
    run = identity()
    store = CheckpointStore.create(tmp_path / "run", run)
    with store.segment("segment-0001") as segment:
        result = publish(segment, run, adapter, 1)
        assert result["pointer"]["status"] == "eligible"
        assert result["payload_bytes"] > 0
        assert set(result["resource_observations"]) == {
            "post_materialize", "post_reload", "precommit", "postcommit", "final_eligibility"}
        segment.finish("failed", 10.0)
    reopened = CheckpointStore.open(tmp_path / "run")
    with reopened.segment("segment-0002", resume=True,
                          preflight_metrics={"swap_used_bytes": 900,
                                             "system_free_percent": 50.0}):
        restored = reopened.load_eligible(codec=NumpyCodec())
        assert restored.manifest["update"] == 1
        assert validate_adapter_tensors(restored.adapter_tensors) == result["manifest"]["adapter_inventory"]


def test_whole_segment_lock_is_nonblocking(tmp_path):
    run = identity()
    first = CheckpointStore.create(tmp_path / "run", run)
    second = CheckpointStore.open(tmp_path / "run")
    with first.segment("segment-0001"):
        with pytest.raises(LockUnavailable):
            with second.segment("segment-0002"):
                pass


def test_failure_before_pointer_preserves_previous_eligible(tmp_path, adapter):
    run = identity()
    store = CheckpointStore.create(tmp_path / "run", run)
    with store.segment("segment-0001") as segment:
        first = publish(segment, run, adapter, 1)

        def fail(stage):
            if stage == "precommit":
                raise RuntimeError("injected")
            return {"stage": stage}

        with pytest.raises(RuntimeError, match="injected"):
            publish(segment, run, adapter, 2, fail)
        assert store._pointer()["manifest_sha256"] == first["pointer"]["manifest_sha256"]
        assert store._pointer()["status"] == "eligible"


def test_postcommit_failure_requires_explicit_resolution(tmp_path, adapter):
    run = identity()
    store = CheckpointStore.create(tmp_path / "run", run)
    with store.segment("segment-0001") as segment:
        first = publish(segment, run, adapter, 1)

        def fail(stage):
            if stage == "final_eligibility":
                raise RuntimeError("cap")
            return {"stage": stage}

        with pytest.raises(RuntimeError, match="cap"):
            publish(segment, run, adapter, 2, fail)
        pointer = store._pointer()
        assert pointer["status"] == "committed"
        with pytest.raises(CheckpointError, match="not resume-eligible"):
            store.load_eligible(codec=NumpyCodec())
        segment.finish("resource_failure", 10.0)
    with store.resolution("resolution-0001") as resolution_segment:
        resolution = resolve_postcommit_failure(store, resolution_segment,
                                                pointer["manifest_sha256"])
        assert resolution["eligible"] == first["pointer"]["manifest_sha256"]
        assert store.load_eligible(codec=NumpyCodec()).manifest["update"] == 1
        resolution_segment.finish("complete", 0.1)


@pytest.mark.parametrize("mutation", ["pointer", "event"])
def test_resolution_cas_rejects_pointer_or_event_mutation(tmp_path, adapter, mutation):
    run = identity()
    store = CheckpointStore.create(tmp_path / "run", run)
    with store.segment("segment-0001") as segment:
        publish(segment, run, adapter, 1)

        def fail(stage):
            if stage == "final_eligibility":
                raise RuntimeError("cap")

        with pytest.raises(RuntimeError):
            publish(segment, run, adapter, 2, fail)
        failed_manifest = store._pointer()["manifest_sha256"]
        segment.finish("resource_failure", 1.0)
    with store.resolution("resolution-0001") as resolution_segment:
        def mutate(stage):
            assert stage == "before_resolution_cas"
            if mutation == "pointer":
                latest = store.root / "latest.json"
                latest.write_bytes(latest.read_bytes() + b" ")
            else:
                resolution_segment.append_event("concurrent_resolution_note")

        with pytest.raises(Exception, match="CAS state changed"):
            resolve_postcommit_failure(store, resolution_segment, failed_manifest,
                                       mutation_hook=mutate)


def test_resolution_validates_predecessor_before_rejection_event(tmp_path, adapter):
    run = identity()
    store = CheckpointStore.create(tmp_path / "run", run)
    with store.segment("segment-0001") as segment:
        first = publish(segment, run, adapter, 1)

        def fail(stage):
            if stage == "final_eligibility":
                raise RuntimeError("cap")

        with pytest.raises(RuntimeError):
            publish(segment, run, adapter, 2, fail)
        failed_manifest = store._pointer()["manifest_sha256"]
        segment.finish("resource_failure", 1.0)
    with (Path(first["path"]) / "adapter.safetensors").open("ab") as handle:
        handle.write(b"corrupt")
    before = len(list((store.root / "ledger").glob("*.json")))
    with store.resolution("resolution-0001") as resolution_segment:
        after_start = len(list((store.root / "ledger").glob("*.json")))
        assert after_start == before + 1
        with pytest.raises(CheckpointError, match="predecessor payload"):
            resolve_postcommit_failure(store, resolution_segment, failed_manifest)
        assert len(list((store.root / "ledger").glob("*.json"))) == after_start


def test_retention_keeps_two_payloads_and_receipt_evidence(tmp_path, adapter):
    run = identity()
    store = CheckpointStore.create(tmp_path / "run", run)
    with store.segment("segment-0001") as segment:
        results = [publish(segment, run, adapter, update) for update in (1, 2)]
        stale_pointer = copy.deepcopy(store._pointer())
        results.append(publish(segment, run, adapter, 3))
        snapshot = store.audit_snapshot()
        assert [entry["update"] for entry in snapshot["latest"]["retained"]] == [3, 2]
        assert len(snapshot["prune_receipts"]) == 1
        oldest = Path(results[0]["path"])
        assert (oldest / "manifest.json").is_file()
        assert not (oldest / "adapter.safetensors").exists()
        assert (Path(results[1]["path"]) / "adapter.safetensors").is_file()
        assert (Path(results[2]["path"]) / "adapter.safetensors").is_file()
        retry_path = store.root / "checkpoints" / "segment-0002" / "update-0003"
        retry_path.parent.mkdir()
        segment._prune_before_commit(stale_pointer, retry_path, "b" * 64)
        assert len(store.audit_snapshot()["prune_receipts"]) == 2


def test_prune_refuses_payload_corruption(tmp_path, adapter):
    run = identity()
    store = CheckpointStore.create(tmp_path / "run", run)
    with store.segment("segment-0001") as segment:
        first = publish(segment, run, adapter, 1)
        second = publish(segment, run, adapter, 2)
        with (Path(first["path"]) / "adapter.safetensors").open("ab") as handle:
            handle.write(b"corruption")
        with pytest.raises(CheckpointError, match="prune victim"):
            publish(segment, run, adapter, 3)
        assert store._pointer()["manifest_sha256"] == second["pointer"]["manifest_sha256"]


@pytest.mark.parametrize("kind", ["file", "directory", "symlink", "fifo"])
def test_prune_rejects_unexpected_victim_entries(tmp_path, adapter, kind):
    run = identity()
    store = CheckpointStore.create(tmp_path / "run", run)
    with store.segment("segment-0001") as segment:
        first = publish(segment, run, adapter, 1)
        second = publish(segment, run, adapter, 2)
        unexpected = Path(first["path"]) / "unexpected"
        if kind == "file":
            unexpected.write_text("x")
        elif kind == "directory":
            unexpected.mkdir()
        elif kind == "symlink":
            unexpected.symlink_to(Path(first["path"]) / "manifest.json")
        else:
            os.mkfifo(unexpected)
        with pytest.raises(CheckpointError, match="unexpected file type"):
            publish(segment, run, adapter, 3)
        assert store._pointer()["manifest_sha256"] == second["pointer"]["manifest_sha256"]


@pytest.mark.parametrize("stage", [
    "after_payload_write", "after_payload_fsync", "after_reload_validation",
    "after_directory_rename", "after_checkpoint_root_fsync",
    "after_precommit_observation", "before_pointer_replace",
])
def test_prepointer_transaction_fault_preserves_prior_eligible(tmp_path, adapter, stage):
    run = identity()
    store = CheckpointStore.create(tmp_path / "run", run)
    with store.segment("segment-0001") as segment:
        first = publish(segment, run, adapter, 1)

        def inject(current):
            if current == stage:
                raise RuntimeError(stage)

        with pytest.raises(RuntimeError, match=stage):
            publish(segment, run, adapter, 2, failure_hook=inject)
        pointer = store._pointer()
        assert pointer["status"] == "eligible"
        assert pointer["manifest_sha256"] == first["pointer"]["manifest_sha256"]
        assert any(Path(store.root / "checkpoints").rglob("*update-0002*"))


@pytest.mark.parametrize("stage", [
    "after_pointer_replace", "after_commit_event", "after_eligibility_event",
])
def test_postpointer_transaction_fault_is_not_resumable(tmp_path, adapter, stage):
    run = identity()
    store = CheckpointStore.create(tmp_path / "run", run)
    with store.segment("segment-0001") as segment:
        publish(segment, run, adapter, 1)

        def inject(current):
            if current == stage:
                raise RuntimeError(stage)

        with pytest.raises(RuntimeError, match=stage):
            publish(segment, run, adapter, 2, failure_hook=inject)
        assert store._pointer()["status"] == "committed"
        with pytest.raises(CheckpointError, match="not resume-eligible"):
            store.load_eligible(codec=NumpyCodec())


def test_corrupt_latest_fails_without_rollback(tmp_path, adapter):
    run = identity()
    store = CheckpointStore.create(tmp_path / "run", run)
    with store.segment("segment-0001") as segment:
        publish(segment, run, adapter, 1)
        segment.finish("failed", 1.0)
    latest = tmp_path / "run" / "latest.json"
    value = json.loads(latest.read_text())
    value["digest"] = "0" * 64
    latest.write_text(json.dumps(value))
    reopened = CheckpointStore.open(tmp_path / "run")
    with pytest.raises(CheckpointError, match="digest"):
        with reopened.segment("segment-0002", resume=True,
                              preflight_metrics={"swap_used_bytes": 1000,
                                                 "system_free_percent": 50.0}):
            pass


def test_symlinked_lock_fails_before_segment_state_read(tmp_path):
    root = tmp_path / "run"
    store = CheckpointStore.create(root, identity())
    (root / "run.lock").unlink()
    (root / "run.lock").symlink_to(root / "run.json")
    with pytest.raises((CheckpointError, OSError)):
        with store.segment("segment-0001"):
            pass


def test_cursor_and_batch_contract_rejects_gaps_and_partial_rows(tmp_path, adapter):
    run = identity()
    store = CheckpointStore.create(tmp_path / "run", run)
    with store.segment("segment-0001") as segment:
        with pytest.raises(CheckpointError, match="first committed"):
            publish(segment, run, adapter, 2)
        first = publish(segment, run, adapter, 1)
        with pytest.raises(CheckpointError, match="extend"):
            publish(segment, run, adapter, 3)
        kwargs = dict(
            codec=NumpyCodec(), adapter_tensors=adapter,
            optimizer_state=optimizer(adapter, 2), update=2, completed_microbatches=8,
            next_schedule={"role": "synthetic", "id": "row-9"},
            batch_records=[{"role": "synthetic", "id": "row-5",
                            "microbatch": 5, "update": 2}], losses=[1.0],
            adapter_config=ADAPTER_CONFIG, optimizer_config=ADAM_CONFIG,
            recipe=run["recipe"], elapsed_seconds=2.0,
            resource_callback=lambda stage: None)
        with pytest.raises(CheckpointError, match="batch size"):
            segment.publish_checkpoint(**kwargs)
        assert store._pointer()["manifest_sha256"] == first["pointer"]["manifest_sha256"]


def test_reserved_event_identity_cannot_be_overridden(tmp_path):
    store = CheckpointStore.create(tmp_path / "run", identity())
    with store.segment("segment-0001") as segment:
        with pytest.raises(CheckpointError, match="reserved"):
            segment.append_event("note", sequence=99)


def test_current_segment_active_time_and_resume_limit(tmp_path, adapter):
    run = identity()
    active = [
        {"kind": "segment_start", "segment_id": "segment-0001"},
        {"kind": "segment_terminal", "segment_id": "segment-0001",
         "terminal_segment_seconds": 100.0},
        {"kind": "resume_start", "segment_id": "segment-0002"},
    ]
    assert cumulative_active_seconds(active, 25.0, "segment-0002") == 125.0
    store = CheckpointStore.create(tmp_path / "run", run)
    with store.segment("segment-0001") as segment:
        publish(segment, run, adapter, 1)
        segment.finish("failed", 1.0)
    for number in (2, 3):
        with store.segment(f"segment-000{number}", resume=True,
                           preflight_metrics={"swap_used_bytes": 1000,
                                              "system_free_percent": 50.0}) as segment:
            segment.finish("failed", 1.0)
    with pytest.raises(CheckpointError, match="resume count"):
        with store.segment("segment-0004", resume=True,
                           preflight_metrics={"swap_used_bytes": 1000,
                                              "system_free_percent": 50.0}):
            pass


def test_final_tail_three_is_the_only_final_batch(tmp_path, adapter):
    run = identity(updates=1, final=3)
    store = CheckpointStore.create(tmp_path / "run", run)
    with store.segment("segment-0001") as segment:
        result = publish(segment, run, adapter, 1)
        assert result["manifest"]["completed_microbatches"] == 3
        assert len(result["manifest"]["batch_records"]) == 3
        assert result["manifest"]["next_expected"] is None


def test_update_time_breach_revokes_eligibility(tmp_path, adapter):
    run = identity()
    store = CheckpointStore.create(tmp_path / "run", run)
    with store.segment("segment-0001") as segment:
        with pytest.raises(CheckpointError, match="120 seconds"):
            segment.publish_checkpoint(
                codec=NumpyCodec(), adapter_tensors=adapter,
                optimizer_state=optimizer(adapter, 1), update=1,
                completed_microbatches=4,
                next_schedule={"role": "synthetic", "id": "row-5"},
                batch_records=[{"role": "synthetic", "id": f"row-{index}",
                                "microbatch": index, "update": 1}
                               for index in range(1, 5)],
                losses=[1.0] * 4, adapter_config=ADAPTER_CONFIG,
                optimizer_config=ADAM_CONFIG, recipe=run["recipe"],
                elapsed_seconds=1.0, update_elapsed_before_checkpoint=121.0,
                resource_callback=lambda stage: None)
        assert store._pointer()["status"] == "committed"
        with pytest.raises(CheckpointError, match="not resume-eligible"):
            store.load_eligible(codec=NumpyCodec())


def test_recipe_extras_are_closed():
    with pytest.raises(CheckpointError, match="recipe extras"):
        make_run_identity(
            logical_run_id="bad", mode="synthetic", baseline_swap_used_bytes=0,
            schedule_sha256=HASH, microbatches=4, updates=1,
            final_update_microbatches=4, source_hashes={"s": HASH},
            config_hashes={"c": HASH}, model_hashes={"m": HASH},
            package_versions={"mlx": "0.31.2"}, created_at="now",
            recipe_extra={"shuffle": True})


def test_fit_identity_requires_exact_reviewed_schedule():
    exact = make_run_identity(
        logical_run_id="fit", mode="fit", baseline_swap_used_bytes=0,
        schedule_sha256=HASH, microbatches=2771, updates=693,
        final_update_microbatches=3, source_hashes={"s": HASH},
        config_hashes={"c": HASH}, model_hashes={"m": HASH},
        package_versions={"mlx": "0.31.2"}, created_at="now")
    assert exact["schedule"]["updates"] == 693
    with pytest.raises(CheckpointError, match="FIT schedule"):
        make_run_identity(
            logical_run_id="fit-bad", mode="fit", baseline_swap_used_bytes=0,
            schedule_sha256=HASH, microbatches=4, updates=1,
            final_update_microbatches=4, source_hashes={"s": HASH},
            config_hashes={"c": HASH}, model_hashes={"m": HASH},
            package_versions={"mlx": "0.31.2"}, created_at="now")


def test_complete_terminal_cannot_resume(tmp_path, adapter):
    run = identity()
    store = CheckpointStore.create(tmp_path / "run", run)
    with store.segment("segment-0001") as segment:
        publish(segment, run, adapter, 1)
        segment.finish("complete", 1.0)
    with pytest.raises(CheckpointError, match="cannot resume"):
        with store.segment("segment-0002", resume=True,
                           preflight_metrics={"swap_used_bytes": 1000,
                                              "system_free_percent": 50.0}):
            pass


def test_terminal_cursor_must_match_pointer(tmp_path, adapter):
    run = identity()
    store = CheckpointStore.create(tmp_path / "run", run)
    with store.segment("segment-0001") as segment:
        publish(segment, run, adapter, 1)
        terminal = segment.finish("failed", 1.0)
    event_path = sorted((store.root / "ledger").glob("*.json"))[-1]
    changed = copy.deepcopy(terminal)
    changed["payload"]["eligible_update"] = 2
    changed = envelope(changed["payload"])
    event_path.write_bytes(canonical_bytes(changed))
    with pytest.raises(CheckpointError, match="cursor"):
        with store.segment("segment-0002", resume=True,
                           preflight_metrics={"swap_used_bytes": 1000,
                                              "system_free_percent": 50.0}):
            pass


def test_prune_receipt_fsync_retry_is_idempotent(tmp_path, adapter):
    run = identity()
    store = CheckpointStore.create(tmp_path / "run", run)
    with store.segment("segment-0001") as segment:
        publish(segment, run, adapter, 1)
        second = publish(segment, run, adapter, 2)

        def stop_after_receipt(stage):
            if stage == "post_prune_receipt_fsync":
                raise RuntimeError("power loss")

        with pytest.raises(RuntimeError, match="power loss"):
            segment.publish_checkpoint(
                codec=NumpyCodec(), adapter_tensors=adapter,
                optimizer_state=optimizer(adapter, 3), update=3,
                completed_microbatches=9, next_schedule=None,
                batch_records=[{"role": "synthetic", "id": "row-9",
                                "microbatch": 9, "update": 3}], losses=[1.0],
                adapter_config=ADAPTER_CONFIG, optimizer_config=ADAM_CONFIG,
                recipe=run["recipe"], elapsed_seconds=3.0,
                resource_callback=lambda stage: None, failure_hook=stop_after_receipt)
        assert store._pointer()["manifest_sha256"] == second["pointer"]["manifest_sha256"]
        segment.finish("failed", 1.0)
    with store.segment("segment-0002", resume=True,
                       preflight_metrics={"swap_used_bytes": 1000,
                                          "system_free_percent": 50.0}) as segment:
        third = publish(segment, run, adapter, 3)
        assert third["pointer"]["status"] == "eligible"
        assert len(store.audit_snapshot()["prune_receipts"]) == 2


def test_actual_mlx_safetensors_roundtrip_when_available(tmp_path, adapter):
    mx = pytest.importorskip("mlx.core")
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_unflatten
    from scripts.long_context_lora_checkpoint_v2 import MlxTensorCodec

    arrays = {name: mx.array(value) for name, value in adapter.items()}
    actual_optimizer = optim.Adam(learning_rate=1e-5, betas=[0.9, 0.999], eps=1e-8,
                                  bias_correction=True)
    actual_optimizer.init(tree_unflatten(list(arrays.items())))
    actual_optimizer.state["step"] = mx.array(1, mx.uint64)
    state = dict(tree_flatten(actual_optimizer.state))
    codec = MlxTensorCodec(mx)
    adapter_path, optimizer_path = tmp_path / "adapter.safetensors", tmp_path / "optimizer.safetensors"
    codec.save(adapter_path, arrays)
    codec.save(optimizer_path, state)
    mx.eval(arrays, state)
    adapter_inventory = validate_adapter_tensors(codec.load(adapter_path))
    validate_optimizer_tensors(codec.load(optimizer_path), 1, adapter_inventory, ADAM_CONFIG)


def test_adapter_config_tamper_blocks_restore(tmp_path, adapter):
    run = identity()
    store = CheckpointStore.create(tmp_path / "run", run)
    with store.segment("segment-0001") as segment:
        result = publish(segment, run, adapter, 1)
        segment.finish("failed", 1.0)
    (Path(result["path"]) / "adapter_config.json").write_text("{}")
    with store.segment("segment-0002", resume=True,
                       preflight_metrics={"swap_used_bytes": 1000,
                                          "system_free_percent": 50.0}):
        with pytest.raises(CheckpointError, match="payload hash"):
            store.load_eligible(codec=NumpyCodec())
