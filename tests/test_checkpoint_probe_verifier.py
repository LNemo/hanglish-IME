"""Mutation checks for the independently implemented checkpoint byte parser."""
import importlib.util
import json
from pathlib import Path
import struct

import pytest


PATH = Path(__file__).resolve().parents[1] / (
    ".omx/artifacts/long-context-lora-work-v2/verify_checkpoint_probe.py")
SPEC = importlib.util.spec_from_file_location("independent_checkpoint_probe_verifier", PATH)
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def write_payload(path, kind="optimizer", mutation=None):
    values = {}
    for name, shape in verifier.adapter_shapes().items():
        count = shape[0] * shape[1]
        keys = [name] if kind == "adapter" else [name + ".m", name + ".v"]
        for key in keys:
            values[key] = ("F32", shape, b"\0" * (4 * count))
    if kind == "optimizer":
        values.update(step=("U64", [], struct.pack("<Q", 2)),
                      learning_rate=("F32", [], struct.pack("<f", 1e-5)))
    if mutation:
        mutation(values)
    header, body = {"__metadata__": None}, bytearray()
    for key, (dtype, shape, data) in values.items():
        start = len(body)
        body.extend(data)
        header[key] = {"dtype": dtype, "shape": shape,
                       "data_offsets": [start, len(body)]}
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + body)


@pytest.mark.parametrize("kind,count", [("adapter", 16), ("optimizer", 34)])
def test_parser_checks_pinned_tensor_inventories_without_mlx(tmp_path, kind, count):
    path = tmp_path / "tensors.safetensors"
    write_payload(path, kind)
    assert len(verifier.inspect_tensors(path, kind=kind, update=2)) == count


@pytest.mark.parametrize("mutation", [
    lambda v: v.update(step=("U64", [], struct.pack("<Q", 1))),
    lambda v: v.update(learning_rate=("F32", [], struct.pack("<f", 2e-5))),
    lambda v: v.pop(next(iter(v))),
    lambda v: v.update(extra=("F32", [], struct.pack("<f", 0))),
    lambda v: v.update(learning_rate=("F32", [], struct.pack("<f", float("nan")))),
    lambda v: v.update(step=("F32", [], struct.pack("<f", 2))),
])
def test_optimizer_corruption_fails_closed(tmp_path, mutation):
    path = tmp_path / "state.safetensors"
    write_payload(path, mutation=mutation)
    with pytest.raises(verifier.VerificationError):
        verifier.inspect_tensors(path, kind="optimizer", update=2)


def test_finite_moment_change_changes_tensor_identity(tmp_path):
    path = tmp_path / "state.safetensors"
    write_payload(path)
    before = verifier.inspect_tensors(path, kind="optimizer", update=2)
    def mutate(values):
        key = next(iter(values))
        dtype, shape, raw = values[key]
        values[key] = (dtype, shape, struct.pack("<f", 1) + raw[4:])
    write_payload(path, mutation=mutate)
    after = verifier.inspect_tensors(path, kind="optimizer", update=2)
    assert before != after


def test_duplicate_json_keys_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"step": 2, "step": 3}')
    with pytest.raises(verifier.VerificationError, match="duplicate"):
        verifier.read_json(path)


def test_artifact_symlink_and_escape_rejected(tmp_path):
    target = tmp_path / "data"
    target.write_text("ok")
    link = tmp_path / "link"
    link.symlink_to(target)
    for name in ("link", "../data", str(target)):
        with pytest.raises(verifier.VerificationError):
            verifier.safe_child(tmp_path, name)


def resources(**changes):
    return {"elapsed_seconds": 900.0, "free_percent": 10.0,
            "mlx_active_bytes": 5, "mlx_cache_bytes": 256 * 1024**2,
            "mlx_peak_bytes": 6 * 1024**3, "rss_peak_bytes": 6 * 1024**3,
            "swap_growth_bytes": 512 * 1024**2,
            "swap_used_bytes": 1000 + 512 * 1024**2, **changes}


def test_original_resource_limits_pass_at_exact_boundary():
    verifier.verify_resource_metrics(resources(), 1000, cache=True)


@pytest.mark.parametrize("changes", [
    {"swap_growth_bytes": 512 * 1024**2 + 1},
    {"elapsed_seconds": 900.0001}, {"free_percent": 9.99},
    {"mlx_peak_bytes": 6 * 1024**3 + 1},
    {"rss_peak_bytes": 6 * 1024**3 + 1},
    {"mlx_cache_bytes": 256 * 1024**2 + 1},
    {"mlx_active_bytes": 6 * 1024**3 + 1},
])
def test_resource_breach_is_not_hidden_by_report_flags(changes):
    with pytest.raises(verifier.VerificationError):
        verifier.verify_resource_metrics(resources(**changes), 1000, cache=True)


def wrapped_write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"payload": payload, "digest": verifier.digest(payload)}))


def store_fixture(root):
    run = {"schema": "long-context-lora-run-v2", "logical_run_id": "fixture",
           "mode": "synthetic", "max_resumes": 2, "limits": verifier.LIMITS, "recipe": verifier.RECIPE,
           "baseline": {"swap_used_bytes": 1000}, "source_hashes": {"fixture": "f" * 64},
           "schedule": {"updates": 2, "microbatches": 8, "final_update_microbatches": 4}}
    wrapped_write(root / "run.json", run)
    (root / "prune-receipts").mkdir()
    events = []
    def event(kind, **fields):
        payload = {"sequence": len(events) + 1,
                   "prior_event_digest": verifier.digest(events[-1]) if events else None,
                   "logical_run_id": "fixture", "segment_id": "s0", "kind": kind, **fields}
        events.append(payload)
        wrapped_write(root / "ledger" / f"{len(events):08d}.json", payload)
        return payload
    event("segment_start")
    previous = None
    for update in (1, 2):
        relative = f"checkpoints/s0/update-{update:04d}"
        directory = root / relative
        directory.mkdir(parents=True)
        write_payload(directory / "adapter.safetensors", "adapter")
        write_payload(directory / "optimizer.safetensors", mutation=lambda v:
                      v.update(step=("U64", [], struct.pack("<Q", update))))
        (directory / "adapter_config.json").write_text(json.dumps(verifier.ADAPTER_CONFIG))
        inventories = {}
        for kind in ("adapter", "optimizer"):
            parsed = verifier.inspect_tensors(directory / f"{kind}.safetensors",
                                               kind=kind, update=update)
            inventories[kind] = {k: {"shape": v["shape"], "sha256": v["checkpoint_sha256"]}
                                 for k, v in parsed.items()}
        manifest = {"recipe": verifier.RECIPE, "optimizer_config": verifier.ADAM,
                    "adapter_config": verifier.ADAPTER_CONFIG, "run_json_sha256": verifier.sha(root / "run.json"),
                    "logical_run_id": "fixture", "segment_id": "s0", "update": update,
                    "completed_updates": update, "completed_microbatches": update * 4,
                    "expected_parent_manifest_sha256": previous,
                    "adapter_inventory": inventories["adapter"],
                    "optimizer_inventory": inventories["optimizer"],
                    "files": {p.name: {"size": p.stat().st_size, "sha256": verifier.sha(p)}
                              for p in directory.iterdir()}}
        manifest["manifest_digest"] = verifier.digest(manifest)
        wrapped_write(directory / "manifest.json", manifest)
        manifest_sha = verifier.sha(directory / "manifest.json")
        event("checkpoint_committed", update=update, manifest_sha256=manifest_sha,
              checkpoint_path=relative)
        last = event("resume_eligible", update=update, manifest_sha256=manifest_sha,
                     checkpoint_path=relative)
        previous = manifest_sha
    event("segment_terminal", terminal_segment_seconds=1.0, status="complete")
    wrapped_write(root / "latest.json", {"status": "eligible", "update": 2,
                  "path": relative, "manifest_sha256": previous,
                  "event_digest": verifier.digest(last)})
    return run


def test_independent_store_byte_and_cursor_verification(tmp_path):
    run = store_fixture(tmp_path)
    result = verifier.verify_store(tmp_path, updates=2, resumes=0,
                                   baseline_swap=1000, source_hashes=run["source_hashes"])
    assert result["retained_payloads_byte_verified"] == 2
    assert result["historical_pruned_payload_bytes_independently_verified"] is False


@pytest.mark.parametrize("mutation", ["missing_terminal", "pointer_old_event",
                                     "step_bytes", "baseline", "broken_chain"])
def test_store_mutations_cannot_pass_via_success_flags(tmp_path, mutation):
    run = store_fixture(tmp_path)
    if mutation == "missing_terminal":
        (tmp_path / "ledger/00000006.json").unlink()
    elif mutation == "pointer_old_event":
        pointer, _ = verifier.read_envelope(tmp_path / "latest.json")
        _, event_digest = verifier.read_envelope(tmp_path / "ledger/00000003.json")
        pointer["event_digest"] = event_digest
        wrapped_write(tmp_path / "latest.json", pointer)
    elif mutation == "step_bytes":
        write_payload(tmp_path / "checkpoints/s0/update-0002/optimizer.safetensors",
                      mutation=lambda v: v.update(step=("U64", [], struct.pack("<Q", 1))))
    elif mutation == "baseline":
        changed = {**run, "baseline": {"swap_used_bytes": 2000}}
        wrapped_write(tmp_path / "run.json", changed)
    else:
        path = tmp_path / "ledger/00000004.json"
        value, _ = verifier.read_envelope(path)
        value["prior_event_digest"] = "0" * 64
        wrapped_write(path, value)
    with pytest.raises(verifier.VerificationError):
        verifier.verify_store(tmp_path, updates=2, resumes=0,
                              baseline_swap=1000, source_hashes=run["source_hashes"])


def complete_observations():
    return [{"stage": stage, "action": "none", "pre": resources(elapsed_seconds=i / 100),
             "post": None, "errors": [], "cleanup_seconds": 0,
             "checkpoint_seconds": .001}
            for i, stage in enumerate(verifier.expected_observation_stages())]


def test_resource_sequence_includes_every_interruption_and_resume():
    observations = complete_observations()
    assert verifier.verify_observations(observations, 1000) == (len(observations) - 1) / 100


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "order", "time", "reset_peak"])
def test_resource_sequence_mutations_are_rejected(mutation):
    rows = complete_observations()
    if mutation == "missing":
        rows.pop(100)
    elif mutation == "duplicate":
        rows.insert(100, rows[100])
    elif mutation == "order":
        rows[100], rows[101] = rows[101], rows[100]
    elif mutation == "time":
        rows[100]["pre"]["elapsed_seconds"] = 0
    else:
        rows[100]["pre"]["mlx_peak_bytes"] -= 1
    with pytest.raises(verifier.VerificationError):
        verifier.verify_observations(rows, 1000)


def test_projection_includes_checkpoint_cost_and_rejects_slow_updates():
    assert verifier.runtime_estimate([10] * 20, [4] * 40)["within_21600_seconds"]
    assert not verifier.runtime_estimate([20] * 20, [4] * 40)["within_21600_seconds"]
    with pytest.raises(verifier.VerificationError):
        verifier.runtime_estimate([121] * 20, [4] * 40)
