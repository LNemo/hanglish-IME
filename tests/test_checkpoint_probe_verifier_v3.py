"""CPU-only mutation tests for the independent validation-v3 verifier."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import struct

import pytest

PATH = Path(__file__).resolve().parents[1] / ".omx/artifacts/long-context-lora-work-v3/verify_checkpoint_probe.py"
SPEC = importlib.util.spec_from_file_location("checkpoint_probe_verifier_v3", PATH)
v = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(v)


def write_payload(path, kind="optimizer", update=2, mutation=None):
    tensors = {}
    for name, shape in v.adapter_shapes().items():
        keys = [name] if kind == "adapter" else [name + ".m", name + ".v"]
        for key in keys: tensors[key] = ("F32", shape, b"\0" * (4 * math.prod(shape)))
    if kind == "optimizer":
        tensors.update(step=("U64", [], struct.pack("<Q", update)),
                       learning_rate=("F32", [], struct.pack("<f", 1e-5)))
    if mutation: mutation(tensors)
    header, body = {"__metadata__": None}, bytearray()
    for key, (dtype, shape, data) in tensors.items():
        begin = len(body); body.extend(data)
        header[key] = {"dtype": dtype, "shape": shape, "data_offsets": [begin, len(body)]}
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + body)


@pytest.mark.parametrize("kind,count", [("adapter", 16), ("optimizer", 34)])
def test_raw_tensor_contract_without_mlx(tmp_path, kind, count):
    path = tmp_path / "state"; write_payload(path, kind)
    assert len(v.inspect_tensors(path, kind=kind, update=2)) == count


@pytest.mark.parametrize("mutation", [
    lambda x: x.update(step=("U64", [], struct.pack("<Q", 1))),
    lambda x: x.update(learning_rate=("F32", [], struct.pack("<f", 2e-5))),
    lambda x: x.pop(next(iter(x))), lambda x: x.update(extra=("F32", [], b"\0" * 4)),
    lambda x: x.update(learning_rate=("F32", [], struct.pack("<f", float("nan")))),
])
def test_raw_tensor_mutations_fail_closed(tmp_path, mutation):
    path = tmp_path / "state"; write_payload(path, mutation=mutation)
    with pytest.raises(v.VerificationError): v.inspect_tensors(path, kind="optimizer", update=2)


def metric(elapsed=0, **changes):
    return {"elapsed_seconds": elapsed, "free_percent": 50.0, "mlx_active_bytes": 1,
            "mlx_cache_bytes": 1, "mlx_peak_bytes": 1, "rss_peak_bytes": 1,
            "swap_growth_bytes": 0, "swap_used_bytes": 1000, **changes}


def observation_fixture():
    return [{"stage": stage, "action": "none", "pre": metric(i / 100), "post": None,
             "errors": [], "cleanup_seconds": 0.0, "checkpoint_seconds": .001}
            for i, stage in enumerate(v.expected_observation_stages())]


def test_exact_497_resource_observations_pass():
    rows = observation_fixture(); assert len(rows) == 497
    assert v.verify_observations(rows, 1000) == pytest.approx(4.96)


@pytest.mark.parametrize("mutation", ("missing", "reordered", "time", "peak", "stage"))
def test_resource_observation_mutations_fail(mutation):
    rows = observation_fixture()
    if mutation == "missing": rows.pop(10)
    elif mutation == "reordered": rows[10], rows[11] = rows[11], rows[10]
    elif mutation == "time": rows[10]["pre"]["elapsed_seconds"] = -1
    elif mutation == "peak": rows[10]["pre"]["mlx_peak_bytes"] = 0
    else: rows[-1]["stage"] = "failure_terminal"
    with pytest.raises(v.VerificationError): v.verify_observations(rows, 1000)


def run_fixture(source_hashes=None):
    descriptor = {"updates": 4, "accumulation": 4,
                  "shapes": [[256, 2, [0]], [512, 3, [0, 2]],
                             [768, 4, [1, 3]], [1024, 5, [0, 2, 4]]]}
    return {"schema": "long-context-lora-run-v2", "logical_run_id": "fixture",
            "created_at": "2026-09-15T00:00:00+00:00", "mode": "synthetic",
            "baseline": {"swap_used_bytes": 1000}, "limits": v.LIMITS, "max_resumes": 2,
            "schedule": {"sha256": __import__("hashlib").sha256(v.canonical(descriptor)).hexdigest(), "microbatches": 16, "updates": 4,
                         "accumulation": 4, "final_update_microbatches": 4},
            "recipe": copy.deepcopy(v.RECIPE), "source_hashes": source_hashes or {"source": "b" * 64},
            "config_hashes": {"plan": v.APPROVED_SOURCE_SHA256[".omx/plans/long-context-lora-validation-v3.md"],
                              "test_spec": v.APPROVED_SOURCE_SHA256[".omx/plans/long-context-lora-validation-v3-test-spec.md"],
                              "handoff": v.APPROVED_SOURCE_SHA256[".omx/plans/long-context-lora-validation-v3-handoff.json"]},
            "model_hashes": {"SOURCE.json": "67928c796b7e4562c072b2bbbdd0d52569e5ab9118ade1302f8d8d1bb4bae819",
                             "model.safetensors": "e56a94b846a2dce7d1cc154ce58b9cfc4b8d99e8ffb74bfe256a265f4cbc7fb9"},
            "package_versions": copy.deepcopy(v.EXPECTED_PACKAGES)}


def test_exact_run_schema_and_schedule_pass():
    run = run_fixture(); v.verify_run(run, updates=4, baseline_swap=1000,
                                      source_hashes=run["source_hashes"])


@pytest.mark.parametrize("mutation", ("extra", "schedule", "package", "baseline", "recipe"))
def test_run_identity_mutations_fail(mutation):
    run = run_fixture()
    if mutation == "extra": run["caller_approved"] = True
    elif mutation == "schedule": run["schedule"]["microbatches"] = 15
    elif mutation == "package": run["package_versions"]["mlx"] = "new"
    elif mutation == "baseline": run["baseline"]["swap_used_bytes"] += 1
    else: run["recipe"]["seed"] += 1
    with pytest.raises(v.VerificationError):
        v.verify_run(run, updates=4, baseline_swap=1000, source_hashes={"source": "b" * 64})


def phase_report(tmp_path):
    report = {"reference": {"r": 1}, "main": {"m": 2},
              "branches": [{"b": n} for n in range(3)],
              "runtime_estimate": {"ok": True}, "scorer_timings_seconds": [.1] * 40,
              "predicate_reads": {"optimized_observed": {"predicate_host_reads": 199, "numeric_host_reads": 167},
                                  "optimized_expected": {"predicate_host_reads": 199, "numeric_host_reads": 167},
                                  "original_reference_expected_static": {"predicate_host_reads": 421, "numeric_host_reads": 21},
                                  "predicate_scope": v.PREDICATE_SCOPE, "numeric_scope": v.NUMERIC_SCOPE,
                                  "other_host_reads": v.OTHER_HOST_READS},
              "durable_phases": []}
    payloads = [report["reference"], report["main"], *report["branches"],
                {k: report[k] for k in ("runtime_estimate", "scorer_timings_seconds",
                                         "predicate_reads")}]
    prior = None
    for sequence, (phase, payload) in enumerate(zip(v.PHASES, payloads), 1):
        snapshot = {"schema": "long-context-lora-phase-snapshot-v3", "sequence": sequence,
                    "phase": phase, "prior_snapshot_sha256": prior, "payload": payload}
        relative = f"phase-snapshots/{sequence:04d}-{phase}.json"; path = tmp_path / relative
        path.parent.mkdir(exist_ok=True); path.write_bytes(v.canonical(snapshot) + b"\n")
        prior = v.sha(path); report["durable_phases"].append(
            {"sequence": sequence, "phase": phase, "path": relative, "sha256": prior})
    return report


def test_full_durable_phase_chain_and_predicate_counts_pass(tmp_path):
    report = phase_report(tmp_path); v.verify_predicate_reads(report["predicate_reads"])
    v.verify_durable_phases(tmp_path, report)


@pytest.mark.parametrize("mutation", ("hash", "prior", "payload", "count", "scope"))
def test_phase_and_predicate_mutations_fail(tmp_path, mutation):
    report = phase_report(tmp_path)
    if mutation == "hash": report["durable_phases"][0]["sha256"] = "0" * 64
    elif mutation in {"prior", "payload"}:
        path = tmp_path / report["durable_phases"][2]["path"]; snap = v.read_json(path)
        snap["prior_snapshot_sha256" if mutation == "prior" else "payload"] = "tampered"
        path.write_bytes(v.canonical(snap) + b"\n"); report["durable_phases"][2]["sha256"] = v.sha(path)
    elif mutation == "count": report["predicate_reads"]["optimized_observed"]["predicate_host_reads"] += 1
    else: report["predicate_reads"]["predicate_scope"] += " changed"
    with pytest.raises(v.VerificationError):
        v.verify_predicate_reads(report["predicate_reads"]); v.verify_durable_phases(tmp_path, report)


def test_update_cursor_batch_position_and_payload_binding_are_strict(monkeypatch):
    record = {"update": 1, "microbatches": 4, "microbatch_losses": [1., 2., 3., 4.],
              "mean_loss": 2.5, "gradient_norm_before_clip": 1., "seconds": 1.,
              "checkpoint_seconds": .1, "checkpoint": {},
              "cursor": {"completed_updates": 1, "completed_microbatches": 4,
                         "next_update": 2, "next_microbatch": 1}}
    v.verify_update_records([record], 1)
    for key in ("completed_microbatches", "next_update"):
        changed = copy.deepcopy(record); changed["cursor"][key] += 1
        with pytest.raises(v.VerificationError): v.verify_update_records([changed], 1)


def test_reference_checkpoint_inventory_drift_is_rejected():
    def store(marker):
        return {"manifests": {str(n): ({"update": n,
                "adapter_inventory": {"x": marker}, "optimizer_inventory": {"x": marker}},
                str(n) * 64) for n in range(1, 5)}}
    v.verify_reference_checkpoint_inventories(store("same"), store("same"))
    changed = store("same"); changed["manifests"]["2"][0]["optimizer_inventory"]["x"] = "drift"
    with pytest.raises(v.VerificationError, match="inventory differs"):
        v.verify_reference_checkpoint_inventories(store("same"), changed)


def test_duplicate_json_and_unsafe_artifact_paths_fail(tmp_path):
    bad = tmp_path / "bad.json"; bad.write_text('{"x":1,"x":2}')
    with pytest.raises(v.VerificationError): v.read_json(bad)
    target = tmp_path / "target"; target.write_text("x"); (tmp_path / "link").symlink_to(target)
    for name in ("link", "../target", str(target)):
        with pytest.raises(v.VerificationError): v.safe_child(tmp_path, name)


def test_all_approved_source_pins_are_frozen_and_match_disk():
    assert all(v.is_sha256(value) for value in v.APPROVED_SOURCE_SHA256.values())
    assert all(v.sha(v.ROOT / name) == expected
               for name, expected in v.APPROVED_SOURCE_SHA256.items())


def test_full_top_level_success_schema_is_wired(tmp_path, monkeypatch):
    """Exercise the complete report orchestration; lower-level byte paths are tested above."""
    base = v.read_json(v.ROOT / ".omx/artifacts/long-context-lora-resource-v1/report.json")["source_hashes"]
    monkeypatch.setattr(v, "verify_sources", lambda before, after: None)
    monkeypatch.setattr(v, "verify_observations", lambda rows, swap: 4.96)
    monkeypatch.setattr(v, "verify_math_and_reload", lambda report: None)
    monkeypatch.setattr(v, "bind_update_manifests", lambda records, store: None)
    monkeypatch.setattr(v, "bind_checkpoint_resources", lambda *args, **kwargs: None)
    monkeypatch.setattr(v, "verify_durable_phases", lambda directory, report: None)
    tensor_hashes = {"tiny": "a" * 64}
    monkeypatch.setattr(v, "inspect_tensors", lambda *args, **kwargs:
                        {"tiny": {"sha256": "a" * 64, "checkpoint_sha256": "b" * 64}})
    stores = []
    def fake_store(root, *, updates, resumes, baseline_swap, source_hashes):
        name = Path(root).name
        segments = ["segment-0001"] + (["segment-0002"] if resumes else [])
        events = [{"kind": "resume_eligible", "update": updates,
                   "checkpoint_path": f"checkpoints/{segments[-1]}/update-{updates:04d}"}]
        events += [{"kind": "segment_terminal", "segment_id": segment,
                    "terminal_segment_seconds": 2.0,
                    "terminal_close_reserve_seconds": 1.0} for segment in segments]
        manifests = {f"checkpoint-{n}": ({"update": n,
                     "adapter_inventory": {"tiny": {"sha256": "b" * 64}},
                     "optimizer_inventory": {"tiny": {"sha256": "b" * 64}}}, "f" * 64)
                     for n in range(1, updates + 1)}
        result = {"events": events, "manifests": manifests, "terminal_active_seconds": 2 * len(segments)}
        stores.append((name, result)); return result
    monkeypatch.setattr(v, "verify_store", fake_store)
    def record(n):
        losses = [float(n)] * 4
        return {"update": n, "microbatches": 4, "microbatch_losses": losses,
                "mean_loss": float(n), "gradient_norm_before_clip": 1.0,
                "seconds": 1.0, "checkpoint_seconds": .1, "checkpoint": {},
                "cursor": {"completed_updates": n, "completed_microbatches": n * 4,
                           "next_update": n + 1, "next_microbatch": 1}}
    main_updates = [record(n) for n in range(1, 21)]
    reference_updates = copy.deepcopy(main_updates[:4])
    def probe_logits():
        return [{"prompt_sha256": h, "prompt_token_count": size, "candidate_count": count,
                 "logits": list(map(float, range(count))), "argmax": count - 1}
                for h, size, count in zip(v.PROMPT_HASHES, (256, 512, 768, 1024), (2, 3, 4, 5))]
    def state(n):
        row = main_updates[n - 1]
        return {"update": n, "adapter_tensor_hashes": tensor_hashes,
                "optimizer_tensor_hashes": tensor_hashes, "optimizer_step": n,
                "cursor": row["cursor"], "microbatch_losses": row["microbatch_losses"],
                "mean_loss": row["mean_loss"], "gradient_norm_before_clip": 1.0}
    states = [state(n) for n in range(1, 5)]
    update4 = {**states[3], "source_only_logits": probe_logits(),
               "model_mode_restored_to_train": True}
    main = {"updates": main_updates, "states": states,
            "update_timings_seconds": [1.0] * 20,
            "initial_trainable_tensor_hashes": {"initial": "0" * 64},
            "final_trainable_tensor_hashes": tensor_hashes,
            "reloaded_trainable_tensor_hashes": tensor_hashes,
            "update4_reference": update4}
    reference = {"updates": reference_updates, "states": copy.deepcopy(states[:4]),
                 "initial_trainable_tensor_hashes": {"initial": "0" * 64},
                 "update4_reference": copy.deepcopy(update4)}
    branches = []
    for count in range(1, 4):
        rows = copy.deepcopy(main_updates[:4])
        branches.append({"interrupted_after_microbatches": count, "resume_count": 1,
                         "prefix_updates": rows[:2], "resumed_updates": rows[2:], "updates": rows,
                         "partial_update": {"update": 3, "computed_microbatches": count,
                            "checkpoint_published": False, "partial_accumulator_discarded": True,
                            "durable_cursor_remains": rows[1]["cursor"],
                            "losses": rows[2]["microbatch_losses"][:count]},
                         "final_state": copy.deepcopy(update4)})
    closures = []
    for name, segments in (("reference", ["segment-0001"]), ("main", ["segment-0001"]),
                           *[(f"restart-after-{n}", ["segment-0001", "segment-0002"])
                             for n in range(1, 4)]):
        closures += [{"run_relative_path": name, "segment_id": segment,
                      "reserve_seconds": 1.0, "actual_close_seconds": .1,
                      "charged_terminal_seconds": 2.0, "completed_within_reserve": True}
                     for segment in segments]
        for close in closures[-len(segments):]:
            close["terminal_event_digest"] = v.digest({"kind":"segment_terminal","segment_id":close["segment_id"],"terminal_segment_seconds":2.0,"terminal_close_reserve_seconds":1.0})
    estimate = v.runtime_estimate([1.0] * 20, [.1] * 40)
    report = {"schema": "long-context-lora-resource-checkpoint-v3", "completed": True,
              "resource_feasible": True, "terminal_status": "success", "synthetic_only": True,
              **{key: False for key in ("real_fit_rows_read", "labels_read", "v1_prediction_cache_read",
                                         "real_training_run", "old_adapter_loaded", "production_modified",
                                         "system_swap_attribution")},
              "source_hashes": base, "source_hashes_after": base,
              "versions": v.EXPECTED_PACKAGES, "versions_after": v.EXPECTED_PACKAGES,
              "limits": v.LIMITS, "system_baseline": {"swap_used_bytes": 1000},
              "resource_observations": [{"stage": "terminal", "pre": metric(4.96), "post": None}],
              "active_gate": "terminal", "last_resource_stage": "terminal",
              "system_terminal": metric(4.96), "elapsed_seconds": 10.0,
              "predicate_reads": {"optimized_observed": {"predicate_host_reads": 199, "numeric_host_reads": 167},
                                  "optimized_expected": {"predicate_host_reads": 199, "numeric_host_reads": 167},
                                  "original_reference_expected_static": {"predicate_host_reads": 421, "numeric_host_reads": 21},
                                  "predicate_scope": v.PREDICATE_SCOPE, "numeric_scope": v.NUMERIC_SCOPE,
                                  "other_host_reads": v.OTHER_HOST_READS},
              "reference": reference, "main": main, "branches": branches,
              "warm_1024_backward": {"completed": True, "update_applied": False, "loss": 1.0,
                                       "finite_gradients": True, "nonzero_gradient": True},
              "original_vs_grouped_parity": {"passed": True,
                  "per_update_exact_state": [{"update": n, "passed": True} for n in range(1, 5)],
                  "update4_logits_and_state": {"passed": True}},
              "segment_closures": closures, "scorer_timings_seconds": [.1] * 40,
              "runtime_estimate": estimate}
    reference["warm_1024_backward"] = copy.deepcopy(report["warm_1024_backward"])
    write_completed_report(tmp_path,report)
    assert v.verify(tmp_path)["passed"] is True
    original=copy.deepcopy(report["segment_closures"][0])
    for field,value in (("reserve_seconds",2.0),("terminal_event_digest","0"*64)):
        report["segment_closures"][0]=copy.deepcopy(original); report["segment_closures"][0][field]=value
        write_completed_report(tmp_path,report)
        with pytest.raises(v.VerificationError): v.verify(tmp_path)


def envelope(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(v.canonical({"payload": payload, "digest": v.digest(payload)}))


def write_completed_report(root,report):
    close={"active_elapsed_observed_seconds":10.0,"reserve_seconds":1.0,
           "charged_elapsed_seconds":900.0,"completion_receipt_required":True,
           "reservation_scope":"report rename/directory fsync and completion receipt atomic write"}
    report.update(final_report_close=close,completion_receipt_path="completion-receipt.json")
    path=root/"report.json"; path.write_bytes(v.canonical(report)+b"\n")
    receipt={"schema":"long-context-lora-v3-completion-receipt","report_sha256":v.sha(path),
             "final_report_close":close,"report_elapsed_through_fsync_seconds":10.1}
    (root/"completion-receipt.json").write_bytes(v.canonical(receipt)+b"\n")


def disk_store(root, total, interruption, records, sources, observations):
    descriptor = {"updates": total, "accumulation": 4,
                  "shapes": [[256, 2, [0]], [512, 3, [0, 2]],
                             [768, 4, [1, 3]], [1024, 5, [0, 2, 4]]]}
    logical = "synthetic-" + (root.name.removeprefix("restart-after-").join(("restart-", ""))
                               if root.name.startswith("restart") else root.name)
    run = run_fixture(sources); run.update(logical_run_id=logical)
    run["schedule"].update(updates=total, microbatches=total * 4,
                           sha256=__import__("hashlib").sha256(v.canonical(descriptor)).hexdigest())
    envelope(root / "run.json", run); (root / "prune-receipts").mkdir()
    events = []
    def event(segment, kind, **extra):
        item = {"sequence": len(events) + 1, "prior_event_digest": v.digest(events[-1]) if events else None,
                "logical_run_id": logical, "segment_id": segment, "kind": kind, **extra}
        events.append(item); envelope(root / "ledger" / f"{len(events):08d}.json", item); return item
    segment = "segment-0001"; event(segment, "segment_start", resume_count=0)
    previous = None; entries = []; manifests = []
    for update in range(1, total + 1):
        if interruption and update == 3:
            event(segment, "segment_terminal", status="failed", terminal_segment_seconds=2.0,
                  completed_updates=2, interrupted_after_microbatches=interruption,
                  terminal_close_reserve_seconds=1.0, pointer_status="eligible",
                  committed_update=2, committed_manifest_sha256=previous,
                  eligible_update=2, eligible_manifest_sha256=previous)
            segment = "segment-0002"; event(segment, "resume_start", resume_count=1)
        rel = f"checkpoints/{segment}/update-{update:04d}"; directory = root / rel; directory.mkdir(parents=True)
        write_payload(directory / "adapter.safetensors", "adapter", update)
        write_payload(directory / "optimizer.safetensors", "optimizer", update)
        (directory / "adapter_config.json").write_bytes(v.canonical(v.ADAPTER_CONFIG))
        inv = {}
        for kind in ("adapter", "optimizer"):
            parsed = v.inspect_tensors(directory / f"{kind}.safetensors", kind=kind, update=update)
            inv[kind] = {k: {"dtype": "uint64" if k=="step" else "float32",
                             "shape": x["shape"], "size": math.prod(x["shape"]),
                             "sha256": x["checkpoint_sha256"]} for k, x in parsed.items()}
        batch = [{"role": "synthetic", "id": f"update-{update:04d}-microbatch-{n:04d}",
                  "update": update, "microbatch": 4 * (update - 1) + n, "position": n,
                  "prompt_sha256": v.PROMPT_HASHES[n - 1], "prompt_tokens": (256,512,768,1024)[n-1]}
                 for n in range(1, 5)]
        files = {p.name: {"size": p.stat().st_size, "sha256": v.sha(p)} for p in directory.iterdir()}
        head = v.digest(events[-1]); manifest = {
            "schema": "long-context-lora-checkpoint-v2", "logical_run_id": logical, "segment_id": segment,
            "update": update, "completed_updates": update, "completed_microbatches": update * 4,
            "next_schedule_position": update * 4, "next_expected": None if update == total else
                {"role": "synthetic", "id": f"update-{update+1:04d}-microbatch-0001"},
            "batch_records": batch, "losses": records[update-1]["microbatch_losses"],
            "adapter_inventory": inv["adapter"], "optimizer_inventory": inv["optimizer"],
            "adapter_config": v.ADAPTER_CONFIG, "optimizer_config": v.ADAM, "recipe": v.RECIPE,
            "run_json_sha256": v.sha(root / "run.json"), "event_chain_head": head,
            "elapsed_seconds": float(update), "expected_parent_manifest_sha256": previous,
            "expected_parent_event_digest": head, "files": files}
        manifest["manifest_digest"] = v.digest(manifest); envelope(directory / "manifest.json", manifest)
        manifest_sha = v.sha(directory / "manifest.json")
        commit = event(segment, "checkpoint_committed", update=update, manifest_sha256=manifest_sha,
                       checkpoint_path=rel, parent_manifest_sha256=previous)
        eligible = event(segment, "resume_eligible", update=update, manifest_sha256=manifest_sha,
                         checkpoint_path=rel)
        entry = {"path": rel, "manifest_sha256": manifest_sha, "update": update,
                 "segment_id": segment, "eligible_event_digest": v.digest(eligible)}
        entries.append(entry); pointer = {"schema": "long-context-lora-checkpoint-v2",
            **{k: entry[k] for k in ("path","manifest_sha256","update","segment_id")},
            "status": "eligible", "event_digest": v.digest(eligible), "commit_event_digest": v.digest(commit),
            "parent_manifest_sha256": previous, "retained": list(reversed(entries[-2:]))}
        stages = ("post_materialize","post_reload","precommit","postcommit","final_eligibility")
        occurrence = 0 if root.name in {"reference","main"} else interruption
        prefix = "reference_" if root.name == "reference" else ""
        samples=[]
        for stage in stages:
            matches=[x for x in observations if x["stage"]==f"{prefix}update_{update}_checkpoint_{stage}"]
            samples.append({"stage":stage,"metrics":copy.deepcopy(matches[occurrence]["pre"])})
        records[update-1]["checkpoint"]={"manifest":manifest,"pointer":pointer,"path":str(directory),
            "timings":{x:.001 for x in stages+('total',)},"resource_observations":samples,
            "payload_bytes":sum(x["size"] for x in files.values())+len(v.canonical({"payload":manifest,"digest":v.digest(manifest)})),
            "update_seconds":.5}
        manifests.append((rel, manifest, manifest_sha)); previous=manifest_sha
    event(segment,"segment_terminal",status="complete",terminal_segment_seconds=2.0,
          completed_updates=total,terminal_close_reserve_seconds=1.0,pointer_status="eligible",
          committed_update=total,committed_manifest_sha256=previous,eligible_update=total,eligible_manifest_sha256=previous)
    envelope(root / "latest.json", pointer)
    for rel, manifest, msha in manifests[:-2]:
        deletion=[]
        for name in ("adapter.safetensors","optimizer.safetensors","adapter_config.json"):
            path=root/rel/name; deletion.append({"name":name,"size":path.stat().st_size,"sha256":v.sha(path)}); path.unlink()
        child=manifests[manifest["update"]+1]
        receipt={"schema":"long-context-lora-checkpoint-v2","kind":"prune_receipt","victim_path":rel,
                 "victim_manifest_sha256":msha,"parent_manifest_sha256":msha,
                 "child_manifest_sha256":child[2],"new_checkpoint_path":child[0],"deletion":deletion}
        child_segment=Path(child[0]).parts[1]
        filename=f"{child_segment}-update-{manifest['update']+2:04d}-prunes-{manifest['segment_id']}-{manifest['update']:04d}.json"
        envelope(root/"prune-receipts"/filename,receipt)


def test_five_disk_stores_pass_all_unmocked_store_checks(tmp_path, monkeypatch):
    rows=observation_fixture(); sources=v.read_json(v.ROOT/".omx/artifacts/long-context-lora-resource-v1/report.json")["source_hashes"]
    all_records={}
    for name,total,interrupt in (("reference",4,0),("main",20,0),
                                 ("restart-after-1",4,1),("restart-after-2",4,2),("restart-after-3",4,3)):
        records=[{"update":n,"microbatches":4,"microbatch_losses":[float(n)]*4,"mean_loss":float(n),
                  "gradient_norm_before_clip":1.,"seconds":1.,"checkpoint_seconds":.1,"checkpoint":{},
                  "cursor":{"completed_updates":n,"completed_microbatches":n*4,"next_update":n+1,"next_microbatch":1}}
                 for n in range(1,total+1)]
        disk_store(tmp_path/name,total,interrupt,records,sources,rows)
        all_records[name]=records
        store=v.verify_store(tmp_path/name,updates=total,resumes=bool(interrupt),baseline_swap=1000,source_hashes=sources)
        v.bind_update_manifests(records,store)
        v.bind_checkpoint_resources(records,rows,"reference_" if name=="reference" else "",
                                    0 if name in {"reference","main"} else interrupt)
    monkeypatch.setattr(v,"verify_sources",lambda before,after: None)
    def hashes(name,update,kind):
        segment="segment-0002" if name.startswith("restart") else "segment-0001"
        parsed=v.inspect_tensors(tmp_path/name/f"checkpoints/{segment}/update-{update:04d}/{kind}.safetensors",kind=kind,update=update)
        return {k:x["sha256"] for k,x in parsed.items()}
    def logits(): return [{"prompt_sha256":h,"prompt_token_count":n,"candidate_count":c,"logits":[float(x) for x in range(c)],"argmax":c-1} for h,n,c in zip(v.PROMPT_HASHES,(256,512,768,1024),(2,3,4,5))]
    def state(n):
        r=all_records["main"][n-1]; return {"update":n,"adapter_tensor_hashes":hashes("reference",4,"adapter"),"optimizer_tensor_hashes":hashes("reference",4,"optimizer"),"optimizer_step":n,"cursor":r["cursor"],"microbatch_losses":r["microbatch_losses"],"mean_loss":r["mean_loss"],"gradient_norm_before_clip":1.}
    states=[state(n) for n in range(1,5)]; u4={**states[3],"source_only_logits":logits(),"model_mode_restored_to_train":True}
    warm={"completed":True,"update_applied":False,"loss":1.,"finite_gradients":True,"nonzero_gradient":True}
    reference={"updates":all_records["reference"],"states":states,"update4_reference":u4,"initial_trainable_tensor_hashes":{"i":"0"*64},"warm_1024_backward":warm}
    main={"updates":all_records["main"],"states":copy.deepcopy(states),"update4_reference":copy.deepcopy(u4),"initial_trainable_tensor_hashes":{"i":"0"*64},"final_trainable_tensor_hashes":hashes("main",20,"adapter"),"reloaded_trainable_tensor_hashes":hashes("main",20,"adapter"),"update_timings_seconds":[1.]*20}
    ordinary=[]; reload=[]
    for h,row in zip(v.PROMPT_HASHES,logits()):
        ordinary.append({"prompt_sha256":h,"selected_scorer_logits":row["logits"],"ordinary_forward_logits":row["logits"],"max_absolute_digit_delta":0.})
        reload.append({"prompt_sha256":h,"expected_logits":row["logits"],"actual_logits":row["logits"],"max_absolute_delta":0.})
    main.update(ordinary_forward_diagnostic=ordinary,fresh_checkpoint_reload={"checks":reload,"atol":1e-5,"rtol":1e-5})
    branches=[]
    for n in range(1,4):
        rs=all_records[f"restart-after-{n}"]; branches.append({"interrupted_after_microbatches":n,"resume_count":1,"prefix_updates":rs[:2],"resumed_updates":rs[2:],"updates":rs,"partial_update":{"update":3,"computed_microbatches":n,"checkpoint_published":False,"partial_accumulator_discarded":True,"durable_cursor_remains":rs[1]["cursor"],"losses":rs[2]["microbatch_losses"][:n]},"final_state":copy.deepcopy(u4)})
    closures=[]
    for name,segs in (("reference",1),("main",1),("restart-after-1",2),("restart-after-2",2),("restart-after-3",2)):
        for i in range(1,segs+1):
            segment=f"segment-{i:04d}"; events=[v.read_envelope(p)[0] for p in sorted((tmp_path/name/"ledger").glob("*.json"))]; terminal=next(e for e in events if e["kind"]=="segment_terminal" and e["segment_id"]==segment)
            closures.append({"run_relative_path":name,"segment_id":segment,"reserve_seconds":1.,"actual_close_seconds":.1,"charged_terminal_seconds":2.,"completed_within_reserve":True,"terminal_event_digest":v.digest(terminal)})
    tiny={"nonfinite_norm_rejected":True,"losses":[{"scores":[0.,math.log(2),math.log(3)],"gold":g,"expected":x,"actual":x} for g,x in zip(([2],[0,2],[0,1,2]),(math.log(2),math.log(1.5),0.))],"gradient_fixtures":{x:{"norm":5.,"clipped":[.6,.8]} for x in ("3","4")}}
    pred={"optimized_observed":{"predicate_host_reads":199,"numeric_host_reads":167},"optimized_expected":{"predicate_host_reads":199,"numeric_host_reads":167},"original_reference_expected_static":{"predicate_host_reads":421,"numeric_host_reads":21},"predicate_scope":v.PREDICATE_SCOPE,"numeric_scope":v.NUMERIC_SCOPE,"other_host_reads":v.OTHER_HOST_READS}; estimate=v.runtime_estimate([1.]*20,[.1]*40)
    report={"schema":"long-context-lora-resource-checkpoint-v3","completed":True,"resource_feasible":True,"terminal_status":"success","synthetic_only":True,**{k:False for k in ("real_fit_rows_read","labels_read","v1_prediction_cache_read","real_training_run","old_adapter_loaded","production_modified","system_swap_attribution")},"source_hashes":sources,"source_hashes_after":sources,"versions":v.EXPECTED_PACKAGES,"versions_after":v.EXPECTED_PACKAGES,"limits":v.LIMITS,"system_baseline":{"swap_used_bytes":1000},"resource_observations":rows,"elapsed_seconds":10.,"active_gate":"terminal","last_resource_stage":"terminal","system_terminal":rows[-1]["pre"],"tiny_math_validation":tiny,"reference":reference,"main":main,"warm_1024_backward":warm,"branches":branches,"segment_closures":closures,"scorer_timings_seconds":[.1]*40,"runtime_estimate":estimate,"predicate_reads":pred,"original_vs_grouped_parity":{"passed":True,"per_update_exact_state":[{"update":n,"passed":True} for n in range(1,5)],"update4_logits_and_state":{"passed":True}},"durable_phases":[]}
    payloads=[reference,main,*branches,{"runtime_estimate":estimate,"scorer_timings_seconds":[.1]*40,"predicate_reads":pred}]; prior=None
    for i,(phase,payload) in enumerate(zip(v.PHASES,payloads),1):
        snap={"schema":"long-context-lora-phase-snapshot-v3","sequence":i,"phase":phase,"prior_snapshot_sha256":prior,"payload":payload}; rel=f"phase-snapshots/{i:04d}-{phase}.json"; path=tmp_path/rel; path.parent.mkdir(exist_ok=True); path.write_bytes(v.canonical(snap)+b"\n"); prior=v.sha(path); report["durable_phases"].append({"sequence":i,"phase":phase,"path":rel,"sha256":prior})
    write_completed_report(tmp_path,report)
    assert v.verify(tmp_path)["passed"] is True


@pytest.mark.parametrize("mutation",("event_link","terminal_status","terminal_cursor","pointer_retained","receipt_child","batch_position","ledger_symlink"))
def test_disk_store_lineage_mutations_fail(tmp_path,monkeypatch,mutation):
    monkeypatch.setattr(v,"adapter_shapes",lambda:{"tiny":[1]}); rows=observation_fixture(); sources={"fixture":"f"*64}
    records=[{"update":n,"microbatches":4,"microbatch_losses":[float(n)]*4,"mean_loss":float(n),
              "gradient_norm_before_clip":1.,"seconds":1.,"checkpoint_seconds":.1,"checkpoint":{},
              "cursor":{"completed_updates":n,"completed_microbatches":n*4,"next_update":n+1,"next_microbatch":1}}
             for n in range(1,5)]
    root=tmp_path/"reference"; disk_store(root,4,0,records,sources,rows)
    if mutation=="event_link": path=root/"ledger/00000003.json"; item,_=v.read_envelope(path); item["prior_event_digest"]="0"*64; envelope(path,item)
    elif mutation=="terminal_status": path=root/"ledger/00000010.json"; item,_=v.read_envelope(path); item["status"]="failed"; envelope(path,item)
    elif mutation=="terminal_cursor": path=root/"ledger/00000010.json"; item,_=v.read_envelope(path); item["completed_updates"]=3; envelope(path,item)
    elif mutation=="pointer_retained": path=root/"latest.json"; item,_=v.read_envelope(path); item["retained"].pop(); envelope(path,item)
    elif mutation=="receipt_child": path=next((root/"prune-receipts").iterdir()); item,_=v.read_envelope(path); item["child_manifest_sha256"]="0"*64; envelope(path,item)
    elif mutation=="ledger_symlink":
        ledger=root/"ledger"; moved=root/"outside-ledger"; ledger.rename(moved); ledger.symlink_to(moved, target_is_directory=True)
    else:
        records[0]["checkpoint"]["manifest"]["batch_records"][0]["position"]=2
        store=v.verify_store(root,updates=4,resumes=0,baseline_swap=1000,source_hashes=sources)
        with pytest.raises(v.VerificationError): v.bind_update_manifests(records,store)
        return
    with pytest.raises(v.VerificationError):
        v.verify_store(root,updates=4,resumes=0,baseline_swap=1000,source_hashes=sources)


@pytest.mark.parametrize("mutation",("segment","resume_count","commit_eligibility","manifest_head"))
def test_branch_segment_and_checkpoint_link_mutations_fail(tmp_path,monkeypatch,mutation):
    monkeypatch.setattr(v,"adapter_shapes",lambda:{"tiny":[1]}); rows=observation_fixture(); sources={"fixture":"f"*64}
    records=[{"update":n,"microbatches":4,"microbatch_losses":[float(n)]*4,"mean_loss":float(n),"gradient_norm_before_clip":1.,"seconds":1.,"checkpoint_seconds":.1,"checkpoint":{},"cursor":{"completed_updates":n,"completed_microbatches":n*4,"next_update":n+1,"next_microbatch":1}} for n in range(1,5)]
    root=tmp_path/"restart-after-1"; disk_store(root,4,1,records,sources,rows)
    if mutation in {"segment","commit_eligibility"}: path=root/"ledger/00000003.json"; item,_=v.read_envelope(path); item["segment_id" if mutation=="segment" else "manifest_sha256"]="0"*64; envelope(path,item)
    elif mutation=="resume_count": path=root/"ledger/00000007.json"; item,_=v.read_envelope(path); item["resume_count"]=2; envelope(path,item)
    else:
        path=root/"checkpoints/segment-0001/update-0002/manifest.json"; item,_=v.read_envelope(path); item["event_chain_head"]="0"*64; envelope(path,item)
    with pytest.raises(v.VerificationError): v.verify_store(root,updates=4,resumes=1,baseline_swap=1000,source_hashes=sources)


@pytest.mark.parametrize("mutation",("loss","timing","payload_size","checkpoint_resource"))
def test_report_checkpoint_binding_mutations_fail(tmp_path,monkeypatch,mutation):
    monkeypatch.setattr(v,"adapter_shapes",lambda:{"tiny":[1]}); rows=observation_fixture(); sources={"fixture":"f"*64}
    records=[{"update":n,"microbatches":4,"microbatch_losses":[float(n)]*4,"mean_loss":float(n),"gradient_norm_before_clip":1.,"seconds":1.,"checkpoint_seconds":.1,"checkpoint":{},"cursor":{"completed_updates":n,"completed_microbatches":n*4,"next_update":n+1,"next_microbatch":1}} for n in range(1,5)]
    root=tmp_path/"reference"; disk_store(root,4,0,records,sources,rows); store=v.verify_store(root,updates=4,resumes=0,baseline_swap=1000,source_hashes=sources)
    if mutation=="loss": records[0]["microbatch_losses"][0]+=1
    elif mutation=="timing": records[0]["checkpoint"]["timings"]["total"]=-1
    elif mutation=="payload_size": records[0]["checkpoint"]["payload_bytes"]+=1
    else: records[0]["checkpoint"]["resource_observations"][0]["metrics"]["rss_peak_bytes"]+=1
    with pytest.raises(v.VerificationError):
        v.bind_update_manifests(records,store); v.bind_checkpoint_resources(records,rows,"reference_")


@pytest.mark.parametrize("mutation",("adapter_name","optimizer_shape","bad_sha","file_size","extra_file"))
def test_historical_manifest_metadata_mutations_fail(tmp_path,mutation):
    adapter={k:{"dtype":"float32","shape":shape,"size":math.prod(shape),"sha256":"a"*64} for k,shape in v.adapter_shapes().items()}
    optimizer={f"{k}.{m}":{"dtype":"float32","shape":shape,"size":math.prod(shape),"sha256":"b"*64} for k,shape in v.adapter_shapes().items() for m in ("m","v")}
    optimizer.update(step={"dtype":"uint64","shape":[],"size":1,"sha256":"c"*64},learning_rate={"dtype":"float32","shape":[],"size":1,"sha256":"d"*64})
    manifest={"adapter_inventory":adapter,"optimizer_inventory":optimizer,
              "files":{n:{"size":1,"sha256":"e"*64} for n in ("adapter.safetensors","optimizer.safetensors","adapter_config.json")}}
    if mutation=="adapter_name": manifest["adapter_inventory"]["extra"]={"shape":[],"sha256":"a"*64}
    elif mutation=="optimizer_shape": manifest["optimizer_inventory"][next(iter(optimizer))]["shape"]=[]
    elif mutation=="bad_sha": manifest["adapter_inventory"][next(iter(adapter))]["sha256"]="bad"
    elif mutation=="file_size": manifest["files"]["adapter.safetensors"]["size"]=0
    else: manifest["files"]["extra"]={"size":1,"sha256":"e"*64}
    with pytest.raises(v.VerificationError): v.verify_manifest_metadata(manifest)


def test_metadata_schema_matches_existing_frozen_helper_manifest():
    path=next((v.ROOT/".omx/artifacts/long-context-lora-resource-checkpoint-v2/main/checkpoints").glob("segment-*/update-*/manifest.json"))
    manifest,_=v.read_envelope(path)
    v.verify_manifest_metadata(manifest)


@pytest.mark.parametrize("mutation",("missing","hash","reserve","through"))
def test_completion_receipt_mutations_fail(tmp_path,mutation):
    report={}; write_completed_report(tmp_path,report)
    path=tmp_path/"completion-receipt.json"
    if mutation=="missing": path.unlink()
    else:
        receipt=v.read_json(path)
        if mutation=="hash": receipt["report_sha256"]="0"*64
        elif mutation=="reserve": receipt["final_report_close"]["reserve_seconds"]=2.
        else: receipt["report_elapsed_through_fsync_seconds"]=900.
        path.write_bytes(v.canonical(receipt)+b"\n")
    with pytest.raises(v.VerificationError):
        v.verify_completion_receipt(tmp_path,report,tmp_path/"report.json")
