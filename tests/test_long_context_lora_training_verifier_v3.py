"""CPU-only structural and mutation tests for the independent TRAIN verifier."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct

import pytest


PATH = (Path(__file__).resolve().parents[1] /
        ".omx/artifacts/long-context-lora-work-v4/verify_training.py")
SPEC = importlib.util.spec_from_file_location("training_verifier_v3", PATH)
v = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(v)


def run_fixture():
    return {
        "schema": "long-context-lora-run-v2", "logical_run_id": "fit-fixture",
        "created_at": "2026-09-15T00:00:00+00:00", "mode": "fit",
        "baseline": {"swap_used_bytes": 1000}, "limits": copy.deepcopy(v.LIMITS),
        "max_resumes": 2,
        "schedule": {"sha256": "a" * 64, "microbatches": 2771, "updates": 693,
                     "accumulation": 4, "final_update_microbatches": 3},
        "recipe": copy.deepcopy(v.RECIPE), "source_hashes": {"source": "b" * 64},
        "config_hashes": {"plan": v.PLAN_SHA256, "test_spec": v.TEST_SPEC_SHA256,
                          "handoff": v.HANDOFF_SHA256,
                          "data_manifest": v.DATA_MANIFEST_SHA256,
                          "resource_qualification": "c" * 64},
        "model_hashes": {
            "SOURCE.json": "67928c796b7e4562c072b2bbbdd0d52569e5ab9118ade1302f8d8d1bb4bae819",
            "model.safetensors": "e56a94b846a2dce7d1cc154ce58b9cfc4b8d99e8ffb74bfe256a265f4cbc7fb9"},
        "package_versions": {"mlx": "0.31.2", "mlx-lm": "0.31.3",
                             "transformers": "5.12.1"},
    }


def test_exact_fit_run_identity_passes():
    v.verify_run(run_fixture(), "a" * 64, "c" * 64)


@pytest.mark.parametrize("mutation", ("budget", "rows", "resume", "recipe", "model", "proof"))
def test_fit_run_identity_mutations_fail(mutation):
    run = run_fixture()
    if mutation == "budget": run["limits"]["whole_seconds"] = 21600.0
    elif mutation == "rows": run["schedule"]["microbatches"] = 2770
    elif mutation == "resume": run["max_resumes"] = 3
    elif mutation == "recipe": run["recipe"]["seed"] += 1
    elif mutation == "model": run["model_hashes"]["model.safetensors"] = "0" * 64
    else: run["config_hashes"]["resource_qualification"] = "d" * 64
    with pytest.raises(v.VerificationError):
        v.verify_run(run, "a" * 64, "c" * 64)


def metric(**changes):
    return {"elapsed_seconds": 1.0, "free_percent": 50.0, "mlx_active_bytes": 1,
            "mlx_cache_bytes": 1, "mlx_peak_bytes": 1, "rss_peak_bytes": 1,
            "swap_growth_bytes": 0, "swap_used_bytes": 1000, **changes}


def observation(stage="model_loaded"):
    return {"stage": stage, "action": "none", "pre": metric(), "post": None,
            "errors": [], "cleanup_seconds": 0.0, "checkpoint_seconds": .01}


def test_resource_observation_passes():
    assert v.verify_resource_observations([observation()], 1000) == 1.0


@pytest.mark.parametrize("mutation", ("rss", "mlx", "cache", "swap", "free", "error", "backward"))
def test_resource_limit_and_trajectory_mutations_fail(mutation):
    rows = [observation(), observation("next")]
    if mutation == "rss": rows[0]["pre"]["rss_peak_bytes"] = 7 * 1024**3
    elif mutation == "mlx": rows[0]["pre"]["mlx_peak_bytes"] = 7 * 1024**3
    elif mutation == "cache": rows[0]["pre"]["mlx_cache_bytes"] = 257 * 1024**2
    elif mutation == "swap": rows[0]["pre"]["swap_used_bytes"] += 512 * 1024**2 + 1
    elif mutation == "free": rows[0]["pre"]["free_percent"] = 9
    elif mutation == "error": rows[0]["errors"] = ["failure"]
    else: rows[1]["pre"]["elapsed_seconds"] = 0.5
    with pytest.raises(v.VerificationError):
        v.verify_resource_observations(rows, 1000)


def update_fixture(update=1):
    batch = [{"role": "fit", "id": f"row-{n}", "microbatch": n, "update": update,
              "tokens_sha256": "a" * 64, "candidate_count": 2,
              "digit_ids": [1, 2], "allowed_gold_positions": [0]} for n in range(1, 5)]
    losses = [1.0, 2.0, 3.0, 4.0]
    manifest = {"batch_records": copy.deepcopy(batch), "losses": copy.deepcopy(losses),
                "files": {"adapter.safetensors": {"size": 3, "sha256": "a" * 64}}}
    record = {"update": update, "microbatches": 4, "batch_records": batch,
              "microbatch_losses": losses, "gradient_norm_before_clip": 1.0,
              "seconds": 2.0, "helper_checkpoint_inclusive_seconds": 1.5,
              "checkpoint_timings": {key: .1 for key in
                  ("post_materialize", "post_reload", "precommit", "postcommit",
                   "final_eligibility", "total")},
              "checkpoint_payload_bytes": sum(item["size"] for item in manifest["files"].values())
                  + len(v.canonical({"payload": manifest, "digest": v.digest(manifest)})),
              "checkpoint_manifest_sha256": "b" * 64}
    return record, manifest, copy.deepcopy(batch)


def test_full_settled_update_record_passes():
    record, manifest, expected = update_fixture()
    v.verify_update_record(record, manifest, expected, 1)


@pytest.mark.parametrize("mutation", ("schedule", "loss", "full", "helper", "tail", "timing"))
def test_update_schedule_loss_and_timing_mutations_fail(mutation):
    record, manifest, expected = update_fixture()
    if mutation == "schedule": record["batch_records"][0]["id"] = "changed"
    elif mutation == "loss": record["microbatch_losses"][0] += 1
    elif mutation == "full": record["seconds"] = 121.0
    elif mutation == "helper": record["helper_checkpoint_inclusive_seconds"] = 3.0
    elif mutation == "tail": record["microbatches"] = 3
    else: record["checkpoint_timings"]["extra"] = 0
    with pytest.raises(v.VerificationError):
        v.verify_update_record(record, manifest, expected, 1)


def write_segment(root, *, report_changes=None, receipt_changes=None):
    segment = "segment-0001"
    event = {"sequence": 4, "prior_event_digest": "a" * 64,
             "logical_run_id": "fit", "segment_id": segment, "kind": "segment_terminal",
             "status": "complete", "terminal_segment_seconds": 11.0,
             "close_reserve_seconds": 1.0,
             "pointer_status": "eligible", "committed_update": 693,
             "committed_manifest_sha256": "b" * 64, "eligible_update": 693,
             "eligible_manifest_sha256": "b" * 64}
    event["digest"] = v.digest({key: value for key, value in event.items() if key != "digest"})
    report = {"schema": "long-context-lora-train-v3-segment", "segment_id": segment,
              "completed": False, "completion_pending": True,
              "ledger_terminal_status": "complete", "terminal_outcome": "success",
              "close_reservation": {"reserve_seconds": 1.0, "prior_charged_seconds": 0.0,
                  "current_active_observed_seconds": 10.0, "current_charged_seconds": 11.0,
                  "cumulative_charged_seconds": 11.0, "terminal_event_digest": event["digest"],
                  "reservation_scope": "terminal event, locked report, unlock, and completion receipt"}}
    report.update(report_changes or {})
    report_path = root / "segments" / segment / "report.json"
    report_path.parent.mkdir(parents=True); report_path.write_text(json.dumps(report))
    receipt = {"schema": "long-context-lora-train-v3-completion-receipt",
               "segment_id": segment, "report_sha256": v.sha(report_path),
               "terminal_event_digest": event["digest"], "ledger_terminal_status": "complete",
               "terminal_outcome": "success", "finalized": True, "training_succeeded": True,
               "reserve_seconds": 1.0, "observed_close_through_unlock_seconds": .2,
               "current_charged_seconds": 11.0, "prior_charged_seconds": 0.0,
               "cumulative_charged_seconds": 11.0}
    receipt.update(receipt_changes or {})
    (report_path.parent / "completion-receipt.json").write_text(json.dumps(receipt))
    return report, event


def test_pending_report_plus_receipt_is_completion_proof(tmp_path):
    report, event = write_segment(tmp_path)
    receipt = v.verify_segment_completion(tmp_path, report, event)
    assert receipt["training_succeeded"] is True


def helper_event_grammar():
    events = []
    prior = None
    def append(kind, **fields):
        payload = {"sequence": len(events) + 1, "prior_event_digest": prior,
                   "logical_run_id": "fit", "segment_id": "segment-0001", "kind": kind,
                   **fields}
        payload["digest"] = v.digest(payload)
        events.append(payload)
        return payload["digest"]
    prior = append("segment_start", resume_count=0)
    prior = append("base_diagnostic_bound")
    for update in range(1, 694):
        checkpoint = f"checkpoints/segment-0001/update-{update:04d}"
        prior = append("checkpoint_committed", update=update, manifest_sha256="b" * 64,
                       checkpoint_path=checkpoint)
        prior = append("resume_eligible", update=update, manifest_sha256="b" * 64,
                       checkpoint_path=checkpoint)
    append("segment_terminal", status="complete", terminal_segment_seconds=101.0,
           close_reserve_seconds=1.0, pointer_status="eligible", committed_update=693,
           committed_manifest_sha256="b" * 64, eligible_update=693,
           eligible_manifest_sha256="b" * 64)
    return events


def test_actual_helper_terminal_event_grammar_passes():
    v.verify_event_grammar(helper_event_grammar())


@pytest.mark.parametrize("mutation", ("old_reserve_name", "invented_completed_updates"))
def test_terminal_event_schema_mutations_fail(mutation):
    events = helper_event_grammar()
    terminal = events[-1]
    if mutation == "old_reserve_name":
        terminal["terminal_close_reserve_seconds"] = terminal.pop("close_reserve_seconds")
    else:
        terminal["completed_updates"] = 693
    with pytest.raises(v.VerificationError):
        v.verify_event_grammar(events)


@pytest.mark.parametrize("outcome,resume_permitted", [
    ("resource_failure", True), ("quality_rejected", False), ("success", False),
    ("failed", False), ("finalization_required", False),
])
def test_impossible_prior_segment_resume_is_rejected(outcome, resume_permitted):
    with pytest.raises(v.VerificationError):
        v.verify_segment_resume_semantics(
            {"terminal_outcome": outcome, "resume_permitted": resume_permitted}, is_final=False)


@pytest.mark.parametrize("outcome", ("failed", "finalization_required"))
def test_diagnosed_prior_interruption_can_be_resumed(outcome):
    v.verify_segment_resume_semantics(
        {"terminal_outcome": outcome, "resume_permitted": True}, is_final=False)


def test_final_success_is_non_resumable():
    v.verify_segment_resume_semantics(
        {"terminal_outcome": "success", "resume_permitted": False}, is_final=True)


@pytest.mark.parametrize("target,key,value", [
    ("report", "completed", True), ("report", "completion_pending", False),
    ("report", "terminal_outcome", "quality_rejected"),
    ("receipt", "report_sha256", "0" * 64),
    ("receipt", "training_succeeded", False),
    ("receipt", "observed_close_through_unlock_seconds", 1.1),
])
def test_report_receipt_and_quality_mutations_fail(tmp_path, target, key, value):
    report, event = write_segment(tmp_path,
                                  report_changes={key: value} if target == "report" else None,
                                  receipt_changes={key: value} if target == "receipt" else None)
    with pytest.raises(v.VerificationError):
        v.verify_segment_completion(tmp_path, report, event)


@pytest.mark.parametrize("kind,count", (("adapter", 16), ("optimizer", 34)))
def test_actual_historical_helper_payload_compatibility(kind, count):
    raw = v._load_raw_verifier()
    path = (v.ROOT / ".omx/artifacts/long-context-lora-resource-checkpoint-v3/reference/"
            "checkpoints/segment-0001/update-0004" / f"{kind}.safetensors")
    assert len(raw.inspect_tensors(path, kind=kind, update=4)) == count


def test_final_source_pins_are_fail_closed_until_review():
    assert v.sha(v.TRAINER) == v.TRAINER_SHA256
    assert v.sha(v.DIAGNOSTIC_VERIFIER) == v.DIAGNOSTIC_VERIFIER_SHA256


def test_actual_reviewed_fit_data_and_enriched_tail_are_compatible():
    manifest = v.read_json(v.DATA / "manifest.json")
    schedule, records = v.load_training_records(v.DATA, manifest)
    assert len(schedule) == len(records) == 2771
    tail = v.expected_batch(schedule, 693, records,
                            manifest["digit_continuation_ids_by_candidate_count"])
    assert len(tail) == 3
    assert set(tail[-1]) == {"role", "id", "microbatch", "update", "tokens_sha256",
                             "candidate_count", "digit_ids", "allowed_gold_positions"}


def test_fixed_reload_probe_identity_and_logits_pass():
    records = [{"role": "fit", "id": f"row-{index}", "tokens": [index],
                "choices": [[0], [1]]} for index in range(2771)]
    checks = []
    for index in (0, 923, 1846, 2770):
        logits = [0.25, 0.75]
        checks.append({"index": index, "role": "fit", "id": f"row-{index}",
                       "tokens_sha256": __import__("hashlib").sha256(
                           json.dumps([index], separators=(",", ":")).encode()).hexdigest(),
                       "digit_logits": logits, "actual_logits": logits,
                       "max_absolute_delta": 0.0})
    v.verify_reload_checks(checks, records)
    changed = copy.deepcopy(checks); changed[-1]["actual_logits"] = [.75, .25]
    with pytest.raises(v.VerificationError):
        v.verify_reload_checks(changed, records)


def _write_json(path, value, *, canonical=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (v.canonical(value) if canonical else
               json.dumps(value, ensure_ascii=False, sort_keys=True).encode())
    path.write_bytes(encoded)


def _envelope(path, payload):
    _write_json(path, {"payload": payload, "digest": v.digest(payload)}, canonical=True)


def _write_payload(path, raw, kind, update):
    tensors = {}
    for name, shape in raw.adapter_shapes().items():
        names = [name] if kind == "adapter" else [name + ".m", name + ".v"]
        for key in names:
            tensors[key] = ("F32", shape, bytes(4 * math.prod(shape)))
    if kind == "optimizer":
        tensors["step"] = ("U64", [], struct.pack("<Q", update))
        tensors["learning_rate"] = ("F32", [], struct.pack("<f", 1e-5))
    header, body = {"__metadata__": None}, bytearray()
    for key, (dtype, shape, data) in tensors.items():
        begin = len(body); body.extend(data)
        header[key] = {"dtype": dtype, "shape": shape,
                       "data_offsets": [begin, len(body)]}
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + body)


def _inventory(raw, path, kind, update):
    parsed = raw.inspect_tensors(path, kind=kind, update=update)
    return {name: {"dtype": "uint64" if name == "step" else "float32",
                   "shape": item["shape"], "size": math.prod(item["shape"]),
                   "sha256": item["checkpoint_sha256"]}
            for name, item in parsed.items()}


@pytest.fixture(scope="module")
def full_training_artifact(tmp_path_factory):
    """Build all 693 real-schema manifests/events; retain only the final two payloads."""
    root = tmp_path_factory.mktemp("train-v3-full")
    qualification_path = root / "qualification.json"
    q_path = v.QUALIFIER
    q_spec = importlib.util.spec_from_file_location("fixture_qualifier", q_path)
    q = importlib.util.module_from_spec(q_spec); q_spec.loader.exec_module(q)
    _write_json(qualification_path, q.qualify_actual())
    qualification_sha = v.sha(qualification_path)

    data_manifest = v.read_json(v.DATA / "manifest.json")
    schedule, records = v.load_training_records(v.DATA, data_manifest)
    schedule_sha = data_manifest["file_hashes"]["schedule.jsonl"]
    source_hashes = v.read_json(
        v.ROOT / ".omx/artifacts/long-context-lora-resource-checkpoint-v3/report.json"
    )["source_hashes"]
    additions = [v.TRAINER, v.HELPER, v.MODEL, v.QUALIFIER,
                 v.ROOT / ".omx/plans/long-context-lora-budget-v4.md",
                 v.ROOT / ".omx/plans/long-context-lora-budget-v4-test-spec.md",
                 v.ROOT / ".omx/plans/long-context-lora-budget-v4-handoff.json"]
    source_hashes = dict(source_hashes)
    for path in additions:
        source_hashes[str(path.relative_to(v.ROOT))] = v.sha(path)
    source_hashes[str(qualification_path)] = qualification_sha
    run = run_fixture()
    run.update(logical_run_id="fit-full-fixture", source_hashes=source_hashes)
    run["schedule"]["sha256"] = schedule_sha
    run["config_hashes"]["resource_qualification"] = qualification_sha
    _envelope(root / "run.json", run)
    run_sha = v.sha(root / "run.json")

    raw = v._load_raw_verifier()
    payload_meta = {}
    for update in (692, 693):
        directory = root / f"checkpoints/segment-0001/update-{update:04d}"
        directory.mkdir(parents=True)
        _write_payload(directory / "adapter.safetensors", raw, "adapter", update)
        _write_payload(directory / "optimizer.safetensors", raw, "optimizer", update)
        _write_json(directory / "adapter_config.json", v.ADAPTER_CONFIG, canonical=True)
        payload_meta[update] = {
            "adapter_inventory": _inventory(raw, directory / "adapter.safetensors",
                                            "adapter", update),
            "optimizer_inventory": _inventory(raw, directory / "optimizer.safetensors",
                                              "optimizer", update),
            "files": {path.name: {"size": path.stat().st_size, "sha256": v.sha(path)}
                      for path in directory.iterdir()},
        }
    base_adapter = payload_meta[693]["adapter_inventory"]
    base_optimizer = payload_meta[693]["optimizer_inventory"]
    base_files = payload_meta[693]["files"]

    events = []
    def event(kind, **fields):
        payload = {"sequence": len(events) + 1,
                   "prior_event_digest": events[-1]["digest"] if events else None,
                   "logical_run_id": run["logical_run_id"], "segment_id": "segment-0001",
                   "kind": kind, **fields}
        event_digest = v.digest(payload)
        _envelope(root / "ledger" / f"{len(events)+1:08d}.json", payload)
        item = {**payload, "digest": event_digest}; events.append(item); return item

    event("segment_start", resume_count=0)
    # Real diagnostic objects and independently replayable prediction files.
    from transformers import AutoTokenizer
    from scripts.long_context_lora_evaluation import apply_logits, prepare_diagnostic, score_predictions
    items = v.read_jsonl(v.DATA / "diagnostic.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(
        v.ROOT / "data/external/qwen3-4b-mlx-4bit", local_files_only=True,
        trust_remote_code=False)
    prepared = prepare_diagnostic(items, tokenizer); del tokenizer
    parent = [apply_logits(record, None) for record in prepared]
    def parent_logits(record):
        values = [0.0] * len(record["choices"])
        values[next(index for index, choice in enumerate(record["choices"]) if choice[0] == 0)] = 1.0
        return values
    base = [apply_logits(record, parent_logits(record) if record["eligible"] else None)
            for record in prepared]
    trained, corrections = copy.deepcopy(base), 0
    for index, (item, record) in enumerate(zip(items, prepared)):
        if item["row"]["result"] in item["row"]["expected"] or not record["eligible"]:
            continue
        for position in range(1, len(record["choices"])):
            logits = [0.0] * len(record["choices"]); logits[position] = 2.0
            candidate = apply_logits(record, logits)
            if candidate["result"] in item["row"]["expected"]:
                trained[index] = candidate; corrections += 1; break
        if corrections >= 6:
            break
    need_gate = score_predictions(items, trained)["gate"]
    assert need_gate["passed"] and corrections >= 5
    base_root = root / "base-diagnostic"; base_root.mkdir()
    def jsonl(path, rows):
        path.write_bytes(b"".join(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                             separators=(",", ":")).encode() + b"\n"
                                  for row in rows))
    jsonl(base_root / "parent_predictions.jsonl", parent)
    jsonl(base_root / "base_predictions.jsonl", base)
    identities = [{"position": index, "role": item["role"], "id": item["row"]["id"]}
                  for index, item in enumerate(items, 1)]
    base_manifest = {"schema": "long-context-lora-base-diagnostic-v3", "rows": 640,
                     "diagnostic_identity_sha256": v.digest(identities),
                     "parent_sha256": v.sha(base_root / "parent_predictions.jsonl"),
                     "base_sha256": v.sha(base_root / "base_predictions.jsonl"),
                     "initial_adapter_hashes": {name: item["sha256"]
                                                 for name, item in base_adapter.items()},
                     "v1_prediction_cache_read": False}
    _write_json(base_root / "manifest.json", base_manifest)
    binding = event("base_diagnostic_bound", rows=640,
                    manifest_sha256=v.sha(base_root / "manifest.json"),
                    diagnostic_identity_sha256=base_manifest["diagnostic_identity_sha256"],
                    parent_sha256=base_manifest["parent_sha256"],
                    base_sha256=base_manifest["base_sha256"])

    manifests, updates, previous_manifest = {}, [], None
    observations, elapsed = [], 0.0
    def observe(stage):
        nonlocal elapsed
        elapsed += .001
        metrics = metric(elapsed_seconds=elapsed)
        observations.append({"stage": stage, "action": "none", "pre": metrics, "post": None,
                             "errors": [], "cleanup_seconds": 0.0,
                             "checkpoint_seconds": .001})
    observe("model_loaded")
    for update in range(1, 694):
        expected = v.expected_batch(schedule, update, records,
                                    data_manifest["digit_continuation_ids_by_candidate_count"])
        count = len(expected)
        for number in range(1, count + 1): observe(f"update_{update}_microbatch_{number}")
        observe(f"update_{update}_averaged_clipped")
        for stage in ("post_materialize", "post_reload", "precommit", "postcommit",
                      "final_eligibility"):
            observe(f"update_{update}_checkpoint_{stage}")
        directory = root / f"checkpoints/segment-0001/update-{update:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        optimizer = copy.deepcopy(base_optimizer)
        optimizer["step"]["sha256"] = hashlib.sha256(
            b"uint64:():" + struct.pack("<Q", update)).hexdigest()
        files = copy.deepcopy(payload_meta.get(update, {}).get("files", base_files))
        losses = [float(update)] * count
        manifest = {"schema": "long-context-lora-checkpoint-v2",
            "logical_run_id": run["logical_run_id"], "segment_id": "segment-0001",
            "update": update, "completed_updates": update,
            "completed_microbatches": min(update * 4, 2771),
            "next_schedule_position": min(update * 4, 2771),
            "next_expected": None if update == 693 else
                {"role": schedule[update * 4]["role"], "id": schedule[update * 4]["id"]},
            "batch_records": expected, "losses": losses,
            "adapter_inventory": copy.deepcopy(base_adapter), "optimizer_inventory": optimizer,
            "adapter_config": copy.deepcopy(v.ADAPTER_CONFIG), "optimizer_config": copy.deepcopy(v.ADAM),
            "recipe": copy.deepcopy(v.RECIPE), "run_json_sha256": run_sha,
            "event_chain_head": events[-1]["digest"], "elapsed_seconds": float(update),
            "expected_parent_manifest_sha256": previous_manifest,
            "expected_parent_event_digest": events[-1]["digest"], "files": files}
        manifest["manifest_digest"] = v.digest(manifest)
        _envelope(directory / "manifest.json", manifest)
        manifest_sha = v.sha(directory / "manifest.json")
        commit = event("checkpoint_committed", update=update, manifest_sha256=manifest_sha,
                       checkpoint_path=f"checkpoints/segment-0001/update-{update:04d}",
                       parent_manifest_sha256=previous_manifest)
        eligible = event("resume_eligible", update=update, manifest_sha256=manifest_sha,
                         checkpoint_path=f"checkpoints/segment-0001/update-{update:04d}")
        observe(f"update_{update}_eligible"); observe(f"update_{update}_settled")
        record = {"update": update, "microbatches": count, "batch_records": expected,
                  "microbatch_losses": losses, "gradient_norm_before_clip": 1.0,
                  "seconds": 1.0, "helper_checkpoint_inclusive_seconds": .5,
                  "checkpoint_timings": {name: .01 for name in
                    ("post_materialize", "post_reload", "precommit", "postcommit",
                     "final_eligibility", "total")},
                  "checkpoint_payload_bytes": sum(item["size"] for item in files.values()) +
                    len(v.canonical({"payload": manifest, "digest": v.digest(manifest)})),
                  "checkpoint_manifest_sha256": manifest_sha}
        updates.append(record); manifests[update] = (manifest, manifest_sha)
        if update >= 3:
            victim_update = update - 2
            victim, victim_sha = manifests[victim_update]
            victim_path = f"checkpoints/segment-0001/update-{victim_update:04d}"
            receipt = {"schema": "long-context-lora-checkpoint-v2", "kind": "prune_receipt",
                "victim_path": victim_path, "victim_manifest_sha256": victim_sha,
                "parent_manifest_sha256": victim_sha, "child_manifest_sha256": manifest_sha,
                "new_checkpoint_path": f"checkpoints/segment-0001/update-{update:04d}",
                "deletion": [{"name": name, **victim["files"][name]} for name in
                    ("adapter.safetensors", "optimizer.safetensors", "adapter_config.json")]}
            name = (f"segment-0001-update-{update:04d}-prunes-segment-0001-"
                    f"{victim_update:04d}.json")
            _envelope(root / "prune-receipts" / name, receipt)
        previous_manifest = manifest_sha

    entries = [{"path": f"checkpoints/segment-0001/update-{update:04d}",
                "manifest_sha256": manifests[update][1], "update": update,
                "segment_id": "segment-0001",
                "eligible_event_digest": next(event["digest"] for event in events
                    if event["kind"] == "resume_eligible" and event["update"] == update)}
               for update in range(1, 694)]
    commits = [event for event in events if event["kind"] == "checkpoint_committed"]
    statuses = [event for event in events if event["kind"] == "resume_eligible"]
    pointer = {"schema": "long-context-lora-checkpoint-v2", **{key: entries[-1][key]
               for key in ("path", "manifest_sha256", "update", "segment_id")},
               "status": "eligible", "event_digest": statuses[-1]["digest"],
               "commit_event_digest": commits[-1]["digest"],
               "parent_manifest_sha256": entries[-2]["manifest_sha256"],
               "retained": list(reversed(entries[-2:]))}
    _envelope(root / "latest.json", pointer)

    attempt = root / "finalization-attempt-0001"; (attempt / "adapter").mkdir(parents=True)
    jsonl(attempt / "trained_predictions.jsonl", trained)
    source_adapter = root / entries[-1]["path"] / "adapter.safetensors"
    (attempt / "adapter/adapters.safetensors").write_bytes(source_adapter.read_bytes())
    (attempt / "adapter/adapter_config.json").write_bytes(
        (root / entries[-1]["path"] / "adapter_config.json").read_bytes())
    diagnostic_spec = importlib.util.spec_from_file_location("fixture_diag", v.DIAGNOSTIC_VERIFIER)
    diagnostic = importlib.util.module_from_spec(diagnostic_spec); diagnostic_spec.loader.exec_module(diagnostic)
    declared_metrics = {name: score_predictions(items, rows) for name, rows in
                        (("fresh_parent", parent), ("fresh_base", base), ("final_adapter", trained))}
    before = declared_metrics["fresh_parent"]["metrics"]["all"]["top1"]
    after = declared_metrics["final_adapter"]["metrics"]["all"]["top1"]
    trained_gate = declared_metrics["final_adapter"]["gate"]
    train_gate = {"passed": after - before >= 5, "net_gain": after - before,
                  "required_net_gain": 5, "parent_correct": before, "trained_correct": after,
                  "corrections": trained_gate["corrections"],
                  "regressions": trained_gate["regressions"]}
    reload_checks = []
    for index in (0, 923, 1846, 2770):
        logits = [0.0] * len(records[index]["choices"])
        reload_checks.append({"index": index, "role": records[index]["role"],
            "id": records[index]["id"], "tokens_sha256": hashlib.sha256(json.dumps(
                records[index]["tokens"], separators=(",", ":")).encode()).hexdigest(),
            "digit_logits": logits, "actual_logits": logits, "max_absolute_delta": 0.0})
    accounting = {"expected_logical": {"predicate_host_reads": 4157,
                                        "numeric_host_reads": 3464},
                  "actual_cumulative": {"predicate_host_reads": 4157,
                                        "numeric_host_reads": 3464},
                  "interrupted_extra": {"predicate_host_reads": 0,
                                        "numeric_host_reads": 0}}
    attempt_manifest = {"schema": "long-context-lora-finalization-attempt-v3",
        "segment_id": "segment-0001", "checkpoint_update": 693,
        "evaluation_completed": True, "quality_outcome": "passed",
        "checkpoint_manifest_sha256": manifests[693][1],
        "adapter": {"adapter_sha256": v.sha(attempt / "adapter/adapters.safetensors"),
                    "config_sha256": v.sha(attempt / "adapter/adapter_config.json")},
        "parent_predictions_sha256": base_manifest["parent_sha256"],
        "base_predictions_sha256": base_manifest["base_sha256"],
        "trained_predictions_sha256": v.sha(attempt / "trained_predictions.jsonl"),
        "reload_checks": reload_checks, "diagnostic_metrics": declared_metrics,
        "train_gate": train_gate, "source_hashes_after": source_hashes,
        "predicate_read_accounting": accounting}
    _write_json(attempt / "manifest.json", attempt_manifest)
    observe("training_model_released"); observe("evaluation_model_released")
    terminal_payload = event("segment_terminal", status="complete", terminal_segment_seconds=101.0,
        close_reserve_seconds=1.0, pointer_status="eligible",
        committed_update=693, committed_manifest_sha256=manifests[693][1], eligible_update=693,
        eligible_manifest_sha256=manifests[693][1])
    report = {"schema": "long-context-lora-train-v3-segment", "mode": "fresh",
        "segment_id": "segment-0001", "completed": False, "completion_pending": True,
        "resume_permitted": False,
        "training_gate_passed": True, "deployed": False, "auto_resume": False,
        "resource_qualification": {"path": str(qualification_path), "sha256": qualification_sha,
            "report_sha256": q.EVIDENCE_SHA256["report"], "qualifier_sha256": v.QUALIFIER_SHA256,
            "helper_sha256": v.HELPER_SHA256}, "source_hashes": source_hashes,
        "source_hashes_after": source_hashes, "versions": run["package_versions"],
        "versions_after": run["package_versions"], "limits": copy.deepcopy(v.LIMITS),
        "fresh_data_read_for_evaluation": False, "development_or_final_read_for_evaluation": False,
        "production_modified": False, "v1_prediction_cache_read": False,
        "resource_observations": observations, "updates": updates,
        "logical_run_id": run["logical_run_id"], "run_json_sha256": run_sha,
        "system_baseline": run["baseline"], "planned_updates": 693, "training_rows": 2771,
        "resume_preflight": None, "prior_terminal_seconds": 0.0,
        "source_only_schedule_verified_before_fit": True, "fit_rows_read": True,
        "base_diagnostic_binding": {"manifest": base_manifest,
            "manifest_sha256": base_manifest["manifest_sha256"] if "manifest_sha256" in base_manifest
                else v.sha(base_root / "manifest.json"), "event_digest": binding["digest"]},
        "completed_updates": 693, "completed_microbatches": 2771,
        "quality_outcome": "passed", "source_inventory_unchanged": True,
        "final_checkpoint": manifests[693][0], "final_adapter": attempt_manifest["adapter"],
        "reload_checks": reload_checks,
        "trained_predictions_sha256": attempt_manifest["trained_predictions_sha256"],
        "diagnostic_metrics": declared_metrics, "train_gate": train_gate,
        "all_diagnostic_labels_consulted_after_model_release": True,
        "predicate_reads_segment": {"predicate_host_reads": 4157, "numeric_host_reads": 3464},
        "predicate_read_accounting": accounting, "terminal_outcome": "success",
        "ledger_terminal_status": "complete", "elapsed_seconds": 100.0,
        "close_reservation": {"reserve_seconds": 1.0, "prior_charged_seconds": 0.0,
            "current_active_observed_seconds": 100.0, "current_charged_seconds": 101.0,
            "cumulative_charged_seconds": 101.0,
            "terminal_event_digest": terminal_payload["digest"],
            "reservation_scope": "terminal event, locked report, unlock, and completion receipt"}}
    report_path = root / "segments/segment-0001/report.json"; _write_json(report_path, report)
    receipt = {"schema": "long-context-lora-train-v3-completion-receipt",
        "segment_id": "segment-0001", "report_sha256": v.sha(report_path),
        "terminal_event_digest": terminal_payload["digest"], "ledger_terminal_status": "complete",
        "terminal_outcome": "success", "finalized": True, "training_succeeded": True,
        "reserve_seconds": 1.0, "observed_close_through_unlock_seconds": .1,
        "current_charged_seconds": 101.0, "prior_charged_seconds": 0.0,
        "cumulative_charged_seconds": 101.0}
    _write_json(root / "segments/segment-0001/completion-receipt.json", receipt)
    return root, qualification_path


def test_full_top_level_real_schema_fixture_passes(full_training_artifact):
    root, qualification = full_training_artifact
    result = v.verify(root, qualification)
    assert result["passed"] is result["training_succeeded"] is True
    assert result["updates_verified"] == 693 and result["rows_verified"] == 2771
