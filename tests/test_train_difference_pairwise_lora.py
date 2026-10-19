import hashlib
import json
from pathlib import Path

import pytest

from scripts import train_difference_pairwise_lora as trainer


def load_real_schedule():
    data = trainer.validate_and_load_data()
    return data


def test_frozen_schedule_has_exact_one_epoch_unit_component_and_tail_accounting():
    data = load_real_schedule()
    assert len(data["units"]) == len(data["schedule"]) == 59
    assert sum(len(unit["references"]) for unit in data["units"]) == 232
    assert [row["update"] for row in data["schedule"]].count(15) == 3
    assert [row["position_in_update"] for row in data["schedule"][-3:]] == [1, 2, 3]


def test_frozen_evaluation_scopes_account_for_all_2037_pair_inferences():
    data = load_real_schedule()
    mechanism, train = trainer._source_rows(data)
    mechanism_pairs = sum(len(trainer._pairs_for_row(
        data, item["role"], item["row"]["id"], "mechanism")) for item in mechanism)
    train_pairs = sum(len(trainer._pairs_for_row(
        data, item["role"], item["row"]["id"], "train640")) for item in train)
    assert (len(mechanism), mechanism_pairs) == (175, 481)
    assert (len(train), train_pairs) == (640, 1556)


def test_schedule_validation_rejects_reorder_duplicate_and_bad_balancing():
    data = load_real_schedule()
    schedule = [dict(row) for row in data["schedule"]]
    schedule[0], schedule[1] = schedule[1], schedule[0]
    with pytest.raises(ValueError, match="unit order"):
        trainer.validate_schedule(data["units"], schedule, data["pair_by_id"], data["token_by_id"])
    units = [dict(row) for row in data["units"]]
    units[0] = {**units[0], "references": [dict(ref) for ref in units[0]["references"]]}
    units[0]["references"][0]["weight"] = 0.4
    with pytest.raises(ValueError, match="weights"):
        trainer.validate_schedule(units, data["schedule"], data["pair_by_id"], data["token_by_id"])


class MemoryProgress:
    def __init__(self):
        self.rows = []

    def append(self, value):
        self.rows.append(value)


def test_injected_schedule_executor_materializes_all_components_and_updates_once():
    data = load_real_schedule()
    seen, update_counts, checkpoints = [], [], []
    progress = MemoryProgress()

    def materialize(components, observer):
        records = []
        for index, component in enumerate(components):
            seen.append((component["pair_id"], component["target"]))
            record = {"component_index": index, "target": component["target"],
                      "weight": trainer.pairwise.unit_weights(
                          item["target"] for item in components)[index],
                      "loss": 0.25, "gradient_seconds": 0.01}
            records.append(record); observer(record)
        return {"loss": 0.25, "gradient": len(components), "components": records,
                "unit_gradient_seconds": 0.02}

    clock_value = [0.0]
    def clock():
        clock_value[0] += 0.01
        return clock_value[0]

    units, updates = trainer.execute_training_schedule(
        data, materialize=materialize,
        accumulate=lambda left, right: (left or 0) + right,
        apply_update=lambda gradient, count: update_counts.append(count) or float(gradient),
        checkpoint=checkpoints.append, progress=progress, monotonic=clock)
    assert len(units) == 59 and len(seen) == 232 and len(updates) == 15
    assert update_counts == [4] * 14 + [3]
    assert [item["actual_unit_count"] for item in updates] == update_counts
    assert len({item["order"] for item in units}) == 59
    assert sum(row["event"] == "unit_complete" for row in progress.rows) == 59
    assert sum(row["event"] == "update_complete" for row in progress.rows) == 15


def test_complete_update_limit_includes_materialization_observers_and_optimizer():
    data = load_real_schedule()
    clock_value = [0.0]

    def clock():
        clock_value[0] += 30.1
        return clock_value[0]

    def materialize(components, observer):
        records = []
        for index, component in enumerate(components):
            record = {"component_index": index, "target": component["target"],
                      "weight": trainer.pairwise.unit_weights(x["target"] for x in components)[index],
                      "loss": 1.0, "gradient_seconds": 1.0}
            observer(record); records.append(record)
        return {"loss": 1.0, "gradient": 1, "components": records,
                "unit_gradient_seconds": 1.0}

    with pytest.raises(TimeoutError, match="120 seconds"):
        trainer.execute_training_schedule(
            data, materialize=materialize,
            accumulate=lambda left, right: (left or 0) + right,
            apply_update=lambda gradient, count: 1.0,
            checkpoint=lambda stage: clock(), progress=MemoryProgress(), monotonic=clock)


def test_schedule_executor_rejects_nonfinite_component_or_update_evidence():
    data = load_real_schedule()
    progress = MemoryProgress()

    def bad_materialize(components, observer):
        weights = trainer.pairwise.unit_weights(item["target"] for item in components)
        records = [{"component_index": index, "target": item["target"], "weight": weight,
                    "loss": float("nan") if index == 0 else 1.0, "gradient_seconds": 0.0}
                   for index, (item, weight) in enumerate(zip(components, weights))]
        return {"loss": 1.0, "gradient": 1, "components": records,
                "unit_gradient_seconds": 0.0}

    with pytest.raises(FloatingPointError, match="component evidence"):
        trainer.execute_training_schedule(
            data, materialize=bad_materialize, accumulate=lambda left, right: right,
            apply_update=lambda gradient, count: 1.0, checkpoint=lambda stage: None,
            progress=progress, monotonic=lambda: 0.0)


def test_schedule_executor_retains_completed_partial_records_on_failure():
    data = load_real_schedule()
    units_out, updates_out, calls = [], [], [0]

    def materialize(components, observer):
        calls[0] += 1
        if calls[0] == 6:
            raise RuntimeError("injected failure")
        weights = trainer.pairwise.unit_weights(item["target"] for item in components)
        records = [{"component_index": index, "target": item["target"], "weight": weight,
                    "loss": 1.0, "gradient_seconds": 0.0}
                   for index, (item, weight) in enumerate(zip(components, weights))]
        return {"loss": 1.0, "gradient": 1, "components": records,
                "unit_gradient_seconds": 0.0}

    with pytest.raises(RuntimeError, match="injected failure"):
        trainer.execute_training_schedule(
            data, materialize=materialize,
            accumulate=lambda left, right: (left or 0) + right,
            apply_update=lambda gradient, count: 1.0,
            checkpoint=lambda stage: None, progress=MemoryProgress(),
            monotonic=lambda: 0.0, units_out=units_out, updates_out=updates_out)
    assert len(units_out) == 5
    assert len(updates_out) == 1
    assert updates_out[0]["actual_unit_count"] == 4


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(trainer.canonical(value) + b"\n")


def test_prelaunch_requires_exact_budget_and_execution_source_closure(tmp_path, monkeypatch):
    root = tmp_path
    monkeypatch.setattr(trainer, "ROOT", root)
    monkeypatch.setattr(trainer, "SOLE_OUTPUT", "out")
    monkeypatch.setattr(trainer, "DEFAULT_PRELAUNCH", root / "prelaunch.json")
    source_hashes = {}
    for index in range(48):
        name = f"source/{index}.txt"
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(str(index))
        source_hashes[name] = trainer.file_sha(path)
    budget_payload = {
        "stage": "budget_qualification", "passed": True, "launch_allowed": True,
        "logical_fit_id": "difference-pairwise-lora-fit-budget-v1",
        "sole_output": ".omx/artifacts/difference-pairwise-lora-train-v1",
        "old_policy": {"launch_allowed": False}, "new_policy": {"limit_seconds": 21600},
        "source_inventory": {"count": 48,
                             "sha256": hashlib.sha256(trainer.canonical(source_hashes)).hexdigest(),
                             "hashes": source_hashes},
    }
    budget = {"schema": "difference-pairwise-lora-budget-proof-v1",
              "payload": budget_payload,
              "digest": hashlib.sha256(trainer.canonical(budget_payload)).hexdigest()}
    budget_path = root / "budget.json"; write_json(budget_path, budget)
    monkeypatch.setattr(trainer, "QUALIFIED_BUDGET", budget_path)
    monkeypatch.setattr(trainer, "QUALIFIED_BUDGET_SHA256", trainer.file_sha(budget_path))
    recovery_path = root / "recovery.json"; write_json(recovery_path, {"synthetic": True})
    monkeypatch.setattr(trainer.recovery, "RECOVERY_PROOF_PATH", "recovery.json")
    for name in ("preflight.json", "checks.json", "review.md"):
        path = root / name; path.write_text(name)
    import os
    import sys
    invocation = Path(os.path.abspath(sys.executable))
    binary = Path(os.path.realpath(sys.executable))
    recovery_payload = {
        "preflight": {"path": "preflight.json", "interpreter": {
            "invocation_path": str(invocation), "resolved_path": str(binary),
            "resolved_sha256": trainer.file_sha(binary),
            "package_versions": trainer.VERSIONS}},
        "checks": {"path": "checks.json"}, "execution_review": {"path": "review.md"},
        "execution_sources": {},
    }
    monkeypatch.setattr(trainer.recovery, "validate_recovery_proof", lambda path: recovery_payload)
    inventory = dict(source_hashes)
    for name in trainer.EXECUTION_ADDITIONS:
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(name)
        inventory[name] = trainer.file_sha(path)
    inventory["budget.json"] = trainer.file_sha(budget_path)
    for name in ("recovery.json", "preflight.json", "checks.json", "review.md"):
        inventory[name] = trainer.file_sha(root / name)
    pre_payload = {"schema": trainer.recovery.PRELAUNCH_SCHEMA,
                   "passed": True, "launch_allowed": True,
                   "logical_fit_id": trainer.LOGICAL_FIT_ID, "sole_output": "out",
                   "package_versions": trainer.VERSIONS,
                   "budget_proof": {"path": "budget.json", "sha256": trainer.file_sha(budget_path)},
                   "recovery_proof": {"path": "recovery.json", "sha256": trainer.file_sha(recovery_path)},
                   "worker_python": recovery_payload["preflight"]["interpreter"],
                   "inventory": inventory}
    prelaunch = root / "prelaunch.json"
    write_json(prelaunch, {"payload": pre_payload,
                           "digest": hashlib.sha256(trainer.canonical(pre_payload)).hexdigest()})
    launch_payload = {"schema": "difference-pairwise-lora-launch-v2",
                      "logical_fit_id": trainer.LOGICAL_FIT_ID, "sole_output": "out",
                      "worker_claim": "claim", "consumed": True, "released": False,
                          "prelaunch": {"sha256": trainer.file_sha(prelaunch)},
                          "budget_proof_sha256": trainer.file_sha(budget_path),
                          "recovery_proof_sha256": trainer.file_sha(recovery_path)}
    launch = root / "out/launch.json"; write_json(launch, launch_payload)
    payload, claimed = trainer.validate_prelaunch(prelaunch, launch, root / "out", "claim")
    assert payload == pre_payload and claimed == launch_payload
    changed_launch = dict(launch_payload); changed_launch["budget_proof_sha256"] = "0" * 64
    write_json(launch, changed_launch)
    with pytest.raises(ValueError, match="budget proof differs"):
        trainer.validate_prelaunch(prelaunch, launch, root / "out", "claim")
    changed_launch = dict(launch_payload); changed_launch["worker_claim"] = "other"
    write_json(launch, changed_launch)
    with pytest.raises(ValueError, match="launch claim differs"):
        trainer.validate_prelaunch(prelaunch, launch, root / "out", "claim")
    write_json(launch, launch_payload)
    (root / "scripts/train_difference_pairwise_lora.py").write_text("mutated")
    with pytest.raises(ValueError, match="inventory differs"):
        trainer.validate_prelaunch(prelaunch, launch, root / "out", "claim")


def test_progress_log_is_append_only_jsonl(tmp_path):
    path = tmp_path / "progress.jsonl"
    log = trainer.ProgressLog(path)
    log.append({"event": "one"}); log.append({"event": "two"})
    assert [json.loads(line)["event"] for line in path.read_bytes().splitlines()] == ["one", "two"]


def test_worker_start_claim_is_exclusive_and_non_releasable(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer, "SOLE_OUTPUT", "out")
    first = trainer.claim_worker_start(tmp_path, "launch-hash", "claim", pid=42)
    assert first["consumed"] is True and first["released"] is False
    with pytest.raises(FileExistsError):
        trainer.claim_worker_start(tmp_path, "launch-hash", "claim", pid=43)
    assert json.loads((tmp_path / "worker-start.json").read_bytes())["worker_pid"] == 42


@pytest.mark.parametrize('failure_kind', ['source', 'data', 'worker_start'])
def test_post_launch_pre_model_failures_have_durable_terminal_evidence(tmp_path, monkeypatch, failure_kind):
    monkeypatch.setattr(trainer, 'ROOT', tmp_path)
    monkeypatch.setattr(trainer, 'SOLE_OUTPUT', 'out')
    output = tmp_path / 'out'; output.mkdir()
    launch = output / 'launch.json'
    write_json(launch, {'schema': 'difference-pairwise-lora-launch-v2',
                        'logical_fit_id': trainer.LOGICAL_FIT_ID, 'sole_output': 'out',
                        'worker_claim': 'test', 'consumed': True, 'released': False,
                        'recovery_proof_sha256': 'a' * 64})
    def bad_source(*args): raise ValueError('source inventory differs')
    def bad_data(): raise ValueError('qualified data differs')
    def bad_claim(*args): raise OSError('worker-start write failed')
    monkeypatch.setattr(trainer, 'validate_prelaunch', bad_source if failure_kind == 'source' else lambda *args: ({}, {}))
    monkeypatch.setattr(trainer, 'validate_and_load_data', bad_data)
    if failure_kind == 'worker_start': monkeypatch.setattr(trainer, 'claim_worker_start', bad_claim)
    report, status = trainer.run(output, tmp_path / 'prelaunch', launch, 0., 'test', monotonic=lambda: 1.)
    assert status == 1 and report['completed'] is False
    assert report['fresh_base_initialized'] is False
    assert report['real_training_run'] is False
    assert report['eligibility'] == {'evaluation': False, 'resume': False, 'deploy': False}
    assert json.loads((output / 'report.json').read_bytes()) == report
    assert json.loads((output / 'progress.jsonl').read_bytes())['event'] == 'terminal'
    import sys
    assert 'mlx.core' not in sys.modules


def test_duplicate_worker_does_not_replace_existing_owner_report(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer, 'ROOT', tmp_path)
    monkeypatch.setattr(trainer, 'SOLE_OUTPUT', 'out')
    output = tmp_path / 'out'; output.mkdir()
    launch = output / 'launch.json'
    write_json(launch, {'schema': 'difference-pairwise-lora-launch-v2',
                        'logical_fit_id': trainer.LOGICAL_FIT_ID, 'sole_output': 'out',
                        'worker_claim': 'test', 'consumed': True, 'released': False,
                        'recovery_proof_sha256': 'a' * 64})
    trainer.claim_worker_start(output, trainer.file_sha(launch), 'test')
    (output / 'report.json').write_text('existing owner report')
    with pytest.raises(FileExistsError):
        trainer.run(output, tmp_path / 'prelaunch', launch, 0., 'test', monotonic=lambda: 1.)
    assert (output / 'report.json').read_text() == 'existing owner report'
