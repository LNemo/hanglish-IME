import json
from pathlib import Path

import pytest

from scripts import train_long_context_lora as train


def make_schedule(rows=2771):
    return [
        {"id": f"row-{index}", "role": "natural_fit",
         "microbatch": index, "update": (index - 1) // 4 + 1}
        for index in range(1, rows + 1)
    ]


def test_schedule_accounts_exact_epoch_updates_and_tail_three():
    batches = train.validate_schedule(make_schedule())
    assert len(batches) == 693
    assert sum(map(len, batches)) == 2771
    assert all(len(batch) == 4 for batch in batches[:-1])
    assert len(batches[-1]) == 3
    assert batches[-1][0]["microbatch"] == 2769
    assert batches[-1][-1]["update"] == 693


def test_schedule_divisible_by_accumulation_has_full_final_batch():
    batches = train.validate_schedule(make_schedule(8), expected_rows=8)
    assert [len(batch) for batch in batches] == [4, 4]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong_update", "wrong_microbatch"])
def test_schedule_rejects_lost_duplicated_or_reordered_examples(mutation):
    schedule = make_schedule()
    if mutation == "missing":
        schedule.pop()
    elif mutation == "duplicate":
        schedule[-1]["id"] = schedule[0]["id"]
    elif mutation == "wrong_update":
        schedule[10]["update"] += 1
    else:
        schedule[10]["microbatch"] += 1
    with pytest.raises(ValueError):
        train.validate_schedule(schedule)


def passing_resource_gate():
    report = {
        "schema": "long-context-lora-resource-v2", "completed": True,
        "resource_feasible": True,
        "real_training_launch_allowed_by_runtime_estimate": True,
        "real_fit_rows_read": False, "real_training_run": False,
        "source_hashes": {"scripts/long_context_lora_resources.py": train.OBSERVER_SHA256},
        "runtime_estimate": {"estimated_training_seconds": 18706.434764570935},
    }
    proof = {
        "passed": True, "training_resource_gate_passed": True,
        "report_sha256": train.RESOURCE_REPORT_SHA256,
        "verifier_sha256": train.RESOURCE_VERIFIER_SHA256,
        "updates": 20, "microbatches": 80, "adapter_tensors": 16,
        "adapter_scalars": 327680, "fresh_reload_parity": True,
        "actual_fit_training_run": False,
        "estimated_training_seconds": 18706.434764570935,
    }
    return report, proof


def test_resource_gate_requires_reviewed_pass_and_runtime_estimate():
    report, proof = passing_resource_gate()
    train.validate_resource_gate(report, proof)
    report["runtime_estimate"]["estimated_training_seconds"] = 21600.1
    with pytest.raises(ValueError, match="runtime estimate"):
        train.validate_resource_gate(report, proof)


def test_resource_gate_rejects_unverified_or_wrong_observer():
    report, proof = passing_resource_gate()
    proof["passed"] = False
    with pytest.raises(ValueError, match="verification"):
        train.validate_resource_gate(report, proof)
    report, proof = passing_resource_gate()
    report["source_hashes"]["scripts/long_context_lora_resources.py"] = "0" * 64
    with pytest.raises(ValueError, match="fixed gate"):
        train.validate_resource_gate(report, proof)


def scored(top1, before, *, passed, raw=True, candidates=True, top5=True, rows=640):
    return {
        "metrics": {"all": {"rows": rows, "before_top1": before, "top1": top1,
                             "before_top5": rows, "top5": rows}},
        "gate": {"passed": passed, "net_top1_gain": top1 - before,
                 "corrections": max(top1 - before, 0), "regressions": 0,
                 "all_rows_accounted": rows == 640,
                 "zero_top5_losses": top5, "raw_preserved": raw,
                 "candidates_preserved": candidates},
    }


def test_train_gate_accepts_net_five_and_reports_regressions_without_zero_requirement():
    parent = scored(500, 500, passed=False)
    trained = scored(505, 500, passed=True)
    trained["gate"].update(corrections=8, regressions=3)
    gate = train.require_train_gate(parent, trained)
    assert gate == {"passed": True, "net_gain": 5, "required_net_gain": 5,
                    "parent_correct": 500, "trained_correct": 505,
                    "corrections": 8, "regressions": 3}


@pytest.mark.parametrize("change", ["gain", "rows", "raw", "candidates", "top5"])
def test_train_gate_rejects_or_fails_required_conditions(change):
    parent = scored(500, 500, passed=False)
    trained = scored(505, 500, passed=True)
    if change == "gain":
        trained = scored(504, 500, passed=False)
        assert train.require_train_gate(parent, trained)["passed"] is False
        return
    if change == "rows":
        trained["metrics"]["all"]["rows"] = 639
    else:
        key = {"raw": "raw_preserved", "candidates": "candidates_preserved",
               "top5": "zero_top5_losses"}[change]
        trained["gate"][key] = False
    with pytest.raises(ValueError):
        train.require_train_gate(parent, trained)


def test_default_output_never_overwrites_existing_run(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        train.run(output)
    assert not (output / "report.json").exists()


def test_training_source_has_no_resource_or_old_adapter_load_path():
    source = Path(train.__file__).read_text()
    assert "torch" not in source
    assert "RESOURCE_REPORT" not in source[source.index("load_adapters("):]
    assert '"resource_adapter_loaded": False' in source
    assert '"old_adapter_loaded": False' in source
    assert "from scripts.long_context_lora_resources import" in source


def test_write_jsonl_is_canonical_and_hashes_exact_bytes(tmp_path):
    rows = [{"b": "한", "a": 1}, {"a": 2}]
    path = tmp_path / "rows.jsonl"
    digest = train.write_jsonl(path, rows)
    expected = '{"a":1,"b":"한"}\n{"a":2}\n'.encode()
    assert path.read_bytes() == expected
    import hashlib
    assert digest == hashlib.sha256(expected).hexdigest()


def test_reload_logits_require_finite_nonempty_same_length_vectors():
    delta, close = train.validate_reload_logits([1.0, 2.0], [1.0, 2.000001])
    assert delta == pytest.approx(0.000001)
    assert close is True
    for expected, actual in (([1.0, float("inf")], [1.0, float("inf")]),
                             ([1.0], []), ([], []), ([1.0], [1.0, 2.0])):
        with pytest.raises(RuntimeError, match="probe logits"):
            train.validate_reload_logits(expected, actual)


def test_late_failure_resets_previously_assembled_success_flags():
    report = {"completed": True, "training_gate_passed": True,
              "diagnostic_metrics": {"computed": True}}
    train.mark_failed(report, RuntimeError("late write"), 21600.1)
    assert report["completed"] is False
    assert report["training_gate_passed"] is False
    assert report["terminal_status"] == "failed"
    assert report["diagnostic_metrics"] == {"computed": True}


def test_terminal_write_time_is_checked_after_serialization_boundary():
    assert train.require_whole_time(10.0, 10.0 + 21600.0) == 21600.0
    with pytest.raises(RuntimeError, match="terminal report write"):
        train.require_whole_time(10.0, 10.0 + 21600.000001)
