import json
import math

import pytest

from scripts import verify_observed_phonetic_training as module


def records(size):
    return [{"roman": "a" * (1 + index % 5), "surface": "가" * (1 + index % 4)}
            for index in range(size)]


def update_rows(values, schedule):
    return [{"update": index, "loss": 1.0 / index, "gradient_norm": 0.5,
             "compute_seconds": 0.01, "padded_cost": module.batch_cost(values, batch),
             "pair_indices": list(batch),
             "parameter_finiteness_checked": module.finiteness_check_expected(
                 index, len(schedule))}
            for index, batch in enumerate(schedule, 1)]


def resource(at, elapsed, stage, operation=None, operation_elapsed=None):
    return {"at": f"2026-09-21T00:00:{at:02d}+00:00", "elapsed_s": elapsed,
            "operation": operation, "operation_elapsed_s": operation_elapsed,
            "rss_bytes": 100, "stage": stage, "swap_growth_bytes": 0,
            "swap_used_bytes": 0}


def test_schedule_and_update_ledger_are_exact_and_deterministic():
    values = records(9)
    first = module.cyclic_schedule(9, seed=3, updates=5, batch_size=4)
    second = module.cyclic_schedule(9, seed=3, updates=5, batch_size=4)
    assert first == second and all(len(batch) == 4 for batch in first)
    evidence = module.verify_updates(update_rows(values, first), values, first, batch_size=4)
    assert evidence["updates"] == 5
    assert evidence["parameter_finiteness_checks"] == 2


@pytest.mark.parametrize("mutation,match", [
    (lambda row: row["pair_indices"].reverse(), "pair indices"),
    (lambda row: row.update(padded_cost=1), "padded cost"),
    (lambda row: row.update(loss=math.nan), "non-finite"),
    (lambda row: row.update(parameter_finiteness_checked=False), "finiteness flag"),
])
def test_update_verifier_rejects_tampered_indices_costs_finiteness_and_flags(mutation, match):
    values = records(130)
    schedule = module.cyclic_schedule(130, seed=7, updates=130, batch_size=128)
    rows = update_rows(values, schedule)
    target = rows[0] if "flag" not in match else rows[127]
    mutation(target)
    with pytest.raises(ValueError, match=match):
        module.verify_updates(rows, values, schedule)


def test_real_finiteness_positions_are_one_every_128_and_final():
    positions = [i for i in range(1, module.UPDATES + 1)
                 if module.finiteness_check_expected(i)]
    assert positions[0] == 1 and positions[-1] == 3954
    assert positions[1:-1] == list(range(128, 3841, 128))
    assert len(positions) == 32


def test_resource_trace_accepts_atomic_operations_and_rejects_tampering():
    rows = [resource(0, 0.0, "operation_start", "one", 0.0),
            resource(1, 0.1, "watchdog", "one", 0.1),
            resource(2, 0.2, "operation_end", "one", 0.2),
            resource(3, 0.3, "final_before_report")]
    result = module.verify_resource_trace(rows, ["one"], "final_before_report")
    assert result["operations"] == 1 and result["final"] == rows[-1]
    changed = json.loads(json.dumps(rows)); changed[2]["operation"] = "other"
    with pytest.raises(ValueError, match="operation end"):
        module.verify_resource_trace(changed, ["one"], "final_before_report")
    changed = json.loads(json.dumps(rows)); changed[2]["operation_elapsed_s"] = 60.1
    with pytest.raises(ValueError, match="operation end"):
        module.verify_resource_trace(changed, ["one"], "final_before_report")


def test_resource_trace_allows_only_idle_watchdog_after_final():
    rows = [resource(0, 0.0, "operation_start", "one", 0.0),
            resource(1, 0.1, "operation_end", "one", 0.1),
            resource(2, 0.2, "final"), resource(3, 0.3, "watchdog")]
    module.verify_resource_trace(rows, ["one"], "final")
    rows[-1]["operation"] = "one"; rows[-1]["operation_elapsed_s"] = 0.1
    with pytest.raises(ValueError, match="watchdog operation"):
        module.verify_resource_trace(rows, ["one"], "final")


def test_resource_trace_rejects_invalid_swap_values_or_changed_baseline():
    rows = [resource(0, 0.0, "operation_start", "one", 0.0),
            resource(1, 0.1, "operation_end", "one", 0.1),
            resource(2, 0.2, "final")]
    rows[0]["swap_used_bytes"] = 100; rows[0]["swap_growth_bytes"] = 10
    rows[1]["swap_used_bytes"] = 100; rows[1]["swap_growth_bytes"] = 10
    rows[2]["swap_used_bytes"] = 100; rows[2]["swap_growth_bytes"] = 10
    result = module.verify_resource_trace(rows, ["one"], "final")
    assert result["baseline_swap_bytes"] == 90
    changed = json.loads(json.dumps(rows)); changed[1]["swap_used_bytes"] = 101
    with pytest.raises(ValueError, match="baseline changed"):
        module.verify_resource_trace(changed, ["one"], "final")
    changed = json.loads(json.dumps(rows)); changed[0]["swap_used_bytes"] = -1
    with pytest.raises(ValueError, match="resource row"):
        module.verify_resource_trace(changed, ["one"], "final")
    changed = json.loads(json.dumps(rows)); changed[0]["swap_growth_bytes"] = 101
    with pytest.raises(ValueError, match="baseline invalid"):
        module.verify_resource_trace(changed, ["one"], "final")


def test_main_and_finalization_traces_must_share_swap_baseline():
    main = {"baseline_swap_bytes": 90}
    finalization = {"baseline_swap_bytes": 90}
    assert module.require_same_swap_baseline(main, finalization) == 90
    finalization["baseline_swap_bytes"] = 91
    with pytest.raises(ValueError, match="main/finalization swap baseline"):
        module.require_same_swap_baseline(main, finalization)


@pytest.mark.parametrize("name", [
    "failure.json", "failure-report.json", "resource-watchdog-failure.json",
])
def test_all_failure_artifacts_are_rejected(tmp_path, name):
    module.require_no_failure_artifacts(tmp_path)
    (tmp_path / name).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match=name):
        module.require_no_failure_artifacts(tmp_path)


def test_source_has_no_torch_or_model_imports():
    source = (module.ROOT / "scripts/verify_observed_phonetic_training.py").read_text()
    assert "import torch" not in source
    assert "hanglish.models" not in source
    assert "torch.load" not in source


def test_parser_requires_run_prepared_and_probe():
    args = module.parser().parse_args(["--run", "r", "--prepared", "p", "--probe", "q"])
    assert str(args.run) == "r" and str(args.prepared) == "p" and str(args.probe) == "q"
