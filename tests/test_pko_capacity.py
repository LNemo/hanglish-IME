import copy
import json
from pathlib import Path

import pytest

from scripts import probe_pko_capacity as capacity


def test_prepare_is_model_free_and_binds_fixed_train_only_boundaries():
    authorization = capacity.prepare()
    assert len(authorization["fit_rows"]) == 64
    assert len(authorization["inner_rows"]) == 64
    assert set(authorization["schedules"]) == {"matched", "deranged"}
    assert all(len(groups) == 128 for groups in authorization["schedules"].values())
    assert all(sum(map(len, groups)) == 1024
               for groups in authorization["schedules"].values())
    assert all("dev.jsonl" not in path and "test.jsonl" not in path
               for path in authorization["pins"])


def test_selection_is_deranged_bijective_quartile_matched_and_quota_exact():
    authorization = capacity.prepare()
    rows = authorization["fit_rows"]
    by_id = {row["id"]: row for row in rows}
    assert len({row["id"] for row in rows}) == 64
    assert len({row["group"] for row in rows}) == 64
    assert {row["id"] for row in rows} == {row["donor_id"] for row in rows}
    assert all(row["id"] != row["donor_id"] for row in rows)
    assert all(row["length_bin"] == by_id[row["donor_id"]]["length_bin"]
               for row in rows)
    counts = {name: sum(row["category"] == name for row in rows)
              for name in capacity.EXPECTED_CATEGORY_COUNTS}
    assert counts == capacity.EXPECTED_CATEGORY_COUNTS


def test_two_arm_schedule_changes_only_source_alignment():
    rows = capacity.prepare()["fit_rows"]
    matched = [row for group in capacity.build_schedule(rows, "matched") for row in group]
    deranged = [row for group in capacity.build_schedule(rows, "deranged") for row in group]
    assert [row["id"] for row in matched] == [row["id"] for row in deranged]
    assert [row["target"] for row in matched] == [row["target"] for row in deranged]
    assert all(left["source_row_id"] == left["id"]
               for left in matched)
    assert all(right["source_row_id"] != right["id"]
               for right in deranged)
    assert [row["source"] for row in matched] != [row["source"] for row in deranged]
    assert [row["cycle"] for row in matched] == [row["cycle"] for row in deranged]


def _metrics(*, normal_nll, normal_lower=64, median_delta=.5,
             exact=40, eos=64):
    conditioning = {"normal_token_nll": normal_nll,
                    "normal_lower_rows": normal_lower,
                    "median_row_delta": median_delta}
    generation = {"whole_exact": exact, "eos_complete": eos}
    return conditioning, generation


def test_capacity_gate_matches_predeclared_plan_thresholds():
    baseline_fit_c, baseline_fit_g = _metrics(
        normal_nll=2.0, median_delta=.2, exact=3)
    inner_c, inner_g = _metrics(normal_nll=2.4, exact=2)
    baseline = {"conditioning": {"fit": baseline_fit_c, "inner": inner_c},
                "generation": {"fit": baseline_fit_g, "inner": inner_g}}
    matched_fit_c, matched_fit_g = _metrics(
        normal_nll=1.0, normal_lower=56, median_delta=.4, exact=32, eos=60)
    deranged_fit_c, deranged_fit_g = _metrics(
        normal_nll=1.1, exact=31, eos=64)
    matched = {"conditioning": {"cycle_16": {"fit": matched_fit_c,
                                                "inner": inner_c}},
               "generation": {"fit": matched_fit_g, "inner": inner_g}}
    deranged = {"conditioning": {"cycle_16": {"fit": deranged_fit_c,
                                                 "inner": inner_c}},
                "generation": {"fit": deranged_fit_g, "inner": inner_g}}
    decision = capacity.capacity_decision(baseline, matched, deranged)
    assert decision["small_sample_reconstruction_gate"] is True
    assert decision["input_alignment_contrast_gate"] is True
    assert decision["supports_bounded_alignment_learnability"] is True
    assert decision["runtime_promotion"] is False

    mutations = [
        ("conditioning", "normal_token_nll", 1.001),
        ("conditioning", "normal_lower_rows", 55),
        ("conditioning", "median_row_delta", .399),
        ("generation", "whole_exact", 31),
        ("generation", "eos_complete", 59),
    ]
    for area, key, value in mutations:
        changed = copy.deepcopy(matched)
        target = (changed["conditioning"]["cycle_16"]["fit"]
                  if area == "conditioning" else changed["generation"]["fit"])
        target[key] = value
        assert capacity.capacity_decision(baseline, changed, deranged)[
            "supports_bounded_alignment_learnability"] is False


def test_capacity_contrast_requires_nll_margin_and_strict_exact_advantage():
    condition, generated = _metrics(normal_nll=2.0, exact=0)
    baseline = {"conditioning": {"fit": condition, "inner": condition},
                "generation": {"fit": generated, "inner": generated}}
    match_c, match_g = _metrics(normal_nll=.8, median_delta=4, exact=40)
    inner_c, inner_g = _metrics(normal_nll=2, exact=0)
    matched = {"conditioning": {"cycle_16": {"fit": match_c, "inner": inner_c}},
               "generation": {"fit": match_g, "inner": inner_g}}
    deranged_c, deranged_g = _metrics(normal_nll=.899, exact=39)
    deranged = {"conditioning": {"cycle_16": {"fit": deranged_c,
                                                 "inner": inner_c}},
                "generation": {"fit": deranged_g, "inner": inner_g}}
    assert capacity.capacity_decision(baseline, matched, deranged)[
        "input_alignment_contrast_gate"] is False
    deranged["conditioning"]["cycle_16"]["fit"]["normal_token_nll"] = .9
    deranged["generation"]["fit"]["whole_exact"] = 40
    assert capacity.capacity_decision(baseline, matched, deranged)[
        "input_alignment_contrast_gate"] is False


def test_checkpoint_persists_optimizer_rng_cursor_and_model(tmp_path):
    torch = pytest.importorskip("torch")

    class Saver:
        def save_pretrained(self, path, **kwargs):
            Path(path).mkdir(exist_ok=True)
            (Path(path) / "artifact.bin").write_bytes(b"safe")

        def state_dict(self):
            return {"weight": torch.tensor([1.0])}

    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4, weight_decay=.01)
    state = {"cycles": 8, "microsteps": 512, "optimizer_updates": 64}
    pointer = capacity.save_arm_checkpoint(
        tmp_path, "matched", 8, Saver(), optimizer, Saver(),
        {"pins": {"input": "a" * 64}}, state)
    assert pointer["arm"] == "matched" and pointer["cycle"] == 8
    assert pointer["state"] == state
    assert pointer["model_files"] and len(pointer["optimizer_sha256"]) == 64
    assert pointer["python_random_state"]
    payload = torch.load(tmp_path / pointer["optimizer_path"], weights_only=True)
    assert set(payload) == {"optimizer", "torch_rng_state"}


def test_output_is_first_run_only(tmp_path):
    output = tmp_path / "capacity"
    assert capacity.prepare_output(output) == output
    with pytest.raises(FileExistsError):
        capacity.prepare_output(output)


def test_resource_and_evaluation_schedule_are_fixed():
    assert capacity.LIMITS == {"rss_bytes": 6 * 1024**3,
                               "swap_growth_bytes": 512 * 1024**2,
                               "whole_seconds": 1200.0,
                               "update_seconds": 120.0}
    assert capacity.CYCLES == 16
    assert capacity.CONDITIONING_CYCLES == {4, 8, 16}
    assert capacity.CHECKPOINT_CYCLES == {8, 16}
    assert capacity.UPDATES_PER_ARM == 128
    assert capacity.MICROSTEPS_PER_ARM == 1024


def test_failure_report_contract_is_train_only_and_nonpromoting():
    source = Path(capacity.__file__).read_text()
    assert '"dev_opened": False' in source
    assert '"test_opened": False' in source
    assert '"runtime_promotion": False' in source
    assert '"production_changed": False' in source
