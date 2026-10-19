import hashlib
import json

import pytest

from scripts import train_observed_phonetic_pair as module


def records(size):
    return [{"pair_id": str(i), "roman": "a" * (2 + i % 7),
             "surface": "가" * (1 + i % 5)} for i in range(size)]


def test_cyclic_schedule_is_full_fixed_deterministic_and_bridges_boundaries():
    first = module.cyclic_schedule(5, seed=9, updates=4, batch_size=3)
    second = module.cyclic_schedule(5, seed=9, updates=4, batch_size=3)
    assert first == second and len(first) == 4
    assert all(len(batch) == 3 for batch in first)
    assert set(index for batch in first for index in batch) == set(range(5))
    summary = module.schedule_summary(records(5), first)
    assert summary["presentations"] == 12
    assert sum(int(exposure) * pairs for exposure, pairs in
               summary["exposure_histogram"].items()) == 12


def test_real_schedule_fixed_compute_and_expected_exposure_ranges():
    control = module.schedule_summary(
        records(51361), module.cyclic_schedule(51361))
    augmented = module.schedule_summary(
        records(84345), module.cyclic_schedule(84345))
    assert control["updates"] == augmented["updates"] == 3954
    assert control["presentations"] == augmented["presentations"] == 506112
    assert (control["min_exposure"], control["max_exposure"]) == (9, 10)
    assert (augmented["min_exposure"], augmented["max_exposure"]) == (6, 7)
    assert control["schedule_sha256"] != augmented["schedule_sha256"]


def test_probe_selection_covers_cost_source_and_target_worst_cases():
    values = records(80)
    values[-1] = {"pair_id": "source", "roman": "a" * 40, "surface": "가"}
    values[-2] = {"pair_id": "target", "roman": "aa", "surface": "가" * 20}
    schedule = module.cyclic_schedule(len(values), seed=3, updates=40, batch_size=8)
    selected = module.select_probe_batches(values, schedule, count=16)
    selected_indices = {index for position in selected for index in schedule[position]}
    assert len(selected) == 16
    assert 79 in selected_indices and 78 in selected_indices
    assert max(module.batch_cost(values, schedule[p]) for p in selected) == max(
        module.batch_cost(values, batch) for batch in schedule)


def test_beam_sample_is_length_stratified_and_projection_uses_worst_bound():
    values = [{"pair_id": str(i), "roman": "a" * (i + 2), "surface": "가"}
              for i in range(40)]
    beams = module.select_beam_inputs(values, count=32, evaluation_max_length=43)
    assert len(beams) == 33 and len(beams[0]["input"]) == 2
    assert beams[-1] == {
        "input": "a" * 43, "kind": "synthetic_evaluation_max_length_resource_only"}
    observations = [{**row, "length": len(row["input"]), "seconds": 0.01}
                    for row in beams]
    projection = module.projected_seconds(
        8.0, 80, 800, observations, {"2": 40, "43": 60})
    expected_units = 40 * module.beam_work_units(2) + 60 * module.beam_work_units(43)
    assert projection["evaluation_work_units"] == expected_units
    assert projection["evaluation_seconds"] == pytest.approx(
        0.01 / module.beam_work_units(2) * expected_units)
    assert projection["evaluation_global_worst_per_input_diagnostic_seconds"] == 1.0
    assert projection["projected_total_seconds"] == pytest.approx(
        60 + 1.3 * (80 + projection["evaluation_seconds"]))


def test_probe_contract_rejects_arm_initial_or_evaluation_drift():
    pins = {"source": "hash"}
    expected_beams = [
        {"input": "aa", "kind": "train_length_stratified"} for _ in range(32)]
    expected_beams.append({
        "input": "a" * 43, "kind": "synthetic_evaluation_max_length_resource_only"})
    positions = list(range(16))
    observations = [{"ordinal": index + 1, "schedule_position": index,
                     "padded_cost": 10, "seconds": 0.1}
                    for index in positions]
    beam_observations = [{**row, "length": len(row["input"]), "seconds": 0.01}
                         for row in expected_beams]
    prepared = {
        "evaluation_inputs": 1046,
        "evaluation": {"input_length_histogram": {"2": 1044, "43": 2}},
        "arms": {"control": {"probe_update_positions": positions,
                               "probe_beam_inputs": expected_beams,
                               "total_padded_cost": 1000}},
    }
    projection = module.projected_seconds(
        sum(row["seconds"] for row in observations), 160, 1000, beam_observations,
        prepared["evaluation"]["input_length_histogram"])
    report = {
        "schema": "observed-phonetic-pair-probe-v1", "completed": True,
        "accepted": True, "arm": "control", "prepared_sha256": "p" * 64,
        "evaluation_inputs": 1046, "source_hashes": pins,
        "source_hashes_after": pins, "prototype_discarded": True,
        "recipe": module.RECIPE, "limits": module.LIMITS,
        "model_updates_run": 16, "beam_inputs_run": 33,
        "train_beam_inputs_run": 32, "synthetic_beam_inputs_run": 1,
        "update_selection": positions,
        "update_selection_sha256": hashlib.sha256(module.canonical(positions)).hexdigest(),
        "beam_selection_sha256": hashlib.sha256(module.canonical(expected_beams)).hexdigest(),
        "coverage": {"maximum_padded_cost": True, "longest_train_source": True,
                     "evaluation_max_length": True,
                     "longest_target_batch_included": True},
        "observations": observations, "beam_observations": beam_observations,
        "projection": projection,
        "final_resource": {"elapsed_s": 10.0, "rss_bytes": 100,
                           "swap_growth_bytes": 0},
        "initial": {"checkpoint_sha256": "c" * 64, "tensor_sha256": "t" * 64},
    }
    completion = {
        "schema": "observed-phonetic-pair-completion-v1", "completed": True,
        "report_sha256": "r" * 64,
        "resource_after_report": {"elapsed_s": 11.0, "rss_bytes": 100,
                                  "swap_growth_bytes": 0},
    }
    module.validate_probe(report, completion, "control", prepared, "p" * 64, pins,
                          "r" * 64)
    for key, value in (("arm", "augmented"), ("evaluation_inputs", 1000)):
        changed = json.loads(json.dumps(report)); changed[key] = value
        with pytest.raises(ValueError, match="probe contract"):
            module.validate_probe(changed, completion, "control", prepared, "p" * 64,
                                  pins, "r" * 64)
    changed = json.loads(json.dumps(report))
    changed["initial"]["tensor_sha256"] = "bad"
    with pytest.raises(ValueError, match="probe contract"):
        module.validate_probe(changed, completion, "control", prepared, "p" * 64,
                              pins, "r" * 64)
    changed_completion = json.loads(json.dumps(completion))
    changed_completion["resource_after_report"]["elapsed_s"] = 1800.1
    with pytest.raises(ValueError, match="completion contract"):
        module.validate_probe(report, changed_completion, "control", prepared,
                              "p" * 64, pins, "r" * 64)


def test_prepare_parser_requires_evaluation_count_and_commands_are_distinct():
    parser = module.parser()
    args = parser.parse_args(["prepare", "--output", "out", "--evaluation-inputs", "10166"])
    assert args.command == "prepare" and args.evaluation_inputs == 10166
    probe = parser.parse_args(["probe", "--arm", "control", "--prepared", "p",
                               "--output", "o"])
    train = parser.parse_args(["train", "--arm", "augmented", "--prepared", "p",
                               "--probe", "q", "--output", "o"])
    assert probe.command == "probe" and train.command == "train"


def test_pin_validation_rejects_changed_source(tmp_path, monkeypatch):
    path = tmp_path / "source"
    path.write_text("original", encoding="utf-8")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    wanted = hashlib.sha256(path.read_bytes()).hexdigest()
    module.verify_pins({"source": wanted})
    path.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="source pin differs"):
        module.verify_pins({"source": wanted})
