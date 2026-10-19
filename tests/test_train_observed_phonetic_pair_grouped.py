import hashlib
import json
from pathlib import Path

import pytest

from scripts import train_observed_phonetic_pair_grouped as module


class Operation:
    def __init__(self, owner, name, update):
        self.owner, self.name, self.update = owner, name, update

    def __enter__(self):
        self.owner.calls.append((self.name, self.update))

    def __exit__(self, *_):
        return False


class Watchdog:
    def __init__(self):
        self.calls = []

    def operation(self, name, *, update=False):
        return Operation(self, name, update)


def test_balanced_groups_are_contiguous_bounded_and_differ_by_at_most_one():
    values = list(range(129))
    groups = module.balanced_groups(values)
    assert [len(group) for group in groups] == [26, 26, 26, 26, 25]
    assert [value for group in groups for value in group] == values
    assert module.balanced_groups(list(range(128))) == [
        list(range(start, start + 32)) for start in range(0, 128, 32)]
    assert module.balanced_groups([]) == []


def test_capture_group_writes_exact_rows_once_and_includes_durable_cost(tmp_path):
    watchdog = Watchdog()
    journal = tmp_path / "predictions.jsonl"
    ticks = iter([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    observed = module.capture_group(
        object(), [{"input": "ga"},
                   {"id": hashlib.sha256(b"na").hexdigest(), "input": "na"}],
        watchdog, journal, "beam_group_1", clock=lambda: next(ticks),
        generator=lambda model, source, beam: [
            (("가" if source == "ga" else "나"), -0.25)],
    )
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    assert watchdog.calls == [("beam_group_1", True)]
    assert rows == [
        {"id": hashlib.sha256(b"ga").hexdigest(), "input": "ga",
         "latency_s": pytest.approx(0.1), "outputs": [{"score": -0.25, "text": "가"}]},
        {"id": hashlib.sha256(b"na").hexdigest(), "input": "na",
         "latency_s": pytest.approx(0.1),
         "outputs": [{"score": -0.25, "text": "나"}]},
    ]
    assert observed == {
        "inputs": ["ga", "na"],
        "work_units": 2 * module.base.beam_work_units(2),
        "seconds": pytest.approx(0.5), "count": 2}


def test_grouped_projection_uses_upper_observed_group_work_rate():
    groups = [
        {"inputs": ["aa"] * 2, "count": 2, "work_units": 20, "seconds": 2.0},
        {"inputs": ["aaaa"], "count": 1, "work_units": 20, "seconds": 1.0},
    ]
    result = module.grouped_projection(8.0, 80, 800, groups, {"2": 10, "4": 5})
    work = (10 * module.base.beam_work_units(2)
            + 5 * module.base.beam_work_units(4))
    assert result["maximum_observed_group_seconds_per_work_unit"] == 0.1
    assert result["evaluation_work_units"] == work
    assert result["evaluation_seconds"] == pytest.approx(0.1 * work)
    assert result["projected_total_seconds"] == pytest.approx(
        60 + 1.3 * (80 + 0.1 * work))


def test_probe_selection_has_128_train_quantiles_and_synthetic_eval_max():
    records = [{"roman": "a" * (index + 1), "surface": "가"}
               for index in range(128)]
    selected = module.select_probe_inputs(records, evaluation_max_length=130)
    assert len(selected) == 129
    assert sum(row["kind"] == "train_length_stratified" for row in selected) == 128
    assert selected[-1] == {
        "input": "a" * 130, "kind": "synthetic_evaluation_max_length_resource_only"}


def test_capture_group_rejects_duplicate_or_oversized_inputs(tmp_path):
    watchdog = Watchdog()
    with pytest.raises(ValueError, match="duplicated"):
        module.capture_group(object(), [{"input": "ga"}, {"input": "ga"}],
                             watchdog, tmp_path / "x", "op", generator=lambda *_: [])
    with pytest.raises(ValueError, match="maximum"):
        module.capture_group(object(), [{"input": f"g{index}"} for index in range(33)],
                             watchdog, tmp_path / "x", "op", generator=lambda *_: [])


def test_grouped_probe_requires_frozen_initial_and_augmented_control_report():
    with pytest.raises(ValueError, match="frozen v1 control initial"):
        module.validate_initial_arguments("control", None, None)
    module.validate_initial_arguments("control", Path("initial.pt"), None)
    with pytest.raises(ValueError, match="grouped control report"):
        module.validate_initial_arguments("augmented", Path("initial.pt"), None)
    module.validate_initial_arguments(
        "augmented", Path("initial.pt"), Path("control/report.json"))
