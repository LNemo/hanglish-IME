import copy
import math

import pytest

from scripts import verify_document_phonetic_capture as module


def requests(count):
    return [{"id": str(index), "input": "a" * (index % 7 + 1)} for index in range(count)]


def test_balanced_groups_are_contiguous_bounded_and_deterministic():
    values = requests(10166)
    groups = module.balanced_groups(values)
    assert len(groups) == 318
    assert {len(group) for group in groups} == {31, 32}
    assert [row for group in groups for row in group] == values
    assert groups == module.balanced_groups(values)


def test_group_rows_reconstruct_inputs_work_and_histogram():
    values = requests(65)
    groups = module.balanced_groups(values)
    rows = []
    for group in groups:
        sources = [row["input"] for row in group]
        rows.append({"inputs": sources, "count": len(group), "seconds": 0.5,
                     "work_units": sum(module.beam_work_units(len(x)) for x in sources)})
    evidence = module.verify_groups(rows, groups)
    assert evidence["groups"] == 3 and evidence["inputs"] == 65
    assert evidence["size_histogram"] == {"21": 1, "22": 2}


@pytest.mark.parametrize("mutation,match", [
    (lambda row: row["inputs"].reverse(), "group differs"),
    (lambda row: row.update(work_units=0), "group differs"),
    (lambda row: row.update(seconds=math.nan), "group differs"),
    (lambda row: row.update(count=99), "group differs"),
])
def test_group_rows_reject_tampered_order_work_timing_and_count(mutation, match):
    groups = module.balanced_groups(requests(5), maximum=3)
    rows = []
    for group in groups:
        sources = [row["input"] for row in group]
        rows.append({"inputs": sources, "count": len(group), "seconds": 0.1,
                     "work_units": sum(module.beam_work_units(len(x)) for x in sources)})
    changed = copy.deepcopy(rows)
    mutation(changed[0])
    with pytest.raises(ValueError, match=match):
        module.verify_groups(changed, groups)


def test_failure_artifacts_include_watchdog_failure(tmp_path):
    module.require_no_failures(tmp_path)
    for name in ("failure.json", "failure-report.json", "resource-watchdog-failure.json"):
        path = tmp_path / name
        path.write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match=name):
            module.require_no_failures(tmp_path)
        path.unlink()


def test_combined_timing_matches_producer_sampling_order():
    module.verify_combined_timing(600.0, 300.0, 899.0, 901.0)
    with pytest.raises(ValueError, match="combined capture timing"):
        module.verify_combined_timing(600.0, 300.0, 901.0, 902.0)
    with pytest.raises(ValueError, match="combined capture timing"):
        module.verify_combined_timing(600.0, 300.0, 899.0, 899.5)


def test_combined_timing_rejects_main_elapsed_before_training():
    with pytest.raises(ValueError, match="combined capture timing"):
        module.verify_combined_timing(600.0, 0.0, 599.0, 600.0)


def test_source_has_no_torch_model_or_reference_loading():
    source = (module.ROOT / "scripts/verify_document_phonetic_capture.py").read_text()
    assert "import torch" not in source
    assert "torch.load" not in source
    assert "load_references(" not in source
    assert "PhoneticTransducer" not in source


def test_parser_requires_capture_authorization_and_training():
    args = module.parser().parse_args([
        "--capture", "c", "--authorization", "a", "--training", "t"])
    assert (str(args.capture), str(args.authorization), str(args.training)) == ("c", "a", "t")
