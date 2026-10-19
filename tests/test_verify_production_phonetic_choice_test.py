import ast
import copy
from pathlib import Path

import pytest

from scripts import verify_production_phonetic_choice_test as verifier
from tests.test_verify_production_phonetic_choice_dev import fixture


def response(text, candidates, ranked=True):
    return {"result": text, "candidates": [{"text": value} for value in candidates],
            "sentence_ranking_applied": ranked}


def comparison_rows():
    rows, labels = [], []
    for index in range(1000):
        identifier = f"test-{index:04d}"
        source = f"roman sentence {index}"
        gold, wrong = f"정답{index}", f"오답{index}"
        parent_top1 = index < 953
        parent_top5 = index < 972
        parent_candidates = [gold if parent_top1 else wrong]
        if parent_top5 and not parent_top1:
            parent_candidates.append(gold)
        proposed_top1 = parent_top1 or index == 953
        proposed_candidates = [gold if proposed_top1 else wrong]
        if parent_top5 and not proposed_top1:
            proposed_candidates.append(gold)
        rows.append({"id": identifier,
                     "request": {"input": source, "context": ""},
                     "parent": response(parent_candidates[0], parent_candidates),
                     "proposed": response(proposed_candidates[0], proposed_candidates),
                     "actual_base_pool": list(dict.fromkeys(proposed_candidates))})
        labels.append({"id": identifier, "input": source, "context": "",
                       "split": "test", "expected": [gold]})
    return rows, labels


def test_verifier_imports_no_test_producer_adapter_projector_or_tensor():
    tree = ast.parse(Path(verifier.__file__).read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    forbidden = ("evaluate_production_phonetic_choice_test",
                 "production_phonetic_choice_converter",
                 "phonetic_choice_features", "torch")
    assert not any(name.endswith(item) or name == item
                   for name in imports for item in forbidden)


def test_test_probe_directly_compares_both_fresh_full_paths(monkeypatch):
    monkeypatch.setattr(verifier.pure, "romanize_word", lambda text: "ga")
    _, _, expected_dev, probe, group = fixture()
    expected_test = {"id": expected_dev["id"], "request": expected_dev["request"],
                     "proposed": expected_dev["parent"]}
    values = verifier.verify_probe(group, probe, expected_test)
    assert values == {"paths": 1, "pairs": 1, "supported_edges": 1}

    broken = copy.deepcopy(probe)
    item = broken["parent_paths"][0]
    broken["parent_paths"][0] = (item[0], 9.0, item[2])
    with pytest.raises(ValueError, match="full path parity"):
        verifier.verify_probe(group, broken, expected_test)


def test_aggregate_replays_strict_1000_regression_gate_and_stage_counts():
    rows, labels = comparison_rows()
    metrics = verifier.aggregate(rows, labels)
    assert metrics["parent"]["counts"] == {
        "rows": 1000, "top1": 953, "top5": 972, "errors": 0}
    assert metrics["proposed"]["counts"]["top1"] == 954
    assert metrics["corrections"] == 1
    assert metrics["regressions"] == metrics["top5_losses"] == 0
    assert metrics["quality_gate"] is True
    assert metrics["actual_base_pool_stage_counts"] == {
        "ranked_rows": 1000, "actual_base_pool_gold": 972,
        "public_top5_gold": 972, "selected_gold": 954,
        "actual_pool_to_public_loss": 0, "actual_pool_to_first_loss": 18}


def test_aggregate_rejects_regression_and_label_reordering():
    rows, labels = comparison_rows()
    damaged = copy.deepcopy(rows)
    gold = labels[0]["expected"][0]
    damaged[0]["proposed"] = response("손실", ["손실", gold])
    metrics = verifier.aggregate(damaged, labels)
    assert metrics["regressions"] == 1
    assert metrics["quality_gate"] is False

    with pytest.raises(ValueError, match="identity"):
        verifier.aggregate(rows, list(reversed(labels)))


def test_expected_test_operation_schedule_has_exact_1038_boundaries():
    requests = [(f"test-{index:04d}", f"roman sentence {index}")
                for index in range(1000)]
    groups = verifier.pure.probe_groups([
        (identifier, source, None) for identifier, source in requests])
    operations = verifier.expected_operations(requests, groups)
    assert len(operations) == 1038
    assert operations[:2] == ["authenticate_dev_then_test_sources", "load_models"]
    assert operations[-4:] == ["close_capture", "late_test_label_join",
                               "write_report", "write_completion"]
    assert sum(name.startswith("convert:") for name in operations) == 1000
