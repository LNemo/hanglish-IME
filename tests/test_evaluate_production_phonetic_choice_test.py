import copy
import json

import pytest

from scripts import evaluate_production_phonetic_choice_test as evaluator
from tests.test_evaluate_production_phonetic_choice_dev import production_fixture


def dev_records():
    hashes = {name: character * 64 for (name, character) in zip(
        evaluator.DEV_EXPECTED, "abcde")}
    weights = json.loads(evaluator.WEIGHTS.read_text())["weights"]
    report = {
        "schema": "production-phonetic-choice-dev-v1",
        "completed": True,
        "split": "dev",
        "quality_gate": True,
        "improvement_gate": True,
        "parent": {"counts": dict(evaluator.EXPECTED_DEV_PARENT)},
        "proposed": {"counts": {"rows": 316, "top1": 263,
                                   "top5": 283, "errors": 0}},
        "regressions": 0,
        "top5_losses": 0,
        "english_ascii_signature": {
            "signature_regressions": 0,
            "literal_signature_regressions": 0,
        },
        "parent_version": evaluator.PRODUCTION_VERSION,
        "version": "experimental-production-phonetic-choice-1-fixture",
        "weights": weights,
        "production_changed": False,
        "test_accessed": False,
        "labels_opened_after_capture": True,
        "gold_not_passed_to_predictor": True,
        "pins": {
            evaluator.path_key(evaluator.ADAPTER):
                evaluator.EXPECTED_ADAPTER_SHA256,
            evaluator.path_key(evaluator.WEIGHTS):
                evaluator.EXPECTED_WEIGHTS_SHA256,
        },
    }
    completion = {"completed": True,
                  "report_sha256": hashes["report.json"]}
    closure = {"schema": "production-phonetic-choice-dev-v1-closure",
               "rows": 316, "capture_sha256": hashes["capture.jsonl"],
               "labels_opened": False}
    proof = {"verified": True,
             "report_sha256": hashes["report.json"],
             "capture_sha256": hashes["capture.jsonl"],
             "closure_sha256": hashes["capture.complete.json"],
             "completion_sha256": hashes["completion.json"],
             "decision": {"quality_gate": True}}
    return report, completion, closure, proof, hashes


def response(text, candidates):
    return {"result": text, "candidates": [{"text": value} for value in candidates]}


def comparison_rows():
    captures, labels = [], []
    for index in range(1000):
        identifier = f"test-{index:04d}"
        source = f"roman sentence {index}"
        gold = f"정답{index}"
        wrong = f"오답{index}"
        parent_top1 = index < 953
        parent_top5 = index < 972
        parent_candidates = [gold if parent_top1 else wrong]
        if parent_top5 and not parent_top1:
            parent_candidates.append(gold)
        proposed_top1 = parent_top1 or index == 953
        proposed_candidates = [gold if proposed_top1 else wrong]
        if parent_top5 and not proposed_top1:
            proposed_candidates.append(gold)
        captures.append({
            "id": identifier,
            "request": {"input": source, "context": ""},
            "parent": response(parent_candidates[0], parent_candidates),
            "proposed": {**response(proposed_candidates[0], proposed_candidates),
                         "sentence_ranking_applied": True},
            "actual_base_pool": list(dict.fromkeys(proposed_candidates)),
        })
        labels.append({"id": identifier, "input": source, "context": "",
                       "split": "test", "expected": [gold]})
    return captures, labels


def test_dev_hashes_are_frozen_and_actual_terminal_gate_authenticates():
    assert all(len(value) == 64 and value != evaluator.UNRESOLVED
               for value in evaluator.DEV_EXPECTED.values())
    identity, report, hashes = evaluator.authenticate_dev()
    assert identity["report_sha256"] == evaluator.DEV_EXPECTED["report.json"]
    assert identity["version"] == "experimental-production-phonetic-choice-1-30e29bb96e4a"
    assert report["proposed"]["counts"]["top1"] == 263
    assert hashes == evaluator.DEV_EXPECTED


def test_unresolved_dev_hash_stops_before_test_authorization():
    unresolved = dict(evaluator.DEV_EXPECTED)
    unresolved["report.json"] = evaluator.UNRESOLVED
    with pytest.raises(ValueError, match="not been frozen"):
        evaluator._resolved_dev_hashes(unresolved)

    calls = []
    def reject_dev():
        calls.append("dev")
        raise ValueError("dev rejected")
    def forbidden_test(split):
        calls.append(split)
        raise AssertionError("TEST authorization ran before DEV")
    with pytest.raises(ValueError, match="dev rejected"):
        evaluator.authorize(dev_loader=reject_dev,
                            reference_loader=forbidden_test)
    assert calls == ["dev"]


@pytest.mark.parametrize("mutation", ["quality", "regression", "top5", "ascii",
                                       "literal", "adapter", "proof"])
def test_dev_gate_rejects_any_quality_or_identity_drift(mutation):
    report, completion, closure, proof, hashes = dev_records()
    if mutation == "quality":
        report["quality_gate"] = False
    elif mutation == "regression":
        report["regressions"] = 1
    elif mutation == "top5":
        report["top5_losses"] = 1
    elif mutation == "ascii":
        report["english_ascii_signature"]["signature_regressions"] = 1
    elif mutation == "literal":
        report["english_ascii_signature"]["literal_signature_regressions"] = 1
    elif mutation == "adapter":
        report["pins"][evaluator.path_key(evaluator.ADAPTER)] = "0" * 64
    else:
        proof["verified"] = False
    with pytest.raises(ValueError):
        evaluator.validate_dev_gate(report, completion, closure, proof,
                                    hashes=hashes)


def test_test_capture_uses_actual_base_paths_without_historical_pool():
    source, frozen, converted, paths = production_fixture()
    evaluator.verify_capture(source, frozen["parent"], converted,
                             paths, [0.0] * 4)
    converted["phonetic_word_union"] = {}
    with pytest.raises(ValueError, match="phonetic proposal"):
        evaluator.verify_capture(source, frozen["parent"], converted,
                                 paths, [0.0] * 4)


def test_test_gate_requires_gain_and_zero_top1_top5_ascii_literal_losses():
    captures, labels = comparison_rows()
    metrics = evaluator.aggregate(captures, labels)
    assert metrics["parent"]["counts"] == evaluator.EXPECTED_PARENT
    assert metrics["proposed"]["counts"]["top1"] == 954
    assert metrics["corrections"] == 1
    assert metrics["regressions"] == metrics["top5_losses"] == 0
    assert metrics["quality_gate"] is True
    assert metrics["actual_base_pool_stage_counts"] == {
        "ranked_rows": 1000,
        "actual_base_pool_gold": 972,
        "public_top5_gold": 972,
        "selected_gold": 954,
        "actual_pool_to_public_loss": 0,
        "actual_pool_to_first_loss": 18,
    }

    damaged = copy.deepcopy(captures)
    gold = labels[0]["expected"][0]
    damaged[0]["proposed"] = response("손실", ["손실", gold])
    metrics = evaluator.aggregate(damaged, labels)
    assert metrics["regressions"] == 1
    assert metrics["quality_gate"] is False


def test_late_test_labels_require_exact_order_source_and_split():
    captures, labels = comparison_rows()
    with pytest.raises(ValueError, match="identities"):
        evaluator.aggregate(captures, list(reversed(labels)))
    labels[0]["split"] = "dev"
    with pytest.raises(ValueError, match="source contract"):
        evaluator.aggregate(captures, labels)


def test_run_requires_explicit_final_before_creating_output(tmp_path):
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="--final"):
        evaluator.run(output, final=False)
    assert not output.exists()
