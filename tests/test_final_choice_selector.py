import copy
import hashlib
import json

import pytest
import torch

import scripts.train_final_choice_selector as module


def segment(source, text, kind="syllable"):
    return {"input_start": 0, "input_end": len(source), "source": source,
            "text": text, "kind": kind}


def candidate(source, text, score):
    return {"text": text, "score": score, "segments": [segment(source, text)]}


def row(parent_correct=True):
    source = "il"
    candidates = [candidate(source, "일", 1.5), candidate(source, "이", .5),
                  {"text": source, "score": None,
                   "segments": [segment(source, source, "raw")] }]
    positive = [0] if parent_correct else [1]
    return {
        "id": "case", "group_id": "group", "origin": "natural",
        "input": source, "context": "문맥", "expected": [candidates[positive[0]]["text"]],
        "result": "일", "candidates": candidates, "finite_indices": [0, 1],
        "accepted_positive_indices": positive, "label_aligned": True,
        "candidate_can_score": [True, True, True], "all_candidates_can_score": True,
        "selector_eligible": True, "ineligible_reasons": [],
        "runtime_version": "runtime", "model_artifact_flags": {
            "expected_count": 16, "response_matches_expected": True},
        "response_flags": {"version": "runtime", "model_artifacts": {
            f"weight-{i}": "a" * 64 for i in range(16)}, "raw": False},
        "raw_bypass_response": {
            "input": source, "context": "문맥", "engine": "neural", "raw": True,
            "version": "runtime", "result": source, "engine_ms": .1,
            "candidates": [{"text": source, "score": 0.0,
                            "segments": [segment(source, source, "raw")]}],
            "model_artifacts": {}, "ranker": None,
        },
        "engine_ms": .2, "raw_engine_ms": .1,
    }


def selection(value):
    return {key: copy.deepcopy(value[key]) for key in
            ("id", "group_id", "origin", "input", "context", "expected")}


def write_capture(tmp_path, role, rows, monkeypatch):
    directory = tmp_path / role
    directory.mkdir()
    selections = [selection(value) for value in rows]
    cases_raw = b"".join((json.dumps(value, sort_keys=True) + "\n").encode() for value in rows)
    selection_raw = b"".join((json.dumps(value, sort_keys=True) + "\n").encode()
                             for value in selections)
    (directory / "cases.jsonl").write_bytes(cases_raw)
    (directory / "selection.jsonl").write_bytes(selection_raw)
    manifest = {
        "completed": True, "role": role,
        "scope": "fresh actual parent TRAIN capture for final-choice learning",
        "partition_manifest_sha256": module.PARTITION_SHA256,
        "expected_rows": len(rows), "rows": len(rows),
        "partition_ordered_ids": [value["id"] for value in rows],
        "selection_sha256_before_inference": hashlib.sha256(selection_raw).hexdigest(),
        "selection_sha256": hashlib.sha256(selection_raw).hexdigest(),
        "cases_sha256": hashlib.sha256(cases_raw).hexdigest(),
        "runtime_version": "runtime",
        "runtime_source_hashes": {"source.py": "b" * 64},
        "model_artifact_hashes": {f"weight-{i}": "a" * 64 for i in range(16)},
        "source_hashes": {"source.py": "b" * 64,
                          "train.jsonl": "c" * 64, "runtime.json": "d" * 64,
                          **{f"weight-{i}": "a" * 64 for i in range(16)}},
        "source_hashes_after": {"source.py": "b" * 64,
                                "train.jsonl": "c" * 64, "runtime.json": "d" * 64,
                                **{f"weight-{i}": "a" * 64 for i in range(16)}},
        "train_inputs": {"natural": {"path": "train.jsonl", "sha256": "c" * 64}},
        "runtime_evidence_path": "runtime.json", "runtime_evidence_sha256": "d" * 64,
        "diagnostics": {"top1": sum(value["result"] in value["expected"] for value in rows),
                        "top5": sum(any(candidate["text"] in value["expected"]
                                        for candidate in value["candidates"]) for value in rows),
                        "label_aligned": sum(value["label_aligned"] for value in rows),
                        "selector_eligible": sum(value["selector_eligible"] for value in rows),
                        "all_candidates_can_score": sum(
                            value["all_candidates_can_score"] for value in rows)},
        "baseline_accuracy_is_diagnostic_only": True,
        "split_or_threshold_selected_from_capture_outcomes": False,
        "training_run": False, "new_selector_inference_run": False,
        "development_data_read": False, "final_data_read": False, "deployed": False,
    }
    raw = json.dumps(manifest).encode()
    (directory / "manifest.json").write_bytes(raw)
    monkeypatch.setitem(module.ROLE_SPECS, role, len(rows))
    return directory, hashlib.sha256(raw).hexdigest()


def test_load_fit_rejects_malicious_role_and_truncated_capture(tmp_path, monkeypatch):
    value = row()
    directory, digest = write_capture(tmp_path, "natural_fit", [value], monkeypatch)
    loaded = module.load_fit_role(directory, digest, "natural_fit")
    assert loaded["rows"][0]["id"] == "case"
    with pytest.raises(ValueError, match="FIT roles only"):
        module.load_fit_role(directory, digest, "natural_calibration")
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["partition_ordered_ids"].append("missing")
    raw = json.dumps(manifest).encode()
    (directory / "manifest.json").write_bytes(raw)
    with pytest.raises(ValueError, match="manifest"):
        module.load_fit_role(directory, hashlib.sha256(raw).hexdigest(), "natural_fit")


def test_load_fit_rejects_modified_candidate_after_manifest(tmp_path, monkeypatch):
    value = row()
    directory, digest = write_capture(tmp_path, "natural_fit", [value], monkeypatch)
    value["candidates"][1]["score"] = .4
    (directory / "cases.jsonl").write_text(json.dumps(value) + "\n")
    with pytest.raises(ValueError, match="hash differs"):
        module.load_fit_role(directory, digest, "natural_fit")


def test_build_example_uses_finite_set_parent_weight_and_multi_positive():
    value = row(parent_correct=False)
    value["expected"] = ["일", "이"]
    value["accepted_positive_indices"] = [0, 1]
    example = module.build_example(value, "natural_fit")
    assert example.positives == (0, 1)
    assert example.finite_source_indices == (0, 1)
    assert example.parent_correct is True
    assert example.loss_weight == module.RECIPE["other_weight"]
    value["expected"] = ["이"]
    value["accepted_positive_indices"] = [1]
    example = module.build_example(value, "natural_fit")
    assert example.parent_correct is False
    assert example.loss_weight == module.RECIPE["parent_error_weight"]


def test_selector_loss_applies_kd_only_to_parent_correct_and_checks_shape():
    correct = module.build_example(row(parent_correct=True), "natural_fit")
    wrong = module.build_example(row(parent_correct=False), "natural_fit")
    scores = torch.tensor([1.0, .2, .1, .9], requires_grad=True)
    actual = module.selector_loss(scores, [correct, wrong])
    no_kd_recipe = {**module.RECIPE, "distillation_weight": 0.}
    without_kd = module.selector_loss(scores, [correct, wrong], no_kd_recipe)
    assert actual > without_kd
    actual.backward()
    assert torch.isfinite(scores.grad).all()
    with pytest.raises(ValueError, match="cover"):
        module.selector_loss(scores[:-1], [correct, wrong])


def test_missing_positive_is_skipped_and_reportable():
    value = row()
    value["accepted_positive_indices"] = []
    value["label_aligned"] = False
    assert module.build_example(value, "natural_fit") is None


def test_oversized_capture_is_retained_but_excluded_from_training():
    value = row()
    value["candidate_can_score"] = [True, False, True]
    value["all_candidates_can_score"] = False
    value["selector_eligible"] = False
    value["ineligible_reasons"] = ["candidate_exceeds_sentence_model_budget"]
    module._validate_row(
        value, selection(value), "natural_fit", "runtime",
        {f"weight-{i}": "a" * 64 for i in range(16)},
    )
    assert module.build_example(value, "natural_fit") is None


class FakeBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.electra = torch.nn.Module()
        self.electra.encoder = torch.nn.Module()
        self.electra.encoder.layer = torch.nn.ModuleList(
            [torch.nn.Linear(1, 1) for _ in range(4)]
        )


class FakeModel(torch.nn.Module):
    base_model_prefix = "base"

    def __init__(self):
        super().__init__()
        self.base = FakeBase()
        self.classifier = torch.nn.Linear(1, 1)

    @property
    def base_model(self):
        return self.base


def test_only_last_two_encoder_layers_and_classifier_are_trainable():
    model = FakeModel()
    details = module.configure_trainable_model(model)
    assert details["unfrozen_encoder_layers"] == 2
    assert all(not parameter.requires_grad for layer in model.base.electra.encoder.layer[:2]
               for parameter in layer.parameters())
    assert all(parameter.requires_grad for layer in model.base.electra.encoder.layer[2:]
               for parameter in layer.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())


def test_resource_bounds_fail_closed(monkeypatch):
    monkeypatch.setattr(module, "rss_bytes", lambda: module.RECIPE["rss_limit_bytes"] + 1)
    with pytest.raises(RuntimeError, match="resource bound"):
        module.require_resources(0.0, "fixture")


@pytest.mark.parametrize("scores", [[float('nan'), 0.], [0., float('inf')], [0.], []])
def test_evaluation_rejects_invalid_score_vectors(scores):
    class Ranker:
        model = torch.nn.Linear(1, 1)

        def score_pairs(self, contexts, candidates):
            return scores

    example = module.build_example(row(), 'natural_fit')
    with pytest.raises(ValueError, match='finite and cover'):
        module.evaluate_examples(Ranker(), [example])


def test_save_load_parity_rejects_nonfinite_missing_and_overflow_scores():
    examples = [module.build_example(row(), 'natural_fit')]
    for left, right in [([[float('nan'), 0.]], [[float('nan'), 0.]]),
                        ([[1., 0.]], [[1.]]), ([[1., 0.]], [])]:
        with pytest.raises(ValueError):
            module.score_parity_delta(left, right, examples)
    with pytest.raises(RuntimeError, match='differs'):
        module.score_parity_delta([[1e308, 0.]], [[-1e308, 0.]], examples)
    assert module.score_parity_delta([[1., 0.]], [[1., 0.]], examples) == 0.


def test_post_step_parameters_cannot_be_nonfinite():
    model = torch.nn.Linear(1, 1)
    module.require_finite_parameters(model)
    with torch.no_grad():
        model.weight.fill_(float('nan'))
    with pytest.raises(RuntimeError, match='nonfinite'):
        module.require_finite_parameters(model)
