from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from scripts import evaluate_balanced_final_choice as evaluation
from tests.test_balanced_final_choice_policy import Ranker, candidates


def row():
    values = candidates()
    raw = deepcopy(values[-1])
    raw["score"] = 0.
    return {"id": "example", "group_id": "document", "input": "imnida",
            "context": "", "candidates": values, "candidate_can_score": [True] * 3,
            "raw_bypass_response": {"candidates": [raw], "result": "imnida"},
            "expected": ["입니다"]}


def test_direct_evaluation_correction_and_object_preservation():
    sample = row()
    before = deepcopy(sample)
    result = evaluation.evaluate_case(Ranker(), sample, "natural_evaluation_v2")
    assert not result["before_top1"] and result["top1"]
    assert result["accepted_action"] and result["chosen_index"] == 1
    assert result["before_top5"] and result["top5"]
    assert result["raw_bypass_preserved"]
    assert result["candidate_objects_and_content_preserved"]
    assert sample == before


def test_labels_accessed_only_after_selector_scoring():
    ranker = Ranker()
    class Guarded(dict):
        def __getitem__(self, key):
            if key == "expected":
                assert ranker.calls
            return super().__getitem__(key)
    assert evaluation.evaluate_case(ranker, Guarded(row()), "natural_evaluation_v2")["top1"]


def test_reject_budget_mismatch_and_mutated_candidate():
    with pytest.raises(ValueError, match="budget"):
        evaluation.evaluate_case(Ranker(too_long="imnida"), row(), "natural_evaluation_v2")
    sample = row()
    class Mutating(Ranker):
        def score_pairs(self, contexts, texts):
            sample["candidates"][0]["score"] += 1
            return super().score_pairs(contexts, texts)
    with pytest.raises(RuntimeError, match="objects"):
        evaluation.evaluate_case(Mutating(), sample, "natural_evaluation_v2")


def test_raw_policy_and_invalid_model_output_fail_closed(monkeypatch):
    with pytest.raises(ValueError, match="residual"):
        evaluation.evaluate_case(Ranker((0., float("nan"), 1.)), row(), "natural_evaluation_v2")
    select = evaluation.select_candidates
    def broken(*args, **kwargs):
        result = select(*args, **kwargs)
        if kwargs.get("raw"):
            result["result"] = "changed"
        return result
    monkeypatch.setattr(evaluation, "select_candidates", broken)
    with pytest.raises(RuntimeError, match="raw"):
        evaluation.evaluate_case(Ranker(), row(), "natural_evaluation_v2")


def summaries(gain=2, regression=False, actions=2):
    return {"natural_evaluation_v2": {"before_top1": 10, "top1": 10 + gain,
            "regressions": ["bad"] if regression else [], "top5_losses": [],
            "accepted_actions": actions},
            "challenge_preservation": {"before_top1": 3, "top1": 3,
            "regressions": [], "top5_losses": [], "accepted_actions": 0}}


def test_gate_needs_gain_actions_and_no_row_losses():
    assert evaluation.evaluation_gate(summaries())["passed"]
    assert not evaluation.evaluation_gate(summaries(gain=1))["passed"]
    assert not evaluation.evaluation_gate(summaries(regression=True))["passed"]
    assert evaluation.evaluation_gate(summaries(actions=0))["inconclusive"]
    values = summaries()
    values["challenge_preservation"]["regressions"] = ["regression"]
    assert not evaluation.evaluation_gate(values)["passed"]


def test_summarize_uses_row_outcomes():
    a = evaluation.evaluate_case(Ranker(), row(), "natural_evaluation_v2")
    b = dict(a, id="regression", before_top1=True, top1=False)
    result = evaluation.summarize([a, b])
    assert result["corrections"] == ["example"]
    assert result["regressions"] == ["regression"]
    assert result["accepted_actions"] == 2


def test_invalid_selector_identity_blocks_all_model_and_case_reads(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "verify_hashes", lambda _: None)
    def forbidden(*args, **kwargs):
        pytest.fail("selector identity must fail before any data or model load")
    monkeypatch.setattr(evaluation, "load_fresh_capture", forbidden)
    monkeypatch.setattr(evaluation, "read_runtime_evidence", forbidden)
    monkeypatch.setattr(evaluation.SentenceRanker, "load", forbidden)
    with pytest.raises(ValueError, match="frozen"):
        evaluation.run(SimpleNamespace(output=tmp_path / "out", selector_report_sha256="wrong"))


def capture_fixture(tmp_path, monkeypatch):
    from scripts.final_choice_capture import capture_case
    selected = [{"id": "example", "group_id": "document", "expected": ["입니다"],
                 "origin": "natural_eval_v2", "input": "imnida", "context": ""}]
    runtime = {"integrated_runtime_source_hashes": {"source": "a" * 64},
               "accepted_weight_hashes": {"model": "b" * 64}}
    normal = {"input": "imnida", "context": "", "engine": "neural", "mode": "batch",
        "raw": False, "version": evaluation.RUNTIME_VERSION, "candidates": candidates(),
        "result": "임니다", "ranker": "hybrid", "model_artifacts": runtime["accepted_weight_hashes"],
        "context_used": False, "external_context_used": False, "ranking_applied": True,
        "sentence_ranking_applied": False, "warnings": [], "engine_ms": 1.25}
    raw_candidate = deepcopy(candidates()[-1])
    raw_candidate["score"] = 0.
    raw_response = dict(normal, raw=True, result="imnida", candidates=[raw_candidate],
                        ranker=None, model_artifacts={}, ranking_applied=False)
    converter = SimpleNamespace(sentence_ranker=Ranker(),
        convert=lambda source, context, raw=False: deepcopy(raw_response if raw else normal))
    sample = capture_case(converter, selected[0], expected_runtime_version=evaluation.RUNTIME_VERSION,
                          expected_model_artifacts=runtime["accepted_weight_hashes"])
    cases_raw = (json.dumps(sample) + "\n").encode()
    selection_raw = (json.dumps(selected[0]) + "\n").encode()
    (tmp_path / "cases.jsonl").write_bytes(cases_raw)
    (tmp_path / "selection.jsonl").write_bytes(selection_raw)
    partition = {"eval_ids": ["example"], "eval_group_ids": ["document"]}
    manifest = {
        "scope": "fresh actual parent capture for balanced final-choice EVAL-v2",
        "role": "natural_evaluation_v2", "completed": True, "rows": 1, "expected_rows": 1,
        "partition_ordered_ids": partition["eval_ids"],
        "partition_ordered_group_ids": partition["eval_group_ids"],
        "evaluation_partition_sha256": evaluation.EVALUATION_PARTITION_SHA256,
        "profile": evaluation.CAPTURE_PROFILE, "runtime_version": evaluation.RUNTIME_VERSION,
        "runtime_source_hashes": runtime["integrated_runtime_source_hashes"],
        "model_artifact_hashes": runtime["accepted_weight_hashes"],
        "runtime_evidence_sha256": evaluation.RUNTIME_EVIDENCE_SHA,
        "selector_report_sha256": evaluation.SELECTOR_REPORT_SHA,
        "selector_checkpoint_sha256": evaluation.SELECTOR_WEIGHT_SHA,
        "source_hashes": {"fixture": "c" * 64}, "source_hashes_after": {"fixture": "c" * 64},
        "cases_sha256": evaluation.digest(cases_raw),
        "selection_sha256": evaluation.digest(selection_raw),
        "selection_sha256_before_inference": evaluation.digest(selection_raw),
        "selector_artifact_validated_without_inference": True,
        "baseline_accuracy_is_diagnostic_only": True,
        "diagnostics": {"top1": 0, "top5": 1, "label_aligned": 1,
                        "selector_eligible": 1, "all_candidates_can_score": 1},
        **{key: False for key in ("selector_inference_run", "training_run",
            "new_selector_inference_run", "development_data_read", "final_data_read",
            "deployed", "split_or_threshold_selected_from_capture_outcomes")},
    }
    monkeypatch.setattr(evaluation, "verify_hashes", lambda hashes: None)
    monkeypatch.setattr(evaluation, "read_evaluation_sources", lambda p: (selected, {}))
    # Exercise the real capture helper and strict row validator together; only
    # filesystem source verification is replaced by this synthetic fixture.
    return manifest, partition, runtime


def load_fixture(tmp_path, manifest, partition, runtime):
    raw = json.dumps(manifest).encode()
    (tmp_path / "manifest.json").write_bytes(raw)
    return evaluation.load_fresh_capture(tmp_path, evaluation.digest(raw), partition, runtime)


def test_fresh_capture_requires_exact_source_selection_and_diagnostics(tmp_path, monkeypatch):
    manifest, partition, runtime = capture_fixture(tmp_path, monkeypatch)
    assert len(load_fixture(tmp_path, manifest, partition, runtime)["rows"]) == 1
    manifest["diagnostics"]["top1"] = 1
    with pytest.raises(ValueError, match="diagnostics"):
        load_fixture(tmp_path, manifest, partition, runtime)
    manifest["diagnostics"]["top1"] = 0
    monkeypatch.setattr(evaluation, "read_evaluation_sources", lambda p: ([], {}))
    with pytest.raises(ValueError, match="selection"):
        load_fixture(tmp_path, manifest, partition, runtime)


@pytest.mark.parametrize("key,value", [("completed", False), ("role", "natural_fit"),
    ("selector_checkpoint_sha256", "changed"), ("source_hashes_after", {}),
    ("partition_ordered_ids", ["other"]), ("new_selector_inference_run", True)])
def test_fresh_capture_rejects_scope_lineage_and_order_changes(tmp_path, monkeypatch, key, value):
    manifest, partition, runtime = capture_fixture(tmp_path, monkeypatch)
    manifest[key] = value
    with pytest.raises(ValueError, match="identity or scope"):
        load_fixture(tmp_path, manifest, partition, runtime)


def test_fresh_capture_rejects_changed_case_bytes(tmp_path, monkeypatch):
    manifest, partition, runtime = capture_fixture(tmp_path, monkeypatch)
    (tmp_path / "cases.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="data hash"):
        load_fixture(tmp_path, manifest, partition, runtime)


@pytest.mark.parametrize("which", ["row", "selection", "both"])
def test_fresh_origin_adapter_rejects_wrong_origin_without_mutating_inputs(tmp_path, monkeypatch, which):
    manifest, partition, runtime = capture_fixture(tmp_path, monkeypatch)
    capture = load_fixture(tmp_path, manifest, partition, runtime)
    value = capture["rows"][0]
    selected = json.loads(capture["selection_raw"].splitlines()[0])
    before = deepcopy((value, selected))
    evaluation.validate_fresh_row(value, selected, runtime)
    assert (value, selected) == before
    if which in ("row", "both"):
        value["origin"] = "natural"
    if which in ("selection", "both"):
        selected["origin"] = "natural"
    with pytest.raises(ValueError, match="origin"):
        evaluation.validate_fresh_row(value, selected, runtime)
