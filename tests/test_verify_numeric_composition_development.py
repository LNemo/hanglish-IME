from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from scripts import verify_numeric_composition_development as verifier


MODEL = {f"weight-{index}": f"{index:064x}" for index in range(16)}


def candidate(text, score=1.0):
    return {"text": text, "score": score, "segments": [{"input_start": 0,
        "input_end": 1, "source": "a", "text": text, "kind": "neural"}]}


def response(text, candidates=None, *, raw=False, sentence=False):
    if raw:
        candidates = [{"text": "a", "score": 0.0, "segments": [{"input_start": 0,
            "input_end": 1, "source": "a", "text": "a", "kind": "raw"}]}]
    elif candidates is None:
        candidates = [candidate(text)]
    return {"input": "a", "context": "", "engine": "neural", "mode": "batch",
        "raw": raw, "version": verifier.RUNTIME_VERSION, "candidates": candidates,
        "result": candidates[0]["text"], "ranker": None if raw else "hybrid",
        "model_artifacts": {} if raw else MODEL, "context_used": False,
        "external_context_used": False, "ranking_applied": sentence,
        "sentence_ranking_applied": sentence, "warnings": []}


def make_capture(source, parent_correct, child_correct):
    parent_text, child_text = ("가" if parent_correct else "나",
                               "가" if child_correct else "나")
    base, child_path = candidate(parent_text), candidate(child_text, .5)
    sentence = parent_text != child_text
    scoring = [deepcopy(base)] + ([deepcopy(child_path)] if sentence else [])
    child_candidates = ([deepcopy(child_path), deepcopy(base)] if sentence
                        else [deepcopy(base)])
    raw = response("a", raw=True)
    return {**source, "parent": response(parent_text, [deepcopy(base)]),
        "child": response(child_text, child_candidates, sentence=sentence),
        "parent_raw": deepcopy(raw), "child_raw": deepcopy(raw),
        "parent_pool": [deepcopy(base)], "parent_base": [deepcopy(base)],
        "trace": {"reason": "sentence_scored" if sentence else "no_accepted_choice",
            "activated": sentence, "production_activated": False, "base_scoreable": True,
            "eligible_piece_count": int(sentence), "retained_combination_count": int(sentence),
            "composition_bypassed": False, "base": [deepcopy(base)],
            "accepted_old_extras": [], "preserved_pool": [deepcopy(base)],
            "production_scoring_pool": [],
            "generated_new_extras": [deepcopy(child_path)] if sentence else [],
            "accepted_new_extras": [deepcopy(child_path)] if sentence else [],
            "scoring_pool": scoring},
        "parent_sentence_scored": False, "child_sentence_scored": sentence,
        "child_score_inputs": [row["text"] for row in scoring] if sentence else None,
        "parent_ms": 1.0, "child_ms": 2.0}


def write_jsonl(path, rows):
    path.write_bytes(b"".join(verifier.canonical(row) + b"\n" for row in rows))


def build_fixture(tmp_path, *, natural=81, reasoning=35, original=140, selection=60):
    directory = tmp_path / "evaluation"; directory.mkdir()
    correct = {"natural_dev": natural, "original_dev": original,
               "selection_dev": selection, "reasoning_dev": reasoning}
    parent_correct = {"natural_dev": 76, "original_dev": 140,
                      "selection_dev": 60, "reasoning_dev": 34}
    sources, targets, captures, references = [], [], [], []
    index = 0
    for suite, count in verifier.SUITE_COUNTS.items():
        for offset in range(count):
            source = {"id": f"r{index}", "group_id": f"g{index}", "origin": suite,
                      "input": "a", "context": ""}
            capture = make_capture(source, offset < parent_correct[suite], offset < correct[suite])
            sources.append(source); captures.append(capture)
            targets.append({"id": source["id"], "role": suite, "expected": ["가"]})
            references.append({"id": source["id"], "suite": suite,
                "candidates": deepcopy(capture["parent"]["candidates"]),
                "sentence_pool": None,
                "flags": {name: capture["parent"].get(name) for name in
                          ("ranker", "context_used", "ranking_applied",
                           "sentence_ranking_applied", "warnings")}})
            index += 1
    for name, rows in (("sources.jsonl", sources), ("targets.jsonl", targets),
                       ("captures.jsonl", captures), ("parent_references.jsonl", references)):
        write_jsonl(directory / name, rows)
    source_hashes = {}
    for index in range(146):
        path = tmp_path / f"source-{index}"; path.write_text(str(index))
        source_hashes[str(path)] = verifier.file_digest(path)
    metrics = verifier._score(targets, captures)
    observations = [{"stage": f"development_capture_{index+1}",
                     "elapsed_seconds": index / 10, "rss_peak_bytes": 1024}
                    for index in range(verifier.TOTAL_ROWS)]
    observations.append({"stage": "after_development_model_release",
                         "elapsed_seconds": 40.0, "rss_peak_bytes": 1024})
    report = {"schema": verifier.REPORT_SCHEMA, "completed": True,
        "terminal_status": "success", "captures_closed": True,
        "models_released_before_target_join": True, "targets_joined": True,
        "model_loaded": False, "captured_rows": verifier.TOTAL_ROWS,
        "rows": verifier.TOTAL_ROWS, "suite_counts": verifier.SUITE_COUNTS,
        "profile": verifier.PROFILE, "runtime_version": verifier.RUNTIME_VERSION,
        "model_artifacts": MODEL,
        "target_access": {"development_target_bytes_accessed": True},
        "final_data_read": False, "promotion_allowed": False, "deployed": False,
        "files": {name: verifier.file_digest(directory / name) for name in
                  ("sources.jsonl", "targets.jsonl", "parent_references.jsonl", "captures.jsonl")},
        "source_hashes_before": source_hashes, "source_hashes_after": source_hashes,
        "resource_observations": observations, "terminal_resources": observations[-1],
        "metrics": metrics, "quality_gate_passed": metrics["gate"]["passed"],
        "final_evaluation_allowed": metrics["gate"]["passed"],
        "latency": {"first_parent_ms": 1.0, "first_child_ms": 2.0,
                    "warm_parent_p95_ms": 1.0, "warm_child_p95_ms": 2.0,
                    "scope": "engine calls after model load; not browser or cold model loading"},
        "elapsed_seconds": 41.0, "created_at": datetime.now(timezone.utc).isoformat()}
    (directory / "report.json").write_bytes(verifier.canonical(report) + b"\n")
    authority = {"weights": MODEL, "source_hashes": source_hashes,
                 "proof": {}, "integration": {}, "reference_report": {}}
    return directory, authority, targets


def verify(directory, authority, targets, touched=None):
    def load(_directory, _authority, _sources):
        if touched is not None:
            touched.append(True)
        return deepcopy(targets)
    return verifier.verify_evaluation(directory, load, lambda: authority)


def test_complete_dev_evidence_passes_exact_gate(tmp_path):
    directory, authority, targets = build_fixture(tmp_path)
    payload = verify(directory, authority, targets)
    assert payload["verification_passed"] is True
    assert payload["quality_gate_passed"] is True
    assert payload["final_evaluation_allowed"] is True
    assert payload["promotion_allowed"] is False and payload["deployed"] is False
    assert payload["metrics"]["suites"]["natural_dev"]["top1"] == 81
    assert payload["metrics"]["suites"]["reasoning_dev"]["top1"] == 35
    assert payload["development_targets_read"] is True
    assert payload["final_fixed_integration_targets_read"] is False


@pytest.mark.parametrize("values", [
    {"natural": 80}, {"original": 139}, {"selection": 59}, {"reasoning": 34}])
def test_each_quality_boundary_is_a_verified_rejection(tmp_path, values):
    directory, authority, targets = build_fixture(tmp_path, **values)
    payload = verify(directory, authority, targets)
    assert payload["verification_passed"] is True
    assert payload["quality_gate_passed"] is False
    assert payload["final_evaluation_allowed"] is False


@pytest.mark.parametrize("defect", ["completed", "capture_count", "source_hash",
                                     "resources", "reference"])
def test_incomplete_or_changed_source_evidence_rejects_before_targets(tmp_path, defect):
    directory, authority, targets = build_fixture(tmp_path)
    report_path = directory / "report.json"; report = json.loads(report_path.read_bytes())
    if defect == "completed":
        report["completed"] = False
    elif defect == "capture_count":
        captures = (directory / "captures.jsonl").read_bytes().splitlines()[:-1]
        (directory / "captures.jsonl").write_bytes(b"\n".join(captures) + b"\n")
        report["files"]["captures.jsonl"] = verifier.file_digest(directory / "captures.jsonl")
    elif defect == "source_hash":
        path = Path(next(iter(authority["source_hashes"])))
        path.write_text(path.read_text() + "changed")
    elif defect == "resources":
        report["resource_observations"] = report["resource_observations"][:-1]
    else:
        references = verifier.read_jsonl(directory / "parent_references.jsonl")
        references[0]["candidates"][0]["text"] = "changed"
        write_jsonl(directory / "parent_references.jsonl", references)
        report["files"]["parent_references.jsonl"] = verifier.file_digest(
            directory / "parent_references.jsonl")
    report_path.write_bytes(verifier.canonical(report) + b"\n")
    touched = []
    with pytest.raises(verifier.VerificationError):
        verify(directory, authority, targets, touched)
    assert touched == []


@pytest.mark.parametrize("defect", ["candidate", "raw", "base", "score", "scorer", "source",
                                     "response_field", "response_flag"])
def test_capture_corruption_rejects_before_targets(tmp_path, defect):
    directory, authority, targets = build_fixture(tmp_path)
    path = directory / "captures.jsonl"; captures = verifier.read_jsonl(path)
    if defect == "candidate":
        captures[0]["child"]["candidates"][0]["segments"][0]["text"] = "다"
    elif defect == "raw":
        captures[0]["child_raw"]["result"] = "changed"
    elif defect == "base":
        captures[0]["trace"]["base"][0]["score"] = 9.0
    elif defect == "score":
        captures[0]["child"]["candidates"][0]["score"] = None
    elif defect == "scorer":
        captures[76]["child_score_inputs"] = ["wrong"]
    elif defect == "source":
        captures[0]["input"] = "changed"
    elif defect == "response_field":
        captures[0]["child"]["unexpected"] = True
    else:
        captures[0]["child"]["context_used"] = "false"
    write_jsonl(path, captures)
    report_path = directory / "report.json"; report = json.loads(report_path.read_bytes())
    report["files"]["captures.jsonl"] = verifier.file_digest(path)
    report_path.write_bytes(verifier.canonical(report) + b"\n")
    touched = []
    with pytest.raises(verifier.VerificationError):
        verify(directory, authority, targets, touched)
    assert touched == []


def test_target_identity_and_answers_are_checked_after_preflight(tmp_path):
    directory, authority, targets = build_fixture(tmp_path)
    targets[0]["expected"] = []
    with pytest.raises(verifier.VerificationError, match="target"):
        verify(directory, authority, targets)


def test_declared_metrics_must_match_independent_recomputation(tmp_path):
    directory, authority, targets = build_fixture(tmp_path)
    report_path = directory / "report.json"; report = json.loads(report_path.read_bytes())
    report["metrics"]["suites"]["natural_dev"]["top1"] = 999
    report_path.write_bytes(verifier.canonical(report) + b"\n")
    with pytest.raises(verifier.VerificationError, match="metrics"):
        verify(directory, authority, targets)


def test_coherent_saved_reference_and_parent_capture_change_rejects_original_binding(tmp_path):
    directory, authority, targets = build_fixture(tmp_path)
    sources = verifier.read_jsonl(directory / "sources.jsonl")
    captures_path = directory / "captures.jsonl"
    references_path = directory / "parent_references.jsonl"
    captures, references = verifier.read_jsonl(captures_path), verifier.read_jsonl(references_path)
    original_reference = deepcopy(references[0])
    captures[0] = make_capture(sources[0], parent_correct=False, child_correct=True)
    references[0] = {"id": sources[0]["id"], "suite": sources[0]["origin"],
        "candidates": deepcopy(captures[0]["parent"]["candidates"]),
        "sentence_pool": None,
        "flags": {name: captures[0]["parent"].get(name) for name in
                  ("ranker", "context_used", "ranking_applied",
                   "sentence_ranking_applied", "warnings")}}
    write_jsonl(captures_path, captures); write_jsonl(references_path, references)
    report_path = directory / "report.json"; report = json.loads(report_path.read_bytes())
    report["files"]["captures.jsonl"] = verifier.file_digest(captures_path)
    report["files"]["parent_references.jsonl"] = verifier.file_digest(references_path)
    report_path.write_bytes(verifier.canonical(report) + b"\n")
    touched = []
    def target_loader(_directory, _authority, _sources):
        touched.append(True)
        verifier._validate_original_reference_binding(original_reference, references[0])
        return targets
    with pytest.raises(verifier.VerificationError, match="accepted DEV reference"):
        verifier.verify_evaluation(directory, target_loader, lambda: authority)
    assert touched == [True]


def test_top5_loss_rejects_quality_without_invalidating_evidence(tmp_path):
    directory, authority, targets = build_fixture(tmp_path)
    path = directory / "captures.jsonl"; captures = verifier.read_jsonl(path)
    wrong = [candidate(text, score) for text, score in
             zip(("나", "다", "라", "마", "바"), (.5, .4, .3, .2, .1))]
    captures[0]["child"] = response("나", deepcopy(wrong), sentence=True)
    captures[0]["trace"]["activated"] = True
    captures[0]["trace"]["reason"] = "sentence_scored"
    captures[0]["trace"]["generated_new_extras"] = deepcopy(wrong)
    captures[0]["trace"]["accepted_new_extras"] = deepcopy(wrong)
    captures[0]["trace"]["scoring_pool"] = (
        deepcopy(captures[0]["trace"]["preserved_pool"]) + deepcopy(wrong))
    captures[0]["child_sentence_scored"] = True
    captures[0]["child_score_inputs"] = ["가", "나", "다", "라", "마", "바"]
    write_jsonl(path, captures)
    report_path = directory / "report.json"; report = json.loads(report_path.read_bytes())
    report["files"]["captures.jsonl"] = verifier.file_digest(path)
    report["metrics"] = verifier._score(targets, captures)
    report["quality_gate_passed"] = False; report["final_evaluation_allowed"] = False
    report_path.write_bytes(verifier.canonical(report) + b"\n")
    payload = verify(directory, authority, targets)
    assert payload["verification_passed"] is True
    assert payload["quality_gate_passed"] is False
    assert payload["metrics"]["suites"]["natural_dev"]["top5_losses"] == ["r0"]


def test_proof_is_immutable_and_cli_exit_codes_are_distinct(tmp_path, monkeypatch):
    directory, authority, targets = build_fixture(tmp_path)
    payload = verify(directory, authority, targets)
    output = tmp_path / "proof.json"; verifier.write_proof(output, payload)
    with pytest.raises(verifier.VerificationError, match="already exists"):
        verifier.write_proof(output, payload)
    result = {"quality_gate_passed": True, "final_evaluation_allowed": True}
    monkeypatch.setattr(verifier, "verify_evaluation", lambda *_: result)
    monkeypatch.setattr(verifier, "write_proof", lambda *_: {})
    assert verifier.main(["--evaluation", str(directory), "--output", str(output)]) == 0
    result["quality_gate_passed"] = False; result["final_evaluation_allowed"] = False
    assert verifier.main(["--evaluation", str(directory), "--output", str(output)]) == 2
    monkeypatch.setattr(verifier, "verify_evaluation",
                        lambda *_: (_ for _ in ()).throw(verifier.VerificationError("bad")))
    assert verifier.main(["--evaluation", str(directory), "--output", str(output)]) == 1
