from copy import deepcopy
from datetime import datetime, timezone
import json

import pytest

from scripts import verify_numeric_composition_downstream as verifier


MODEL = {"model.bin": "a" * 64}


def segment_candidate(text, score=1.0):
    return {"text": text, "score": score, "segments": [{"input_start": 0,
        "input_end": 1, "source": "a", "text": text, "kind": "neural"}]}


def response(text, candidates=None, *, raw=False, sentence=False):
    if raw:
        candidates = [{"text": "a", "score": 0.0, "segments": [{"input_start": 0,
            "input_end": 1, "source": "a", "text": "a", "kind": "raw"}]}]
    elif candidates is None:
        candidates = [segment_candidate(text)]
    return {"input": "a", "context": "", "engine": "neural", "mode": "batch",
            "raw": raw, "version": verifier.RUNTIME_VERSION,
            "candidates": candidates, "result": candidates[0]["text"],
            "ranker": None if raw else "hybrid", "model_artifacts": {} if raw else MODEL,
            "context_used": False, "external_context_used": False,
            "ranking_applied": sentence, "sentence_ranking_applied": sentence,
            "warnings": []}


def capture(source, *, corrected=False, regressed=False):
    parent_text = "나" if corrected else "가"
    child_text = "나" if regressed else "가"
    parent_path, child_path = segment_candidate(parent_text), segment_candidate(child_text, .5)
    parent = response(parent_text, [deepcopy(parent_path)])
    child_candidates = [deepcopy(child_path)]
    if child_text != parent_text:
        child_candidates.append(deepcopy(parent_path))
    child = response(child_text, child_candidates, sentence=True)
    raw = response("a", raw=True)
    preserved = [deepcopy(parent_path)]
    scoring = preserved + ([] if child_text == parent_text else [deepcopy(child_path)])
    trace = {"reason": "sentence_scored", "activated": True,
             "production_activated": False, "base_scoreable": True,
             "eligible_piece_count": 1, "retained_combination_count": 1,
             "composition_bypassed": False, "base": deepcopy(preserved),
             "accepted_old_extras": [], "preserved_pool": deepcopy(preserved),
             "production_scoring_pool": [], "generated_new_extras": deepcopy(scoring[1:]),
             "accepted_new_extras": deepcopy(scoring[1:]), "scoring_pool": deepcopy(scoring)}
    return {**source, "parent": parent, "child": child, "parent_raw": deepcopy(raw),
            "child_raw": deepcopy(raw), "parent_pool": deepcopy(preserved),
            "parent_base": deepcopy(preserved), "trace": trace,
            "parent_sentence_scored": False, "child_sentence_scored": True,
            "child_score_inputs": [row["text"] for row in scoring],
            "parent_ms": 1.0, "child_ms": 2.0}


def write_jsonl(path, rows):
    path.write_bytes(b"".join(verifier.canonical(row) + b"\n" for row in rows))


def build_fixture(tmp_path, *, corrections=12, regressions=0, protected_failure=False):
    directory = tmp_path / "evaluation"
    directory.mkdir()
    sources, targets, captures, baselines = [], [], [], []
    fresh_group_ids = [f"g{i}" for i in range(verifier.FRESH_GROUPS)]
    preservation_ids = [f"p{i}" for i in range(verifier.PRESERVATION_ROWS)]
    for index in range(verifier.TOTAL_ROWS):
        fresh = index < verifier.FRESH_ROWS
        identity = f"f{index}" if fresh else preservation_ids[index-verifier.FRESH_ROWS]
        group = fresh_group_ids[index % verifier.FRESH_GROUPS] if fresh else f"pg{index}"
        source = {"id": identity, "group_id": group,
                  "origin": "fresh_natural" if fresh else "challenge", "input": "a", "context": ""}
        is_correction = fresh and index < corrections
        is_regression = fresh and corrections <= index < corrections + regressions
        value = capture(source, corrected=is_correction, regressed=is_regression)
        sources.append(source); captures.append(value)
        targets.append({"id": identity, "group_id": group,
                        "role": "fresh_natural" if fresh else "challenge_preservation",
                        "expected": ["나" if is_regression else "가"]})
        if not fresh:
            if protected_failure and index == verifier.FRESH_ROWS:
                targets[-1]["expected"] = ["다"]
            baselines.append({"id": identity, "result": value["parent"]["result"],
                "candidates": deepcopy(value["parent"]["candidates"]),
                "response_flags": {name: value["parent"].get(name)
                                   for name in verifier.FLAG_FIELDS},
                "runtime_version": verifier.RUNTIME_VERSION,
                "raw_bypass_response": deepcopy(value["parent_raw"])})
    for name, rows in (("sources.jsonl", sources), ("targets.jsonl", targets),
                       ("captures.jsonl", captures),
                       ("preservation_baselines.jsonl", baselines)):
        write_jsonl(directory / name, rows)
    source_hashes = {}
    for index in range(126):
        path = tmp_path / f"source-{index}"
        path.write_text(str(index))
        source_hashes[str(path)] = verifier.file_digest(path)
    metrics = verifier._score(targets, captures)
    observations = [{"stage": f"capture_{index+1}", "elapsed_seconds": index / 10,
                     "rss_peak_bytes": 1024} for index in range(verifier.TOTAL_ROWS)]
    observations.append({"stage": "after_model_release", "elapsed_seconds": 90.0,
                         "rss_peak_bytes": 1024})
    files = {name: verifier.file_digest(directory / name) for name in (
        "sources.jsonl", "targets.jsonl", "preservation_baselines.jsonl", "captures.jsonl")}
    report = {"schema": verifier.REPORT_SCHEMA, "completed": True,
              "terminal_status": "success", "deployed": False, "promotion_allowed": False,
              "profile": verifier.PROFILE,
              "target_access": {"fresh_target_bytes_accessed": True,
                                "preservation_target_bytes_accessed": True},
              "captures_closed": True, "models_released_before_target_join": True,
              "targets_joined": True, "model_loaded": False,
              "captured_rows": verifier.TOTAL_ROWS, "rows": verifier.TOTAL_ROWS,
              "fresh_rows": verifier.FRESH_ROWS, "fresh_groups": verifier.FRESH_GROUPS,
              "preservation_rows": verifier.PRESERVATION_ROWS,
              "runtime_version": verifier.RUNTIME_VERSION, "model_artifacts": MODEL,
              "historical_qwen_limitations": verifier.LIMITATIONS,
              "files": files, "source_hashes_before": source_hashes,
              "source_hashes_after": source_hashes, "resource_observations": observations,
              "terminal_resources": observations[-1], "model_load_seconds": 1.0,
              "metrics": metrics, "quality_gate_passed": metrics["gate"]["passed"],
              "downstream_evaluation_allowed": metrics["gate"]["passed"],
              "latency": {"first_parent_ms": 1.0, "first_child_ms": 2.0,
                          "warm_parent_p95_ms": 1.0, "warm_child_p95_ms": 2.0,
                          "scope": "engine calls after model load; not browser or cold model loading"},
              "elapsed_seconds": 91.0,
              "created_at": datetime.now(timezone.utc).isoformat()}
    (directory / "report.json").write_bytes(verifier.canonical(report) + b"\n")
    authority = {"selection": [{"id": f"f{i}",
                    "group_id": fresh_group_ids[i % verifier.FRESH_GROUPS]}
                    for i in range(verifier.FRESH_ROWS)],
                 "fresh_manifest": {},
                 "preservation_manifest": {"partition_ordered_ids": preservation_ids},
                 "model_artifacts": MODEL, "source_hashes": source_hashes,
                 "fit_proof_sha256": "a" * 64}
    return directory, authority, targets


def verify(directory, authority, targets, touched=None):
    def target_loader(_directory, _authority, _sources):
        if touched is not None:
            touched.append(True)
        return deepcopy(targets)
    return verifier.verify_evaluation(directory, target_loader,
                                      authority_loader=lambda: authority)


def test_current_authority_binds_128_or_more_sources_without_target_access():
    authority = verifier._validate_authority()
    assert len(authority["source_hashes"]) >= 126
    assert len(authority["selection"]) == verifier.FRESH_ROWS
    assert len(authority["preservation_manifest"]["partition_ordered_ids"]) == 376


def test_valid_complete_evidence_recomputes_passing_gate(tmp_path):
    directory, authority, targets = build_fixture(tmp_path)
    payload = verify(directory, authority, targets)
    assert payload["verification_passed"] is True
    assert payload["quality_gate_passed"] is True
    assert payload["downstream_evaluation_allowed"] is True
    assert payload["promotion_allowed"] is False and payload["deployed"] is False
    assert payload["fresh_targets_read"] is True
    assert payload["development_final_targets_read"] is False
    assert payload["metrics"]["bootstrap"]["net_gain"] == 12
    assert payload["metrics"]["bootstrap"]["resamples"] == 2000
    assert payload["metrics"]["bootstrap"]["seed"] == 20260915


def test_valid_quality_failure_is_verified_and_proof_is_immutable(tmp_path):
    directory, authority, targets = build_fixture(tmp_path, corrections=7)
    payload = verify(directory, authority, targets)
    assert payload["verification_passed"] is True
    assert payload["quality_gate_passed"] is False
    assert payload["downstream_evaluation_allowed"] is False
    proof = tmp_path / "proof.json"
    envelope = verifier.write_proof(proof, payload)
    assert envelope["digest"] == verifier.digest(verifier.canonical(payload))
    with pytest.raises(verifier.VerificationError, match="already exists"):
        verifier.write_proof(proof, payload)


@pytest.mark.parametrize("defect", ["completed", "captures", "source_hashes", "resources"])
def test_incomplete_evidence_rejects_before_target_loader(tmp_path, defect):
    directory, authority, targets = build_fixture(tmp_path)
    report_path = directory / "report.json"
    report = json.loads(report_path.read_bytes())
    if defect == "completed":
        report["completed"] = False
    elif defect == "captures":
        rows = (directory / "captures.jsonl").read_bytes().splitlines()[:-1]
        (directory / "captures.jsonl").write_bytes(b"\n".join(rows) + b"\n")
        report["files"]["captures.jsonl"] = verifier.file_digest(directory / "captures.jsonl")
    elif defect == "source_hashes":
        report["source_hashes_after"] = {}
    else:
        report["resource_observations"] = report["resource_observations"][:-1]
    report_path.write_bytes(verifier.canonical(report) + b"\n")
    touched = []
    with pytest.raises(verifier.VerificationError):
        verify(directory, authority, targets, touched)
    assert touched == []


@pytest.mark.parametrize("defect", ["raw", "base", "old_pool", "scorer", "nonfinite",
                                     "response_flag", "trace_flag", "none_score"])
def test_capture_invariants_reject_before_target_loader(tmp_path, defect):
    directory, authority, targets = build_fixture(tmp_path)
    captures_path = directory / "captures.jsonl"
    captures = [json.loads(line) for line in captures_path.read_bytes().splitlines()]
    if defect == "raw":
        captures[0]["child_raw"]["result"] = "changed"
    elif defect == "base":
        captures[0]["trace"]["base"][0]["score"] = 9.0
    elif defect == "old_pool":
        captures[0]["trace"]["preserved_pool"] = []
    elif defect == "scorer":
        captures[0]["child_score_inputs"] = ["wrong"]
    elif defect == "response_flag":
        captures[0]["child"]["sentence_ranking_applied"] = False
    elif defect == "trace_flag":
        captures[0]["trace"]["activated"] = False
    elif defect == "none_score":
        captures[0]["child"]["candidates"][0]["score"] = None
    else:
        captures[0]["parent_ms"] = float("inf")
        with pytest.raises(verifier.VerificationError):
            verifier.canonical(captures)
        captures[0]["parent_ms"] = -1.0
    write_jsonl(captures_path, captures)
    report_path = directory / "report.json"
    report = json.loads(report_path.read_bytes())
    report["files"]["captures.jsonl"] = verifier.file_digest(captures_path)
    report_path.write_bytes(verifier.canonical(report) + b"\n")
    touched = []
    with pytest.raises(verifier.VerificationError):
        verify(directory, authority, targets, touched)
    assert touched == []


def test_declared_metrics_must_match_independent_recomputation(tmp_path):
    directory, authority, targets = build_fixture(tmp_path)
    report_path = directory / "report.json"
    report = json.loads(report_path.read_bytes())
    report["metrics"]["bootstrap"]["net_gain"] = 999
    report_path.write_bytes(verifier.canonical(report) + b"\n")
    with pytest.raises(verifier.VerificationError, match="metrics"):
        verify(directory, authority, targets)


def test_protected_failure_is_a_valid_quality_failure(tmp_path):
    directory, authority, targets = build_fixture(tmp_path, protected_failure=True)
    report_path = directory / "report.json"
    report = json.loads(report_path.read_bytes())
    metrics = verifier._score(targets, read_jsonl(directory / "captures.jsonl"))
    report["metrics"] = metrics
    report["quality_gate_passed"] = False
    report["downstream_evaluation_allowed"] = False
    report_path.write_bytes(verifier.canonical(report) + b"\n")
    result = verify(directory, authority, targets)
    assert result["verification_passed"] is True
    assert result["quality_gate_passed"] is False
    assert result["metrics"]["gate"]["preservation_all_correct"] is False


def test_fresh_top5_loss_is_a_valid_quality_failure(tmp_path):
    directory, authority, targets = build_fixture(tmp_path)
    captures_path = directory / "captures.jsonl"
    captures = read_jsonl(captures_path)
    wrong = [segment_candidate(text, score) for text, score in
             zip(("나", "다", "라", "마", "바"), (.5, .4, .3, .2, .1))]
    captures[20]["child"] = response("나", deepcopy(wrong), sentence=True)
    captures[20]["trace"]["generated_new_extras"] = deepcopy(wrong)
    captures[20]["trace"]["accepted_new_extras"] = deepcopy(wrong)
    captures[20]["trace"]["scoring_pool"] = (
        deepcopy(captures[20]["trace"]["preserved_pool"]) + deepcopy(wrong))
    captures[20]["child_score_inputs"] = ["가", "나", "다", "라", "마", "바"]
    write_jsonl(captures_path, captures)
    report_path = directory / "report.json"
    report = json.loads(report_path.read_bytes())
    report["files"]["captures.jsonl"] = verifier.file_digest(captures_path)
    report["metrics"] = verifier._score(targets, captures)
    report["quality_gate_passed"] = False
    report["downstream_evaluation_allowed"] = False
    report_path.write_bytes(verifier.canonical(report) + b"\n")
    result = verify(directory, authority, targets)
    assert result["verification_passed"] is True
    assert result["quality_gate_passed"] is False
    assert result["metrics"]["roles"]["fresh_natural"]["top5_losses"] == ["f20"]


def read_jsonl(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def test_wrong_group_count_fails_fixed_bootstrap_gate(tmp_path):
    directory, authority, targets = build_fixture(tmp_path)
    targets[386]["group_id"] = targets[0]["group_id"]
    captures = read_jsonl(directory / "captures.jsonl")
    metrics = verifier._score(targets, captures)
    assert metrics["bootstrap"]["groups"] == verifier.FRESH_GROUPS - 1
    assert metrics["gate"]["passed"] is False


def test_bootstrap_requires_positive_frozen_lower_percentile():
    outcomes = [{"id": f"r{i}", "group_id": ("one" if i < 8 else f"g{i}"),
                 "before_correct": False, "after_correct": i < 8}
                for i in range(verifier.FRESH_ROWS)]
    result = verifier._bootstrap(outcomes)
    assert result["net_gain"] == 8
    assert result["lower_index"] == 49
    assert result["lower_2_5_percentile"] == 0
    assert result["gain_and_interval_passed"] is False


def test_original_raw_binding_removes_only_valid_engine_timing():
    closed = response("a", raw=True)
    original = {**deepcopy(closed), "engine_ms": .013}
    assert verifier._validate_original_raw_binding(original, closed) is True
    for mutation in ("missing_timing", "nonfinite_timing", "changed_raw"):
        changed = deepcopy(original)
        if mutation == "missing_timing":
            changed.pop("engine_ms")
        elif mutation == "nonfinite_timing":
            changed["engine_ms"] = float("inf")
        else:
            changed["result"] = "changed"
        with pytest.raises(verifier.VerificationError, match="raw|timing"):
            verifier._validate_original_raw_binding(changed, closed)


def test_cli_exit_codes_distinguish_pass_quality_fail_and_invalid(tmp_path, monkeypatch):
    evaluation, output = tmp_path / "evaluation", tmp_path / "proof.json"
    payload = {"quality_gate_passed": True, "downstream_evaluation_allowed": True,
               "promotion_allowed": False}
    monkeypatch.setattr(verifier, "verify_evaluation", lambda *_: payload)
    monkeypatch.setattr(verifier, "write_proof", lambda *_: {})
    assert verifier.main(["--evaluation", str(evaluation), "--output", str(output)]) == 0
    payload["quality_gate_passed"] = False
    payload["downstream_evaluation_allowed"] = False
    assert verifier.main(["--evaluation", str(evaluation), "--output", str(output)]) == 2
    def invalid(*_):
        raise verifier.VerificationError("invalid")
    monkeypatch.setattr(verifier, "verify_evaluation", invalid)
    assert verifier.main(["--evaluation", str(evaluation), "--output", str(output)]) == 1
