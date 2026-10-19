from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from scripts import verify_numeric_composition_fit as verifier


MODEL = {"model.bin": "a" * 64}


def path(text, source="1in", score=1.0):
    return {"text": text, "score": score, "segments": [
        {"input_start": 0, "input_end": 1, "source": source[:1],
         "text": text[:1], "kind": "raw"},
        {"input_start": 1, "input_end": len(source), "source": source[1:],
         "text": text[1:], "kind": "numeric"},
    ]}


def response(row, candidates, *, raw=False, sentence=False):
    if raw:
        candidates = [{"text": row["input"], "score": 0.0, "segments": [{
            "input_start": 0, "input_end": len(row["input"]), "source": row["input"],
            "text": row["input"], "kind": "raw",
        }]}]
    return {"input": row["input"], "context": row["context"], "engine": "neural",
            "mode": "neural", "raw": raw, "version": verifier.RUNTIME_VERSION,
            "ranker": None if raw else "hybrid", "model_artifacts": {} if raw else MODEL,
            "context_used": bool(row["context"]), "external_context_used": False,
            "ranking_applied": not raw, "sentence_ranking_applied": sentence,
            "warnings": [], "result": candidates[0]["text"], "candidates": candidates}


def build_fixture(tmp_path, *, quality_pass=True):
    directory = tmp_path / "evaluation"
    directory.mkdir()
    rows, sources, captures = [], [], []
    correction_id = "r0"
    for index in range(verifier.ROWS):
        row_id = f"r{index}"
        source = {"id": row_id, "input": "1in", "context": f"context {index}"}
        parent_path = path("1인")
        new_path = path("1닌", score=0.5)
        parent = response(source, [deepcopy(parent_path)], sentence=False)
        child = response(source, [deepcopy(new_path), deepcopy(parent_path)], sentence=True)
        raw = response(source, [], raw=True)
        expected = ["1닌"] if quality_pass and index == 0 else ["없는정답"]
        row = {**source, "expected": expected, "result": parent["result"],
               "candidates": deepcopy(parent["candidates"]),
               "runtime_version": verifier.RUNTIME_VERSION,
               "response_flags": {name: parent.get(name) for name in verifier.FLAG_FIELDS}}
        trace = {"base": [deepcopy(parent_path)], "accepted_old_extras": [],
                 "accepted_new_extras": [deepcopy(new_path)],
                 "scoring_pool": [deepcopy(parent_path), deepcopy(new_path)],
                 "eligible_piece_count": 1}
        capture = {**source, "parent": parent, "child": child,
                   "parent_raw": deepcopy(raw), "child_raw": deepcopy(raw),
                   "parent_pool": [deepcopy(parent_path)], "parent_base": [deepcopy(parent_path)],
                   "trace": trace, "parent_sentence_scored": False,
                   "child_sentence_scored": True,
                   "child_score_inputs": ["1인", "1닌"],
                   "parent_ms": 1.0, "child_ms": 2.0 + index / 1000}
        rows.append(row); sources.append(source); captures.append(capture)
    write_jsonl(directory / "sources.jsonl", sources)
    write_jsonl(directory / "captures.jsonl", captures)
    counts = {"rows": verifier.ROWS, "before_top1": 0,
              "top1": 1 if quality_pass else 0, "before_top5": 0,
              "top5": 1 if quality_pass else 0, "before_pool": 0,
              "pool": 1 if quality_pass else 0, "eligible_rows": verifier.ROWS,
              "new_path_rows": verifier.ROWS}
    source_hashes = verifier._expected_source_hashes()
    gate = {"passed": quality_pass, "original_pool_preserved": True,
            "raw_preserved": True, "zero_first_choice_losses": True,
            "zero_top5_losses": True, "at_least_one_new_top5": quality_pass}
    timings = sorted(capture["child_ms"] for capture in captures[1:])
    report = {"schema": verifier.REPORT_SCHEMA, "completed": True, "deployed": False,
              "scope": "first512 natural FIT only", "runtime_version": verifier.RUNTIME_VERSION,
              "source_hashes_before": source_hashes, "source_hashes_after": source_hashes,
              "rows_captured_before_accuracy": verifier.ROWS,
              "captures_sha256": verifier.file_digest(directory / "captures.jsonl"),
              "sources_sha256": verifier.file_digest(directory / "sources.jsonl"),
              "counts": counts, "correction_ids": [correction_id] if quality_pass else [],
              "regression_ids": [], "top5_gain_ids": [correction_id] if quality_pass else [],
              "top5_loss_ids": [], "gate": gate, "first_child_ms": 2.0,
              "warm_child_p95_ms": timings[-26],
              "latency_scope": "engine calls after model load; not browser or cold model loading",
              "elapsed_seconds": 10.0, "created_at": datetime.now(timezone.utc).isoformat()}
    (directory / "report.json").write_bytes(verifier.canonical(report) + b"\n")
    return directory, rows


def write_jsonl(path, rows):
    path.write_bytes(b"".join(verifier.canonical(row) + b"\n" for row in rows))


def loader(rows):
    return lambda: {"rows": rows, "manifest": {"model_artifact_hashes": MODEL}}


def test_valid_complete_evidence_recomputes_passing_gate(tmp_path):
    directory, rows = build_fixture(tmp_path)
    result = verifier.verify_evaluation(directory, loader(rows))
    assert result["verification_passed"] is True
    assert result["quality_gate_passed"] is True
    assert result["downstream_evaluation_allowed"] is True
    assert result["promotion_allowed"] is False
    assert result["deployed"] is False
    assert result["metrics"]["counts"]["rows"] == 512
    assert result["metrics"]["correction_ids"] == ["r0"]


def test_valid_failed_quality_gate_is_still_verified_and_proof_is_immutable(tmp_path):
    directory, rows = build_fixture(tmp_path, quality_pass=False)
    payload = verifier.verify_evaluation(directory, loader(rows))
    assert payload["verification_passed"] is True
    assert payload["quality_gate_passed"] is False
    assert payload["downstream_evaluation_allowed"] is False
    assert payload["promotion_allowed"] is False
    proof = tmp_path / "proof.json"
    envelope = verifier.write_proof(proof, payload)
    assert envelope["digest"] == verifier.digest(verifier.canonical(payload))
    with pytest.raises(verifier.VerificationError, match="already exists"):
        verifier.write_proof(proof, payload)


@pytest.mark.parametrize("mutation", ["completed", "capture_hash", "source_hash", "row_count"])
def test_incomplete_evidence_is_rejected_before_fit_loader_reads_labels(tmp_path, mutation):
    directory, rows = build_fixture(tmp_path)
    report_path = directory / "report.json"
    report = json.loads(report_path.read_bytes())
    if mutation == "completed":
        report["completed"] = False
    elif mutation == "capture_hash":
        report["captures_sha256"] = "0" * 64
    elif mutation == "source_hash":
        report["source_hashes_after"] = {}
    else:
        captures = (directory / "captures.jsonl").read_bytes().splitlines()
        (directory / "captures.jsonl").write_bytes(b"\n".join(captures[:-1]) + b"\n")
        report["captures_sha256"] = verifier.file_digest(directory / "captures.jsonl")
    report_path.write_bytes(verifier.canonical(report) + b"\n")
    touched = []
    def forbidden_loader():
        touched.append(True)
        raise AssertionError("FIT labels were opened")
    with pytest.raises(verifier.VerificationError):
        verifier.verify_evaluation(directory, forbidden_loader)
    assert touched == []


def test_frozen_source_identity_candidate_and_flags_are_exact(tmp_path):
    directory, rows = build_fixture(tmp_path)
    for defect in ("input", "candidates", "flags", "model"):
        changed = deepcopy(rows)
        if defect == "input":
            changed[0]["input"] = "2in"
        elif defect == "candidates":
            changed[0]["candidates"][0]["score"] = 99.0
        elif defect == "flags":
            changed[0]["response_flags"]["ranker"] = "changed"
        else:
            bad_loader = lambda: {"rows": changed,
                                  "manifest": {"model_artifact_hashes": {"other": "b" * 64}}}
            with pytest.raises(verifier.VerificationError, match="model artifacts"):
                verifier.verify_evaluation(directory, bad_loader)
            continue
        with pytest.raises(verifier.VerificationError):
            verifier.verify_evaluation(directory, loader(changed))


def test_unrounded_or_nonfinite_candidate_score_is_rejected_before_labels(tmp_path):
    directory, _rows = build_fixture(tmp_path)
    captures_path = directory / "captures.jsonl"
    for bad in (1.00001, float("inf")):
        captures = [json.loads(line) for line in captures_path.read_bytes().splitlines()]
        captures[0]["parent"]["candidates"][0]["score"] = bad
        if bad == float("inf"):
            # JSON cannot represent infinity under the canonical contract; exercise the validator directly.
            with pytest.raises(verifier.VerificationError, match="score"):
                verifier._validate_segments(
                    captures[0]["parent"]["candidates"][0], "1in", rounded_score=True
                )
            continue
        write_jsonl(captures_path, captures)
        report_path = directory / "report.json"
        report = json.loads(report_path.read_bytes())
        report["captures_sha256"] = verifier.file_digest(captures_path)
        report_path.write_bytes(verifier.canonical(report) + b"\n")
        touched = []
        with pytest.raises(verifier.VerificationError, match="score"):
            verifier.verify_evaluation(directory, lambda: touched.append(True))
        assert touched == []


def test_instrumented_pool_retains_finite_unrounded_internal_scores():
    value = path("1인", score=1.00001)
    verifier._validate_path(value, "1in")
    with pytest.raises(verifier.VerificationError, match="score"):
        verifier._validate_segments(value, "1in", rounded_score=True)


@pytest.mark.parametrize("defect", ["raw", "base", "score_inputs", "old_pool"])
def test_capture_preservation_and_actual_score_inputs_are_verified_before_labels(tmp_path, defect):
    directory, rows = build_fixture(tmp_path)
    captures_path = directory / "captures.jsonl"
    captures = [json.loads(line) for line in captures_path.read_bytes().splitlines()]
    capture = captures[0]
    if defect == "raw":
        capture["child_raw"]["result"] = "changed"
    elif defect == "base":
        capture["trace"]["base"][0]["score"] = 7.0
    elif defect == "score_inputs":
        capture["child_score_inputs"] = ["wrong"]
    else:
        capture["parent_sentence_scored"] = True
        capture["trace"]["accepted_old_extras"] = [path("1닌", score=0.5)]
    write_jsonl(captures_path, captures)
    report_path = directory / "report.json"
    report = json.loads(report_path.read_bytes())
    report["captures_sha256"] = verifier.file_digest(captures_path)
    report_path.write_bytes(verifier.canonical(report) + b"\n")
    touched = []
    with pytest.raises(verifier.VerificationError):
        verifier.verify_evaluation(directory, lambda: touched.append(True))
    assert touched == []


@pytest.mark.parametrize("field", ["counts", "correction_ids", "regression_ids",
                                    "top5_gain_ids", "top5_loss_ids", "gate"])
def test_declared_quality_metrics_must_match_independent_recomputation(tmp_path, field):
    directory, rows = build_fixture(tmp_path)
    report_path = directory / "report.json"
    report = json.loads(report_path.read_bytes())
    report[field] = {} if isinstance(report[field], dict) else ["tampered"]
    report_path.write_bytes(verifier.canonical(report) + b"\n")
    with pytest.raises(verifier.VerificationError, match=field):
        verifier.verify_evaluation(directory, loader(rows))


def test_prior_correct_first_choice_and_top5_losses_reject_gate(tmp_path):
    directory, rows = build_fixture(tmp_path)
    rows[1]["expected"] = ["1인"]
    captures_path = directory / "captures.jsonl"
    captures = [json.loads(line) for line in captures_path.read_bytes().splitlines()]
    captures[1]["child"]["candidates"] = list(reversed(captures[1]["child"]["candidates"]))
    captures[1]["child"]["result"] = "1인"
    write_jsonl(captures_path, captures)
    report_path = directory / "report.json"
    report = json.loads(report_path.read_bytes())
    report["captures_sha256"] = verifier.file_digest(captures_path)
    report["counts"].update({"before_top1": 1, "top1": 2, "before_top5": 1, "top5": 2,
                             "before_pool": 1, "pool": 2})
    # Child still retains 1인 in top5, so this control remains preserved and the gate passes.
    report_path.write_bytes(verifier.canonical(report) + b"\n")
    result = verifier.verify_evaluation(directory, loader(rows))
    assert result["quality_gate_passed"] is True
    captures = [json.loads(line) for line in captures_path.read_bytes().splitlines()]
    captures[1]["child"]["candidates"] = [captures[1]["child"]["candidates"][0]]
    captures[1]["child"]["candidates"] = [path("1닌", score=0.5)]
    captures[1]["child"]["result"] = "1닌"
    write_jsonl(captures_path, captures)
    report = json.loads(report_path.read_bytes())
    report["captures_sha256"] = verifier.file_digest(captures_path)
    report["counts"].update({"top1": 1, "top5": 1, "pool": 2})
    report["regression_ids"] = ["r1"]
    report["top5_loss_ids"] = ["r1"]
    report["gate"].update({"passed": False, "zero_first_choice_losses": False,
                           "zero_top5_losses": False})
    report_path.write_bytes(verifier.canonical(report) + b"\n")
    result = verifier.verify_evaluation(directory, loader(rows))
    assert result["verification_passed"] is True
    assert result["quality_gate_passed"] is False
    assert result["downstream_evaluation_allowed"] is False
    assert result["promotion_allowed"] is False
