import ast
from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts import verify_document_phonetic_shard as verifier


@pytest.fixture
def probe():
    with (verifier.ROOT / ".omx/artifacts/document-phonetic-choice-preflight-v1/capture.jsonl").open() as stream:
        return json.loads(next(stream))


def capture_record(probe):
    return {
        "id": probe["id"],
        "request": probe["request"],
        "baseline": {
            "kind": verifier.BASELINE_KIND,
            "version": verifier.PRODUCTION_VERSION,
            "candidates": probe["parent"]["candidates"],
            "result": probe["parent"]["result"],
            "sentence_ranking_applied": probe["parent"]["sentence_ranking_applied"],
        },
        "proposed": probe["proposed"],
        "base_paths": probe["base_paths"],
    }


def test_verifier_has_no_capture_producer_neural_or_tensor_imports():
    tree = ast.parse(Path(verifier.__file__).read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    forbidden = (
        "capture_document_phonetic_shard", "torch", "phonetic_likelihood",
        "production_phonetic_choice_converter", "preflight_document_phonetic_choice",
    )
    assert not any(item in name for name in imports for item in forbidden)


def test_actual_probe_replays_baseline_and_all_calibration_features(probe):
    record = capture_record(probe)
    counts = verifier.verify_record(
        {"id": probe["id"], "input": probe["request"]["input"]}, record)
    assert counts["paths"] == len(probe["base_paths"])
    assert counts["pairs"] == len(
        probe["proposed"]["phonetic_choice"]["score_records"])
    assert counts["supported_edges"] > 0


def test_baseline_candidate_or_kind_tampering_is_rejected(probe):
    record = capture_record(probe)
    record["baseline"]["kind"] = "actual_parent_inference"
    with pytest.raises(ValueError, match="baseline identity"):
        verifier.verify_record(
            {"id": probe["id"], "input": probe["request"]["input"]}, record)

    record = capture_record(probe)
    record["baseline"]["candidates"][0]["text"] = "손상"
    with pytest.raises(ValueError):
        verifier.verify_record(
            {"id": probe["id"], "input": probe["request"]["input"]}, record)


def test_adjusted_score_tampering_is_rejected(probe):
    record = capture_record(probe)
    record["proposed"]["phonetic_choice"]["rows"][0]["adjusted_score"] += 0.5
    with pytest.raises(ValueError, match="score differs"):
        verifier.verify_record(
            {"id": probe["id"], "input": probe["request"]["input"]}, record)


def test_record_requires_exact_source_order_and_context(probe):
    record = capture_record(probe)
    record["request"]["context"] = "forbidden"
    with pytest.raises(ValueError, match="source/schema sequence"):
        verifier.verify_record(
            {"id": probe["id"], "input": probe["request"]["input"]}, record)


def test_resolved_producer_fails_closed_until_frozen():
    with pytest.raises(ValueError, match="not frozen"):
        verifier.resolved_producer(verifier.UNRESOLVED)
    assert verifier.resolved_producer("a" * 64) == "a" * 64


def test_resource_reader_rejects_blank_or_empty_trace(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(ValueError, match="empty"):
        verifier.read_exact_jsonl_unbounded(empty)
    blank = tmp_path / "blank.jsonl"
    blank.write_text("{}\n\n")
    with pytest.raises(ValueError, match="blank"):
        verifier.read_exact_jsonl_unbounded(blank)
