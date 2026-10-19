from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts import evaluate_numeric_composition_development as module


MODEL = {"model": "a" * 64}


def candidate(text):
    return {"text": text, "score": 1.0, "segments": [{"input_start": 0,
        "input_end": 1, "source": "a", "text": text, "kind": "neural"}]}


def rows(reasoning_correct=35):
    sources, targets, references = [], [], []
    index = 0
    for suite, count in module.SUITE_COUNTS.items():
        for offset in range(count):
            identity = f"r{index}"
            source = {"id": identity, "group_id": f"g{index}", "origin": suite,
                      "input": "a", "context": ""}
            expected = ["가"]
            parent_correct = (suite in ("original_dev", "selection_dev")
                              or suite == "natural_dev" and offset < 80
                              or suite == "reasoning_dev" and offset < 34)
            child_correct = (suite in ("original_dev", "selection_dev")
                             or suite == "natural_dev" and offset < 81
                             or suite == "reasoning_dev" and offset < reasoning_correct)
            parent_text, child_text = ("가" if parent_correct else "나",
                                       "가" if child_correct else "나")
            path = candidate(parent_text)
            capture = {**source,
                "parent": {"result": parent_text, "candidates": [candidate(parent_text)]},
                "child": {"result": child_text, "candidates": [candidate(child_text),
                    candidate("가")] if not child_correct else [candidate("가")]},
                "parent_raw": {"result": "a"}, "child_raw": {"result": "a"},
                "parent_pool": [path], "parent_base": [path],
                "trace": {"base": [path], "accepted_old_extras": [],
                    "accepted_new_extras": [], "scoring_pool": [candidate(child_text)]},
                "parent_sentence_scored": False, "child_sentence_scored": False,
                "child_score_inputs": None, "parent_ms": 1.0, "child_ms": 2.0}
            sources.append(source)
            targets.append({"id": identity, "role": suite, "expected": expected})
            references.append({"id": identity, "capture": capture})
            index += 1
    return sources, targets, references


def entry():
    path = module.ROOT / ".omx/artifacts/numeric-composition-downstream-v1-proof.json"
    return {"weights": MODEL, "source_hashes": {
        module._path(path): module.downstream.file_digest(path)}}


def fake_capture(_bundle, source, reference, _weights):
    return deepcopy(reference["capture"])


def test_fixed_interface_and_gate_constants():
    assert module.SCHEMA == "numeric-composition-development-v1"
    assert module.SUITE_COUNTS == {"natural_dev": 128, "original_dev": 140,
                                   "selection_dev": 60, "reasoning_dev": 40}
    assert module.TOTAL_ROWS == 368
    assert callable(module.validate_downstream_entry)
    assert callable(module.load_and_split_development)
    assert callable(module.capture_development)
    assert callable(module.score_development)


def test_pipeline_closes_all_captures_releases_then_joins_targets(tmp_path, monkeypatch):
    sources, targets, references = rows()
    events = []
    def loader(_entry, access):
        access["development_target_bytes_accessed"] = True
        return sources, targets, references, entry()["source_hashes"]
    def release(bundle):
        events.append("release")
        bundle.clear()
    original = module.downstream.read_jsonl
    def reopen(path):
        events.append("join")
        assert events[-2] == "release"
        return original(path)
    monkeypatch.setattr(module.downstream, "read_jsonl", reopen)
    report = module.execute_pipeline(tmp_path / "out", entry(), loader,
        lambda _: {"loaded": True}, fake_capture, release, lambda *_: {"ok": True})
    assert report["completed"] is True
    assert report["captured_rows"] == module.TOTAL_ROWS
    assert report["captures_closed"] is True
    assert report["models_released_before_target_join"] is True
    assert report["targets_joined"] is True
    assert events == ["release", "join"]
    assert len(report["resource_observations"]) == module.TOTAL_ROWS + 1


def test_only_source_and_target_free_reference_reach_capture(tmp_path):
    sources, targets, references = rows()
    seen = []
    def capture(bundle, source, reference, weights):
        assert set(source) == {"id", "group_id", "origin", "input", "context"}
        assert "expected" not in source and "expected" not in reference
        seen.append(source["id"])
        return fake_capture(bundle, source, reference, weights)
    report = module.execute_pipeline(tmp_path / "out", entry(),
        lambda _entry, access: (sources, targets, references, entry()["source_hashes"]),
        lambda _: {}, capture, lambda bundle: None, lambda *_: {})
    assert report["completed"] and len(seen) == module.TOTAL_ROWS


def test_synthetic_dev_loader_splits_targets_and_accepted_parent_references(
        tmp_path, monkeypatch):
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    suites, reference_rows, source_hashes = [], [], {}
    index = 0
    for suite, count in module.SUITE_COUNTS.items():
        corpus, identity = tmp_path / f"{suite}.jsonl", tmp_path / f"{suite}.json"
        suite_rows = []
        for _ in range(count):
            row = {"id": f"r{index}", "input": "a", "context": "", "expected": ["가"]}
            suite_rows.append(row)
            reference_rows.append({**row, "suite": suite, "candidates": [candidate("가")],
                "sentence_pool": None, "flags": {}})
            index += 1
        corpus.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                  for row in suite_rows))
        identity.write_text("{}\n")
        suites.append((suite, corpus, identity))
    cases = reference_dir / "cases.jsonl"
    cases.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                             for row in reference_rows))
    original_path = module._path
    monkeypatch.setattr(module, "_path", lambda path: (
        original_path(path) if module.ROOT in Path(path).resolve().parents
        else str(Path(path).resolve())))
    for _, corpus, identity in suites:
        source_hashes[module._path(corpus)] = module.downstream.file_digest(corpus)
        source_hashes[module._path(identity)] = module.downstream.file_digest(identity)
    monkeypatch.setattr(module, "REFERENCE", reference_dir)
    monkeypatch.setattr(module, "REFERENCE_CASES_SHA256",
                        module.downstream.file_digest(cases))
    monkeypatch.setattr(module, "SUITES", tuple(suites))
    access = {}
    sources, targets, references, hashes = module.load_and_split_development(
        {"reference_report": {"source_hashes": source_hashes}}, access)
    assert access["development_target_bytes_accessed"] is True
    assert len(sources) == len(targets) == len(references) == module.TOTAL_ROWS
    assert "expected" not in sources[0] and "expected" not in references[0]
    assert targets[0]["expected"] == ["가"] and hashes


def test_exact_development_gate_passes_and_never_promotes():
    sources, targets, references = rows()
    result = module.score_development(
        sources, targets, [row["capture"] for row in references])
    assert result["suites"]["natural_dev"]["top1"] == 81
    assert result["suites"]["original_dev"]["top1"] == 140
    assert result["suites"]["selection_dev"]["top1"] == 60
    assert result["suites"]["reasoning_dev"]["top1"] == 35
    assert result["gate"]["passed"] is True
    assert result["gate"]["promotion_allowed"] is False


@pytest.mark.parametrize("mutation", ["natural", "original", "selection", "reasoning", "top5"])
def test_each_fixed_quality_boundary_fails(mutation):
    sources, targets, references = rows(reasoning_correct=34 if mutation == "reasoning" else 35)
    captures = [row["capture"] for row in references]
    if mutation == "natural":
        captures[80]["child"] = {"result": "나", "candidates": [candidate("나"), candidate("가")]}
    elif mutation == "original":
        index = module.SUITE_COUNTS["natural_dev"]
        captures[index]["child"] = {"result": "나", "candidates": [candidate("나"), candidate("가")]}
    elif mutation == "selection":
        index = module.SUITE_COUNTS["natural_dev"] + module.SUITE_COUNTS["original_dev"]
        captures[index]["child"] = {"result": "나", "candidates": [candidate("나"), candidate("가")]}
    elif mutation == "top5":
        captures[0]["parent"] = {"result": "가", "candidates": [candidate("가")]}
        captures[0]["child"] = {"result": "나", "candidates": [candidate("나")]}
    assert module.score_development(sources, targets, captures)["gate"]["passed"] is False


def test_parent_reference_requires_exact_candidates_pool_and_flags():
    capture = {"id": "one", "parent": {"candidates": [candidate("가")],
        "ranker": "hybrid", "context_used": False, "ranking_applied": False,
        "sentence_ranking_applied": False, "warnings": []},
        "parent_pool": [candidate("가")], "parent_sentence_scored": False}
    reference = {"id": "one", "candidates": [candidate("가")],
        "sentence_pool": None, "flags": {"ranker": "hybrid", "context_used": False,
            "ranking_applied": False, "sentence_ranking_applied": False, "warnings": []}}
    module.validate_parent_reference(capture, reference)
    changed = deepcopy(reference)
    changed["candidates"][0]["text"] = "나"
    with pytest.raises(ValueError, match="candidates"):
        module.validate_parent_reference(capture, changed)


def test_partial_failure_releases_model_and_never_joins_targets(tmp_path):
    sources, targets, references = rows()
    released = []
    def fail(*_args):
        raise RuntimeError("capture failed")
    report = module.execute_pipeline(tmp_path / "out", entry(),
        lambda _entry, access: (sources, targets, references, entry()["source_hashes"]),
        lambda _: {"loaded": True}, fail, lambda bundle: released.append(bundle),
        lambda *_: {})
    assert report["completed"] is False
    assert report["targets_joined"] is False
    assert report["promotion_allowed"] is False
    assert report["models_released_on_failure"] is True
    assert released == [{"loaded": True}]
    assert json.loads((tmp_path / "out/report.json").read_bytes()) == report


def test_invalid_proof_hash_rejects_before_any_dev_loader(monkeypatch):
    original = module.downstream.file_digest
    monkeypatch.setattr(module.downstream, "file_digest",
        lambda path: "0" * 64 if Path(path) == module.DOWNSTREAM_PROOF else original(path))
    with pytest.raises(ValueError, match="proof changed"):
        module.validate_downstream_entry()
