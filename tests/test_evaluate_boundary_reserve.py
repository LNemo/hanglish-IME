from copy import deepcopy
import hashlib
import json

import pytest

import scripts.evaluate_boundary_reserve as module


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode()


def candidate(source, text, kind="neural", score=1.0):
    return {"text": text, "score": score, "segments": [{
        "input_start": 0, "input_end": len(source), "source": source,
        "text": text, "kind": kind,
    }]}


def response(source, text):
    item = candidate(source, text)
    return {"candidates": [item], "result": text,
            "sentence_ranking_applied": False}


def unsupported_capture(index):
    source = {"id": f"id-{index}", "group_id": f"group-{index}",
              "input": f"source-{index}", "context": ""}
    parent = response(source["input"], "정답")
    raw = response(source["input"], source["input"])
    capture = {**source, "parent": parent, "child": deepcopy(parent),
        "parent_raw": raw, "child_raw": deepcopy(raw),
        "parent_pool": deepcopy(parent["candidates"]),
        "trace": {"activated": False, "reason": "production_branch_bypass",
            "preserved_pool": [], "generated_new_extras": [],
            "accepted_new_extras": [], "augmented_piece_count": 0,
            "derived_options": []},
        "parent_sentence_scored": False, "child_sentence_scored": False,
        "parent_score_inputs": None, "parent_scores": None,
        "child_score_inputs": None, "child_scores": None,
        "child_score_calls": [], "parent_ms": 1.0, "child_ms": 2.0}
    return source, capture


def test_verify_capture_replays_parent_pool_scores_and_source_policy():
    source = {"id": "id", "group_id": "group", "input": "gungnip",
              "context": "ctx"}
    parent_path = candidate("gungnip", "궁닙")
    extra = candidate("gungnip", "국립", "morphology")
    parent = {"candidates": module.legacy.display(
        [parent_path], [1.0], "gungnip"), "result": "궁닙",
        "sentence_ranking_applied": True}
    child = {"candidates": module.legacy.display(
        [parent_path, extra], [1.0, 2.0], "gungnip"), "result": "국립",
        "sentence_ranking_applied": True}
    provenance = {"source": "gungnip", "text": "국립",
        "left_lexeme": "국", "right_lexeme": "립", "left_key": "guk",
        "right_key": "rip", "boundary": "changed", "surface": "ngn",
        "suffix": None}
    raw = response("gungnip", "gungnip")
    capture = {**source, "parent": parent, "child": child,
        "parent_raw": raw, "child_raw": deepcopy(raw),
        "parent_pool": [parent_path],
        "trace": {"activated": True, "reason": "accepted_new_extras",
            "preserved_pool": [parent_path], "generated_new_extras": [extra],
            "accepted_new_extras": [extra], "augmented_piece_count": 1,
            "derived_options": [provenance]},
        "parent_sentence_scored": True, "child_sentence_scored": True,
        "parent_score_inputs": ["궁닙"], "parent_scores": [1.0],
        "child_score_inputs": ["궁닙", "국립"], "child_scores": [1.0, 2.0],
        "child_score_calls": [
            {"contexts": ["ctx"], "candidates": ["궁닙"], "scores": [1.0]},
            {"contexts": ["ctx", "ctx"], "candidates": ["궁닙", "국립"],
             "scores": [1.0, 2.0]}]}
    assert module.verify_capture(capture, source) is True
    broken = deepcopy(capture)
    broken["trace"]["derived_options"] = []
    with pytest.raises(ValueError, match="disallowed"):
        module.verify_capture(broken, source)


def test_targets_are_not_loaded_until_complete_capture_stream_is_closed(tmp_path):
    pairs = [unsupported_capture(index) for index in range(512)]
    sources = [pair[0] for pair in pairs]
    captures = [pair[1] for pair in pairs]
    first_source = sources[0]["input"]
    wrong = response(first_source, "오답")
    captures[0]["parent"] = wrong
    captures[0]["child"] = deepcopy(wrong)
    captures[0]["parent_pool"] = [
        candidate(first_source, f"오답{index}") for index in range(5)
    ] + [candidate(first_source, "정답")]
    path = tmp_path / "captures.jsonl"
    path.write_bytes(b"".join(canonical(row) + b"\n" for row in captures))
    stream = path.open("rb")
    calls = []

    def load_targets():
        assert stream.closed
        calls.append(True)
        return {source["id"]: ["정답"] for source in sources}

    with pytest.raises(ValueError, match="must close"):
        module.summarize_closed(path, sources, load_targets,
                                capture_stream=stream, models_released=True)
    assert calls == []
    stream.close()
    with pytest.raises(ValueError, match="model references"):
        module.summarize_closed(path, sources, load_targets,
                                capture_stream=stream, models_released=False)
    assert calls == []
    result = module.summarize_closed(path, sources, load_targets,
                                     capture_stream=stream, models_released=True)
    assert calls == [True]
    assert result["counts"] == {"rows": 512, "before_top1": 511,
        "top1": 511, "before_top5": 511, "top5": 511,
        "before_pool": 512, "pool": 512}
    assert result["gate"]["passed"] is False


def test_reserve_selection_is_validated_before_target_derived_sources(
        tmp_path, monkeypatch):
    directory = tmp_path / "selection"
    directory.mkdir()
    selected = []
    raw_rows = []
    index = 0
    for group_index in range(394):
        size = 2 if group_index < 118 else 1
        for _ in range(size):
            identity = f"id-{index}"
            group = f"group-{group_index}"
            selected.append({"id": identity, "group_id": group})
            raw_rows.append({"id": identity, "group_id": group,
                "split": "train", "text": f"문장{index}",
                "left_context": f"문맥{index}"})
            index += 1
    selection_raw = b"".join(canonical(row) + b"\n" for row in selected)
    raw_path = tmp_path / "raw.jsonl"
    raw_bytes = b"".join(canonical(row) + b"\n" for row in raw_rows)
    raw_path.write_bytes(raw_bytes)
    raw_sha = hashlib.sha256(raw_bytes).hexdigest()
    manifest = {"schema": "boundary-reserve-selection-v1", "rows": 512,
        "groups": 394, "selection_sha256": hashlib.sha256(selection_raw).hexdigest(),
        "raw_target_bytes_read": True, "raw_json_rows_parsed": True,
        "raw_target_fields_parsed": True, "selection_uses_target_text": False,
        "target_fields_used_for_selection": False, "target_text_copied": False,
        "selection_fields_used": ["id", "group_id"],
        "whole_groups_preserved": True, "model_inference_run": False,
        "training_run": False, "source_rows_modified": False,
        "source_hashes": {"raw.jsonl": raw_sha}}
    manifest_raw = json.dumps(manifest).encode()
    (directory / "selection.jsonl").write_bytes(selection_raw)
    (directory / "manifest.json").write_bytes(manifest_raw)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "PINNED_INPUTS", {"raw.jsonl": raw_sha})
    monkeypatch.setattr(module, "SELECTION_MANIFEST_SHA256",
                        hashlib.sha256(manifest_raw).hexdigest())
    monkeypatch.setattr(module, "SELECTION_SHA256",
                        hashlib.sha256(selection_raw).hexdigest())
    access = []
    sources, load_targets, lineage = module.load_reserve(
        directory, raw_path, target_access=access)
    assert len(sources) == 512 and len({row["group_id"] for row in sources}) == 394
    assert sources[0]["input"] == module.romanize_word("문장0")
    assert access == [{"reason": "target text is required to derive romanized inference input",
        "path": "raw.jsonl", "sha256": raw_sha, "before_raw_read": True}]
    raw_path.unlink()
    with pytest.raises(FileNotFoundError):
        load_targets()
    raw_path.write_bytes(raw_bytes)
    assert load_targets()["id-0"] == ["문장0"]
    assert lineage["target_bytes_needed_for_input_derivation"] is True


def test_fit_gate_is_required_before_reserve_open(tmp_path):
    path = tmp_path / "fit.json"
    good = {"schema": "boundary-provenance-runtime-fit-v1", "completed": True,
        "gate": {"passed": True}, "correction_ids": ["fix"],
        "regression_ids": [], "top5_loss_ids": [],
        "source_hashes_before": {}, "source_hashes_after": {}}
    path.write_text(json.dumps(good))
    report_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    proof = {"verified": True, "quality_gate_passed": True,
        "report_sha256": report_sha, "rows": 512,
        "raw_and_parent_pool_preserved": True,
        "all_extras_pass_source_policy": True}
    independent = tmp_path / "independent.json"
    independent.write_text(json.dumps(proof))
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(module, "FIT_REPORT_SHA256", report_sha)
    monkeypatch.setattr(module, "FIT_INDEPENDENT_SHA256",
                        hashlib.sha256(independent.read_bytes()).hexdigest())
    assert module.require_fit_gate(path, independent) == good
    good["regression_ids"] = ["loss"]
    path.write_text(json.dumps(good))
    report_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    proof["report_sha256"] = report_sha
    independent.write_text(json.dumps(proof))
    with pytest.raises(ValueError, match="regression-free"):
        module.FIT_REPORT_SHA256 = report_sha
        module.FIT_INDEPENDENT_SHA256 = hashlib.sha256(
            independent.read_bytes()).hexdigest()
        module.require_fit_gate(path, independent)
    monkeypatch.undo()


def test_normal_trace_is_snapshotted_before_raw_conversion_resets_it():
    normal_trace = {"activated": True, "accepted_new_extras": [{"text": "국립"}]}

    class Converter:
        def __init__(self, trace):
            self.boundary_trace = trace

        def convert(self, text, context, raw=False):
            if raw:
                self.boundary_trace = {"activated": False,
                                       "accepted_new_extras": []}
            return {"result": text, "raw": raw}

    parent = Converter({})
    child = Converter(normal_trace)
    trace, _, _ = module.capture_raw_pair(parent, child, "gungnip", "")
    assert trace == normal_trace
    assert trace is not normal_trace
    assert child.boundary_trace["accepted_new_extras"] == []
