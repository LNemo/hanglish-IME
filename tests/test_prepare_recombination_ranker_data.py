from copy import deepcopy
import hashlib
import json

import pytest

from scripts import prepare_recombination_ranker_data as preparation


SOURCE = "abcdef"


def segment(start, end, text, kind="fixture"):
    return {"input_start": start, "input_end": end,
            "source": SOURCE[start:end], "text": text, "kind": kind}


def candidate(text, segments, score=0.0):
    return {"text": text, "score": score, "segments": deepcopy(segments)}


def capture(identity="one"):
    first = [segment(0, 2, "A"), segment(2, 4, "B"),
             segment(4, 6, "C")]
    second = [segment(0, 2, "X"), segment(2, 4, "Y"),
              segment(4, 6, "Z")]
    candidates = [candidate("ABC", first), candidate("XYZ", second)]
    return {"id": identity, "input": SOURCE, "result": "ABC",
            "candidates": candidates,
            "displayed_candidates": deepcopy(candidates)}


def label(identity="one", **changes):
    value = {"id": identity, "input": SOURCE, "expected": ["AYC"],
             "group_id": "group-" + identity,
             "source_ids": ["source-" + identity], "split": "train",
             "input_contract": "roman_sentence_only"}
    value.update(changes)
    return value


def test_prepare_record_is_source_only_stable_and_uses_saved_paths():
    row = capture()
    before = deepcopy(row)
    first = preparation.prepare_record(row)
    second = preparation.prepare_record(row)
    assert first == second
    assert row == before
    assert "expected" not in first
    assert first["request"] == {"input": SOURCE, "context": ""}
    assert first["parent_result"] == "ABC"
    assert all(item["candidate_id"] == item["path_id"]
               and item["origin"] == "saved_roman_train"
               and item["origins"] == ["saved_roman_train"]
               for item in first["pool"])
    assert "AYC" in {item["text"] for item in first["proposals"]}
    assert all(set(item["edge_ids"]) <=
               {edge["edge_id"] for edge in first["edge_inventory"]}
               for item in first["proposals"])


def test_labels_cannot_influence_source_only_proposals():
    row = capture()
    leaked = deepcopy(row)
    leaked["expected"] = ["secret target"]
    assert preparation.prepare_record(row) == preparation.prepare_record(leaked)


def test_join_conserves_originals_dedupes_and_never_injects_gold():
    saved = capture()
    saved["displayed_candidates"] = [saved["candidates"][1]]
    source = preparation.prepare_record(saved)
    source["pool"].append(deepcopy(source["pool"][0]))
    prepared = preparation.join_labels([source], [label(expected=["gold absent"])])[0]
    assert prepared["parent_candidates"] == ["XYZ"]
    assert prepared["original_candidates"] == ["ABC", "XYZ"]
    assert prepared["recombined_candidates"] == [
        *prepared["original_candidates"], *prepared["novel_candidates"]]
    assert len(prepared["recombined_candidates"]) == len(
        set(prepared["recombined_candidates"]))
    assert "gold absent" not in prepared["recombined_candidates"]


def test_missing_duplicate_or_changed_train_identity_is_rejected():
    source = preparation.prepare_record(capture())
    with pytest.raises(ValueError, match="missing"):
        preparation.join_labels([source], [])
    with pytest.raises(ValueError, match="duplicate"):
        preparation.join_labels([source], [label(), label()])
    with pytest.raises(ValueError, match="identity"):
        preparation.join_labels([source], [label(input="different")])


def metadata(identity, group, sources=(), source=None, expected=None):
    return {"id": identity, "group_id": group,
            "source_ids": list(sources), "input": source or identity,
            "expected": list(expected or [identity + " target"])}


def test_component_assignment_is_transitive_and_order_independent():
    rows = [metadata("a", "ga", ["s1"]),
            metadata("b", "gb", ["s1", "s2"]),
            metadata("c", "gc", ["s2"]), metadata("d", "gd")]
    first = preparation.assign_components(rows)
    assert first == preparation.assign_components(list(reversed(rows)))
    assert len({first[key]["component_id"] for key in "abc"}) == 1
    assert first["a"]["split"] == first["b"]["split"] == first["c"]["split"]


def test_normalized_input_and_any_target_join_components():
    rows = [metadata("a", "ga", source="a  b"),
            metadata("b", "gb", source="a b", expected=["target one"]),
            metadata("c", "gc", expected=["other", "target  one"])]
    assigned = preparation.assign_components(rows)
    assert len({value["component_id"] for value in assigned.values()}) == 1


def test_upstream_hash_and_report_contract_are_validated(tmp_path):
    capture_path = tmp_path / "capture.jsonl"
    capture_path.write_text("{}\n")
    digest = hashlib.sha256(capture_path.read_bytes()).hexdigest()
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({
        "completed": True, "input_contract": "roman_sentence_only",
        "split": "train", "rows": 1,
        "sha256": {"capture.jsonl": digest}}))
    preparation._validate_upstream(report_path, capture_path, digest, 1)
    capture_path.write_text("{\"changed\":true}\n")
    with pytest.raises(ValueError, match="hash"):
        preparation._validate_upstream(report_path, capture_path, digest, 1)


def test_component_metadata_types_and_duplicate_ids_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        preparation.assign_components([
            metadata("a", "g"), metadata("a", "h")])
    bad = metadata("a", "g")
    bad["source_ids"] = "not-a-list"
    with pytest.raises(ValueError, match="source_ids"):
        preparation.assign_components([bad])


def test_run_writes_closed_source_capture_then_compact_rows_without_overwrite(
        tmp_path):
    upstream = tmp_path / "upstream.jsonl"
    upstream.write_text(json.dumps(capture()) + "\n")
    upstream_sha = preparation.sha256_file(upstream)
    report = tmp_path / "report.json"
    report.write_text(json.dumps({
        "completed": True, "input_contract": "roman_sentence_only",
        "split": "train", "rows": 1,
        "sha256": {"capture.jsonl": upstream_sha}}))
    proof = tmp_path / "proof.json"
    proof.write_text(json.dumps({"completed": True, "passed": True,
                                 "rows": 1,
                                 "model_inference_run": False}))
    train = tmp_path / "train.jsonl"
    train.write_text(json.dumps(label()) + "\n")
    output = tmp_path / "output"
    manifest = preparation.run(
        output, capture_path=upstream, report_path=report, proof_path=proof,
        train_path=train, capture_sha=upstream_sha,
        report_sha=preparation.sha256_file(report),
        proof_sha=preparation.sha256_file(proof),
        train_sha=preparation.sha256_file(train), expected_rows=1)
    assert manifest["outputs"]["source-capture.jsonl"]["rows"] == 1
    assert manifest["outputs"]["prepared.jsonl"]["rows"] == 1
    closure = json.loads((output / "capture.complete.json").read_text())
    assert closure["source_only"] is True and closure["targets_joined"] is False
    prepared = json.loads((output / "prepared.jsonl").read_text())
    assert set(prepared) == {
        "id", "input", "expected", "group_id", "source_ids",
        "component_id", "split", "parent_result", "original_candidates",
        "parent_candidates", "recombined_candidates", "novel_candidates"}
    with pytest.raises(FileExistsError):
        preparation.run(
            output, capture_path=upstream, report_path=report,
            proof_path=proof, train_path=train, capture_sha=upstream_sha,
            report_sha=preparation.sha256_file(report),
            proof_sha=preparation.sha256_file(proof),
            train_sha=preparation.sha256_file(train), expected_rows=1)
