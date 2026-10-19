import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


PATH = Path(__file__).parents[1] / "scripts/freeze_document_reserve.py"
SPEC = importlib.util.spec_from_file_location("freeze_document_reserve", PATH)
freezer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = freezer
SPEC.loader.exec_module(freezer)


def dump_json(path: Path, value) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def dump_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8")


def fixture(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    rows = [
        {"id": "a1", "group_id": "a", "expected": ["ignored"]},
        {"id": "a2", "group_id": "a", "expected": ["ignored"]},
        {"id": "b1", "group_id": "b", "expected": ["ignored"]},
        {"id": "c1", "group_id": "c", "expected": ["ignored"]},
        {"id": "d1", "group_id": "d", "expected": ["ignored"]},
        {"id": "e1", "group_id": "e", "expected": ["ignored"]},
        {"id": "f1", "group_id": "f", "expected": ["ignored"]},
        {"id": "g1", "group_id": "g", "expected": ["ignored"]},
        {"id": "h1", "group_id": "h", "expected": ["ignored"]},
    ]
    paths = {name: tmp_path / name for name in
             ("raw", "final", "balanced", "finite", "long", "boundary")}
    paths["audits"] = tuple(tmp_path / f"audit-{n}" for n in range(3))
    dump_jsonl(paths["raw"], rows)
    dump_json(paths["final"], {"natural": {
        "reserve_ids": [r["id"] for r in rows],
        "reserve_group_ids": list("abcdefgh")}})
    dump_json(paths["balanced"], {
        "eval_ids": ["h1"], "eval_group_ids": ["h"],
        "remaining_ids": ["a1", "a2", "b1", "c1", "d1", "e1", "f1", "g1"],
        "remaining_group_ids": list("abcdefg")})
    dump_json(paths["finite"], {
        "eval_ids": ["g1"], "eval_group_ids": ["g"],
        "remaining_ids": ["a1", "a2", "b1", "c1", "d1", "e1", "f1"],
        "remaining_group_ids": list("abcdef")})
    dump_jsonl(paths["long"], [{"id": "d1", "group_id": "d"},
                                {"id": "e1", "group_id": "e"}])
    dump_jsonl(paths["boundary"], [{"id": "f1", "group_id": "f"}])
    for path in paths["audits"]:
        dump_jsonl(path, [{"id": "a1", "group_id": "a", "expected": "ignored"},
                          {"id": "b1", "group_id": "b", "expected": "ignored"},
                          {"id": "dev-1", "group_id": "dev", "expected": "ignored"}])
    all_paths = [value for value in paths.values() if isinstance(value, Path)]
    all_paths += list(paths["audits"])
    hashes = {path: freezer.sha256_file(path) for path in all_paths}
    counts = freezer.Counts(
        raw_rows=9, final_reserve_rows=9, balanced_eval_rows=1,
        balanced_remaining_rows=8, finite_eval_rows=1, finite_eval_groups=1,
        finite_remaining_rows=7, finite_remaining_groups=6,
        long_rows=2, long_groups=2, boundary_rows=1, boundary_groups=1,
        nominal_rows=4, nominal_groups=3, audit_overlap_ids=2,
        audit_overlap_groups=2, excluded_whole_group_rows=3,
        output_rows=1, output_groups=1)
    return paths, hashes, counts


def test_freezes_metadata_and_excludes_entire_exposed_group(tmp_path, monkeypatch):
    paths, hashes, counts = fixture(tmp_path)
    output = tmp_path / "output"
    monkeypatch.setattr(freezer, "ROOT", tmp_path)
    freezer.run(output, paths, hashes, counts)
    selection = [json.loads(line) for line in
                 (output / "selection.jsonl").read_text().splitlines()]
    assert selection == [{"group_id": "c", "id": "c1"}]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["schema"] == "document-reserve-v2"
    assert manifest["counts"]["rows"] == 1
    assert manifest["counts"]["excluded_whole_group_rows"] == 3
    assert manifest["contracts"]["model_inference"] is False
    assert manifest["contracts"]["training_allowed"] is False
    assert manifest["outputs"]["selection.jsonl"]["fields"] == ["id", "group_id"]


def test_rejects_partial_group(tmp_path):
    paths, hashes, counts = fixture(tmp_path)
    dump_jsonl(paths["long"], [{"id": "a1", "group_id": "a"},
                                {"id": "d1", "group_id": "d"}])
    hashes[paths["long"]] = freezer.sha256_file(paths["long"])
    with pytest.raises(ValueError, match="partial group"):
        freezer.derive_reserve(paths, hashes, counts)


def test_rejects_identity_group_mapping_mismatch(tmp_path):
    paths, hashes, counts = fixture(tmp_path)
    dump_jsonl(paths["boundary"], [{"id": "f1", "group_id": "wrong"}])
    hashes[paths["boundary"]] = freezer.sha256_file(paths["boundary"])
    with pytest.raises(ValueError, match="group mapping"):
        freezer.derive_reserve(paths, hashes, counts)


def test_rejects_swapped_row_groups_even_when_group_set_matches(tmp_path):
    paths, hashes, counts = fixture(tmp_path)
    dump_jsonl(paths["long"], [{"id": "d1", "group_id": "e"},
                                {"id": "e1", "group_id": "d"}])
    hashes[paths["long"]] = freezer.sha256_file(paths["long"])
    with pytest.raises(ValueError, match="id/group mapping"):
        freezer.derive_reserve(paths, hashes, counts)


def test_rejects_bad_audit_mapping_and_unknown_nominal_group(tmp_path):
    paths, hashes, counts = fixture(tmp_path)
    bad = [{"id": "a1", "group_id": "b"},
           {"id": "b1", "group_id": "a"},
           {"id": "dev-1", "group_id": "dev"}]
    dump_jsonl(paths["audits"][0], bad)
    hashes[paths["audits"][0]] = freezer.sha256_file(paths["audits"][0])
    with pytest.raises(ValueError, match="id/group mapping"):
        freezer.derive_reserve(paths, hashes, counts)

    paths, hashes, counts = fixture(tmp_path / "unknown")
    bad = [{"id": "a1", "group_id": "a"},
           {"id": "b1", "group_id": "b"},
           {"id": "dev-1", "group_id": "c"}]
    dump_jsonl(paths["audits"][0], bad)
    hashes[paths["audits"][0]] = freezer.sha256_file(paths["audits"][0])
    with pytest.raises(ValueError, match="unknown id in nominal group"):
        freezer.derive_reserve(paths, hashes, counts)


def test_rejects_tampered_source_before_parsing(tmp_path):
    paths, hashes, counts = fixture(tmp_path)
    paths["raw"].write_text(paths["raw"].read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="frozen source changed"):
        freezer.derive_reserve(paths, hashes, counts)
