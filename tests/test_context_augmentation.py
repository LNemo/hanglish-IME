import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data/challenge/train.jsonl"
SIDECAR = ROOT / "data/challenge/context-augmentation/train.jsonl"
MANIFEST = SIDECAR.with_suffix(".manifest.json")


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _load_builder():
    path = ROOT / "scripts/build_context_augmentation.py"
    spec = importlib.util.spec_from_file_location("build_context_augmentation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _train_senses(rows):
    return {
        (row["input"], expected)
        for row in rows if row["category"] == "homophone_context"
        for expected in row["expected"]
    }


def _train_contracts(rows):
    contracts = {}
    for row in rows:
        if row["category"] != "homophone_context":
            continue
        for expected in row["expected"]:
            contracts.setdefault((row["input"], expected), set()).add(
                (row["boundary"], row["context_role"])
            )
    return contracts


def test_sidecar_covers_every_train_homophone_sense_with_new_contexts():
    train_rows = _rows(TRAIN)
    rows = _rows(SIDECAR)
    train_senses = _train_senses(train_rows)

    counts = Counter((row["input"], row["expected"][0]) for row in rows)
    assert set(counts) == train_senses
    assert set(counts.values()) == {12}
    assert len(rows) == 240
    assert len({row["input"] for row in rows}) == 9

    train_contexts = {row["context"] for row in train_rows}
    contexts = [row["context"] for row in rows]
    assert len(contexts) == len(set(contexts))
    assert not set(contexts) & train_contexts
    assert all(len(context) >= 20 and context.endswith(".") for context in contexts)


def test_sidecar_schema_labels_and_provenance_are_train_only():
    rows = _rows(SIDECAR)
    train_rows = _rows(TRAIN)
    train_senses = _train_senses(train_rows)
    train_contracts = _train_contracts(train_rows)
    required = {
        "id", "input", "context", "expected", "category", "family",
        "split", "provenance", "context_role", "boundary",
    }
    identifiers = set()
    for row in rows:
        assert required <= set(row)
        assert row["id"] not in identifiers
        identifiers.add(row["id"])
        assert row["split"] == "train"
        assert row["category"] == "homophone_context"
        assert row["provenance"] == "independently-authored-train-only-post-first-test-exposure"
        assert len(row["expected"]) == 1
        sense = (row["input"], row["expected"][0])
        assert sense in train_senses
        assert (row["boundary"], row["context_role"]) in train_contracts[sense]
        assert row["expected"][0] not in row["context"]

    stem_rows = [row for row in rows if row["input"] == "mat" and row["expected"] == ["맞"]]
    assert len(stem_rows) == 12
    assert all(row["boundary"] == "stem_fragment" for row in stem_rows)
    assert all("어간" in row["context"] for row in stem_rows)


def test_checked_in_sidecar_is_deterministic_and_bound_to_train(tmp_path):
    builder = _load_builder()
    generated = tmp_path / "train.jsonl"
    generated_manifest = builder.write_sidecar(TRAIN, generated)

    assert generated.read_bytes() == SIDECAR.read_bytes()
    checked_manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert generated_manifest == checked_manifest
    assert checked_manifest["augmentation_sha256"] == hashlib.sha256(SIDECAR.read_bytes()).hexdigest()
    assert checked_manifest["source_train_sha256"] == hashlib.sha256(TRAIN.read_bytes()).hexdigest()
    assert checked_manifest["source_scope"] == ["data/challenge/train.jsonl"]
    assert checked_manifest["held_out_disjointness_verification"] == "requires-independent-attestation"
    assert checked_manifest["frozen_corpus_modified"] is False
