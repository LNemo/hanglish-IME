import hashlib
import json

import pytest

from hanglish.models.training_guard import validate_training_augmentation, validate_training_splits


def _row(identifier, split):
    return {
        "id": identifier, "input": "gachi", "context": "문맥",
        "expected": ["같이"], "category": "homophone_context",
        "family": f"{split}:family", "split": split, "provenance": "authored",
        "boundary": "isolated_pronunciation", "context_role": "semantic_hint",
    }


def _bytes(row):
    return (json.dumps(row, ensure_ascii=False) + "\n").encode()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _fixture(tmp_path):
    train_data = _bytes(_row("train-1", "train"))
    dev_data = _bytes(_row("dev-1", "dev"))
    test_data = _bytes(_row("test-1", "test"))
    for name, data in (("train.jsonl", train_data), ("dev.jsonl", dev_data),
                       ("test.jsonl", test_data)):
        (tmp_path / name).write_bytes(data)
    train_lexicon = {
        "schema_version": 2, "provenance": {"source_split": "train"},
        "train_sha256": _sha(train_data), "entries": {},
    }
    train_lexicon_data = json.dumps(train_lexicon).encode()
    (tmp_path / "train_lexicon.json").write_bytes(train_lexicon_data)
    manifest = {
        "schema_version": 2, "corpus_version": "synthetic",
        "counts": {"train": 1, "dev": 1, "test": 1},
        "sha256": {"train.jsonl": _sha(train_data), "dev.jsonl": _sha(dev_data),
                   "test.jsonl": _sha(test_data),
                   "train_lexicon.json": _sha(train_lexicon_data)},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _validate_splits(tmp_path, manifest):
    return validate_training_splits(
        tmp_path / "train.jsonl", tmp_path / "dev.jsonl", manifest,
        canonical_train_lexicon=tmp_path / "train_lexicon.json")


def _rebind_train_lexicon(tmp_path, manifest, train_hash):
    lexicon_path = tmp_path / "train_lexicon.json"
    lexicon = json.loads(lexicon_path.read_text()); lexicon["train_sha256"] = train_hash
    content = json.dumps(lexicon).encode(); lexicon_path.write_bytes(content)
    data = json.loads(manifest.read_text()); data["sha256"]["train_lexicon.json"] = _sha(content)
    manifest.write_text(json.dumps(data))


def _augmentation_fixture(tmp_path):
    frozen_manifest = _fixture(tmp_path)
    guard = _validate_splits(tmp_path, frozen_manifest)
    row = _row("augmentation-1", "train")
    row.update({"context": "증강 전용 문맥", "provenance": "context-augmentation:v1"})
    data = _bytes(row)
    augmentation = tmp_path / "context-augmentation.jsonl"
    augmentation.write_bytes(data)
    frozen = json.loads(frozen_manifest.read_text())
    sidecar = {
        "schema_version": 1, "split": "train", "count": 1,
        "augmentation_sha256": _sha(data),
        "source_train_sha256": guard["source_hashes"][str(tmp_path / "train.jsonl")],
        "provenance_prefix": "context-augmentation:",
    }
    sidecar_path = tmp_path / "context-augmentation.manifest.json"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    attestation = {
        "schema_version": 1, "independent_verification": True,
        "augmentation_sha256": _sha(data), "context_count": 1,
        "frozen_split_hashes": {
            "dev.jsonl": frozen["sha256"]["dev.jsonl"],
            "test.jsonl": frozen["sha256"]["test.jsonl"],
        },
        "exact_context_overlap_count": {"dev": 0, "test": 0},
        "verifier_provenance": "independent synthetic held-out auditor",
    }
    attestation_path = tmp_path / "context-augmentation.attestation.json"
    attestation_path.write_text(json.dumps(attestation), encoding="utf-8")
    return frozen_manifest, guard, augmentation, sidecar_path, attestation_path


def _validate_augmentation(frozen_manifest, guard, augmentation, sidecar, attestation):
    return validate_training_augmentation(
        augmentation, sidecar, attestation, frozen_manifest.parent / "train.jsonl",
        frozen_manifest.parent / "dev.jsonl", frozen_manifest, guard)


def test_validates_manifest_hash_schema_and_split_without_test_rows(tmp_path):
    manifest = _fixture(tmp_path)
    result = _validate_splits(tmp_path, manifest)
    assert result["row_counts"] == {"train": 1, "dev": 1}
    assert result["held_out_test_read"] is False


def test_rejects_held_out_symlink_before_hashing_as_train(tmp_path):
    manifest = _fixture(tmp_path)
    alias = tmp_path / "renamed-training.jsonl"
    alias.symlink_to(tmp_path / "test.jsonl")
    with pytest.raises(ValueError, match="held-out test path"):
        validate_training_splits(alias, tmp_path / "dev.jsonl", manifest,
                                 canonical_train_lexicon=tmp_path / "train_lexicon.json")


def test_rejects_copied_held_out_content_by_manifest_hash(tmp_path):
    manifest = _fixture(tmp_path)
    disguised = tmp_path / "innocent-name.jsonl"
    disguised.write_bytes((tmp_path / "test.jsonl").read_bytes())
    with pytest.raises(ValueError, match="held-out test content"):
        validate_training_splits(disguised, tmp_path / "dev.jsonl", manifest,
                                 canonical_train_lexicon=tmp_path / "train_lexicon.json")


def test_rejects_declared_split_mismatch_even_with_matching_manifest_hash(tmp_path):
    manifest = _fixture(tmp_path)
    bad = _bytes(_row("wrong-1", "test"))
    (tmp_path / "train.jsonl").write_bytes(bad)
    data = json.loads(manifest.read_text())
    data["sha256"]["train.jsonl"] = _sha(bad)
    manifest.write_text(json.dumps(data), encoding="utf-8")
    _rebind_train_lexicon(tmp_path, manifest, _sha(bad))
    with pytest.raises(ValueError, match="declares split"):
        _validate_splits(tmp_path, manifest)


def test_rejects_schema_missing_expected_values(tmp_path):
    manifest = _fixture(tmp_path)
    row = _row("train-1", "train")
    row.pop("expected")
    bad = _bytes(row)
    (tmp_path / "train.jsonl").write_bytes(bad)
    data = json.loads(manifest.read_text())
    data["sha256"]["train.jsonl"] = _sha(bad)
    manifest.write_text(json.dumps(data), encoding="utf-8")
    _rebind_train_lexicon(tmp_path, manifest, _sha(bad))
    with pytest.raises(ValueError, match="schema"):
        _validate_splits(tmp_path, manifest)


def test_rejects_stale_train_lexicon_hash(tmp_path):
    manifest = _fixture(tmp_path)
    data = json.loads(manifest.read_text()); data["sha256"]["train_lexicon.json"] = "0" * 64
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="lexicon hash"):
        _validate_splits(tmp_path, manifest)


def test_rejects_train_lexicon_bound_to_different_train(tmp_path):
    manifest = _fixture(tmp_path)
    lexicon = json.loads((tmp_path / "train_lexicon.json").read_text())
    lexicon["train_sha256"] = "0" * 64
    content = json.dumps(lexicon).encode(); (tmp_path / "train_lexicon.json").write_bytes(content)
    data = json.loads(manifest.read_text()); data["sha256"]["train_lexicon.json"] = _sha(content)
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="provenance"):
        _validate_splits(tmp_path, manifest)


def test_rejects_alternate_manifest_lexicon_not_consumed_by_generator(tmp_path):
    manifest = _fixture(tmp_path)
    other = tmp_path / "canonical"; other.mkdir()
    (other / "train_lexicon.json").write_bytes((tmp_path / "train_lexicon.json").read_bytes())
    with pytest.raises(ValueError, match="consumed lexicon"):
        validate_training_splits(
            tmp_path / "train.jsonl", tmp_path / "dev.jsonl", manifest,
            canonical_train_lexicon=other / "train_lexicon.json")


def test_validates_train_bound_augmentation_without_test_rows(tmp_path):
    frozen, guard, augmentation, sidecar, attestation = _augmentation_fixture(tmp_path)
    result = _validate_augmentation(frozen, guard, augmentation, sidecar, attestation)
    assert result["row_count"] == 1
    assert result["raw_input_overlap_with_train_allowed"] is True
    assert result["held_out_test_read"] is False


def test_augmentation_rejects_label_absent_from_original_train(tmp_path):
    frozen, guard, augmentation, sidecar, attestation = _augmentation_fixture(tmp_path)
    row = _row("augmentation-1", "train")
    row.update({"context": "증강 전용 문맥", "expected": ["가치"],
                "provenance": "context-augmentation:v1"})
    data = _bytes(row); augmentation.write_bytes(data)
    manifest = json.loads(sidecar.read_text()); manifest["augmentation_sha256"] = _sha(data)
    sidecar.write_text(json.dumps(manifest))
    statement = json.loads(attestation.read_text()); statement["augmentation_sha256"] = _sha(data)
    attestation.write_text(json.dumps(statement))
    with pytest.raises(ValueError, match="label absent"):
        _validate_augmentation(frozen, guard, augmentation, sidecar, attestation)


def test_augmentation_rejects_context_equal_to_frozen_dev(tmp_path):
    frozen, guard, augmentation, sidecar, attestation = _augmentation_fixture(tmp_path)
    row = _row("augmentation-1", "train")
    row.update({"context": "문맥", "provenance": "context-augmentation:v1"})
    data = _bytes(row); augmentation.write_bytes(data)
    manifest = json.loads(sidecar.read_text()); manifest["augmentation_sha256"] = _sha(data)
    sidecar.write_text(json.dumps(manifest))
    statement = json.loads(attestation.read_text()); statement["augmentation_sha256"] = _sha(data)
    attestation.write_text(json.dumps(statement))
    with pytest.raises(ValueError, match="overlaps the frozen dev"):
        _validate_augmentation(frozen, guard, augmentation, sidecar, attestation)


def test_augmentation_requires_independent_attestation_bound_to_split_hashes(tmp_path):
    frozen, guard, augmentation, sidecar, attestation = _augmentation_fixture(tmp_path)
    statement = json.loads(attestation.read_text())
    statement["frozen_split_hashes"]["test.jsonl"] = "0" * 64
    attestation.write_text(json.dumps(statement))
    with pytest.raises(ValueError, match="independent held-out context attestation"):
        _validate_augmentation(frozen, guard, augmentation, sidecar, attestation)


def test_augmentation_rejects_boundary_contract_drift(tmp_path):
    frozen, guard, augmentation, sidecar, attestation = _augmentation_fixture(tmp_path)
    row = _row("augmentation-1", "train")
    row.update({"context": "증강 전용 문맥", "boundary": "single_token",
                "provenance": "context-augmentation:v1"})
    data = _bytes(row); augmentation.write_bytes(data)
    manifest = json.loads(sidecar.read_text()); manifest["augmentation_sha256"] = _sha(data)
    sidecar.write_text(json.dumps(manifest))
    statement = json.loads(attestation.read_text()); statement["augmentation_sha256"] = _sha(data)
    attestation.write_text(json.dumps(statement))
    with pytest.raises(ValueError, match="boundary or context_role"):
        _validate_augmentation(frozen, guard, augmentation, sidecar, attestation)


def test_augmentation_rejects_held_out_symlink_before_hashing(tmp_path):
    frozen, guard, _, sidecar, attestation = _augmentation_fixture(tmp_path)
    alias = tmp_path / "augmentation-alias.jsonl"
    alias.symlink_to(tmp_path / "test.jsonl")
    with pytest.raises(ValueError, match="held-out test path"):
        _validate_augmentation(frozen, guard, alias, sidecar, attestation)
