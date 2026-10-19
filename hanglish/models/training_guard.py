"""Fail-closed validation for challenge-corpus training inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REQUIRED_FIELDS = {
    "id", "input", "context", "expected", "category", "family",
    "split", "provenance",
}
ROOT = Path(__file__).resolve().parents[2]
CANONICAL_TRAIN_LEXICON = ROOT / "data/challenge/train_lexicon.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_rows(path: Path, expected_split: str, expected_count: int) -> int:
    count = 0
    identifiers: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(row, dict) or not REQUIRED_FIELDS.issubset(row):
                raise ValueError(f"invalid challenge schema at {path}:{line_number}")
            scalar_fields = ("id", "input", "context", "category", "family", "split", "provenance")
            if any(not isinstance(row[field], str) for field in scalar_fields):
                raise ValueError(f"invalid challenge field type at {path}:{line_number}")
            expected = row["expected"]
            if not isinstance(expected, list) or not expected or any(
                    not isinstance(value, str) or not value for value in expected):
                raise ValueError(f"invalid expected values at {path}:{line_number}")
            if row["split"] != expected_split:
                raise ValueError(
                    f"{path}:{line_number} declares split {row['split']!r}, expected {expected_split!r}")
            if not row["id"] or row["id"] in identifiers:
                raise ValueError(f"missing or duplicate id at {path}:{line_number}")
            identifiers.add(row["id"])
            count += 1
    if count != expected_count:
        raise ValueError(f"{expected_split} row count {count} does not match manifest {expected_count}")
    return count


def validate_training_splits(train: str | Path, dev: str | Path,
                             manifest: str | Path, *,
                             canonical_train_lexicon: str | Path | None = None) -> dict:
    """Validate train/dev identity without opening the held-out test file."""
    train_path, dev_path, manifest_path = Path(train), Path(dev), Path(manifest)
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest_data.get("schema_version") != 2:
        raise ValueError("unsupported challenge manifest schema")
    hashes = manifest_data.get("sha256")
    counts = manifest_data.get("counts")
    if not isinstance(hashes, dict) or not isinstance(counts, dict):
        raise ValueError("challenge manifest lacks hashes or counts")
    required_hashes = {"train.jsonl", "dev.jsonl", "test.jsonl", "train_lexicon.json"}
    if not required_hashes.issubset(hashes):
        raise ValueError("challenge manifest lacks a required split hash")

    held_out_path = manifest_path.parent / "test.jsonl"
    resolved_test = held_out_path.resolve(strict=False)
    for role, path in (("train", train_path), ("dev", dev_path)):
        if path.resolve(strict=True) == resolved_test:
            raise ValueError(f"held-out test path cannot be used as {role}")

    train_hash, dev_hash = _sha256(train_path), _sha256(dev_path)
    held_out_hash = hashes["test.jsonl"]
    if train_hash == held_out_hash or dev_hash == held_out_hash:
        raise ValueError("held-out test content cannot be used for training or selection")
    if train_hash != hashes["train.jsonl"]:
        raise ValueError("train file hash does not match the manifest")
    if dev_hash != hashes["dev.jsonl"]:
        raise ValueError("dev file hash does not match the manifest")
    if train_hash == dev_hash or train_path.resolve() == dev_path.resolve():
        raise ValueError("train and dev must be distinct splits")

    consumed_lexicon = Path(canonical_train_lexicon or CANONICAL_TRAIN_LEXICON)
    manifest_lexicon = manifest_path.parent / "train_lexicon.json"
    if manifest_lexicon.resolve(strict=True) != consumed_lexicon.resolve(strict=True):
        raise ValueError("manifest train lexicon differs from CandidateGenerator's consumed lexicon")
    lexicon_hash = _sha256(consumed_lexicon)
    if lexicon_hash != hashes["train_lexicon.json"]:
        raise ValueError("train lexicon hash does not match the manifest")
    lexicon = json.loads(consumed_lexicon.read_text(encoding="utf-8"))
    provenance = lexicon.get("provenance")
    if (lexicon.get("schema_version") != 2 or not isinstance(provenance, dict)
            or provenance.get("source_split") != "train"
            or lexicon.get("train_sha256") != train_hash):
        raise ValueError("train lexicon provenance does not match the validated train split")

    train_count = _validate_rows(train_path, "train", int(counts["train"]))
    dev_count = _validate_rows(dev_path, "dev", int(counts["dev"]))
    return {
        "manifest_sha256": _sha256(manifest_path),
        "corpus_version": manifest_data.get("corpus_version"),
        "source_hashes": {str(train_path): train_hash, str(dev_path): dev_hash},
        "row_counts": {"train": train_count, "dev": dev_count},
        "candidate_lexicon": {"path": str(consumed_lexicon), "sha256": lexicon_hash,
                              "source_train_sha256": lexicon["train_sha256"]},
        "held_out_test_sha256": held_out_hash,
        "held_out_test_read": False,
    }


def _load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def validate_training_augmentation(augmentation: str | Path,
                                   augmentation_manifest: str | Path,
                                   held_out_attestation: str | Path,
                                   train: str | Path, dev: str | Path,
                                   frozen_manifest: str | Path,
                                   training_guard: dict) -> dict:
    """Validate train-derived context augmentation without opening held-out test rows."""
    augmentation_path = Path(augmentation)
    sidecar_path = Path(augmentation_manifest)
    attestation_path = Path(held_out_attestation)
    train_path, dev_path = Path(train), Path(dev)
    frozen_manifest_path = Path(frozen_manifest)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    frozen = json.loads(frozen_manifest_path.read_text(encoding="utf-8"))
    if sidecar.get("schema_version") != 1:
        raise ValueError("unsupported augmentation manifest schema")
    if sidecar.get("split") != "train":
        raise ValueError("augmentation manifest must declare split train")
    held_out_path = frozen_manifest_path.parent / "test.jsonl"
    if augmentation_path.resolve(strict=True) == held_out_path.resolve(strict=False):
        raise ValueError("held-out test path cannot be used as augmentation")
    augmentation_hash = _sha256(augmentation_path)
    if sidecar.get("augmentation_sha256") != augmentation_hash:
        raise ValueError("augmentation file hash does not match its manifest")
    source_hashes = training_guard.get("source_hashes", {})
    train_hash = source_hashes.get(str(train_path))
    if sidecar.get("source_train_sha256") != train_hash:
        raise ValueError("augmentation is not bound to the validated train split")
    frozen_manifest_hash = _sha256(frozen_manifest_path)
    count = sidecar.get("count")
    if not isinstance(count, int) or count <= 0:
        raise ValueError("augmentation manifest requires a positive row count")
    _validate_rows(augmentation_path, "train", count)

    held_out_hash = frozen.get("sha256", {}).get("test.jsonl")
    if not isinstance(held_out_hash, str) or augmentation_hash == held_out_hash:
        raise ValueError("held-out test content cannot be used as augmentation")

    expected_split_hashes = {
        name: frozen.get("sha256", {}).get(name) for name in ("dev.jsonl", "test.jsonl")
    }
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    if (attestation.get("schema_version") != 1
            or attestation.get("independent_verification") is not True
            or attestation.get("augmentation_sha256") != augmentation_hash
            or attestation.get("frozen_split_hashes") != expected_split_hashes
            or attestation.get("context_count") != count
            or attestation.get("exact_context_overlap_count") != {"dev": 0, "test": 0}
            or not isinstance(attestation.get("verifier_provenance"), str)
            or not attestation["verifier_provenance"].strip()):
        raise ValueError("augmentation lacks a valid independent held-out context attestation")
    provenance_prefix = sidecar.get("provenance_prefix")
    if not isinstance(provenance_prefix, str) or not provenance_prefix:
        raise ValueError("augmentation manifest requires a provenance prefix")

    train_rows = _load_rows(train_path)
    dev_contexts = {row["context"] for row in _load_rows(dev_path)}
    augmentation_rows = _load_rows(augmentation_path)
    train_ids = {row["id"] for row in train_rows}
    known_labels: dict[str, set[str]] = {}
    known_categories: dict[str, set[tuple[str, str]]] = {}
    known_contracts: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for row in train_rows:
        known_labels.setdefault(row["input"], set()).update(row["expected"])
        known_categories.setdefault(row["input"], set()).add((row["category"], row["family"]))
        for label in row["expected"]:
            known_contracts.setdefault((row["input"], label), set()).add(
                (row.get("boundary"), row.get("context_role")))
    contexts: set[str] = set()
    for row in augmentation_rows:
        if row["id"] in train_ids:
            raise ValueError("augmentation ids must be disjoint from original train ids")
        if not row["context"].strip() or row["context"] in contexts:
            raise ValueError("augmentation contexts must be nonempty and unique")
        contexts.add(row["context"])
        if row["context"] in dev_contexts:
            raise ValueError("augmentation context overlaps the frozen dev split")
        if not row["provenance"].startswith(provenance_prefix):
            raise ValueError("augmentation row has unrecognized provenance")
        labels = known_labels.get(row["input"])
        if labels is None or not set(row["expected"]).issubset(labels):
            raise ValueError("augmentation contains a label absent from original train")
        if (row["category"], row["family"]) not in known_categories[row["input"]]:
            raise ValueError("augmentation category or family differs from original train")
        if not isinstance(row.get("boundary"), str) or not isinstance(
                row.get("context_role"), str):
            raise ValueError("augmentation requires boundary and context_role")
        contract = (row["boundary"], row["context_role"])
        if any(contract not in known_contracts[(row["input"], label)]
               for label in row["expected"]):
            raise ValueError("augmentation boundary or context_role differs from original train label")
    return {
        "manifest_sha256": _sha256(sidecar_path),
        "augmentation_sha256": augmentation_hash,
        "row_count": count,
        "source_train_sha256": train_hash,
        "source_manifest_sha256": frozen_manifest_hash,
        "provenance_prefix": provenance_prefix,
        "held_out_context_attestation_sha256": _sha256(attestation_path),
        "held_out_context_attestation": attestation,
        "raw_input_overlap_with_train_allowed": True,
        "held_out_test_read": False,
    }


__all__ = ["validate_training_augmentation", "validate_training_splits"]
