#!/usr/bin/env python3
"""Build a deterministic raw natural-context window corpus from pinned KLUE train data."""

from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import re
import resource
import sys
import time
import unicodedata


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data/external/klue-v1.1-pilot"
DEFAULT_OUTPUT = ROOT / "data/natural_context/klue-v1.1-pilot-v2"
EXPECTED_REVISION = "3efd98708a40ff49251fddde35453f8fbb11f536"
EXPECTED_SOURCE_BYTES = 66_759_150
TRUSTED_ATTRIBUTION = {
    "license": "CC-BY-SA-4.0",
    "attribution": "KLUE Team, KLUE: Korean Language Understanding Evaluation (2021), https://arxiv.org/abs/2105.09680",
}
TRUSTED_SOURCE_FILES = {
    "klue-dp-v1.1_train.tsv": {
        "bytes": 6_229_730,
        "sha256": "0e549204078481f9abe35b5978a07376202ee69b037380b2a22f3eab93d1ca9d",
    },
    "klue-nli-v1.1_train.json": {
        "bytes": 12_576_683,
        "sha256": "8d32e7e4ff97804a23f277fee8740f1341ac82e1c0c7b34f1be21952e4813996",
    },
    "klue-mrc-v1.1_train.json": {
        "bytes": 47_952_737,
        "sha256": "c52c7b82a6015c09ea8be1fea49c912e73012d8b44d3653256d91c65d519c3df",
    },
}
MEMORY_LIMIT_BYTES = int(1.5 * 1024**3)
DISK_LIMIT_BYTES = 1024**3
MAX_WINDOWS = 10_000
SELECTION_SALT = "hanglish-natural-context-klue-v1.1-pilot-selection-v1"
SPLIT_SALT = "hanglish-natural-context-klue-v1.1-pilot-split-v1"
NORMALIZATION_VERSION = "nfc-whitespace-v1"
EXTRACTION_VERSION = "sentence-window-v1"
SPLIT_VERSION = "duplicate-component-90-10-v1"
HANGUL = re.compile(r"[가-힣]")
WHITESPACE = re.compile(r"\s+")
SENTENCE_BOUNDARY = re.compile(r"[.!?。！？]+[\"'”’」』)\]]*\s+")
DP_HEADER = re.compile(r"^##\s+(klue-dp-v1_train_([0-9]+)_([^\s\t]+))\t(.*)$")


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def guard_memory(stage: str) -> int:
    rss = rss_bytes()
    if rss > MEMORY_LIMIT_BYTES:
        raise MemoryError(f"RSS exceeded 1.5 GiB at {stage}: {rss / 1024**3:.3f} GiB")
    return rss


def normalize_text(value: str | None) -> str:
    return WHITESPACE.sub(" ", unicodedata.normalize("NFC", value or "")).strip()


def stable_hash(salt: str, value: str) -> str:
    return hashlib.sha256((salt + "\0" + value).encode()).hexdigest()


def sentence_spans(text: str):
    """Yield normalized sentence text and offsets without consuming future text."""
    normalized = normalize_text(text)
    start = 0
    sentence_index = 0
    for match in SENTENCE_BOUNDARY.finditer(normalized):
        end = match.end() - 1
        sentence = normalized[start:end].strip()
        if sentence:
            actual_start = start
            while actual_start < end and normalized[actual_start].isspace():
                actual_start += 1
            yield sentence_index, sentence, actual_start, end, normalized
            sentence_index += 1
        start = match.end()
    sentence = normalized[start:].strip()
    if sentence:
        actual_start = start
        while actual_start < len(normalized) and normalized[actual_start].isspace():
            actual_start += 1
        yield sentence_index, sentence, actual_start, len(normalized), normalized


def passage_windows(text: str, parent_group: str, provenance_base: dict, rejects: Counter):
    windows = []
    for sentence_index, sentence, start, end, normalized in sentence_spans(text):
        rejects["sentence_units"] += 1
        length = len(sentence)
        if length < 10:
            rejects["shorter_than_10"] += 1
            continue
        if length > 160:
            rejects["longer_than_160"] += 1
            continue
        if len(HANGUL.findall(sentence)) < 8:
            rejects["fewer_than_8_hangul"] += 1
            continue
        preceding_source = normalized[:start]
        expected_left_context = preceding_source[-240:].rstrip()
        left_context = normalized[max(0, start - 240):start].rstrip()
        if left_context != expected_left_context:
            raise AssertionError("left context is not the expected preceding source substring")
        rejects["left_context_precedence_checks"] += 1
        provenance = dict(provenance_base)
        provenance.update({
            "sentence_index": sentence_index,
            "normalized_start": start,
            "normalized_end": end,
        })
        windows.append({
            "text": sentence,
            "left_context": left_context,
            "parent_group": parent_group,
            "provenance": provenance,
        })
        rejects["retained_occurrences"] += 1
    return windows


def extract_mrc(document: dict, rejects: Counter):
    windows = []
    for article_index, article in enumerate(document["data"]):
        title = normalize_text(article["title"])
        source = normalize_text(article["source"])
        category = normalize_text(article.get("news_category", ""))
        parent = "mrc:" + stable_hash("mrc-parent-v1", source + "\0" + title)
        for paragraph_index, paragraph in enumerate(article["paragraphs"]):
            provenance = {
                "dataset": "mrc",
                "source_id": f"mrc:{article_index}:{paragraph_index}",
                "article_index": article_index,
                "paragraph_index": paragraph_index,
                "title": title,
                "publisher": source,
                "category": category,
            }
            windows.extend(passage_windows(
                paragraph["context"], parent, provenance, rejects
            ))
    return windows


def extract_nli(rows: list[dict], rejects: Counter):
    windows = []
    for row in rows:
        premise = normalize_text(row["premise"])
        parent = "nli:" + stable_hash("nli-parent-v1", premise)
        provenance = {
            "dataset": "nli",
            "source_id": row["guid"],
            "genre": normalize_text(row["genre"]),
            "category": normalize_text(row["genre"]),
        }
        windows.extend(passage_windows(row["premise"], parent, provenance, rejects))
    return windows


def extract_dp_text(content: str, rejects: Counter):
    windows = []
    for line in content.splitlines():
        match = DP_HEADER.fullmatch(line)
        if match is None:
            continue
        guid, numeric_id, genre, sentence = match.groups()
        parent = "dp:" + guid
        provenance = {
            "dataset": "dp",
            "source_id": guid,
            "dp_numeric_id": numeric_id,
            "genre": genre,
            "category": genre,
        }
        windows.extend(passage_windows(sentence, parent, provenance, rejects))
    return windows


class UnionFind:
    def __init__(self):
        self.parent = {}

    def add(self, value):
        self.parent.setdefault(value, value)

    def find(self, value):
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left, right):
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if right_root < left_root:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root


def canonicalize_occurrences(occurrences: list[dict]):
    """Union parent groups on duplicate text, then emit one canonical row per text."""
    union = UnionFind()
    first_group_by_text = {}
    for occurrence in occurrences:
        group = occurrence["parent_group"]
        union.add(group)
        previous = first_group_by_text.setdefault(occurrence["text"], group)
        union.union(previous, group)

    members = defaultdict(list)
    for group in union.parent:
        members[union.find(group)].append(group)
    component_id = {}
    for component in members.values():
        group_id = "group-" + stable_hash("component-v1", "\n".join(sorted(component)))[:20]
        for group in component:
            component_id[group] = group_id

    by_text = defaultdict(list)
    for occurrence in occurrences:
        by_text[occurrence["text"]].append(occurrence)
    rows = []
    for text, duplicates in by_text.items():
        ordered = sorted(duplicates, key=lambda item: (
            json.dumps(item["provenance"], ensure_ascii=False, sort_keys=True),
            item["left_context"], item["parent_group"],
        ))
        canonical = ordered[0]
        provenance = []
        seen = set()
        for occurrence in ordered:
            serialized = json.dumps(occurrence["provenance"], ensure_ascii=False, sort_keys=True)
            if serialized not in seen:
                seen.add(serialized)
                provenance.append(occurrence["provenance"])
        rows.append({
            "text": text,
            "left_context": canonical["left_context"],
            "group_id": component_id[canonical["parent_group"]],
            "provenance": provenance,
            "source_ids": sorted({
                item["dataset"] + ":" + item["source_id"] for item in provenance
            }),
            "duplicate_occurrences": len(ordered),
        })
    return rows


def select_and_split(rows: list[dict], limit=MAX_WINDOWS,
                     selection_salt=SELECTION_SALT, split_salt=SPLIT_SALT):
    ranked = sorted(rows, key=lambda row: (
        stable_hash(selection_salt, row["text"]), row["text"]
    ))[:limit]
    output = []
    for row in ranked:
        split_value = int(stable_hash(split_salt, row["group_id"])[:16], 16) % 10
        split = "dev" if split_value == 0 else "train"
        item = dict(row)
        item.update({
            "id": "nc-" + stable_hash("window-id-v1", row["text"])[:20],
            "split": split,
        })
        output.append(item)
    return output


def verify_sources(source_dir: Path):
    manifest_path = source_dir / "SOURCE.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("revision") != EXPECTED_REVISION:
        raise ValueError("KLUE source revision does not match the pinned revision")
    if any(manifest.get(key) != value for key, value in TRUSTED_ATTRIBUTION.items()):
        raise ValueError("KLUE attribution does not match the trusted metadata")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(TRUSTED_SOURCE_FILES):
        raise ValueError("KLUE source manifest file inventory is invalid")
    verified = {"SOURCE.json": {
        "bytes": len(manifest_bytes), "sha256": sha256_bytes(manifest_bytes)
    }}
    total = 0
    for name, trusted in sorted(TRUSTED_SOURCE_FILES.items()):
        metadata = files[name]
        path = source_dir / name
        size = path.stat().st_size
        digest = sha256_file(path)
        if size != trusted["bytes"] or digest != trusted["sha256"]:
            raise ValueError(f"KLUE source does not match trusted pin: {name}")
        if metadata.get("bytes") != trusted["bytes"] or metadata.get("sha256") != trusted["sha256"]:
            raise ValueError(f"KLUE source integrity mismatch: {name}")
        total += size
        verified[name] = {"bytes": size, "sha256": digest}
    if total != EXPECTED_SOURCE_BYTES or manifest.get("transferred_bytes_this_run") != total:
        raise ValueError("KLUE source byte total does not match the pinned pilot")
    return manifest, verified, total


def split_isolation_assertions(train, dev):
    train_groups = {row["group_id"] for row in train}
    dev_groups = {row["group_id"] for row in dev}
    train_texts = {row["text"] for row in train}
    dev_texts = {row["text"] for row in dev}
    train_source_ids = {source_id for row in train for source_id in row["source_ids"]}
    dev_source_ids = {source_id for row in dev for source_id in row["source_ids"]}
    embedded = sum(
        any(dev_text in row["left_context"] for row in train)
        for dev_text in dev_texts
    )
    assertions = {
        "group_overlap_count": len(train_groups & dev_groups),
        "exact_text_overlap_count": len(train_texts & dev_texts),
        "source_id_overlap_count": len(train_source_ids & dev_source_ids),
        "dev_text_embedded_in_train_left_context_count": embedded,
        "selected_texts_unique": len(train_texts | dev_texts) == len(train) + len(dev),
    }
    expected = {
        "group_overlap_count": 0,
        "exact_text_overlap_count": 0,
        "source_id_overlap_count": 0,
        "dev_text_embedded_in_train_left_context_count": 0,
        "selected_texts_unique": True,
    }
    if assertions != expected:
        raise AssertionError(assertions)
    return assertions


def jsonl_bytes(rows):
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode()


def build(source_dir: Path, output_dir: Path, limit=MAX_WINDOWS):
    output_paths = [output_dir / name for name in ("train.jsonl", "dev.jsonl", "manifest.json")]
    existing = [str(path) for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite natural-context outputs: " + ", ".join(existing))
    started = time.perf_counter()
    source_manifest, source_hashes, source_bytes = verify_sources(source_dir)
    peak_rss = guard_memory("source verification")

    rejects = Counter()
    mrc = json.loads((source_dir / "klue-mrc-v1.1_train.json").read_text())
    occurrences = extract_mrc(mrc, rejects)
    del mrc
    peak_rss = max(peak_rss, guard_memory("MRC extraction"))
    nli = json.loads((source_dir / "klue-nli-v1.1_train.json").read_text())
    occurrences.extend(extract_nli(nli, rejects))
    del nli
    peak_rss = max(peak_rss, guard_memory("NLI extraction"))
    dp_text = (source_dir / "klue-dp-v1.1_train.tsv").read_text()
    occurrences.extend(extract_dp_text(dp_text, rejects))
    del dp_text
    peak_rss = max(peak_rss, guard_memory("DP extraction"))

    unique_rows = canonicalize_occurrences(occurrences)
    occurrence_count = len(occurrences)
    parent_group_count = len({item["parent_group"] for item in occurrences})
    del occurrences
    selected = select_and_split(unique_rows, limit=limit)
    peak_rss = max(peak_rss, guard_memory("selection and split"))
    train = [row for row in selected if row["split"] == "train"]
    dev = [row for row in selected if row["split"] == "dev"]

    assertions = split_isolation_assertions(train, dev)
    assertions["left_context_precedence_checks"] = rejects["left_context_precedence_checks"]
    assertions["left_context_precedence_checks_match_retained_occurrences"] = (
        rejects["left_context_precedence_checks"] == occurrence_count
    )
    if not assertions["left_context_precedence_checks_match_retained_occurrences"]:
        raise AssertionError(assertions)

    train_bytes, dev_bytes = jsonl_bytes(train), jsonl_bytes(dev)
    output_size = len(train_bytes) + len(dev_bytes)
    if source_bytes + output_size > DISK_LIMIT_BYTES:
        raise OSError("source plus pilot outputs would exceed 1 GiB")

    canonical_datasets = Counter(row["provenance"][0]["dataset"] for row in selected)
    provenance_datasets = Counter(
        item["dataset"] for row in selected for item in row["provenance"]
    )
    provenance_categories = Counter(
        item["dataset"] + ":" + item.get("category", "")
        for row in selected for item in row["provenance"]
    )
    manifest = {
        "schema_version": 1,
        "name": "klue-v1.1-pilot-natural-context-v2",
        "description": "Deterministic raw natural-context sentence windows; not trainer-format data.",
        "scope": "KLUE v1.1 official MRC, NLI, and DP train sources only; context/premise/DP header sentence fields only.",
        "excluded_fields": ["MRC qas", "NLI hypothesis", "NLI labels", "DP dependency token and label rows"],
        "license": source_manifest["license"],
        "attribution": source_manifest["attribution"],
        "upstream_revision": source_manifest["revision"],
        "normalization_version": NORMALIZATION_VERSION,
        "extraction_version": EXTRACTION_VERSION,
        "split_version": SPLIT_VERSION,
        "selection": {
            "limit": limit,
            "salt": SELECTION_SALT,
            "policy": "ascending salted SHA-256 of globally deduplicated normalized text before splitting",
        },
        "split": {
            "salt": SPLIT_SALT,
            "policy": "duplicate-unioned parent-group hash modulo 10; 0 dev, 1..9 train",
            "target_ratio": {"train": 0.9, "dev": 0.1},
            "forced_counts": False,
        },
        "window_policy": {
            "sentence_boundary": "terminal .!?。！？ plus closing quote/bracket characters followed by whitespace",
            "unicode_characters": {"min": 10, "max": 160},
            "minimum_hangul_syllables": 8,
            "left_context": "up to 240 Unicode characters preceding the normalized source sentence only",
        },
        "counts": {
            "source_sentence_units": rejects["sentence_units"],
            "retained_occurrences": occurrence_count,
            "unique_eligible_texts": len(unique_rows),
            "parent_groups": parent_group_count,
            "selected": len(selected),
            "train": len(train),
            "dev": len(dev),
            "duplicate_occurrences_collapsed": occurrence_count - len(unique_rows),
        },
        "reject_counts": {
            key: rejects[key] for key in (
                "shorter_than_10", "longer_than_160", "fewer_than_8_hangul"
            )
        },
        "category_counts": {
            "canonical_dataset": dict(sorted(canonical_datasets.items())),
            "all_provenance_dataset": dict(sorted(provenance_datasets.items())),
            "all_provenance_category": dict(sorted(provenance_categories.items())),
        },
        "assertions": assertions,
        "source_files": source_hashes,
        "outputs": {
            "train.jsonl": {"bytes": len(train_bytes), "sha256": sha256_bytes(train_bytes), "rows": len(train)},
            "dev.jsonl": {"bytes": len(dev_bytes), "sha256": sha256_bytes(dev_bytes), "rows": len(dev)},
        },
        "resource_limits": {
            "memory_bytes": MEMORY_LIMIT_BYTES,
            "disk_bytes": DISK_LIMIT_BYTES,
            "source_bytes": source_bytes,
            "output_jsonl_bytes": output_size,
            "peak_rss_bytes": peak_rss,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "next_stage_api": {
            "input_files": ["train.jsonl", "dev.jsonl"],
            "fields": ["id", "text", "left_context", "group_id", "split", "provenance", "source_ids"],
            "contract": "Romanize and measure candidate coverage from text/left_context without adding expected labels; preserve group_id and split.",
        },
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    if source_bytes + output_size + len(manifest_bytes) > DISK_LIMIT_BYTES:
        raise OSError("source plus complete pilot outputs would exceed 1 GiB")

    output_dir.mkdir(parents=True, exist_ok=True)
    for path, content in zip(output_paths, (train_bytes, dev_bytes, manifest_bytes)):
        with path.open("xb") as stream:
            stream.write(content)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=MAX_WINDOWS)
    args = parser.parse_args()
    if not 1 <= args.limit <= MAX_WINDOWS:
        parser.error(f"--limit must be from 1 to {MAX_WINDOWS}")
    manifest = build(args.source_dir, args.output_dir, args.limit)
    print(json.dumps({
        "status": "complete",
        "output": str(args.output_dir),
        "counts": manifest["counts"],
        "reject_counts": manifest["reject_counts"],
        "assertions": manifest["assertions"],
        "peak_rss_gib": manifest["resource_limits"]["peak_rss_bytes"] / 1024**3,
        "elapsed_seconds": manifest["elapsed_seconds"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
