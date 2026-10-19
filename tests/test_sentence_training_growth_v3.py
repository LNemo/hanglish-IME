import hashlib
import json
from pathlib import Path

from scripts.build_sentence_training_extra import NearDuplicateIndex
from scripts.build_sentence_training_growth_v3 import (
    CORPUS, NEW_PREFIX, OUTPUT, QUOTAS, REVIEW_OUTPUT, V2, build,
    read_jsonl, source_tokens, target_tokens, valid_model_pair,
)
from scripts.sentence_data import read_split


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_pure_python_model_contract_covers_lengths_and_full_hangul_range():
    assert valid_model_pair("roman full sentence", "완전한 문장")
    assert source_tokens("a a") == 5
    assert target_tokens("가A") == 5
    assert not valid_model_pair("roman 가 sentence", "완전한 문장")
    assert not valid_model_pair("roman ᄀ sentence", "완전한 문장")
    assert not valid_model_pair("a " + "b" * 509, "짧은 문장")
    assert not valid_model_pair("roman full sentence", "가" * 171)


def test_published_v3_retains_v2_bytes_and_has_exact_quota_contract():
    manifest = json.loads((OUTPUT / "manifest.json").read_text())
    raw = (OUTPUT / "train.jsonl").read_bytes()
    retained = (V2 / "train.jsonl").read_bytes()
    assert raw[:len(retained)] == retained
    assert manifest["corpus_version"] == "sentence-train-extra-v3"
    assert manifest["counts"] == {"train": 2048}
    assert manifest["retained_rows"] == manifest["new_rows"] == 1024
    assert manifest["new_selection_strata"] == QUOTAS
    assert manifest["retained_prefix_bytes"] == len(retained)
    assert manifest["retained_train_sha256"] == sha(V2 / "train.jsonl")
    assert manifest["sha256"]["train.jsonl"] == sha(OUTPUT / "train.jsonl")
    assert manifest["sha256"]["train_lexicon.json"] == sha(OUTPUT / "train_lexicon.json")
    rows = read_split(OUTPUT, "train")
    assert len(rows) == 2048
    assert all(valid_model_pair(row["input"], row["expected"][0]) for row in rows)
    assert all(row["id"].startswith(NEW_PREFIX) for row in rows[1024:])
    assert all(row["source_row_ids"] for row in rows[1024:])


def test_new_rows_are_disjoint_and_near_duplicate_free_against_all_exclusions():
    base = [row for split in ("train", "dev", "test")
            for row in read_jsonl(CORPUS / f"{split}.jsonl")]
    retained = read_jsonl(V2 / "train.jsonl")
    additions = read_jsonl(OUTPUT / "train.jsonl")[1024:]
    excluded = base + retained
    documents = {row["group_id"] for row in excluded}
    source_ids = {item for row in excluded for item in row.get("source_ids", ())}
    source_rows = {item for row in excluded for item in row.get("source_row_ids", ())}
    inputs = {row["input"] for row in excluded}
    index = NearDuplicateIndex(target for row in excluded for target in row["expected"])
    for row in additions:
        assert row["group_id"] not in documents
        assert not set(row["source_ids"]) & source_ids
        assert not set(row["source_row_ids"]) & source_rows
        assert row["input"] not in inputs
        assert not index.contains(row["expected"][0])
        documents.add(row["group_id"])
        source_ids.update(row["source_ids"])
        source_rows.update(row["source_row_ids"])
        inputs.add(row["input"])
        index.add(row["expected"][0])


def test_train_lexicon_is_bound_to_combined_train_only_labels():
    rows = read_jsonl(OUTPUT / "train.jsonl")
    lexicon = json.loads((OUTPUT / "train_lexicon.json").read_text())
    assert lexicon["source_split"] == "train"
    assert lexicon["source_rows"] == 2048
    assert lexicon["retained_rows"] == lexicon["new_rows"] == 1024
    assert lexicon["train_sha256"] == sha(OUTPUT / "train.jsonl")
    assert lexicon["provenance"]["context_included"] is False
    assert lexicon["provenance"]["runtime_dictionary_injection"] is False
    assert sum(entry["count"] for entry in lexicon["entries"]) >= len(rows)


def test_hash12_review_is_label_independent_and_reproducible():
    additions = read_jsonl(OUTPUT / "train.jsonl")[1024:]
    expected = sorted(additions,
                      key=lambda row: hashlib.sha256(row["id"].encode()).hexdigest())[:12]
    review = json.loads(REVIEW_OUTPUT.read_text())
    assert review["selection"] == "first 12 new rows by SHA-256(id); label-independent"
    assert [row["id"] for row in review["rows"]] == [row["id"] for row in expected]
    assert review["train_sha256"] == sha(OUTPUT / "train.jsonl")
    assert review["labels_corrected"] is False
    assert all(row["source_row_ids"] for row in review["rows"])


def test_full_rebuild_matches_published_bytes(tmp_path):
    rebuilt = tmp_path / "sentence-train-extra-v3"
    build(rebuilt)
    for name in ("train.jsonl", "train_lexicon.json", "manifest.json", "ATTRIBUTION.md"):
        assert (rebuilt / name).read_bytes() == (OUTPUT / name).read_bytes()
