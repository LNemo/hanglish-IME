"""Build a 2,048-row TRAIN-only supplement retaining v2 byte-for-byte."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hanglish.core.phonology import romanize_word
from scripts.build_sentence_challenge import merge_sentences
from scripts.build_sentence_training_extra import (
    NearDuplicateIndex, REVIEWED_QUARANTINE, eligible_strata, row_features,
    select_growth,
)


NATURAL = ROOT / "data/natural_context/runtime-collision-source-v5"
CORPUS = ROOT / "data/challenge/sentence-v3"
V2 = ROOT / "data/challenge/sentence-train-extra-v2"
REVIEW = ROOT / ".omx/artifacts/runtime-collision-source-review-v2.json"
OUTPUT = ROOT / "data/challenge/sentence-train-extra-v3"
REVIEW_OUTPUT = ROOT / ".omx/artifacts/sentence-train-extra-v3-hash12-review.json"
QUOTAS = {"structural_oov": 256, "mixed_oov": 384, "plain_oov": 384}
NEW_PREFIX = "s3xv3-train-"
V3_REVIEWED_QUARANTINE = {
    "rcc-800fbaf01cd8cf9df18d": "source joins sentence boundary as '있다.부품'; preserve source and exclude",
    "rcc-9236f0e1adbc68225bbe": "same merged source with missing sentence-boundary spacing",
    "rcc-f4f0f780ee43b6ed9ad4": "same merged source with missing sentence-boundary spacing",
    "rcc-33a48ff0e807a196422c": "source uses nonstandard spelling '트랜드'; preserve source and exclude",
    "rcc-76a6b1d1afca06fc954e": "same merged source with nonstandard spelling '트랜드'",
    "rcc-1839ec83f5bedafec934": "source has nonstandard dependent-noun spacing '나아가는데 있어'; preserve source and exclude",
    "rcc-9d6dc8d307acc3272193": "same merged source with nonstandard spacing '나아가는데 있어'",
    "rcc-bb9dd51dcb378eb97986": "same merged source with nonstandard spacing '나아가는데 있어'",
    "rcc-3401f93bd7285b289874": "source attaches honorific dependent noun in '시원(최시원·28)씨가'; preserve source and exclude",
}
HANGUL_SOURCE = re.compile("[가-힣ᄀ-ᇿ㄰-㆏ꥠ-꥿ힰ-퟿]")
MAX_TOKENS = 512


def check(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def source_tokens(text):
    return len(text.encode("utf-8")) + 2


def target_tokens(text):
    return sum(3 if "가" <= char <= "힣" else len(char.encode("utf-8"))
               for char in text) + 1


def valid_model_pair(source, target):
    return (not HANGUL_SOURCE.search(source) and len(source.split()) >= 2
            and source_tokens(source) <= MAX_TOKENS
            and target_tokens(target) <= MAX_TOKENS)


def pinned_rows(directory, split):
    manifest = json.loads((directory / "manifest.json").read_text())
    name = split + ".jsonl"
    path = directory / name
    check(manifest["sha256"][name] == sha256(path), f"{directory.name} {name} hash differs")
    rows = read_jsonl(path)
    check(len(rows) == manifest["counts"][split]
          and all(row.get("split") == split for row in rows),
          f"{directory.name} {split} identity differs")
    return rows, manifest


def known_chunks(corpus, v2):
    chunks = set()
    for path in (corpus / "train_lexicon.json", v2 / "train_lexicon.json"):
        payload = json.loads(path.read_text())
        chunks.update(entry["text"] for entry in payload["entries"])
    return chunks


def source_candidates(natural, corpus, v2, review_path):
    natural_manifest = json.loads((natural / "manifest.json").read_text())
    natural_proof = json.loads((natural / "independent-verification.json").read_text())
    source_path = natural / "train.jsonl"
    check(natural_manifest.get("completed") is True
          and natural_manifest.get("model_inference_performed") is False,
          "natural source contract differs")
    check(natural_proof.get("verified") is True
          and natural_proof.get("manifest_sha256") == sha256(natural / "manifest.json"),
          "natural source proof differs")
    check(natural_manifest["outputs"]["train"]["sha256"] == sha256(source_path),
          "natural TRAIN source differs")

    existing = []
    corpus_hashes = {}
    corpus_manifest = json.loads((corpus / "manifest.json").read_text())
    for split in ("train", "dev", "test"):
        rows, _ = pinned_rows(corpus, split)
        existing.extend(rows)
        corpus_hashes[split] = sha256(corpus / f"{split}.jsonl")
        check(corpus_manifest["sha256"][f"{split}.jsonl"] == corpus_hashes[split],
              "base corpus split hash differs")
    v2_rows, v2_manifest = pinned_rows(v2, "train")
    check(len(v2_rows) == 1024 and v2_manifest.get("corpus_version") == "sentence-train-extra-v2",
          "retained v2 supplement differs")
    check(all(valid_model_pair(row["input"], row["expected"][0]) for row in v2_rows),
          "retained v2 row violates the model input/target contract")
    existing.extend(v2_rows)

    review = json.loads(review_path.read_text())
    quarantine_forms = review.get("quarantined_forms", {})
    v2_review = v2_manifest.get("source_review", {})
    check(set(v2_review.get("quarantined_forms", ())).issubset(quarantine_forms),
          "v2 quarantined forms are absent from the active review")
    quarantined_ids = (set(review.get("quarantined_ids", ()))
                       | set(REVIEWED_QUARANTINE)
                       | set(v2_review.get("quarantined_source_ids", ()))
                       | set(V3_REVIEWED_QUARANTINE))
    excluded_documents = {row["group_id"] for row in existing}
    excluded_sources = {source for row in existing for source in row.get("source_ids", ())}
    excluded_source_rows = {source for row in existing
                            for source in row.get("source_row_ids", ())}
    excluded_inputs = {row["input"] for row in existing}
    excluded_targets = {target for row in existing for target in row["expected"]}
    near_index = NearDuplicateIndex(excluded_targets)
    chunks = known_chunks(corpus, v2)
    rejection_counts = Counter()
    candidates = []
    raw_rows = read_jsonl(source_path)
    check(all(row.get("split") == "train" for row in raw_rows),
          "non-TRAIN natural row encountered")
    for row in merge_sentences(raw_rows, quarantine_forms):
        origins = set(row["origins"])
        if origins & quarantined_ids:
            rejection_counts["quarantined_source_row"] += 1
            continue
        if origins & excluded_source_rows:
            rejection_counts["existing_source_row"] += 1
            continue
        if row["group_id"] in excluded_documents:
            rejection_counts["existing_document"] += 1
            continue
        if set(row["source_ids"]) & excluded_sources:
            rejection_counts["existing_source_id"] += 1
            continue
        if row["text"] in excluded_targets or near_index.contains(row["text"]):
            rejection_counts["existing_target_or_near_duplicate"] += 1
            continue
        source = romanize_word(row["text"])
        if source in excluded_inputs:
            rejection_counts["existing_input"] += 1
            continue
        if HANGUL_SOURCE.search(source) or len(source.split()) < 2:
            rejection_counts["invalid_full_roman_source"] += 1
            continue
        if source_tokens(source) > MAX_TOKENS:
            rejection_counts["source_token_cap"] += 1
            continue
        if target_tokens(row["text"]) > MAX_TOKENS:
            rejection_counts["target_token_cap"] += 1
            continue
        features = row_features(row["text"], chunks)
        strata = eligible_strata(features)
        if not strata:
            rejection_counts["outside_growth_strata"] += 1
            continue
        candidates.append({"source_row_id": min(row["origins"]),
                           "source_row_ids": sorted(row["origins"]),
                           "group_id": row["group_id"],
                           "source_ids": sorted(row["source_ids"]),
                           "focus_families": sorted(row["collisions"]),
                           "input": source, "target": row["text"],
                           "features": features, "eligible_strata": strata})
    metadata = {"natural_manifest": natural_manifest, "v2_manifest": v2_manifest,
                "corpus_hashes": corpus_hashes, "rejections": rejection_counts,
                "quarantined_ids": quarantined_ids,
                "quarantine_forms": set(quarantine_forms),
                "existing": existing, "near_index": near_index,
                "excluded_sources": excluded_sources,
                "excluded_source_rows": excluded_source_rows}
    return candidates, metadata


def new_rows(selected):
    rows = []
    for index, row in enumerate(selected, 1):
        rows.append({"id": f"{NEW_PREFIX}{index:05d}", "input": row["input"],
                     "context": "", "expected": [row["target"]],
                     "category": "natural_sentence_oov_growth_v3",
                     "family": "natural-extra-v3:" + row["group_id"],
                     "split": "train",
                     "provenance": "KLUE-TRAIN-source-reference+synthetic-Roman-sentence",
                     "context_role": "none", "boundary": "word_spaced",
                     "group_id": row["group_id"], "source_ids": row["source_ids"],
                     "source_row_ids": row["source_row_ids"],
                     "focus_families": row["focus_families"],
                     "selection_stratum": row["selection_stratum"],
                     "structural_features": row["features"],
                     "input_contract": "roman_sentence_only",
                     "reference_policy": "observed_source_sentence; alternative readings not exhaustively annotated"})
    return rows


def write_review_sample(path, rows, train_sha):
    selected = sorted(rows, key=lambda row: hashlib.sha256(row["id"].encode()).hexdigest())[:12]
    payload = {"completed": True, "selection": "first 12 new rows by SHA-256(id); label-independent",
               "train_sha256": train_sha, "rows": [{key: row[key] for key in
                   ("id", "input", "expected", "group_id", "source_ids", "source_row_ids",
                    "selection_stratum")} for row in selected],
               "labels_corrected": False, "model_inference_performed": False}
    path.parent.mkdir(parents=True, exist_ok=True)
    check(not path.exists(), "preserve existing review sample")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return payload


def build(output=OUTPUT, *, natural=NATURAL, corpus=CORPUS, v2=V2,
          review_path=REVIEW, quotas=QUOTAS, review_output=None):
    output, natural, corpus, v2, review_path = map(
        Path, (output, natural, corpus, v2, review_path))
    check(not output.exists(), "preserve existing v3 supplement")
    check(quotas == QUOTAS and sum(quotas.values()) == 1024,
          "v3 growth requires the fixed 1,024-row quota")
    candidates, metadata = source_candidates(natural, corpus, v2, review_path)
    selected, stratum_counts = select_growth(candidates, quotas, metadata["near_index"])
    additions = new_rows(selected)
    check(len(additions) == 1024 and stratum_counts == QUOTAS,
          "new growth count or quota differs")
    retained_raw = (v2 / "train.jsonl").read_bytes()
    retained_rows = read_jsonl(v2 / "train.jsonl")
    combined = retained_rows + additions
    check(len(combined) == 2048 and len({row["id"] for row in combined}) == 2048,
          "combined row identity differs")
    check(len({row["input"] for row in combined}) == 2048,
          "combined input identity differs")
    check(len({row["group_id"] for row in combined}) == 2048,
          "combined document identity differs")
    check(not ({source for row in additions for source in row["source_ids"]}
               & metadata["excluded_sources"]), "new source IDs overlap")
    check(not ({source for row in additions for source in row["source_row_ids"]}
               & metadata["excluded_source_rows"]), "new source row IDs overlap")

    source_path = natural / "train.jsonl"
    pin_paths = (source_path, natural / "manifest.json",
                 natural / "independent-verification.json", corpus / "manifest.json",
                 corpus / "train_lexicon.json", review_path, v2 / "train.jsonl",
                 v2 / "train_lexicon.json", v2 / "manifest.json",
                 v2 / "independent-verification.json", Path(__file__),
                 ROOT / "scripts/build_sentence_training_extra.py",
                 ROOT / "scripts/build_sentence_challenge.py", ROOT / "hanglish/core/phonology.py")
    pins = {str(path.relative_to(ROOT)): sha256(path) for path in pin_paths}
    pins.update({str((corpus / f"{split}.jsonl").relative_to(ROOT)):
                 metadata["corpus_hashes"][split] for split in ("train", "dev", "test")})
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".sentence-train-extra-v3-", dir=output.parent) as temp:
        staged = Path(temp) / "corpus"
        staged.mkdir()
        train_path = staged / "train.jsonl"
        with train_path.open("wb") as stream:
            stream.write(retained_raw)
            stream.write("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                                 for row in additions).encode())
        check(train_path.read_bytes()[:len(retained_raw)] == retained_raw,
              "retained v2 rows changed at byte level")
        words = Counter(word for row in combined for target in row["expected"]
                        for word in re.findall("[가-힣]+", target))
        lexicon = {"schema_version": 1, "source_split": "train",
                   "source_rows": len(combined), "retained_rows": 1024,
                   "new_rows": 1024, "train_sha256": sha256(train_path),
                   "provenance": {"fields": ["expected"], "context_included": False,
                                  "runtime_dictionary_injection": False,
                                  "selection_known_vocabulary": "base TRAIN plus retained v2 TRAIN only"},
                   "entries": [{"text": word, "count": count}
                               for word, count in sorted(words.items())]}
        lexicon_path = staged / "train_lexicon.json"
        lexicon_path.write_text(json.dumps(lexicon, ensure_ascii=False) + "\n")
        manifest = {"schema_version": 1, "corpus_version": "sentence-train-extra-v3",
                    "completed": True, "input_contract": "roman_sentence_only",
                    "model_input_fields": ["input"], "label_fields": ["expected"],
                    "split": "train", "rows": 2048, "counts": {"train": 2048},
                    "retained_rows": 1024, "new_rows": 1024,
                    "retained_corpus": "sentence-train-extra-v2",
                    "retained_train_sha256": sha256(v2 / "train.jsonl"),
                    "retained_prefix_bytes": len(retained_raw),
                    "new_selection_strata": stratum_counts,
                    "candidate_rows": len(candidates),
                    "rejections": dict(metadata["rejections"]),
                    "selection_policy": "retain v2 byte-exact; select 1,024 additional TRAIN rows with reused deterministic family round robin and fixed structural/OOV quotas",
                    "known_vocabulary_policy": "base TRAIN lexicon plus retained v2 TRAIN lexicon only",
                    "exclusion_policy": "base train/dev/test plus retained v2 used for document/source row/source ID/input/target exclusion and normalized containment or character5gram Jaccard>=.8",
                    "test_use": "exclusion identities only; no test-derived examples or selection features",
                    "romanization_policy": "pinned forward romanize_word over the complete observed TRAIN source sentence",
                    "source_review": {"all_rows_expert_verified": False,
                                      "labels_corrected": False,
                                      "observed_source_labels_preserved": True,
                                      "quarantined_forms": sorted(metadata["quarantine_forms"]),
                                      "quarantined_source_ids": sorted(metadata["quarantined_ids"]),
                                      "reviewed_quarantine_reasons": {
                                          **REVIEWED_QUARANTINE, **V3_REVIEWED_QUARANTINE}},
                    "limitations": [
                        "Synthetic Roman inputs use the same forward rules as validation; round trips are not independent phonological review.",
                        "Observed TRAIN source sentences are not fully expert- or sense-annotated.",
                        "The supplement is TRAIN-only and is not evaluation evidence."],
                    "license": metadata["natural_manifest"]["license"],
                    "attribution": metadata["natural_manifest"]["attribution"],
                    "pins": pins,
                    "sha256": {"train.jsonl": sha256(train_path),
                               "train_lexicon.json": sha256(lexicon_path)},
                    "training_run": False, "model_inference_performed": False,
                    "production_changed": False}
        (staged / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        (staged / "ATTRIBUTION.md").write_text(
            "# Attribution\n\nNatural sentence references: "
            + metadata["natural_manifest"]["attribution"] + "\n\nLicense: "
            + metadata["natural_manifest"]["license"]
            + ". Romanized derivatives retain this attribution and license.\n")
        staged.rename(output)
    if review_output is not None:
        write_review_sample(Path(review_output), additions, manifest["sha256"]["train.jsonl"])
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--review-output", type=Path, default=REVIEW_OUTPUT)
    args = parser.parse_args()
    result = build(args.output, review_output=args.review_output)
    print(json.dumps({key: result[key] for key in
                      ("completed", "rows", "retained_rows", "new_rows",
                       "new_selection_strata", "candidate_rows", "rejections")},
                     ensure_ascii=False))
