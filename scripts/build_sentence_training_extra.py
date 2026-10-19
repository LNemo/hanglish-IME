"""Build a disjoint TRAIN-only whole-sentence supplement from pinned KLUE sources."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hanglish.core.phonology import romanize_word
from scripts.build_sentence_challenge import merge_sentences, natural_eligible


NATURAL = ROOT / "data/natural_context/runtime-collision-source-v5"
CORPUS = ROOT / "data/challenge/sentence-v3"
REVIEW = ROOT / ".omx/artifacts/runtime-collision-source-review-v2.json"
OUTPUT = ROOT / "data/challenge/sentence-train-extra-v2"
QUOTAS = {"structural_oov": 256, "mixed_oov": 384, "plain_oov": 384}
SALT = "sentence-train-extra-v2"
REVIEWED_QUARANTINE = {
    "rcc-2b7887d8a6b5a2c0919c": "source has nonstandard spacing '체감 할'; preserve source and exclude",
    "rcc-3aa6f9295d81eb376b6b": "duplicate occurrence of source with nonstandard spacing '체감 할'",
    "rcc-6d461255f29f4b9b1b42": "duplicate occurrence of source with nonstandard spacing '체감 할'",
    "rcc-1d2a1b310f5a16cc433e": "source has nonstandard spacing '1년전'; preserve source and exclude",
}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256((SALT + "\0" + value).encode()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def normalized_text(text):
    value = unicodedata.normalize("NFKC", text).lower()
    return "".join(character for character in value if character.isalnum())


class NearDuplicateIndex:
    """Detect normalized containment or character-fivegram similarity."""

    def __init__(self, texts=()):
        self.values = []
        self.exact = set()
        self.shingles = defaultdict(set)
        self.fivegrams = []
        self.fivegram_index = defaultdict(set)
        for text in texts:
            self.add(text)

    @staticmethod
    def _keys(value):
        if len(value) < 8:
            return (value,)
        return {value[index:index + 8] for index in range(len(value) - 7)}

    @staticmethod
    def _fivegrams(value):
        if len(value) < 5:
            return {value}
        return {value[index:index + 5] for index in range(len(value) - 4)}

    def add(self, text):
        value = normalized_text(text)
        if not value or value in self.exact:
            return
        index = len(self.values)
        self.values.append(value)
        self.exact.add(value)
        for key in self._keys(value):
            self.shingles[key].add(index)
        grams = self._fivegrams(value)
        self.fivegrams.append(grams)
        for gram in grams:
            self.fivegram_index[gram].add(index)

    def contains(self, text):
        value = normalized_text(text)
        if not value or value in self.exact:
            return True
        possible = set()
        for key in self._keys(value):
            possible.update(self.shingles.get(key, ()))
        if any(min(len(value), len(self.values[index])) >= 12
               and (value in self.values[index] or self.values[index] in value)
               for index in possible):
            return True
        grams = self._fivegrams(value)
        similar = set()
        for gram in grams:
            similar.update(self.fivegram_index.get(gram, ()))
        return any(len(grams & self.fivegrams[index]) / len(grams | self.fivegrams[index]) >= .8
                   for index in similar)


def row_features(text, known_chunks):
    chunks = re.findall("[가-힣]+", text)
    oov = sorted(set(chunks) - known_chunks)
    ascii_letters = bool(re.search("[A-Za-z]", text))
    digits = bool(re.search(r"\d", text))
    punctuation = len(re.findall(r"[^가-힣A-Za-z0-9\s]", text))
    structural = (len(text) >= 80 or punctuation >= 4
                  or bool(re.search("[()\\[\\]{}“”‘’'\"]", text)))
    return {"oov_target_chunks": oov, "has_ascii": ascii_letters,
            "has_digits": digits, "punctuation_count": punctuation,
            "long_or_punctuated": structural}


def eligible_strata(features):
    oov_count = len(features["oov_target_chunks"])
    mixed = features["has_ascii"] or features["has_digits"]
    result = []
    if features["long_or_punctuated"] and oov_count >= 2:
        result.append("structural_oov")
    if mixed and oov_count >= 2:
        result.append("mixed_oov")
    if not mixed and oov_count >= 4:
        result.append("plain_oov")
    return result


def select_growth(candidates, quotas, near_index):
    """Deterministic family round robin with one row per document/source."""
    selected = []
    selected_ids, documents, sources, inputs = set(), set(), set(), set()
    counts = Counter()
    for stratum in ("structural_oov", "mixed_oov", "plain_oov"):
        wanted = quotas.get(stratum, 0)
        buckets = defaultdict(list)
        for row in candidates:
            if stratum not in row["eligible_strata"]:
                continue
            family = min(row["focus_families"], key=lambda value: digest("family\0" + value))
            buckets[family].append(row)
        queues = {family: deque(sorted(rows, key=lambda row: digest("row\0" + row["source_row_id"])))
                  for family, rows in buckets.items()}
        families = sorted(queues, key=lambda value: digest("order\0" + value))
        while counts[stratum] < wanted:
            progressed = False
            for family in families:
                queue = queues[family]
                while queue:
                    row = queue.popleft()
                    row_sources = set(row["source_ids"])
                    if (row["source_row_id"] in selected_ids or row["group_id"] in documents
                            or row_sources & sources or row["input"] in inputs
                            or near_index.contains(row["target"])):
                        continue
                    chosen = dict(row, selection_stratum=stratum)
                    selected.append(chosen)
                    selected_ids.add(row["source_row_id"])
                    documents.add(row["group_id"])
                    sources.update(row_sources)
                    inputs.add(row["input"])
                    near_index.add(row["target"])
                    counts[stratum] += 1
                    progressed = True
                    break
                if counts[stratum] == wanted:
                    break
            check(progressed, f"not enough disjoint rows for {stratum}")
    return selected, dict(counts)


def build(output=OUTPUT, *, natural=NATURAL, corpus=CORPUS, review_path=REVIEW,
          quotas=QUOTAS):
    output, natural, corpus, review_path = map(Path, (output, natural, corpus, review_path))
    check(not output.exists(), "preserve existing supplement; use a new output path")
    check(sum(quotas.values()) <= 2048 and sum(quotas.values()) >= 512,
          "supplement must contain 512 through 2048 rows")
    natural_manifest = json.loads((natural / "manifest.json").read_text())
    natural_proof = json.loads((natural / "independent-verification.json").read_text())
    check(natural_manifest.get("completed") is True and natural_manifest.get("model_inference_performed") is False,
          "natural source contract differs")
    check(natural_proof.get("verified") is True
          and natural_proof.get("manifest_sha256") == sha256(natural / "manifest.json"),
          "natural source verification differs")
    source_path = natural / "train.jsonl"
    check(sha256(source_path) == natural_manifest["outputs"]["train"]["sha256"],
          "natural TRAIN source differs")
    review = json.loads(review_path.read_text())
    quarantine = review.get("quarantined_forms", {})
    quarantined_ids = set(review.get("quarantined_ids", ())) | set(REVIEWED_QUARANTINE)

    existing = []
    split_hashes = {}
    for split in ("train", "dev", "test"):
        path = corpus / f"{split}.jsonl"
        rows = read_jsonl(path)
        check(all(row.get("split") == split for row in rows), "existing split identity differs")
        existing.extend(rows)
        split_hashes[split] = sha256(path)
    corpus_manifest = json.loads((corpus / "manifest.json").read_text())
    check(all(corpus_manifest["sha256"][f"{split}.jsonl"] == split_hashes[split]
              for split in ("train", "dev", "test")), "existing corpus hash differs")

    excluded_documents = {row["group_id"] for row in existing}
    excluded_sources = {source for row in existing for source in row.get("source_ids", ())}
    excluded_inputs = {row["input"] for row in existing}
    excluded_targets = {target for row in existing for target in row["expected"]}
    near_index = NearDuplicateIndex(excluded_targets)
    train_lexicon = json.loads((corpus / "train_lexicon.json").read_text())
    known_chunks = {entry["text"] for entry in train_lexicon["entries"]}

    raw_rows = read_jsonl(source_path)
    check(all(row.get("split") == "train" for row in raw_rows), "non-TRAIN source row encountered")
    merged = merge_sentences(raw_rows, quarantine)
    rejection_counts = Counter()
    candidates = []
    for row in merged:
        if quarantined_ids.intersection(row["origins"]):
            rejection_counts["quarantined_source_row"] += 1
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
        if (any("가" <= character <= "힣" for character in source)
                or len(source.split()) < 2):
            rejection_counts["invalid_full_roman_source"] += 1
            continue
        features = row_features(row["text"], known_chunks)
        strata = eligible_strata(features)
        if not strata:
            rejection_counts["outside_growth_strata"] += 1
            continue
        candidates.append({"source_row_id": min(row["origins"]), "source_row_ids": sorted(row["origins"]),
            "group_id": row["group_id"], "source_ids": sorted(row["source_ids"]),
            "focus_families": sorted(row["collisions"]), "input": source, "target": row["text"],
            "features": features, "eligible_strata": strata})

    selected, stratum_counts = select_growth(candidates, quotas, near_index)
    rows = []
    for index, row in enumerate(selected, 1):
        rows.append({"id": f"s3x-train-{index:05d}", "input": row["input"], "context": "",
            "expected": [row["target"]], "category": "natural_sentence_oov_growth",
            "family": "natural-extra:" + row["group_id"], "split": "train",
            "provenance": "KLUE-TRAIN-source-reference+synthetic-Roman-sentence",
            "context_role": "none", "boundary": "word_spaced", "group_id": row["group_id"],
            "source_ids": row["source_ids"], "source_row_ids": row["source_row_ids"],
            "focus_families": row["focus_families"], "selection_stratum": row["selection_stratum"],
            "structural_features": row["features"], "input_contract": "roman_sentence_only",
            "reference_policy": "observed_source_sentence; alternative readings not exhaustively annotated"})
    check(len(rows) == sum(quotas.values()), "supplement count differs")
    check(len({row["input"] for row in rows}) == len(rows), "duplicate supplement input")
    check(len({row["group_id"] for row in rows}) == len(rows), "duplicate supplement document")
    check(not {source for row in rows for source in row["source_ids"]} & excluded_sources,
          "supplement source overlaps existing corpus")

    pins = {str(path.relative_to(ROOT)): sha256(path) for path in
            (source_path, natural / "manifest.json", natural / "independent-verification.json",
             corpus / "manifest.json", corpus / "train_lexicon.json", review_path,
             Path(__file__), ROOT / "scripts/build_sentence_challenge.py",
             ROOT / "hanglish/core/phonology.py")}
    pins.update({str((corpus / f"{split}.jsonl").relative_to(ROOT)): split_hashes[split]
                 for split in ("train", "dev", "test")})
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".sentence-train-extra-", dir=output.parent) as temporary:
        staged = Path(temporary) / "corpus"
        staged.mkdir()
        train_path = staged / "train.jsonl"
        train_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                                      for row in rows))
        words = Counter(word for row in rows for answer in row["expected"]
                        for word in re.findall("[가-힣]+", answer))
        lexicon = {"schema_version": 1, "source_split": "train", "source_rows": len(rows),
            "train_sha256": sha256(train_path),
            "provenance": {"fields": ["expected"], "context_included": False,
                           "runtime_dictionary_injection": False},
            "entries": [{"text": word, "count": count} for word, count in sorted(words.items())]}
        lexicon_path = staged / "train_lexicon.json"
        lexicon_path.write_text(json.dumps(lexicon, ensure_ascii=False) + "\n")
        manifest = {"schema_version": 1, "corpus_version": "sentence-train-extra-v2",
            "completed": True, "input_contract": "roman_sentence_only", "model_input_fields": ["input"],
            "label_fields": ["expected"], "split": "train", "rows": len(rows),
            "counts": {"train": len(rows)},
            "selection_strata": stratum_counts, "candidate_rows": len(candidates),
            "rejections": dict(rejection_counts), "selection_policy":
            "TRAIN-only; one document and disjoint source IDs per row; family hash round robin; fixed structural/OOV quotas",
            "exclusion_policy":
            "all sentence-v3 splits used only for document/source/input/target exclusion; normalized containment or character5gram Jaccard>=.8",
            "test_use": "exclusion identities only; no test-derived examples or selection features",
            "romanization_policy": "pinned forward romanize_word over the complete observed source sentence",
            "source_review": {"all_rows_expert_verified": False, "labels_corrected": False,
                              "quarantined_forms": sorted(quarantine),
                              "quarantined_source_ids": sorted(quarantined_ids),
                              "reviewed_quarantine_reasons": REVIEWED_QUARANTINE},
            "limitations": [
                "Synthetic Roman inputs are generated by the same forward rules used in validation; round trips are not independent phonological review.",
                "Observed source sentences and collision alternatives are not fully expert- or sense-annotated.",
                "The supplement is TRAIN-only and does not constitute evaluation evidence."],
            "license": natural_manifest["license"], "attribution": natural_manifest["attribution"],
            "pins": pins, "sha256": {"train.jsonl": sha256(train_path),
                                       "train_lexicon.json": sha256(lexicon_path)},
            "training_run": False, "model_inference_performed": False, "production_changed": False}
        (staged / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        (staged / "ATTRIBUTION.md").write_text(
            "# Attribution\n\nNatural sentence references: " + natural_manifest["attribution"]
            + "\n\nLicense: " + natural_manifest["license"]
            + ". Romanized derivatives retain this attribution and license.\n")
        staged.rename(output)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = build(args.output)
    print(json.dumps({key: result[key] for key in
                      ("completed", "rows", "selection_strata", "candidate_rows", "rejections")},
                     ensure_ascii=False))
