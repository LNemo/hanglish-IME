#!/usr/bin/env python3
"""Audit leakage-safe family splits for a broader TRAIN-only phonetic corpus."""

from __future__ import annotations

import argparse
import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hanglish.core.lattice import CandidateGenerator, suffix_allowed
from hanglish.core.lexical_data import FILES
from hanglish.core.phonology import romanize_word


TRAIN = ROOT / "data/natural_context/final-choice-v1/natural_fit/cases.jsonl"
TRAIN_MANIFEST = TRAIN.with_name("manifest.json")
MECAB = ROOT / "data/external/mecab-ko-dic"
OUTPUT = ROOT / ".omx/artifacts/phonetic-training-family-audit-v1.json"
ORIGINAL_PAIRS = 50384
ORIGINAL_SHA256 = "b0a5ff8d112d0afe910164134394db3daca48d5a21c9f3c4efe0342acfb73e2b"
OBSERVED_ADDITIONS = 13837
ANCHORED_ADDITIONS = 12453
EXPECTED_UNION = 76674
LEXICAL_POS = frozenset({"VV", "VA", "VX", "VCP", "VCN", "NNG", "NNP", "NP", "NR", "XR"})
HANGUL = re.compile(r"[가-힣]+")
HANGUL_FULL = re.compile(r"[가-힣]+\Z")
HOMONYM_DIGITS = re.compile(r"(?<=[가-힣])[0-9]+\Z")
PRODUCTIVE_ENDINGS = (
    "하다", "하기", "합니다", "해요", "하고", "하는", "하면", "했다", "했어요",
    "했습니다", "하세요",
)
SOURCE_FILES = (
    "data/morphology_lexicon.json",
    "data/workbench_lexicon.json",
    "data/challenge/train_lexicon.json",
    "data/common_english.txt",
    "data/it_terms.txt",
    "hanglish/core/lattice.py",
    "hanglish/core/lexical_data.py",
    "hanglish/core/phonology.py",
    "scripts/audit_phonetic_training_families.py",
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pair_sha(pairs) -> str:
    return hashlib.sha256(
        "\n".join(f"{roman}\t{surface}" for roman, surface in sorted(pairs)).encode()
    ).hexdigest()


def stable_bucket(value: str, modulo: int = 20) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") % modulo


def normalize_lexeme(text: str) -> str:
    """Normalize only MeCab's trailing homonym index."""
    return HOMONYM_DIGITS.sub("", text)


def eligible_pair(surface: str) -> tuple[str, str] | None:
    if not HANGUL_FULL.fullmatch(surface) or len(surface) > 24:
        return None
    roman = romanize_word(surface)
    if not (roman.isascii() and roman.isalpha() and roman.islower()
            and 2 <= len(roman) <= 40):
        return None
    return roman, surface


def read_inflect(path: Path = MECAB / "Inflect.csv") -> dict[str, set[tuple]]:
    analyses: dict[str, set[tuple]] = {}
    for row in csv.reader(path.read_text(encoding="utf-8").splitlines()):
        if len(row) < 12 or not HANGUL_FULL.fullmatch(row[0]):
            continue
        parts = []
        for item in row[11].split("+"):
            fields = item.split("/")
            if len(fields) < 2 or not fields[0] or not fields[1]:
                parts = []
                break
            parts.append((normalize_lexeme(fields[0]), fields[1]))
        if parts and any(pos in LEXICAL_POS for _, pos in parts):
            analyses.setdefault(row[0], set()).add(tuple(parts))
    return analyses


def first_lexical(analysis: tuple) -> tuple[int, str, str] | None:
    for index, (surface, pos) in enumerate(analysis):
        if pos in LEXICAL_POS:
            return index, surface, pos
    return None


def load_train_words(path: Path = TRAIN) -> tuple[list[dict], set[str]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if (len(rows) != 2048 or any(row.get("origin") != "natural" for row in rows)
            or any(not isinstance(row.get("expected"), list) or len(row["expected"]) != 1
                   for row in rows)):
        raise ValueError("natural_fit TRAIN rows differ")
    words = {word for row in rows for sentence in row["expected"] for word in HANGUL.findall(sentence)}
    return rows, words


def reconstruct_original(generator: CandidateGenerator) -> set[tuple[str, str]]:
    words = {
        option.text for options in generator.index.values() for option in options.values()
        if option.text and HANGUL_FULL.fullmatch(option.text)
    }
    current = {
        (roman, word) for word in words
        if (roman := romanize_word(word)).isascii() and roman.isalpha()
    }
    later_noun_gi = {
        pair for pair in current
        if pair[1].endswith("하기") and pair[1][:-2] in generator.nouns
    }
    original = current - later_noun_gi
    if (len(later_noun_gi) != 360 or len(original) != ORIGINAL_PAIRS
            or pair_sha(original) != ORIGINAL_SHA256):
        raise ValueError("original checkpoint training pairs do not reconstruct exactly")
    return original


def build_corpus(generator: CandidateGenerator, analyses: dict[str, set[tuple]],
                 train_words: set[str]):
    original = reconstruct_original(generator)
    observed = {pair for word in train_words if (pair := eligible_pair(word)) is not None}
    anchored_roots = {
        (root, pos)
        for surface in train_words
        for analysis in analyses.get(surface, ())
        if (lexical := first_lexical(analysis)) is not None
        for _, root, pos in (lexical,)
    }
    anchored = {
        pair for surface, surface_analyses in analyses.items()
        if any((lexical[1], lexical[2]) in anchored_roots
               for analysis in surface_analyses
               if (lexical := first_lexical(analysis)) is not None)
        if (pair := eligible_pair(surface)) is not None
    }
    observed_new = observed - original
    anchored_new = anchored - original - observed
    union = original | observed | anchored
    if (len(observed_new) != OBSERVED_ADDITIONS
            or len(anchored_new) != ANCHORED_ADDITIONS
            or len(union) != EXPECTED_UNION):
        raise ValueError("expanded TRAIN-only corpus counts differ")
    return original, observed_new, anchored_new, union, anchored_roots


class UnionFind:
    def __init__(self):
        self.parent = {}
        self.size = {}

    def add(self, item):
        if item not in self.parent:
            self.parent[item] = item
            self.size[item] = 1

    def find(self, item):
        self.add(item)
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.size[left] < self.size[right]:
            left, right = right, left
        self.parent[right] = left
        self.size[left] += self.size[right]


def productive_aliases(generator: CandidateGenerator, pairs: set[tuple[str, str]]):
    surfaces = {surface for _, surface in pairs}
    aliases: dict[str, set[str]] = {}
    for noun in generator.nouns:
        if noun in surfaces:
            aliases.setdefault(noun, set()).add(noun)
        for suffix in generator.suffixes:
            form = noun + suffix
            if form in surfaces and suffix_allowed(noun, suffix):
                aliases.setdefault(form, set()).add(noun)
        for ending in PRODUCTIVE_ENDINGS:
            form = noun + ending
            if form in surfaces:
                aliases.setdefault(form, set()).add(noun)
    return aliases


def analysis_roots_and_signatures(surface: str, analyses: dict[str, set[tuple]], aliases):
    roots, signatures = set(), set()
    for analysis in analyses.get(surface, ()):
        lexical = first_lexical(analysis)
        if lexical is None:
            continue
        index, root, pos = lexical
        roots.add((root, pos))
        tail = analysis[index + 1:]
        if tail:
            signatures.add(tuple(tail))
    for root in aliases.get(surface, ()):
        roots.add((root, "AUTHORED"))
        if surface != root:
            signatures.add(((surface[len(root):], "PRODUCTIVE"),))
    return roots, signatures


def family_components(pairs: set[tuple[str, str]], analyses, aliases):
    ordered = sorted(pairs)
    uf = UnionFind()
    by_roman: dict[str, list[int]] = {}
    roots_by_pair, signatures_by_pair = {}, {}
    root_owner = {}
    for index, (roman, surface) in enumerate(ordered):
        uf.add(index)
        by_roman.setdefault(roman, []).append(index)
        roots, signatures = analysis_roots_and_signatures(surface, analyses, aliases)
        roots_by_pair[index], signatures_by_pair[index] = roots, signatures
        for root in roots:
            if root in root_owner:
                uf.union(index, root_owner[root])
            else:
                root_owner[root] = index
    pre_collision_hazards = sum(
        len({uf.find(index) for index in indices}) > 1 for indices in by_roman.values()
    )
    for indices in by_roman.values():
        for index in indices[1:]:
            uf.union(indices[0], index)
    components: dict[int, list[int]] = {}
    for index in range(len(ordered)):
        components.setdefault(uf.find(index), []).append(index)
    canonical = {
        root: hashlib.sha256("\n".join(
            f"{ordered[index][0]}\t{ordered[index][1]}" for index in indices
        ).encode()).hexdigest()
        for root, indices in components.items()
    }
    return ordered, components, canonical, roots_by_pair, signatures_by_pair, pre_collision_hazards


def split_audit(pairs, analyses, aliases):
    (ordered, components, family_keys, roots_by_pair, signatures_by_pair,
     pre_collision_hazards) = family_components(pairs, analyses, aliases)
    component_for = {index: root for root, indices in components.items() for index in indices}
    lexical_roots = {root for root, key in family_keys.items() if stable_bucket(key) == 0}

    combo_uf = UnionFind()
    signature_owner = {}
    roman_owner = {}
    morph_eligible = set()
    for index, (roman, _) in enumerate(ordered):
        if component_for[index] in lexical_roots:
            continue
        combo_uf.add(index)
        if roman in roman_owner:
            combo_uf.union(index, roman_owner[roman])
        else:
            roman_owner[roman] = index
        for signature in signatures_by_pair[index]:
            morph_eligible.add(index)
            key = (component_for[index], signature)
            if key in signature_owner:
                combo_uf.union(index, signature_owner[key])
            else:
                signature_owner[key] = index

    combo_groups: dict[int, list[int]] = {}
    for index in combo_uf.parent:
        combo_groups.setdefault(combo_uf.find(index), []).append(index)
    morph_roots = set()
    for root, indices in combo_groups.items():
        if not any(index in morph_eligible for index in indices):
            continue
        key = hashlib.sha256("\n".join(
            f"{ordered[index][0]}\t{ordered[index][1]}" for index in sorted(indices)
        ).encode()).hexdigest()
        if stable_bucket(key) == 1:
            morph_roots.add(root)

    roles = {}
    for index in range(len(ordered)):
        if component_for[index] in lexical_roots:
            roles[index] = "lexical_holdout"
        elif combo_uf.find(index) in morph_roots:
            roles[index] = "morph_combination_holdout"
        else:
            roles[index] = "train"

    family_role_sets = {
        root: {roles[index] for index in indices} for root, indices in components.items()
    }
    signature_roles = {}
    roman_roles = {}
    roman_surfaces = {}
    for index, (roman, _) in enumerate(ordered):
        roman_roles.setdefault(roman, set()).add(roles[index])
        roman_surfaces.setdefault(roman, set()).add(ordered[index][1])
        for signature in signatures_by_pair[index]:
            signature_roles.setdefault((component_for[index], signature), set()).add(roles[index])
    counts = {role: sum(value == role for value in roles.values()) for role in set(roles.values())}
    largest = []
    for root, indices in sorted(components.items(), key=lambda item: (-len(item[1]), family_keys[item[0]]))[:20]:
        roots = sorted({f"{surface}/{pos}" for index in indices for surface, pos in roots_by_pair[index]})
        largest.append({
            "component_sha256": family_keys[root], "pairs": len(indices),
            "roman_inputs": len({ordered[index][0] for index in indices}),
            "surfaces": len({ordered[index][1] for index in indices}),
            "root_examples": roots[:12],
            "surface_examples": [ordered[index][1] for index in indices[:12]],
        })
    return {
        "counts": counts,
        "fractions": {role: count / len(ordered) for role, count in counts.items()},
        "families": len(components),
        "largest_components": largest,
        "largest_component_pairs": max(map(len, components.values())),
        "largest_component_fraction": max(map(len, components.values())) / len(ordered),
        "lexical_family_cross_split": sum(len(values) > 1 for values in family_role_sets.values()
                                          if "lexical_holdout" in values),
        "morph_signature_cross_train_holdout": sum(
            "train" in values and "morph_combination_holdout" in values
            for values in signature_roles.values()
        ),
        "roman_input_cross_split": sum(len(values) > 1 for values in roman_roles.values()),
        "ambiguous_roman_keys": sum(
            len(surfaces) > 1 for surfaces in roman_surfaces.values()
        ),
        "pre_collision_family_hazards": pre_collision_hazards,
        "morph_eligible_pairs": len(morph_eligible),
        "morph_combination_groups": sum(
            any(index in morph_eligible for index in indices) for indices in combo_groups.values()
        ),
    }


def source_hashes():
    paths = [ROOT / name for name in SOURCE_FILES]
    paths.extend((TRAIN, TRAIN_MANIFEST, MECAB / "SOURCE.json"))
    paths.extend(MECAB / f"{name}.csv" for name in FILES)
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in paths}


def build_report():
    before = source_hashes()
    generator = CandidateGenerator()
    analyses = read_inflect()
    rows, words = load_train_words()
    original, observed, anchored, pairs, roots = build_corpus(generator, analyses, words)
    aliases = productive_aliases(generator, pairs)
    split = split_audit(pairs, analyses, aliases)
    after = source_hashes()
    if before != after:
        raise ValueError("audit sources changed")
    report = {
        "schema": "phonetic-training-family-audit-v1",
        "completed": True,
        "scope": "TRAIN-only data preparation; no model, development, reserve, or final access",
        "corpus": {
            "original_pairs": len(original), "original_pairs_sha256": pair_sha(original),
            "natural_fit_rows": len(rows), "natural_fit_unique_hangul_words": len(words),
            "observed_additions": len(observed), "anchored_inflect_additions": len(anchored),
            "union_pairs": len(pairs), "union_pairs_sha256": pair_sha(pairs),
            "anchored_lexical_roots": len(roots),
            "ambiguous_roman_keys": split["ambiguous_roman_keys"],
        },
        "family_recipe": {
            "lexical_roots": sorted(LEXICAL_POS),
            "normalization": "remove trailing homonym digits only",
            "multiple_analyses": "union every valid first lexical root",
            "known_aliases": "authored noun+validated suffix and noun+productive-ending forms",
            "collision_closure": "all outputs with the same Roman input stay in one split",
            "lexical_holdout": "sha256(closed family pairs) modulo 20 == 0",
            "morph_combination_holdout": (
                "among non-lexical families, close identical family+suffix/POS signatures and "
                "same-input outputs; sha256(closed group pairs) modulo 20 == 1"
            ),
        },
        "split": split,
        "feasible": (
            split["lexical_family_cross_split"] == 0
            and split["morph_signature_cross_train_holdout"] == 0
            and split["roman_input_cross_split"] == 0
            and split["largest_component_pairs"] < len(pairs) // 2
            and split["counts"].get("lexical_holdout", 0) > 0
            and split["counts"].get("morph_combination_holdout", 0) > 0
        ),
        "limitations": [
            "Inflect.csv surface attestation does not prove that every form is grammatical in every context.",
            "The audit measures deterministic split structure, not model accuracy or runtime candidate coverage.",
            "A new family-holdout evaluation is independent only for a model trained from scratch; initializing the existing transducer would leak held pairs through its weights.",
            "TRAIN corpus word extraction is orthographic and does not claim a full morphological analysis.",
            ("Conservative multi-analysis and same-input closure creates a "
             f"{split['largest_component_pairs']:,}-pair component "
             f"({split['largest_component_fraction']:.1%} of the corpus), reducing effective "
             "family resolution but not collapsing most pairs."),
        ],
        "training_requirement": "train the transducer from scratch after the full existing-beam baseline closes",
        "model_loaded": False, "model_inference_run": False, "training_run": False,
        "development_data_read": False, "reserve_data_read": False, "final_data_read": False,
        "external_versions": {"wordfreq": generator.external_version or version("wordfreq")},
        "generator_provenance": {
            "external_word_count": generator.external_word_count,
            "external_fingerprint": generator.external_fingerprint,
            "dictionary_fingerprint": generator.dictionary_fingerprint,
            "dictionary_rejected_rows": generator.dictionary_rejected_rows,
        },
        "source_hashes": before,
        "source_hashes_after": after,
    }
    return report


def main(output: Path = OUTPUT):
    if output.exists():
        raise FileExistsError(output)
    report = build_report()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"completed": True, "feasible": report["feasible"],
                      "corpus": report["corpus"], "split": report["split"]["counts"]},
                     ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    main(parser.parse_args().output)
