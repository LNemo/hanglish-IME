#!/usr/bin/env python3
"""Audit provenance quality of phonetic pairs without judging unattested forms invalid."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wordfreq import get_frequency_dict, top_n_list

from hanglish.core.lattice import CandidateGenerator, suffix_allowed
from hanglish.core.lexical_data import FILES, HANGUL, invalid_past_duplicate
from hanglish.core.phonology import romanize_word
import scripts.audit_phonetic_training_families as family


OUTPUT = ROOT / ".omx/artifacts/phonetic-training-source-quality-v1.json"
ORIGINAL_ENDINGS = (
    "하다", "합니다", "해요", "하고", "하는", "하면", "했다", "했어요", "했습니다", "하세요",
)
SOURCE_FILES = (
    "scripts/audit_phonetic_pair_sources.py",
    "scripts/audit_phonetic_training_families.py",
    "hanglish/core/lattice.py",
    "hanglish/core/lexical_data.py",
    "hanglish/core/phonology.py",
    "data/morphology_lexicon.json",
    "data/workbench_lexicon.json",
    "data/challenge/train_lexicon.json",
    "data/natural_context/final-choice-v1/natural_fit/cases.jsonl",
    "data/natural_context/final-choice-v1/natural_fit/manifest.json",
    "data/external/mecab-ko-dic/SOURCE.json",
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def as_pairs(surfaces) -> set[tuple[str, str]]:
    return {(romanize_word(surface), surface) for surface in surfaces}


def source_sets(generator: CandidateGenerator, analyses, train_words):
    seed = json.loads((ROOT / "data/morphology_lexicon.json").read_text())
    challenge = json.loads((ROOT / "data/challenge/train_lexicon.json").read_text())
    workbench = json.loads((ROOT / "data/workbench_lexicon.json").read_text())
    authored_surfaces = set(seed["nouns"].split()) | set(seed["forms"].split())
    authored_surfaces.update(item["text"] for item in challenge["entries"])
    authored_surfaces.update(word for words in workbench.values() for word in words)

    frequencies = get_frequency_dict("ko")
    frequency_surfaces = {
        word for word in top_n_list("ko", 50000)
        if family.HANGUL_FULL.fullmatch(word) and len(word) <= 12
    }
    frequency_fingerprint = hashlib.sha256(
        (generator.external_version + json.dumps(
            {word: frequencies[word] for word in frequency_surfaces},
            ensure_ascii=False, sort_keys=True,
        )).encode()
    ).hexdigest()

    direct_surfaces, direct_pos = set(), {}
    for name in FILES:
        if name == "Inflect":
            continue
        path = family.MECAB / f"{name}.csv"
        for row in csv.reader(path.read_text(encoding="utf-8").splitlines()):
            if len(row) >= 5 and HANGUL.fullmatch(row[0]):
                direct_surfaces.add(row[0])
                direct_pos.setdefault(row[0], set()).add(row[4])

    inflect_path = family.MECAB / "Inflect.csv"
    raw_rows = list(csv.reader(inflect_path.read_text(encoding="utf-8").splitlines()))
    siblings = {(row[0], row[11]) for row in raw_rows if len(row) >= 12}
    accepted_inflect = {
        row[0] for row in raw_rows
        if len(row) >= 12 and HANGUL.fullmatch(row[0])
        and not invalid_past_duplicate(row, siblings)
        and row[0] in analyses
    }

    synthetic_suffix = set()
    synthetic_ha = set()
    for noun in generator.nouns:
        for suffix in generator.suffixes:
            if suffix_allowed(noun, suffix):
                synthetic_suffix.add(noun + suffix)
        synthetic_ha.update(noun + ending for ending in ORIGINAL_ENDINGS)

    return {
        "train_observed": as_pairs(train_words),
        "explicit_authored": as_pairs(authored_surfaces),
        "external_frequency": as_pairs(frequency_surfaces),
        "mecab_direct": as_pairs(direct_surfaces),
        "inflect": as_pairs(accepted_inflect),
        "synthetic_noun_suffix": as_pairs(synthetic_suffix),
        "synthetic_noun_ha": as_pairs(synthetic_ha),
    }, direct_pos, frequency_fingerprint


def categorize_original(original, sources):
    independent = sources["external_frequency"] | sources["mecab_direct"] | sources["inflect"]
    overlap = {name: len(original & values) for name, values in sources.items()}
    overlap["independently_attested_union"] = len(original & independent)
    exclusive = {name: 0 for name in (
        "train_observed", "explicit_authored", "independently_attested",
        "unsupported_synthetic_suffix_only", "unsupported_synthetic_ha_only",
        "unsupported_synthetic_overlap", "unclassified",
    )}
    members = {name: [] for name in exclusive}
    category_pairs = {name: set() for name in exclusive}
    for pair in sorted(original):
        if pair in sources["train_observed"]:
            category = "train_observed"
        elif pair in sources["explicit_authored"]:
            category = "explicit_authored"
        elif pair in independent:
            category = "independently_attested"
        else:
            suffix = pair in sources["synthetic_noun_suffix"]
            noun_ha = pair in sources["synthetic_noun_ha"]
            category = (
                "unsupported_synthetic_overlap" if suffix and noun_ha else
                "unsupported_synthetic_suffix_only" if suffix else
                "unsupported_synthetic_ha_only" if noun_ha else "unclassified"
            )
        exclusive[category] += 1
        category_pairs[category].add(pair)
        if len(members[category]) < 30:
            members[category].append({"roman": pair[0], "surface": pair[1]})
    if sum(exclusive.values()) != len(original):
        raise ValueError("exclusive original provenance does not close")
    hashes = {name: family.pair_sha(values) for name, values in category_pairs.items()}
    return independent, overlap, exclusive, members, hashes


def nominal_ha_evidence(original, sources, direct_pos, analyses, generator):
    synthetic = original & sources["synthetic_noun_ha"]
    predicate_pos = {"VV", "VA", "VX", "VCP", "VCN", "XR"}
    analyzed_predicate_roots = {
        (lexical[1], lexical[2])
        for form_analyses in analyses.values() for analysis in form_analyses
        if (lexical := family.first_lexical(analysis)) is not None
        and lexical[2] in predicate_pos
    }
    base_nominal_attested = 0
    predicate_stem_attested = 0
    exact_surface_attested = 0
    for _, surface in synthetic:
        match = next((ending for ending in ORIGINAL_ENDINGS if surface.endswith(ending)), None)
        if match is None:
            continue
        noun = surface[:-len(match)]
        stem = noun + "하"
        base_nominal_attested += bool(direct_pos.get(noun, set()) & {"NNG", "NNP", "NP", "NR"})
        stem_direct = bool(direct_pos.get(stem, set()) & predicate_pos)
        stem_analysis = any((stem, pos) in analyzed_predicate_roots for pos in predicate_pos)
        predicate_stem_attested += stem_direct or stem_analysis
        exact_surface_attested += pair_in_any((romanize_word(surface), surface), (
            sources["mecab_direct"], sources["inflect"], sources["external_frequency"],
            sources["train_observed"], sources["explicit_authored"],
        ))
    return {
        "original_synthetic_noun_ha_pairs": len(synthetic),
        "base_noun_has_nominal_POS_attestation": base_nominal_attested,
        "noun_plus_ha_has_predicate_root_attestation": predicate_stem_attested,
        "exact_surface_observed_authored_or_independently_attested": exact_surface_attested,
        "interpretation": (
            "Nominal and predicate-root evidence is provenance only; it does not prove every generated ending is acceptable."
        ),
    }


def pair_in_any(pair, collections) -> bool:
    return any(pair in values for values in collections)


def split_coverage(pairs, analyses, aliases):
    ordered, components, family_keys, roots_by_pair, signatures, _ = family.family_components(
        pairs, analyses, aliases
    )
    rooted_pairs = {index for index, roots in roots_by_pair.items() if roots}
    rooted_components = sum(any(index in rooted_pairs for index in indices)
                            for indices in components.values())
    split = family.split_audit(pairs, analyses, aliases)

    # Recreate only the published deterministic role assignment to disclose
    # whether a morph holdout still has train examples in its closed family.
    component_for = {index: root for root, indices in components.items() for index in indices}
    lexical = {root for root, key in family_keys.items() if family.stable_bucket(key) == 0}
    combo = family.UnionFind()
    signature_owner, roman_owner, eligible = {}, {}, set()
    for index, (roman, _) in enumerate(ordered):
        if component_for[index] in lexical:
            continue
        combo.add(index)
        if roman in roman_owner:
            combo.union(index, roman_owner[roman])
        else:
            roman_owner[roman] = index
        for signature in signatures[index]:
            eligible.add(index)
            key = (component_for[index], signature)
            if key in signature_owner:
                combo.union(index, signature_owner[key])
            else:
                signature_owner[key] = index
    groups = {}
    for index in combo.parent:
        groups.setdefault(combo.find(index), []).append(index)
    morph_groups = set()
    for root, indices in groups.items():
        if any(index in eligible for index in indices):
            key = hashlib.sha256("\n".join(
                f"{ordered[index][0]}\t{ordered[index][1]}" for index in sorted(indices)
            ).encode()).hexdigest()
            if family.stable_bucket(key) == 1:
                morph_groups.add(root)
    roles = {}
    for index in range(len(ordered)):
        roles[index] = (
            "lexical_holdout" if component_for[index] in lexical else
            "morph_combination_holdout" if combo.find(index) in morph_groups else "train"
        )
    family_roles = {}
    for index, role in roles.items():
        family_roles.setdefault(component_for[index], set()).add(role)
    morph_pairs = [index for index, role in roles.items() if role == "morph_combination_holdout"]
    represented = sum("train" in family_roles[component_for[index]] for index in morph_pairs)
    return split, {
        "pairs_with_any_family_root": len(rooted_pairs),
        "pairs_without_family_root": len(ordered) - len(rooted_pairs),
        "components_with_rooted_members": rooted_components,
        "components_without_rooted_members": len(components) - rooted_components,
        "morph_holdout_pairs_with_train_in_same_family": represented,
        "morph_holdout_pairs_in_morph_only_family": len(morph_pairs) - represented,
    }


def source_hashes():
    paths = [ROOT / name for name in SOURCE_FILES]
    paths.extend(family.MECAB / f"{name}.csv" for name in FILES)
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in paths}


def build_report():
    before = source_hashes()
    generator = CandidateGenerator()
    analyses = family.read_inflect()
    rows, train_words = family.load_train_words()
    original, observed_new, anchored_new, full_union, _ = family.build_corpus(
        generator, analyses, train_words
    )
    sources, direct_pos, frequency_fingerprint = source_sets(generator, analyses, train_words)
    independent, overlap, exclusive, examples, exclusive_hashes = categorize_original(original, sources)
    supported_original = original & (sources["train_observed"] | sources["explicit_authored"] | independent)
    observed_all = {pair for word in train_words if (pair := family.eligible_pair(word)) is not None}
    alternative = supported_original | observed_all | anchored_new
    aliases = family.productive_aliases(generator, alternative)
    alternative_split, coverage = split_coverage(alternative, analyses, aliases)
    after = source_hashes()
    if before != after:
        raise ValueError("source-quality audit inputs changed")
    return {
        "schema": "phonetic-training-source-quality-v1",
        "completed": True,
        "scope": "TRAIN-only pair provenance and alternative-corpus evidence; no model activity",
        "original": {
            "pairs": len(original), "pairs_sha256": family.pair_sha(original),
            "overlapping_provenance_counts": overlap,
            "mutually_exclusive_counts": exclusive,
            "mutually_exclusive_pair_sha256": exclusive_hashes,
            "mutually_exclusive_examples": examples,
            "unsupported_synthetic_pairs": (
                exclusive["unsupported_synthetic_suffix_only"]
                + exclusive["unsupported_synthetic_ha_only"]
                + exclusive["unsupported_synthetic_overlap"]
            ),
            "unsupported_synthetic_pairs_sha256": family.pair_sha(
                original - supported_original
            ),
            "generation_trace": {
                "source": "hanglish/core/lattice.py",
                "authored_and_external_index": "CandidateGenerator.__init__ lines 86-147",
                "noun_suffix_and_ha_generation": "CandidateGenerator.__init__ lines 148-156",
                "checkpoint_lineage": "remove the later 360 noun+하기 pairs from the current index",
            },
        },
        "nominal_ha_evidence": nominal_ha_evidence(
            original, sources, direct_pos, analyses, generator
        ),
        "attested_only_alternative": {
            "policy": (
                "all eligible natural_fit TRAIN observed pairs + TRAIN-anchored exact accepted Inflect "
                "pairs + original explicitly authored or independently attested exact surfaces"
            ),
            "pairs": len(alternative), "pairs_sha256": family.pair_sha(alternative),
            "removed_original_unsupported_synthetic_pairs": len(original - supported_original),
            "split": alternative_split, "family_coverage": coverage,
            "feasible_split": (
                alternative_split["lexical_family_cross_split"] == 0
                and alternative_split["morph_signature_cross_train_holdout"] == 0
                and alternative_split["roman_input_cross_split"] == 0
                and alternative_split["counts"].get("lexical_holdout", 0) > 0
                and alternative_split["counts"].get("morph_combination_holdout", 0) > 0
            ),
        },
        "relationship_to_family_audit": {
            "full_union_pairs": len(full_union),
            "full_union_observed_additions": len(observed_new),
            "full_union_anchored_additions": len(anchored_new),
            "family_audit_did_not_validate_pair_correctness": True,
            "family_audit_did_not_approve_the_76674_union_for_training": True,
        },
        "limitations": [
            "An unattested synthetic form is not asserted to be linguistically invalid.",
            "Dictionary or frequency-list presence is evidence of independent attestation, not contextual grammaticality.",
            "This proposal remains data preparation evidence and does not authorize training or deployment.",
            "Any family-holdout model must be trained from scratch; the existing checkpoint contains held pairs.",
        ],
        "external": {
            "wordfreq_version": generator.external_version,
            "wordfreq_frequency_fingerprint": frequency_fingerprint,
            "runtime_external_fingerprint": generator.external_fingerprint,
            "mecab_dictionary_fingerprint": generator.dictionary_fingerprint,
        },
        "source_hashes": before, "source_hashes_after": after,
        "model_loaded": False, "model_inference_run": False, "training_run": False,
        "development_data_read": False, "reserve_data_read": False, "final_data_read": False,
    }


def main(output: Path = OUTPUT):
    if output.exists():
        raise FileExistsError(output)
    report = build_report()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    print(json.dumps({
        "original": report["original"]["pairs"],
        "unsupported_synthetic": report["original"]["unsupported_synthetic_pairs"],
        "alternative": report["attested_only_alternative"]["pairs"],
        "split": report["attested_only_alternative"]["split"]["counts"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    main(parser.parse_args().output)
