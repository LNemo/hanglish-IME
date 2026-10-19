#!/usr/bin/env python3
"""Serialize the reviewed TRAIN-only phonetic pair corpus without training a model."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hanglish.core.lattice import CandidateGenerator
import scripts.audit_phonetic_pair_sources as quality
import scripts.audit_phonetic_training_families as family


OUTPUT = ROOT / ".omx/artifacts/phonetic-supported-data-v1"
QUALITY_REPORT = ROOT / ".omx/artifacts/phonetic-training-source-quality-v1.json"
EXPECTED_PAIRS = 54676
EXPECTED_PAIR_SHA256 = "5764355e6bfebc1451a961fe6a6d3998902394888e0fec35e0bd23f160828ccf"
EXPECTED_SPLIT = {"train": 51361, "lexical_holdout": 2552, "morph_combination_holdout": 763}
QUALITY_REPORT_SHA256 = "faf67c089d1ccf2fe7ca896edcf420fc4c5729a85cb85e3ae4b94557f22daf30"


def canonical(value) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pair_digest(pairs) -> str:
    return family.pair_sha(pairs)


def _split_assignment(pairs, analyses, aliases):
    ordered, components, family_keys, roots, signatures, _ = family.family_components(
        pairs, analyses, aliases
    )
    component_for = {
        index: component for component, indices in components.items() for index in indices
    }
    lexical = {
        component for component, key in family_keys.items() if family.stable_bucket(key) == 0
    }
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
    combo_groups = defaultdict(list)
    for index in combo.parent:
        combo_groups[combo.find(index)].append(index)
    morph_groups = set()
    for root, indices in combo_groups.items():
        if not any(index in eligible for index in indices):
            continue
        key = hashlib.sha256("\n".join(
            f"{ordered[index][0]}\t{ordered[index][1]}" for index in sorted(indices)
        ).encode()).hexdigest()
        if family.stable_bucket(key) == 1:
            morph_groups.add(root)
    roles = {
        index: (
            "lexical_holdout" if component_for[index] in lexical else
            "morph_combination_holdout" if combo.find(index) in morph_groups else
            "train"
        )
        for index in range(len(ordered))
    }
    return ordered, components, family_keys, roots, signatures, component_for, roles


def build_records(pairs, analyses, aliases, source_flags):
    (ordered, components, family_keys, roots, signatures,
     component_for, roles) = _split_assignment(pairs, analyses, aliases)
    family_roles = {
        component: sorted({roles[index] for index in indices})
        for component, indices in components.items()
    }
    records = []
    for index, (roman, surface) in enumerate(ordered):
        component = component_for[index]
        records.append({
            "index": index,
            "roman": roman,
            "surface": surface,
            "split": roles[index],
            "family_sha256": family_keys[component],
            "family_splits": family_roles[component],
            "family_has_train": "train" in family_roles[component],
            "known_family_root": bool(roots[index]),
            "family_roots": [
                {"surface": root, "pos": pos} for root, pos in sorted(roots[index])
            ],
            "morphology_signatures": [
                [{"surface": part, "pos": pos} for part, pos in signature]
                for signature in sorted(signatures[index])
            ],
            "source_flags": dict(sorted(source_flags[(roman, surface)].items())),
        })
    family_records = []
    for component, indices in sorted(components.items(), key=lambda item: family_keys[item[0]]):
        family_records.append({
            "family_sha256": family_keys[component],
            "pair_indices": sorted(indices),
            "splits": family_roles[component],
            "has_train": "train" in family_roles[component],
            "has_held": any(role != "train" for role in family_roles[component]),
            "has_known_root": any(roots[index] for index in indices),
        })
    return records, family_records


def build_dataset():
    quality_raw = QUALITY_REPORT.read_bytes()
    if hashlib.sha256(quality_raw).hexdigest() != QUALITY_REPORT_SHA256:
        raise ValueError("reviewed source-quality report hash differs")
    quality_report = json.loads(quality_raw)
    if (quality_report.get("schema") != "phonetic-training-source-quality-v1"
            or quality_report["attested_only_alternative"].get("pairs") != EXPECTED_PAIRS
            or quality_report["attested_only_alternative"].get("pairs_sha256")
            != EXPECTED_PAIR_SHA256):
        raise ValueError("reviewed source-quality report does not match the supported recipe")
    pins = reviewed_pins(quality_report)
    verify_pins(pins)

    generator = CandidateGenerator()
    analyses = family.read_inflect()
    _, train_words = family.load_train_words()
    original, _, anchored_new, _, _ = family.build_corpus(generator, analyses, train_words)
    sources, _, frequency_fingerprint = quality.source_sets(generator, analyses, train_words)
    verify_external_provenance(quality_report["external"], {
        "wordfreq_version": generator.external_version,
        "wordfreq_frequency_fingerprint": frequency_fingerprint,
        "runtime_external_fingerprint": generator.external_fingerprint,
        "mecab_dictionary_fingerprint": generator.dictionary_fingerprint,
    })
    independent = sources["external_frequency"] | sources["mecab_direct"] | sources["inflect"]
    supported_original = original & (
        sources["train_observed"] | sources["explicit_authored"] | independent
    )
    observed = {
        pair for word in train_words if (pair := family.eligible_pair(word)) is not None
    }
    pairs = supported_original | observed | anchored_new
    if len(pairs) != EXPECTED_PAIRS or pair_digest(pairs) != EXPECTED_PAIR_SHA256:
        raise ValueError("supported pair corpus differs from reviewed quality audit")

    flags = {
        pair: source_flags(pair, original, anchored_new, sources, independent)
        for pair in pairs
    }
    aliases = family.productive_aliases(generator, pairs)
    records, families = build_records(pairs, analyses, aliases, flags)
    counts = {
        name: sum(record["split"] == name for record in records)
        for name in EXPECTED_SPLIT
    }
    if counts != EXPECTED_SPLIT:
        raise ValueError("supported corpus split differs from reviewed quality audit")
    roman_roles = defaultdict(set)
    for record in records:
        roman_roles[record["roman"]].add(record["split"])
    if any(len(values) != 1 for values in roman_roles.values()):
        raise ValueError("a Roman input crosses supported corpus splits")
    verify_pins(pins)
    return records, families, dict(sorted(pins.items()))


def reviewed_pins(quality_report):
    pins = dict(quality_report["source_hashes"])
    pins[str(QUALITY_REPORT.relative_to(ROOT))] = QUALITY_REPORT_SHA256
    pins["scripts/prepare_phonetic_supported_training.py"] = sha256(Path(__file__))
    return pins


def source_flags(pair, original, anchored_new, sources, independent):
    return {
        "original_checkpoint": pair in original,
        "natural_train_observed": pair in sources["train_observed"],
        "train_anchored_inflect_addition": pair in anchored_new,
        "explicit_authored": pair in sources["explicit_authored"],
        "external_frequency": pair in sources["external_frequency"],
        "mecab_direct": pair in sources["mecab_direct"],
        "inflect": pair in sources["inflect"],
        "independently_attested": pair in independent,
    }


def verify_external_provenance(reviewed, current):
    names = {
        "wordfreq_version", "wordfreq_frequency_fingerprint",
        "runtime_external_fingerprint", "mecab_dictionary_fingerprint",
    }
    if set(current) != names or any(reviewed.get(name) != current[name] for name in names):
        raise ValueError("external provenance differs from reviewed source-quality audit")


def verify_pins(pins):
    for name, expected in pins.items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"supported-data source changed: {name}")


def write_jsonl(path: Path, values) -> str:
    raw = b"".join(canonical(value) + b"\n" for value in values)
    with path.open("xb") as stream:
        stream.write(raw)
    return hashlib.sha256(raw).hexdigest()


def prepare(output: Path = OUTPUT):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    records, families, pins = build_dataset()
    output.mkdir(parents=True)
    pairs_sha = write_jsonl(output / "pairs.jsonl", records)
    families_sha = write_jsonl(output / "families.jsonl", families)
    split_counts = {
        name: sum(record["split"] == name for record in records)
        for name in EXPECTED_SPLIT
    }
    root_counts = {
        name: {
            "known_root": sum(record["split"] == name and record["known_family_root"]
                              for record in records),
            "root_unknown": sum(record["split"] == name and not record["known_family_root"]
                                for record in records),
        }
        for name in EXPECTED_SPLIT
    }
    morph_only = sum(
        record["split"] == "morph_combination_holdout" and not record["family_has_train"]
        for record in records
    )
    manifest = {
        "schema": "phonetic-supported-data-v1",
        "completed": True,
        "scope": "TRAIN-only reviewed source-supported pair serialization; no model activity",
        "pairs": len(records),
        "pair_set_sha256": pair_digest((r["roman"], r["surface"]) for r in records),
        "pairs_jsonl_sha256": pairs_sha,
        "families": len(families),
        "families_jsonl_sha256": families_sha,
        "split_counts": split_counts,
        "root_coverage_by_split": root_counts,
        "morph_holdout_pairs_without_train_family_member": morph_only,
        "roman_input_cross_split": 0,
        "policy": quality_report_policy(),
        "source_hashes": pins,
        "limitations": [
            "Root-unknown records do not establish that related forms are absent.",
            "The 23 morph-held records without a train member in their known family remain held and are not reclassified.",
            "Word-frequency and dictionary presence are attestation evidence, not proof of grammaticality.",
            "Explicitly authored records can still contain errors and require downstream quality controls.",
            "This dataset does not authorize model training, evaluation, or deployment.",
            "A family-holdout evaluation requires training a model from scratch.",
        ],
        "model_loaded": False,
        "model_inference_run": False,
        "training_run": False,
        "development_data_read": False,
        "reserve_data_read": False,
        "final_data_read": False,
    }
    (output / "manifest.json").write_bytes(canonical(manifest) + b"\n")
    verify_pins(pins)
    return manifest


def quality_report_policy():
    return (
        "TRAIN-only: eligible natural_fit TRAIN observed pairs, TRAIN-anchored exact accepted Inflect pairs, "
        "and original pairs with exact authored or independent frequency/MeCab/Inflect attestation"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output), ensure_ascii=False, sort_keys=True))
