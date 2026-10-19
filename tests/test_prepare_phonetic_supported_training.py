import json

import pytest

import scripts.prepare_phonetic_supported_training as module


def flags(pairs):
    return {pair: {"fixture": True} for pair in pairs}


def test_records_preserve_roots_signatures_flags_and_family_membership(monkeypatch):
    pairs = {("ga", "가"), ("gayo", "가요"), ("na", "나")}
    analyses = {
        "가": {(('가', 'VV'),)},
        "가요": {(('가', 'VV'), ('요', 'EF'))},
    }
    monkeypatch.setattr(module.family, "stable_bucket", lambda value: 2)
    records, families = module.build_records(pairs, analyses, {}, flags(pairs))
    by_surface = {record["surface"]: record for record in records}
    assert by_surface["가요"]["family_roots"] == [{"surface": "가", "pos": "VV"}]
    assert by_surface["가요"]["morphology_signatures"] == [
        [{"surface": "요", "pos": "EF"}]
    ]
    assert by_surface["가요"]["source_flags"] == {"fixture": True}
    assert by_surface["나"]["known_family_root"] is False
    ga_family = next(value for value in families if len(value["pair_indices"]) == 2)
    assert ga_family["has_train"] is True
    assert ga_family["has_held"] is False


def test_same_roman_outputs_are_closed_into_one_split(monkeypatch):
    pairs = {("nun", "눈"), ("nun", "는")}
    analyses = {surface: {((surface, "NNG"),)} for _, surface in pairs}
    monkeypatch.setattr(module.family, "stable_bucket", lambda value: 0)
    records, _ = module.build_records(pairs, analyses, {}, flags(pairs))
    assert {record["split"] for record in records} == {"lexical_holdout"}
    assert len({record["family_sha256"] for record in records}) == 1


def test_morph_only_family_is_disclosed_without_reclassification(monkeypatch):
    pairs = {("gayo", "가요")}
    analyses = {"가요": {(('가', 'VV'), ('요', 'EF'))}}
    monkeypatch.setattr(module.family, "stable_bucket", lambda value: 1)
    records, families = module.build_records(pairs, analyses, {}, flags(pairs))
    assert records[0]["split"] == "morph_combination_holdout"
    assert records[0]["family_has_train"] is False
    assert families[0]["has_train"] is False
    assert families[0]["has_held"] is True


def test_prepare_refuses_existing_output(tmp_path):
    with pytest.raises(FileExistsError):
        module.prepare(tmp_path)


def test_manifest_contract_uses_reviewed_limits():
    assert module.EXPECTED_PAIRS == 54676
    assert module.EXPECTED_SPLIT == {
        "train": 51361,
        "lexical_holdout": 2552,
        "morph_combination_holdout": 763,
    }
    assert "TRAIN-only" in module.quality_report_policy()


def test_inherited_audit_pin_is_not_replaced_by_current_hash():
    report = json.loads(module.QUALITY_REPORT.read_text())
    report["source_hashes"]["scripts/audit_phonetic_pair_sources.py"] = "0" * 64
    pins = module.reviewed_pins(report)
    assert pins["scripts/audit_phonetic_pair_sources.py"] == "0" * 64
    with pytest.raises(ValueError, match="source changed"):
        module.verify_pins(pins)


def test_external_provenance_requires_every_reviewed_fingerprint():
    reviewed = {
        "wordfreq_version": "v", "wordfreq_frequency_fingerprint": "frequency",
        "runtime_external_fingerprint": "runtime", "mecab_dictionary_fingerprint": "mecab",
    }
    module.verify_external_provenance(reviewed, dict(reviewed))
    changed = dict(reviewed, runtime_external_fingerprint="changed")
    with pytest.raises(ValueError, match="external provenance"):
        module.verify_external_provenance(reviewed, changed)


def test_train_observation_flag_is_literal_source_evidence_not_eligibility():
    pair = ("e", "에")
    sources = {
        "train_observed": {pair}, "explicit_authored": set(),
        "external_frequency": set(), "mecab_direct": set(), "inflect": set(),
    }
    value = module.source_flags(pair, {pair}, set(), sources, set())
    assert module.family.eligible_pair("에") is None
    assert value["natural_train_observed"] is True
