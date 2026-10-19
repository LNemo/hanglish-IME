import scripts.audit_phonetic_pair_sources as module
import scripts.audit_phonetic_training_families as family


def test_exclusive_categories_close_without_calling_unattested_invalid():
    original = {("a", "가"), ("b", "나"), ("c", "다"), ("d", "라"), ("e", "마")}
    sources = {
        "train_observed": {("a", "가")},
        "explicit_authored": {("a", "가"), ("b", "나")},
        "external_frequency": {("c", "다")},
        "mecab_direct": set(), "inflect": set(),
        "synthetic_noun_suffix": {("d", "라")},
        "synthetic_noun_ha": {("e", "마")},
    }
    _, _, exclusive, _, hashes = module.categorize_original(original, sources)
    assert exclusive == {
        "train_observed": 1, "explicit_authored": 1, "independently_attested": 1,
        "unsupported_synthetic_suffix_only": 1, "unsupported_synthetic_ha_only": 1,
        "unsupported_synthetic_overlap": 0, "unclassified": 0,
    }
    assert set(hashes) == set(exclusive)


def test_multiple_roots_and_productive_alias_union_on_one_surface():
    pairs = {("georeo", "걸어"), ("geotda", "걷다"), ("georeul", "걸을")}
    analyses = {
        "걸어": {(("걷", "VV"), ("어", "EC")), (("걸", "NNG"), ("어", "JX"))},
        "걷다": {(("걷", "VV"), ("다", "EF"))},
    }
    aliases = {"걸어": {"걸"}, "걸을": {"걸"}}
    _, components, _, roots, _, _ = family.family_components(pairs, analyses, aliases)
    assert len(components) == 1
    assert any({("걷", "VV"), ("걸", "NNG"), ("걸", "AUTHORED")} <= values
               for values in roots.values())


def test_repository_quality_report_closes_and_stays_train_only():
    report = module.build_report()
    original = report["original"]
    assert original["pairs"] == 50384
    assert sum(original["mutually_exclusive_counts"].values()) == 50384
    assert report["relationship_to_family_audit"]["family_audit_did_not_approve_the_76674_union_for_training"]
    alternative = report["attested_only_alternative"]
    assert alternative["pairs"] < 76674
    assert alternative["feasible_split"] is True
    assert alternative["split"]["roman_input_cross_split"] == 0
    assert alternative["family_coverage"]["pairs_with_any_family_root"] > 0
    assert report["model_loaded"] is False
    assert report["development_data_read"] is False
    assert report["reserve_data_read"] is False
    assert report["final_data_read"] is False
