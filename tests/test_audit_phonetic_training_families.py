import scripts.audit_phonetic_training_families as module


def test_homonym_normalization_is_trailing_only():
    assert module.normalize_lexeme("단편3") == "단편"
    assert module.normalize_lexeme("3단편") == "3단편"
    assert module.normalize_lexeme("단3편") == "단3편"


def test_multiple_analyses_union_every_lexical_root_and_signature():
    analyses = {
        "걸어": {
            (("걷", "VV"), ("어", "EC")),
            (("걸", "NNG"), ("어", "JX")),
        }
    }
    roots, signatures = module.analysis_roots_and_signatures("걸어", analyses, {})
    assert roots == {("걷", "VV"), ("걸", "NNG")}
    assert signatures == {(("어", "EC"),), (("어", "JX"),)}


def test_family_split_closes_roots_signatures_and_same_input_collisions():
    pairs = {
        ("gana", "가나"), ("gana", "간아"),
        ("gada", "가다"), ("gago", "가고"),
        ("nada", "나다"), ("nago", "나고"),
    }
    analyses = {
        "가다": {(("가", "VV"), ("다", "EF"))},
        "가고": {(("가", "VV"), ("고", "EC"))},
        "나다": {(("나", "VV"), ("다", "EF"))},
        "나고": {(("나", "VV"), ("고", "EC"))},
    }
    result = module.split_audit(pairs, analyses, {})
    assert result["families"] < len(pairs)
    assert result["roman_input_cross_split"] == 0
    assert result["lexical_family_cross_split"] == 0
    assert result["morph_signature_cross_train_holdout"] == 0
    assert result["ambiguous_roman_keys"] == 1


def test_repository_corpus_reproduces_frozen_pair_contract():
    generator = module.CandidateGenerator()
    analyses = module.read_inflect()
    rows, words = module.load_train_words()
    original, observed, anchored, union, roots = module.build_corpus(generator, analyses, words)
    assert len(rows) == 2048
    assert len(original) == 50384
    assert module.pair_sha(original) == module.ORIGINAL_SHA256
    assert len(observed) == 13837
    assert len(anchored) == 12453
    assert len(union) == 76674
    assert roots


def test_report_requires_from_scratch_training_and_has_no_eval_access():
    report = module.build_report()
    assert report["training_requirement"].startswith("train the transducer from scratch")
    assert report["model_loaded"] is False
    assert report["model_inference_run"] is False
    assert report["training_run"] is False
    assert report["development_data_read"] is False
    assert report["reserve_data_read"] is False
    assert report["final_data_read"] is False
    assert report["split"]["roman_input_cross_split"] == 0
