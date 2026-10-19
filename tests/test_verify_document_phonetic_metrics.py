import hashlib

from scripts import verify_document_phonetic_metrics as verifier


def row(source, outputs):
    return {"id": hashlib.sha256(source.encode()).hexdigest(), "input": source,
            "outputs": [{"text": text, "score": -float(index + 1)}
                        for index, text in enumerate(outputs)],
            "latency_s": 0.01}


def fixture():
    occurrences = [("same", "가"), ("same", "가"), ("same", "나"),
                   ("delta", "다"), ("unsupported!", "라")]
    counts = verifier.Counter(occurrences)
    references = {
        "pair_counts": counts,
        "variants": {"same": {"가", "나"}, "delta": {"다"},
                     "unsupported!": {"라"}},
        "supported_inputs": {"same", "delta"},
    }
    strata = {
        "control_pairs": {("same", "가")},
        "delta_pairs": {("delta", "다")},
        "control_sources": {"same"}, "delta_sources": {"delta"},
    }
    return references, strata


def test_independent_metrics_retain_all_spellings_occurrences_and_unsupported():
    references, strata = fixture()
    result = verifier.compute_metrics(
        [row("same", ["가", "나"]), row("delta", [])], references, strata)
    assert result["roman_acceptable_variant"]["acceptable_variant_top1"] == {
        "correct": 1, "denominator": 3, "rate": 1 / 3}
    assert result["all_spelling_variants"]["variant_recall_top8"] == {
        "correct": 2, "denominator": 4, "rate": 0.5}
    assert result["all_spelling_variants"]["occurrence_weighted_top8"] == {
        "correct": 3, "denominator": 5, "rate": 0.6}
    assert result["unsupported"] == {
        "roman_keys": 1, "reference_pairs": 1, "occurrences": 1}
    assert result["by_exact_pair_stratum"]["unseen_augmented"]["pairs"] == 2


def test_exact_length_metrics_include_missing_unsupported_key():
    references, strata = fixture()
    result = verifier.compute_metrics(
        [row("same", ["가", "나"]), row("delta", [])], references, strata)
    lengths = result["by_input_length"]
    assert set(lengths) == {"4", "5", "12"}
    assert lengths["4"]["all_spelling_variants"]["occurrence_weighted_top8"] == {
        "correct": 3, "denominator": 3, "rate": 1.0}
    assert lengths["12"]["roman_acceptable_variant"]["acceptable_variant_top8"] == {
        "correct": 0, "denominator": 1, "rate": 0.0}
    assert lengths["12"]["all_spelling_variants"]["variant_recall_top8"] == {
        "correct": 0, "denominator": 1, "rate": 0.0}


def test_supported_capture_set_must_match_reference_projection():
    references, strata = fixture()
    try:
        verifier.compute_metrics([row("same", ["가"])], references, strata)
    except ValueError as error:
        assert "supported inputs differ" in str(error)
    else:
        raise AssertionError("missing supported capture input was accepted")


def test_parser_requires_closed_captures_evaluation_and_output():
    args = verifier.parser().parse_args([
        "--evaluation", "eval", "--control-capture", "control",
        "--augmented-capture", "augmented", "--output", "proof.json"])
    assert str(args.evaluation) == "eval" and str(args.output) == "proof.json"
