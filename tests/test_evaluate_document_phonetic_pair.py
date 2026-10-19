import hashlib
import json
import math

import pytest

from scripts import evaluate_document_phonetic_pair as module


def prediction(source, outputs):
    return {"id": hashlib.sha256(source.encode()).hexdigest(), "input": source,
            "outputs": [{"text": text, "score": score} for text, score in outputs],
            "latency_s": 0.01}


def references():
    occurrences = [("same", "가"), ("same", "가"), ("same", "나"),
                   ("delta", "다"), ("unsupported!", "라")]
    counts = module.Counter(occurrences)
    return {"occurrences": occurrences, "pair_counts": counts,
            "variants": {"same": {"가", "나"}, "delta": {"다"},
                         "unsupported!": {"라"}},
            "supported_inputs": {"same", "delta"}, "source_hashes": {}}


def strata():
    return {"control_pairs": {("same", "가")}, "delta_pairs": {("delta", "다")},
            "control_romans": {"same"}, "delta_romans": {"delta"}}


def test_metrics_count_any_variant_all_spellings_occurrences_and_strata():
    rows = [prediction("same", [("가", -0.1), ("나", -0.2)]),
            prediction("delta", [])]
    result = module.score_arm(rows, references(), strata())
    assert result["roman_acceptable_variant"]["acceptable_variant_top1"] == {
        "correct": 1, "denominator": 3, "rate": 1 / 3}
    assert result["roman_acceptable_variant"]["acceptable_variant_top8"]["correct"] == 1
    assert result["all_spelling_variants"]["variant_recall_top1"]["correct"] == 1
    assert result["all_spelling_variants"]["variant_recall_top8"] == {
        "correct": 2, "denominator": 4, "rate": 0.5}
    assert result["all_spelling_variants"]["occurrence_weighted_top8"] == {
        "correct": 3, "denominator": 5, "rate": 0.6}
    assert result["empty_supported_outputs"] == 1
    assert result["unsupported"] == {"roman_keys": 1, "reference_pairs": 1,
                                      "occurrences": 1}
    assert result["by_exact_pair_stratum"]["seen_control"]["pairs"] == 1
    assert result["by_exact_pair_stratum"]["unseen_augmented"]["pairs"] == 2
    assert result["by_roman_key_stratum"]["delta_only"]["roman_keys"] == 1


def test_input_length_metrics_keep_unsupported_and_multioutput_denominators():
    rows = [prediction("same", [("가", -0.1), ("나", -0.2)]),
            prediction("delta", [])]
    by_length = module.score_arm(rows, references(), strata())["by_input_length"]

    assert set(by_length) == {"4", "5", "12"}
    assert by_length["4"]["roman_acceptable_variant"]["acceptable_variant_top1"] == {
        "correct": 1, "denominator": 1, "rate": 1.0}
    assert by_length["4"]["all_spelling_variants"]["variant_recall_top8"] == {
        "correct": 2, "denominator": 2, "rate": 1.0}
    assert by_length["4"]["all_spelling_variants"]["occurrence_weighted_top8"] == {
        "correct": 3, "denominator": 3, "rate": 1.0}
    assert by_length["5"]["roman_acceptable_variant"]["acceptable_variant_top8"] == {
        "correct": 0, "denominator": 1, "rate": 0.0}
    assert by_length["12"]["roman_acceptable_variant"]["acceptable_variant_top8"] == {
        "correct": 0, "denominator": 1, "rate": 0.0}
    assert by_length["12"]["all_spelling_variants"]["occurrence_weighted_top8"] == {
        "correct": 0, "denominator": 1, "rate": 0.0}


def test_output_validation_requires_unique_hangul_forward_valid_finite_descending():
    romanize = {"가": "ga", "나": "na"}.__getitem__
    assert module.validate_outputs("ga", [{"text": "가", "score": -1.0}], romanize)
    invalid = [
        [{"text": "가", "score": math.inf}],
        [{"text": "가", "score": -2.0}, {"text": "가", "score": -3.0}],
        [{"text": "가", "score": -2.0}, {"text": "나", "score": -1.0}],
        [{"text": "나", "score": -1.0}],
        [{"text": "latin", "score": -1.0}],
        [{"text": "가", "score": -1.0, "rank": 1}],
    ]
    for outputs in invalid:
        with pytest.raises(ValueError):
            module.validate_outputs("ga", outputs, romanize)
    with pytest.raises(ValueError, match="beam8"):
        module.validate_outputs("ga", [{"text": "가", "score": -float(i)}
                                        for i in range(9)], romanize)


def test_capture_authentication_rejects_missing_duplicate_and_bad_hash(tmp_path, monkeypatch):
    requests = [{"id": hashlib.sha256(b"ga").hexdigest(), "input": "ga"}]
    directory = tmp_path / "capture"; directory.mkdir()
    predictions = directory / "predictions.jsonl"
    predictions.write_text(json.dumps(prediction("ga", [])) + "\n")
    source = tmp_path / "source"; source.write_text("pinned")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    report = {"schema": "document-phonetic-capture-v1", "completed": True,
              "arm": "control", "input_manifest_sha256": module.INPUT_MANIFEST_SHA256,
              "inputs_sha256": module.INPUTS_SHA256,
              "predictions_sha256": module.sha256_file(predictions),
              "checkpoint_sha256": module.sha256_file(source),
              "training_report_sha256": module.sha256_file(source),
              "paired_initial_checkpoint_sha256": module.sha256_file(source),
              "prediction_count": 1, "beam_size": 8, "reference_targets_read": False,
              "source_hashes": {"source": module.sha256_file(source)},
              "limits": module.LIMITS,
              "final_resource": {"elapsed_s": 1.0, "rss_bytes": 1,
                                 "swap_growth_bytes": 0}}
    (directory / "report.json").write_bytes(module.canonical(report) + b"\n")
    completion = {"schema": "observed-phonetic-pair-completion-v1", "completed": True,
                  "report_sha256": module.sha256_file(directory / "report.json"),
                  "resource_after_report": {"elapsed_s": 2.0, "rss_bytes": 1,
                                             "swap_growth_bytes": 0}}
    (directory / "completion.json").write_bytes(module.canonical(completion) + b"\n")
    assert module.authenticate_capture(directory, "control", requests)["predictions"][0]["input"] == "ga"

    predictions.write_text(json.dumps(prediction("ga", [])) + "\n ")
    with pytest.raises(ValueError, match="prediction hash"):
        module.authenticate_capture(directory, "control", requests)

    predictions.write_text("")
    report["predictions_sha256"] = module.sha256_file(predictions)
    (directory / "report.json").write_bytes(module.canonical(report) + b"\n")
    completion["report_sha256"] = module.sha256_file(directory / "report.json")
    (directory / "completion.json").write_bytes(module.canonical(completion) + b"\n")
    with pytest.raises(ValueError, match="missing inputs"):
        module.authenticate_capture(directory, "control", requests)

    duplicate = json.dumps(prediction("ga", [])) + "\n"
    predictions.write_text(duplicate + duplicate)
    report["prediction_count"] = 2
    report["predictions_sha256"] = module.sha256_file(predictions)
    (directory / "report.json").write_bytes(module.canonical(report) + b"\n")
    completion["report_sha256"] = module.sha256_file(directory / "report.json")
    (directory / "completion.json").write_bytes(module.canonical(completion) + b"\n")
    with pytest.raises(ValueError):
        module.authenticate_capture(directory, "control", requests * 2)


def test_run_never_opens_references_until_both_captures_authenticate(tmp_path):
    requests = ([{"id": hashlib.sha256(b"ga").hexdigest(), "input": "ga"}], {})
    events = []
    def authenticate(path, arm, rows):
        events.append(arm)
        if arm == "augmented":
            raise ValueError("capture rejected")
        return {"report": {"paired_initial_checkpoint_sha256": "a" * 64},
                "predictions": [], "report_sha256": "r", "completion_sha256": "c",
                "predictions_sha256": "p", "arm": arm}
    def references_opened():
        events.append("references")
        raise AssertionError("references opened early")
    with pytest.raises(ValueError, match="capture rejected"):
        module.run(tmp_path / "c", tmp_path / "a", tmp_path / "out",
                   request_loader=lambda: requests, capture_authenticator=authenticate,
                   reference_loader=references_opened)
    assert events == ["control", "augmented"]
    assert not (tmp_path / "out").exists()


def test_run_emits_aggregate_report_without_examples_or_gate(tmp_path):
    reqs = [{"id": hashlib.sha256(x.encode()).hexdigest(), "input": x}
            for x in ("same", "delta")]
    captures = {
        "control": [prediction("same", [("가", -1.0)]), prediction("delta", [])],
        "augmented": [prediction("same", [("가", -1.0)]), prediction("delta", [("다", -1.0)])],
    }
    def authenticate(path, arm, rows):
        path.mkdir()
        for name in ("report.json", "completion.json", "predictions.jsonl"):
            (path / name).write_text(arm + "-" + name)
        return {"arm": arm, "report": {"paired_initial_checkpoint_sha256": "i" * 64,
                "checkpoint_sha256": arm[0] * 64, "training_report_sha256": arm[-1] * 64},
                "predictions": captures[arm],
                "report_sha256": module.sha256_file(path / "report.json"),
                "completion_sha256": module.sha256_file(path / "completion.json"),
                "predictions_sha256": module.sha256_file(path / "predictions.jsonl")}
    output = tmp_path / "evaluation"
    result = module.run(tmp_path / "control", tmp_path / "augmented", output,
        request_loader=lambda: (reqs, {"counts": {"reference_occurrences": 5}}),
        capture_authenticator=authenticate, reference_loader=references,
        strata_loader=strata)
    assert result["completed"] and result["quality_gate_applied"] is False
    assert result["checkpoint_selected"] is False
    assert result["metrics"]["augmented"]["roman_acceptable_variant"][
        "acceptable_variant_top1"]["correct"] == 2
    serialized=(output / "report.json").read_text()
    assert '"examples"' not in serialized and '"errors"' not in serialized


def test_parser_requires_both_captures_and_output():
    args = module.parser().parse_args(["--control-capture", "c", "--augmented-capture", "a",
                                       "--output", "o"])
    assert str(args.control_capture) == "c" and str(args.augmented_capture) == "a"
