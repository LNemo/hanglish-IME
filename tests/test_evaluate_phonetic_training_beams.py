import json

import pytest

from scripts import evaluate_phonetic_training_beams as module


def write_jsonl(path, rows):
    path.write_bytes(b"".join(module.canonical(row) + b"\n" for row in rows))
    return module.sha(path)


def piece(source, start, *texts):
    return {"source": source, "start": start, "end": start + len(source),
            "baseline": [{"text": text} for text in texts], "proposals": []}


def closed_fixture(tmp_path, *, completed=True):
    tmp_path.mkdir(parents=True, exist_ok=True)
    mechanical = tmp_path / "mechanical.jsonl"
    rows = [{"id": "r1", "input": "gana!",
             "pieces": [piece("gana", 0, "가너"), piece("!", 4, "!")]}]
    mechanical_sha = write_jsonl(mechanical, rows)
    mechanical_report = tmp_path / "mechanical-report.json"
    mechanical_report.write_text("fixed\n")
    census = tmp_path / "census"
    census.mkdir()
    selection = [{"index": 0, "input": "gana", "row_ids": ["r1"],
                  "protected_row_ids": [], "unprotected_row_ids": ["r1"]}]
    capture = [{"index": 0, "input": "gana",
                "outputs": [{"text": "가나", "score": -0.1}]}]
    selection_sha = write_jsonl(census / "selection.jsonl", selection)
    capture_sha = write_jsonl(census / "capture.jsonl", capture)
    dependency = tmp_path / "dependency"
    dependency.write_text("fixed")
    pins = {str(dependency): module.sha(dependency)}
    report = {
        "schema": "phonetic-training-beam-census-v1", "completed": completed,
        "source_role": "natural_fit", "source_rows": 1, "unique_inputs": 1,
        "captured_inputs": 1, "target_fields_read": False,
        "model_inference_run": True, "training_run": False,
        "source_hashes_unchanged": True, "profile": {"beam_size": 8},
        "selection_order": (
            "256 evenly spaced quantiles of (length,input), including endpoints; "
            "then remaining (length,input) order"),
        "validation": {"all_selected_inputs_captured_once": True,
                       "outputs_are_unique_forward_valid_hangul": True,
                       "scores_are_finite_non_increasing": True,
                       "maximum_outputs_per_input": 8},
        "source_hashes_start": pins, "source_hashes_end": pins,
        "selection_sha256": selection_sha, "capture_sha256": capture_sha,
        "ordered_inputs_sha256": module.digest_json(["gana"]),
        "outputs": 1, "empty_outputs": 0,
    }
    report["selection"] = module.independent_selection_summary(rows, selection)
    (census / "report.json").write_text(json.dumps(report))
    return census, module.sha(census / "report.json"), mechanical, mechanical_sha, mechanical_report


def validate_fixture(values):
    census, report_sha, mechanical, mechanical_sha, mechanical_report = values
    return module.validate_closed_capture(
        census, report_sha=report_sha, mechanical=mechanical,
        mechanical_sha=mechanical_sha, mechanical_report=mechanical_report,
        mechanical_report_sha=module.sha(mechanical_report), expected_rows=1,
        expected_inputs=1, pilot_rows=1)


def test_closed_capture_validates_membership_hashes_and_forward_beams(tmp_path):
    closed = validate_fixture(closed_fixture(tmp_path))
    assert closed["eligible_occurrences"] == 1
    assert closed["beams"]["gana"]["unprotected_row_ids"] == {"r1"}


def test_unclosed_or_corrupt_capture_is_rejected_before_target_join(tmp_path):
    values = closed_fixture(tmp_path, completed=False)
    with pytest.raises(ValueError, match="incomplete"):
        validate_fixture(values)

    values = closed_fixture(tmp_path / "second")
    (values[0] / "capture.jsonl").write_text("corrupt\n")
    with pytest.raises(ValueError, match="payload hash"):
        validate_fixture(values)

    values = closed_fixture(tmp_path / "third")
    (values[0] / "failure.json").write_text("{}")
    with pytest.raises(ValueError, match="failed census"):
        validate_fixture(values)


def test_evaluate_rejects_unclosed_report_before_reading_targets(tmp_path, monkeypatch):
    census = tmp_path / "census"
    census.mkdir()
    (census / "report.json").write_text("{}")
    target_reads = []
    monkeypatch.setattr(module, "load_targets", lambda: target_reads.append(True))
    with pytest.raises(ValueError, match="closed census report"):
        module.evaluate(census, tmp_path / "result.json",
                        report_sha="PENDING_AFTER_CLOSED_CAPTURE")
    assert target_reads == []


def test_closed_payloads_are_each_read_once_for_hash_and_parse(tmp_path, monkeypatch):
    values = closed_fixture(tmp_path)
    watched = {values[0] / "report.json", values[0] / "selection.jsonl",
               values[0] / "capture.jsonl", values[2], values[4]}
    counts = {path: 0 for path in watched}
    original = module.Path.read_bytes

    def counted(path):
        if path in counts:
            counts[path] += 1
        return original(path)

    monkeypatch.setattr(module.Path, "read_bytes", counted)
    validate_fixture(values)
    assert set(counts.values()) == {1}


def test_source_pin_corruption_is_rejected(tmp_path):
    values = closed_fixture(tmp_path)
    report = json.loads((values[0] / "report.json").read_text())
    dependency = next(iter(report["source_hashes_start"]))
    open(dependency, "w").write("changed")
    with pytest.raises(ValueError, match="source pin"):
        validate_fixture(values)


def test_forward_alignment_preserves_punctuation_and_mixed_english():
    aligned = module.forward_alignment("gana, testreul!", "가나, test를!")
    assert aligned == [{"text": "가나", "start": 0, "end": 4},
                       {"text": "를", "start": 10, "end": 14}]
    pieces = [piece("gana", 0, "가나"), piece(", ", 4, ", "),
              piece("testreul", 6, "test를"), piece("!", 14, "!")]
    beams = {"gana": {"outputs": [{"text": "가나", "score": 0.0}],
                       "protected_row_ids": frozenset(),
                       "unprotected_row_ids": frozenset({"r"})},
             "testreul": {"outputs": [], "protected_row_ids": frozenset(),
                           "unprotected_row_ids": frozenset({"r"})}}
    row = {"id": "r", "input": "gana, testreul!", "pieces": pieces}
    assert module.chunk_piece_indices(aligned[1], pieces, beams, row) == (
        None, "partial_source_piece")
    assert module.eligible_occurrences(" ".join(["aa"] * 129)) == []
    assert module.forward_alignment("gareul", "가를") == [
        {"text": "가를", "start": 0, "end": 6}]
    assert module.forward_alignment("gareul", "가reul") == [
        {"text": "가", "start": 0, "end": 2}]
    assert module.option_groups([piece("aa", 0, "aa")], {
        "aa": {"outputs": [{"text": "아아", "score": 0.0}],
               "protected_row_ids": frozenset(),
               "unprotected_row_ids": frozenset({"other-row"})}}, 8,
        {"id": "bypassed-row", "input": " ".join(["aa"] * 129)}) == [("aa",)]


def test_metrics_use_all_accepted_targets_and_keep_protected_beams_diagnostic_only():
    rows = [
        {"id": "r1", "input": "gana!",
         "pieces": [piece("gana", 0, "가너"), piece("!", 4, "!")]},
        {"id": "r2", "input": "`gana`",
         "pieces": [piece("`gana`", 0, "`gana`")]},
        {"id": "r3", "input": "testreul!",
         "pieces": [piece("testreul", 0, "test를"), piece("!", 8, "!")]},
    ]
    beams = {
        "gana": {"outputs": [{"text": "가나", "score": -0.1}],
                 "protected_row_ids": frozenset({"r2"}),
                 "unprotected_row_ids": frozenset({"r1"})},
        "testreul": {"outputs": [], "protected_row_ids": frozenset(),
                     "unprotected_row_ids": frozenset({"r3"})},
    }
    targets = [
        {"id": "r1", "input": "gana!", "expected": ["가나!", "간아!"],
         "finite_indices": [0], "candidates": [{"text": "가너!"}]},
        {"id": "r2", "input": "`gana`", "expected": ["`가나`"],
         "finite_indices": [], "candidates": []},
        {"id": "r3", "input": "testreul!", "expected": ["test를!"],
         "finite_indices": [0], "candidates": [{"text": "test를!"}]},
    ]
    result = module.evaluate_metrics(rows, beams, targets)
    counts = result["counts"]
    assert counts["mechanical_full_rows"] == 1
    assert counts["union_top1_full_rows"] == 2
    assert counts["union_top1_full_gain_rows"] == 1
    assert counts["direct_protected_beam_top1_covered"] == 1
    assert counts["direct_unprotected_pure_hangul_chunks"] == 1
    assert counts["direct_unprotected_beam_top1_covered"] == 1
    assert counts["pure_hangul_chunks"] == 1
    assert counts["accepted_hangul_forms"] == 4
    assert counts["beam_top1_chunk_covered"] == 1
    assert counts["saved_finite_full_rows"] == 1
    assert sum(counts[key] for key in (
        "top8_gain_yes__saved_finite_yes", "top8_gain_yes__saved_finite_no",
        "top8_gain_no__saved_finite_yes", "top8_gain_no__saved_finite_no")) == 3
    assert result["alignment_exclusions"]["partial_source_piece"] >= 2


def test_forward_validity_and_selection_membership_corruption_fail_closed(tmp_path):
    values = closed_fixture(tmp_path)
    capture_path = values[0] / "capture.jsonl"
    capture = json.loads(capture_path.read_text())
    capture["outputs"][0]["text"] = "다나"
    capture_sha = write_jsonl(capture_path, [capture])
    report_path = values[0] / "report.json"
    report = json.loads(report_path.read_text())
    report["capture_sha256"] = capture_sha
    report_path.write_text(json.dumps(report))
    values = (values[0], module.sha(report_path), *values[2:])
    with pytest.raises(ValueError, match="forward-valid"):
        validate_fixture(values)
