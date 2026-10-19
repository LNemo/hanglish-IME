import json
import hashlib

import pytest

from scripts.prepare_document_sentence_inputs import canonical, project, write_artifacts


def selected(identifier, group):
    return {"id": identifier, "group_id": group}


def raw(identifier, group, text):
    return {"id": identifier, "group_id": group, "text": text,
            "label_only_field": "must-not-be-emitted"}


def test_full_text_preserves_boundaries_case_ascii_punctuation_and_space():
    requests, summary = project(
        [selected("a", "g")],
        [raw("a", "g", "  Test를 시행, ABC-12!  ")],
    )
    assert requests == [{"id": "a", "input": "  Testreul sihaeng, ABC-12!  "}]
    assert summary["counts"]["non_hangul_literal_preservation_failures"] == 0
    assert summary["counts"]["multi_token_sources"] == 1
    assert set(requests[0]) == {"id", "input"}


def test_duplicate_inputs_keep_distinct_ids_and_exact_selection_order():
    selection = [selected("b", "g2"), selected("a", "g1")]
    references = [raw("a", "g1", "가"), raw("b", "g2", "가")]
    requests, summary = project(selection, references)
    assert requests == [{"id": "b", "input": "ga"},
                        {"id": "a", "input": "ga"}]
    assert summary["counts"]["rows"] == 2
    assert summary["counts"]["groups"] == 2
    assert summary["counts"]["unique_inputs"] == 1
    assert summary["counts"]["duplicate_input_rows"] == 1


def test_whole_reference_is_passed_to_romanizer_without_trimming_or_lowering():
    calls = []
    def romanize(value):
        calls.append(value)
        return value.replace("한", "han")
    requests, _ = project([selected("a", "g")],
                          [raw("a", "g", " X한 Y ")], romanize)
    assert calls[0] == " X한 Y "
    assert requests[0]["input"] == " Xhan Y "


@pytest.mark.parametrize("selection,references,message", [
    ([selected("a", "g"), selected("a", "g")],
     [raw("a", "g", "가")], "selection"),
    ([{"id": "a", "group_id": "g", "expected": "가"}],
     [raw("a", "g", "가")], "selection"),
    ([selected("a", "g")], [raw("a", "other", "가")], "mapping"),
    ([selected("missing", "g")], [raw("a", "g", "가")], "missing"),
    ([selected("a", "g")], [raw("a", "g", "")], "reference"),
    ([selected("a", "g")], [raw("a", "g", "가"), raw("a", "g", "나")],
     "duplicate raw"),
])
def test_invalid_selection_reference_and_group_mapping_are_rejected(
        selection, references, message):
    with pytest.raises(ValueError, match=message):
        project(selection, references)


def test_unsupported_lengths_and_residual_jamo_are_counted_not_filtered():
    long_text = "가" * 2100
    requests, summary = project(
        [selected("jamo", "g"), selected("long", "g")],
        [raw("jamo", "g", "ㄱ가"), raw("long", "g", long_text)],
    )
    assert [row["id"] for row in requests] == ["jamo", "long"]
    counts = summary["counts"]
    assert counts["rows"] == 2
    assert counts["ctc_token_limit_exceeded_rows"] == 1
    assert counts["source_utf8_over_512_rows"] == 1
    assert counts["ui_character_limit_exceeded_rows"] == 1
    assert counts["residual_jamo_codepoints"] == 1
    assert counts["rows_with_residual_jamo"] == 1
    assert counts["residual_precomposed_hangul_codepoints"] == 0


def test_non_hangul_boundary_mutation_is_rejected():
    def broken(value):
        return value.lower().replace("가", "ga")
    with pytest.raises(ValueError, match="literal boundaries"):
        project([selected("a", "g")], [raw("a", "g", "A가")], broken)


def test_output_creation_is_exclusive_and_has_no_reference_or_group_fields(tmp_path):
    requests = [{"id": "a", "input": "testreul handa."},
                {"id": "b", "input": "ga"}]
    content = b"".join(canonical(row) + b"\n" for row in requests)
    manifest = {"schema": "fixture",
                "inputs_sha256": hashlib.sha256(content).hexdigest()}
    output = tmp_path / "output"
    write_artifacts(output, requests, manifest)
    rows = [json.loads(line) for line in (output / "inputs.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert all(set(row) == {"id", "input"} for row in rows)
    with pytest.raises(FileExistsError):
        write_artifacts(output, requests, manifest)
