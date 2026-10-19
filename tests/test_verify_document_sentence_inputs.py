import ast
import copy
from pathlib import Path

import pytest

from scripts import verify_document_sentence_inputs as verifier


def rows():
    selection = [
        {"id": "a", "group_id": "g1"},
        {"id": "b", "group_id": "g1"},
        {"id": "c", "group_id": "g2"},
    ]
    raw = [
        {"id": "a", "group_id": "g1", "text": "입니다 API 2.0!"},
        {"id": "b", "group_id": "g1", "text": "입니다 API 2.0!"},
        {"id": "c", "group_id": "g2", "text": "한글 test를 시행"},
    ]
    return selection, raw


def test_verifier_does_not_import_producer_model_or_tensor_modules():
    tree = ast.parse(Path(verifier.__file__).read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    forbidden = ("prepare_document_sentence_inputs", "torch", "phonetic_likelihood",
                 "production_phonetic_choice_converter")
    assert not any(name.endswith(item) or name == item
                   for name in imports for item in forbidden)


def test_replay_romanizes_full_text_and_preserves_non_hangul_literals():
    selection, raw = rows()
    requests, projection = verifier.replay(selection, raw, expected_rows=3)
    assert requests[0]["input"].endswith(" API 2.0!")
    assert requests[2]["input"].endswith(" testreul sihaeng")
    assert all(set(row) == {"id", "input"} for row in requests)
    assert projection["counts"]["rows"] == 3
    assert projection["counts"]["groups"] == 2
    assert projection["counts"]["residual_precomposed_hangul_codepoints"] == 0
    assert projection["counts"]["residual_jamo_codepoints"] == 0


def test_replay_retains_same_input_rows_with_distinct_ids():
    selection, raw = rows()
    requests, projection = verifier.replay(selection, raw, expected_rows=3)
    assert requests[0]["input"] == requests[1]["input"]
    assert requests[0]["id"] != requests[1]["id"]
    assert projection["counts"]["unique_inputs"] == 2
    assert projection["counts"]["duplicate_input_rows"] == 1


def test_replay_rejects_partial_group_or_changed_mapping():
    selection, raw = rows()
    with pytest.raises(ValueError, match="whole-document-group"):
        verifier.replay([selection[0], selection[2]], raw, expected_rows=2)

    changed = copy.deepcopy(selection)
    changed[0]["group_id"] = "g2"
    with pytest.raises(ValueError):
        verifier.replay(changed, raw, expected_rows=3)


def test_replay_rejects_duplicate_id_and_extra_selection_field():
    selection, raw = rows()
    duplicate = copy.deepcopy(selection)
    duplicate[1]["id"] = duplicate[0]["id"]
    with pytest.raises(ValueError, match="selected"):
        verifier.replay(duplicate, raw, expected_rows=3)

    extra = copy.deepcopy(selection)
    extra[0]["target"] = "금지"
    with pytest.raises(ValueError, match="selection schema"):
        verifier.replay(extra, raw, expected_rows=3)


def test_digest_lines_is_order_sensitive_and_summary_is_exact():
    assert verifier.digest_lines(["a", "b"]) != verifier.digest_lines(["b", "a"])
    assert verifier.summary([2, 2, 4]) == {
        "minimum": 2, "maximum": 4, "histogram": {"2": 2, "4": 1}}
