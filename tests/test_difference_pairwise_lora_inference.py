from copy import deepcopy
import json

import pytest

from scripts.difference_pairwise_lora_evaluation import resolve_selection
from scripts.difference_pairwise_lora_inference import (
    EXPECTED_DIGIT_CONTINUATION_IDS,
    prepare_inference_selection,
    qualify_inference_tokens,
)
from scripts.prepare_difference_pairwise_lora_data import (
    MAX_PROMPT_TOKENS,
    build_pair,
    make_messages,
)


def segment(source, start, end, text, kind="morphology"):
    return {
        "input_start": start,
        "input_end": end,
        "source": source[start:end],
        "text": text,
        "kind": kind,
    }


def candidate(source, text, score, segments):
    return {"text": text, "score": score, "segments": segments}


def simple_row(*, parent_score=3.0, alternative_scores=(2.0, 1.0)):
    source = "imnida"
    candidates = [
        candidate(source, "임니다", parent_score, [segment(source, 0, len(source), "임니다")]),
        candidate(source, "입니다", alternative_scores[0],
                  [segment(source, 0, len(source), "입니다")]),
        candidate(source, "입니다", alternative_scores[1],
                  [segment(source, 0, len(source), "입니다")]),
    ]
    return {
        "id": "row-1",
        "group_id": "group-1",
        "input": source,
        "context": "앞 문장에서 설명한 대상입니다.",
        "expected": ["입니다"],
        "candidates": candidates,
        "raw_bypass_response": {"result": source, "candidates": [{"text": source}]},
    }


class LabelTrap(dict):
    def __getitem__(self, key):
        if key in {"expected", "targets", "accepted", "answer"}:
            raise AssertionError(f"opened label field: {key}")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in {"expected", "targets", "accepted", "answer"}:
            raise AssertionError(f"opened label field: {key}")
        return super().get(key, default)

    def __iter__(self):
        raise AssertionError("iterated the untrusted source row")

    def items(self):
        raise AssertionError("iterated the untrusted source row")


def test_source_projection_never_opens_labels_and_preserves_object_identity():
    row = simple_row()
    trapped = LabelTrap(row)
    selection = prepare_inference_selection("fresh", trapped)
    assert selection["candidates"] is row["candidates"]
    assert selection["raw_bypass_response"] is row["raw_bypass_response"]
    assert selection["group_id"] == row["group_id"]
    assert selection["alternative_indices"] == [1, 2]
    assert "expected" not in json.dumps(selection["pairs"], ensure_ascii=False)


def test_pairs_and_compact_prompts_match_frozen_builder_for_any_labels():
    row = simple_row()
    selection = prepare_inference_selection("fresh", row)
    for expected in (["입니다"], ["임니다"], ["전혀 다른 정답"]):
        labelled = {**row, "expected": expected}
        built = [build_pair("fresh", labelled, index)[0] for index in (1, 2)]
        assert selection["pairs"] == built
        assert [make_messages(pair) for pair in selection["pairs"]] == [
            make_messages(pair) for pair in built
        ]


def mixed_row():
    source = "imnida test 2026"
    bounds = [(0, 6), (6, 7), (7, 11), (11, 12), (12, 16)]
    sources = [source[left:right] for left, right in bounds]

    def parts(first, english="test", number="2026", english_kind="raw", number_kind="numeric"):
        texts = [first, " ", english, " ", number]
        kinds = ["morphology", "raw", english_kind, "raw", number_kind]
        return [segment(source, left, right, text, kind)
                for (left, right), text, kind in zip(bounds, texts, kinds)]

    candidates = [
        candidate(source, "임니다 test 2026", 4.0, parts("임니다")),
        candidate(source, "입니다 test 2026", 3.0, parts("입니다")),
        candidate(source, "임니다 테스트 2026", 2.0, parts("임니다", "테스트", english_kind="neural")),
        candidate(source, "임니다 test 이천이십육", 1.0,
                  parts("임니다", number="이천이십육", number_kind="neural")),
    ]
    return {
        "id": "mixed", "group_id": "mixed-group", "input": source, "context": "",
        "candidates": candidates, "raw_bypass_response": {"result": source},
    }, sources


def test_english_and_numeric_spans_stay_protected_while_korean_pair_is_built():
    row, _ = mixed_row()
    selection = prepare_inference_selection("fresh", row)
    assert selection["alternative_indices"] == [1]
    assert selection["pairs"][0]["changed_spans"] == [{
        "segment_index": 0,
        "input_start": 0,
        "input_end": 6,
        "roman": "imnida",
        "left_context": "",
        "right_context": " test 2026",
        "current_text": "임니다",
        "current_kind": "morphology",
        "alternative_text": "입니다",
        "alternative_kind": "morphology",
    }]


def test_none_parent_produces_no_pairs_but_nonfinite_scores_fail():
    row = simple_row(parent_score=None)
    selection = prepare_inference_selection("fresh", row)
    assert selection["pairs"] == []
    assert selection["alternative_indices"] == []
    row = simple_row(alternative_scores=(float("nan"), 1.0))
    with pytest.raises(ValueError, match="finite or None"):
        prepare_inference_selection("fresh", row)


def test_raw_mode_and_raw_parent_bypass_pair_construction(monkeypatch):
    from scripts import difference_pairwise_lora_inference as module

    row = simple_row()
    monkeypatch.setattr(
        module,
        "permitted_pairs",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("constructed raw pairs")),
    )
    selection = prepare_inference_selection("fresh", row, raw_mode=True)
    assert selection["pairs"] == []
    result = resolve_selection(selection, {})
    assert result["complete"] is True
    assert result["reason"] == "raw_mode"

    monkeypatch.undo()
    row["candidates"][0]["segments"][0]["kind"] = "raw"
    assert prepare_inference_selection("fresh", row)["pairs"] == []


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.update(group_id=""),
        lambda row: row.update(group_id=7),
        lambda row: row.update(id=""),
        lambda row: row.update(context=None),
        lambda row: row.update(candidates={}),
        lambda row: row.update(raw_bypass_response=[]),
    ],
)
def test_identity_and_source_field_validation(mutation):
    row = simple_row()
    mutation(row)
    with pytest.raises(ValueError, match="malformed"):
        prepare_inference_selection("fresh", row)


def test_malformed_alignment_is_not_a_permitted_pair():
    row = simple_row()
    row["candidates"][1]["segments"][0]["input_end"] -= 1
    selection = prepare_inference_selection("fresh", row)
    assert selection["alternative_indices"] == [2]
    assert all(pair["alternative_source_index"] != 1 for pair in selection["pairs"])


def test_complete_pair_coverage_and_resolver_integration():
    row = simple_row()
    selection = prepare_inference_selection("fresh", row)
    logits = {
        selection["pairs"][0]["pair_id"]: [0.0, 1.0],
        selection["pairs"][1]["pair_id"]: [0.0, 3.0],
    }
    result = resolve_selection(selection, logits)
    assert result["complete"] is True
    assert result["expected_pair_count"] == result["scored_pair_count"] == 2
    assert result["chosen_index"] == 2
    assert result["candidates"][0] is row["candidates"][2]
    assert result["raw_bypass_response"] is row["raw_bypass_response"]


class FakeTokenizer:
    def __init__(self, *, prompt_tokens=32, digit_ids=(16, 17)):
        self.prompt_tokens = prompt_tokens
        self.digit_ids = digit_ids

    def encode(self, text, add_special_tokens=False):
        if text.endswith("1"):
            return [7] * self.prompt_tokens + [self.digit_ids[0]]
        if text.endswith("2"):
            return [7] * self.prompt_tokens + [self.digit_ids[1]]
        return [7] * self.prompt_tokens

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt,
                            enable_thinking, return_dict=False):
        assert add_generation_prompt is True and enable_thinking is False
        rendered = "<prefix>"
        return self.encode(rendered) if tokenize else rendered


def test_token_qualification_uses_frozen_ceiling_and_digit_ids():
    selection = prepare_inference_selection("fresh", simple_row())
    qualified = qualify_inference_tokens(FakeTokenizer(), selection)
    assert [item["pair_id"] for item in qualified] == [
        pair["pair_id"] for pair in selection["pairs"]
    ]
    assert all(item["digit_continuation_ids"] == EXPECTED_DIGIT_CONTINUATION_IDS
               for item in qualified)
    assert all(item["prompt_tokens"] == 32 for item in qualified)

    with pytest.raises(ValueError, match="exceeds"):
        qualify_inference_tokens(FakeTokenizer(prompt_tokens=MAX_PROMPT_TOKENS + 1), selection)
    with pytest.raises(ValueError, match="continuation IDs"):
        qualify_inference_tokens(FakeTokenizer(digit_ids=(701, 702)), selection)


class TokenizerCallTrap:
    def apply_chat_template(self, *_args, **_kwargs):
        raise AssertionError("tokenizer called before source evidence validation")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda selection: selection["pairs"][0]["changed_spans"][0].update(
            alternative_text="변조"
        ),
        lambda selection: selection["pairs"][0]["alternative"].update(rank=99),
        lambda selection: selection["pairs"][0]["match_key"].update(context="empty"),
        lambda selection: selection["pairs"].pop(),
        lambda selection: selection["pairs"].append(deepcopy(selection["pairs"][0])),
    ],
)
def test_token_qualification_rejects_modified_pair_evidence_before_tokenizer(mutation):
    selection = prepare_inference_selection("fresh", simple_row())
    mutation(selection)
    with pytest.raises(ValueError, match="differs from rebuilt"):
        qualify_inference_tokens(TokenizerCallTrap(), selection)


@pytest.mark.parametrize("field", ["role", "row_id", "group_id", "alternative_indices",
                                    "candidate_snapshot", "raw_snapshot"])
def test_token_qualification_rejects_missing_selection_fields(field):
    selection = prepare_inference_selection("fresh", simple_row())
    selection.pop(field)
    with pytest.raises(ValueError, match="fields are missing"):
        qualify_inference_tokens(TokenizerCallTrap(), selection)


def test_token_qualification_rejects_candidate_or_raw_mutation_before_tokenizer():
    selection = prepare_inference_selection("fresh", simple_row())
    selection["candidates"][0]["score"] = 10.0
    with pytest.raises(ValueError, match="differs from rebuilt"):
        qualify_inference_tokens(TokenizerCallTrap(), selection)

    selection = prepare_inference_selection("fresh", simple_row())
    selection["raw_bypass_response"]["result"] = "tampered"
    with pytest.raises(ValueError, match="differs from rebuilt"):
        qualify_inference_tokens(TokenizerCallTrap(), selection)


def test_raw_selection_is_rebuilt_and_qualified_without_tokenizer_calls():
    selection = prepare_inference_selection("fresh", simple_row(), raw_mode=True)
    assert qualify_inference_tokens(TokenizerCallTrap(), selection) == []
