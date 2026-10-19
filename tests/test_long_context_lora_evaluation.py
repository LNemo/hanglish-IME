from copy import deepcopy

import pytest

from scripts import long_context_lora_evaluation as module


class Tokenizer:
    def __init__(self, length=10):
        self.length = length

    def apply_chat_template(self, messages, *, tokenize, **kwargs):
        assert kwargs["enable_thinking"] is False
        assert kwargs["add_generation_prompt"] is True
        return list(range(self.length)) if tokenize else "prompt"

    def encode(self, text, **kwargs):
        return list(range(self.length)) + ([] if text == "prompt" else [100 + int(text[-1])])


def item(identity="x", source="imnida", parent="임니다", target="입니다"):
    def candidate(text, score=1.0):
        return {"text": text, "score": score, "segments": [{"source": source,
            "text": text, "kind": "lexicon", "input_start": 0, "input_end": len(source)}]}
    return {"role": "natural_fit", "stratum": "natural", "row": {"id": identity, "input": source,
        "context": "문맥", "result": parent, "expected": [target],
        "candidates": [candidate(parent), candidate(target), candidate(source, None)],
        "raw_bypass_response": {"result": source, "candidates": [candidate(source, None)]}}}


@pytest.fixture(autouse=True)
def system(monkeypatch):
    monkeypatch.setattr(module, "system_message", lambda: {"role": "system", "content": "select"})


def target_logits(record, text):
    return [2.0 if candidate == text else -1.0 for _, candidate in record["choices"]]


def test_source_only_prompt_and_eligibility_ignore_answers():
    first = item()
    second = deepcopy(first)
    second["row"]["expected"] = ["PRIVATE-ANSWER"]
    second["row"]["category"] = "PRIVATE-CATEGORY"
    assert module.prepare_diagnostic([first], Tokenizer()) == module.prepare_diagnostic([second], Tokenizer())


def test_assimilated_spelling_reordered_with_exact_objects_and_raw():
    source = item()
    record = module.prepare_diagnostic([source], Tokenizer())[0]
    result = module.apply_logits(record, target_logits(record, "입니다"))
    assert result["result"] == "입니다"
    assert result["candidates"][0] is source["row"]["candidates"][1]
    assert result["candidates"][-1] is source["row"]["candidates"][2]
    score = module.score_predictions([source], [result])
    assert score["metrics"]["all"]["corrections"] == 1
    assert score["gate"]["raw_preserved"]
    assert not score["gate"]["passed"]


def test_overlength_keeps_parent_in_denominator():
    source = item()
    record = module.prepare_diagnostic([source], Tokenizer(1025))[0]
    assert not record["eligible"]
    assert record["fallback_reason"] == "prompt_over_1024"
    result = module.apply_logits(record, None)
    assert result["result"] == "임니다"
    assert module.score_predictions([source], [result])["metrics"]["all"]["rows"] == 1
    with pytest.raises(ValueError):
        module.apply_logits(record, [1.0, 2.0])


def test_protected_ascii_changes_are_denied():
    source = item(source="test", parent="test", target="테스트")
    record = module.prepare_diagnostic([{**source, "row": {**source["row"],
        "candidates": source["row"]["candidates"][:2]}}], Tokenizer())[0]
    result = module.apply_logits(record, target_logits(record, "테스트"))
    assert result["result"] == "test"
    assert result["reason"] == "protected_or_unaligned"


@pytest.mark.parametrize("logits", [[float("nan"), 1.0], [1.0], [True, 1.0]])
def test_invalid_logits_fail(logits):
    record = module.prepare_diagnostic([item()], Tokenizer())[0]
    with pytest.raises(ValueError):
        module.apply_logits(record, logits)


def test_snapshot_mutation_and_prediction_tampering_fail():
    source = item()
    record = module.prepare_diagnostic([source], Tokenizer())[0]
    result = module.apply_logits(record, None)
    result["raw_bypass_response"]["result"] = "tampered"
    with pytest.raises(ValueError, match="raw preservation"):
        module.score_predictions([source], [result])
    record["candidates"][0]["score"] += 1
    with pytest.raises(ValueError, match="snapshot changed"):
        module.apply_logits(record, None)


def test_all640_gate_and_regression_accounting():
    sources = [item(str(i)) for i in range(640)]
    records = module.prepare_diagnostic(sources, Tokenizer())
    predictions = [module.apply_logits(record, target_logits(record, "입니다") if i < 5 else None)
                   for i, record in enumerate(records)]
    score = module.score_predictions(sources, predictions)
    assert score["gate"]["passed"]
    assert score["gate"]["net_top1_gain"] == 5
    with pytest.raises(ValueError, match="identity"):
        module.score_predictions(sources, list(reversed(predictions)))
    with pytest.raises(ValueError, match="accounted"):
        module.score_predictions(sources, predictions[:-1])


def test_insufficient_choices_falls_back_and_tie_uses_prompt_order():
    source = item()
    source["row"]["candidates"] = source["row"]["candidates"][:1]
    record = module.prepare_diagnostic([source], Tokenizer())[0]
    assert record["fallback_reason"] == "insufficient_choices"
    assert module.apply_logits(record, None)["chosen_index"] == 0
    record = module.prepare_diagnostic([item()], Tokenizer())[0]
    assert module.apply_logits(record, [1.0, 1.0])["proposed_index"] == record["choices"][0][0]


def test_regression_is_reported_and_all_allowed_answers_count():
    source = item(parent="입니다", target="임니다")
    source["row"]["expected"] = ["입니다"]
    record = module.prepare_diagnostic([source], Tokenizer())[0]
    result = module.apply_logits(record, target_logits(record, "임니다"))
    score = module.score_predictions([source], [result])
    assert score["gate"]["regressions"] == 1
    assert score["gate"]["net_top1_gain"] == -1
    source["row"]["expected"].append("임니다")
    assert module.score_predictions([source], [result])["gate"]["regressions"] == 0


def test_cross_role_duplicate_ids_fail_preparation_and_scoring():
    left, right = item(), item()
    right["role"] = "challenge_fit"
    with pytest.raises(ValueError, match="identity"):
        module.prepare_diagnostic([left, right], Tokenizer())
    predictions = [module.apply_logits(module.prepare_diagnostic([source], Tokenizer())[0], None)
                   for source in (left, right)]
    with pytest.raises(ValueError, match="identity"):
        module.score_predictions([left, right], predictions)


def test_strata_and_saved_mapping_are_verified():
    sources = [item(str(i)) for i in range(3)]
    for source, stratum in zip(sources, ("natural", "challenge_single", "challenge_sentence")):
        source["stratum"] = stratum
    records = module.prepare_diagnostic(sources, Tokenizer())
    predictions = [module.apply_logits(record, target_logits(record, "입니다")) for record in records]
    score = module.score_predictions(sources, predictions)
    assert {key: value["rows"] for key, value in score["strata"].items()} == {
        "natural": 1, "challenge_single": 1, "challenge_sentence": 1}
    predictions[0]["choices"] = list(reversed(predictions[0]["choices"]))
    with pytest.raises(ValueError, match="mapping"):
        module.score_predictions(sources, predictions)


def test_saved_logits_cannot_disagree_with_selection():
    source = item()
    record = module.prepare_diagnostic([source], Tokenizer())[0]
    result = module.apply_logits(record, target_logits(record, "입니다"))
    result["logits"] = target_logits(record, "임니다")
    with pytest.raises(ValueError, match="selection"):
        module.score_predictions([source], [result])
