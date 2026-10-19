import json
from pathlib import Path

import pytest

from scripts import prepare_long_context_lora_data as module


def candidate(source, text, kind="lexicon"):
    return {"text": text, "score": 1.0,
            "segments": [{"input_start": 0, "input_end": len(source),
                          "source": source, "text": text, "kind": kind}]}


def row(identity="case-1", expected=None):
    source = "na"
    candidates = [candidate(source, "나"), candidate(source, "나", "neural")]
    candidates[1]["text"] = "너"
    candidates[1]["segments"][0]["text"] = "너"
    return {"id": identity, "group_id": "group-a", "input": source,
            "context": "문맥", "expected": expected or ["나"],
            "category": "SECRET-CATEGORY", "candidates": candidates}


def test_schedule_is_salted_deterministic_and_complete():
    records = [{"role": "natural_fit", "id": f"id-{i}"} for i in range(11)]
    first = module.training_schedule(records)
    assert first == module.training_schedule(list(records))
    assert sorted(item["id"] for item in first) == sorted(item["id"] for item in records)
    assert [item["microbatch"] for item in first] == list(range(1, 12))
    assert first[-1]["update"] == 3


def test_prompt_never_reads_or_serializes_answer_fields():
    item = row(expected=["SECRET-ANSWER"])
    presented, messages = module.make_prompt(item, {"role": "system", "content": "system"})
    rendered = json.dumps(messages, ensure_ascii=False)
    assert "SECRET-ANSWER" not in rendered
    assert "SECRET-CATEGORY" not in rendered
    assert len(presented) == 2


def test_eligibility_precedence_and_multi_gold(monkeypatch):
    item = row(expected=["나", "너"])
    presented = [(0, "나"), (1, "너")]
    monkeypatch.setattr(module, "proposal_allowed", lambda *_: True)
    assert module.classify(item, presented, 10) == ("eligible", [0, 1])
    assert module.classify(item, presented[:1], 5000) == ("insufficient_choices", [])
    assert module.classify(item, presented, 1025) == ("prompt_over_1024", [])
    item["expected"] = ["missing"]
    assert module.classify(item, presented, 10) == ("gold_not_in_choices", [])
    item["expected"] = ["너"]
    monkeypatch.setattr(module, "proposal_allowed", lambda *_: False)
    assert module.classify(item, presented, 10) == ("no_allowed_gold", [])


def test_whole_group_collision_excludes_every_member(monkeypatch):
    monkeypatch.setattr(module, "romanize_word", lambda text: text.lower())
    raw = [
        {"id": "a", "group_id": "g1", "text": "A", "left_context": "c"},
        {"id": "b", "group_id": "g1", "text": "B", "left_context": "c"},
        {"id": "c", "group_id": "g2", "text": "C", "left_context": "c"},
        {"id": "d", "group_id": "g3", "text": "D", "left_context": "c"},
    ]
    partition = {"remaining_ids": ["a", "b", "c", "d"],
                 "remaining_group_ids": ["g1", "g2", "g3"]}
    groups, selected, excluded, _, _ = module.select_fresh(
        raw, partition, {"a"}, set(), set(), target=2,
        expected_rows=4, expected_groups=3)
    assert "g1" in excluded and excluded["g1"] == ["id"]
    assert not {"a", "b"} & {r["id"] for r in selected}
    assert set(groups) == {r["group_id"] for r in selected}


def test_partial_reserve_group_fails(monkeypatch):
    monkeypatch.setattr(module, "romanize_word", lambda text: text.lower())
    raw = [{"id": "a", "group_id": "g1", "text": "A", "left_context": "c"},
           {"id": "b", "group_id": "g1", "text": "B", "left_context": "c"}]
    partition = {"remaining_ids": ["a"], "remaining_group_ids": ["g1"]}
    with pytest.raises(ValueError, match="partial document group"):
        module.select_fresh(raw, partition, set(), set(), set(), target=1,
                            expected_rows=1, expected_groups=1)


def test_swapped_equal_count_record_fails():
    expected = [{"role": "natural_fit", "id": "a", "allowed_gold_positions": [0]},
                {"role": "challenge_fit", "id": "b", "allowed_gold_positions": [1]}]
    with pytest.raises(ValueError, match="identity or gold mapping"):
        module.validate_frozen_records(list(reversed(expected)), expected)


def test_source_pin_tamper_fails(tmp_path):
    source = tmp_path / "source.json"
    source.write_text("original")
    pins = {str(source): module.digest(b"original")}
    module.verify_hashes(pins)
    source.write_text("tampered")
    with pytest.raises(ValueError, match="source changed"):
        module.verify_hashes(pins)
