import json

from scripts.symmetric_pair_text_layout import semantic_views


def test_full_text_and_policy_roles_survive_swapping_and_escaping():
    row = {"input": 'roman"\ninput', "context": "문맥"}
    a, b = {"text": "기존\n표기"}, {"text": '대안"표기'}
    first, second = map(json.loads, semantic_views(row, a, b))
    assert first[:2] == second[:2] == [row["input"], row["context"]]
    assert first[2:] == [["대안", b["text"]], ["현재", a["text"]]]
    assert second[2:] == list(reversed(first[2:]))


def test_only_context_is_bounded_and_bookkeeping_does_not_enter_encoder():
    row = {"input": "r"*2000, "context": "a"*128+"b"*128, "expected": ["secret"]}
    a, b = {"text": "가"*1000, "origins": {"score": 99}}, {"text": "나"*1000}
    views = semantic_views(row, a, b)
    decoded = json.loads(views[0])
    assert decoded[0] == row["input"]
    assert decoded[1] == "a"*64+"[…중략…]"+"b"*64
    assert decoded[2][1] == b["text"] and decoded[3][1] == a["text"]
    row["expected"] = ["different"]
    a["origins"]["score"] = -999
    assert semantic_views(row, a, b) == views
