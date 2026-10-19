import json
from pathlib import Path

import pytest

import scripts.prepare_difference_pairwise_lora_data as module


class FakeTokenizer:
    chat_template = "frozen-fake-template"

    def encode(self, text, add_special_tokens=False):
        if text.endswith("1") or text.endswith("2"):
            return self.encode(text[:-1], add_special_tokens=False) + [700 + int(text[-1])]
        return [len(text), *(ord(char) % 697 for char in text[-4:])]

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt,
                            enable_thinking, return_dict=False):
        assert add_generation_prompt is True
        assert enable_thinking is False
        rendered = "<system>" + messages[0]["content"] + "<user>" + messages[1]["content"] + "<assistant>"
        return self.encode(rendered) if tokenize else rendered


def candidate(source="imnida", text="입니다", kind="morphology", score=1.0):
    return {"score": score, "text": text, "segments": [{
        "input_start": 0, "input_end": len(source), "source": source,
        "text": text, "kind": kind,
    }]}


def row(parent="임니다", alternative="입니다", expected=("입니다",), context=""):
    return {
        "id": "row-1", "group_id": "group-1", "input": "imnida", "context": context,
        "expected": list(expected),
        "candidates": [candidate(text=parent, score=2.0), candidate(text=alternative, score=1.0)],
        "raw_bypass_response": {"result": "imnida"},
    }


def key(**changes):
    value = {"shape": "single", "context": "empty", "span_count": "one",
             "roman_length": "5_8", "kind_signature": "morphology>morphology",
             "neighbor_class": "none"}
    value.update(changes)
    return value


def pool_pair(identity, match_key):
    return {"pair_id": identity, "match_key": match_key}


def test_context_window_is_bounded_and_deterministic():
    context = "앞" * 300 + "imnida" + "뒤" * 300
    centered = module.external_context_window(context, "imnida")
    assert centered["strategy"] == "centered_source"
    assert len(centered["text"]) == 256
    assert "imnida" in centered["text"]

    missing = module.external_context_window("가" * 600, "imnida")
    assert missing["strategy"] == "head_tail_gap"
    assert len(missing["head"]) == len(missing["tail"]) == 128
    assert missing["source_chars"] == 256
    assert missing["gap_marker"] == module.GAP_MARKER
    assert missing == module.external_context_window("가" * 600, "imnida")


def test_pair_identity_and_prompt_are_source_only(monkeypatch):
    monkeypatch.setattr(module, "proposal_allowed", lambda *_: True)
    pair, target = module.build_pair("natural_fit", row(), 1)
    assert target == "SWITCH"
    assert pair["pair_id"] == module.digest(module.canonical({
        "role": "natural_fit", "row_id": "row-1", "current_source_index": 0,
        "alternative_source_index": 1,
    }))
    payload = module.prompt_payload(pair)
    assert list(sorted(payload)) == ["ALTERNATIVE", "CURRENT", "changed_spans",
                                     "external_context", "mapping", "raw_mode", "roman_input"]
    assert payload["mapping"] == {"1": "KEEP", "2": "SWITCH"}
    assert not ({"expected", "target", "answer", "accepted"} & set(payload))
    assert pair["changed_spans"][0]["left_context"] == ""
    assert pair["changed_spans"][0]["right_context"] == ""


def semantic_payload():
    return {
        "roman_input": "imnida testreul", "external_context": {"text": "문맥"},
        "raw_mode": False, "mapping": {"1": "KEEP", "2": "SWITCH"},
        "CURRENT": {"text": "임니다 test를", "score": 1.5, "rank": 1,
                    "kinds": list(module.KIND_CODEBOOK.values())},
        "ALTERNATIVE": {"text": "입니다 test를", "score": 0, "rank": 2,
                        "kinds": ["syllable", *list(module.KIND_CODEBOOK.values())[1:6], "lexicon"]},
        "changed_spans": [{"roman": "imnida", "current_kind": "syllable",
                           "alternative_kind": "morphology"}],
    }


def test_compact_codec_all_codes_boundary_edits_and_exact_round_trip():
    original = semantic_payload()
    compact = module.encode_compact_prompt_payload(original)
    assert compact["K"] == {"c": "LXMNDRS", "a": "0=S,6=L"}
    assert "kinds" not in compact["CURRENT"] and "kinds" not in compact["ALTERNATIVE"]
    assert module.canonical(module.decode_compact_prompt_payload(compact)) == module.canonical(original)
    assert original == semantic_payload()


def test_compact_codec_empty_edits_and_long_alternating_sequence():
    original = semantic_payload()
    original["CURRENT"]["kinds"] = ["lexicon", "raw"] * 80
    original["ALTERNATIVE"]["kinds"] = original["CURRENT"]["kinds"].copy()
    compact = module.encode_compact_prompt_payload(original)
    assert compact["K"] == {"c": "LR" * 80, "a": ""}
    assert module.decode_compact_prompt_payload(compact) == original


@pytest.mark.parametrize("edits", [
    "-1=N", "+1=N", "01=N", "1.0=N", "7=N", "0=N,0=S", "6=L,0=S", "=N",
    "0N", "0=N,", ",0=N", " 0=N", "0 =N", "0= N", "0=n", "0=Z", "0=한", "0=L",
])
def test_compact_decoder_rejects_malformed_edits(edits):
    compact = module.encode_compact_prompt_payload(semantic_payload())
    compact["K"]["a"] = edits
    with pytest.raises(ValueError):
        module.decode_compact_prompt_payload(compact)


@pytest.mark.parametrize("codes", ["", "lX", "LZ", "L한", "L R", 7, None])
def test_compact_decoder_rejects_malformed_codes(codes):
    compact = module.encode_compact_prompt_payload(semantic_payload())
    compact["K"]["c"] = codes
    with pytest.raises(ValueError):
        module.decode_compact_prompt_payload(compact)


@pytest.mark.parametrize("mutation", ["collision", "empty", "unequal", "unknown"])
def test_compact_encoder_rejects_invalid_semantics(mutation):
    original = semantic_payload()
    if mutation == "collision":
        original["K"] = {}
    elif mutation == "empty":
        original["CURRENT"]["kinds"] = original["ALTERNATIVE"]["kinds"] = []
    elif mutation == "unequal":
        original["CURRENT"]["kinds"].pop()
    else:
        original["CURRENT"]["kinds"][0] = "unknown"
    with pytest.raises(ValueError):
        module.encode_compact_prompt_payload(original)


def test_make_messages_uses_compact_payload_without_changing_semantics(monkeypatch):
    monkeypatch.setattr(module, "proposal_allowed", lambda *_: True)
    pair, _ = module.build_pair("natural_fit", row(), 1)
    semantic = module.prompt_payload(pair)
    messages = module.make_messages(pair)
    assert messages[0]["content"] == module.SYSTEM_TEXT
    compact = json.loads(messages[1]["content"])
    assert "K" in compact and "kinds" not in compact["CURRENT"]
    assert module.canonical(module.decode_compact_prompt_payload(compact)) == module.canonical(semantic)


@pytest.mark.parametrize(
    "parent,alternative,expected,target",
    [
        ("입니다", "임니다", ("입니다",), "KEEP"),
        ("입니다", "임니다", ("입니다", "임니다"), "KEEP"),
        ("임니다", "입니다", ("입니다",), "SWITCH"),
        ("임니다", "입이다", ("입니다",), "KEEP"),
    ],
)
def test_pair_target_contract(monkeypatch, parent, alternative, expected, target):
    monkeypatch.setattr(module, "proposal_allowed", lambda *_: True)
    assert module.build_pair("natural_fit", row(parent, alternative, expected), 1)[1] == target


def test_matching_relaxes_cumulatively_and_reuses_only_after_exhaustion():
    repair = key(context="present", roman_length="3_4")
    exact = pool_pair("exact", key(context="present", roman_length="3_4"))
    relaxed = pool_pair("relaxed", key(context="empty", roman_length="3_4"))
    counts = module.Counter()
    chosen, evidence = module.choose_control("unit", repair, [relaxed, exact], counts)
    assert chosen["pair_id"] == "exact"
    assert evidence == {"relaxation_depth": 0, "prior_use_count": 0,
                        "reuse_after_unused_exhaustion": False}
    chosen, evidence = module.choose_control("unit-2", repair, [relaxed, exact], counts)
    assert chosen["pair_id"] == "relaxed"
    assert evidence["relaxation_depth"] == 1
    assert evidence["reuse_after_unused_exhaustion"] is False
    chosen, evidence = module.choose_control("unit-3", repair, [relaxed, exact], counts)
    assert evidence["reuse_after_unused_exhaustion"] is True
    assert evidence["prior_use_count"] == 1


@pytest.mark.parametrize(
    "field,depth",
    [("context", 1), ("roman_length", 2), ("neighbor_class", 3),
     ("shape", 4), ("kind_signature", 5)],
)
def test_each_matching_field_drops_only_at_approved_depth(field, depth):
    repair = key()
    control = key()
    replacements = {"context": "present", "roman_length": "1_2",
                    "neighbor_class": "latin", "shape": "sentence",
                    "kind_signature": "lexicon>neural"}
    control[field] = replacements[field]
    assert not module._relaxed_equal(repair, control, depth - 1)
    assert module._relaxed_equal(repair, control, depth)
    control["span_count"] = "multi"
    assert not module._relaxed_equal(repair, control, 5)


def test_tokenizer_requires_false_thinking_single_digit_boundary(monkeypatch):
    monkeypatch.setattr(module, "proposal_allowed", lambda *_: True)
    pair, _ = module.build_pair("natural_fit", row(), 1)
    messages, tokens, ids, rendered = module.tokenize_pair(FakeTokenizer(), pair)
    assert messages[-1]["role"] == "user"
    assert tokens == FakeTokenizer().encode(rendered)
    assert ids == {"1": 701, "2": 702}


def test_full_frozen_fit_construction_is_deterministic(tmp_path, monkeypatch):
    bridge = tmp_path / "bridge.json"
    stable_verifier = tmp_path / "verifier.py"
    stable_verifier.write_bytes(module.VERIFIER.read_bytes())
    monkeypatch.setattr(module, "VERIFIER", stable_verifier)
    monkeypatch.setattr(module, "COMPACT_BRIDGE", bridge)
    module.fit_bridge(bridge)
    monkeypatch.setattr(module, "_source_pins", lambda _: {})
    monkeypatch.setattr(module.importlib.metadata, "version", lambda name: {
        "mlx": "0.31.2", "mlx-lm": "0.31.3", "transformers": "5.12.1",
    }[name])
    monkeypatch.setattr(module, "CHAT_TEMPLATE_SHA256",
                        module.digest(FakeTokenizer.chat_template.encode()))

    first, second = tmp_path / "first", tmp_path / "second"
    one = module.prepare(first, bridge, tokenizer_loader=FakeTokenizer)
    two = module.prepare(second, bridge, tokenizer_loader=FakeTokenizer)
    assert one["counts"] == two["counts"]
    assert one["representation_version"] == "shared-alpha-kind-edits-v1"
    assert one["prompt_contract"]["codebook"] == module.KIND_CODEBOOK
    assert one["prompt_contract"]["decoded_semantics_byte_equal"] is True
    assert one["counts"] | {
        "fit_rows": 3492, "repair_units": 59, "repair_switch_references": 59,
        "natural_keep_references": 59, "challenge_keep_references": 59,
        "same_row_keep_references": 55, "repair_rows_without_same_row_keep": 4,
        "component_references": 232, "updates": 15, "final_update_units": 3,
    } == one["counts"]
    for name in one["file_hashes"]:
        assert (first / name).read_bytes() == (second / name).read_bytes()
    units = [json.loads(line) for line in (first / "units.jsonl").read_text().splitlines()]
    assert [unit["order"] for unit in units] == list(range(1, 60))
    assert all(abs(sum(ref["weight"] for ref in unit["references"]) - 1.0) < 1e-12
               for unit in units)
    prompts = [json.loads(line) for line in (first / "prompts.jsonl").read_text().splitlines()]
    assert len(prompts) == one["counts"]["required_unique_pairs"]
    assert all("target" not in module.canonical(prompt).decode() for prompt in prompts)
