from collections import Counter
from copy import deepcopy

import pytest

from hanglish.core.ungated_phonetic_proposals import generate_proposals
from scripts.verify_ungated_phonetic_fit import verify_counts, verify_row


class Tokenizer:
    def __call__(self, text, *, add_special_tokens, truncation):
        assert add_special_tokens is False and truncation is False
        return {"input_ids": text.split()}

    def num_special_tokens_to_add(self, *, pair):
        assert pair is True
        return 3


def segment(source, text=None, *, start=0, kind="raw"):
    return {"input_start": start, "input_end": start + len(source),
            "source": source, "text": source if text is None else text, "kind": kind}


def fixture():
    request = {"input": "gim", "context": ""}
    parent = {"text": "gim", "segments": [segment("gim", kind="neural")]}
    raw = {"text": "gim", "segments": [segment("gim")], "score": None}
    public_parent = {**parent, "score": -1.0}
    source = {
        "id": "s3-train-00001", "request": request,
        "response": {"candidates": [public_parent, raw]},
    }
    entry = {"id": source["id"], "shard": 0, "row_index": 2}
    generated = generate_proposals(
        request["input"], [parent], lambda word, beam_size=8: [["김", -0.1]])

    def supported(value, status="supported"):
        return {**value, "sentence_tokens": 4, "sentence_support": status}

    captured = {
        "id": source["id"], "request": request, "source_shard": 0,
        "source_row_index": 2, "fallback_public": deepcopy([public_parent, raw]),
        **generated,
    }
    captured["parents"] = [supported(row) for row in generated["parents"]]
    captured["proposals"] = [supported(row) for row in generated["proposals"]]
    report = {"known_english": [], "max_sources": 96, "max_extras": 96,
              "max_sentence_tokens": 96}
    return captured, entry, source, report


def test_replays_captured_beams_and_independently_checks_support():
    captured, entry, source, report = fixture()
    counts = verify_row(captured, entry, source, Tokenizer(), report)
    assert counts == {"parents": 1, "proposals": 1, "generation_requests": 1,
                      "parent_supported": 1, "supported_proposals": 1}


def test_tampered_support_status_is_rejected_without_dropping_candidate():
    captured, entry, source, report = fixture()
    captured["proposals"][0]["sentence_support"] = "unsupported"
    with pytest.raises(ValueError, match="tokenizer support"):
        verify_row(captured, entry, source, Tokenizer(), report)


def test_tampered_parent_or_full_fallback_identity_is_rejected():
    captured, entry, source, report = fixture()
    captured["fallback_public"][0]["text"] = "changed"
    with pytest.raises(ValueError, match="fallback identity"):
        verify_row(captured, entry, source, Tokenizer(), report)


def test_malformed_recorded_beam_is_rejected_by_shared_helper_contract():
    captured, entry, source, report = fixture()
    captured["generated_by_source"]["gim"] = [["한", -0.1]]
    with pytest.raises(ValueError, match="forward romanization"):
        verify_row(captured, entry, source, Tokenizer(), report)


def test_counter_tampering_and_unsupported_drop_are_rejected():
    captured, _, _, _ = fixture()
    counts = Counter(rows=1, parents=1, proposals=1, supported_proposals=1,
                     unsupported_proposals=0, generation_requests=1)
    verify_counts(counts, dict(counts), [captured])
    changed = dict(counts)
    changed["proposals"] = 2
    with pytest.raises(ValueError, match="counters"):
        verify_counts(counts, changed, [captured])
    dropped = Counter(counts)
    dropped["supported_proposals"] = 0
    with pytest.raises(ValueError, match="dropped"):
        verify_counts(dropped, dict(dropped), [captured])
