from copy import deepcopy

import pytest

from hanglish.core.lattice import WordOption
from scripts import verify_ungated_mixed_boundary_fit as verifier


class Tokenizer:
    def __init__(self, lengths=None, default_length=None):
        self.lengths = lengths or {}
        self.default_length = default_length
        self.calls = []

    def __call__(self, text, *, add_special_tokens, truncation):
        self.calls.append((text, add_special_tokens, truncation))
        length = self.lengths.get(
            text, len(text) if self.default_length is None else self.default_length)
        return {"input_ids": list(range(length))}

    def num_special_tokens_to_add(self, *, pair):
        assert pair is True
        return 3


def segment(source, start, end, text, kind):
    return {"input_start": start, "input_end": end, "source": source[start:end],
            "text": text, "kind": kind}


def fixture(tokenizer=None):
    source = "api3betasang"
    converted = [segment(source, 0, 8, "api3beta", "raw"),
                 segment(source, 8, 12, "상", "legacy")]
    raw = [segment(source, 0, 12, source, "raw")]
    source_row = {"id": "s3-train-1",
                  "request": {"input": source, "context": ""},
                  "response": {"input": source, "candidates": [
                      {"text": "api3beta상", "segments": converted},
                      {"text": source, "segments": raw}]}}
    audit = {"id": source_row["id"], "request": source_row["request"],
             "source_shard": 0, "source_row_index": 1,
             "opaque_spans": [{"flags": ["digits"], "parents": [
                 {"kind": "raw", "source_copied_exactly": True}]}]}
    parents = verifier.public_parents(source_row)
    records = [{"source": "api", "learned": True, "options": [
        {"text": "아피", "score": 2.0, "kind": "fixture"}]},
        {"source": "beta", "learned": True, "options": [
        {"text": "베타", "score": 1.0, "kind": "fixture"}]}]
    values, _ = verifier.validate_callback_records(records)
    callback = verifier.SavedOptions(values)
    from hanglish.core.mixed_span_proposals import sentence_extras
    extras = sentence_extras(
        source, [(row["text"], row["score"], deepcopy(row["segments"]))
                 for row in parents], callback, limit=96)
    tokenizer = tokenizer or Tokenizer()
    proposals = [{"text": text, "score": score, "segments": segments,
                  "token_support": verifier.token_support(tokenizer, text)}
                 for text, score, segments in extras]
    captured = {"id": audit["id"], "request": audit["request"],
                "source_shard": 0, "source_row_index": 1,
                "opaque_spans": deepcopy(audit["opaque_spans"]),
                "preserved_parents": parents, "callback_records": records,
                "proposals": proposals, "proposal_count": len(proposals),
                "zero_proposal": not proposals,
                "protected_or_no_option": not proposals,
                "elapsed_seconds": 0.1}
    return audit, source_row, captured, tokenizer


def test_replays_saved_options_without_requiring_inherited_whole_roundtrip():
    audit, source, captured, tokenizer = fixture()
    counts = verifier.replay_row(audit, source, captured, tokenizer)
    assert counts["proposals"] == len(captured["proposals"]) > 0
    assert captured["preserved_parents"][0]["segments"][-1]["text"] == "상"
    assert all(any(segment["source"] == "3" and segment["text"] == "3"
                   and segment["kind"] == "raw"
                   for segment in proposal["segments"])
               for proposal in captured["proposals"])


def test_tokenizer_is_untruncated_and_unsupported_is_retained():
    tokenizer = Tokenizer(default_length=94)
    audit, source, captured, tokenizer = fixture(tokenizer)
    counts = verifier.replay_row(audit, source, captured, tokenizer)
    assert counts["unsupported"] == counts["proposals"]
    assert captured["proposals"]
    assert all(not row["token_support"]["supported"]
               for row in captured["proposals"])
    assert all(not special and truncation is False
               for _, special, truncation in tokenizer.calls)


def test_rejects_callback_nonfinite_and_uncaptured_requests():
    with pytest.raises(ValueError, match="WordOption"):
        verifier.validate_callback_records([{"source": "api", "learned": True,
            "options": [{"text": "에이피", "score": float("nan"), "kind": "x"}]}])
    callback = verifier.SavedOptions({"api": [WordOption("에이피", 1.0, "x")]})
    with pytest.raises(ValueError, match="uncaptured"):
        callback("missing")


def test_rejects_changed_raw_copy_or_nonforward_new_edge():
    audit, source, captured, tokenizer = fixture()
    broken = deepcopy(captured)
    raw = next(segment for segment in broken["proposals"][0]["segments"]
               if segment["kind"] == "raw")
    raw["text"] = "X"
    with pytest.raises(ValueError, match="raw COPY|render|proposal"):
        verifier.replay_row(audit, source, broken, tokenizer)


def test_selection_requires_digit_and_exact_raw_parent_copy():
    audit, _, _, _ = fixture()
    assert verifier.selected_audit_row(audit)
    audit["opaque_spans"][0]["parents"][0]["source_copied_exactly"] = False
    assert not verifier.selected_audit_row(audit)
    audit["opaque_spans"][0]["parents"][0]["source_copied_exactly"] = True
    audit["opaque_spans"][0]["flags"] = ["uppercase"]
    assert not verifier.selected_audit_row(audit)
