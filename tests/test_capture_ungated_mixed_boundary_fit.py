from hanglish.core.lattice import WordOption
from scripts.capture_ungated_mixed_boundary_fit import (
    RecordingOptions, capture_row, selected_audit_row,
)


def segment(source, start, end, text, kind):
    return {"input_start": start, "input_end": end, "source": source[start:end],
            "text": text, "kind": kind}


def audit(selected=True):
    return {"id": "s3-train-1", "request": {"input": "api3betasang", "context": ""},
            "source_shard": 0, "source_row_index": 1, "opaque_spans": [{
                "flags": ["digits"], "parents": [{"kind": "raw",
                "source_copied_exactly": selected}]}]}


def source():
    text = "api3betasang"
    converted = [segment(text, 0, 8, "api3beta", "raw"),
                 segment(text, 8, 12, "상", "lexicon")]
    raw = [segment(text, 0, len(text), text, "raw")]
    return {"id": "s3-train-1", "request": {"input": text, "context": ""},
            "response": {"input": text, "candidates": [
                {"text": "api3beta상", "segments": converted},
                {"text": text, "segments": raw}]}}


class Tokenizer:
    def __call__(self, text, **kwargs): return {"input_ids": list(range(len(text)))}
    def num_special_tokens_to_add(self, pair=True): return 2


class Ranker:
    tokenizer = Tokenizer(); max_length = 96
    def can_score(self, text): return len(text) + 2 <= self.max_length


def options(word, learned=True):
    values = {"api": WordOption("아피", 2.0, "lexicon"),
              "beta": WordOption("베타", 1.0, "lexicon")}
    return [values[word]] if word in values else [WordOption(word, 0, "raw")]


def test_selects_copied_raw_digit_in_nonreserve_parent():
    assert selected_audit_row(audit())
    assert not selected_audit_row(audit(False))
    row = audit(); row["opaque_spans"][0]["flags"] = ["uppercase"]
    assert not selected_audit_row(row)


def test_capture_excludes_raw_reserve_and_keeps_roundtrip_proposals():
    row = capture_row(audit(), source(), options, Ranker())
    assert len(row["preserved_parents"]) == 1
    assert all(parent["text"] != "api3betasang" for parent in row["preserved_parents"])
    assert all(any(segment["source"] == "3" and segment["text"] == "3"
                   for segment in proposal["segments"])
               for proposal in row["proposals"])
    assert all(proposal["token_support"]["supported"] for proposal in row["proposals"])
    assert row["callback_records"][0]["source"] == "api"


def test_unsupported_proposal_is_retained():
    ranker = Ranker(); ranker.max_length = 2
    row = capture_row(audit(), source(), options, ranker)
    assert row["proposals"]
    assert all(not proposal["token_support"]["supported"] for proposal in row["proposals"])


def test_recording_options_caches_and_serializes_finite_values():
    calls = []
    def callback(source, learned=True):
        calls.append(source); return [WordOption("상", 1.25, "lexicon")]
    recorder = RecordingOptions(callback)
    assert recorder("sang")[0].text == "상"
    assert recorder("sang")[0].text == "상"
    assert calls == ["sang"]
    assert recorder.calls == [{"source": "sang", "learned": True,
                                "options": [{"text": "상", "score": 1.25,
                                             "kind": "lexicon"}]}]


def test_preserved_parent_legacy_edge_need_not_roundtrip_as_whole_sentence():
    row = capture_row(audit(), source(), options, Ranker())
    # The inherited `sang` -> `상` edge is not revalidated by whole-sentence
    # romanization; only the newly changed raw `api3` chunk is forward-checked.
    assert row["preserved_parents"][0]["segments"][-1]["text"] == "상"
    assert row["proposals"]
