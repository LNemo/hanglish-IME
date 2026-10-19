import hashlib
import json
from pathlib import Path
import re

import pytest

from hanglish.core.ctc_word_proposals import aligned_words, sentence_extras


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / ".omx/artifacts/mixed-span-sentence-dev-v1/capture.jsonl"
PROPOSALS = ROOT / ".omx/artifacts/ctc-word-candidates-dev-v1/proposals.jsonl"


def paths(source):
    chunks = list(re.finditer(r"\S+|\s+", source))
    result = []
    for number in range(4):
        segments = []
        for index, match in enumerate(chunks):
            text = match.group()
            if index == 0:
                text = f"기준{number}"
            segments.append({"input_start": match.start(), "input_end": match.end(),
                             "source": match.group(), "text": text, "kind": "fixture"})
        result.append(("".join(item["text"] for item in segments),
                       float(4 - number), segments))
    return result


def test_mixed_alignment_is_leftmost_and_preserves_literals():
    repeated = aligned_words("gana gana", "가나")
    assert [(item["source_chunk_index"], item["output_chunk_index"])
            for item in repeated] == [(0, 0)]
    assert aligned_words("Ksaneun je13hoe", "K사는 제13회") == [
        {"source_chunk_index": 0, "output_chunk_index": 0,
         "input_start": 0, "input_end": 7, "source": "Ksaneun", "text": "K사는",
         "forward_roman": "Ksaneun", "literal_signature": "K"},
        {"source_chunk_index": 1, "output_chunk_index": 1,
         "input_start": 8, "input_end": 15, "source": "je13hoe", "text": "제13회",
         "forward_roman": "je13hoe", "literal_signature": "13"},
    ]
    assert aligned_words("Ksaneun", "X사는") == []


@pytest.mark.parametrize("source", [
    "https://a.kr", "a@b.kr", "`code`", "a_b", "a/b", "a\\b",
    "v1.2.3", "file.txt", "AB12", "code12",
])
def test_protected_chunks_never_align(source):
    assert aligned_words(source, "가") == []


def test_candidate_cap_parent_retention_and_segment_kind():
    source = " ".join(["ga"] * 30)
    parent = paths(source)
    frozen = json.loads(json.dumps(parent))
    # JSON round-trip changes tuples to lists, so compare the serializable form.
    extras = sentence_extras(source, " ".join(["가"] * 30), parent)
    assert len(extras) == 96
    assert json.loads(json.dumps(parent)) == frozen
    assert not ({item[0] for item in extras} & {item[0] for item in parent})
    assert all(any(segment["kind"] == "ctc_mixed_word_probe"
                   for segment in item[2]) for item in extras)


def test_malformed_inputs_and_duplicate_pool_are_rejected():
    assert aligned_words(None, "가") == []
    with pytest.raises(ValueError, match="duplicates"):
        sentence_extras("ga", "가", [paths("ga")[0]] * 4)
    with pytest.raises(ValueError, match="limit"):
        sentence_extras("ga", "가", paths("ga"), limit=97)


def test_frozen_mixed_probe_parity_all_316_rows():
    assert hashlib.sha256(BASELINE.read_bytes()).hexdigest() == (
        "ec4a9450d9a878b4c989d782bf546ddf71aebd773fa842293b3c71d973ffd0ba")
    assert hashlib.sha256(PROPOSALS.read_bytes()).hexdigest() == (
        "cff2f538c7e1a9379081b486a4646381a6b0cfd6eb64393fd9249d779072e225")
    count = 0
    with BASELINE.open() as baseline, PROPOSALS.open() as proposals:
        for base_line, proposal_line in zip(baseline, proposals, strict=True):
            base, proposed = json.loads(base_line), json.loads(proposal_line)
            assert base["id"] == proposed["id"] and base["input"] == proposed["input"]
            parent = [(item["text"], float(item["score"] or 0.0), item["segments"])
                      for item in base["scored_pool"]]
            extras = sentence_extras(base["input"], proposed["prediction"]["text"], parent)
            expected = [(item["text"], item["segments"])
                        for item in proposed["mixed"]["proposals"]]
            assert [(item[0], item[2]) for item in extras] == expected
            count += 1
    assert count == 316
