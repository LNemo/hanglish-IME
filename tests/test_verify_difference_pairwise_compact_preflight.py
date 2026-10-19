"""Pure mutation tests for the independent compact-kind preflight verifier."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest


PATH = (Path(__file__).resolve().parents[1] /
        "scripts/verify_difference_pairwise_compact_preflight.py")
SPEC = importlib.util.spec_from_file_location("compact_preflight_verifier", PATH)
v = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v)


class FakeTokenizer:
    chat_template = "fake-template"

    def encode(self, text, add_special_tokens=False):
        if text.endswith("1"):
            return self.encode(text[:-1]) + [16]
        if text.endswith("2"):
            return self.encode(text[:-1]) + [17]
        return [ord(char) % 1000 for char in text]

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt,
                            enable_thinking, return_dict=False):
        assert add_generation_prompt is True and enable_thinking is False
        rendered = "<s>" + messages[0]["content"] + "<u>" + messages[1]["content"] + "<a>"
        return self.encode(rendered) if tokenize else rendered


def semantic(current=None, alternative=None):
    left = (["lexicon", "mixed", "morphology", "neural",
             "numeric", "raw", "syllable"] if current is None else current)
    right = list(left) if alternative is None else alternative
    return {"roman_input": "imnida", "external_context": {"text": "", "strategy": "complete",
            "original_chars": 0}, "raw_mode": False, "mapping": {"1": "KEEP", "2": "SWITCH"},
            "CURRENT": {"source_index": 0, "text": "임니다", "score": 2.0,
                        "rank": 1, "kinds": left},
            "ALTERNATIVE": {"source_index": 1, "text": "입니다", "score": 1.0,
                            "rank": 2, "kinds": right},
            "changed_spans": [{"roman": "imnida", "current_kind": "morphology",
                               "alternative_kind": "morphology"}]}


def test_golden_codec_covers_every_code_and_empty_edits():
    source = semantic()
    compact = v.encode_payload(source)
    assert compact["K"] == {"c": "LXMNDRS", "a": ""}
    assert "kinds" not in compact["CURRENT"] and "kinds" not in compact["ALTERNATIVE"]
    assert v.canonical_bytes(v.decode_payload(compact)) == v.canonical_bytes(source)


def test_first_last_and_multiple_edits_are_canonical():
    source = semantic(alternative=["syllable", "mixed", "neural", "neural",
                                   "numeric", "raw", "lexicon"])
    compact = v.encode_payload(source)
    assert compact["K"] == {"c": "LXMNDRS", "a": "0=S,2=N,6=L"}
    assert v.decode_payload(compact) == source


def test_long_alternating_sequence_round_trips():
    current = ["lexicon", "raw"] * 100
    alternative = ["neural", "syllable"] * 100
    source = semantic(current, alternative)
    compact = v.encode_payload(source)
    assert len(compact["K"]["c"]) == 200
    assert v.decode_payload(compact) == source


def test_reserved_k_and_kind_length_errors_fail():
    source = semantic(); source["K"] = {}
    with pytest.raises(v.VerificationError, match="reserved K"):
        v.encode_payload(source)
    with pytest.raises(v.VerificationError, match="equal length"):
        v.encode_payload(semantic(["lexicon"], ["lexicon", "raw"]))
    with pytest.raises(v.VerificationError, match="nonempty"):
        v.encode_payload(semantic([], []))


@pytest.mark.parametrize("record", [
    {"c": "", "a": ""}, {"c": "l", "a": ""}, {"c": "한", "a": ""},
    {"c": "L", "a": "-1=N"}, {"c": "L", "a": "+0=N"},
    {"c": "L", "a": "00=N"}, {"c": "L", "a": "0.0=N"},
    {"c": "LL", "a": "1=N,1=S"}, {"c": "LL", "a": "1=N,0=S"},
    {"c": "L", "a": "1=N"}, {"c": "L", "a": "0N"},
    {"c": "L", "a": "0=N,"}, {"c": "L", "a": " 0=N"},
    {"c": "L", "a": "0=L"}, {"c": "L", "a": "0=Z"},
])
def test_malformed_compact_grammar_fails(record):
    compact = v.encode_payload(semantic(["lexicon"], ["neural"]))
    compact["K"] = record
    with pytest.raises(v.VerificationError):
        v.decode_payload(compact)


def test_missing_or_extra_k_fields_fail():
    compact = v.encode_payload(semantic())
    compact["K"] = {"c": "L"}
    with pytest.raises(v.VerificationError, match="exactly"):
        v.decode_payload(compact)
    compact["K"] = {"c": "L", "a": "", "x": "R"}
    with pytest.raises(v.VerificationError, match="exactly"):
        v.decode_payload(compact)


def test_omitted_changed_index_cannot_reconstruct_source_semantics():
    source = semantic(["lexicon", "raw"], ["neural", "syllable"])
    compact = v.encode_payload(source)
    compact["K"]["a"] = "0=N"
    decoded = v.decode_payload(compact)
    assert v.canonical_bytes(decoded) != v.canonical_bytes(source)


@pytest.mark.parametrize("field", ["roman_input", "external_context", "CURRENT",
                                    "ALTERNATIVE", "changed_spans"])
def test_semantic_field_mutation_changes_digest(field):
    source = semantic(); changed = deepcopy(source)
    if field == "roman_input": changed[field] += "x"
    elif field == "external_context": changed[field]["original_chars"] = 1
    elif field in ("CURRENT", "ALTERNATIVE"): changed[field]["rank"] += 1
    else: changed[field][0]["roman"] += "x"
    assert v.digest_value(changed) != v.digest_value(source)


def test_token_replay_records_payload_and_boundary_digests(monkeypatch):
    tokenizer = FakeTokenizer()
    monkeypatch.setattr(v, "CHAT_TEMPLATE_SHA256",
                        v.hashlib.sha256(tokenizer.chat_template.encode()).hexdigest())
    source = semantic(["lexicon"], ["neural"])
    compact = v.encode_payload(source)
    observations, stats, ordered = v._tokenize(
        {"pair-a": (source, compact, v.decode_payload(compact))}, tokenizer)
    assert observations[0]["pair_id"] == "pair-a"
    assert observations[0]["digit_continuation_ids"] == {"1": 16, "2": 17}
    assert stats["count"] == 1 and stats["overflow"] == 0
    assert ordered == v.digest_value([observations[0]["tokens_sha256"]])


def test_count_preserving_token_or_order_mutation_differs():
    tokens = [v.digest_value([1, 2]), v.digest_value([3, 4])]
    assert v.digest_value(tokens) != v.digest_value(list(reversed(tokens)))
    changed = [tokens[0], v.digest_value([3, 5])]
    assert v.digest_value(tokens) != v.digest_value(changed)


def test_source_pin_mutation_fails(monkeypatch):
    report = {"source_hashes_before": {"required": "a" * 64},
              "source_hashes_after": {"required": "b" * 64}}
    with pytest.raises(v.VerificationError, match="source hashes differ"):
        v._verify_source_pins(report)


def test_exit_two_writes_no_success_proof(tmp_path, monkeypatch):
    monkeypatch.setattr(v, "verify_preflight", lambda *_: {"passed": False})
    output = tmp_path / "proof.json"
    assert v.main(["--preflight", str(tmp_path / "report.json"),
                   "--output", str(output)]) == 2
    assert not output.exists()


def test_invalid_report_writes_no_proof(tmp_path):
    report = tmp_path / "report.json"; report.write_text("{}")
    output = tmp_path / "proof.json"
    assert v.main(["--preflight", str(report), "--output", str(output)]) == 1
    assert not output.exists()


def test_success_proof_is_canonical_and_immutable(tmp_path):
    output = tmp_path / "proof.json"; payload = {"passed": True, "stage": "compact"}
    envelope = v.write_proof(output, payload)
    assert json.loads(output.read_bytes()) == envelope
    assert envelope["digest"] == v.digest_value(payload)
    with pytest.raises(v.VerificationError, match="already exists"):
        v.write_proof(output, payload)
