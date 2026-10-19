"""CPU-only mutation tests for the independent pairwise data verifier."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts/verify_difference_pairwise_lora.py"
SPEC = importlib.util.spec_from_file_location("difference_pairwise_verifier", PATH)
v = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v)


def candidate(source, text, *, kind="lexicon", score=1.0):
    return {"text": text, "score": score, "segments": [{
        "input_start": 0, "input_end": len(source), "source": source,
        "text": text, "kind": kind,
    }]}


def segmented_pair():
    source = "na test 12!"
    parent = {"text": "나 test 12!", "score": 1.0, "segments": [
        {"input_start": 0, "input_end": 2, "source": "na", "text": "나", "kind": "lexicon"},
        {"input_start": 2, "input_end": 8, "source": " test ", "text": " test ", "kind": "raw"},
        {"input_start": 8, "input_end": 11, "source": "12!", "text": "12!", "kind": "raw"},
    ]}
    alternative = deepcopy(parent)
    alternative["text"] = "너 test 12!"
    alternative["segments"][0]["text"] = "너"
    alternative["segments"][0]["kind"] = "neural"
    return source, parent, alternative


def test_pair_and_unit_id_are_canonical_and_domain_separated():
    identity = v.pair_id("natural_fit", "row-a", 3)
    assert identity == v.digest_value({"role": "natural_fit", "row_id": "row-a",
                                       "current_source_index": 0,
                                       "alternative_source_index": 3})
    assert v.unit_id(identity) != identity
    assert v.unit_id(identity) == v.unit_id(identity)


def test_changed_span_windows_and_signature_are_reconstructed(monkeypatch):
    source, parent, alternative = segmented_pair()
    monkeypatch.setattr(v, "romanize_word", lambda text: {"너": "na"}[text])
    assert v.proposal_allowed(source, parent, alternative)
    spans = v.changed_spans(source, parent, alternative)
    assert spans == [{
        "segment_index": 0, "input_start": 0, "input_end": 2, "roman": "na",
        "left_context": "", "right_context": " test 12!", "current_text": "나",
        "current_kind": "lexicon", "alternative_text": "너", "alternative_kind": "neural",
    }]
    assert v.match_key(source, " 문맥 ", parent, alternative) == {
        "shape": "sentence", "context": "present", "span_count": "one",
        "roman_length": "1_2", "kind_signature": "lexicon>neural",
        "neighbor_class": "mixed",
    }


@pytest.mark.parametrize("mutation", ("misaligned", "protected", "nonroman", "unchanged"))
def test_proposal_predicate_rejects_invalid_alternatives(monkeypatch, mutation):
    source = "na"
    parent, alternative = candidate(source, "나"), candidate(source, "너", kind="neural")
    monkeypatch.setattr(v, "romanize_word", lambda text: "na")
    if mutation == "misaligned":
        alternative["segments"][0]["input_start"] = 1
    elif mutation == "protected":
        alternative["segments"][0]["kind"] = "raw"
    elif mutation == "nonroman":
        alternative["segments"][0]["source"] = "n1"
    else:
        alternative = deepcopy(parent)
    assert not v.proposal_allowed(source, parent, alternative)


def key(**changes):
    value = {"shape": "single", "context": "present", "span_count": "one",
             "roman_length": "3_4", "kind_signature": "lexicon>neural",
             "neighbor_class": "none"}
    value.update(changes)
    return value


@pytest.mark.parametrize("depth,change,expected", [
    (0, {}, True), (0, {"context": "empty"}, False),
    (1, {"context": "empty"}, True),
    (1, {"roman_length": "5_8"}, False),
    (2, {"context": "empty", "roman_length": "5_8"}, True),
    (3, {"neighbor_class": "latin"}, True),
    (4, {"shape": "sentence"}, True),
    (5, {"kind_signature": "neural>lexicon"}, True),
])
def test_matching_relaxes_only_in_the_frozen_order(depth, change, expected):
    assert v.matches_at_depth(key(), key(**change), depth) is expected


def pair(identity, signature):
    return {"pair_id": identity, "matching_key": signature}


def test_control_allocation_is_hash_ordered_unused_first(monkeypatch):
    pool = [pair("a", key()), pair("b", key())]
    uses = v.Counter()
    monkeypatch.setattr(v, "control_order", lambda unit, identity: {"a": "2", "b": "1"}[identity])
    chosen, proof = v.allocate_control("unit", key(), pool, uses)
    assert chosen["pair_id"] == "b"
    assert proof == {"relaxation_depth": 0, "prior_use_count": 0,
                     "reuse_after_unused_exhaustion": False}
    chosen, proof = v.allocate_control("unit", key(), pool, uses)
    assert chosen["pair_id"] == "a" and proof["prior_use_count"] == 0
    chosen, proof = v.allocate_control("unit", key(), pool, uses)
    assert chosen["pair_id"] == "b"
    assert proof == {"relaxation_depth": 0, "prior_use_count": 1,
                     "reuse_after_unused_exhaustion": True}


def test_control_pool_excludes_keep_pair_from_parent_wrong_row():
    pairs = {
        "valid": {"role": "natural_fit", "parent_accepted": True},
        "wrong-keep": {"role": "natural_fit", "parent_accepted": False},
        "challenge": {"role": "challenge_fit", "parent_accepted": True},
    }
    pools = v._control_pools(pairs)
    assert pools["natural_fit"] == [pairs["valid"]]
    assert pools["challenge_fit"] == [pairs["challenge"]]


def test_required_pair_set_rejects_omitted_nontraining_pair():
    expected_scopes = {"training": {"training", "mechanism"},
                       "eval-only": {"train640"}}
    with pytest.raises(v.VerificationError, match="omits or adds"):
        v._verify_pairs([], {}, {"natural_fit": [], "challenge_fit": []}, expected_scopes)


def test_unit_weights_and_references_are_independently_checked():
    pairs = {name: {"pair_id": name} for name in ("s", "n", "c", "w")}
    targets = [{"pair_id": "s", "target": "SWITCH"}] + [
        {"pair_id": name, "target": "KEEP"} for name in ("n", "c", "w")]
    expected = [{"unit_id": "u", "switch_pair_id": "s", "references": [
                     {"component": "repair_switch", "pair_id": "s", "target": "SWITCH", "weight": .5},
                     {"component": "natural_keep", "pair_id": "n", "target": "KEEP", "weight": 1 / 6,
                      "match": {"relaxation_depth": 0}},
                     {"component": "challenge_keep", "pair_id": "c", "target": "KEEP", "weight": 1 / 6,
                      "match": {"relaxation_depth": 1}},
                     {"component": "same_row_keep", "pair_id": "w", "target": "KEEP", "weight": 1 / 6,
                      "match": {"same_row": True}},
                 ]}]
    # The production verifier additionally enforces 59/232 aggregate counts;
    # use 59 copies with distinct IDs so this test reaches the weight checks.
    units, wanted, all_pairs, all_targets = [], [], {}, []
    for index in range(59):
        suffix = str(index)
        mapped = {name: name + suffix for name in pairs}
        item = deepcopy(expected[0])
        item["switch_pair_id"] = mapped[item["switch_pair_id"]]
        item["unit_id"] = v.unit_id(item["switch_pair_id"])
        item["order"] = index + 1
        for component in item["references"]:
            component["pair_id"] = mapped[component["pair_id"]]
            if component["component"] == "same_row_keep":
                component["match"] = {"same_row": True, "relaxation_depth": None,
                                      "prior_use_count": 0,
                                      "reuse_after_unused_exhaustion": False}
        units.append(deepcopy(item)); wanted.append(deepcopy(item))
        for name in mapped.values(): all_pairs[name] = {"pair_id": name}
        all_targets.extend({"pair_id": mapped[row["pair_id"]], "target": row["target"]} for row in targets)
    with pytest.raises(v.VerificationError, match="component reference target counts"):
        v._verify_units(units, wanted, all_pairs, all_targets)
    # 59 four-component units intentionally have 177 KEEP references, proving
    # the global 173 KEEP contract rejects an otherwise locally valid layout.


def test_schedule_requires_14_full_updates_and_tail_three():
    units = [{"unit_id": f"u-{n}"} for n in range(59)]
    rows = [{"unit_order": n + 1, "unit_id": unit["unit_id"],
             "update": n // 4 + 1, "position_in_update": n % 4 + 1}
            for n, unit in enumerate(units)]
    v._verify_schedule(rows, units)
    rows[-1]["position_in_update"] = 4
    with pytest.raises(v.VerificationError, match="schedule differs"):
        v._verify_schedule(rows, units)


def test_duplicate_valid_unit_identity_is_rejected_before_lookup():
    identity = v.unit_id("switch")
    units = [{"unit_id": identity, "switch_pair_id": "switch", "order": n + 1,
              "references": []} for n in range(59)]
    expected = [{"unit_id": f"expected-{n}", "switch_pair_id": f"s-{n}",
                 "references": []} for n in range(59)]
    with pytest.raises(v.VerificationError, match="ordered unit identities"):
        v._verify_units(units, expected, {}, [])


def test_same_row_match_requires_explicit_fixed_metadata():
    exact = {"same_row": True, "relaxation_depth": None, "prior_use_count": 0,
             "reuse_after_unused_exhaustion": False}
    v._verify_same_row_match(exact)
    for field, replacement in (("same_row", False), ("relaxation_depth", 0),
                               ("prior_use_count", 1),
                               ("reuse_after_unused_exhaustion", True)):
        changed = deepcopy(exact)
        changed[field] = replacement
        with pytest.raises(v.VerificationError, match="same-row"):
            v._verify_same_row_match(changed)


def test_missing_source_context_uses_exact_128_plus_128_metadata():
    context = "가" * 300 + "나" * 300
    assert v.external_context_window(context, "not-present") == {
        "head": "가" * 128, "tail": "나" * 128, "strategy": "head_tail_gap",
        "original_chars": 600, "source_chars": 256, "gap_marker": "[…중략…]",
    }


def compact_semantic():
    return {
        "roman_input": "na test", "external_context": {"text": "", "strategy": "complete",
                                                           "original_chars": 0},
        "raw_mode": False, "mapping": {"1": "KEEP", "2": "SWITCH"},
        "CURRENT": {"source_index": 0, "text": "나 test", "score": 1.0, "rank": 1,
                    "kinds": ["lexicon", "raw"]},
        "ALTERNATIVE": {"source_index": 1, "text": "너 test", "score": .5, "rank": 2,
                        "kinds": ["neural", "raw"]},
        "changed_spans": [{"segment_index": 0}],
    }


def test_compact_kind_codec_is_lossless_and_canonical():
    semantic = compact_semantic()
    compact = v.encode_compact_payload(semantic)
    assert compact["K"] == {"c": "LR", "a": "0=N"}
    assert "kinds" not in compact["CURRENT"] and "kinds" not in compact["ALTERNATIVE"]
    assert v.canonical_bytes(v.decode_compact_payload(compact)) == v.canonical_bytes(semantic)


@pytest.mark.parametrize("mutation", [
    lambda row: row.update(K={"c": "LR", "a": "0=N,0=L"}),
    lambda row: row.update(K={"c": "LR", "a": "1=R"}),
    lambda row: row.update(K={"c": "L?", "a": ""}),
    lambda row: row["CURRENT"].update(kinds=["lexicon", "raw"]),
])
def test_compact_kind_decoder_rejects_malformed_or_noncanonical_values(mutation):
    compact = v.encode_compact_payload(compact_semantic())
    mutation(compact)
    with pytest.raises(v.VerificationError):
        v.decode_compact_payload(compact)


def test_compact_kind_encoder_rejects_reserved_K_collision():
    semantic = compact_semantic()
    semantic["K"] = {"c": "L", "a": ""}
    with pytest.raises(v.VerificationError, match="reserved K"):
        v.encode_compact_payload(semantic)


class FakeTokenizer:
    chat_template = "fake"

    def apply_chat_template(self, messages, *, tokenize, **_kwargs):
        rendered = v.canonical_bytes(messages).decode()
        return self.encode(rendered, add_special_tokens=False) if tokenize else rendered

    def encode(self, value, *, add_special_tokens=False):
        assert add_special_tokens is False
        if value.endswith("1"):
            return [ord(char) for char in value[:-1]] + [16]
        if value.endswith("2"):
            return [ord(char) for char in value[:-1]] + [17]
        return [ord(char) for char in value]


def compact_pair_record():
    semantic = compact_semantic()
    return {
        "pair_id": "p", "source": semantic["roman_input"],
        "external_context": semantic["external_context"], "raw_mode": False,
        "current_source_index": 0, "alternative_source_index": 1,
        "current": {key: value for key, value in semantic["CURRENT"].items()
                    if key != "source_index"},
        "alternative": {key: value for key, value in semantic["ALTERNATIVE"].items()
                        if key != "source_index"},
        "changed_spans": semantic["changed_spans"], "_expected_target": "KEEP",
    }


def compact_prompt_fixture():
    pair = compact_pair_record()
    semantic = v.semantic_payload(pair)
    compact = v.encode_compact_payload(semantic)
    decoded = v.decode_compact_payload(compact)
    messages = v.expected_messages(pair)
    tokenizer = FakeTokenizer()
    tokens = tokenizer.apply_chat_template(messages, tokenize=True,
                                           add_generation_prompt=True,
                                           enable_thinking=False, return_dict=False)
    rendered = tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True,
                                             enable_thinking=False)
    hashes = {"semantic": v.digest_value(semantic), "compact": v.digest_value(compact),
              "decoded": v.digest_value(decoded), "tokens": v.digest_value(tokens)}
    prompt = {"pair_id": "p", "messages": messages,
              "semantic_payload_sha256": hashes["semantic"],
              "compact_payload_sha256": hashes["compact"],
              "decoded_payload_sha256": hashes["decoded"],
              "payload_sha256": hashes["compact"],
              "rendered_prefix_sha256": v.hashlib.sha256(rendered.encode()).hexdigest()}
    token = {"pair_id": "p", "tokens": tokens, "tokens_sha256": hashes["tokens"],
             "prompt_tokens": len(tokens), "digit_continuation_ids": {"1": 16, "2": 17}}
    target = {"pair_id": "p", "target": "KEEP", "target_digit": 1}
    observation = {"pair_id": "p", "semantic_sha256": hashes["semantic"],
                   "compact_sha256": hashes["compact"], "decoded_sha256": hashes["decoded"],
                   "tokens_sha256": hashes["tokens"], "token_count": len(tokens),
                   "digit_continuation_ids": {"1": 16, "2": 17}}
    preflight = {"observations": [observation],
                 "ordered_pair_ids_sha256": v.digest_value(["p"])}
    return pair, prompt, token, target, preflight, tokenizer


def test_prompt_verifier_checks_compact_hashes_and_ordered_digests():
    pair, prompt, token, target, preflight, tokenizer = compact_prompt_fixture()
    stats, digests = v._verify_prompt_and_tokens(
        [pair], [prompt], [token], [target], tokenizer, preflight)
    assert stats["count"] == 1 and stats["overflow"] == 0
    assert digests == {
        "ordered_semantic_sha256": v.digest_value([prompt["semantic_payload_sha256"]]),
        "ordered_compact_sha256": v.digest_value([prompt["compact_payload_sha256"]]),
        "ordered_decoded_sha256": v.digest_value([prompt["decoded_payload_sha256"]]),
        "ordered_token_sha256": v.digest_value([token["tokens_sha256"]]),
    }


@pytest.mark.parametrize("field", ["semantic_payload_sha256", "compact_payload_sha256",
                                    "decoded_payload_sha256", "payload_sha256"])
def test_prompt_verifier_rejects_each_payload_digest_mutation(field):
    pair, prompt, token, target, preflight, tokenizer = compact_prompt_fixture()
    prompt[field] = "0" * 64
    with pytest.raises(v.VerificationError, match="prompt semantic"):
        v._verify_prompt_and_tokens([pair], [prompt], [token], [target], tokenizer, preflight)


def test_prompt_verifier_rejects_preflight_observation_mutation():
    pair, prompt, token, target, preflight, tokenizer = compact_prompt_fixture()
    preflight["observations"][0]["compact_sha256"] = "0" * 64
    with pytest.raises(v.VerificationError, match="successful compact preflight"):
        v._verify_prompt_and_tokens([pair], [prompt], [token], [target], tokenizer, preflight)


def test_global_ordered_digest_mutation_is_rejected():
    actual = {name: str(index) * 64 for index, name in enumerate((
        "ordered_semantic_sha256", "ordered_compact_sha256",
        "ordered_decoded_sha256", "ordered_token_sha256"), 1)}
    manifest = {"payload_digests": deepcopy(actual)}
    v._verify_payload_digest_summary(manifest, actual)
    manifest["payload_digests"]["ordered_token_sha256"] = "0" * 64
    with pytest.raises(v.VerificationError, match="ordered payload"):
        v._verify_payload_digest_summary(manifest, actual)


def test_manifest_and_bridge_lineage_mutations_are_rejected():
    preflight = {"ordered_pair_ids_sha256": "a" * 64,
                 "ordered_token_digest": "b" * 64}
    proof, archive = {"digest": "c" * 64}, {"files": [{}, {}]}
    expected = v._expected_compact_lineage(preflight, proof, archive, "d" * 64)
    bridge = {
        "intent": expected["intent"],
        "preflight": {**expected["preflight"], **preflight},
        "preflight_proof": expected["preflight_proof"],
        "archive_manifest": expected["archive_manifest"],
    }
    manifest = {"compact_lineage": deepcopy(expected)}
    v._verify_lineage_records(manifest, bridge, expected, preflight)
    manifest["compact_lineage"]["bridge"]["sha256"] = "0" * 64
    with pytest.raises(v.VerificationError, match="manifest compact lineage"):
        v._verify_lineage_records(manifest, bridge, expected, preflight)
    manifest["compact_lineage"] = deepcopy(expected)
    bridge["preflight"]["ordered_token_digest"] = "0" * 64
    with pytest.raises(v.VerificationError, match="compact bridge lineage"):
        v._verify_lineage_records(manifest, bridge, expected, preflight)


def test_bridge_torch_provenance_does_not_require_torch_in_tokenizer_environment(monkeypatch):
    predecessor = v.read_json(v.STRICT_FIT_BRIDGE)
    bridge = {"torch_environment_required": True,
              "torch_version": predecessor["torch_version"]}

    def no_torch_distribution(name):
        if name == "torch":
            raise v.importlib.metadata.PackageNotFoundError(name)
        return "unused"

    monkeypatch.setattr(v.importlib.metadata, "version", no_torch_distribution)
    v._verify_bridge_torch_provenance(bridge)
    bridge["torch_version"] = "0.0.0"
    with pytest.raises(v.VerificationError, match="Torch provenance"):
        v._verify_bridge_torch_provenance(bridge)


def test_identity_digests_bind_ordered_pairs_and_produced_unit_bytes():
    pairs = [{"pair_id": "b"}, {"pair_id": "a"}]
    units = [{"unit_id": "u", "references": [{"pair_id": "a"}]}]
    expected = {"ordered_pair_ids_sha256": v.digest_value(["a", "b"]),
                "produced_units_sha256": v.digest_value(units)}
    manifest = {"identity_digests": deepcopy(expected)}
    assert v._verify_identity_digests(manifest, pairs, units) == expected
    manifest["identity_digests"]["ordered_pair_ids_sha256"] = "0" * 64
    with pytest.raises(v.VerificationError, match="identity digests"):
        v._verify_identity_digests(manifest, pairs, units)
    manifest["identity_digests"] = deepcopy(expected)
    mutated_units = deepcopy(units)
    mutated_units[0]["references"][0]["pair_id"] = "b"
    with pytest.raises(v.VerificationError, match="identity digests"):
        v._verify_identity_digests(manifest, pairs, mutated_units)


def test_proof_is_canonical_digest_envelope_and_no_overwrite(tmp_path):
    output = tmp_path / "proof.json"
    payload = {"stage": "data", "passed": True}
    envelope = v.write_proof(output, payload)
    assert envelope["schema"] == v.SCHEMA
    assert envelope["digest"] == v.digest_value(payload)
    with pytest.raises(v.VerificationError, match="already exists"):
        v.write_proof(output, payload)


def test_later_stage_flags_fail_closed(tmp_path):
    assert v.main(["--data", str(tmp_path / "missing"), "--output", str(tmp_path / "proof"),
                   "--resource", str(tmp_path / "resource")]) == 1
    assert not (tmp_path / "proof").exists()


def test_tokenizer_overflow_report_cannot_be_certified_as_data(tmp_path):
    report = tmp_path / "report.json"
    report.write_text('{"schema":"difference-pairwise-lora-data-overflow-v1",'
                      '"completed":true,"terminal_outcome":"tokenizer_ceiling_rejected"}')
    output = tmp_path / "proof.json"
    assert v.main(["--data", str(report), "--output", str(output)]) == 1
    assert not output.exists()
