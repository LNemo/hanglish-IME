import itertools
import json
import math

import pytest
import torch

from hanglish.models.sentence_ctc import (
    CTC_VOCAB,
    SentenceCTC,
    SentenceCTCConfig,
    collapse_ctc,
    ctc_labels,
    ctc_prefix_beam,
    decode_ctc_labels,
    required_ctc_frames,
    subframe_memory,
    training_batch,
    valid_label_prefix,
)
from hanglish.models.sentence_transducer import (
    BOS, BYTE_OFFSET, CODA_OFFSET, EOS, ONSET_OFFSET, PAD, VOWEL_OFFSET)


TINY = SentenceCTCConfig(
    width=12, heads=2, encoder_layers=1, feedforward=24, dropout=0.,
    max_source_tokens=32, embedding_scale=1., beam_size=8, token_shortlist=8)
A = BYTE_OFFSET + ord("a") - 2


def test_adjacent_repeat_requires_blank_and_feasibility_audit():
    assert collapse_ctc([A, A]) == [A]
    assert collapse_ctc([A, 0, A]) == [A, A]
    assert required_ctc_frames([A, A, A]) == 5
    source, targets, input_lengths, target_lengths = training_batch(
        [("a a", "aa")], TINY)
    assert source.shape == (1, 5)
    assert targets.tolist() == [A, A]
    assert input_lengths.tolist() == [9]
    assert target_lengths.tolist() == [2]
    with pytest.raises(ValueError, match="frame budget"):
        training_batch([("a a", "aaaaaa")], TINY)


def test_training_batch_excludes_bos_eos_and_audits_complete_limits():
    source, _, input_lengths, _ = training_batch(
        [("a a", "가"), ("ab cd", "나")], TINY)
    assert source.shape == (2, 7)
    assert source[0, 0] == BOS and source[0, 4] == EOS
    assert source[0, 5:].tolist() == [PAD, PAD]
    assert source[1, 0] == BOS and source[1, 6] == EOS
    assert input_lengths.tolist() == [9, 15]
    with pytest.raises(ValueError, match="source exceeds"):
        training_batch([("a" * 29 + " b", "가")], TINY)
    with pytest.raises(ValueError, match="511-token"):
        training_batch([("a a", "a" * 512)], TINY)


def test_subframes_use_only_raw_byte_positions_and_keep_exact_lengths():
    memory = torch.arange(2 * 7 * 2, dtype=torch.float32).reshape(2, 7, 2)
    offsets = torch.zeros(3, 2)
    frames, lengths = subframe_memory(memory, torch.tensor([5, 7]), offsets)
    assert frames.shape == (2, 15, 2)
    assert lengths.tolist() == [9, 15]
    assert torch.equal(frames[0, 0], memory[0, 1])
    assert torch.equal(frames[0, 2], memory[0, 1])
    assert torch.equal(frames[0, 3], memory[0, 2])
    assert torch.equal(frames[0, 8], memory[0, 3])
    assert not torch.equal(frames[0, 0], memory[0, 0])
    assert not torch.equal(frames[0, 8], memory[0, 4])


def test_strict_decoder_reports_invalid_utf8_and_mandatory_jamo_parts():
    invalid_utf8 = BYTE_OFFSET + 255 - 2
    onset = ONSET_OFFSET - 2
    vowel = VOWEL_OFFSET - 2
    coda = CODA_OFFSET - 2
    assert decode_ctc_labels([invalid_utf8])["reason"] == "invalid_utf8"
    assert decode_ctc_labels([onset])["reason"] == "invalid_jamo"
    assert decode_ctc_labels([onset, vowel])["reason"] == "invalid_jamo"
    assert decode_ctc_labels([])["reason"] == "empty_output"
    assert decode_ctc_labels([onset, vowel, coda]) == {
        "text": "가", "reason": "complete", "generated_tokens": 3}
    assert valid_label_prefix([onset])
    assert not valid_label_prefix([onset], final=True)
    assert valid_label_prefix([onset, vowel])
    assert not valid_label_prefix([onset, vowel], final=True)
    assert valid_label_prefix([onset, vowel, coda], final=True)


def test_literal_prefix_requires_valid_utf8_boundaries():
    # 0xEA starts a valid three-byte UTF-8 prefix but cannot be closed by an
    # onset token until both continuation bytes have arrived.
    lead = BYTE_OFFSET + 0xEA - 2
    continuation_1 = BYTE_OFFSET + 0xB0 - 2
    continuation_2 = BYTE_OFFSET + 0x80 - 2
    onset = ONSET_OFFSET - 2
    assert valid_label_prefix([lead])
    assert not valid_label_prefix([lead], final=True)
    assert not valid_label_prefix([lead, onset])
    assert valid_label_prefix([lead, continuation_1, continuation_2], final=True)


def test_prefix_beam_matches_exhaustive_ctc_blank_repeat_distribution():
    probabilities = ((.4, .6), (.5, .5), (.3, .7))
    log_probs = torch.full((3, CTC_VOCAB), -torch.inf, dtype=torch.float64)
    for frame, (blank, label) in enumerate(probabilities):
        log_probs[frame, 0] = math.log(blank)
        log_probs[frame, A] = math.log(label)
    beams, audit = ctc_prefix_beam(log_probs, beam_size=16, token_shortlist=1)
    actual = {tuple(item["labels"]): math.exp(item["log_probability"])
              for item in beams}
    expected = {}
    for path in itertools.product((0, A), repeat=3):
        probability = math.prod(probabilities[index][0 if token == 0 else 1]
                                for index, token in enumerate(path))
        prefix = tuple(collapse_ctc(path))
        expected[prefix] = expected.get(prefix, 0.) + probability
    assert actual.keys() == expected.keys()
    for prefix, probability in expected.items():
        assert actual[prefix] == pytest.approx(probability, abs=1e-12)
    assert audit == {
        "frames": 3, "beam_size": 16, "token_shortlist": 1,
        "max_label_tokens": 511,
        "shortlist_pruned": True, "beam_pruned": False,
        "repeat_tokens_restored": 0,
        "approximate_pruning": True}


def test_prefix_beam_restores_repeat_mass_outside_global_shortlist():
    b = A + 1
    log_probs = torch.full((2, CTC_VOCAB), -torch.inf, dtype=torch.float64)
    log_probs[0, 0], log_probs[0, A] = math.log(.1), math.log(.9)
    log_probs[1, 0] = math.log(.1)
    log_probs[1, A], log_probs[1, b] = math.log(.2), math.log(.7)
    beams, audit = ctc_prefix_beam(log_probs, beam_size=16, token_shortlist=1)
    scores = {tuple(item["labels"]): math.exp(item["log_probability"])
              for item in beams}
    # A→blank and A→A both collapse to the existing A prefix.  The latter is
    # retained even though B is the frame's global top-one nonblank token.
    assert scores[(A,)] == pytest.approx(.9 * .1 + .9 * .2)
    assert audit["repeat_tokens_restored"] >= 1


def test_prefix_beam_caps_labels_but_keeps_blank_and_repeat_mass():
    b = A + 1
    log_probs = torch.full((512, CTC_VOCAB), -torch.inf, dtype=torch.float64)
    for frame in range(511):
        log_probs[frame, A if frame % 2 == 0 else b] = 0.
    # Repeating the 511th label does not append a 512th label and therefore
    # remains legal at the declared AR-compatible output cap.
    log_probs[511, A] = 0.
    beams, audit = ctc_prefix_beam(log_probs, beam_size=1, token_shortlist=1)
    assert len(beams) == 1 and len(beams[0]["labels"]) == 511
    assert beams[0]["labels"][-2:] == [b, A]
    assert audit["max_label_tokens"] == 511


def test_config_rejects_odd_width_before_positional_construction():
    odd = SentenceCTCConfig(width=9, heads=1, encoder_layers=1,
                            feedforward=18, max_source_tokens=32)
    with pytest.raises(ValueError, match="must be even"):
        training_batch([("a a", "가")], odd)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_prefix_beam_rejects_nan_positive_infinity_and_empty_support(bad):
    values = torch.full((1, CTC_VOCAB), -torch.inf)
    values[0, 0] = bad
    with pytest.raises(ValueError, match="invalid log-probability"):
        ctc_prefix_beam(values)
    with pytest.raises(ValueError, match="invalid log-probability"):
        ctc_prefix_beam(torch.full((1, CTC_VOCAB), -torch.inf))


def test_ctc_label_mapping_has_no_special_tokens():
    labels = ctc_labels("가 a")
    assert labels and all(1 <= label < CTC_VOCAB for label in labels)
    assert 0 not in labels


# These two tests instantiate and execute a neural encoder.  During the
# concurrent runtime evaluation they remain written but intentionally unrun.
def test_model_forward_and_loss_tiny_cpu():
    source, targets, input_lengths, target_lengths = training_batch(
        [("a a", "가"), ("ab cd", "나")], TINY)
    model = SentenceCTC(TINY)
    logits, actual_lengths = model(source)
    assert logits.shape == (15, 2, CTC_VOCAB)
    assert actual_lengths.tolist() == [9, 15]
    loss = model.loss(source, targets, input_lengths, target_lengths)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_model_serialization_round_trip_tiny_cpu(tmp_path):
    torch.manual_seed(17)
    model = SentenceCTC(TINY).eval()
    source = training_batch([("a a", "가")], TINY)[0]
    before = model(source)[0]
    checkpoint = tmp_path / "ctc.pt"
    model.save(checkpoint, {"input_contract": "roman_sentence_only"})
    restored = SentenceCTC.load(checkpoint)
    after = restored(source)[0]
    assert torch.equal(before, after)
    metadata = json.loads(checkpoint.with_suffix(".json").read_text())
    assert metadata["provenance"] == {"input_contract": "roman_sentence_only"}
    with pytest.raises(FileExistsError):
        model.save(checkpoint, {})
