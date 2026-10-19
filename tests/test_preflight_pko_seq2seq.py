import math

import pytest
import torch
from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

from scripts import preflight_pko_seq2seq as preflight


class FakeTokenizer:
    all_special_tokens = ["</s>", "<pad>", "<extra_id_0>"]

    class Backend:
        normalizer = "NFKC"
        padding = {"strategy": "BatchLongest"}
        truncation = {"max_length": 12}

        def no_padding(self):
            self.padding = None

        def no_truncation(self):
            self.truncation = None

        def to_str(self):
            return '{"normalizer":%s,"padding":%s,"truncation":%s}' % (
                "null" if self.normalizer is None else '"NFKC"',
                "null" if self.padding is None else '"enabled"',
                "null" if self.truncation is None else '"enabled"')

    def __init__(self):
        self.backend_tokenizer = self.Backend()
        self.split_special_tokens = False
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append(kwargs)
        return {"input_ids": [ord(char) for char in text] + [1]}

    def decode(self, ids, **kwargs):
        return "".join(chr(value) for value in ids if value != 1)


def test_tokenizer_audit_covers_train_targets_and_dev_sources_without_truncation():
    tokenizer = FakeTokenizer()
    train = [{"id": "a", "source": "roman<extra_id_0> source", "target": "한글 target"}]
    dev = [{"id": "b", "source": "dev source"}]
    before = preflight.tokenizer_fingerprint(tokenizer)
    assert preflight.adapt_lossless_tokenizer(tokenizer) is tokenizer
    after = preflight.tokenizer_fingerprint(tokenizer)
    stats, records = preflight.audit_tokenizer(tokenizer, train, dev)
    assert {row["scope"] for row in records} == {
        "train_source", "train_target", "dev_source"}
    assert stats["train_source"]["max_tokens"] == len("roman<extra_id_0> source") + 1
    assert all(row["roundtrip_exact"] for row in records)
    assert all(row["tokens"] == len(row["text"]) + 1 for row in records)
    assert before != after and tokenizer.backend_tokenizer.normalizer is None
    assert tokenizer.split_special_tokens is True
    assert tokenizer.backend_tokenizer.padding is None
    assert tokenizer.backend_tokenizer.truncation is None
    assert stats["train_source"]["special_token_literals"] == 1
    assert stats["train_source"]["special_token_losses"] == 0
    assert all(call["split_special_tokens"] is True for call in tokenizer.calls)


def test_adapted_fast_tokenizer_fingerprint_survives_encode_save_and_reload(tmp_path):
    backend = Tokenizer(BPE(unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    trainer = BpeTrainer(vocab_size=256, special_tokens=["<unk>", "<pad>", "</s>"])
    backend.train_from_iterator(
        ["imnida 입니다 ㈜3.3㎡14㎝ literal<extra_id_0>tag <pad> 한글"], trainer)
    backend.normalizer = normalizers.NFKC()
    eos_id = backend.token_to_id("</s>")
    backend.post_processor = processors.TemplateProcessing(
        single="$A </s>", special_tokens=[("</s>", eos_id)])
    backend.enable_padding(pad_id=backend.token_to_id("<pad>"), pad_token="<pad>")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>", eos_token="</s>")
    preflight.adapt_lossless_tokenizer(tokenizer)
    expected = preflight.tokenizer_fingerprint(tokenizer)
    probes = ["imnida", "입니다", "㈜3.3㎡14㎝", "literal<extra_id_0>tag", "<pad>", "한글"]
    for text in probes:
        ids = preflight._token_ids(tokenizer, text)
        assert tokenizer.decode(ids, skip_special_tokens=True,
                                clean_up_tokenization_spaces=False) == text
    assert preflight.tokenizer_fingerprint(tokenizer) == expected
    tokenizer.save_pretrained(tmp_path)
    restored = PreTrainedTokenizerFast.from_pretrained(tmp_path, local_files_only=True)
    preflight.adapt_lossless_tokenizer(restored)
    assert preflight.tokenizer_fingerprint(restored) == expected
    for text in probes:
        ids = preflight._token_ids(restored, text)
        assert restored.decode(ids, skip_special_tokens=True,
                               clean_up_tokenization_spaces=False) == text


def test_resource_guard_rejects_live_rss_swap_whole_and_update_bounds():
    base = dict(elapsed_s=1., operation_elapsed_s=1., rss=10, swap=20,
                baseline_swap=20,
                limits={"rss_bytes": 100, "swap_growth_bytes": 50,
                        "whole_seconds": 10., "update_seconds": 5.})
    assert preflight.evaluate_observation(**base)["swap_growth_bytes"] == 0
    for change, message in (
        ({"rss": 101}, "RSS"), ({"swap": 71}, "swap-growth"),
        ({"elapsed_s": 11.}, "whole"), ({"operation_elapsed_s": 6.}, "update"),
    ):
        with pytest.raises(preflight.ResourceExceeded, match=message):
            preflight.evaluate_observation(**(base | change))


def test_selection_is_deterministic_unique_and_retains_required_case_types():
    rows = []
    for index in range(20):
        source = f"roman sentence {index}"
        target = f"한글 문장 {index}"
        if index == 4:
            target += " OpenAI"
        if index == 5:
            source = "https://example.com/reul yeoreoyo"
        if index == 6:
            source = "2026nyeon suyireul hwaginhaeyo"
        if index == 7:
            source = "gwalho(reul) hwaginhaeyo!"
        rows.append({"id": f"r{index:02d}", "source": source, "target": target,
                     "source_tokens": index + 5, "target_tokens": index + 7})
    first = preflight.select_probe_rows(rows)
    second = preflight.select_probe_rows(list(reversed(rows)))
    assert [(row["id"], row["selection_reason"]) for row in first] == [
        (row["id"], row["selection_reason"]) for row in second]
    assert len({row["id"] for row in first}) == 8
    assert {row["selection_reason"] for row in first} >= {
        "longest_total", "longest_source", "longest_target", "median_total",
        "mixed_ascii_target", "url_or_email", "numeric", "punctuated",
    }


def test_schedule_estimate_accounts_for_three_epochs_generation_checkpoint_and_pilot():
    result = preflight.schedule_estimate([2.] * 16, generation_seconds=8.,
                                         generation_rows=8, checkpoint_seconds=10.)
    assert result["three_epoch_fit_updates"] == 6916 * 3
    assert result["heldout_inner_validation_rows"] == 788
    assert result["estimated_three_epoch_training_s"] == 6916 * 3 * 2. * 1.25
    assert result["estimated_dev_generation_s"] == 316 * 1.25
    assert result["estimated_full_schedule_s"] == (
        result["estimated_three_epoch_training_s"] + 316 * 1.25 + 10.)
    assert 0 < result["bounded_pilot_updates"] <= 6916
    assert result["estimated_bounded_pilot_s"] <= 3600.
    assert math.isfinite(result["estimated_full_schedule_s"])


def test_logit_parity_accepts_fp32_noise_but_rejects_large_or_decision_changes():
    expected = torch.tensor([[[0.25, 1.0, -0.5]]], dtype=torch.float32)
    close = expected + torch.tensor([[[2e-6, -2e-6, 1e-6]]])
    report = preflight.validate_logit_parity(expected, close)
    assert report == {
        "max_logit_delta": pytest.approx(2e-6, abs=1e-7),
        "logits_bitwise_equal": False,
        "logits_allclose": True,
        "logit_atol": 1e-5,
        "logit_rtol": 1e-6,
        "argmax_same": True,
    }
    with pytest.raises(RuntimeError, match="output parity differs"):
        preflight.validate_logit_parity(expected, expected + 1e-3)
    near_tie = torch.tensor([[[1e-7, 0.0]]])
    changed_choice = torch.tensor([[[0.0, 1e-7]]])
    with pytest.raises(RuntimeError, match="argmax_same=False"):
        preflight.validate_logit_parity(near_tie, changed_choice)
