from types import SimpleNamespace

import pytest
import torch

from hanglish.models.electra_plausibility import (
    ElectraPlausibility, MAX_CONTEXT_TOKENS, _common_difference_spans,
    _discriminator_head_complete, _loading_info_valid,
)


class Tokenizer:
    unk_token_id = 99

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def _one(self, text, special):
        ids, offsets = [], []
        cursor = 0
        for word in text.split():
            start = text.index(word, cursor)
            cursor = start + len(word)
            ids.append(99 if "?" in word else (len(word) % 20 + 3))
            offsets.append((start, cursor))
        if special:
            ids = [1, *ids, 2]
            offsets = [(0, 0), *offsets, (0, 0)]
        return ids, offsets

    def __call__(self, value, *, add_special_tokens=True, padding=False,
                 truncation=False, max_length=None, return_offsets_mapping=False,
                 return_special_tokens_mask=False, return_tensors=None):
        values = value if isinstance(value, list) else [value]
        encoded = [self._one(text, add_special_tokens) for text in values]
        max_size = max(len(ids) for ids, _ in encoded)
        rows, masks, offsets, specials = [], [], [], []
        for ids, spans in encoded:
            pad = max_size - len(ids) if padding else 0
            rows.append(ids + [0] * pad)
            masks.append([1] * len(ids) + [0] * pad)
            offsets.append(spans + [(0, 0)] * pad)
            specials.append([int(span == (0, 0)) for span in spans] + [1] * pad)
        result = {"input_ids": rows, "attention_mask": masks}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        if return_special_tokens_mask:
            result["special_tokens_mask"] = specials
        if return_tensors == "pt":
            result = {name: torch.tensor(item) for name, item in result.items()}
        elif not isinstance(value, list):
            result = {name: item[0] for name, item in result.items()}
        return result


class Model:
    def __init__(self, logits=None):
        self.fixed_logits = logits

    def __call__(self, **inputs):
        shape = inputs["input_ids"].shape
        logits = (torch.zeros(shape) if self.fixed_logits is None
                  else self.fixed_logits[:shape[0], :shape[1]])
        return SimpleNamespace(logits=logits)


def scorer(logits=None):
    return ElectraPlausibility(Model(logits), Tokenizer(), {"bundle": "a" * 64})


def test_zero_logit_is_log_half_and_masks_special_context_tokens():
    result = scorer().score_candidates("old context", ["good candidate"])[0]
    assert result["mean_original_logprob"] == pytest.approx(-0.69314718)
    assert result["candidate_tokens"] == 2 and result["context_tokens"] == 2
    assert result["changed_tokens"] == 2 and result["changed_span_fallback"] is True


def test_positive_replaced_logit_reduces_original_logprob():
    # [CLS], context, candidate, [SEP]
    logits = torch.tensor([[0., 0., 3., 0.], [0., 0., -3., 0.]])
    values = scorer(logits).score_candidates("ctx", ["bad", "good"])
    assert values[0]["mean_original_logprob"] < values[1]["mean_original_logprob"]


def test_nonfinite_discriminator_logits_fail_closed():
    with pytest.raises(ValueError, match="non-finite"):
        scorer(torch.tensor([[0., 0., float("nan"), 0.]])).score_candidates(
            "ctx", ["candidate"])


def test_context_is_left_trimmed_to_recent_96_tokens_and_candidate_is_complete():
    context = " ".join(f"c{i}" for i in range(140))
    result = scorer().score_candidates(context, ["one two three"])[0]
    assert result["context_tokens"] == MAX_CONTEXT_TOKENS
    assert result["candidate_tokens"] == 3


def test_every_candidate_uses_one_context_tail_sized_for_longest_candidate():
    value = scorer()
    context = " ".join(f"c{i}" for i in range(150))
    long_candidate = " ".join("word" for _ in range(300))
    prepared = value._prepare(context, ["short", long_candidate])
    short_prefix = prepared[0]["text"][:-len("short")].rstrip()
    long_prefix = prepared[1]["text"][:-len(long_candidate)].rstrip()
    assert short_prefix == long_prefix
    assert len(short_prefix.split()) == 82


def test_context_tail_is_retokenized_when_cut_starts_inside_wordpiece():
    class WordpieceTokenizer(Tokenizer):
        def _one(self, text, special):
            ids, offsets = super()._one(text, False)
            if text.startswith("prefixsuffix"):
                ids[:1] = [7, 8]
                offsets[:1] = [(0, 6), (6, 12)]
            elif text.startswith("suffix"):
                ids[:1] = [9, 10, 11]
                offsets[:1] = [(0, 2), (2, 4), (4, 6)]
            if special:
                ids = [1, *ids, 2]
                offsets = [(0, 0), *offsets, (0, 0)]
            return ids, offsets

    value = ElectraPlausibility(Model(), WordpieceTokenizer(), {})
    context = "prefixsuffix " + " ".join("ctx" for _ in range(95))
    candidates = ["one", "two three"]
    result = value.score_candidates(context, candidates)
    assert [row["candidate_tokens"] for row in result] == [1, 2]
    assert all(row["context_tokens"] <= MAX_CONTEXT_TOKENS for row in result)
    prepared = value._prepare(context, candidates)
    assert prepared[0]["text"][:-3] == prepared[1]["text"][:-9]


def test_unknown_candidate_or_oversized_candidate_returns_row_fallback():
    assert scorer().score_candidates("ctx", ["known ?unknown"]) is None
    oversized = " ".join("word" for _ in range(383))
    assert scorer().score_candidates("ctx", [oversized]) is None


def test_changed_word_region_is_order_invariant():
    candidates = ["나는 좋은 사람", "나는 나쁜 사람", "나는 멋진 사람"]
    first = scorer().score_candidates("문맥", candidates)
    order = [2, 0, 1]
    second = scorer().score_candidates("문맥", [candidates[i] for i in order])
    assert [value["changed_tokens"] for value in first] == [1, 1, 1]
    assert [second[order.index(i)] for i in range(3)] == first


def test_insertion_deletion_expands_an_adjacent_word():
    values = ["alpha beta", "alpha very beta"]
    spans = _common_difference_spans(values)
    assert spans[0][0] < spans[0][1] and spans[1][0] < spans[1][1]
    assert [value[start:end].strip() for value, (start, end, _) in zip(values, spans)] \
        == ["beta", "very"]
    result = scorer().score_candidates("", values)
    assert all(value["changed_tokens"] >= 1 for value in result)


def test_appended_word_does_not_expand_over_unchanged_prefix_word():
    values = ["alpha", "alpha beta"]
    spans = _common_difference_spans(values)
    assert values[0][spans[0][0]:spans[0][1]] == "alpha"
    assert values[1][spans[1][0]:spans[1][1]] == "beta"


def test_candidate_limit_and_types_are_strict():
    with pytest.raises(ValueError):
        scorer().score_candidates("", [])
    with pytest.raises(ValueError):
        scorer().score_candidates("", ["x"] * 11)
    with pytest.raises(TypeError):
        scorer().score_candidates(None, ["x"])


def test_loading_info_rejects_missing_trained_head_but_allows_position_buffer():
    assert _loading_info_valid({"missing_keys": ["electra.embeddings.position_ids"]})
    assert not _loading_info_valid({"missing_keys": [
        "discriminator_predictions.dense.weight"]})


def test_discriminator_head_requires_all_four_trained_parameters():
    class Head:
        def __init__(self, names):
            self.names = names

        def named_parameters(self):
            return [(name, object()) for name in self.names]

    names = [
        "discriminator_predictions.dense.weight",
        "discriminator_predictions.dense.bias",
        "discriminator_predictions.dense_prediction.weight",
        "discriminator_predictions.dense_prediction.bias",
    ]
    assert _discriminator_head_complete(Head(names))
    assert not _discriminator_head_complete(Head(names[:-1]))


def test_source_hashes_property_is_a_copy():
    value = scorer()
    hashes = value.source_hashes
    hashes["bundle"] = "changed"
    assert value.source_hashes["bundle"] == "a" * 64


def test_bundle_ignores_download_cache_but_rejects_shadowing_sidecar(tmp_path, monkeypatch):
    import json
    import hanglish.models.electra_plausibility as module
    for name in module.PINNED_HASHES:
        (tmp_path / name).write_text("placeholder")
    (tmp_path / "SOURCE.json").write_text(json.dumps({
        "revision": module.SOURCE_REVISION,
        "files": {name: value for name, value in module.PINNED_HASHES.items()
                  if name != "SOURCE.json"},
    }))
    (tmp_path / "config.json").write_text(json.dumps({
        "architectures": ["ElectraForPreTraining"], "model_type": "electra",
    }))
    (tmp_path / ".cache").mkdir()
    (tmp_path / ".cache" / "download.lock").touch()
    monkeypatch.setattr(module, "_sha", lambda path: module.PINNED_HASHES[path.name])
    assert len(module._validate_bundle(tmp_path)) == 5
    (tmp_path / "tokenizer.json").write_text("unverified")
    with pytest.raises(ValueError, match="unpinned"):
        module._validate_bundle(tmp_path)
