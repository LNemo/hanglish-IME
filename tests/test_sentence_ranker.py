import json

import pytest
import torch

from hanglish.core.lattice import WordOption
from hanglish.models.sentence_ranker import SentenceRanker, encode_sentence_pairs
from scripts.train_sentence_ranker import (
    _surface_spans, balanced_order, build_runtime_sentence_examples,
    build_sentence_examples, runtime_proposal_fingerprint, selection_key,
)


class _Tokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, sentences, contexts=None, **kwargs):
        if contexts is None:
            values = [ord(char) for char in sentences]
            return {"input_ids": values}
        self.calls.append((sentences, contexts, kwargs))
        return {"input_ids": torch.tensor([[len(value)] for value in sentences])}

    def num_special_tokens_to_add(self, pair=False):
        return 3 if pair else 2


class _Output:
    def __init__(self, logits):
        self.logits = logits


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids):
        return _Output(input_ids.float())


class _Generator:
    def options(self, source):
        values = {
            "gachi": (WordOption("같이", 7.0, "lexicon"), WordOption("가치", 6.5, "lexicon")),
            "eomneun": (WordOption("없는", 7.0, "lexicon"),),
        }
        return values.get(source, (WordOption(source, 0.0, "raw"),))


def _row(expected):
    return {
        "id": "case-1", "input": "gachi", "context": "가격과 쓸모를 말한다.",
        "expected": expected, "category": "homophone_context", "family": "gachi",
        "split": "dev", "provenance": "synthetic",
    }


def test_encoding_places_complete_sentence_once_and_context_separately():
    tokenizer = _Tokenizer()
    encode_sentence_pairs(tokenizer, ["  차에  받힌 자전거  "], [" 사고 문맥  "], 96)
    sentences, contexts, kwargs = tokenizer.calls[0]
    assert sentences == ["차에 받힌 자전거"]
    assert contexts == ["사고 문맥"]
    assert kwargs["truncation"] == "only_second"
    assert "차에 받힌 자전거" not in contexts[0]


@pytest.mark.parametrize('context', ['', '문맥', '문맥 ' * 20])
def test_exact_candidate_budget_drops_context_without_truncating_sentence(context):
    tokenizers = pytest.importorskip('tokenizers')
    transformers = pytest.importorskip('transformers')
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(
        {'[UNK]': 0, '[CLS]': 1, '[SEP]': 2, '[PAD]': 3, '가': 4, '문맥': 5}, unk_token='[UNK]'))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    backend.post_processor = tokenizers.processors.TemplateProcessing(
        single='[CLS] $A [SEP]', pair='[CLS] $A [SEP] $B:1 [SEP]:1',
        special_tokens=[('[CLS]', 1), ('[SEP]', 2)])
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=backend,
        unk_token='[UNK]', cls_token='[CLS]', sep_token='[SEP]', pad_token='[PAD]')
    encoded = encode_sentence_pairs(tokenizer, [' '.join(['가']*93), '가'], [context, '문맥'], 96)
    assert encoded['input_ids'].shape == (2, 96)
    assert (encoded['input_ids'][0] == 4).sum().item() == 93
    assert (encoded['input_ids'][0] == 5).sum().item() == 0
    assert (encoded['input_ids'][1] == 5).sum().item() == 1
    assert encoded['attention_mask'][0].sum().item() == 96


def test_score_pairs_uses_compatible_deterministic_api():
    tokenizer = _Tokenizer()
    ranker = SentenceRanker(_Model(), tokenizer, batch_size=2)
    assert ranker.score_pairs(["문맥", ""], ["짧다", "더 긴 문장"]) == [2.0, 6.0]
    assert ranker.score_pairs([], []) == []
    with pytest.raises(ValueError):
        ranker.score_pairs(["문맥"], ["하나", "둘"])
    with pytest.raises(TypeError):
        ranker.score_pairs(("문맥",), ["하나"])


def test_surface_spans_preserve_spaces_around_merged_alignment():
    assert _surface_spans("이번에 간다", ["이번에", "간다"]) == [(0, 3), (4, 6)]
    assert _surface_spans("이 번에 간다", ["이번에", "간다"]) == [(0, 4), (5, 7)]


def test_dev_builder_does_not_inject_missing_gold(tmp_path):
    path = tmp_path / "dev.jsonl"
    path.write_text(json.dumps(_row(["값이"]), ensure_ascii=False) + "\n", encoding="utf-8")
    examples, coverage = build_sentence_examples(
        path, _Generator(), candidate_limit=4, inject_gold=False)
    assert examples[0].positives == ()
    assert "값이" not in examples[0].candidates
    assert coverage["gold_aligned_bounded_candidate_coverage"] == 0.0
    assert coverage["train_gold_injection"] is False


def test_train_builder_keeps_injected_positive_and_hard_negative(tmp_path):
    path = tmp_path / "train.jsonl"
    row = _row(["가치"]); row["split"] = "train"
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    examples, coverage = build_sentence_examples(
        path, _Generator(), candidate_limit=2, inject_gold=True)
    assert examples[0].candidates == ("가치", "같이")
    assert examples[0].positives == (0,)
    assert coverage["oracle_segment_coverage"] == 1.0


def test_score_pairs_rejects_candidate_that_would_be_truncated():
    tokenizer = _Tokenizer()
    ranker = SentenceRanker(_Model(), tokenizer, max_length=5)
    assert ranker.can_score("두글")
    assert not ranker.can_score("여섯글자")
    with pytest.raises(ValueError, match="full-sentence token budget"):
        ranker.score_pairs(["문맥"], ["여섯글자"])


def test_balanced_order_is_deterministic_and_repeats_rare_category():
    from scripts.train_sentence_ranker import SentenceExample
    examples = [
        SentenceExample("", ("a", "b"), (0,), "1", "common"),
        SentenceExample("", ("a", "b"), (0,), "2", "common"),
        SentenceExample("", ("a", "b"), (0,), "3", "rare"),
    ]
    first = balanced_order(examples, 3, 9)
    assert first == balanced_order(examples, 3, 9)
    assert len(first) == 6
    assert sum(examples[index].category == "rare" for index in first) == 3


def test_sentence_selection_key_prefers_exact_zero_nll():
    exact = {"sentence_accuracy": 1.0, "covered_mean_nll": 0.0}
    positive = {"sentence_accuracy": 1.0, "covered_mean_nll": 0.01}
    assert selection_key(exact) > selection_key(positive)
    unavailable = {"sentence_accuracy": 1.0, "covered_mean_nll": None}
    assert selection_key(exact) > selection_key(unavailable)
    assert selection_key(unavailable) == selection_key(dict(unavailable))


def test_runtime_dev_proposals_are_bounded_and_independent_of_expected(tmp_path):
    first = _row(["가치"])
    second = _row(["같이"]); second["id"] = "case-2"
    path = tmp_path / "dev.jsonl"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in (first, second)) + "\n")

    class Converter:
        def sentence_proposals(self, text, context):
            assert text == "gachi" and context == "가격과 쓸모를 말한다."
            return {"paths": [(f"후보-{index}", 0.0, []) for index in range(60)]}

    examples, coverage = build_runtime_sentence_examples(path, Converter())
    assert examples[0].candidates == examples[1].candidates
    assert len(examples[0].candidates) == 48
    assert examples[0].positives == () and examples[1].positives == ()
    assert coverage["candidate_limit"] == 48
    assert runtime_proposal_fingerprint(examples) == runtime_proposal_fingerprint(list(examples))


def test_load_rejects_unsupported_or_incomplete_bundle(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps({"format_version": 2}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        SentenceRanker.load(tmp_path)
    (tmp_path / "metadata.json").write_text(
        json.dumps({"format_version": 1, "artifact_hashes": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="complete artifact"):
        SentenceRanker.load(tmp_path)


@pytest.mark.parametrize("field,value", [
    ("max_length", True), ("max_length", 3), ("max_length", 513),
    ("inference_batch_size", False), ("inference_batch_size", 0),
    ("inference_batch_size", -1), ("inference_batch_size", 65),
])
def test_load_rejects_invalid_inference_bounds_before_artifact_loading(tmp_path, field, value):
    names = ["model.safetensors", "config.json", "tokenizer.json", "vocab.txt",
             "tokenizer_config.json", "special_tokens_map.json"]
    metadata = {"format_version": 1, "artifact_hashes": dict.fromkeys(names, "abc"),
                "max_length": 96, "inference_batch_size": 8}
    metadata[field] = value
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid inference bounds"):
        SentenceRanker.load(tmp_path)


@pytest.mark.parametrize("batch_size", [True, 0, -1, 65])
def test_constructor_rejects_invalid_batch_size(batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        SentenceRanker(_Model(), _Tokenizer(), batch_size=batch_size)
