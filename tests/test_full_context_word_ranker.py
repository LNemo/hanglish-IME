import pytest
import torch

from hanglish.models.full_context_word_ranker import FullContextWordRanker


class _Tokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, sentences, contexts=None, **kwargs):
        if contexts is None:
            values = sentences if isinstance(sentences, list) else [sentences]
            rows = [[ord(character) for character in value] for value in values]
            return {"input_ids": rows if isinstance(sentences, list) else rows[0]}
        self.calls.append((list(sentences), list(contexts), dict(kwargs)))
        return {"input_ids": torch.tensor([[len(value)] for value in sentences])}

    def num_special_tokens_to_add(self, pair=False):
        return 3 if pair else 2


class _SentenceRanker:
    def __init__(self, max_length=256):
        self.tokenizer = _Tokenizer()
        self.max_length = max_length
        self.score_calls = []

    def can_score(self, candidate):
        return len(candidate) + self.tokenizer.num_special_tokens_to_add(pair=True) \
            <= self.max_length

    def score_pairs(self, contexts, candidates):
        self.score_calls.append((list(contexts), list(candidates)))
        return [float(len(candidate)) for candidate in candidates]


def test_inference_retains_distant_left_and_right_context_and_replaces_last_mask():
    sentence_ranker = _SentenceRanker(max_length=512)
    ranker = FullContextWordRanker(sentence_ranker)
    left = "LEFT-CUE-" + "가" * 90
    right = "나" * 50 + "-RIGHT-CUE"
    context = f"{left} [MASK] middle [MASK] {right}"

    scores = ranker.score_pairs([context], ["후보"])

    delegated_contexts, delegated_candidates = sentence_ranker.score_calls[0]
    assert delegated_contexts == [""]
    assert delegated_candidates == [f"{left} [MASK] middle 후보 {right}"]
    assert "LEFT-CUE" in delegated_candidates[0]
    assert "RIGHT-CUE" in delegated_candidates[0]
    assert scores == [float(len(delegated_candidates[0]))]


def test_full_render_must_fit_before_inference_or_training_encoding():
    sentence_ranker = _SentenceRanker(max_length=12)
    ranker = FullContextWordRanker(sentence_ranker)
    assert ranker.can_score("[MASK]", "짧다")
    assert not ranker.can_score("아주 긴 왼쪽 문맥 [MASK] 오른쪽", "후보")

    with pytest.raises(ValueError, match="full context"):
        ranker.score_pairs(["아주 긴 왼쪽 문맥 [MASK] 오른쪽"], ["후보"])
    assert sentence_ranker.score_calls == []

    with pytest.raises(ValueError, match="full context"):
        ranker.encode_pairs(["아주 긴 왼쪽 문맥 [MASK] 오른쪽"], ["후보"])
    assert sentence_ranker.tokenizer.calls == []


def test_inference_and_training_use_identical_full_context_rendering():
    sentence_ranker = _SentenceRanker(max_length=256)
    ranker = FullContextWordRanker(sentence_ranker)
    contexts = ["앞 [MASK] 중간 [MASK] 뒤", "마스크 없는 문맥"]
    candidates = ["첫째", "둘째"]

    ranker.score_pairs(contexts, candidates)
    ranker.encode_pairs(contexts, candidates)

    inference_contexts, inference_sentences = sentence_ranker.score_calls[0]
    training_sentences, training_contexts, kwargs = sentence_ranker.tokenizer.calls[0]
    assert inference_sentences == training_sentences
    assert inference_contexts == training_contexts == ["", ""]
    assert inference_sentences == ["앞 [MASK] 중간 첫째 뒤", "마스크 없는 문맥 둘째"]
    assert kwargs["truncation"] == "only_second"
    assert kwargs["max_length"] == sentence_ranker.max_length


@pytest.mark.parametrize(
    "method,args,error",
    [
        ("score_pairs", (("문맥",), ["후보"]), TypeError),
        ("encode_pairs", (["문맥"], ("후보",)), TypeError),
        ("score_pairs", (["문맥"], ["후보", "다른 후보"]), ValueError),
        ("encode_pairs", ([1], ["후보"]), TypeError),
        ("score_pairs", (["문맥"], [None]), TypeError),
    ],
)
def test_pair_apis_validate_lengths_and_types(method, args, error):
    ranker = FullContextWordRanker(_SentenceRanker())
    with pytest.raises(error):
        getattr(ranker, method)(*args)
