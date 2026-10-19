import inspect
from types import SimpleNamespace

import pytest

import scripts.context_proposal_union as module


def path(text, score=0.0):
    return (text, score, [{"text": text}])


class SentenceRanker:
    def __init__(self, long=()):
        self.long = set(long)
        self.calls = []

    def can_score(self, text):
        return text not in self.long

    def score_pairs(self, contexts, candidates):
        self.calls.append((contexts, candidates))
        return [float(len(candidate)) for candidate in candidates]


class Converter:
    def __init__(self, parent_word, static, student_word, *, sentence=None,
                 student_error=None, student_static=None):
        self.parent = object()
        self.student = object()
        self.ranker = self.parent
        self.sentence_ranker = sentence or SentenceRanker()
        self.parent_word = parent_word
        self.static = static
        self.student_word = student_word
        self.student_error = student_error
        self.student_static = static if student_static is None else student_static
        self.parent_calls = 0
        self.student_calls = 0

    def _prepare_candidates(self, text, context, *, learned, raw):
        assert learned is True and raw is False
        if self.ranker is self.parent:
            self.parent_calls += 1
            return self.parent_word, self.static, [], False, True
        assert self.ranker is self.student
        self.student_calls += 1
        if self.student_error:
            raise self.student_error
        return self.student_word, self.student_static, [], False, True


@pytest.fixture(autouse=True)
def identity_beam(monkeypatch):
    monkeypatch.setattr(module, "beam_paths", lambda pieces: list(pieces))


def test_parent_prefix_dedup_and_max_72_are_preserved():
    parent_word = [path(f"w{i}") for i in range(24)]
    static = [path(f"s{i}") for i in range(24)]
    student = [path(f"n{i}") for i in range(23)] + [path("w0")]
    converter = Converter(parent_word, static, student)

    result = module.propose(converter, converter.student, "two words", "context")

    assert result["parent_paths"] == parent_word + static
    assert result["union_paths"][:48] == result["parent_paths"]
    assert len(result["union_paths"]) == 71
    assert [item[0] for item in result["additions"]] == [f"n{i}" for i in range(23)]
    assert converter.parent_calls == converter.student_calls == 1

    converter = Converter(parent_word, static, [path(f"n{i}") for i in range(24)])
    assert len(module.propose(converter, converter.student, "two words")["union_paths"]) == 72


def test_long_student_extra_is_filtered_without_disabling_baseline_sentence_ranking():
    sentence = SentenceRanker(long={"student candidate that is too long"})
    converter = Converter(
        [path("parent a"), path("parent b")], [path("parent b")],
        [path("student candidate that is too long"), path("new")], sentence=sentence,
    )

    proposal = module.propose(converter, converter.student, "two words")
    chosen = module.choose(proposal, sentence, "context")

    assert proposal["sentence_eligible"] is True
    assert [item[0] for item in proposal["filtered_long"]] == ["student candidate that is too long"]
    assert [item[0] for item in proposal["union_paths"]] == ["parent a", "parent b", "new"]
    assert chosen["sentence_ranking_applied"] is True
    assert all(item[0] != "student candidate that is too long" for item in chosen["paths"])


@pytest.mark.parametrize("text,parent_word,static", [
    ("single", [path("a"), path("b")], [path("a")]),
    ("two words", [path("only")], [path("only")]),
    ("two words", [path("a"), path("b")], [path("too long")]),
])
def test_ineligible_parent_never_calls_student_and_uses_parent_word_fallback(
        text, parent_word, static):
    sentence = SentenceRanker(long={"too long"})
    converter = Converter(parent_word, static, [path("new")], sentence=sentence)

    proposal = module.propose(converter, converter.student, text)
    chosen = module.choose(proposal, sentence)

    assert converter.student_calls == 0
    assert proposal["union_paths"] == proposal["parent_paths"]
    assert chosen == {"paths": parent_word, "sentence_ranking_applied": False}
    assert sentence.calls == []


def test_parent_ranker_is_restored_when_student_preparation_fails():
    converter = Converter([path("a"), path("b")], [path("a")], [],
                          student_error=RuntimeError("student failed"))

    with pytest.raises(RuntimeError, match="student failed"):
        module.propose(converter, converter.student, "two words")

    assert converter.ranker is converter.parent


def test_static_lattice_mismatch_is_rejected_after_ranker_restore():
    converter = Converter([path("a"), path("b")], [path("a")], [path("new")],
                          student_static=[path("changed")])

    with pytest.raises(ValueError, match="static lattice"):
        module.propose(converter, converter.student, "two words")

    assert converter.ranker is converter.parent


def test_choose_uses_sentence_score_then_text_and_does_not_mutate_pool():
    class Scores(SentenceRanker):
        def score_pairs(self, contexts, candidates):
            return [1., 2., 2.]
    paths = [path("c"), path("b"), path("a")]
    proposal = {"sentence_eligible": True, "union_paths": list(paths)}

    result = module.choose(proposal, Scores(), "ctx")

    assert [item[0] for item in result["paths"]] == ["a", "b", "c"]
    assert [item[1] for item in result["paths"]] == [2., 2., 1.]
    assert proposal["union_paths"] == paths


def test_public_proposal_api_has_no_label_or_expected_argument():
    assert list(inspect.signature(module.propose).parameters) == [
        "converter", "student", "text", "context"
    ]
