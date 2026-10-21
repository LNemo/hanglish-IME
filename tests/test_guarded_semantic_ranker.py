import math

import pytest

from hanglish.models.guarded_semantic_ranker import GuardedSemanticRanker


class FakeRanker:
    def __init__(self, values):
        self.values = list(values)
        self.calls = []

    def score_pairs(self, contexts, candidates):
        self.calls.append((list(contexts), list(candidates)))
        return self.values[:len(candidates)]


def guarded(teacher, student, *, teacher_margin=0.2, student_margin=0.5):
    return GuardedSemanticRanker(
        teacher, student, max_teacher_margin=teacher_margin,
        min_student_margin=student_margin)


def test_interleaved_context_groups_keep_output_positions_stable():
    teacher = FakeRanker([1.0, 3.0, .9, 2.8])
    student = FakeRanker([0.0, 2.0, 0.0, 1.0])
    ranker = guarded(teacher, student, teacher_margin=.3, student_margin=1.)

    scores = ranker.score_pairs(["a", "b", "a", "b"], ["갑", "병", "을", "정"])

    assert scores == [-1.0, 2.0, 1.0, 3.0]
    assert student.calls == [(["a", "a", "b", "b"], ["갑", "을", "병", "정"])]


def test_singleton_and_agreement_return_teacher_scores_exactly():
    teacher = FakeRanker([1, 5, 4])
    student = FakeRanker([3, 2])
    ranker = guarded(teacher, student, teacher_margin=1.)

    scores = ranker.score_pairs(["single", "group", "group"], ["하나", "둘", "셋"])

    assert scores == [1.0, 5.0, 4.0]
    assert student.calls == [(["group", "group"], ["둘", "셋"])]


def test_threshold_boundaries_are_inclusive():
    teacher = FakeRanker([1.0, .8])
    student = FakeRanker([0.0, .5])
    ranker = guarded(teacher, student, teacher_margin=.2, student_margin=.5)

    assert ranker.score_pairs(["c", "c"], ["기존", "신규"]) == [.5, 1.0]


def test_override_is_rejected_outside_either_margin_gate():
    assert guarded(
        FakeRanker([1.0, .7]), FakeRanker([0.0, 2.0]),
        teacher_margin=.2, student_margin=1.,
    ).score_pairs(["c", "c"], ["기존", "신규"]) == [1.0, .7]
    assert guarded(
        FakeRanker([1.0, .9]), FakeRanker([0.0, .4]),
        teacher_margin=.2, student_margin=.5,
    ).score_pairs(["c", "c"], ["기존", "신규"]) == [1.0, .9]


def test_confident_teacher_skips_student_inference():
    student = FakeRanker([])
    ranker = guarded(FakeRanker([3., 1.]), student, teacher_margin=.5)
    assert ranker.score_pairs(['c', 'c'], ['기존', '신규']) == [3., 1.]
    assert student.calls == []


@pytest.mark.parametrize(
    ("teacher_margin", "student_margin"),
    [(-1, 0), (0, -1), (math.inf, 0), (0, math.nan), (True, 0)],
)
def test_thresholds_must_be_finite_nonnegative(teacher_margin, student_margin):
    with pytest.raises(ValueError):
        guarded(FakeRanker([]), FakeRanker([]),
                teacher_margin=teacher_margin, student_margin=student_margin)


@pytest.mark.parametrize(
    ("teacher_values", "student_values", "message"),
    [([1.0], [0.0, 1.0], "teacher"),
     ([1.0, 0.9], [math.nan, 1.0], "student")],
)
def test_malformed_or_nonfinite_scores_are_rejected(
        teacher_values, student_values, message):
    ranker = guarded(FakeRanker(teacher_values), FakeRanker(student_values))
    with pytest.raises(RuntimeError, match=message):
        ranker.score_pairs(["c", "c"], ["하나", "둘"])


def test_repeated_candidate_text_is_scored_once_and_does_not_form_runner_up():
    teacher = FakeRanker([1.0, 1.0, .9])
    student = FakeRanker([0.0, 2.0])
    ranker = guarded(teacher, student, teacher_margin=.1, student_margin=2.)

    scores = ranker.score_pairs(["c"] * 3, ["같음", "같음", "다름"])

    assert scores == [-1.0, -1.0, 1.0]
    assert student.calls == [(["c", "c"], ["같음", "다름"])]


def test_student_override_preserves_candidate_order_and_maps_winner_to_teacher_max():
    teacher = FakeRanker([7.0, 6.9, 6.8])
    student = FakeRanker([-3.0, -1.0, 2.0])
    ranker = guarded(teacher, student, teacher_margin=.1, student_margin=3.)

    assert ranker.score_pairs(["c"] * 3, ["가", "나", "다"]) == [2.0, 4.0, 7.0]


def test_empty_and_invalid_inputs_do_not_call_models():
    teacher, student = FakeRanker([]), FakeRanker([])
    ranker = guarded(teacher, student)
    assert ranker.score_pairs([], []) == []
    assert teacher.calls == student.calls == []
    with pytest.raises(TypeError):
        ranker.score_pairs(("c",), ["가"])
    with pytest.raises(ValueError):
        ranker.score_pairs(["c"], ["가", "나"])


def test_checkpoint_identity_is_exposed_and_requires_both_paths(tmp_path):
    teacher_path, student_path = tmp_path / "teacher", tmp_path / "student"
    teacher_path.mkdir(); student_path.mkdir()
    (teacher_path / "metadata.json").write_text("teacher")
    (student_path / "metadata.json").write_text("student")
    ranker = GuardedSemanticRanker(
        FakeRanker([]), FakeRanker([]), max_teacher_margin=.2,
        min_student_margin=.5, teacher_checkpoint=teacher_path,
        student_checkpoint=student_path)

    assert ranker.metadata["experimental"] is True
    assert ranker.artifact_paths == {
        "teacher": str(teacher_path.resolve()),
        "student": str(student_path.resolve()),
    }
    assert set(ranker.artifact_hashes) == {"teacher", "student"}
    assert set(ranker.artifact_hashes["teacher"]) == {"metadata.json"}
    with pytest.raises(ValueError, match="provided together"):
        GuardedSemanticRanker(
            FakeRanker([]), FakeRanker([]), max_teacher_margin=.2,
            min_student_margin=.5, teacher_checkpoint=teacher_path)
