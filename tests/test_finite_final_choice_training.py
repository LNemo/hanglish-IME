from copy import deepcopy

import pytest

import scripts.finite_final_choice_training as module


def raw(source):
    return {"text": source, "score": None, "segments": [{
        "input_start": 0, "input_end": len(source), "source": source,
        "text": source, "kind": "raw",
    }]}


def row(accepted=(1,), old_eligible=False):
    source = "roman"
    candidates = [
        {"text": "로만", "score": 1.5, "segments": []},
        {"text": "로먼", "score": .7, "segments": []},
        raw(source),
    ]
    expected = ([candidates[index]["text"] for index in accepted]
                if accepted else ["후보에 없음"])
    return {
        "id": "id", "group_id": "group", "input": source, "context": "문맥",
        "candidates": candidates, "finite_indices": [0, 1],
        "accepted_positive_indices": list(accepted), "expected": expected,
        "selector_eligible": old_eligible,
        "all_candidates_can_score": old_eligible,
    }


class Ranker:
    def __init__(self, allowed=None):
        self.allowed = allowed
        self.seen = []

    def can_score(self, text):
        self.seen.append(text)
        return self.allowed is None or text in self.allowed


def test_none_original_is_never_budget_scored_and_old_false_flag_is_ignored():
    value = row(old_eligible=False)
    ranker = Ranker()
    example = module.build_example(value, "natural_fit", ranker)
    assert ranker.seen == ["로만", "로먼"]
    assert example is not None
    assert example.finite_source_indices == (0, 1)
    assert example.candidates == ("로만", "로먼")
    assert example.loss_weight == 1.


def test_training_view_uses_policy_indices_and_exact_compact_positive_remap():
    value = row(accepted=(0, 1))
    ranker = Ranker()
    example = module.build_example(value, "natural_fit", ranker)
    assert example.finite_source_indices == (0, 1)
    assert example.positives == (0, 1)
    assert example.teacher_scores == (1.5, .7)
    assert example.parent_correct is True

    source = "roman"
    value["candidates"] = [
        {"text": "로만", "score": 1.5, "segments": []},
        raw(source),
        {"text": "로먼", "score": .7, "segments": []},
    ]
    value["finite_indices"] = [0, 2]
    value["accepted_positive_indices"] = [2]
    value["expected"] = ["로먼"]
    example = module.build_example(value, "natural_fit", Ranker())
    assert example.finite_source_indices == (0, 2)
    assert example.positives == (1,)
    assert example.teacher_scores == (1.5, .7)


def test_budget_failure_and_no_positive_never_create_fake_examples():
    assert module.build_example(
        row(), "natural_fit", Ranker(allowed={"로만"}),
    ) is None
    value = row(accepted=())
    assert module.build_example(value, "natural_fit", Ranker()) is None


def test_malformed_stored_finite_or_positive_alignment_fails_closed():
    value = row()
    value["finite_indices"] = [0, 1, 2]
    with pytest.raises(ValueError, match="finite candidate indices"):
        module.build_example(value, "natural_fit", Ranker())
    value = row()
    value["accepted_positive_indices"] = [0]
    with pytest.raises(ValueError, match="positive alignment"):
        module.build_example(value, "natural_fit", Ranker())


def test_build_example_does_not_mutate_captured_row():
    value = row(accepted=(0, 1))
    before = deepcopy(value)
    module.build_example(value, "challenge_fit", Ranker())
    assert value == before


class LabelGuard(dict):
    budget_complete = False

    def get(self, key, default=None):
        if key in ("accepted_positive_indices", "expected") and not self.budget_complete:
            raise AssertionError("label read before source-only budget")
        return super().get(key, default)


def test_labels_are_read_only_after_complete_finite_budget():
    value = LabelGuard(row())

    class GuardRanker(Ranker):
        def can_score(self, text):
            result = super().can_score(text)
            if len(self.seen) == 2:
                value.budget_complete = True
            return result

    assert module.build_example(value, "natural_fit", GuardRanker()) is not None


def example(identity, correct):
    return module.SelectorExample(
        identity, "group", "natural_fit", "", ("a", "b"), (0,),
        (1., 0.), (0, 1), correct, 1.,
    )


def test_balanced_schedule_is_deterministic_preserves_once_and_cycles_repairs():
    values = [*(example(f"p{i}", True) for i in range(10)),
              *(example(f"r{i}", False) for i in range(2))]
    left = module.balanced_schedule(values)
    right = module.balanced_schedule(values)
    assert [[item.case_id for item in batch] for batch in left] == [
        [item.case_id for item in batch] for batch in right
    ]
    preserve = [item.case_id for batch in left for item in batch[1:]]
    repairs = [batch[0].case_id for batch in left]
    assert sorted(preserve) == [f"p{i}" for i in range(10)]
    assert len(repairs) == 4
    assert set(repairs) == {"r0", "r1"}
    assert all(batch[0].parent_correct is False for batch in left)
    assert all(item.loss_weight == 1. for batch in left for item in batch)


def test_build_all_fit_reports_reasons_without_enforcing_production_counts():
    natural = [row(accepted=(0,)), row(accepted=())]
    natural[1]["id"] = "missing"
    examples, summary = module.build_all_fit(
        natural, [], Ranker(), enforce_expected=False,
    )
    assert len(examples) == 1
    assert summary["parent_correct"] == 1
    assert summary["parent_error"] == 0
    assert summary["excluded"] == {"no_eligible_positive": 1}
