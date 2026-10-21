import pytest

from hanglish.models.guarded_semantic_ranker import GuardedSemanticRanker
from scripts import evaluate_guarded_context as evaluation


class MappingScorer:
    marker = "delegated"

    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def score_pairs(self, contexts, candidates):
        pairs = list(zip(contexts, candidates))
        self.calls.append(pairs)
        return [self.scores[pair] for pair in pairs]


def selection_row(*, corrections=1, regressions=0, top5_regressions=0,
                  errors=0, changed_results=1, thresholds=(1.0, 1.0)):
    return {
        "thresholds": list(thresholds),
        "changed_results": changed_results,
        "comparison": {
            "baseline": {"counts": {"errors": 0}},
            "corrections": [object()] * corrections,
            "regressions": [object()] * regressions,
            "top5_regressions": [object()] * top5_regressions,
            "proposed": {"counts": {"errors": errors}},
        },
    }


def test_metrics_returns_summary_and_outcomes_separately(monkeypatch):
    before, after = object(), object()
    cases = [
        {"id": "natural:1", "suite": "natural", "category": "stale"},
        {"id": "reasoning:1", "suite": "reasoning"},
    ]
    captured = {}

    def compare_captures(actual_before, actual_after, actual_cases):
        captured.update(before=actual_before, after=actual_after, cases=actual_cases)
        return {
            "baseline": {"counts": {"top1": 1}},
            "proposed": {"counts": {"top1": 2}},
            "outcomes": [{"id": "natural:1", "status": "corrected"}],
        }

    monkeypatch.setattr(evaluation.rr, "compare_captures", compare_captures)

    report, outcomes = evaluation.metrics(before, after, cases)

    assert captured["before"] is before
    assert captured["after"] is after
    assert [row["category"] for row in captured["cases"]] == [
        "natural", "reasoning"]
    assert report == {
        "baseline": {"counts": {"top1": 1}},
        "proposed": {"counts": {"top1": 2}},
    }
    assert outcomes == [{"id": "natural:1", "status": "corrected"}]


@pytest.mark.parametrize(
    "overrides",
    [
        {"corrections": 0},
        {"regressions": 1},
        {"top5_regressions": 1},
        {"errors": 1},
    ],
)
def test_select_variant_rejects_every_strict_failure(overrides):
    assert evaluation.select_variant({"unsafe": selection_row(**overrides)}) is None


def test_select_variant_ignores_unsafe_rows_when_a_safe_row_exists():
    reports = {
        "row-regression": selection_row(corrections=9, regressions=1),
        "top5-loss": selection_row(corrections=9, top5_regressions=1),
        "runtime-error": selection_row(corrections=9, errors=1),
        "safe": selection_row(corrections=1),
    }

    assert evaluation.select_variant(reports) == "safe"


def test_select_variant_uses_declared_tie_break_order():
    reports = {
        "more-changes": selection_row(changed_results=3, thresholds=(0.5, 4.0)),
        "looser-teacher": selection_row(changed_results=2, thresholds=(1.0, 4.0)),
        "looser-student": selection_row(changed_results=2, thresholds=(0.5, 1.0)),
        "declared-winner": selection_row(
            changed_results=2, thresholds=(0.5, 4.0)),
    }

    assert evaluation.select_variant(reports) == "declared-winner"


def test_select_variant_prioritizes_more_corrections_before_tie_breakers():
    reports = {
        "one": selection_row(
            corrections=1, changed_results=1, thresholds=(0.5, 4.0)),
        "two": selection_row(
            corrections=2, changed_results=99, thresholds=(2.0, 1.0)),
    }

    assert evaluation.select_variant(reports) == "two"


def test_cached_scores_deduplicates_hits_and_stays_bounded():
    scorer = MappingScorer({
        ("a", "가"): 1.0,
        ("b", "나"): 2.0,
        ("c", "다"): 3.0,
    })
    cached = evaluation.CachedScores(scorer, limit=2)

    assert cached.score_pairs(
        ["a", "a", "b"], ["가", "가", "나"]) == [1.0, 1.0, 2.0]
    assert scorer.calls == [[("a", "가"), ("b", "나")]]

    assert cached.score_pairs(["b"], ["나"]) == [2.0]
    assert len(scorer.calls) == 1
    assert cached.marker == "delegated"

    assert cached.score_pairs(["c"], ["다"]) == [3.0]
    assert list(cached.cache) == [("b", "나"), ("c", "다")]

    assert cached.score_pairs(["a", "b"], ["가", "나"]) == [1.0, 2.0]
    assert scorer.calls[-1] == [("a", "가")]
    assert len(cached.cache) == 2
    assert list(cached.cache) == [("c", "다"), ("a", "가")]


def test_guard_override_preserves_student_deltas_under_teacher_ceiling():
    teacher = MappingScorer({
        ("ctx", "가"): 5.0,
        ("ctx", "나"): 4.9,
        ("ctx", "다"): 4.8,
    })
    student = MappingScorer({
        ("ctx", "가"): 10.0,
        ("ctx", "나"): 14.0,
        ("ctx", "다"): 12.0,
    })
    guard = GuardedSemanticRanker(
        teacher, student, max_teacher_margin=0.1, min_student_margin=2.0)

    scores = guard.score_pairs(["ctx"] * 3, ["가", "나", "다"])

    assert scores == pytest.approx([1.0, 5.0, 3.0])
    assert max(scores) == pytest.approx(5.0)
    assert scores[1] - scores[2] == pytest.approx(2.0)
    assert scores[2] - scores[0] == pytest.approx(2.0)


def configuration():
    variants = {f't{a:g}-s{b:g}': selection_row(thresholds=(a, b))
                for a, b in evaluation.THRESHOLDS}
    return dict(mode='development', completed=True, test_tuning=False,
                variants=variants, selected=evaluation.select_variant(variants))


@pytest.mark.parametrize('baseline_errors', [None, 1])
def test_selection_requires_successful_baseline(baseline_errors):
    row = selection_row()
    row['comparison']['baseline']['counts']['errors'] = baseline_errors
    assert evaluation.select_variant({'bad': row}) is None


@pytest.mark.parametrize('change', ['subset', 'threshold', 'selected', 'test_tuning'])
def test_configuration_rejects_changed_plan_or_selection(change):
    config = configuration()
    name = config['selected']
    if change == 'subset':
        config['variants'].pop(next(iter(config['variants'])))
    elif change == 'threshold':
        config['variants'][name]['thresholds'] = [99, 99]
    elif change == 'selected':
        config['selected'] = 't2-s1'
    else:
        config['test_tuning'] = True
    with pytest.raises(ValueError):
        evaluation.validate_configuration(config)


@pytest.mark.parametrize('change', [None, 'hash', 'regression', 'error', 'cached'])
def test_final_evaluation_requires_successful_bound_fresh_gate(change):
    config = configuration()
    name = config['selected']
    fresh = dict(mode='development-fresh', completed=True, test_tuning=False,
                 memoization_used=False, configuration_sha256='frozen',
                 selected=name, variants={name: selection_row(
                     thresholds=config['variants'][name]['thresholds'])})
    if change == 'hash':
        fresh['configuration_sha256'] = 'different'
    elif change == 'regression':
        fresh['variants'][name]['comparison']['regressions'] = ['case']
    elif change == 'error':
        fresh['variants'][name]['comparison']['proposed']['counts']['errors'] = 1
    elif change == 'cached':
        fresh['memoization_used'] = True
    if change is None:
        evaluation.validate_configuration(config, fresh, 'frozen')
    else:
        with pytest.raises(ValueError, match='fresh development'):
            evaluation.validate_configuration(config, fresh, 'frozen')


def test_cache_rejects_wrong_input_and_output_lengths():
    scorer = MappingScorer({})
    cached = evaluation.CachedScores(scorer)
    with pytest.raises(ValueError, match='equal length'):
        cached.score_pairs(['ctx'], [])
    scorer.score_pairs = lambda contexts, candidates: []
    with pytest.raises(ValueError, match='number of scores'):
        cached.score_pairs(['ctx'], ['word'])


def test_generation_cache_is_bounded_and_isolates_mutable_results():
    class Model:
        calls = 0
        def generate(self, sources):
            self.calls += 1
            return [{'text': text, 'nested': {'status': 'original'}} for text in sources]
    model = Model()
    cached = evaluation.CachedGeneration(model, limit=1)
    result = cached.generate(['one'])
    result[0]['nested']['status'] = 'modified'
    assert cached.generate(['one'])[0]['nested']['status'] == 'original'
    assert model.calls == 1
    cached.generate(['two'])
    assert len(cached.cache) == 1
    cached.generate(['one'])
    assert model.calls == 3
