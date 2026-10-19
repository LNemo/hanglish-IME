import pytest

from scripts.audit_phonetic_calibration_fit_signal import audit, summarize


def row(base, features, positive=(1,)):
    return {"id": "fit", "candidates": [chr(0xAC00 + i) for i in range(len(base))],
            "base_scores": base, "features": features, "positive_indices": list(positive)}


@pytest.mark.parametrize("weights", [[0.] * 4, [100., -30., 9., 2.], [-100., 30., -9., -2.]])
def test_identical_features_cannot_overturn_wrong_base_rank(weights):
    result = audit([row([2., 1.], [[.1, .2, .3, .4]] * 2)], weights)
    assert result["counts"]["calibrated_correct"] == 0
    assert result["counts"]["errors_with_all_positives_blocked_by_identical_features"] == 1
    assert result["base_error_gold_residual_advantage"]["max"] == 0


def test_signal_that_crosses_margin_corrects_selection():
    result = audit([row([1., 0.], [[0.] * 4, [1., 0., 0., 0.]])], [2., 0., 0., 0.])
    assert result["counts"]["corrections"] == 1
    assert result["counts"]["errors_phonetic_likelihood_prefers_best_gold"] == 1
    assert result["base_error_gold_gap"]["max"] == 1
    assert result["base_error_gold_residual_advantage"]["max"] == 2


def test_one_blocked_positive_does_not_make_all_positives_impossible():
    item = row([2., 1., 0.], [[0.] * 4, [0.] * 4, [1., 0., 0., 0.]], (1, 2))
    result = audit([item], [3., 0., 0., 0.])
    assert result["counts"]["calibrated_correct"] == 1
    assert result["counts"]["errors_best_gold_same_features_as_winner"] == 1
    assert result["counts"]["errors_with_all_positives_blocked_by_identical_features"] == 0


def test_different_features_do_not_guarantee_correct_calibration():
    result = audit([row([0., 1.], [[2., 0., 0., 0.], [0.] * 4])], [1., 0., 0., 0.])
    assert result["counts"]["regressions"] == 1
    assert result["counts"]["base_errors"] == 0


def test_lexical_tie_break_preserved():
    result = audit([row([0., 0.], [[0.] * 4] * 2)], [0.] * 4)
    assert result["counts"]["errors_with_all_positives_blocked_by_identical_features"] == 1


def test_denominator_and_nonfinite_rejected():
    item = row([1., 0.], [[0.] * 4] * 2)
    with pytest.raises(ValueError, match="duplicate"):
        audit([item, item], [0.] * 4)
    with pytest.raises(ValueError, match="empty"):
        audit([], [0.] * 4)
    item["features"][0] = [float("nan")] * 4
    with pytest.raises(ValueError, match="numeric"):
        audit([item], [0.] * 4)


def test_summary_uses_observations_without_interpolation():
    assert summarize([]) == {"count": 0}
    assert summarize([4, 1, 2, 3]) == {"count": 4, "min": 1, "max": 4, "p50": 2, "p95": 4}
