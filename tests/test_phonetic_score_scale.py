import hashlib
import numpy as np
import pytest

from scripts import phonetic_score_scale as core
from scripts.train_phonetic_score_scale import check_gate, projection


def fixture():
    rng = np.random.default_rng(17)
    matrix = rng.normal(size=(3, 4, 5))
    valid = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]], dtype=bool)
    positive = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 1]], dtype=bool)
    return matrix, valid, positive


def test_analytical_gradient_matches_finite_differences():
    matrix, valid, positive = fixture()
    parameters = np.array([.8, .2, -.1, .3, -.2])
    _, analytic = core.objective(parameters, matrix, valid, positive, True)
    numerical = []
    for index in range(5):
        shift = np.zeros(5)
        shift[index] = 1e-6
        hi = core.objective(parameters + shift, matrix, valid, positive, True)[0]
        lo = core.objective(parameters - shift, matrix, valid, positive, True)[0]
        numerical.append((hi - lo) / 2e-6)
    np.testing.assert_allclose(analytic, numerical, atol=2e-9, rtol=2e-7)


def test_masked_padding_cannot_change_loss_or_gradient():
    matrix, valid, positive = fixture()
    baseline = core.objective(core.ANCHOR, matrix, valid, positive, True)
    matrix[~valid] = 1e8
    actual = core.objective(core.ANCHOR, matrix, valid, positive, True)
    assert actual[0] == baseline[0]
    np.testing.assert_array_equal(actual[1], baseline[1])


def test_multiple_positive_probability_is_summed():
    matrix = np.zeros((1, 3, 5))
    valid = np.ones((1, 3), bool)
    positive = np.array([[True, True, False]])
    loss, gradient = core.objective(core.ANCHOR, matrix, valid, positive, True)
    assert loss == pytest.approx(np.log(3 / 2))
    np.testing.assert_array_equal(gradient, np.zeros(5))


def test_identity_parameters_preserve_base_even_after_scaling():
    base = np.array([[4., -2., 0.]])
    features = np.arange(12, dtype=float).reshape(1, 3, 4)
    valid = np.array([[True, True, False]])
    positives = np.array([[False, True, False]])
    matrix = core.design(base, features, np.array([.1, 10., 1., .03]))
    _, scores, _ = core.losses_and_scores(core.ANCHOR, matrix, valid, positives)
    np.testing.assert_array_equal(scores, np.where(valid, base, -np.inf))


def test_scale_uses_only_training_rows_and_equal_row_weight():
    features = np.zeros((3, 4, 4))
    features[0, :2, 0] = [0., 2.]
    features[1, :4, 0] = [0., 4., 0., 4.]
    features[2, :, :] = 1e8
    valid = np.array([[1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]], bool)
    scale = core.feature_scale(features, valid, [0, 1], True)
    assert scale[0] == pytest.approx(np.sqrt(2.5))
    np.testing.assert_array_equal(scale[1:], np.ones(3))
    features[2] *= -3000
    np.testing.assert_array_equal(core.feature_scale(features, valid, [0, 1], True), scale)
    np.testing.assert_array_equal(core.feature_scale(features, valid, [0, 1], False), np.ones(4))


def test_fixed_alpha_remains_identity_across_updates():
    matrix, valid, positive = fixture()
    parameters, first, second = core.ANCHOR.copy(), np.zeros(5), np.zeros(5)
    for step in range(1, 8):
        parameters, first, second, record = core.adam_step(parameters, first, second, step,
                                                         matrix, valid, positive, False)
        assert parameters[0] == 1 and record["gradient"][0] == 0


def test_learned_alpha_is_projected_to_nonnegative_domain():
    matrix = np.zeros((1, 2, 5))
    matrix[0, 0, 0] = 10
    parameters = np.array([.001, 0., 0., 0., 0.])
    result = core.adam_step(parameters, np.zeros(5), np.zeros(5), 1, matrix,
                            np.ones((1, 2), bool), np.array([[False, True]]), True)
    assert result[0][0] == 0
    assert np.isfinite(result[0]).all()


def test_component_fold_is_predeclared_hash():
    for component in ("same-document", "component-1", "component-2"):
        digest = hashlib.sha256(("phonetic-score-scale-v1:" + component).encode()).hexdigest()
        assert core.fold_for(component) == int(digest, 16) % 4
        assert core.fold_for(component) == core.fold_for(component)


def test_invalid_positive_mask_is_rejected():
    matrix, valid, positive = fixture()
    positive[0, 2] = True
    with pytest.raises(ValueError, match="loss masks"):
        core.objective(core.ANCHOR, matrix, valid, positive, True)


def test_gate_requires_both_first_choice_and_loss_improvements():
    summaries = {"fixed_unscaled": {"top1": 10, "cross_entropy": .5},
                 "learned_scaled": {"top1": 11, "cross_entropy": .4}}
    assert check_gate(summaries)
    summaries["learned_scaled"]["top1"] = 10
    assert not check_gate(summaries)
    summaries["learned_scaled"].update(top1=11, cross_entropy=.5)
    assert not check_gate(summaries)


def test_resource_projection_includes_full_fit_and_slowest_probe():
    assert projection(10., [.01, .02, .03]) == pytest.approx(1.3 * (10 + 3400 * .03) + 60)
    with pytest.raises(ValueError, match="probe"):
        projection(10., [.01, .02])
