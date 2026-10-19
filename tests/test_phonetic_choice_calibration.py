"""Small numerical objective checks; no neural models or real FIT training."""
import numpy as np
import pytest

from scripts.train_phonetic_choice_calibration import adam_step, objective


def example():
    base = np.array([[.5, -.2, -np.inf], [2., -3., .7]], dtype=np.float64)
    features = np.array([
        [[-1., .8, .1, .2], [-2., .7, .3, .1], [0., 0., 0., 0.]],
        [[-.4, .4, .5, .3], [-.8, .5, .2, .1], [-.9, .7, .1, .2]],
    ])
    positives = np.array([[False, True, False], [True, False, True]])
    return base, features, positives


def test_analytical_gradient_matches_finite_difference_with_padding_and_multiple_positives():
    base, features, positives = example()
    weights = np.array([.2, -.1, .3, -.4])
    _, actual = objective(weights, base, features, positives)
    numerical = []
    for i in range(4):
        offset = np.eye(4)[i] * 1e-6
        high = objective(weights + offset, base, features, positives)[0]
        low = objective(weights - offset, base, features, positives)[0]
        numerical.append((high - low) / 2e-6)
    np.testing.assert_allclose(actual, numerical, atol=2e-10, rtol=1e-8)


def test_rows_have_equal_weight_despite_different_candidate_counts():
    base, features, positives = example()
    weights = np.array([.2, -.1, .3, -.4])
    combined_loss, combined_grad = objective(weights, base, features, positives)
    separate = [objective(weights, base[i:i+1], features[i:i+1], positives[i:i+1])
                for i in range(2)]
    assert combined_loss == pytest.approx(sum(x[0] for x in separate) / 2)
    np.testing.assert_allclose(combined_grad, np.mean([x[1] for x in separate], axis=0))


def test_uniform_row_score_shift_does_not_change_loss_or_gradient():
    base, features, positives = example()
    weights = np.array([.2, -.1, .3, -.4])
    loss, grad = objective(weights, base, features, positives)
    shifted_loss, shifted_grad = objective(weights, base + np.array([[1000.], [-1000.]]), features, positives)
    assert shifted_loss == pytest.approx(loss, abs=2e-13)
    np.testing.assert_allclose(shifted_grad, grad, atol=2e-13)


def test_all_candidates_positive_leaves_only_declared_ridge():
    base, features, _ = example()
    weights = np.array([.2, -.1, .3, -.4])
    loss, gradient = objective(weights, base, features, np.isfinite(base))
    assert loss == pytest.approx(.01 * np.dot(weights, weights))
    np.testing.assert_allclose(gradient, .02 * weights, atol=1e-15)


def test_candidate_permutation_preserves_multi_positive_objective():
    base, features, positives = example()
    weights = np.array([.2, -.1, .3, -.4])
    loss, grad = objective(weights, base, features, positives)
    order = [2, 0, 1]
    permuted_loss, permuted_grad = objective(weights, base[:, order], features[:, order], positives[:, order])
    assert permuted_loss == pytest.approx(loss)
    np.testing.assert_allclose(permuted_grad, grad, atol=1e-15)


def test_optimizer_clips_large_gradient_to_exact_unit_norm():
    base, features, positives = example()
    features *= 100.
    zero = np.zeros(4, dtype=np.float64)
    _, gradient = objective(zero, base, features, positives)
    assert np.linalg.norm(gradient) > 1.
    _, first, second, _ = adam_step(zero, zero, zero, 1, base, features, positives)
    clipped = first / .1
    np.testing.assert_allclose(clipped, gradient / np.linalg.norm(gradient), rtol=1e-14)
    assert np.linalg.norm(clipped) == pytest.approx(1., abs=1e-14)
    np.testing.assert_allclose(second, .001 * clipped * clipped, rtol=1e-14)
