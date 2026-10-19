import copy

import pytest

import scripts.fit_electra_score_calibration as module


def row(index, *, expected=None, candidates=None, plausibility=True, component=None,
        group_id=None):
    candidates = candidates or [f"current-{index}", f"better-{index}"]
    value = {
        "id": f"row-{index}", "component": component or f"component-{index}",
        "fold": index % 4, "source_role": "natural_fit" if index % 2 else "challenge_fit",
        "current_result": candidates[0], "candidate_texts": candidates,
        "baseline_scores": [2.0, 1.0][:len(candidates)],
        "plausibility": ([
            {"mean_original_logprob": -3.0, "changed_original_logprob": -2.0},
            {"mean_original_logprob": -0.1, "changed_original_logprob": -0.2},
        ][:len(candidates)] if plausibility else None),
        "expected": expected if expected is not None else [candidates[-1]],
    }
    if group_id is not None:
        value["group_id"] = group_id
    return value


def four_rows():
    return [row(index) for index in range(4)]


def test_cross_fitting_never_uses_held_fold_labels_for_its_weights():
    rows = four_rows()
    original = module.evaluate_calibration(rows)
    changed = copy.deepcopy(rows)
    changed[0]["expected"] = [changed[0]["candidate_texts"][0]]
    modified = module.evaluate_calibration(changed)
    assert original["folds"][0]["weights"] == modified["folds"][0]["weights"]
    assert "row-0" not in original["folds"][0]["train_row_ids"]
    assert original["oof_outcomes"][0]["calibrated_correct"] \
        != modified["oof_outcomes"][0]["calibrated_correct"]


def test_multi_gold_loss_uses_all_accepted_candidates():
    rows = four_rows()
    rows[0]["candidate_texts"] = ["a", "b", "c"]
    rows[0]["baseline_scores"] = [0.0, 0.0, 0.0]
    rows[0]["plausibility"] = [
        {"mean_original_logprob": 0.0, "changed_original_logprob": 0.0},
        {"mean_original_logprob": 1.0, "changed_original_logprob": 0.0},
        {"mean_original_logprob": 2.0, "changed_original_logprob": 0.0},
    ]
    rows[0]["current_result"] = "a"
    rows[0]["expected"] = ["a", "c"]
    weights = module.torch.tensor([1.0, 1.0, 0.0], dtype=module.torch.float64)
    loss_multi = module._objective(weights, [rows[0]])
    rows[0]["expected"] = ["a"]
    loss_single = module._objective(weights, [rows[0]])
    assert loss_multi < loss_single


def test_vectorized_objective_matches_scalar_loss_and_gradients():
    rows = four_rows()[:3]
    rows[0]["expected"] = rows[0]["candidate_texts"]
    vector_weights = module.torch.tensor([0.7, 0.2, 0.4], dtype=module.torch.float64,
                                         requires_grad=True)
    vector_loss = module._packed_objective(vector_weights, module._pack(rows))
    vector_loss.backward()

    scalar_weights = vector_weights.detach().clone().requires_grad_(True)
    losses = []
    for item in rows:
        scores = module._features(item).mv(scalar_weights)
        positives = module.torch.tensor(module._positives(item), dtype=module.torch.long)
        losses.append(module.torch.logsumexp(scores, 0)
                      - module.torch.logsumexp(scores.index_select(0, positives), 0))
    anchor = scalar_weights.new_tensor(module.RECIPE["initial_weights"])
    scalar_loss = module.torch.stack(losses).mean() + module.RECIPE["l2"] * \
        module.torch.square(scalar_weights - anchor).sum()
    scalar_loss.backward()

    assert module.torch.allclose(vector_loss, scalar_loss, atol=0, rtol=0)
    assert module.torch.allclose(vector_weights.grad, scalar_weights.grad,
                                 atol=1e-15, rtol=1e-15)


def test_plausibility_diagnostics_are_allowed_but_do_not_enter_features():
    item = row(0)
    before = module._features(item)
    for value in item["plausibility"]:
        value.update(candidate_tokens=12, changed_tokens=3, context_tokens=9,
                     changed_span_fallback=False)
    rows = four_rows()
    rows[0] = item
    assert module.torch.equal(module._features(item), before)
    assert module.evaluate_calibration(rows)["rows"] == 4


def test_uncovered_and_missing_plausibility_stay_in_denominator_and_fallback_raw():
    rows = four_rows()
    rows[0]["expected"] = ["not-a-candidate"]
    rows[0]["plausibility"] = None
    result = module.evaluate_calibration(rows)
    outcome = next(value for value in result["oof_outcomes"] if value["id"] == "row-0")
    assert result["metrics"]["all"]["rows"] == 4
    assert result["metrics"]["all"]["none_gold"] == 1
    assert outcome["calibrated_choice"] == rows[0]["current_result"]
    assert outcome["calibrated_scores"] is None
    assert all("row-0" not in fold["train_row_ids"] for fold in result["folds"])


def test_candidate_permutation_preserves_selected_identity_with_text_tie_breaking():
    base = row(0, candidates=["z", "a"], expected=["a"])
    base["baseline_scores"] = [1.0, 1.0]
    base["plausibility"] = [
        {"mean_original_logprob": 0.0, "changed_original_logprob": 0.0},
        {"mean_original_logprob": 0.0, "changed_original_logprob": 0.0},
    ]
    weights = module.torch.tensor([1.0, 2.0, 3.0], dtype=module.torch.float64)
    choice, _ = module._choose(base, weights)
    permuted = copy.deepcopy(base)
    for key in ("candidate_texts", "baseline_scores", "plausibility"):
        permuted[key] = list(reversed(permuted[key]))
    assert choice == "a" and module._choose(permuted, weights)[0] == "a"


def test_single_candidate_returns_it_without_feature_training():
    rows = four_rows()
    rows[0] = row(0, candidates=["only"], expected=["only"], plausibility=False)
    rows[0]["baseline_scores"] = [7.0]
    result = module.evaluate_calibration(rows)
    outcome = next(value for value in result["oof_outcomes"] if value["id"] == "row-0")
    assert outcome["calibrated_choice"] == "only"


@pytest.mark.parametrize("key", ["component", "group_id"])
def test_component_or_group_may_not_cross_folds(key):
    rows = four_rows()
    if key == "group_id":
        for item in rows:
            item["group_id"] = f"group-{item['fold']}"
    rows[1][key] = rows[0][key]
    with pytest.raises(ValueError, match=f"{key} crosses folds"):
        module.evaluate_calibration(rows)
