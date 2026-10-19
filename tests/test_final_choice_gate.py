import copy
import math

import pytest

from scripts.final_choice_gate import (
    GateLabelsInsufficient,
    dumps_gate,
    fit_gate,
    loads_gate,
    predict_accept,
    predict_probability,
    transform_features,
)


def row(group, *, accept, features=None, role="natural_calibration"):
    return {
        "role": role,
        "group_id": group,
        "reason": "eligible_disagreement",
        "features": features if features is not None else [float(accept), 1, 1, 1, 1, 1],
        "parent_accepted": False,
        "proposal_accepted": accept,
    }


def sufficient_rows():
    return ([row(f"accept-{index}", accept=True) for index in range(8)]
            + [row(f"stay-{index}", accept=False) for index in range(8)])


@pytest.mark.parametrize("role", [
    "natural_fit", "natural_holdout", "challenge_fit", "challenge_preservation",
])
def test_fit_rejects_every_non_calibration_role(role):
    rows = sufficient_rows()
    rows[0] = row("leak", accept=True, role=role)
    with pytest.raises(ValueError, match="only natural_calibration"):
        fit_gate(rows)


def test_fit_requires_eight_distinct_groups_in_each_derived_class():
    rows = sufficient_rows()
    rows[7]["group_id"] = rows[6]["group_id"]
    with pytest.raises(GateLabelsInsufficient, match="labels insufficient"):
        fit_gate(rows)

    both_accepted = row("not-a-repair", accept=True)
    both_accepted["parent_accepted"] = True
    rows = sufficient_rows() + [both_accepted]
    artifact = fit_gate(rows)
    assert artifact["counts"] == {
        "rows": 17, "groups": 17, "accept_rows": 8, "stay_rows": 9,
        "accept_groups": 8, "stay_groups": 9,
    }


def test_fit_rejects_non_disagreements_and_is_deterministic():
    rows = sufficient_rows()
    rows[0]["reason"] = "parent_selected"
    with pytest.raises(ValueError, match="eligible disagreements"):
        fit_gate(rows)
    assert fit_gate(sufficient_rows()) == fit_gate(sufficient_rows())


def test_population_scaling_marks_constants_and_rejects_nonfinite_features():
    rows = []
    for index in range(8):
        rows.append(row(f"accept-{index}", accept=True,
                        features=[float(index + 1), 3, 3, 3, 3, 3]))
        rows.append(row(f"stay-{index}", accept=False,
                        features=[float(-index - 1), 3, 3, 3, 3, 3]))
    artifact = fit_gate(rows)
    assert artifact["mean"] == pytest.approx([0, 3, 3, 3, 3, 3])
    assert artifact["scale"][0] == pytest.approx(math.sqrt(25.5))
    assert artifact["scale"][1:] == [1.0] * 5
    assert artifact["constant_features"] == artifact["settings"]["feature_names"][1:]
    assert transform_features(artifact, [1000, 3, 3, 3, 3, 3]) == [8.0, 0, 0, 0, 0, 0]

    rows[0]["features"][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        fit_gate(rows)

    overflowing = copy.deepcopy(artifact)
    overflowing["mean"][0] = -1e308
    overflowing["scale"][0] = 1.0
    with pytest.raises(ValueError, match="non-finite"):
        transform_features(overflowing, [1e308, 3, 3, 3, 3, 3])


def test_json_round_trip_preserves_scores_and_half_threshold():
    artifact = fit_gate(sufficient_rows())
    restored = loads_gate(dumps_gate(artifact))
    for features in ([1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1]):
        assert predict_probability(restored, features) == predict_probability(artifact, features)
        assert predict_accept(restored, features) == predict_accept(artifact, features)

    zero = copy.deepcopy(artifact)
    zero["weights"] = [0.0] * 6
    zero["bias"] = 0.0
    assert predict_probability(loads_gate(dumps_gate(zero)), [0] * 6) == 0.5
    assert predict_accept(loads_gate(dumps_gate(zero)), [0] * 6) is True


@pytest.mark.parametrize("field,value", [
    ("weights", [float("inf"), 0, 0, 0, 0, 0]),
    ("bias", float("nan")),
    ("scale", [0, 1, 1, 1, 1, 1]),
])
def test_prediction_rejects_invalid_saved_coefficients(field, value):
    artifact = fit_gate(sufficient_rows())
    artifact[field] = value
    with pytest.raises(ValueError):
        predict_accept(artifact, [0] * 6)
