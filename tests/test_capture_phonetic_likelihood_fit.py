import pytest

from scripts.capture_phonetic_likelihood_fit import groups_for, probe_indices, validate_scores, work_units


def request(i, source="ga", target="가"):
    return {"pair_id": str(i), "source": source, "target": target}


def test_balanced_groups_cover_all_requests_and_work_including_tails():
    rows = [request(i, "a" * (i % 40 + 1), "가" * (i % 20 + 1)) for i in range(129)]
    groups = groups_for(rows)
    assert list(map(len, groups)) == [43, 43, 43]
    assert sorted(r["pair_id"] for g in groups for r in g) == sorted(r["pair_id"] for r in rows)
    assert [r for g in groups for r in g] == sorted(rows, key=lambda r: (len(r["target"]), len(r["source"]), r["pair_id"]))
    assert work_units([request(1)]) == 8


def test_probe_contains_length_extremes_and_highest_work_without_labels():
    groups = [[request(i, "a" * (i % 10 + 1), "가" * (i % 7 + 1))] for i in range(40)]
    chosen = probe_indices(groups)
    assert len(chosen) == len(set(chosen))
    assert max(range(40), key=lambda i: (work_units(groups[i]), i)) in chosen
    assert max(range(40), key=lambda i: (len(groups[i][0]["source"]), i)) in chosen
    assert max(range(40), key=lambda i: (len(groups[i][0]["target"]), i)) in chosen


def test_score_rows_bind_identity_and_eos_normalization():
    score = {"source": "ga", "target": "가", "log_probability": -4.,
             "token_count": 4, "mean_log_probability": -1.}
    assert validate_scores([request(3)], [score]) == [{"pair_id": "3", **score}]
    for key, value in [("source", "na"), ("target", "나"), ("token_count", 3),
                       ("mean_log_probability", -2.), ("log_probability", float("nan")),
                       ("log_probability", 1.)]:
        with pytest.raises(ValueError):
            validate_scores([request(3)], [{**score, key: value}])
    with pytest.raises(ValueError):
        validate_scores([request(3)], [])
