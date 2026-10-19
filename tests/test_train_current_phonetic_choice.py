from copy import deepcopy
from contextlib import contextmanager
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import phonetic_score_scale as core
from scripts import train_current_phonetic_choice as trainer


def component_for(fold):
    index = 0
    while True:
        value = f"component-{fold}-{index}"
        if core.fold_for(value) == fold:
            return value
        index += 1


def row(identifier="row-1", component=None, candidates=None, scores=None,
        expected=None, production=None, ranked=True, public=None,
        training_eligible=None, source="raw"):
    candidates = ["정답", "오답"] if candidates is None else candidates
    scores = list(range(len(candidates), 0, -1)) if scores is None else scores
    expected = ["정답"] if expected is None else expected
    positives = [i for i, value in enumerate(candidates) if value in set(expected)]
    if training_eligible is None:
        training_eligible = bool(ranked and positives)
    return {"id": identifier, "input": source,
            "component_id": component or component_for(0), "source_shard": 0,
            "ranked": ranked, "training_eligible": training_eligible,
            "candidates": candidates, "base_scores": scores,
            "features": [[float(i), .1, .2, .3]
                         for i in range(len(candidates))],
            "positive_indices": positives,
            "production_indices": (list(range(len(candidates)))
                                   if production is None else production),
            "public_candidates": (list(candidates[:5]) if public is None else public),
            "expected": expected}


def source_map(rows):
    return {value["id"]: {"id": value["id"], "input": value["input"],
                          "component_id": value["component_id"]}
            for value in rows}


def test_validate_recomputes_training_and_gradient_eligibility():
    rows = [
        row("gradient", candidates=["정답", "오답"]),
        row("all-positive", candidates=["정답", "허용"],
            expected=["정답", "허용"]),
        row("no-positive", candidates=["오답", "다른 오답"], expected=["정답"],
            training_eligible=False),
        row("no-candidate", candidates=[], scores=[], expected=["정답"],
            production=[], ranked=False, public=["원문"], training_eligible=False),
    ]
    trainer.validate_rows(rows, source_map(rows))
    assert [value["training_eligible"] for value in rows] == \
        [True, True, False, False]
    assert [value["gradient_eligible"] for value in rows] == \
        [True, False, False, False]


def test_validate_rejects_untrusted_preparer_eligibility_flag():
    value = row(training_eligible=False)
    with pytest.raises(ValueError, match="eligibility"):
        trainer.validate_rows([value], source_map([value]))


def test_validate_requires_production_subset_for_ranked_row():
    value = row(production=[])
    with pytest.raises(ValueError, match="domain"):
        trainer.validate_rows([value], source_map([value]))


def test_eligible_arrays_excludes_all_positive_and_uses_component_folds():
    rows = []
    for fold in range(4):
        rows.append(row(f"gradient-{fold}", component=component_for(fold)))
        rows.append(row(f"all-positive-{fold}", component=component_for(fold),
                        candidates=["정답"], expected=["정답"]))
    trainer.validate_rows(rows, source_map(rows))
    eligible, base, features, valid, positives, folds = trainer.eligible_arrays(rows)
    assert [value["id"] for value in eligible] == \
        [f"gradient-{fold}" for fold in range(4)]
    assert set(folds.tolist()) == {0, 1, 2, 3}
    assert base.shape[:2] == valid.shape == positives.shape
    assert features.shape[2] == 4


def test_public_top5_replaces_latent_fifth_with_raw_source():
    value = row(candidates=["일", "이", "삼", "사", "정답", "raw"],
                scores=[6., 5., 4., 3., 2., 1.], expected=["정답"], source="raw")
    outcome = trainer.score_outcome(value, core.ANCHOR, np.ones(4))
    assert outcome["selected_index"] == 0
    assert outcome["first"] is False
    assert outcome["latent_top5"] is True
    assert outcome["top5"] is False
    assert outcome["public_top5"] == ["일", "이", "삼", "사", "raw"]


def test_public_top5_keeps_five_when_raw_source_already_present():
    value = row(candidates=["일", "raw", "삼", "사", "정답", "육"],
                scores=[6., 5., 4., 3., 2., 1.], expected=["정답"], source="raw")
    outcome = trainer.score_outcome(value, core.ANCHOR, np.ones(4))
    assert outcome["top5"] is True
    assert outcome["public_top5"] == ["일", "raw", "삼", "사", "정답"]


def test_all_positive_row_has_zero_ce_but_is_not_gradient_eligible():
    value = row(candidates=["정답", "허용"], expected=["정답", "허용"])
    trainer.validate_rows([value], source_map([value]))
    outcome = trainer.score_outcome(value, core.ANCHOR, np.ones(4))
    assert value["training_eligible"] is True
    assert value["gradient_eligible"] is False
    assert outcome["covered"] is True
    assert outcome["cross_entropy"] == pytest.approx(0.0)


def test_empty_candidate_row_uses_immutable_public_fallback():
    value = row(candidates=[], scores=[], production=[], ranked=False,
                public=["정답", "raw"], expected=["정답"],
                training_eligible=False)
    trainer.validate_rows([value], source_map([value]))
    outcome = trainer.score_outcome(value, np.array([9., 8., 7., 6., 5.]),
                                    np.ones(4))
    assert outcome == {"first": True, "top5": True, "latent_top5": True,
                       "covered": False, "cross_entropy": None,
                       "selected_index": None, "public_top5": ["정답", "raw"],
                       "latent_top5_indices": [], "fallback_public": True}


def test_production_subset_is_scored_separately_from_full_pool():
    value = row(candidates=["정답", "오답"], scores=[10., 1.],
                expected=["정답"], production=[1], source="raw")
    full = trainer.score_outcome(value, core.ANCHOR, np.ones(4))
    production = trainer.score_outcome(value, core.ANCHOR, np.ones(4), [1])
    assert full["first"] is True
    assert production["first"] is False
    assert production["covered"] is False


def test_summary_reports_corrections_regressions_and_ce_denominator():
    first = row("one", component=component_for(0))
    second = row("two", component=component_for(1), candidates=[], scores=[],
                 production=[], ranked=False, public=["정답"],
                 training_eligible=False)
    trainer.validate_rows([first, second], source_map([first, second]))
    predictions = [
        trainer.evaluate_row(first, core.ANCHOR, np.ones(4),
                             core.ANCHOR, np.ones(4)),
        trainer.evaluate_row(second, core.ANCHOR, np.ones(4),
                             core.ANCHOR, np.ones(4)),
    ]
    summary = trainer.summarize(predictions)
    assert summary["base_full"]["rows"] == 2
    assert summary["base_full"]["coverage"] == 1
    assert summary["base_full"]["coverage_denominator"] == 1
    assert summary["base_full"]["fallback_public_rows"] == 1
    assert summary["base_full"]["fallback_public_first"] == 1
    assert summary["base_full"]["covered_cross_entropy_rows"] == 1
    assert summary["base_full"]["evaluated_id_sha256"] == \
        summary["proposed"]["evaluated_id_sha256"]
    assert summary["base_full"]["covered_cross_entropy_id_sha256"] == \
        summary["proposed"]["covered_cross_entropy_id_sha256"]
    assert summary["candidate_expansion_deployed_production_to_full"] == {
        "corrections": 0, "regressions": 0}
    assert summary["refit_deployed_full_to_proposed"] == {
        "corrections": 0, "regressions": 0}
    assert summary["versus_deployed_production"] == {"corrections": 0,
                                                       "regressions": 0}


def test_qualification_is_strict_and_predeclared():
    shared = {"rows": trainer.ROWS, "evaluated_id_sha256": "all",
              "covered_cross_entropy_id_sha256": "covered"}
    summary = {"base_full": {**shared, "first": 10,
                             "covered_cross_entropy": 1.0},
               "proposed": {**shared, "first": 11,
                            "covered_cross_entropy": .9}}
    assert trainer.qualified(summary) is True
    changed = deepcopy(summary); changed["proposed"]["first"] = 10
    assert trainer.qualified(changed) is False
    changed = deepcopy(summary); changed["proposed"]["covered_cross_entropy"] = 1.0
    assert trainer.qualified(changed) is False
    changed = deepcopy(summary)
    changed["proposed"]["covered_cross_entropy_id_sha256"] = "different"
    with pytest.raises(ValueError, match="domains differ"):
        trainer.qualified(changed)
    changed = deepcopy(summary); changed["proposed"]["rows"] -= 1
    with pytest.raises(ValueError, match="domains differ"):
        trainer.qualified(changed)


def test_score_rejects_negative_base_coefficient():
    value = row()
    with pytest.raises(ValueError, match="score model"):
        trainer.score_outcome(value, np.array([-1., 0., 0., 0., 0.]),
                              np.ones(4))


def test_resource_projection_uses_maximum_one_thousand_updates():
    assert trainer.projection(5.0, [.01, .02, .03]) == pytest.approx(
        1.3 * (5.0 + .03 * 1000) + 60.0)


def test_fit_restarts_from_anchor_and_is_deterministic():
    rows = [row(f"row-{fold}", component=component_for(fold))
            for fold in range(4)]
    trainer.validate_rows(rows, source_map(rows))
    _, base, features, valid, positives, _ = trainer.eligible_arrays(rows)
    scale = core.feature_scale(features, valid, np.arange(4), True)
    matrix = core.design(base, features, scale)
    first, records_a = trainer.fit(matrix, valid, positives, steps=2)
    second, records_b = trainer.fit(matrix, valid, positives, steps=2)
    assert np.array_equal(first, second)
    assert records_a == records_b
    assert records_a[0]["step"] == 1 and records_a[-1]["step"] == 2


def test_fixed_numerical_core_and_feature_preparer_hashes_are_current():
    assert trainer.sha(core.__file__) == trainer.CORE_SHA256
    assert trainer.sha(trainer.FEATURE_PREPARER) == \
        trainer.FEATURE_PREPARER_SHA256


@pytest.mark.parametrize("gate", [True, False])
def test_synthetic_end_to_end_runner_covers_full_fit_and_no_model_branch(
        tmp_path, monkeypatch, gate):
    rows = [row(f"gradient-{fold}", component=component_for(fold),
                candidates=["오답", "정답"], scores=[1.0, .9])
            for fold in range(4)]
    rows.extend([
        row("all-positive", component=component_for(0), candidates=["정답"],
            scores=[1.0], expected=["정답"]),
        row("no-positive", component=component_for(1),
            candidates=["오답", "다른 오답"], scores=[1.0, .5],
            expected=["정답"], training_eligible=False),
        row("no-candidate", component=component_for(2), candidates=[], scores=[],
            expected=["정답"], production=[], ranked=False, public=["raw"],
            training_eligible=False),
        row("all-positive-two", component=component_for(3),
            candidates=["정답", "허용"], scores=[1.0, .5],
            expected=["정답", "허용"]),
    ])
    trainer.validate_rows(rows, source_map(rows))
    features = tmp_path / "features"
    features.mkdir()
    (features / "rows.jsonl").write_text("synthetic rows\n")
    expected_report_sha = "a" * 64
    deployed = {"parameters": core.ANCHOR.tolist(),
                "scale": [1.0] * 4}

    def fake_authenticate(path, digest):
        assert Path(path) == features and digest == expected_report_sha
        return rows, deployed, {}, {"counts": {"rows": len(rows)}}

    class Guard:
        def __init__(self, *unused, **kwargs):
            del kwargs

        def __enter__(self):
            return self

        def __exit__(self, *unused):
            return False

        @contextmanager
        def operation(self, unused, update=False):
            del unused, update
            yield

        def sample_and_append(self, stage):
            return {"stage": stage, "elapsed_s": 0.01,
                    "rss_bytes": 1, "swap_growth_bytes": 0}

    monkeypatch.setattr(trainer, "ROWS", len(rows))
    monkeypatch.setattr(trainer, "STEPS", 2)
    monkeypatch.setattr(trainer, "authenticate", fake_authenticate)
    monkeypatch.setattr(trainer, "AtomicResourceWatchdog", Guard)
    monkeypatch.setattr(trainer, "system_swap_bytes", lambda: 0)
    monkeypatch.setattr(trainer, "projection", lambda elapsed, durations: 1.0)
    monkeypatch.setattr(trainer, "qualified", lambda summary: gate)
    output = tmp_path / "output"
    report = trainer.run(features, expected_report_sha, output)
    predictions = [json.loads(line) for line in
                   (output / "predictions.jsonl").read_text().splitlines()]
    models = json.loads((output / "fold-models.json").read_text())["models"]
    assert len(predictions) == len(rows)
    assert len({value["id"] for value in predictions}) == len(rows)
    assert len(models) == 4
    assert all(model["training_eligible_rows"] == 3 for model in models)
    assert report["counts"]["gradient_eligible_rows"] == 4
    assert report["counts"]["feature_training_eligible_rows"] == 6
    assert report["counts"]["no_candidate_rows"] == 1
    assert report["counts"]["no_positive_rows"] == 1
    assert report["counts"]["all_positive_rows"] == 2
    assert report["qualification_passed"] is gate
    assert report["full_fit_performed"] is gate
    assert report["counts"]["completed_updates"] == (10 if gate else 8)
    assert (output / "model.json").exists() is gate
    assert (output / "full-fit-updates.jsonl").exists() is gate
    if gate:
        model = json.loads((output / "model.json").read_text())
        assert model["features_manifest_sha256"] == expected_report_sha
        assert model["features_rows_sha256"] == trainer.sha(features / "rows.jsonl")
        assert model["numerical_core_sha256"] == trainer.CORE_SHA256
        assert model["recipe"] == \
            "component-fold-learned-scaled-200-adam-anchor-v1"
    completion = json.loads((output / "completion.json").read_text())
    assert completion["completed"] is True
    assert completion["report_sha256"] == trainer.sha(output / "report.json")
