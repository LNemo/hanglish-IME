import hashlib
import json

import pytest

from scripts.verify_supported_phonetic_transducer import (
    EXPECTED_CONFIG, EXPECTED_RECIPE, LEGACY_REPORT_SHA, recompute_metrics, replay_predictions,
    validate_probe, validate_training_header, verify_run,
)


def example():
    records = [
        {"index": 1, "split": "lexical_holdout", "roman": "gachi", "surface": "같이",
         "known_family_root": True, "family_has_train": False},
        {"index": 2, "split": "lexical_holdout", "roman": "gachi", "surface": "가치",
         "known_family_root": False, "family_has_train": False},
    ]
    predictions = [{"split": "lexical_holdout", "roman": "gachi", "pair_indices": [1, 2],
                    "outputs": [{"text": "가치", "logprob": -1.}]}]
    return records, predictions


def test_accepted_set_success_does_not_hide_missing_spelling_variant():
    records, predictions = example()
    result = recompute_metrics(records, predictions)["lexical_holdout"]
    assert result["all"]["inputs"] == result["all"]["accepted_top8"] == 1
    assert result["all"]["variants"] == 2
    assert result["all"]["variants_top8"] == 1
    assert result["known_root"]["accepted_top8"] == 0
    assert result["root_unknown"]["accepted_top8"] == 1


@pytest.mark.parametrize("mode", ["duplicate", "missing", "membership", "invalid", "nan"])
def test_incomplete_or_invalid_predictions_rejected(mode):
    records, predictions = example()
    if mode == "duplicate": predictions *= 2
    elif mode == "missing": predictions.clear()
    elif mode == "membership": predictions[0]["pair_indices"] = [1]
    elif mode == "invalid": predictions[0]["outputs"][0]["text"] = "갗"
    else: predictions[0]["outputs"][0]["logprob"] = float("nan")
    with pytest.raises(ValueError):
        recompute_metrics(records, predictions)


def test_empty_beam_and_family_without_train_remain_in_denominator():
    records, predictions = example()
    for row in records: row["split"] = "morph_combination_holdout"
    predictions[0]["split"] = "morph_combination_holdout"
    predictions[0]["outputs"] = []
    result = recompute_metrics(records, predictions)["morph_combination_holdout"]
    assert result["all"]["empty"] == result["all"]["inputs"] == 1
    assert result["family_without_train"]["variants"] == 2
    assert result["family_has_train"]["inputs"] == 0


def test_fresh_replay_compares_text_and_scores():
    _, predictions = example()
    class Model:
        def generate(self, roman, beam_size):
            assert (roman, beam_size) == ("gachi", 8)
            return [("가치", -1.)]
    assert replay_predictions(Model(), predictions) == 1
    predictions[0]["outputs"][0]["logprob"] = -2.
    with pytest.raises(ValueError, match="scores differ"):
        replay_predictions(Model(), predictions)


def probe_fixture():
    contract = {"recipe": EXPECTED_RECIPE, "model_config": EXPECTED_CONFIG,
                "source_hashes": {"source": "digest"}}
    probe = {
        "schema": "supported-phonetic-transducer-probe-v1", "completed": True, "accepted": True,
        "contract": contract, "source_hashes": contract["source_hashes"],
        "source_hashes_after": contract["source_hashes"], "prototype_discarded": True,
        "training_run": False, "held_evaluation_run": False, "morph_validation_forward_run": True,
        "lexical_validation_run": False, "model_updates_run": 16, "beam_inputs_run": 256,
        "update_observations": [{"padded_cost": 100, "seconds": .01} for _ in range(16)],
        "timing": {"updates_seconds": .16, "morph_validation_seconds": .1, "beams_seconds": 1.,
                   "probe_elapsed_seconds": 2.,
                   "projected_full_train_and_all_held_beam_seconds": 62 + 1.3 * (24.12 + .6 + 3292 / 256)},
        "projection": {"train_update_batches": 2412, "morph_selection_full_passes": 6,
                       "held_unique_beam_inputs": 3292, "held_beam_evaluation_passes": 1,
                       "multiplier": 1.3, "margin_seconds": 60,
                       "observed_update_padded_cost": 1600, "full_training_padded_cost": 241200},
        "peak_rss_bytes": 1000,
    }
    return probe, contract


@pytest.mark.parametrize("corruption", [None, "rss", "projection", "phase", "observations", "source"])
def test_probe_acceptance_needs_actual_closed_resource_evidence(corruption):
    probe, contract = probe_fixture()
    if corruption == "rss": probe["peak_rss_bytes"] = 3 * 1024**3
    elif corruption == "projection": probe["timing"]["projected_full_train_and_all_held_beam_seconds"] = 1.
    elif corruption == "phase": probe["lexical_validation_run"] = True
    elif corruption == "observations": probe["update_observations"].pop()
    elif corruption == "source": probe["source_hashes_after"] = {}
    if corruption is None:
        validate_probe(probe, contract)
    else:
        with pytest.raises(ValueError): validate_probe(probe, contract)


def test_verify_run_rejects_changed_recipe_before_model_loading(tmp_path):
    recipe = {**EXPECTED_RECIPE, "seed": 0}
    report = {
        "schema": "supported-phonetic-transducer-training-v1", "completed": True,
        "training_run": True, "from_scratch": True, "lexical_used_for_training": False,
        "lexical_used_for_selection": False, "source_hashes": {}, "source_hashes_after": {},
        "contract": {"recipe": recipe, "model_config": EXPECTED_CONFIG, "source_hashes": {}},
    }
    raw = json.dumps(report).encode()
    (tmp_path / "report.json").write_bytes(raw)
    with pytest.raises(ValueError, match="frozen recipe differs"):
        verify_run(tmp_path, hashlib.sha256(raw).hexdigest(), tmp_path / "absent-probe.json",
                   tmp_path / "verification.json")
    assert not (tmp_path / "verification.json").exists()


def test_legacy_metadata_schema_collision_only_accepts_exact_closed_training_report():
    report = {"schema": "supported-phonetic-transducer-checkpoint-v1",
              "completed": True, "training_run": True}
    assert validate_training_header(report, LEGACY_REPORT_SHA) is True
    with pytest.raises(ValueError, match="schema unrecognized"):
        validate_training_header(report, "0" * 64)
    report["training_run"] = False
    with pytest.raises(ValueError, match="training incomplete"):
        validate_training_header(report, LEGACY_REPORT_SHA)
