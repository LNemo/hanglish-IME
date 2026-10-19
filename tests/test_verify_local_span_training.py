from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_local_span_training.py"
SPEC = importlib.util.spec_from_file_location("verify_local_span_training", SCRIPT)
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def write_jsonl(path, values):
    path.write_text("".join(json.dumps(value, sort_keys=True) + "\n" for value in values))


class Helper:
    @staticmethod
    def argmax(values):
        return max(range(len(values)), key=values.__getitem__)


def test_softplus_replay_is_stable_for_large_margins_and_empty_rows():
    assert verifier.mean_softplus_negative([]) == 0.
    assert verifier.mean_softplus_negative([1000.]) == pytest.approx(0.)
    assert verifier.mean_softplus_negative([-1000.]) == pytest.approx(1000.)
    assert verifier.mean_softplus_negative([0.]) == pytest.approx(math.log(2.))
    tolerance = verifier.softplus_loss_tolerance([-1000., 1000.], 500.)
    assert tolerance >= 8 * 2. ** (math.frexp(1000.)[1] - 24)
    assert tolerance < .01


def update_records(rows, epochs=2):
    values = []
    update = 0
    for epoch in range(epochs):
        for row in rows:
            update += 1
            local, candidate = .5, .75
            total = verifier.fp32(verifier.fp32(candidate)
                                  + verifier.fp32(.25 * verifier.fp32(local)))
            values.append({"update": update, "epoch": epoch, "id": row["id"],
                           "losses": {"total": total, "candidate": candidate,
                                      "local": local},
                           "gradient_norm": 1.,
                           "gradient_groups": {name: float(update)
                                               for name in verifier.GRADIENT_GROUPS}})
    return values


def test_updates_replay_exact_two_epoch_order_loss_and_gradients(tmp_path, monkeypatch):
    rows = [{"id": "a"}, {"id": "b"}]
    monkeypatch.setattr(verifier, "EPOCHS", 2)
    monkeypatch.setattr(verifier, "UPDATES", 4)
    path = tmp_path / "updates.jsonl"
    write_jsonl(path, update_records(rows))
    result = verifier.verify_updates(rows, path)
    assert result["updates"] == 4
    assert result["epoch_mean_losses"][0] == {
        "total": .875, "candidate": .75, "local": .5}
    assert all(value == 4. for value in result["gradient_maxima"].values())


@pytest.mark.parametrize("mutation,match", [
    ("order", "order"), ("combined", "combined"), ("gradient", "gradient")])
def test_updates_reject_tampered_order_loss_and_gradients(
        tmp_path, monkeypatch, mutation, match):
    rows = [{"id": "a"}, {"id": "b"}]
    monkeypatch.setattr(verifier, "EPOCHS", 2)
    monkeypatch.setattr(verifier, "UPDATES", 4)
    values = update_records(rows)
    if mutation == "order":
        values[1]["id"] = "a"
    elif mutation == "combined":
        values[1]["losses"]["total"] += .1
    else:
        values[1]["gradient_groups"][verifier.GRADIENT_GROUPS[0]] = float("nan")
    path = tmp_path / "updates.jsonl"; write_jsonl(path, values)
    with pytest.raises(ValueError, match=match):
        verifier.verify_updates(rows, path)


def test_control_replay_requires_exact_historical_scores_and_softplus(tmp_path):
    rows = [{"id": "a"}, {"id": "b"}]
    labels = [{"preferences": [{"witnesses": [{}]}]}, {"preferences": []}]
    historical = [{"scores": [1., 2.]}, {"scores": [3., 4.]}]
    values = [
        {"id": "a", "scores": [1., 2.], "local_margins": [.5],
         "local_loss": verifier.mean_softplus_negative([.5])},
        {"id": "b", "scores": [3., 4.], "local_margins": [], "local_loss": 0.},
    ]
    path = tmp_path / "control.jsonl"; write_jsonl(path, values)
    result = verifier.verify_control(rows, labels, historical, path)
    assert result["preferences"] == 1 and result["positive_margins"] == 1
    changed = deepcopy(values); changed[0]["scores"][0] += 1
    write_jsonl(path, changed)
    with pytest.raises(ValueError, match="scores differ"):
        verifier.verify_control(rows, labels, historical, path)


def after_fixture():
    rows = [{"id": "a", "candidates": ["x", "y"], "positive_indices": [1]}]
    labels = [{"preferences": [{"witnesses": [{}]}]}]
    before = [{"scores": [verifier.fp32(.1), verifier.fp32(.2)]}]
    old = [{"correct": 0}]
    residuals = [verifier.fp32(.3), verifier.fp32(.4)]
    scores = [verifier.fp32(before[0]["scores"][index] + residuals[index])
              for index in range(2)]
    margins = [.7]
    values = [{"id": "a", "correct": 1, "scores": scores,
               "base_scores": before[0]["scores"], "residuals": residuals,
               "local_margins": margins,
               "local_loss": verifier.mean_softplus_negative(margins)}]
    return rows, labels, before, old, values


def test_after_replays_fp32_scores_argmax_gains_and_local_loss(tmp_path):
    rows, labels, before, old, values = after_fixture()
    path = tmp_path / "after.jsonl"; write_jsonl(path, values)
    result = verifier.verify_after(Helper, rows, labels, before, old, path)
    assert (result["correct"], result["gains"], result["losses"]) == (1, 1, 0)
    assert result["local_positive_margins"] == result["local_margin_count"] == 1


@pytest.mark.parametrize("field,match", [
    ("score", "base plus residual"), ("correct", "argmax"), ("loss", "softplus")])
def test_after_rejects_score_argmax_and_softplus_tampering(tmp_path, field, match):
    rows, labels, before, old, values = after_fixture()
    if field == "score":
        values[0]["scores"][0] += .01
    elif field == "correct":
        values[0]["correct"] = 0
    else:
        values[0]["local_loss"] += .01
    path = tmp_path / "after.jsonl"; write_jsonl(path, values)
    with pytest.raises(ValueError, match=match):
        verifier.verify_after(Helper, rows, labels, before, old, path)


def resource_row(stage, operation, elapsed, operation_elapsed):
    return {"at": "time", "elapsed_s": elapsed, "rss_bytes": 100,
            "swap_used_bytes": 10, "swap_growth_bytes": 0,
            "operation_elapsed_s": operation_elapsed,
            "stage": stage, "operation": operation}


def test_resources_require_complete_atomic_operation_order(tmp_path, monkeypatch):
    rows = [{"id": "a"}]
    monkeypatch.setattr(verifier, "UPDATES", 1)
    operations = verifier.expected_operations(rows)
    values = []
    elapsed = 0.
    elapsed += .05; values.append(resource_row("watchdog", None, elapsed, None))
    for operation in operations:
        elapsed += .1; values.append(resource_row("operation_start", operation, elapsed, 0.))
        elapsed += .1; values.append(resource_row("operation_end", operation, elapsed, .1))
    path = tmp_path / "resources.jsonl"; write_jsonl(path, values)
    final = {"stage": "final", "operation": None, "operation_elapsed_s": None,
             "elapsed_s": elapsed + .1, "rss_bytes": 100,
             "swap_used_bytes": 10, "swap_growth_bytes": 0,
             "at": "time"}
    report = {"baseline_swap_bytes": 10, "final_resources": final,
              "elapsed_s": elapsed + .2}
    result = verifier.verify_resources(rows, report, path)
    assert result["operation_pairs"] == len(operations)
    assert result["atomic_operation_attribution_verified"] is True
    changed = deepcopy(values); changed[3]["operation"] = "wrong"
    write_jsonl(path, changed)
    with pytest.raises(ValueError, match="order"):
        verifier.verify_resources(rows, report, path)


def test_one_witness_contract_is_necessary_for_margin_loss_replay():
    preference = {"witnesses": [{}, {}]}
    assert len(preference["witnesses"]) != 1
    # A mean margin cannot recover mean softplus when witnesses differ.
    witness_margins = [-2., 2.]
    assert verifier.mean_softplus_negative(witness_margins) != pytest.approx(
        verifier.mean_softplus_negative([sum(witness_margins) / 2]))


def test_verifier_has_no_torch_model_or_trainer_imports():
    source = SCRIPT.read_text()
    assert "import torch" not in source
    assert "from hanglish" not in source
    assert "from scripts.train_local_span_objective" not in source
    assert "from hanglish.models.local_span_objective" not in source


def test_alternate_run_is_not_authorized(tmp_path):
    with pytest.raises(ValueError, match="alternate"):
        verifier.verify(tmp_path)
