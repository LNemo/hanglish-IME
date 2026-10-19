import math

import pytest

from scripts import evaluate_difference_pairwise_lora_fresh as worker


def test_rejected_train_gate_precedes_every_fresh_or_runtime_operation(monkeypatch):
    def rejected():
        raise ValueError("accepted pairwise TRAIN proof required")

    def forbidden(*args, **kwargs):
        pytest.fail("rejected TRAIN must not open downstream resources")

    monkeypatch.setattr(worker.bridge, "require_training_proof", rejected)
    monkeypatch.setattr(worker, "worker_identity", forbidden)
    monkeypatch.setattr(worker.bridge, "prepare", forbidden)
    monkeypatch.setattr(worker, "read_inputs", forbidden)
    monkeypatch.setattr(worker, "load_verified_runtime", forbidden)
    with pytest.raises(ValueError, match="accepted pairwise TRAIN proof required"):
        worker.prepare_launch()


@pytest.mark.parametrize("count", [-1, True, 1.5, 4000])
def test_invalid_or_over_budget_complete_workload_rejected(count):
    with pytest.raises(ValueError):
        worker.estimate_work(count, {"pair_inference_p95": 6.3263,
                                    "save_reload_finalization_seconds": 18.841})


@pytest.mark.parametrize("seconds", [0, -1, True, math.nan, math.inf])
def test_missing_or_invalid_timing_cannot_qualify_work(seconds):
    with pytest.raises(ValueError):
        worker.estimate_work(888, {"pair_inference_p95": seconds,
                                  "save_reload_finalization_seconds": 18.841})
