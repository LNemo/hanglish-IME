"""Fixed four-weight FIT calibration of already captured neural scores.

No neural network is loaded. NumPy computes the exact analytical gradient of
the frozen row-mean multi-positive listwise objective plus squared L2 penalty.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

# Set before NumPy is imported, including in child numerical libraries.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "2"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.source_span_resources import AtomicResourceWatchdog, system_swap_bytes

PREPARED = ROOT / ".omx/artifacts/phonetic-choice-features-v1"
PLAN = ROOT / ".omx/plans/phonetic-choice-calibration-v1.md"
LIMITS = {"whole_seconds": 600., "update_seconds": 60.,
          "rss_bytes": 2 * 1024**3, "swap_growth_bytes": 512 * 1024**2}
STEPS = 200
PREPARED_PINS = {
    "manifest.json": "b884847fc4fc5f69d20acb95628135f82ba5fa6596479e540600f29e59517a84",
    "rows.jsonl": "20071a1874e057f346b4d3f131767c3a13182806576c98ac773c09f389ebccec",
    "independent-verification.json": "9eed6ed3642be4fe34a5741886f8c9b9a36e8654e88eb0fec27e4eda77baef32",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def objective(weights, base, features, positives):
    import numpy as np
    logits = base + np.einsum("rcf,f->rc", features, weights)
    maximum = np.max(logits, axis=1, keepdims=True)
    exp_all = np.exp(logits - maximum)
    denominator = exp_all.sum(axis=1, keepdims=True)
    positive_logits = np.where(positives, logits, -np.inf)
    positive_max = np.max(positive_logits, axis=1, keepdims=True)
    exp_positive = np.exp(positive_logits - positive_max)
    positive_denominator = exp_positive.sum(axis=1, keepdims=True)
    row_loss = maximum + np.log(denominator) - positive_max - np.log(positive_denominator)
    loss = float(row_loss.mean() + .01 * np.dot(weights, weights))
    delta = exp_all / denominator - exp_positive / positive_denominator
    gradient = np.einsum("rc,rcf->f", delta, features) / len(base) + .02 * weights
    require(np.isfinite(loss) and np.isfinite(gradient).all(), "nonfinite objective")
    return loss, gradient


def adam_step(weights, first, second, step, base, features, positives):
    import numpy as np
    loss, gradient = objective(weights, base, features, positives)
    norm = float(np.linalg.norm(gradient))
    clipped = gradient / norm if norm > 1. else gradient
    first = .9 * first + .1 * clipped
    second = .999 * second + .001 * clipped * clipped
    corrected_first = first / (1. - .9 ** step)
    corrected_second = second / (1. - .999 ** step)
    weights = weights - .05 * corrected_first / (np.sqrt(corrected_second) + 1e-8)
    require(np.isfinite(weights).all(), "nonfinite optimizer weights")
    return weights, first, second, {
        "step": step, "loss_before_update": loss, "gradient": gradient.tolist(),
        "gradient_norm_before_clip": norm, "weights_after_update": weights.tolist(),
    }


def authenticate():
    require(all(sha(PREPARED / name) == value for name, value in PREPARED_PINS.items()),
            "prepared artifact identity differs")
    manifest = read(PREPARED / "manifest.json")
    proof = read(PREPARED / "independent-verification.json")
    require(proof.get("verified") is True
            and proof.get("manifest_sha256") == sha(PREPARED / "manifest.json"),
            "feature preparation has no matching independent proof")
    require(manifest["counts"] == {"rows": 1483, "candidates": 202603,
                                  "paths": 202607, "positive_indices": 1483,
                                  "joined_pairs": 26544}, "FIT denominator differs")
    require(manifest["rows_sha256"] == sha(PREPARED / "rows.jsonl"), "feature data differs")
    pins = dict(manifest["pins"])
    for path in (Path(__file__), PLAN, PREPARED / "manifest.json",
                 PREPARED / "independent-verification.json", PREPARED / "rows.jsonl",
                 ROOT / "scripts/source_span_resources.py", ROOT / "scripts/preflight_pko_seq2seq.py"):
        key = str(path.relative_to(ROOT))
        require(key not in pins or pins[key] == sha(path), "prepared implementation changed")
        pins[key] = sha(path)
    require(all(sha(ROOT / name) == value for name, value in pins.items()), "input pin changed")
    return manifest, pins


def run(output):
    started = time.monotonic()
    baseline = system_swap_bytes()
    output.mkdir(parents=True, exist_ok=False)
    try:
        with AtomicResourceWatchdog(started, baseline, output / "resources.jsonl",
                                    limits=LIMITS, interval_s=.5) as watchdog:
            with watchdog.operation("authenticate", update=True):
                manifest, pins = authenticate()
                write(output / "pins.json", pins)
            with watchdog.operation("prepare_arrays", update=True):
                import numpy as np
                rows = [json.loads(line) for line in (PREPARED / "rows.jsonl").read_text().splitlines()]
                width = max(len(r["base_scores"]) for r in rows)
                base = np.full((len(rows), width), -np.inf, dtype=np.float64)
                features = np.zeros((len(rows), width, 4), dtype=np.float64)
                positives = np.zeros((len(rows), width), dtype=bool)
                for i, row in enumerate(rows):
                    count = len(row["base_scores"])
                    base[i, :count] = row["base_scores"]
                    features[i, :count] = row["features"]
                    positives[i, row["positive_indices"]] = True
                require(np.isfinite(features).all() and positives.any(axis=1).all(), "invalid arrays")
                zero = np.zeros(4, dtype=np.float64)
                require(np.array_equal(base + np.einsum("rcf,f->rc", features, zero), base),
                        "zero-weight base score parity failed")
                initial_loss, _ = objective(zero, base, features, positives)
                # Candidate text/order is unchanged; exactly equal scores preserve
                # whichever stable tie ordering the runtime adapter uses.
                del rows
            weights = zero.copy(); first = zero.copy(); second = zero.copy()
            durations = []
            probe_records = []
            with (output / "probe-updates.jsonl").open("x") as stream:
                for step in range(1, 4):
                    tick = time.monotonic()
                    with watchdog.operation(f"probe:{step}", update=True):
                        weights, first, second, record = adam_step(
                            weights, first, second, step, base, features, positives)
                        probe_records.append(record)
                        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                    durations.append(time.monotonic() - tick)
            projected = 1.3 * (time.monotonic() - started + max(durations) * STEPS * 2) + 60.
            write(output / "qualification.json", {
                "qualified": projected <= LIMITS["whole_seconds"],
                "probe_step_seconds": durations, "probe_updates": probe_records,
                "projected_seconds": projected,
                "projection": "1.3*(elapsed+max_probe_seconds*400)+60",
            })
            require(projected <= LIMITS["whole_seconds"], "resource projection exceeded")
            trained = None
            for arm in ("fit", "replay"):
                weights = zero.copy(); first = zero.copy(); second = zero.copy()
                with (output / f"{arm}-updates.jsonl").open("x") as stream:
                    for step in range(1, STEPS + 1):
                        with watchdog.operation(f"{arm}:{step}", update=True):
                            weights, first, second, record = adam_step(
                                weights, first, second, step, base, features, positives)
                            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                            stream.flush()
                            os.fsync(stream.fileno())
                if arm == "fit":
                    trained = weights.copy()
                else:
                    require(np.array_equal(trained, weights), "deterministic replay failed")
            with watchdog.operation("finalize", update=True):
                require(sha(output / "fit-updates.jsonl") == sha(output / "replay-updates.jsonl"),
                        "full optimizer trajectory replay differs")
                require(all(sha(ROOT / name) == value for name, value in pins.items()), "input changed")
                final_loss, _ = objective(trained, base, features, positives)
                write(output / "weights.json", {"schema": "phonetic-choice-calibration-v1",
                    "feature_names": manifest["feature_names"], "weights": trained.tolist(),
                    "steps": STEPS, "intercept": 0., "feature_manifest_sha256": sha(PREPARED / "manifest.json")})
                reloaded = np.array(read(output / "weights.json")["weights"], dtype=np.float64)
                fitted_scores = base + np.einsum("rcf,f->rc", features, trained)
                require(np.array_equal(fitted_scores, base + np.einsum("rcf,f->rc", features, reloaded)),
                        "checkpoint score reload differs")
                report = {"schema": "phonetic-choice-calibration-training-v1", "completed": True,
                    "counts": manifest["counts"], "steps": STEPS,
                    "initial_loss": initial_loss, "final_loss": final_loss,
                    "weights": trained.tolist(), "feature_names": manifest["feature_names"],
                    "numpy_version": np.__version__, "limits": LIMITS,
                    "weights_sha256": sha(output / "weights.json"),
                    "updates_sha256": sha(output / "fit-updates.jsonl"),
                    "replay_sha256": sha(output / "replay-updates.jsonl"),
                    "qualification_sha256": sha(output / "qualification.json"),
                    "zero_weight_score_parity": True, "all_step_replay_exact": True,
                    "reload_scores_exact": True, "neural_model_loaded": False,
                    "dev_test_reserve_accessed": False, "production_changed": False,
                    "quality_evaluated": False, "pins": pins}
                write(output / "report.json", report)
            terminal = watchdog.sample_and_append("after_report")
            with watchdog.operation("write_completion", update=True):
                write(output / "completion.json", {"completed": True,
                    "report_sha256": sha(output / "report.json"), "resource_after_report": terminal})
            watchdog.sample_and_append("after_completion")
        return report
    except BaseException as error:
        write(output / "failure.json", {"completed": False, "error": f"{type(error).__name__}: {error}"})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    result = run(parser.parse_args().output)
    print(json.dumps({key: result[key] for key in ("completed", "initial_loss", "final_loss", "weights")}))
