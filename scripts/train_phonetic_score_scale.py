"""Run the predeclared four-arm component-fold TRAIN calibration comparison."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "2"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_phonetic_choice_calibration import authenticate, read, write, sha, PREPARED
from scripts.source_span_resources import AtomicResourceWatchdog, system_swap_bytes
from scripts import phonetic_score_scale as core
import numpy as np

SCHEMA = "phonetic-score-scale-training-v1"
PLAN = ROOT / ".omx/plans/phonetic-score-scale-training-comparison-v1.md"
LIMITS = {"whole_seconds": 600., "update_seconds": 60.,
          "rss_bytes": 2 * 1024**3, "swap_growth_bytes": 512 * 1024**2}
STEPS = 200


def append(stream, record):
    stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def projection(elapsed, durations):
    core.need(len(durations) == 3 and all(d >= 0 and np.isfinite(d) for d in durations), "probe durations differ")
    # Every probe uses all rows, while fold fits use fewer rows.
    return 1.3 * (elapsed + max(durations) * (16 * STEPS + STEPS)) + 60.


def check_gate(summaries):
    before, after = summaries["fixed_unscaled"], summaries["learned_scaled"]
    return after["top1"] > before["top1"] and after["cross_entropy"] < before["cross_entropy"]


def run(output):
    output = Path(output)
    started = time.monotonic()
    baseline_swap = system_swap_bytes()
    output.mkdir(parents=True, exist_ok=False)
    completed_updates = 0
    try:
        with AtomicResourceWatchdog(started, baseline_swap, output / "resources.jsonl",
                                    limits=LIMITS, interval_s=.5) as watchdog:
            with watchdog.operation("authenticate_train", update=True):
                manifest, pins = authenticate()
                for path in (Path(__file__), PLAN, Path(core.__file__)):
                    pins[str(path.relative_to(ROOT))] = sha(path)
                write(output / "pins.json", pins)
                with (PREPARED / "rows.jsonl").open() as stream:
                    rows = [json.loads(line) for line in stream]
                base, features, valid, positives, folds = core.arrays(rows)
                core.need(len(rows) == 1483 and int(valid.sum()) == 202603, "fixed TRAIN denominator differs")
                write(output / "folds.json", [{"id": r["id"], "component_id": r["component_id"], "fold": int(f)}
                                              for r, f in zip(rows, folds)])
            all_indices = np.arange(len(rows))
            with watchdog.operation("prepare_probe", update=True):
                probe_scale = core.feature_scale(features, valid, all_indices, True)
                probe_matrix = core.design(base, features, probe_scale)
                identity = core.losses_and_scores(core.ANCHOR, probe_matrix, valid, positives)[1]
                core.need(np.array_equal(identity, np.where(valid, base, -np.inf)), "initial base parity differs")
            parameters, first, second = core.ANCHOR.copy(), np.zeros(5), np.zeros(5)
            durations = []
            with (output / "probe-updates.jsonl").open("x") as stream:
                for step in range(1, 4):
                    tick = time.monotonic()
                    with watchdog.operation(f"probe:{step}", update=True):
                        parameters, first, second, record = core.adam_step(
                            parameters, first, second, step, probe_matrix, valid, positives, True)
                        append(stream, record)
                    durations.append(time.monotonic() - tick)
            projected = projection(time.monotonic() - started, durations)
            with watchdog.operation("qualify_resources", update=True):
                write(output / "qualification.json", {"qualified": projected <= LIMITS["whole_seconds"],
                    "projected_seconds": projected, "probe_durations_s": durations,
                    "max_fit_updates": 3400, "performance_used_for_selection": False})
                core.need(projected <= LIMITS["whole_seconds"], "resource projection exceeds limit")
            del probe_matrix, identity
            models = []
            out_of_fold = {arm: [] for arm in core.ARMS}
            with (output / "updates.jsonl").open("x") as updates, (output / "predictions.jsonl").open("x") as predictions:
                for fold in range(4):
                    training = np.flatnonzero(folds != fold)
                    testing = np.flatnonzero(folds == fold)
                    core.need(len(training) and len(testing) and not set(training) & set(testing), "fold domain differs")
                    for arm in core.ARMS:
                        learn_alpha = arm.startswith("learned_")
                        scaled = arm.endswith("_scaled")
                        with watchdog.operation(f"prepare:{fold}:{arm}", update=True):
                            scale = core.feature_scale(features, valid, training, scaled)
                            matrix = core.design(base[training], features[training], scale)
                            mask, target = valid[training], positives[training]
                            # Every arm/fold restarts; probe/fold states are never reused.
                            parameters, first, second = core.ANCHOR.copy(), np.zeros(5), np.zeros(5)
                        for step in range(1, STEPS + 1):
                            with watchdog.operation(f"fit:{fold}:{arm}:{step}", update=True):
                                parameters, first, second, record = core.adam_step(
                                    parameters, first, second, step, matrix, mask, target, learn_alpha)
                                append(updates, {"fold": fold, "arm": arm, **record})
                                completed_updates += 1
                        with watchdog.operation(f"evaluate:{fold}:{arm}", update=True):
                            model = {"fold": fold, "arm": arm, "scale": scale.tolist(),
                                     "parameters": parameters.tolist(), "training_rows": len(training),
                                     "held_rows": len(testing), "steps": STEPS}
                            models.append(model)
                            held = core.design(base[testing], features[testing], scale)
                            result = core.predictions(rows, testing, parameters, held, valid[testing], positives[testing])
                            for record in result:
                                append(predictions, {"fold": fold, "arm": arm, **record})
                            out_of_fold[arm].extend(result)
                        print(json.dumps({"fold": fold, "arm": arm, "completed_updates": completed_updates,
                                          "elapsed_s": time.monotonic() - started}), flush=True)
            with watchdog.operation("close_fold_comparison", update=True):
                write(output / "fold-models.json", models)
                summaries = {arm: core.summary(sorted(result, key=lambda r: r["id"])) for arm, result in out_of_fold.items()}
                core.need(all(s["rows"] == 1483 for s in summaries.values()), "OOF coverage differs")
                # Reload every saved fold model and reproduce its actual held predictions.
                for model in read(output / "fold-models.json"):
                    testing = np.flatnonzero(folds == model["fold"])
                    held = core.design(base[testing], features[testing], np.array(model["scale"]))
                    repeated = core.predictions(rows, testing, np.array(model["parameters"]), held, valid[testing], positives[testing])
                    expected = [r for r in out_of_fold[model["arm"]] if core.fold_for(r["component_id"]) == model["fold"]]
                    core.need(repeated == expected, "fold model reload differs")
                selected = check_gate(summaries)
                write(output / "oof-decision.json", {"summaries": summaries,
                    "preselected_arm": "learned_scaled", "qualification_passed": selected,
                    "recipe_selected_using_evaluation_data": False})
            full_model = None
            if selected:
                with watchdog.operation("prepare_full_fit", update=True):
                    scale = core.feature_scale(features, valid, all_indices, True)
                    matrix = core.design(base, features, scale)
                    parameters, first, second = core.ANCHOR.copy(), np.zeros(5), np.zeros(5)
                with (output / "full-fit-updates.jsonl").open("x") as stream:
                    for step in range(1, STEPS + 1):
                        with watchdog.operation(f"full_fit:{step}", update=True):
                            parameters, first, second, record = core.adam_step(
                                parameters, first, second, step, matrix, valid, positives, True)
                            append(stream, record)
                            completed_updates += 1
                with watchdog.operation("save_full_fit", update=True):
                    full_model = {"schema": "phonetic-score-scale-model-v1", "arm": "learned_scaled",
                        "scale": scale.tolist(), "parameters": parameters.tolist(), "steps": STEPS,
                        "features_manifest_sha256": sha(PREPARED / "manifest.json")}
                    write(output / "model.json", full_model)
                    reloaded = read(output / "model.json")
                    core.need(np.array_equal(np.einsum("rcf,f->rc", matrix, parameters),
                        np.einsum("rcf,f->rc", matrix, np.array(reloaded["parameters"]))), "full model reload differs")
            with watchdog.operation("verify_pins", update=True):
                core.need(all(sha(ROOT / name) == digest for name, digest in pins.items()), "runtime pin changed")
            report = {"schema": SCHEMA, "completed": True, "pins": pins, "limits": LIMITS,
                "rows": 1483, "candidates": 202603, "fold_rows": {str(f): int((folds == f).sum()) for f in range(4)},
                "summaries": summaries, "preselected_arm": "learned_scaled", "qualification_passed": selected,
                "completed_updates": completed_updates, "full_fit_performed": selected,
                "model": full_model, "production_changed": False, "neural_model_loaded": False,
                "evaluation_examples_read": False,
                "artifacts": {name: sha(output / name) for name in ("folds.json", "fold-models.json",
                    "updates.jsonl", "predictions.jsonl", "oof-decision.json", "qualification.json", "probe-updates.jsonl")},
                "limitations": ["Only calibration labels are held out by component; frozen upstream models have seen TRAIN.",
                                "The FIT candidate pool predates the latest phonetic candidate union.",
                                "OOF qualification is not DEV/TEST success or deployment authorization."]}
            if selected:
                report["artifacts"].update({name: sha(output / name) for name in ("model.json", "full-fit-updates.jsonl")})
            with watchdog.operation("write_report", update=True):
                write(output / "report.json", report)
            terminal = watchdog.sample_and_append("after_report")
            with watchdog.operation("write_completion", update=True):
                write(output / "completion.json", {"completed": True,
                    "report_sha256": sha(output / "report.json"), "resource_after_report": terminal})
            watchdog.sample_and_append("after_completion")
        return report
    except BaseException as error:
        write(output / "failure.json", {"completed": False, "completed_updates": completed_updates,
            "error": f"{type(error).__name__}: {error}", "evaluation_examples_read": False})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    result = run(parser.parse_args().output)
    print(json.dumps({key: result[key] for key in ("completed", "summaries", "qualification_passed", "completed_updates")}))
