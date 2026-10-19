"""Run the cache-observing v2 synthetic long-context LoRA resource gate."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
import time


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import probe_long_context_lora_resources as v1
from scripts.long_context_lora_model import (
    LORA_CONFIG, LORA_LAYERS, SEED, add_materialized_gradients,
    apply_materialized_update, average_and_clip_gradients, configure_fresh_lora,
    create_optimizer, materialize_microbatch, mlx_log_mass_loss,
    ordinary_forward_choice_scores, require_matching_gradient_tree,
    require_trainable_contract, save_adapter, score_choices, tensor_hashes,
)
from scripts.long_context_lora_resources import (
    ResourceObservationError, observe_resources,
)


DEFAULT_OUTPUT = ROOT / ".omx/artifacts/long-context-lora-resource-v2"
AMENDMENT = ROOT / ".omx/plans/long-context-lora-cache-observation-amendment-v1.md"
CACHE_HANDOFF = ROOT / ".omx/plans/long-context-lora-cache-handoff-v1.json"
FAILED_V1_REPORT = ROOT / ".omx/artifacts/long-context-lora-resource-v1/report.json"
MLX_API_STUB = ROOT / ".venv-mlx/lib/python3.12/site-packages/mlx/core/__init__.pyi"
FIXED_V2_INPUTS = {
    AMENDMENT: "5857a6487f5d4437d380f6e94ac4403bdb3ff2b374bc2bdee18dc6b96a330df2",
    CACHE_HANDOFF: "697155362e8b7a9ba6b04aba1d22474b286ec459299d6d02760a1022d47df1df",
    FAILED_V1_REPORT: "7e8175c327a0f81bf2bcaeb2fda97f0954c6abd09c6b53d6f17d19046f6159fd",
    MLX_API_STUB: "36a0c96841e9a18daa17a8e4438bdd3f94a80a07527567ab3eddd3f2e7c47add",
    ROOT / "scripts/long_context_lora_model.py":
        "eb9b6c23ef6f639773e34782f04e1499810a448eee3a688a3385d9ef29c12015",
    ROOT / "scripts/probe_long_context_lora_resources.py":
        "52e489c4ab17521cfd34a12eedeef8e34bf5693c80e3afdc9aaeae81ae01e2aa",
}


def capture_provenance() -> tuple[dict, dict]:
    hashes, versions = v1.capture_provenance()
    handoff = json.loads(CACHE_HANDOFF.read_text())
    if (handoff.get("status") != "complete"
            or handoff.get("amendment_sha256") != FIXED_V2_INPUTS[AMENDMENT]
            or handoff.get("failed_resource_report_sha256") != FIXED_V2_INPUTS[FAILED_V1_REPORT]
            or handoff.get("architect", {}).get("verdict") != "APPROVE"
            or handoff.get("critic", {}).get("verdict") != "APPROVE"
            or handoff.get("critic", {}).get("after_completed_architect") is not True
            or handoff.get("ralplan_execution_handoff", {}).get("authorized") is not True):
        raise RuntimeError("approved cache-observer handoff differs")
    for path, expected in FIXED_V2_INPUTS.items():
        actual = v1.sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"fixed v2 input differs: {v1.relative_path(path)}")
        hashes[v1.relative_path(path)] = actual
    for path in (Path(__file__), ROOT / "scripts/long_context_lora_resources.py"):
        hashes[v1.relative_path(path)] = v1.sha256_file(path)
    return dict(sorted(hashes.items())), versions


def max_abs_delta(left, right) -> float:
    return max(abs(float(a) - float(b)) for a, b in zip(left, right))


def run(output: Path = DEFAULT_OUTPUT) -> dict:
    if output.exists():
        raise FileExistsError(f"resource artifact already exists: {output}")
    output.mkdir(parents=True)
    progress = output / "progress.jsonl"
    report_path = output / "report.json"
    started = time.monotonic()
    observations = []
    report = {
        "schema": "long-context-lora-resource-v2", "completed": False,
        "resource_feasible": False, "synthetic_only": True,
        "real_fit_rows_read": False, "real_training_run": False,
        "old_adapter_loaded": False, "limits": v1.LIMITS,
        "cache_observer": "scripts/long_context_lora_resources.py",
        "resource_observations": observations,
        "supersedes_failed_report": v1.relative_path(FAILED_V1_REPORT),
        "superseded_report_remains_failed": True,
    }
    model = tokenizer = backend_mx = checkpoint = None
    try:
        before, versions = capture_provenance()
        baseline = v1.system_resources()
        report.update({"source_hashes": before, "versions": versions,
                       "system_baseline": baseline})
        v1.write_progress(progress, "provenance_verified", files=len(before))

        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm.tuner.utils import load_adapters
        from mlx_lm.utils import load
        backend_mx = mx
        mx.set_memory_limit(v1.LIMITS["mlx_peak_bytes"])
        mx.set_cache_limit(v1.LIMITS["cache_bytes"])

        def checkpoint(stage: str, update_started: float | None = None):
            try:
                settled, record = observe_resources(
                    stage=stage, mx=mx, started=started, baseline=baseline,
                    limits=v1.LIMITS, system_reader=v1.system_resources,
                    rss_reader=v1.rss_bytes)
            except ResourceObservationError as error:
                observations.append(error.record)
                report["failure_resource_observation"] = error.record
                settled = error.record.get("post") or error.record.get("pre")
                if settled is not None:
                    report["failure_resources"] = settled
                v1.write_progress(progress, "resource_observation_failed",
                                  observation=error.record)
                raise
            observations.append(record)
            v1.write_progress(progress, "resource_observation", observation=record)
            if (update_started is not None
                    and time.monotonic() - update_started > v1.LIMITS["update_seconds"]):
                record["errors"].append("optimizer update time bound exceeded")
                report["failure_resource_observation"] = record
                report["failure_resources"] = settled
                raise RuntimeError("optimizer update time bound exceeded")
            return settled

        tiny_math = v1.validate_tiny_math(mx)
        checkpoint("tiny_math")
        model, tokenizer = load(
            str(v1.MODEL_DIR), tokenizer_config={"trust_remote_code": False})
        load_resources = checkpoint("model_load")
        if not bool(model.args.tie_word_embeddings):
            raise RuntimeError("pinned base unexpectedly has an untied output head")
        model.eval()
        rows, digit_ids = v1.exact_prompts(tokenizer)
        initial = configure_fresh_lora(mx, model)
        initial_hashes = tensor_hashes(initial)
        del initial
        checkpoint("lora_initialization")
        optimizer = create_optimizer()
        model.train()

        def loss_fn(active_model, row):
            scores = score_choices(
                mx, active_model, row["prompt_tokens"],
                digit_ids[:row["candidate_count"]])
            return mlx_log_mass_loss(mx, scores, row["allowed_gold_positions"])

        value_and_grad = nn.value_and_grad(model, loss_fn)
        warm_loss, warm_grads = value_and_grad(model, rows[-1])
        materialize_microbatch(mx, warm_loss, warm_grads)
        warm_flat = require_matching_gradient_tree(model, warm_grads)
        if not bool(mx.isfinite(warm_loss).item()):
            raise RuntimeError("warm 1024-token loss is nonfinite")
        if any(not bool(mx.all(mx.isfinite(value)).item()) for value in warm_flat.values()):
            raise RuntimeError("warm 1024-token gradient is nonfinite")
        warm_nonzero = any(bool(mx.any(value != 0).item()) for value in warm_flat.values())
        if not warm_nonzero:
            raise RuntimeError("warm 1024-token gradient is entirely zero")
        checkpoint("warm_1024_backward")
        v1.write_progress(progress, "warm_backward", loss=float(warm_loss.item()))

        update_times, update_records = [], []
        for update_index in range(v1.UPDATES):
            update_started = time.monotonic()

            def require_update_time():
                if time.monotonic() - update_started > v1.LIMITS["update_seconds"]:
                    raise RuntimeError("optimizer update time bound exceeded")

            accumulated, losses = None, []
            for microbatch_index, row in enumerate(rows, 1):
                loss, gradients = value_and_grad(model, row)
                materialize_microbatch(mx, loss, gradients)
                gradient_flat = require_matching_gradient_tree(model, gradients)
                if not bool(mx.isfinite(loss).item()) or any(
                        not bool(mx.all(mx.isfinite(value)).item())
                        for value in gradient_flat.values()):
                    raise RuntimeError("nonfinite loss or gradient")
                accumulated = add_materialized_gradients(mx, accumulated, gradients)
                checkpoint(
                    f"update_{update_index + 1}_microbatch_{microbatch_index}",
                    update_started)
                require_update_time()
                losses.append(float(loss.item()))
            clipped, norm = average_and_clip_gradients(
                mx, accumulated, v1.ACCUMULATION)
            checkpoint(f"update_{update_index + 1}_averaged_clipped", update_started)
            require_update_time()
            apply_materialized_update(mx, model, optimizer, clipped)
            resources = checkpoint(f"update_{update_index + 1}_updated", update_started)
            require_update_time()
            elapsed = time.monotonic() - update_started
            if elapsed > v1.LIMITS["update_seconds"]:
                raise RuntimeError("optimizer update time bound exceeded")
            update_times.append(elapsed)
            record = {
                "update": update_index + 1, "microbatches": v1.ACCUMULATION,
                "microbatch_losses": losses, "mean_loss": sum(losses) / len(losses),
                "gradient_norm_before_clip": float(norm.item()),
                "seconds": elapsed, "resources": resources,
            }
            update_records.append(record)
            report["updates"] = update_records
            report["update_timings_seconds"] = update_times
            v1.write_progress(progress, "optimizer_update", **record)

        final = require_trainable_contract(model)
        final_hashes = tensor_hashes(final)
        if initial_hashes == final_hashes:
            raise RuntimeError("adapter did not change")

        model.eval()
        equivalence, trained_logits = [], []
        for row in rows:
            selected = score_choices(
                mx, model, row["prompt_tokens"], digit_ids[:row["candidate_count"]])
            ordinary = ordinary_forward_choice_scores(
                mx, model, row["prompt_tokens"], digit_ids[:row["candidate_count"]])
            mx.eval(selected, ordinary)
            checkpoint(f"ordinary_forward_{row['prompt_token_count']}")
            left, right = selected.tolist(), ordinary.tolist()
            if not all(math.isfinite(float(value)) for value in [*left, *right]):
                raise RuntimeError("diagnostic logits are nonfinite")
            delta = max_abs_delta(left, right)
            if (delta > v1.FULL_FORWARD_TOLERANCE
                    or left.index(max(left)) != right.index(max(right))):
                raise RuntimeError("final-hidden scorer differs from ordinary forward")
            equivalence.append({
                "prompt_tokens": row["prompt_token_count"],
                "candidate_count": row["candidate_count"],
                "prompt_token_ids": row["prompt_tokens"],
                "prompt_sha256": row["prompt_sha256"],
                "selected_scorer_logits": left, "ordinary_forward_logits": right,
                "max_absolute_digit_delta": delta, "argmax": left.index(max(left)),
            })
            trained_logits.append(left)
            del ordinary

        scorer_times = []
        for row in rows:
            for repetition in range(v1.INFERENCE_REPETITIONS):
                timing_started = time.monotonic()
                scores = score_choices(
                    mx, model, row["prompt_tokens"], digit_ids[:row["candidate_count"]])
                mx.eval(scores)
                checkpoint(
                    f"inference_{row['prompt_token_count']}_{repetition + 1}")
                scorer_times.append(time.monotonic() - timing_started)
        report["scorer_timings_seconds"] = scorer_times

        adapter_dir = output / "adapter"
        adapter_dir.mkdir()
        adapter_path = adapter_dir / "adapters.safetensors"
        saved_hashes = save_adapter(mx, model, adapter_path)
        checkpoint("adapter_save")
        adapter_config = {
            "fine_tune_type": "lora", "num_layers": LORA_LAYERS,
            "lora_parameters": LORA_CONFIG,
        }
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps(adapter_config, sort_keys=True, indent=2) + "\n")
        adapter_sha = v1.sha256_file(adapter_path)
        del model, tokenizer, optimizer, value_and_grad, loss_fn, warm_grads, warm_flat
        del accumulated, clipped, final, gradients, gradient_flat, loss, scores
        model = tokenizer = None
        gc.collect()
        checkpoint("first_model_released")

        model, tokenizer = load(
            str(v1.MODEL_DIR), tokenizer_config={"trust_remote_code": False})
        model.freeze()
        load_adapters(model, str(adapter_dir))
        require_trainable_contract(model)
        model.eval()
        fresh_rows, fresh_digit_ids = v1.exact_prompts(tokenizer)
        if fresh_rows != rows or fresh_digit_ids != digit_ids:
            raise RuntimeError("fresh tokenizer synthetic prompt identity differs")
        reload_checks = []
        for row, expected in zip(rows, trained_logits):
            actual_array = score_choices(
                mx, model, row["prompt_tokens"], digit_ids[:row["candidate_count"]])
            mx.eval(actual_array)
            checkpoint(f"reload_{row['prompt_token_count']}")
            actual = actual_array.tolist()
            close = all(
                abs(float(a) - float(b))
                <= v1.RELOAD_ATOL + v1.RELOAD_RTOL * abs(float(b))
                for a, b in zip(actual, expected))
            if not close or actual.index(max(actual)) != expected.index(max(expected)):
                raise RuntimeError("fresh base plus adapter scorer parity failed")
            reload_checks.append({
                "prompt_tokens": row["prompt_token_count"],
                "expected_logits": expected, "actual_logits": actual,
                "max_absolute_delta": max_abs_delta(actual, expected),
                "prediction": actual.index(max(actual)),
            })

        estimate = v1.runtime_estimate(update_times, scorer_times)
        after, versions_after = capture_provenance()
        if before != after or versions != versions_after:
            raise RuntimeError("pinned sources or package versions changed during probe")
        terminal = checkpoint("terminal")
        report.update({
            "completed": True, "resource_feasible": True,
            "terminal_status": "success", "source_hashes_after": after,
            "versions_after": versions_after, "seed": SEED,
            "tiny_math_validation": tiny_math, "load_resources": load_resources,
            "shapes": [{key: value for key, value in row.items()
                        if key != "prompt_tokens"} for row in rows],
            "digit_token_ids": digit_ids,
            "warm_1024_backward": {
                "completed": True, "update_applied": False,
                "finite_gradients": True, "nonzero_gradient": warm_nonzero,
            },
            "updates": update_records, "update_timings_seconds": update_times,
            "scorer_timings_seconds": scorer_times, "runtime_estimate": estimate,
            "ordinary_forward_diagnostic": equivalence,
            "initial_trainable_tensor_hashes": initial_hashes,
            "final_trainable_tensor_hashes": final_hashes,
            "saved_trainable_tensor_hashes": saved_hashes,
            "adapter": {
                "sha256": adapter_sha,
                "config_sha256": v1.sha256_file(adapter_dir / "adapter_config.json"),
                "only_16_trainable_tensors": True,
            },
            "fresh_reload": {
                "atol": v1.RELOAD_ATOL, "rtol": v1.RELOAD_RTOL,
                "checks": reload_checks,
            },
            "real_training_launch_allowed_by_runtime_estimate":
                estimate["within_21600_seconds"],
            "system_terminal": terminal,
            "elapsed_seconds": time.monotonic() - started,
        })
        v1.write_progress(
            progress, "terminal_success", elapsed_seconds=report["elapsed_seconds"])
        return report
    except BaseException as error:
        if callable(checkpoint) and "failure_resource_observation" not in report:
            try:
                report["failure_resources"] = checkpoint("failure_terminal")
            except BaseException as observation_error:
                report["failure_observation_error"] = (
                    f"{type(observation_error).__name__}: {observation_error}")
        report.update({
            "terminal_status": "failed", "failure_type": type(error).__name__,
            "failure": str(error), "elapsed_seconds": time.monotonic() - started,
        })
        v1.write_progress(progress, "terminal_failure",
                          failure_type=type(error).__name__, failure=str(error))
        raise
    finally:
        model = tokenizer = optimizer = value_and_grad = loss_fn = None
        gc.collect()
        if backend_mx is not None:
            try:
                backend_mx.synchronize()
            except Exception as error:
                report["terminal_cleanup_error"] = f"{type(error).__name__}: {error}"
        if "source_hashes" in report and "source_hashes_after" not in report:
            try:
                after, versions_after = capture_provenance()
                report["source_hashes_after"] = after
                report["versions_after"] = versions_after
                report["provenance_unchanged"] = (
                    report["source_hashes"] == after
                    and report["versions"] == versions_after)
            except Exception as provenance_error:
                report["after_provenance_failure"] = str(provenance_error)
        report_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run(args.output)
    print(json.dumps({
        "completed": report["completed"],
        "resource_feasible": report["resource_feasible"],
        "output": str(args.output), "runtime_estimate": report["runtime_estimate"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
