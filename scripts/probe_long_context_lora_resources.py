"""Run the approved synthetic-only long-context LoRA resource gate."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.long_context_lora_model import (
    LORA_CONFIG, LORA_LAYERS, SEED, add_materialized_gradients,
    apply_materialized_update, average_and_clip_gradients, configure_fresh_lora,
    create_optimizer, materialize_microbatch, mlx_log_mass_loss,
    ordinary_forward_choice_scores, reference_log_mass_loss, require_matching_gradient_tree,
    require_trainable_contract, save_adapter, score_choices, tensor_hashes,
)


DEFAULT_OUTPUT = ROOT / ".omx/artifacts/long-context-lora-resource-v1"
MODEL_DIR = ROOT / "data/external/qwen3-4b-mlx-4bit"
MODEL_SOURCE = MODEL_DIR / "SOURCE.json"
PLAN = ROOT / ".omx/plans/long-context-lora-v1.md"
TEST_SPEC = ROOT / ".omx/plans/test-spec-long-context-lora-v1.md"
HANDOFF = ROOT / ".omx/plans/long-context-lora-handoff-v1.json"
REFERENCES = ROOT / ".omx/context/long-context-lora-references-v1.json"
TOKENIZER_LINEAGE = ROOT / ".omx/context/long-context-lora-tokenizer-lineage-v1.json"
MODEL_REVISION = "52a5ab34fa604bc8af6d3ce0cac0cab10b7eb495"
MODEL_WEIGHT_SHA256 = "e56a94b846a2dce7d1cc154ce58b9cfc4b8d99e8ffb74bfe256a265f4cbc7fb9"
PACKAGE_VERSIONS = {"mlx": "0.31.2", "mlx-lm": "0.31.3", "transformers": "5.12.1"}
SHAPES = ((256, 2, (0,)), (512, 3, (0, 2)),
          (768, 4, (1, 3)), (1024, 5, (0, 2, 4)))
LIMITS = {
    "rss_bytes": 6 * 1024**3,
    "mlx_peak_bytes": 6 * 1024**3,
    "cache_bytes": 256 * 1024**2,
    "whole_seconds": 900.0,
    "update_seconds": 120.0,
    "minimum_system_free_percent": 10.0,
    "maximum_swap_growth_bytes": 512 * 1024**2,
}
UPDATES = 20
ACCUMULATION = 4
INFERENCE_REPETITIONS = 10
FULL_FORWARD_TOLERANCE = 0.125
RELOAD_ATOL = 1e-5
RELOAD_RTOL = 1e-5
PINNED_ARTIFACT_SHA256 = {
    PLAN: "cfe51fa6ccf6f408cc763a654170ae01b9cb81bfb7fa01f948f12ebfdb39449e",
    TEST_SPEC: "4799276085e525dc66a94dce7610f0edfbb168eb4c1b2ce28aa9f1d5ece90189",
    HANDOFF: "e1d3644a51514b7bcda899b6f27820de60b660f9520016143a01c1e26a99872d",
    REFERENCES: "9b3f25c4c765d716f87b4778c4d63c9c47178251a8cd06ea063bc79ec7033d57",
    TOKENIZER_LINEAGE: "0bae533c74ec1ffcf2afa674df0becb1c95119049202c732f44daa525c8cbd5d",
    MODEL_SOURCE: "67928c796b7e4562c072b2bbbdd0d52569e5ab9118ade1302f8d8d1bb4bae819",
}
OFFICIAL_TEMPLATE_SHA256 = "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def parse_memory_pressure(text: str) -> float:
    match = re.search(r"System-wide memory free percentage:\s*([0-9]+(?:\.[0-9]+)?)%", text)
    if match is None:
        raise ValueError("system free-memory percentage unavailable")
    value = float(match.group(1))
    if not 0 <= value <= 100:
        raise ValueError("system free-memory percentage is invalid")
    return value


def parse_swap_usage(text: str) -> int:
    match = re.search(r"\bused\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGT])", text, re.I)
    if match is None:
        raise ValueError("system swap usage unavailable")
    scale = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}[match.group(2).upper()]
    return round(float(match.group(1)) * scale)


def system_resources(run=subprocess.run) -> dict:
    pressure = run(["memory_pressure", "-Q"], check=True, capture_output=True,
                   text=True, timeout=10)
    swap = run(["sysctl", "vm.swapusage"], check=True, capture_output=True,
               text=True, timeout=10)
    return {"free_percent": parse_memory_pressure(pressure.stdout),
            "swap_used_bytes": parse_swap_usage(swap.stdout)}


def p95(values) -> float:
    numbers = tuple(float(value) for value in values)
    if not numbers or not all(math.isfinite(value) and value >= 0 for value in numbers):
        raise ValueError("p95 requires nonempty finite nonnegative timings")
    ordered = sorted(numbers)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def runtime_estimate(update_times, scorer_times) -> dict:
    if len(update_times) != 20 or len(scorer_times) != 40:
        raise ValueError("runtime estimate requires exactly 20 update and 40 scorer timings")
    update_p95, scorer_p95 = p95(update_times), p95(scorer_times)
    estimate = 1.5 * (693 * update_p95 + 1280 * scorer_p95) + 300
    return {"update_p95_seconds": update_p95, "probe_scoring_p95_seconds": scorer_p95,
            "formula": "1.5*(693*update_p95+1280*probe_scoring_p95)+300",
            "estimated_training_seconds": estimate,
            "within_21600_seconds": estimate <= 21600}


def capture_provenance() -> tuple[dict, dict]:
    versions = {name: importlib.metadata.version(name) for name in PACKAGE_VERSIONS}
    if versions != PACKAGE_VERSIONS:
        raise RuntimeError(f"package versions differ: {versions}")
    expected_hashes = dict(PINNED_ARTIFACT_SHA256)
    source = json.loads(MODEL_SOURCE.read_text())
    if (source.get("revision") != MODEL_REVISION
            or source.get("files", {}).get("model.safetensors") != MODEL_WEIGHT_SHA256):
        raise RuntimeError("pinned base identity differs")
    handoff = json.loads(HANDOFF.read_text())
    if (handoff.get("status") != "complete"
            or handoff.get("ralplan_execution_handoff", {}).get("authorized") is not True
            or handoff.get("planning_artifacts", {}).get(str(PLAN.relative_to(ROOT)))
            != PINNED_ARTIFACT_SHA256[PLAN]
            or handoff.get("planning_artifacts", {}).get(str(TEST_SPEC.relative_to(ROOT)))
            != PINNED_ARTIFACT_SHA256[TEST_SPEC]):
        raise RuntimeError("approved sequential handoff differs")
    dynamic = [Path(__file__), ROOT / "scripts/long_context_lora_model.py"]
    for name, digest in source["files"].items():
        path = MODEL_DIR / name
        if path.name != name:
            raise RuntimeError(f"pinned base contains a nonlocal path: {name}")
        if path in expected_hashes and expected_hashes[path] != digest:
            raise RuntimeError(f"pinned base hash conflict: {name}")
        expected_hashes[path] = digest
    references = json.loads(REFERENCES.read_text())
    if references.get("versions") != PACKAGE_VERSIONS:
        raise RuntimeError("reference package pins differ")
    lineage = json.loads(TOKENIZER_LINEAGE.read_text())
    if lineage.get("versions") != PACKAGE_VERSIONS:
        raise RuntimeError("tokenizer lineage package pins differ")
    for collection in (references.get("source_hashes", {}), lineage.get("source_hashes", {})):
        for raw_path, digest in collection.items():
            path = Path(raw_path)
            if not path.is_absolute():
                path = ROOT / path
            if path in expected_hashes and expected_hashes[path] != digest:
                raise RuntimeError(f"pinned source hash conflict: {raw_path}")
            expected_hashes[path] = digest
    observed = {}
    for path, digest in expected_hashes.items():
        actual = sha256_file(path)
        if actual != digest:
            raise RuntimeError(f"pinned source differs: {relative_path(path)}")
        observed[relative_path(path)] = actual
    for path in dynamic:
        observed[relative_path(path)] = sha256_file(path)
    return dict(sorted(observed.items())), versions


def check_resources(started: float, mx, baseline: dict) -> dict:
    current = system_resources()
    observed = {
        "elapsed_seconds": time.monotonic() - started,
        "rss_peak_bytes": rss_bytes(),
        "mlx_peak_bytes": int(mx.get_peak_memory()),
        "mlx_cache_bytes": int(mx.get_cache_memory()),
        **current,
        "swap_growth_bytes": current["swap_used_bytes"] - baseline["swap_used_bytes"],
    }
    if observed["elapsed_seconds"] > LIMITS["whole_seconds"]:
        raise RuntimeError("whole resource probe time bound exceeded")
    if observed["rss_peak_bytes"] > LIMITS["rss_bytes"]:
        raise RuntimeError("RSS bound exceeded")
    if observed["mlx_peak_bytes"] > LIMITS["mlx_peak_bytes"]:
        raise RuntimeError("MLX peak-memory bound exceeded")
    if observed["mlx_cache_bytes"] > LIMITS["cache_bytes"]:
        raise RuntimeError("MLX cache bound exceeded")
    if observed["free_percent"] < LIMITS["minimum_system_free_percent"]:
        raise RuntimeError("system free-memory bound exceeded")
    if observed["swap_growth_bytes"] > LIMITS["maximum_swap_growth_bytes"]:
        raise RuntimeError("swap-growth bound exceeded")
    return observed


def write_progress(path: Path, stage: str, **values) -> None:
    record = {"at": datetime.now(timezone.utc).isoformat(), "stage": stage, **values}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()


def exact_prompts(tokenizer) -> tuple[list[dict], list[int]]:
    if hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() != OFFICIAL_TEMPLATE_SHA256:
        raise RuntimeError("official tokenizer chat template differs")
    messages = [{"role": "system", "content": "Select the best numbered neutral option."},
                {"role": "user", "content": "Neutral synthetic resource probe."}]
    body = tokenizer.apply_chat_template(messages, add_generation_prompt=False,
                                         enable_thinking=False, return_dict=False)
    complete = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                             enable_thinking=False, return_dict=False)
    if not isinstance(body, list) or not isinstance(complete, list) or complete[:len(body)] != body:
        raise RuntimeError("official false-thinking generation prefix is not a stable suffix")
    generation_prefix = complete[len(body):]
    filler = tokenizer.encode(" neutral", add_special_tokens=False)
    if len(filler) != 1 or not generation_prefix:
        raise RuntimeError("deterministic one-token filler or generation prefix is unavailable")
    digit_ids = []
    for digit in range(1, 6):
        encoded = tokenizer.encode(str(digit), add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(f"choice digit {digit} is not a single continuation token")
        digit_ids.append(encoded[0])
    if len(set(digit_ids)) != 5:
        raise RuntimeError("choice digit token IDs are not distinct")
    rows = []
    for target, count, gold in SHAPES:
        needed = target - len(body) - len(generation_prefix)
        if needed < 0:
            raise RuntimeError("neutral template exceeds a fixed prompt shape")
        tokens = [*body, *([filler[0]] * needed), *generation_prefix]
        if len(tokens) != target:
            raise AssertionError("exact synthetic prompt construction failed")
        if any(digit_id in tokens for digit_id in digit_ids):
            raise RuntimeError("synthetic prompt unexpectedly contains an answer digit token")
        rendered = tokenizer.decode(tokens)
        for digit, digit_id in enumerate(digit_ids[:count], 1):
            boundary = tokenizer.encode(rendered + str(digit), add_special_tokens=False)
            if boundary != [*tokens, digit_id]:
                raise RuntimeError(f"choice digit {digit} is not one token at prompt boundary")
        rows.append({"prompt_tokens": tokens, "prompt_token_count": target,
                     "prompt_sha256": hashlib.sha256(
                         json.dumps(tokens, separators=(",", ":")).encode()).hexdigest(),
                     "candidate_count": count, "allowed_gold_positions": list(gold),
                     "synthetic_labels_only": True})
    return tuple(rows), digit_ids


def validate_tiny_math(mx) -> dict:
    fixtures = [([0.0, math.log(2.0), math.log(3.0)], (2,)),
                ([0.0, math.log(2.0), math.log(3.0)], (0, 2)),
                ([0.0, math.log(2.0), math.log(3.0)], (0, 1, 2))]
    losses = []
    for scores, gold in fixtures:
        actual = mlx_log_mass_loss(mx, mx.array(scores), gold)
        mx.eval(actual)
        expected = reference_log_mass_loss(scores, gold)
        if not math.isclose(float(actual.item()), expected, abs_tol=1e-6, rel_tol=1e-6):
            raise RuntimeError("MLX finite-choice loss differs from high-precision fixture")
        losses.append({"scores": scores, "gold": list(gold),
                       "actual": float(actual.item()), "expected": expected})
    gradient_fixtures = {}
    for count in (4, 3):
        accumulated = {"known": mx.array([3.0 * count, 4.0 * count])}
        clipped, norm = average_and_clip_gradients(mx, accumulated, count)
        mx.eval(clipped, norm)
        actual = clipped["known"].tolist()
        if not math.isclose(float(norm.item()), 5.0, abs_tol=1e-6) or any(
                not math.isclose(float(value), expected, abs_tol=1e-6)
                for value, expected in zip(actual, (0.6, 0.8))):
            raise RuntimeError("actual-count averaging/global clipping fixture failed")
        gradient_fixtures[str(count)] = {"norm": float(norm.item()), "clipped": actual}
    overflow_rejected = False
    try:
        average_and_clip_gradients(
            mx, {"overflow": mx.array([3e38, 3e38], dtype=mx.float32)}, 1)
    except FloatingPointError:
        overflow_rejected = True
    if not overflow_rejected:
        raise RuntimeError("nonfinite float32 gradient norm was not rejected")
    return {"losses": losses, "gradient_fixtures": gradient_fixtures,
            "nonfinite_norm_rejected": True}


def max_abs_delta(left, right) -> float:
    return max(abs(float(a) - float(b)) for a, b in zip(left, right))


def run(output: Path = DEFAULT_OUTPUT) -> dict:
    if output.exists():
        raise FileExistsError(f"resource artifact already exists: {output}")
    output.mkdir(parents=True)
    progress = output / "progress.jsonl"
    report_path = output / "report.json"
    started = time.monotonic()
    report = {"schema": "long-context-lora-resource-v1", "completed": False,
              "resource_feasible": False, "synthetic_only": True,
              "real_fit_rows_read": False, "real_training_run": False,
              "old_adapter_loaded": False, "limits": LIMITS}
    model = tokenizer = backend_mx = None
    try:
        before, versions = capture_provenance()
        baseline = system_resources()
        report.update({"source_hashes": before, "versions": versions,
                       "system_baseline": baseline})
        write_progress(progress, "provenance_verified", files=len(before))

        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm.tuner.utils import load_adapters
        from mlx_lm.utils import load
        backend_mx = mx
        mx.set_memory_limit(LIMITS["mlx_peak_bytes"])
        mx.set_cache_limit(LIMITS["cache_bytes"])
        tiny_math = validate_tiny_math(mx)
        check_resources(started, mx, baseline)
        model, tokenizer = load(str(MODEL_DIR),
                                tokenizer_config={"trust_remote_code": False})
        load_resources = check_resources(started, mx, baseline)
        if not bool(model.args.tie_word_embeddings):
            raise RuntimeError("pinned base unexpectedly has an untied output head")
        model.eval()
        rows, digit_ids = exact_prompts(tokenizer)
        initial = configure_fresh_lora(mx, model)
        initial_hashes = tensor_hashes(initial)
        del initial
        optimizer = create_optimizer()
        model.train()

        def loss_fn(active_model, row):
            scores = score_choices(mx, active_model, row["prompt_tokens"],
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
        check_resources(started, mx, baseline)
        write_progress(progress, "warm_backward", loss=float(warm_loss.item()))

        update_times, update_records = [], []
        for update_index in range(UPDATES):
            update_started = time.monotonic()
            def require_update_time():
                if time.monotonic() - update_started > LIMITS["update_seconds"]:
                    raise RuntimeError("optimizer update time bound exceeded")
            accumulated = None
            losses = []
            for row in rows:
                loss, gradients = value_and_grad(model, row)
                materialize_microbatch(mx, loss, gradients)
                require_update_time()
                gradient_flat = require_matching_gradient_tree(model, gradients)
                if not bool(mx.isfinite(loss).item()) or any(
                        not bool(mx.all(mx.isfinite(value)).item())
                        for value in gradient_flat.values()):
                    raise RuntimeError("nonfinite loss or gradient")
                accumulated = add_materialized_gradients(mx, accumulated, gradients)
                require_update_time()
                check_resources(started, mx, baseline)
                losses.append(float(loss.item()))
            clipped, norm = average_and_clip_gradients(mx, accumulated, ACCUMULATION)
            require_update_time()
            check_resources(started, mx, baseline)
            apply_materialized_update(mx, model, optimizer, clipped)
            require_update_time()
            elapsed = time.monotonic() - update_started
            if elapsed > LIMITS["update_seconds"]:
                raise RuntimeError("optimizer update time bound exceeded")
            resources = check_resources(started, mx, baseline)
            update_times.append(elapsed)
            record = {"update": update_index + 1, "microbatches": ACCUMULATION,
                      "microbatch_losses": losses,
                      "mean_loss": sum(losses) / len(losses),
                      "gradient_norm_before_clip": float(norm.item()),
                      "seconds": elapsed, "resources": resources}
            update_records.append(record)
            write_progress(progress, "optimizer_update", **record)

        final = require_trainable_contract(model)
        final_hashes = tensor_hashes(final)
        if initial_hashes == final_hashes:
            raise RuntimeError("adapter did not change")

        model.eval()
        equivalence, trained_logits = [], []
        for row in rows:
            selected = score_choices(mx, model, row["prompt_tokens"],
                                     digit_ids[:row["candidate_count"]])
            ordinary = ordinary_forward_choice_scores(
                mx, model, row["prompt_tokens"], digit_ids[:row["candidate_count"]])
            mx.eval(selected, ordinary)
            check_resources(started, mx, baseline)
            left, right = selected.tolist(), ordinary.tolist()
            if not all(math.isfinite(float(value)) for value in [*left, *right]):
                raise RuntimeError("diagnostic logits are nonfinite")
            delta = max_abs_delta(left, right)
            if delta > FULL_FORWARD_TOLERANCE or left.index(max(left)) != right.index(max(right)):
                raise RuntimeError("final-hidden scorer differs from ordinary forward")
            equivalence.append({"prompt_tokens": row["prompt_token_count"],
                                "candidate_count": row["candidate_count"],
                                "prompt_token_ids": row["prompt_tokens"],
                                "prompt_sha256": row["prompt_sha256"],
                                "selected_scorer_logits": left,
                                "ordinary_forward_logits": right,
                                "max_absolute_digit_delta": delta,
                                "argmax": left.index(max(left))})
            trained_logits.append(left)
            del ordinary
            mx.clear_cache()
            check_resources(started, mx, baseline)

        scorer_times = []
        for row in rows:
            for _ in range(INFERENCE_REPETITIONS):
                timing_started = time.monotonic()
                scores = score_choices(mx, model, row["prompt_tokens"],
                                       digit_ids[:row["candidate_count"]])
                mx.eval(scores)
                scorer_times.append(time.monotonic() - timing_started)
                check_resources(started, mx, baseline)

        adapter_dir = output / "adapter"
        adapter_dir.mkdir()
        adapter_path = adapter_dir / "adapters.safetensors"
        saved_hashes = save_adapter(mx, model, adapter_path)
        check_resources(started, mx, baseline)
        adapter_config = {"fine_tune_type": "lora", "num_layers": LORA_LAYERS,
                          "lora_parameters": LORA_CONFIG}
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps(adapter_config, sort_keys=True, indent=2) + "\n")
        adapter_sha = sha256_file(adapter_path)
        del model, tokenizer, optimizer, value_and_grad, loss_fn, warm_grads, warm_flat
        del accumulated, clipped, final, gradients, gradient_flat, loss, scores
        model = tokenizer = None
        gc.collect()
        mx.synchronize()
        mx.clear_cache()

        model, tokenizer = load(str(MODEL_DIR),
                                tokenizer_config={"trust_remote_code": False})
        model.freeze()
        load_adapters(model, str(adapter_dir))
        require_trainable_contract(model)
        model.eval()
        fresh_rows, fresh_digit_ids = exact_prompts(tokenizer)
        if fresh_rows != rows or fresh_digit_ids != digit_ids:
            raise RuntimeError("fresh tokenizer synthetic prompt identity differs")
        reload_checks = []
        for row, expected in zip(rows, trained_logits):
            actual_array = score_choices(mx, model, row["prompt_tokens"],
                                         digit_ids[:row["candidate_count"]])
            mx.eval(actual_array)
            check_resources(started, mx, baseline)
            actual = actual_array.tolist()
            close = all(abs(float(a) - float(b)) <= RELOAD_ATOL + RELOAD_RTOL * abs(float(b))
                        for a, b in zip(actual, expected))
            if not close or actual.index(max(actual)) != expected.index(max(expected)):
                raise RuntimeError("fresh base plus adapter scorer parity failed")
            reload_checks.append({"prompt_tokens": row["prompt_token_count"],
                                  "expected_logits": expected, "actual_logits": actual,
                                  "max_absolute_delta": max_abs_delta(actual, expected),
                                  "prediction": actual.index(max(actual))})

        estimate = runtime_estimate(update_times, scorer_times)
        after, versions_after = capture_provenance()
        if before != after or versions != versions_after:
            raise RuntimeError("pinned sources or package versions changed during probe")
        terminal = check_resources(started, mx, baseline)
        report.update({
            "completed": True, "resource_feasible": True,
            "terminal_status": "success", "source_hashes_after": after,
            "versions_after": versions_after, "seed": SEED,
            "tiny_math_validation": tiny_math, "load_resources": load_resources,
            "shapes": [{k: value for k, value in row.items() if k != "prompt_tokens"}
                       for row in rows], "digit_token_ids": digit_ids,
            "warm_1024_backward": {"completed": True, "update_applied": False,
                                   "finite_gradients": True, "nonzero_gradient": warm_nonzero},
            "updates": update_records, "update_timings_seconds": update_times,
            "scorer_timings_seconds": scorer_times, "runtime_estimate": estimate,
            "ordinary_forward_diagnostic": equivalence,
            "initial_trainable_tensor_hashes": initial_hashes,
            "final_trainable_tensor_hashes": final_hashes,
            "saved_trainable_tensor_hashes": saved_hashes,
            "adapter": {"sha256": adapter_sha,
                        "config_sha256": sha256_file(adapter_dir / "adapter_config.json"),
                        "only_16_trainable_tensors": True},
            "fresh_reload": {"atol": RELOAD_ATOL, "rtol": RELOAD_RTOL,
                             "checks": reload_checks},
            "real_training_launch_allowed_by_runtime_estimate":
                estimate["within_21600_seconds"],
            "system_terminal": terminal,
            "elapsed_seconds": time.monotonic() - started,
        })
        write_progress(progress, "terminal_success", elapsed_seconds=report["elapsed_seconds"])
        return report
    except Exception as error:
        report.update({"terminal_status": "failed", "failure_type": type(error).__name__,
                       "failure": str(error), "elapsed_seconds": time.monotonic() - started})
        write_progress(progress, "terminal_failure", failure_type=type(error).__name__,
                       failure=str(error))
        raise
    finally:
        model = tokenizer = optimizer = value_and_grad = loss_fn = None
        gc.collect()
        if backend_mx is not None:
            try:
                backend_mx.synchronize()
                backend_mx.clear_cache()
            except Exception:
                pass
        if "source_hashes" in report and "source_hashes_after" not in report:
            try:
                after, versions_after = capture_provenance()
                report["source_hashes_after"] = after
                report["versions_after"] = versions_after
                report["provenance_unchanged"] = (
                    report["source_hashes"] == after and report["versions"] == versions_after)
            except Exception as provenance_error:
                report["after_provenance_failure"] = str(provenance_error)
        report_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run(args.output)
    print(json.dumps({"completed": report["completed"],
                      "resource_feasible": report["resource_feasible"],
                      "output": str(args.output),
                      "runtime_estimate": report["runtime_estimate"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
