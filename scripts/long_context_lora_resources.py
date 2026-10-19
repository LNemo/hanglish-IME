"""Shared fail-closed MLX resource observer for long-context LoRA runs."""
from __future__ import annotations

import math
import time


REQUIRED_LIMITS = {
    "rss_bytes", "mlx_peak_bytes", "cache_bytes", "whole_seconds",
    "minimum_system_free_percent", "maximum_swap_growth_bytes",
}


class ResourceObservationError(RuntimeError):
    """Resource failure carrying every observation available at the failure."""

    def __init__(self, message: str, record: dict):
        super().__init__(message)
        self.record = record


def _finite_number(value, name: str, errors: list[str]):
    if type(value) not in (int, float) or not math.isfinite(value):
        errors.append(f"{name} unavailable")
        return None
    return value


def _read_observation(*, mx, started, baseline, system_reader, rss_reader,
                      monotonic) -> tuple[dict, list[str]]:
    observed, errors = {}, []
    readers = {
        "rss_peak_bytes": rss_reader,
        "mlx_active_bytes": mx.get_active_memory,
        "mlx_cache_bytes": mx.get_cache_memory,
        "mlx_peak_bytes": mx.get_peak_memory,
    }
    for name, reader in readers.items():
        try:
            value = _finite_number(reader(), name, errors)
            if value is not None:
                observed[name] = value
        except Exception as error:
            errors.append(f"{name} unavailable: {type(error).__name__}: {error}")
    try:
        system = system_reader()
        if not isinstance(system, dict):
            raise TypeError("system observation is not an object")
        for name in ("free_percent", "swap_used_bytes"):
            value = _finite_number(system.get(name), name, errors)
            if value is not None:
                observed[name] = value
    except Exception as error:
        errors.append(f"system metrics unavailable: {type(error).__name__}: {error}")
    baseline_swap = _finite_number(
        baseline.get("swap_used_bytes") if isinstance(baseline, dict) else None,
        "baseline swap_used_bytes", errors)
    if baseline_swap is not None and "swap_used_bytes" in observed:
        observed["swap_growth_bytes"] = observed["swap_used_bytes"] - baseline_swap
    value = _finite_number(monotonic() - started, "elapsed_seconds", errors)
    if value is not None:
        observed["elapsed_seconds"] = value
    return observed, errors


def _noncache_breaches(observed: dict, limits: dict) -> list[str]:
    checks = (
        ("elapsed_seconds", lambda value: value > limits["whole_seconds"],
         "whole resource time bound exceeded"),
        ("rss_peak_bytes", lambda value: value > limits["rss_bytes"],
         "RSS bound exceeded"),
        ("mlx_peak_bytes", lambda value: value > limits["mlx_peak_bytes"],
         "MLX peak-memory bound exceeded"),
        ("free_percent", lambda value: value < limits["minimum_system_free_percent"],
         "system free-memory bound exceeded"),
        ("swap_growth_bytes", lambda value: value > limits["maximum_swap_growth_bytes"],
         "swap-growth bound exceeded"),
    )
    return [message for name, breached, message in checks if breached(observed[name])]


def observe_resources(*, stage: str, mx, started: float, baseline: dict,
                      limits: dict, system_reader, rss_reader,
                      monotonic=time.monotonic) -> tuple[dict, dict]:
    """Synchronize, observe, conditionally reclaim unused cache, and enforce caps.

    Returns ``(settled_observation, full_record)``. Failures raise
    :class:`ResourceObservationError` with the same full record attached.
    """
    if not isinstance(stage, str) or not stage:
        raise ValueError("resource observation stage is required")
    if not isinstance(limits, dict) or not REQUIRED_LIMITS <= set(limits):
        raise ValueError("resource limits are incomplete")
    checkpoint_started = monotonic()
    record = {"stage": stage, "action": "none", "pre": None, "post": None,
              "errors": [], "cleanup_seconds": 0.0, "checkpoint_seconds": None}

    def fail(message: str):
        record["checkpoint_seconds"] = monotonic() - checkpoint_started
        raise ResourceObservationError(message, record)

    try:
        mx.synchronize()
    except Exception as error:
        record["errors"].append(f"pre-observation synchronize failed: {type(error).__name__}: {error}")
        fail("resource synchronization failed")
    pre, errors = _read_observation(
        mx=mx, started=started, baseline=baseline, system_reader=system_reader,
        rss_reader=rss_reader, monotonic=monotonic)
    record["pre"], record["errors"] = pre, errors
    if errors:
        fail("required resource metrics unavailable")
    breaches = _noncache_breaches(pre, limits)
    if breaches:
        record["errors"].extend(breaches)
        fail(breaches[0])

    settled = pre
    if pre["mlx_cache_bytes"] > limits["cache_bytes"]:
        record["action"] = "clear_unused_mlx_cache"
        cleanup_started = monotonic()
        try:
            mx.clear_cache()
            mx.synchronize()
        except Exception as error:
            record["errors"].append(
                f"unused MLX cache reclamation failed: {type(error).__name__}: {error}")
            record["cleanup_seconds"] = monotonic() - cleanup_started
            fail("unused MLX cache reclamation failed")
        post, errors = _read_observation(
            mx=mx, started=started, baseline=baseline, system_reader=system_reader,
            rss_reader=rss_reader, monotonic=monotonic)
        record["post"], settled = post, post
        record["cleanup_seconds"] = monotonic() - cleanup_started
        record["errors"].extend(errors)
        if errors:
            fail("required post-reclamation resource metrics unavailable")
        breaches = _noncache_breaches(post, limits)
        if breaches:
            record["errors"].extend(breaches)
            fail(breaches[0])
        if post["mlx_cache_bytes"] > limits["cache_bytes"]:
            record["errors"].append("MLX residual cache bound exceeded")
            fail("MLX residual cache bound exceeded")

    record["checkpoint_seconds"] = monotonic() - checkpoint_started
    return settled, record


__all__ = ["ResourceObservationError", "observe_resources"]
