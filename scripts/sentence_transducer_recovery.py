"""Atomic recovery checkpoints for sentence-transducer training."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import re

import torch


FORMAT = "sentence-transducer-recovery-v1"
REQUIRED_STATE = {
    "updates", "next_epoch", "next_batch", "epochs_report",
    "epoch_loss_sum", "epoch_token_sum", "training_elapsed_s", "peak_rss_bytes",
}
OPTIONAL_STATE = {"discarded_tail_updates", "discarded_tail_archive"}


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _finite_number(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value))


def _validate_tensors(value, label):
    if torch.is_tensor(value):
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
            raise ValueError(f"{label} contains a nonfinite tensor")
        return
    if isinstance(value, dict):
        for item in value.values():
            _validate_tensors(item, label)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _validate_tensors(item, label)


def _validate_state(state):
    _need(isinstance(state, dict) and set(state) >= REQUIRED_STATE
          and set(state) <= REQUIRED_STATE | OPTIONAL_STATE,
          "recovery state keys differ")
    for key in ("updates", "next_epoch", "next_batch", "epoch_token_sum", "peak_rss_bytes"):
        value = state[key]
        _need(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
              f"recovery state {key} is invalid")
    for key in ("epoch_loss_sum", "training_elapsed_s"):
        _need(_finite_number(state[key]) and state[key] >= 0,
              f"recovery state {key} is invalid")
    reports = state["epochs_report"]
    _need(isinstance(reports, list), "recovery epoch reports are invalid")
    for report in reports:
        _need(isinstance(report, dict), "recovery epoch report is invalid")
        if "epoch" in report:
            _need(isinstance(report["epoch"], int) and not isinstance(report["epoch"], bool)
                  and report["epoch"] >= 1, "recovery epoch identity is invalid")
        for key in ("target_loss", "tokens"):
            if key in report:
                _need(_finite_number(report[key]) and report[key] >= 0,
                      "recovery epoch metric is invalid")
    if "discarded_tail_updates" in state:
        value = state["discarded_tail_updates"]
        _need(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
              "discarded tail update count is invalid")
    if "discarded_tail_archive" in state:
        value = state["discarded_tail_archive"]
        _need(value is None or isinstance(value, str),
              "discarded tail archive is invalid")


def _sha_prefix(path, size):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        remaining = size
        while remaining:
            chunk = stream.read(min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError("update log is shorter than committed prefix")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_checkpoint(output, model, optimizer, identity: dict, state: dict, stream):
    """Atomically commit model/optimizer/RNG state after an fsynced log prefix."""
    output = Path(output)
    _need(output.is_dir(), "recovery output directory is missing")
    _need(isinstance(identity, dict), "recovery identity must be a dictionary")
    _validate_state(state)
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    _validate_tensors(model_state, "model state")
    _validate_tensors(optimizer_state, "optimizer state")
    log_path = Path(stream.name).resolve()
    _need(log_path.parent == output.resolve(), "update log must belong to recovery output")
    stream.flush()
    os.fsync(stream.fileno())
    prefix_bytes = log_path.stat().st_size
    prefix_sha256 = _sha_prefix(log_path, prefix_bytes)
    payload = {
        "format": FORMAT, "identity": identity, "state": dict(state),
        "model_state": model_state, "optimizer_state": optimizer_state,
        "torch_rng_state": torch.get_rng_state(), "python_rng_state": random.getstate(),
        "log": {"name": log_path.name, "prefix_bytes": prefix_bytes,
                "prefix_sha256": prefix_sha256},
    }
    destination = output / "recovery.pt"
    temporary = output / ".recovery.pt.tmp"
    try:
        with temporary.open("wb") as checkpoint_stream:
            torch.save(payload, checkpoint_stream)
            checkpoint_stream.flush()
            os.fsync(checkpoint_stream.fileno())
        os.replace(temporary, destination)
        _fsync_directory(output)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return {"path": destination.name, "updates": state["updates"],
            "log_prefix_bytes": prefix_bytes, "log_prefix_sha256": prefix_sha256,
            "checkpoint_sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}


def _load_payload(path, identity):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("recovery checkpoint cannot be safely loaded") from error
    _need(isinstance(payload, dict) and payload.get("format") == FORMAT,
          "recovery checkpoint format differs")
    _need(payload.get("identity") == identity, "recovery identity differs")
    _validate_state(payload.get("state"))
    _validate_tensors(payload.get("model_state"), "model state")
    _validate_tensors(payload.get("optimizer_state"), "optimizer state")
    rng = payload.get("torch_rng_state")
    _need(torch.is_tensor(rng) and rng.dtype == torch.uint8 and rng.ndim == 1,
          "torch RNG state is invalid")
    _need(isinstance(payload.get("python_rng_state"), tuple), "Python RNG state is invalid")
    log = payload.get("log")
    _need(isinstance(log, dict) and isinstance(log.get("name"), str)
          and Path(log["name"]).name == log["name"], "recovery log identity is invalid")
    _need(isinstance(log.get("prefix_bytes"), int) and not isinstance(log["prefix_bytes"], bool)
          and log["prefix_bytes"] >= 0, "recovery log prefix size is invalid")
    _need(re.fullmatch(r"[0-9a-f]{64}", log.get("prefix_sha256", "")) is not None,
          "recovery log prefix hash is invalid")
    return payload


def _archive_tail(output, log_path, offset, tail):
    base = f"{log_path.stem}.uncommitted-{offset}-{offset + len(tail)}{log_path.suffix}"
    archive = output / base
    counter = 1
    while archive.exists():
        archive = output / f"{base}.{counter}"
        counter += 1
    with archive.open("xb") as stream:
        stream.write(tail)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(output)
    return archive


def _tail_resources(tail):
    updates = 0
    elapsed = 0.0
    peak = 0
    for raw in tail.splitlines():
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(row, dict):
            continue
        updates += 1
        value = row.get("training_elapsed_s")
        if _finite_number(value) and value >= 0:
            elapsed = max(elapsed, float(value))
        value = row.get("peak_rss_bytes")
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            peak = max(peak, value)
    return updates, elapsed, peak


def restore_checkpoint(output, model, optimizer, identity):
    """Restore the committed checkpoint and roll back an uncommitted log tail."""
    output = Path(output)
    _need(isinstance(identity, dict), "recovery identity must be a dictionary")
    path = output / "recovery.pt"
    _need(path.is_file(), "recovery checkpoint is missing")
    payload = _load_payload(path, identity)
    log = payload["log"]
    log_path = output / log["name"]
    _need(log_path.is_file(), "recovery update log is missing")
    size = log_path.stat().st_size
    offset = log["prefix_bytes"]
    _need(size >= offset, "update log is shorter than committed prefix")
    _need(_sha_prefix(log_path, offset) == log["prefix_sha256"],
          "committed update log prefix differs")
    with log_path.open("rb") as stream:
        stream.seek(offset)
        tail = stream.read()
    discarded, tail_elapsed, tail_peak = _tail_resources(tail)
    # Validate compatibility before changing the only live copy of the update
    # log. RNG is restored last so loading cannot consume the committed stream.
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    archive = None
    if tail:
        archive = _archive_tail(output, log_path, offset, tail)
        with log_path.open("r+b") as stream:
            stream.truncate(offset)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(output)
    torch.set_rng_state(payload["torch_rng_state"])
    random.setstate(payload["python_rng_state"])
    state = dict(payload["state"])
    state["training_elapsed_s"] = max(float(state["training_elapsed_s"]), tail_elapsed)
    state["peak_rss_bytes"] = max(state["peak_rss_bytes"], tail_peak)
    state["discarded_tail_updates"] = state.get("discarded_tail_updates", 0) + discarded
    if archive is not None:
        state["discarded_tail_archive"] = str(archive)
    else:
        state.setdefault("discarded_tail_archive", None)
    return state


__all__ = ["save_checkpoint", "restore_checkpoint"]
