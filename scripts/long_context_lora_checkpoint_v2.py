"""Atomic, fail-closed checkpoints for the reviewed long-context LoRA v2 path.

MLX is imported lazily.  The module deliberately stores only completed optimizer
updates; partial gradient accumulation is outside this checkpoint contract.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import time
from typing import Any, Callable, Mapping
import uuid


SCHEMA = "long-context-lora-checkpoint-v2"
RUN_SCHEMA = "long-context-lora-run-v2"
MAX_RESUMES = 2
FINAL_UPDATE = 693
GIB = 1024 ** 3
MIB = 1024 ** 2
EXPECTED_LIMITS = {
    "rss_bytes": 6 * GIB,
    "mlx_peak_bytes": 6 * GIB,
    "cache_bytes": 256 * MIB,
    "minimum_system_free_percent": 10.0,
    "maximum_swap_growth_bytes": 512 * MIB,
    "update_seconds": 120.0,
}
ADAM_CONFIG = {
    "learning_rate": 1e-5,
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "bias_correction": True,
}
LORA_CONFIG = {
    "rank": 8,
    "scale": 20.0,
    "dropout": 0.0,
    "layers": 4,
    "keys": ["self_attn.q_proj", "self_attn.v_proj"],
}
ADAPTER_CONFIG = {
    "fine_tune_type": "lora",
    "lora_parameters": {"rank": 8, "scale": 20.0, "dropout": 0.0,
                        "keys": ["self_attn.q_proj", "self_attn.v_proj"]},
    "num_layers": 4,
}


class CheckpointError(RuntimeError):
    pass


class LockUnavailable(CheckpointError):
    pass


class CheckpointConflict(CheckpointError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def envelope(payload: Mapping[str, Any]) -> dict[str, Any]:
    body = copy.deepcopy(dict(payload))
    if "digest" in body:
        raise ValueError("digest cannot occur inside an envelope payload")
    return {"payload": body, "digest": canonical_digest(body)}


def validate_envelope(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"payload", "digest"}:
        raise CheckpointError("invalid envelope")
    payload, digest = value["payload"], value["digest"]
    if not isinstance(payload, dict) or "digest" in payload:
        raise CheckpointError("invalid envelope payload")
    if not isinstance(digest, str) or digest != canonical_digest(payload):
        raise CheckpointError("envelope digest differs")
    return payload


def expected_adapter_shapes() -> dict[str, tuple[int, int]]:
    shapes: dict[str, tuple[int, int]] = {}
    for layer in range(32, 36):
        for projection, output in (("q_proj", 4096), ("v_proj", 1024)):
            prefix = f"model.layers.{layer}.self_attn.{projection}"
            shapes[f"{prefix}.lora_a"] = (2560, 8)
            shapes[f"{prefix}.lora_b"] = (8, output)
    return shapes


def _array_metadata(value: Any) -> tuple[str, tuple[int, ...], int, bytes]:
    import numpy as np

    array = np.asarray(value)
    dtype = str(array.dtype)
    shape = tuple(int(item) for item in array.shape)
    return dtype, shape, int(array.size), array.tobytes(order="C")


def _tensor_inventory(arrays: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, value in sorted(arrays.items()):
        dtype, shape, size, raw = _array_metadata(value)
        domain = f"{dtype}:{shape}:".encode("utf-8")
        result[name] = {
            "dtype": dtype, "shape": list(shape), "size": size,
            "sha256": hashlib.sha256(domain + raw).hexdigest(),
        }
    return result


def _all_finite(value: Any) -> bool:
    import numpy as np
    return bool(np.all(np.isfinite(np.asarray(value))))


def _scalar(value: Any) -> Any:
    import numpy as np
    array = np.asarray(value)
    if array.shape != ():
        raise CheckpointError("optimizer scalar has non-scalar shape")
    return array.item()


def validate_adapter_tensors(arrays: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    shapes = expected_adapter_shapes()
    if set(arrays) != set(shapes):
        raise CheckpointError("adapter tensor names differ")
    inventory = _tensor_inventory(arrays)
    if len(inventory) != 16 or sum(item["size"] for item in inventory.values()) != 327680:
        raise CheckpointError("adapter tensor count or scalar count differs")
    for name, expected_shape in shapes.items():
        item = inventory[name]
        if item["dtype"] != "float32" or tuple(item["shape"]) != expected_shape:
            raise CheckpointError(f"adapter dtype or shape differs: {name}")
        if not _all_finite(arrays[name]):
            raise CheckpointError(f"adapter tensor is nonfinite: {name}")
    return inventory


def validate_optimizer_tensors(flat_state: Mapping[str, Any], update: int,
                               adapter_inventory: Mapping[str, Mapping[str, Any]],
                               config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if dict(config) != ADAM_CONFIG:
        raise CheckpointError("Adam constructor configuration differs")
    expected = {"step", "learning_rate"}
    for name in adapter_inventory:
        expected.update((f"{name}.m", f"{name}.v"))
    if set(flat_state) != expected:
        raise CheckpointError("optimizer state names differ")
    inventory = _tensor_inventory(flat_state)
    if inventory["step"]["dtype"] != "uint64" or inventory["step"]["shape"] != []:
        raise CheckpointError("Adam step dtype or shape differs")
    if int(_scalar(flat_state["step"])) != update:
        raise CheckpointError("Adam step differs from checkpoint update")
    lr_item = inventory["learning_rate"]
    if lr_item["dtype"] != "float32" or lr_item["shape"] != []:
        raise CheckpointError("serialized learning-rate dtype or shape differs")
    import numpy as np
    serialized_lr = np.asarray(flat_state["learning_rate"])
    expected_lr = np.asarray(ADAM_CONFIG["learning_rate"], dtype=np.float32)
    if serialized_lr.tobytes() != expected_lr.tobytes():
        raise CheckpointError("serialized learning rate differs")
    for name, adapter_item in adapter_inventory.items():
        for moment in ("m", "v"):
            key = f"{name}.{moment}"
            item = inventory[key]
            if item["dtype"] != "float32" or item["shape"] != adapter_item["shape"]:
                raise CheckpointError(f"Adam moment dtype or shape differs: {key}")
            if not _all_finite(flat_state[key]):
                raise CheckpointError(f"Adam moment is nonfinite: {key}")
    return inventory


def validate_adam_constructor(optimizer: Any) -> None:
    import numpy as np

    try:
        learning_rate = np.asarray(optimizer.learning_rate)
    except Exception as error:
        raise CheckpointError("Adam constructor learning rate is unavailable") from error
    expected_lr = np.asarray(ADAM_CONFIG["learning_rate"], dtype=np.float32)
    if (learning_rate.dtype != np.dtype("float32") or learning_rate.shape != ()
            or learning_rate.tobytes() != expected_lr.tobytes()
            or list(optimizer.betas) != ADAM_CONFIG["betas"]
            or optimizer.eps != ADAM_CONFIG["eps"]
            or optimizer.bias_correction is not True):
        raise CheckpointError("restored Adam constructor differs")


def validate_run_identity(run: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(run))
    required = {"schema", "logical_run_id", "created_at", "mode", "baseline",
                "limits", "max_resumes", "schedule", "recipe", "source_hashes",
                "config_hashes", "model_hashes", "package_versions"}
    if set(value) != required or value.get("schema") != RUN_SCHEMA:
        raise CheckpointError("run identity fields differ")
    if not isinstance(value["logical_run_id"], str) or not value["logical_run_id"]:
        raise CheckpointError("logical run id is empty")
    if not isinstance(value["created_at"], str) or not value["created_at"]:
        raise CheckpointError("run creation timestamp is empty")
    if value["mode"] not in {"fit", "synthetic"}:
        raise CheckpointError("run mode differs")
    if value["max_resumes"] != MAX_RESUMES:
        raise CheckpointError("resume limit differs")
    baseline = value["baseline"]
    if (not isinstance(baseline, dict) or set(baseline) != {"swap_used_bytes"}
            or type(baseline["swap_used_bytes"]) is not int
            or baseline["swap_used_bytes"] < 0):
        raise CheckpointError("original swap baseline differs")
    limits = value["limits"]
    whole = 21600.0 if value["mode"] == "fit" else 900.0
    if not isinstance(limits, dict) or limits != {**EXPECTED_LIMITS, "whole_seconds": whole}:
        raise CheckpointError("hard resource limits differ")
    schedule = value["schedule"]
    if (not isinstance(schedule, dict)
            or set(schedule) != {"sha256", "microbatches", "updates", "accumulation",
                                "final_update_microbatches"}
            or not _is_sha256(schedule["sha256"])
            or type(schedule["microbatches"]) is not int or schedule["microbatches"] < 1
            or type(schedule["updates"]) is not int or schedule["updates"] < 1
            or schedule["accumulation"] != 4
            or schedule["final_update_microbatches"] not in {1, 2, 3, 4}
            or (schedule["updates"] - 1) * 4 + schedule["final_update_microbatches"]
            != schedule["microbatches"]):
        raise CheckpointError("schedule identity differs")
    if value["mode"] == "fit" and schedule != {
            "sha256": schedule["sha256"], "microbatches": 2771, "updates": 693,
            "accumulation": 4, "final_update_microbatches": 3}:
        raise CheckpointError("FIT schedule must be exactly 2771 rows / 693 updates / tail 3")
    recipe = value["recipe"]
    expected_recipe = {
        "seed": 20260915, "lora": LORA_CONFIG, "adam": ADAM_CONFIG,
        "clip_norm": 1.0, "accumulation": 4,
        "loss": "finite_choice_log_mass_float32",
        "order": "immutable_schedule_no_shuffle",
        "post_initialization_random_consumers": [],
        "synthetic_only": value["mode"] == "synthetic",
    }
    if not isinstance(recipe, dict) or recipe != expected_recipe:
        raise CheckpointError("deterministic recipe differs")
    for field in ("source_hashes", "config_hashes", "model_hashes"):
        hashes = value[field]
        if not isinstance(hashes, dict) or not hashes or any(
                not isinstance(name, str) or not name or not _is_sha256(digest)
                for name, digest in hashes.items()):
            raise CheckpointError(f"{field} differs")
    versions = value["package_versions"]
    if not isinstance(versions, dict) or not versions or any(
            not isinstance(k, str) or not k or not isinstance(v, str) or not v
            for k, v in versions.items()):
        raise CheckpointError("package versions differ")
    return value


def make_run_identity(*, logical_run_id: str, mode: str,
                      baseline_swap_used_bytes: int, schedule_sha256: str,
                      microbatches: int, updates: int,
                      final_update_microbatches: int,
                      source_hashes: Mapping[str, str], config_hashes: Mapping[str, str],
                      model_hashes: Mapping[str, str], package_versions: Mapping[str, str],
                      created_at: str, recipe_extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    extras = dict(recipe_extra or {})
    if extras not in ({}, {"synthetic_only": True}) or extras.get(
            "synthetic_only", mode == "synthetic") != (mode == "synthetic"):
        raise CheckpointError("recipe extras are not in the reviewed deterministic contract")
    recipe = {
        "seed": 20260915, "lora": copy.deepcopy(LORA_CONFIG),
        "adam": copy.deepcopy(ADAM_CONFIG), "clip_norm": 1.0, "accumulation": 4,
        "loss": "finite_choice_log_mass_float32",
        "order": "immutable_schedule_no_shuffle",
        "post_initialization_random_consumers": [],
        "synthetic_only": mode == "synthetic",
    }
    value = {
        "schema": RUN_SCHEMA, "logical_run_id": logical_run_id, "created_at": created_at,
        "mode": mode, "baseline": {"swap_used_bytes": baseline_swap_used_bytes},
        "limits": {**EXPECTED_LIMITS, "whole_seconds": 21600.0 if mode == "fit" else 900.0},
        "max_resumes": MAX_RESUMES,
        "schedule": {"sha256": schedule_sha256, "microbatches": microbatches,
                     "updates": updates, "accumulation": 4,
                     "final_update_microbatches": final_update_microbatches},
        "recipe": recipe, "source_hashes": dict(source_hashes),
        "config_hashes": dict(config_hashes), "model_hashes": dict(model_hashes),
        "package_versions": dict(package_versions),
    }
    return validate_run_identity(value)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value)


def schedule_cursor(schedule: list[Mapping[str, Any]], completed_update: int) -> dict[str, Any]:
    if type(completed_update) is not int or completed_update < 0:
        raise ValueError("completed update must be nonnegative")
    completed = [row for row in schedule if row.get("update", 0) <= completed_update]
    if any(row.get("microbatch") != index for index, row in enumerate(schedule, 1)):
        raise CheckpointError("schedule microbatch sequence differs")
    next_row = schedule[len(completed)] if len(completed) < len(schedule) else None
    return {
        "completed_updates": completed_update,
        "completed_microbatches": len(completed),
        "next_schedule_position": len(completed),
        "next_expected": None if next_row is None else {
            "role": next_row.get("role"), "id": next_row.get("id"),
        },
    }


def cumulative_active_seconds(events: list[Mapping[str, Any]],
                              current_segment_elapsed: float = 0.0,
                              current_segment_id: str | None = None) -> float:
    if not math.isfinite(current_segment_elapsed) or current_segment_elapsed < 0:
        raise CheckpointError("current segment elapsed time differs")
    terminals: dict[str, float] = {}
    starts: set[str] = set()
    for event in events:
        kind, segment = event.get("kind"), event.get("segment_id")
        if kind in {"segment_start", "resume_start", "resolution_start"}:
            if segment in starts:
                raise CheckpointError("duplicate segment start")
            starts.add(segment)
        elif kind == "segment_terminal":
            seconds = event.get("terminal_segment_seconds")
            if (segment not in starts or segment in terminals
                    or type(seconds) not in (int, float)
                    or not math.isfinite(seconds) or seconds < 0):
                raise CheckpointError("terminal segment accounting differs")
            terminals[segment] = float(seconds)
    unmatched = starts - terminals.keys()
    if unmatched and unmatched != ({current_segment_id} if current_segment_id else set()):
        raise CheckpointError("prior segment has no durable terminal duration")
    if current_segment_id is not None and current_segment_id not in starts:
        raise CheckpointError("current segment has no durable start event")
    return sum(terminals.values()) + float(current_segment_elapsed)


def resume_preflight(run: Mapping[str, Any], events: list[Mapping[str, Any]],
                     metrics: Mapping[str, Any]) -> dict[str, float]:
    checked = validate_run_identity(run)
    required = {"swap_used_bytes", "system_free_percent"}
    if not isinstance(metrics, Mapping) or not required <= set(metrics):
        raise CheckpointError("resume resource metrics are unavailable")
    swap, free = metrics["swap_used_bytes"], metrics["system_free_percent"]
    if type(swap) is not int or swap < 0 or type(free) not in (int, float) or not math.isfinite(free):
        raise CheckpointError("resume resource metrics differ")
    growth = swap - checked["baseline"]["swap_used_bytes"]
    if growth > checked["limits"]["maximum_swap_growth_bytes"]:
        raise CheckpointError("original-baseline swap growth exceeds the hard limit")
    if free < checked["limits"]["minimum_system_free_percent"]:
        raise CheckpointError("system free memory is below the hard limit")
    if sum(event.get("kind") == "resume_start" for event in events) >= checked["max_resumes"]:
        raise CheckpointError("resume count is exhausted")
    active = cumulative_active_seconds(events)
    if active >= checked["limits"]["whole_seconds"]:
        raise CheckpointError("whole logical-run active time is exhausted")
    return {"swap_growth_bytes": float(growth), "system_free_percent": float(free),
            "prior_terminal_seconds": active}


def _validate_resume_terminal(run: Mapping[str, Any], events: list[Mapping[str, Any]],
                              pointer: Mapping[str, Any]) -> None:
    terminals = [event for event in events if event.get("kind") == "segment_terminal"
                 and str(event.get("segment_id", "")).startswith("segment-")]
    if not terminals:
        raise CheckpointError("resume requires an orderly terminal training segment")
    terminal = terminals[-1]
    if terminal.get("status") not in {"failed", "resource_failure", "finalization_required"}:
        raise CheckpointError("completed or nonrecoverable segment cannot resume")
    if (terminal.get("eligible_manifest_sha256") != pointer.get("manifest_sha256")
            or terminal.get("eligible_update") != pointer.get("update")):
        raise CheckpointError("terminal segment cursor does not match the eligible pointer")
    final = pointer.get("update") == run["schedule"]["updates"]
    if final != (terminal.get("status") == "finalization_required"):
        raise CheckpointError("final checkpoint requires finalization-only terminal status")


class MlxTensorCodec:
    def __init__(self, mx: Any):
        self.mx = mx

    def flatten(self, tree: Any) -> dict[str, Any]:
        from mlx.utils import tree_flatten
        return dict(tree_flatten(tree))

    def unflatten(self, flat: Mapping[str, Any]) -> Any:
        from mlx.utils import tree_unflatten
        return tree_unflatten(list(flat.items()))

    def save(self, path: Path, arrays: Mapping[str, Any]) -> None:
        self.mx.save_safetensors(str(path), dict(arrays))

    def load(self, path: Path) -> dict[str, Any]:
        value = self.mx.load(str(path))
        if not isinstance(value, dict):
            raise CheckpointError("safetensors payload is not a flat mapping")
        return value

    def materialize(self, *values: Any) -> None:
        self.mx.eval(*values)


@dataclass(frozen=True)
class RestoredCheckpoint:
    path: Path
    manifest: dict[str, Any]
    adapter_tensors: dict[str, Any]
    optimizer_state: Any


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path: Path, value: Any, *, replace: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    try:
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            temporary.unlink()
        _fsync_directory(path.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _read_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise CheckpointError(f"required regular file is missing: {path}")
    try:
        return json.loads(path.read_bytes())
    except Exception as error:
        raise CheckpointError(f"invalid JSON: {path}") from error


def _safe_component(value: str, label: str) -> str:
    if not isinstance(value, str) or not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise CheckpointError(f"unsafe {label}")
    return value


class CheckpointStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self._active: Segment | None = None

    @classmethod
    def create(cls, root: Path, run_identity: Mapping[str, Any]) -> "CheckpointStore":
        path = Path(root)
        path.mkdir(parents=True, exist_ok=False)
        for child in ("ledger", "checkpoints", "prune-receipts"):
            (path / child).mkdir()
        _write_json(path / "run.json", envelope(validate_run_identity(run_identity)), replace=False)
        (path / "run.lock").touch(exist_ok=False)
        _fsync_directory(path)
        return cls(path)

    @classmethod
    def open(cls, root: Path) -> "CheckpointStore":
        path = Path(root)
        if path.is_symlink() or not path.is_dir():
            raise CheckpointError("logical-run root is unavailable")
        for name in ("ledger", "checkpoints", "prune-receipts"):
            child = path / name
            if child.is_symlink() or not child.is_dir():
                raise CheckpointError(f"logical-run directory is unsafe: {name}")
        lock = path / "run.lock"
        if lock.is_symlink() or not lock.is_file():
            raise CheckpointError("logical-run lock is unsafe")
        return cls(path)

    def segment(self, segment_id: str, *, resume: bool = False,
                preflight_metrics: Mapping[str, Any] | None = None) -> "Segment":
        safe = _safe_component(segment_id, "segment id")
        if not safe.startswith("segment-") or not safe.removeprefix("segment-").isdigit():
            raise CheckpointError("segment id must be segment-qualified")
        return Segment(self, safe,
                       "resume" if resume else "fresh", preflight_metrics)

    def resolution(self, resolution_id: str) -> "Segment":
        safe = _safe_component(resolution_id, "resolution id")
        if not safe.startswith("resolution-") or not safe.removeprefix("resolution-").isdigit():
            raise CheckpointError("resolution id must be resolution-qualified")
        return Segment(self, safe,
                       "resolution", None)

    def _validate_layout(self) -> None:
        if self.root.is_symlink() or not self.root.is_dir():
            raise CheckpointError("logical-run root is unsafe")
        for name in ("ledger", "checkpoints", "prune-receipts"):
            child = self.root / name
            if child.is_symlink() or not child.is_dir():
                raise CheckpointError(f"logical-run directory is unsafe: {name}")
        for name in ("run.json", "run.lock"):
            child = self.root / name
            if child.is_symlink() or not child.is_file():
                raise CheckpointError(f"logical-run file is unsafe: {name}")

    def _run(self) -> tuple[dict[str, Any], str]:
        raw = _read_json(self.root / "run.json")
        payload = validate_run_identity(validate_envelope(raw))
        return payload, sha256_file(self.root / "run.json")

    def _events(self) -> list[dict[str, Any]]:
        files = sorted((self.root / "ledger").glob("*.json"))
        events, prior = [], None
        for sequence, path in enumerate(files, 1):
            if path.name != f"{sequence:08d}.json":
                raise CheckpointError("event sequence filenames differ")
            value = _read_json(path)
            payload = validate_envelope(value)
            if payload.get("sequence") != sequence or payload.get("prior_event_digest") != prior:
                raise CheckpointError("event hash chain differs")
            prior = value["digest"]
            events.append(payload)
        return events

    def _pointer(self, required: bool = False) -> dict[str, Any] | None:
        path = self.root / "latest.json"
        if not path.exists():
            if required:
                raise CheckpointError("latest pointer is missing")
            return None
        pointer = validate_envelope(_read_json(path))
        required_fields = {"schema", "path", "manifest_sha256", "update", "segment_id",
                           "status", "event_digest", "parent_manifest_sha256", "retained"}
        optional = {"commit_event_digest", "eligible_event_digest", "resolution_of",
                    "resolution_event_digest"}
        if (not required_fields <= set(pointer) or not (set(pointer) - required_fields <= optional)
                or pointer.get("schema") != SCHEMA or pointer.get("status") not in {"committed", "eligible"}
                or not _is_sha256(pointer.get("manifest_sha256"))
                or type(pointer.get("update")) is not int or pointer["update"] < 1
                or not isinstance(pointer.get("retained"), list)
                or not 1 <= len(pointer["retained"]) <= 2):
            raise CheckpointError("latest pointer fields differ")
        for item in pointer["retained"]:
            if (not isinstance(item, dict)
                    or set(item) - {"path", "manifest_sha256", "update", "segment_id",
                                       "eligible_event_digest"}
                    or not {"path", "manifest_sha256", "update", "segment_id"} <= set(item)
                    or not _is_sha256(item["manifest_sha256"])
                    or type(item["update"]) is not int or item["update"] < 1):
                raise CheckpointError("retained pointer entry differs")
        first = pointer["retained"][0]
        if any(pointer[key] != first[key]
               for key in ("path", "manifest_sha256", "update", "segment_id")):
            raise CheckpointError("latest pointer does not match its first retained entry")
        if (not _is_sha256(pointer["event_digest"])
                or (pointer["parent_manifest_sha256"] is not None
                    and not _is_sha256(pointer["parent_manifest_sha256"]))):
            raise CheckpointError("latest pointer hash fields differ")
        return pointer

    def load_eligible(self, mx: Any = None, *, codec: Any = None) -> RestoredCheckpoint:
        if self._active is None:
            raise CheckpointError("eligible checkpoint may be read only under the segment lock")
        pointer = self._pointer(required=True)
        assert pointer is not None
        if pointer.get("status") != "eligible":
            raise CheckpointError("latest checkpoint is not resume-eligible")
        events = self._events()
        matching_events = [event for event in events
                           if canonical_digest(event) == pointer.get("event_digest")]
        if (len(matching_events) != 1
                or matching_events[0].get("kind") not in {"resume_eligible", "checkpoint_rejected"}):
            raise CheckpointError("eligible pointer status event differs")
        status_event = matching_events[0]
        if status_event["kind"] == "resume_eligible":
            event_manifest = status_event.get("manifest_sha256")
            event_path = status_event.get("checkpoint_path")
            event_update = status_event.get("update")
        else:
            event_manifest = status_event.get("eligible_manifest_sha256")
            event_path = status_event.get("eligible_checkpoint_path")
            event_update = status_event.get("eligible_update")
        if (event_manifest != pointer.get("manifest_sha256")
                or event_path != pointer.get("path") or event_update != pointer.get("update")):
            raise CheckpointError("eligible event does not bind the current pointer")
        status_index = events.index(status_event)
        if any(event.get("kind") == "postcommit_cap_failure"
               and event.get("manifest_sha256") == pointer.get("manifest_sha256")
               for event in events[status_index + 1:]):
            raise CheckpointError("eligible checkpoint has a newer cap failure")
        relative = Path(pointer.get("path", ""))
        path = self._resolved_checkpoint_path(relative)
        manifest_file = path / "manifest.json"
        if manifest_file.is_symlink() or not manifest_file.is_file() \
                or sha256_file(manifest_file) != pointer.get("manifest_sha256"):
            raise CheckpointError("checkpoint manifest file hash differs")
        manifest = validate_envelope(_read_json(manifest_file))
        required_manifest = {
            "schema", "logical_run_id", "segment_id", "update", "completed_updates",
            "completed_microbatches", "next_schedule_position", "next_expected",
            "batch_records", "losses", "adapter_inventory", "optimizer_inventory",
            "adapter_config", "optimizer_config", "recipe", "run_json_sha256",
            "event_chain_head", "elapsed_seconds", "expected_parent_manifest_sha256",
            "expected_parent_event_digest", "files", "manifest_digest",
        }
        if set(manifest) != required_manifest or manifest.get("schema") != SCHEMA:
            raise CheckpointError("checkpoint manifest fields differ")
        if manifest.get("manifest_digest") != canonical_digest(
                {key: value for key, value in manifest.items() if key != "manifest_digest"}):
            raise CheckpointError("checkpoint manifest digest differs")
        run, run_hash = self._run()
        expected_completed = min(manifest.get("update", 0) * 4,
                                 run["schedule"]["microbatches"])
        if (manifest.get("run_json_sha256") != run_hash
                or manifest.get("logical_run_id") != run["logical_run_id"]
                or manifest.get("segment_id") != pointer.get("segment_id")
                or manifest.get("update") != pointer.get("update")
                or manifest.get("completed_updates") != manifest.get("update")
                or manifest.get("completed_microbatches") != expected_completed
                or manifest.get("next_schedule_position") != expected_completed
                or manifest.get("recipe") != run["recipe"]
                or manifest.get("adapter_config") != ADAPTER_CONFIG
                or manifest.get("optimizer_config") != ADAM_CONFIG
                or manifest.get("expected_parent_manifest_sha256")
                != pointer.get("parent_manifest_sha256")
                or manifest.get("event_chain_head") != manifest.get("expected_parent_event_digest")
                or type(manifest.get("elapsed_seconds")) not in (int, float)
                or not math.isfinite(manifest["elapsed_seconds"])
                or manifest["elapsed_seconds"] < 0):
            raise CheckpointError("checkpoint run identity differs")
        if (manifest["update"] == run["schedule"]["updates"]) != (manifest["next_expected"] is None):
            raise CheckpointError("checkpoint next schedule identity differs")
        batch_count = (run["schedule"]["final_update_microbatches"]
                       if manifest["update"] == run["schedule"]["updates"] else 4)
        first_microbatch = expected_completed - batch_count + 1
        if (len(manifest["batch_records"]) != batch_count
                or len(manifest["losses"]) != batch_count
                or any(type(loss) not in (int, float) or not math.isfinite(loss)
                       for loss in manifest["losses"])):
            raise CheckpointError("checkpoint manifest batch accounting differs")
        seen = set()
        for offset, record in enumerate(manifest["batch_records"]):
            identity = (record.get("role"), record.get("id")) if isinstance(record, dict) else None
            if (identity is None or not all(isinstance(item, str) and item for item in identity)
                    or record.get("update") != manifest["update"]
                    or record.get("microbatch") != first_microbatch + offset
                    or identity in seen):
                raise CheckpointError("checkpoint manifest batch identity differs")
            seen.add(identity)
        expected_path = Path("checkpoints") / manifest["segment_id"] / f"update-{manifest['update']:04d}"
        if relative != expected_path:
            raise CheckpointError("checkpoint pointer path does not bind manifest identity")
        commits = [event for event in events if event.get("kind") == "checkpoint_committed"
                   and event.get("manifest_sha256") == pointer["manifest_sha256"]
                   and event.get("checkpoint_path") == pointer["path"]
                   and event.get("update") == pointer["update"]
                   and event.get("segment_id") == pointer["segment_id"]]
        if (len(commits) != 1 or status_event.get("prior_event_digest") != canonical_digest(commits[0])
                or pointer.get("commit_event_digest", canonical_digest(commits[0]))
                != canonical_digest(commits[0])
                or commits[0].get("prior_event_digest") != manifest["expected_parent_event_digest"]
                or commits[0].get("parent_manifest_sha256")
                != manifest["expected_parent_manifest_sha256"]):
            raise CheckpointError("checkpoint commit/eligibility event linkage differs")
        if pointer.get("resolution_of") is not None:
            resolutions = [event for event in events
                           if canonical_digest(event) == pointer.get("resolution_event_digest")]
            if (len(resolutions) != 1 or resolutions[0].get("kind") != "checkpoint_rejected"
                    or resolutions[0].get("manifest_sha256") != pointer["resolution_of"]
                    or resolutions[0].get("eligible_manifest_sha256")
                    != pointer["manifest_sha256"]):
                raise CheckpointError("checkpoint resolution event linkage differs")
        used_codec = codec or MlxTensorCodec(mx)
        adapter_path, optimizer_path = path / "adapter.safetensors", path / "optimizer.safetensors"
        if set(manifest["files"]) != {"adapter.safetensors", "optimizer.safetensors",
                                     "adapter_config.json"}:
            raise CheckpointError("checkpoint payload file inventory differs")
        for filename, expected in ((adapter_path, manifest["files"]["adapter.safetensors"]),
                                   (optimizer_path, manifest["files"]["optimizer.safetensors"]),
                                   (path / "adapter_config.json", manifest["files"]["adapter_config.json"])):
            if filename.is_symlink() or sha256_file(filename) != expected["sha256"]:
                raise CheckpointError(f"checkpoint payload hash differs: {filename.name}")
            if filename.stat().st_size != expected["size"]:
                raise CheckpointError(f"checkpoint payload size differs: {filename.name}")
        if _read_json(path / "adapter_config.json") != ADAPTER_CONFIG:
            raise CheckpointError("serialized adapter configuration differs")
        adapter = used_codec.load(adapter_path)
        optimizer_flat = used_codec.load(optimizer_path)
        adapter_inventory = validate_adapter_tensors(adapter)
        optimizer_inventory = validate_optimizer_tensors(
            optimizer_flat, manifest["update"], adapter_inventory, manifest["optimizer_config"])
        if adapter_inventory != manifest["adapter_inventory"] or optimizer_inventory != manifest["optimizer_inventory"]:
            raise CheckpointError("reloaded tensor inventory differs from manifest")
        return RestoredCheckpoint(path, manifest, adapter,
                                  used_codec.unflatten(optimizer_flat))

    def audit_snapshot(self) -> dict[str, Any]:
        if self._active is None:
            raise CheckpointError("audit snapshot requires the segment lock")
        run, run_hash = self._run()
        events = self._events()
        manifests = []
        for path in sorted((self.root / "checkpoints").glob("segment-*/update-*/manifest.json")):
            if path.is_symlink():
                raise CheckpointError("checkpoint manifest symlink is unsafe")
            manifests.append({
                "path": str(path.parent.relative_to(self.root)),
                "file_sha256": sha256_file(path),
                "payload": validate_envelope(_read_json(path)),
                "payload_files_present": sorted(
                    child.name for child in path.parent.iterdir()
                    if child.is_file() and not child.is_symlink()),
            })
        receipts = []
        for path in sorted((self.root / "prune-receipts").glob("*.json")):
            receipts.append({"name": path.name, "file_sha256": sha256_file(path),
                             "payload": validate_envelope(_read_json(path))})
        return {
            "run": run, "run_json_sha256": run_hash,
            "events": [{"payload": event, "digest": canonical_digest(event)} for event in events],
            "latest": self._pointer(), "manifests": manifests, "prune_receipts": receipts,
        }

    def _resolved_checkpoint_path(self, relative: Path) -> Path:
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 3 \
                or relative.parts[0] != "checkpoints":
            raise CheckpointError("checkpoint pointer path differs")
        path = self.root.joinpath(relative)
        if path.is_symlink() or not path.is_dir() or path.resolve().parent.parent != (self.root / "checkpoints").resolve():
            raise CheckpointError("checkpoint path is unsafe")
        return path


class Segment:
    def __init__(self, store: CheckpointStore, segment_id: str, mode: str,
                 preflight_metrics: Mapping[str, Any] | None):
        self.store, self.segment_id, self.mode = store, segment_id, mode
        self.preflight_metrics = preflight_metrics
        self._lock_handle: Any = None
        self._terminal = False

    def __enter__(self) -> "Segment":
        if self.store._active is not None:
            raise CheckpointError("a segment already owns this store")
        lock_path = self.store.root / "run.lock"
        flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(lock_path, flags)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise CheckpointError("logical-run lock is not a regular file")
        self._lock_handle = os.fdopen(descriptor, "r+")
        try:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self._lock_handle.close()
            raise LockUnavailable("logical-run lock is already held") from error
        self.store._active = self
        try:
            self.store._validate_layout()
            run, _ = self.store._run()
            events = self.store._events()
            if any(event.get("segment_id") == self.segment_id for event in events):
                raise CheckpointError("segment id already exists")
            if self.mode == "resume":
                pointer = self.store._pointer(required=True)
                if pointer is None or pointer.get("status") != "eligible":
                    raise CheckpointError("resume requires an eligible checkpoint")
                _validate_resume_terminal(run, events, pointer)
                resume_preflight(run, events, self.preflight_metrics or {})
                kind = "resume_start"
            elif self.mode == "resolution":
                cumulative_active_seconds(events)
                pointer = self.store._pointer(required=True)
                if pointer is None or pointer.get("status") != "committed":
                    raise CheckpointError("resolution requires an unresolved committed pointer")
                if not any(event.get("kind") == "postcommit_cap_failure"
                           and event.get("manifest_sha256") == pointer["manifest_sha256"]
                           for event in events):
                    raise CheckpointError("resolution target has no durable cap failure")
                kind = "resolution_start"
            elif events:
                raise CheckpointError("non-resume segment cannot join an existing logical run")
            else:
                kind = "segment_start"
            self.append_event(kind, resume_count=sum(
                event.get("kind") == "resume_start" for event in events)
                + int(self.mode == "resume"))
            return self
        except Exception:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._lock_handle is not None:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
            self._lock_handle.close()
            self._lock_handle = None
        self.store._active = None

    def append_event(self, kind: str, **fields: Any) -> dict[str, Any]:
        if self.store._active is not self:
            raise CheckpointError("segment lock is not held")
        reserved = {"sequence", "prior_event_digest", "logical_run_id", "segment_id", "kind"}
        if reserved & fields.keys():
            raise CheckpointError("event fields override reserved hash-chain identity")
        events = self.store._events()
        payload = {
            "sequence": len(events) + 1,
            "prior_event_digest": None if not events else canonical_digest(events[-1]),
            "logical_run_id": self.store._run()[0]["logical_run_id"],
            "segment_id": self.segment_id,
            "kind": _safe_component(kind, "event kind"),
            **copy.deepcopy(fields),
        }
        wrapped = envelope(payload)
        _write_json(self.store.root / "ledger" / f"{payload['sequence']:08d}.json",
                    wrapped, replace=False)
        return wrapped

    def finish(self, status: str, terminal_segment_seconds: float, **fields: Any) -> dict[str, Any]:
        if self._terminal:
            raise CheckpointError("segment already has a terminal event")
        if status not in {"complete", "resource_failure", "failed", "finalization_required"}:
            raise CheckpointError("terminal status differs")
        if type(terminal_segment_seconds) not in (int, float) \
                or not math.isfinite(terminal_segment_seconds) or terminal_segment_seconds < 0:
            raise CheckpointError("terminal segment duration differs")
        pointer = self.store._pointer()
        eligible = None
        if pointer is not None:
            if pointer["status"] == "eligible":
                eligible = pointer
            elif len(pointer["retained"]) > 1:
                eligible = pointer["retained"][1]
        accounting = {
            "pointer_status": None if pointer is None else pointer["status"],
            "committed_update": None if pointer is None else pointer["update"],
            "committed_manifest_sha256": None if pointer is None else pointer["manifest_sha256"],
            "eligible_update": None if eligible is None else eligible["update"],
            "eligible_manifest_sha256": None if eligible is None else eligible["manifest_sha256"],
        }
        for key, expected in accounting.items():
            if key in fields and fields[key] != expected:
                raise CheckpointError("terminal checkpoint accounting differs")
        fields = {**fields, **accounting}
        result = self.append_event("segment_terminal", status=status,
                                   terminal_segment_seconds=float(terminal_segment_seconds),
                                   **fields)
        self._terminal = True
        return result

    def publish_checkpoint(self, *, mx: Any = None, codec: Any = None,
                           adapter_tensors: Mapping[str, Any], optimizer_state: Any,
                           update: int, completed_microbatches: int,
                           next_schedule: Mapping[str, Any] | None,
                           batch_records: list[Mapping[str, Any]], losses: list[float],
                           adapter_config: Mapping[str, Any],
                           optimizer_config: Mapping[str, Any], recipe: Mapping[str, Any],
                           elapsed_seconds: float,
                           resource_callback: Callable[[str], Any],
                           update_elapsed_before_checkpoint: float = 0.0,
                           failure_hook: Callable[[str], Any] | None = None) -> dict[str, Any]:
        transaction_started = time.monotonic()
        timings: dict[str, float] = {}
        observations: dict[str, Any] = {}

        def observe(stage: str) -> None:
            started = time.monotonic()
            observations[stage] = resource_callback(stage)
            timings[stage] = time.monotonic() - started

        if self.store._active is not self:
            raise CheckpointError("segment lock is not held")
        if type(update) is not int or update < 1 or type(completed_microbatches) is not int \
                or completed_microbatches < 1:
            raise CheckpointError("checkpoint cursor differs")
        if not losses or len(losses) != len(batch_records) or any(
                type(loss) not in (int, float) or not math.isfinite(loss) for loss in losses):
            raise CheckpointError("checkpoint batch losses differ")
        run, run_hash = self.store._run()
        if (dict(adapter_config) != ADAPTER_CONFIG or dict(recipe) != run["recipe"]
                or type(elapsed_seconds) not in (int, float) or not math.isfinite(elapsed_seconds)
                or elapsed_seconds < 0 or type(update_elapsed_before_checkpoint) not in (int, float)
                or not math.isfinite(update_elapsed_before_checkpoint)
                or update_elapsed_before_checkpoint < 0):
            raise CheckpointError("checkpoint recipe, adapter config, or elapsed time differs")
        expected_microbatches = min(update * 4, run["schedule"]["microbatches"])
        if update > run["schedule"]["updates"] or completed_microbatches != expected_microbatches:
            raise CheckpointError("checkpoint schedule cursor differs")
        if update == run["schedule"]["updates"] and next_schedule is not None:
            raise CheckpointError("final checkpoint has a next schedule row")
        if update < run["schedule"]["updates"] and not isinstance(next_schedule, Mapping):
            raise CheckpointError("nonfinal checkpoint lacks its next schedule row")
        used_codec = codec or MlxTensorCodec(mx)
        flat_adapter = dict(adapter_tensors)
        flat_optimizer = used_codec.flatten(optimizer_state)
        used_codec.materialize(flat_adapter, flat_optimizer)
        adapter_inventory = validate_adapter_tensors(flat_adapter)
        optimizer_inventory = validate_optimizer_tensors(
            flat_optimizer, update, adapter_inventory, optimizer_config)
        observe("post_materialize")

        pointer = self.store._pointer()
        if pointer is None:
            if update != 1:
                raise CheckpointError("the first committed checkpoint must be update 1")
        elif pointer.get("status") != "eligible" or update != pointer.get("update") + 1:
            raise CheckpointError("checkpoint update does not extend the eligible cursor")
        expected_batch_count = (run["schedule"]["final_update_microbatches"]
                                if update == run["schedule"]["updates"] else 4)
        if len(batch_records) != expected_batch_count:
            raise CheckpointError("checkpoint batch size differs from the immutable schedule")
        first_microbatch = completed_microbatches - expected_batch_count + 1
        identities: set[tuple[str, str]] = set()
        for offset, record in enumerate(batch_records):
            if (not isinstance(record, Mapping)
                    or not isinstance(record.get("role"), str) or not record["role"]
                    or not isinstance(record.get("id"), str) or not record["id"]
                    or ("update" in record and record["update"] != update)
                    or ("microbatch" in record
                        and record["microbatch"] != first_microbatch + offset)):
                raise CheckpointError("checkpoint batch identity differs")
            identity = (record["role"], record["id"])
            if identity in identities:
                raise CheckpointError("checkpoint batch identities repeat")
            identities.add(identity)
        if next_schedule is not None and (
                not isinstance(next_schedule.get("role"), str) or not next_schedule.get("role")
                or not isinstance(next_schedule.get("id"), str) or not next_schedule.get("id")):
            raise CheckpointError("next schedule identity differs")
        events = self.store._events()
        parent_manifest = None if pointer is None else pointer.get("manifest_sha256")
        parent_event = None if not events else canonical_digest(events[-1])
        segment_path = self.store.root / "checkpoints" / self.segment_id
        segment_path.mkdir(exist_ok=True)
        if segment_path.is_symlink() or segment_path.resolve().parent != (self.store.root / "checkpoints").resolve():
            raise CheckpointError("segment checkpoint path is unsafe")
        final = segment_path / f"update-{update:04d}"
        if final.exists():
            raise FileExistsError(final)
        staging = segment_path / f".update-{update:04d}.{uuid.uuid4().hex}.staging"
        staging.mkdir()
        try:
            adapter_path, optimizer_path = staging / "adapter.safetensors", staging / "optimizer.safetensors"
            used_codec.save(adapter_path, flat_adapter)
            used_codec.save(optimizer_path, flat_optimizer)
            if failure_hook is not None:
                failure_hook("after_payload_write")
            for path in (adapter_path, optimizer_path):
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
            _write_json(staging / "adapter_config.json", dict(adapter_config), replace=False)
            files = {
                name: {"size": (staging / name).stat().st_size,
                       "sha256": sha256_file(staging / name)}
                for name in ("adapter.safetensors", "optimizer.safetensors", "adapter_config.json")
            }
            manifest = {
                "schema": SCHEMA, "logical_run_id": run["logical_run_id"],
                "segment_id": self.segment_id, "update": update,
                "completed_updates": update, "completed_microbatches": completed_microbatches,
                "next_schedule_position": completed_microbatches,
                "next_expected": None if next_schedule is None else {
                    "role": next_schedule.get("role"), "id": next_schedule.get("id")},
                "batch_records": copy.deepcopy(batch_records), "losses": list(losses),
                "adapter_inventory": adapter_inventory,
                "optimizer_inventory": optimizer_inventory,
                "adapter_config": copy.deepcopy(dict(adapter_config)),
                "optimizer_config": copy.deepcopy(dict(optimizer_config)),
                "recipe": copy.deepcopy(dict(recipe)), "run_json_sha256": run_hash,
                "event_chain_head": parent_event, "elapsed_seconds": float(elapsed_seconds),
                "expected_parent_manifest_sha256": parent_manifest,
                "expected_parent_event_digest": parent_event, "files": files,
            }
            manifest["manifest_digest"] = canonical_digest(manifest)
            _write_json(staging / "manifest.json", envelope(manifest), replace=False)
            _fsync_directory(staging)
            if failure_hook is not None:
                failure_hook("after_payload_fsync")
            reloaded_adapter = used_codec.load(adapter_path)
            reloaded_optimizer = used_codec.load(optimizer_path)
            if (validate_adapter_tensors(reloaded_adapter) != adapter_inventory
                    or validate_optimizer_tensors(reloaded_optimizer, update, adapter_inventory,
                                                  optimizer_config) != optimizer_inventory):
                raise CheckpointError("checkpoint reload differs")
            if failure_hook is not None:
                failure_hook("after_reload_validation")
            observe("post_reload")
            os.replace(staging, final)
            if failure_hook is not None:
                failure_hook("after_directory_rename")
            _fsync_directory(segment_path)
            _fsync_directory(self.store.root / "checkpoints")
            if failure_hook is not None:
                failure_hook("after_checkpoint_root_fsync")
            observe("precommit")
            if failure_hook is not None:
                failure_hook("after_precommit_observation")
            manifest_sha = sha256_file(final / "manifest.json")
            retained = self._prune_before_commit(pointer, final, manifest_sha, failure_hook)
            current_pointer = self.store._pointer()
            current_events = self.store._events()
            current_event = None if not current_events else canonical_digest(current_events[-1])
            if ((None if current_pointer is None else current_pointer.get("manifest_sha256"))
                    != parent_manifest or current_event != parent_event):
                raise CheckpointConflict("checkpoint parent changed before CAS")
            if failure_hook is not None:
                failure_hook("before_pointer_replace")
            entry = {"path": str(final.relative_to(self.store.root)),
                     "manifest_sha256": manifest_sha, "update": update,
                     "segment_id": self.segment_id}
            committed_pointer = {
                "schema": SCHEMA, **entry, "status": "committed",
                "event_digest": parent_event, "parent_manifest_sha256": parent_manifest,
                "retained": [entry, *retained][:2],
            }
            _write_json(self.store.root / "latest.json", envelope(committed_pointer), replace=True)
            if failure_hook is not None:
                failure_hook("after_pointer_replace")
            commit = self.append_event("checkpoint_committed", update=update,
                                       manifest_sha256=manifest_sha,
                                       checkpoint_path=entry["path"],
                                       parent_manifest_sha256=parent_manifest)
            if failure_hook is not None:
                failure_hook("after_commit_event")
            try:
                observe("postcommit")
            except Exception:
                self.append_event("postcommit_cap_failure", update=update,
                                  manifest_sha256=manifest_sha, checkpoint_path=entry["path"])
                raise
            eligible = self.append_event("resume_eligible", update=update,
                                         manifest_sha256=manifest_sha,
                                         checkpoint_path=entry["path"])
            if failure_hook is not None:
                failure_hook("after_eligibility_event")
            after_commit = self.store._pointer(required=True)
            if after_commit != committed_pointer:
                raise CheckpointConflict("committed pointer changed before eligibility CAS")
            eligible_entry = {**entry, "eligible_event_digest": eligible["digest"]}
            eligible_pointer = {**committed_pointer, "status": "eligible",
                                "event_digest": eligible["digest"],
                                "commit_event_digest": commit["digest"],
                                "retained": [eligible_entry, *retained][:2]}
            _write_json(self.store.root / "latest.json", envelope(eligible_pointer), replace=True)
            try:
                observe("final_eligibility")
            except Exception:
                unresolved = {**committed_pointer, "event_digest": eligible["digest"],
                              "commit_event_digest": commit["digest"]}
                _write_json(self.store.root / "latest.json", envelope(unresolved), replace=True)
                failure = self.append_event("postcommit_cap_failure", update=update,
                                            manifest_sha256=manifest_sha,
                                            checkpoint_path=entry["path"],
                                            failure_stage="final_eligibility")
                unresolved = {**unresolved, "event_digest": failure["digest"]}
                _write_json(self.store.root / "latest.json", envelope(unresolved), replace=True)
                raise
            timings["total"] = time.monotonic() - transaction_started
            update_seconds = update_elapsed_before_checkpoint + timings["total"]
            if update_seconds > run["limits"]["update_seconds"]:
                unresolved = {**committed_pointer, "event_digest": eligible["digest"],
                              "commit_event_digest": commit["digest"]}
                _write_json(self.store.root / "latest.json", envelope(unresolved), replace=True)
                failure = self.append_event("postcommit_cap_failure", update=update,
                                            manifest_sha256=manifest_sha,
                                            checkpoint_path=entry["path"],
                                            update_seconds=update_seconds)
                unresolved = {**unresolved, "event_digest": failure["digest"]}
                _write_json(self.store.root / "latest.json", envelope(unresolved), replace=True)
                raise CheckpointError("optimizer update including checkpoint exceeded 120 seconds")
            payload_bytes = sum((final / name).stat().st_size for name in
                                ("adapter.safetensors", "optimizer.safetensors",
                                 "adapter_config.json", "manifest.json"))
            return {"manifest": manifest, "pointer": eligible_pointer, "path": final,
                    "timings": timings, "resource_observations": observations,
                    "payload_bytes": payload_bytes, "update_seconds": update_seconds}
        except Exception:
            # Staging/orphan evidence is intentionally preserved; no cleanup guesses state.
            raise

    def _prune_before_commit(self, pointer: dict[str, Any] | None, final: Path,
                             manifest_sha: str,
                             failure_hook: Callable[[str], Any] | None = None) -> list[dict[str, Any]]:
        if pointer is None:
            return []
        retained = list(pointer.get("retained", []))
        if not retained:
            retained = [{key: pointer[key] for key in ("path", "manifest_sha256", "update", "segment_id")}]
        if len(retained) < 2:
            return retained
        victim = retained[1]
        victim_path = self.store._resolved_checkpoint_path(Path(victim["path"]))
        victim_manifest_path = victim_path / "manifest.json"
        if sha256_file(victim_manifest_path) != victim["manifest_sha256"]:
            raise CheckpointError("prune victim manifest hash differs")
        victim_manifest = validate_envelope(_read_json(victim_manifest_path))
        if (victim_manifest.get("segment_id") != victim["segment_id"]
                or victim_manifest.get("update") != victim["update"]
                or set(victim_manifest.get("files", {}))
                != {"adapter.safetensors", "optimizer.safetensors", "adapter_config.json"}):
            raise CheckpointError("prune victim manifest identity differs")
        entries = list(victim_path.iterdir())
        names = {entry.name for entry in entries}
        full_names = {"manifest.json", "adapter.safetensors", "optimizer.safetensors",
                      "adapter_config.json"}
        if (names not in (full_names, {"manifest.json"})
                or any(entry.is_symlink() or not entry.is_file() for entry in entries)):
            raise CheckpointError("prune victim contains an unexpected file type or path")
        deletion = []
        presence = []
        for name in ("adapter.safetensors", "optimizer.safetensors", "adapter_config.json"):
            path = victim_path / name
            if path.is_symlink():
                raise CheckpointError("prune victim payload is a symlink")
            presence.append(path.is_file())
            if path.is_file():
                item = {"name": name, "size": path.stat().st_size,
                        "sha256": sha256_file(path)}
                if {"size": item["size"], "sha256": item["sha256"]} != victim_manifest["files"][name]:
                    raise CheckpointError("prune victim payload differs from its manifest")
                deletion.append(item)
        original_receipts = []
        for receipt_path in (self.store.root / "prune-receipts").glob("*.json"):
            receipt = validate_envelope(_read_json(receipt_path))
            if (receipt.get("kind") == "prune_receipt"
                    and receipt.get("victim_path") == victim["path"]
                    and receipt.get("victim_manifest_sha256") == victim["manifest_sha256"]):
                original_receipts.append(receipt)
        if len(original_receipts) > 1:
            raise CheckpointError("prune victim has duplicate original receipts")
        if not any(presence):
            if len(original_receipts) != 1:
                raise CheckpointError("deleted prune victim lacks one exact durable receipt")
            original = original_receipts[0]
            expected_names = {"adapter.safetensors", "optimizer.safetensors", "adapter_config.json"}
            if ({item.get("name") for item in original.get("deletion", [])} != expected_names
                    or original.get("victim_manifest_sha256") != victim["manifest_sha256"]):
                raise CheckpointError("prune receipt deletion inventory differs")
            reuse = {
                "schema": SCHEMA, "kind": "prune_receipt_reuse",
                "victim_path": victim["path"],
                "victim_manifest_sha256": victim["manifest_sha256"],
                "prior_receipt_digest": canonical_digest(original),
                "child_manifest_sha256": manifest_sha,
                "new_checkpoint_path": str(final.relative_to(self.store.root)),
                "deletion": original["deletion"],
            }
            reuse_name = (f"{self.segment_id}-update-{final.name.split('-')[-1]}-"
                          f"reuses-prune-{victim['segment_id']}-{victim['update']:04d}.json")
            _write_json(self.store.root / "prune-receipts" / reuse_name,
                        envelope(reuse), replace=False)
            return retained[:1]
        if not all(presence):
            raise CheckpointError("prune victim is partially deleted")
        if original_receipts:
            original = original_receipts[0]
            if original.get("deletion") != deletion:
                raise CheckpointError("durable prune receipt differs from remaining payloads")
            reuse = {
                "schema": SCHEMA, "kind": "prune_receipt_reuse",
                "victim_path": victim["path"],
                "victim_manifest_sha256": victim["manifest_sha256"],
                "prior_receipt_digest": canonical_digest(original),
                "child_manifest_sha256": manifest_sha,
                "new_checkpoint_path": str(final.relative_to(self.store.root)),
                "deletion": deletion,
            }
            reuse_name = (f"{self.segment_id}-update-{final.name.split('-')[-1]}-"
                          f"reuses-prune-{victim['segment_id']}-{victim['update']:04d}.json")
            _write_json(self.store.root / "prune-receipts" / reuse_name,
                        envelope(reuse), replace=False)
            if failure_hook is not None:
                failure_hook("post_prune_receipt_fsync")
            for item in deletion:
                (victim_path / item["name"]).unlink()
            _fsync_directory(victim_path)
            _fsync_directory(self.store.root / "checkpoints")
            return retained[:1]
        receipt_payload = {
            "schema": SCHEMA, "kind": "prune_receipt",
            "victim_path": victim["path"],
            "victim_manifest_sha256": victim["manifest_sha256"],
            "parent_manifest_sha256": pointer.get("parent_manifest_sha256"),
            "child_manifest_sha256": manifest_sha,
            "new_checkpoint_path": str(final.relative_to(self.store.root)),
            "deletion": deletion,
        }
        receipt_name = f"{self.segment_id}-update-{final.name.split('-')[-1]}-prunes-{victim['segment_id']}-{victim['update']:04d}.json"
        _write_json(self.store.root / "prune-receipts" / receipt_name,
                    envelope(receipt_payload), replace=False)
        if failure_hook is not None:
            failure_hook("post_prune_receipt_fsync")
        for item in deletion:
            (victim_path / item["name"]).unlink()
        _fsync_directory(victim_path)
        _fsync_directory(self.store.root / "checkpoints")
        return retained[:1]


def restore_model_optimizer(mx: Any, model: Any, optimizer: Any,
                            restored: RestoredCheckpoint) -> dict[str, Any]:
    manifest = restored.manifest
    adapter_inventory = validate_adapter_tensors(restored.adapter_tensors)
    if manifest["adapter_inventory"] != adapter_inventory:
        raise CheckpointError("restored adapter inventory differs")
    model.load_weights(str(restored.path / "adapter.safetensors"), strict=False)
    from scripts.long_context_lora_model import require_trainable_contract

    trainable = require_trainable_contract(model)
    mx.eval(trainable)
    if validate_adapter_tensors(trainable) != adapter_inventory:
        raise CheckpointError("model adapter hashes differ after restore")
    if manifest["update"] == FINAL_UPDATE:
        if optimizer is not None:
            raise CheckpointError("finalization-only restore must not construct an optimizer")
        model.eval()
        return {"finalization_only": True, "next_update": None, "train_mode": False}
    if optimizer is None:
        raise CheckpointError("training restore requires an optimizer")
    validate_adam_constructor(optimizer)
    optimizer.state = restored.optimizer_state
    optimizer.init(model.trainable_parameters())
    mx.eval(model.state, optimizer.state)
    codec = MlxTensorCodec(mx)
    validate_optimizer_tensors(codec.flatten(optimizer.state), manifest["update"],
                               adapter_inventory, manifest["optimizer_config"])
    model.train()
    if getattr(model, "training", True) is not True:
        raise CheckpointError("model did not enter train mode before loss construction")
    return {"finalization_only": False, "next_update": manifest["update"] + 1,
            "train_mode": True}


def resolve_postcommit_failure(store: CheckpointStore, segment: Segment,
                               expected_manifest_sha256: str,
                               mutation_hook: Callable[[str], Any] | None = None) -> dict[str, Any]:
    if store._active is not segment:
        raise CheckpointError("resolution requires the segment lock")
    if segment.mode != "resolution":
        raise CheckpointError("postcommit failure requires a dedicated offline resolution segment")
    pointer = store._pointer(required=True)
    assert pointer is not None
    if pointer.get("status") != "committed" or pointer.get("manifest_sha256") != expected_manifest_sha256:
        raise CheckpointConflict("postcommit resolution target differs")
    events = store._events()
    failures = [event for event in events if event.get("kind") == "postcommit_cap_failure"
                and event.get("manifest_sha256") == expected_manifest_sha256]
    if len(failures) != 1:
        raise CheckpointError("postcommit failure status is not authoritative")
    retained = pointer.get("retained", [])
    prior = retained[1] if len(retained) > 1 else None
    prior_event = prior_parent = None
    if prior is not None:
        prior_event = prior.get("eligible_event_digest")
        if not _is_sha256(prior_event) or not any(
                canonical_digest(event) == prior_event and event.get("kind") == "resume_eligible"
                and event.get("manifest_sha256") == prior["manifest_sha256"]
                and event.get("checkpoint_path") == prior["path"]
                and event.get("update") == prior["update"] for event in events):
            raise CheckpointError("preserved predecessor lacks an exact eligibility event")
        prior_path = store._resolved_checkpoint_path(Path(prior["path"]))
        manifest_path = prior_path / "manifest.json"
        if manifest_path.is_symlink() or sha256_file(manifest_path) != prior["manifest_sha256"]:
            raise CheckpointError("preserved predecessor manifest hash differs")
        prior_manifest = validate_envelope(_read_json(manifest_path))
        if (prior_manifest.get("segment_id") != prior["segment_id"]
                or prior_manifest.get("update") != prior["update"]):
            raise CheckpointError("preserved predecessor manifest identity differs")
        for name, item in prior_manifest.get("files", {}).items():
            path = prior_path / name
            if (path.is_symlink() or not path.is_file() or path.stat().st_size != item.get("size")
                    or sha256_file(path) != item.get("sha256")):
                raise CheckpointError("preserved predecessor payload differs")
        prior_parent = prior_manifest.get("expected_parent_manifest_sha256")
    pointer_file_hash = sha256_file(store.root / "latest.json")
    event_head = None if not events else canonical_digest(events[-1])
    rejected = segment.append_event(
        "checkpoint_rejected", manifest_sha256=expected_manifest_sha256,
        checkpoint_path=pointer["path"], update=pointer["update"],
        eligible_manifest_sha256=None if prior is None else prior["manifest_sha256"],
        eligible_checkpoint_path=None if prior is None else prior["path"],
        eligible_update=None if prior is None else prior["update"])
    if mutation_hook is not None:
        mutation_hook("before_resolution_cas")
    current_events = store._events()
    if (sha256_file(store.root / "latest.json") != pointer_file_hash
            or store._pointer(required=True) != pointer
            or len(current_events) != len(events) + 1
            or canonical_digest(current_events[-1]) != rejected["digest"]
            or current_events[-1].get("prior_event_digest") != event_head):
        raise CheckpointConflict("postcommit resolution CAS state changed")
    if prior is None:
        (store.root / "latest.json").unlink()
        _fsync_directory(store.root)
    else:
        restored_pointer = {
            "schema": SCHEMA, **prior, "status": "eligible",
            "event_digest": prior_event, "parent_manifest_sha256": prior_parent,
            "retained": [prior], "resolution_of": expected_manifest_sha256,
            "resolution_event_digest": rejected["digest"],
        }
        _write_json(store.root / "latest.json", envelope(restored_pointer), replace=True)
    return {"rejected": expected_manifest_sha256, "eligible": prior["manifest_sha256"],
            "event_digest": rejected["digest"]} if prior is not None else {
                "rejected": expected_manifest_sha256, "eligible": None,
                "event_digest": rejected["digest"]}


__all__ = [
    "ADAM_CONFIG", "ADAPTER_CONFIG", "CheckpointConflict", "CheckpointError", "CheckpointStore",
    "EXPECTED_LIMITS", "FINAL_UPDATE", "LORA_CONFIG", "LockUnavailable",
    "MAX_RESUMES", "MlxTensorCodec", "RestoredCheckpoint", "RUN_SCHEMA", "SCHEMA",
    "canonical_bytes", "canonical_digest", "cumulative_active_seconds", "envelope",
    "expected_adapter_shapes", "make_run_identity", "resolve_postcommit_failure", "restore_model_optimizer",
    "resume_preflight", "schedule_cursor", "sha256_file", "validate_adapter_tensors",
    "validate_adam_constructor", "validate_envelope", "validate_optimizer_tensors",
    "validate_run_identity",
]
