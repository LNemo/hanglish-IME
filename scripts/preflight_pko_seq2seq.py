"""Bounded local resource preflight for the pinned pko-T5-small checkpoint.

This program audits the complete 7,704-row TRAIN union and DEV *sources*, then
runs sixteen batch-one FP32 optimizer updates on eight deterministic TRAIN
examples.  It never opens the TEST split and never performs DEV inference.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import resource
import shutil
import signal
import subprocess
import sys
import threading
import time
import unicodedata
from datetime import datetime, timezone


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/challenge/sentence-v3"
EXTRA = ROOT / "data/challenge/sentence-train-extra-v3"
PLAN = ROOT / ".omx/plans/pretrained-seq2seq-options-v1.md"
PARTITION = ROOT / ".omx/artifacts/pko-training-partition-v1.json"
MODEL_REPO = "paust/pko-t5-small"
MODEL_REVISION = "91c8f3b9a762807daeb344129012593934a63f52"
MODEL_MANIFEST_SHA256 = "61afe725bb49c87de6c45bbb370bf588b10ed468f4d3095ada72c0c67245e8e9"
WEIGHT_SHA256 = "dc0594a2931c30cc0af084e3546aeb2d8bdf39dfacadf3f4e86ce6419b92ae8c"
WEIGHT_BYTES = 382_575_813
MODEL_FILE_SHA256 = {
    "config.json": "7bc0e09da49734a09cecf38f411e4c469688955b266f9bdd0b88c744c9fb1e3e",
    "tokenizer.json": "9dd2e4ba2658bd72fde0043b58064840daf0cf59e417581e184dfb1956517d4e",
    "tokenizer_config.json": "c97c75a0521afed2ca5720e8053625724e93e6d22440301e89dc74db212e398e",
    "special_tokens_map.json": "78c2fc0dfddf295e5e63b8e6872d33a105126907a53287ccd4a8ee654d7926b1",
    "README.md": "1dc4d20657b7a199639acec5cc70ac8dc3bb889b6928d421531b621f962917da",
    "pytorch_model.bin": WEIGHT_SHA256,
}
REQUIRED_MODEL_FILES = (
    "config.json", "tokenizer.json", "tokenizer_config.json", "pytorch_model.bin",
)
EXPECTED_FILES = {
    BASE / "train.jsonl": "e73ac3b3baf41075028c3b5629b4412336926faafdb8353b20c61a200bda4c42",
    BASE / "dev.jsonl": "593e1a032b54bf4d2c541da6754cfc68daf759cdcee7f8fa1f0a746459910ac9",
    BASE / "manifest.json": "ec5629a4742311f75854077cb58c343911cafecb3262c50589ca4dda30531062",
    EXTRA / "train.jsonl": "e0729e728c0c5bf2827c83b666380c4f8c4757ee204cdb248578327667a0539b",
    EXTRA / "manifest.json": "27d3e2d7afa5c76b5db82ffa8dc269054d58747b67f3a352fd4c9b0e8280d504",
    EXTRA / "independent-verification.json": "f3cbd142282c90115c3b98ad550e60e9940127e517a9fa96509cab67aba43926",
    PLAN: "4f81438aae6a7a3e00336f7cbe7066efd009202fbb9847fa0c7349631a9c445a",
    PARTITION: "8bc60b9d04f9c82c823fdb6bbecc98401d1c0a947e542b018def44fe6ac30c23",
    ROOT / ".omx/artifacts/pko-training-partition-verification-v1.json":
        "55c359a7ded8891d5e2df944e1b1d519b224abe2257695ad3d2e5fd70c150c6b",
    ROOT / ".omx/artifacts/audit-pko-tokenizer-v1.py":
        "417fc95496c2fbf73ee45ea3df6bf4491b33c937d23183b6d48c2fafa69b5588",
    ROOT / ".omx/artifacts/pko-tokenizer-audit-v1.json":
        "bf9147f7bc38990561cf59131934df9773602ead6cd440a5fb43c0887337b57b",
}
LIMITS = {
    "rss_bytes": 6 * 1024**3,
    "swap_growth_bytes": 512 * 1024**2,
    "whole_seconds": 900.0,
    "update_seconds": 120.0,
}
SEED = 20260921
MAX_TOKENS = 512
TRAIN_ROWS = 7_704
FIT_ROWS = 6_916
DEV_ROWS = 316
UPDATES = 16


class ResourceExceeded(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_key(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_json(path: Path, value: dict) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def append_jsonl(path: Path, value: dict, lock: threading.Lock | None = None) -> None:
    manager = lock if lock is not None else contextlib.nullcontext()
    with manager:
        with Path(path).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


def partial_evidence(output: Path) -> dict:
    output = Path(output)
    evidence = {"completed_updates": 0, "last_resource_observation": None}
    updates = output / "updates.jsonl"
    if updates.is_file():
        evidence["completed_updates"] = sum(bool(line) for line in updates.read_bytes().splitlines())
    progress = output / "progress.jsonl"
    if progress.is_file():
        for line in reversed(progress.read_bytes().splitlines()):
            if line:
                try:
                    evidence["last_resource_observation"] = json.loads(line)
                except json.JSONDecodeError:
                    pass
                break
    return evidence


def rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def parse_swap_usage(text: str) -> int:
    match = re.search(r"\bused\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGT])", text, re.I)
    if match is None:
        raise ValueError("system swap usage unavailable")
    scale = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    return round(float(match.group(1)) * scale[match.group(2).upper()])


def system_swap_bytes(run=subprocess.run) -> int:
    if sys.platform == "darwin":
        result = run(["sysctl", "vm.swapusage"], check=True, capture_output=True,
                     text=True, timeout=10)
        return parse_swap_usage(result.stdout)
    total = 0
    with Path("/proc/swaps").open(encoding="utf-8") as stream:
        next(stream, None)
        for line in stream:
            fields = line.split()
            if len(fields) >= 4:
                total += int(fields[3]) * 1024
    return total


def evaluate_observation(*, elapsed_s: float, operation_elapsed_s: float | None,
                         rss: int, swap: int, baseline_swap: int,
                         limits: dict = LIMITS) -> dict:
    observation = {
        "elapsed_s": float(elapsed_s), "rss_bytes": int(rss),
        "swap_used_bytes": int(swap),
        "swap_growth_bytes": int(swap - baseline_swap),
        "operation_elapsed_s": (None if operation_elapsed_s is None
                                 else float(operation_elapsed_s)),
    }
    if observation["elapsed_s"] > limits["whole_seconds"]:
        raise ResourceExceeded("whole preflight time bound exceeded")
    if observation["rss_bytes"] > limits["rss_bytes"]:
        raise ResourceExceeded("RSS bound exceeded")
    if observation["swap_growth_bytes"] > limits["swap_growth_bytes"]:
        raise ResourceExceeded("swap-growth bound exceeded")
    if (operation_elapsed_s is not None
            and operation_elapsed_s > limits["update_seconds"]):
        raise ResourceExceeded("optimizer-update time bound exceeded")
    return observation


class ResourceWatchdog:
    """Sample hard bounds while Torch is inside otherwise blocking operations."""

    def __init__(self, started: float, baseline_swap: int, trace: Path,
                 *, interval_s: float = 0.25, limits: dict = LIMITS):
        self.started = started
        self.baseline_swap = baseline_swap
        self.trace = Path(trace)
        self.interval_s = interval_s
        self.limits = dict(limits)
        self.operation_started: float | None = None
        self.operation_name: str | None = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.failure: str | None = None
        self.thread: threading.Thread | None = None
        self.old_handler = None

    def _signal_failure(self, signum, frame):  # pragma: no cover - actual OS delivery
        raise ResourceExceeded(self.failure or "resource watchdog interrupted the preflight")

    def _sample(self, stage: str) -> dict:
        now = time.monotonic()
        return evaluate_observation(
            elapsed_s=now - self.started,
            operation_elapsed_s=(None if self.operation_started is None
                                 else now - self.operation_started),
            rss=rss_bytes(), swap=system_swap_bytes(), baseline_swap=self.baseline_swap,
            limits=self.limits,
        ) | {"at": datetime.now(timezone.utc).isoformat(), "stage": stage,
             "operation": self.operation_name}

    def _monitor(self):
        while not self.stop_event.wait(self.interval_s):
            try:
                observed = self._sample("watchdog")
                append_jsonl(self.trace, observed, self.lock)
            except BaseException as error:
                self.failure = f"{type(error).__name__}: {error}"
                write_json(self.trace.parent / "report.json", {
                    "schema": "pko-t5-small-resource-preflight-v1",
                    "completed": False, "qualifies": False,
                    "error_type": type(error).__name__, "error": str(error),
                    "watchdog_failure": self.failure,
                    "partial_evidence": partial_evidence(self.trace.parent),
                    "test_opened": False, "production_changed": False,
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                })
                os.kill(os.getpid(), signal.SIGUSR1)
                # CPython dispatches signals on the main thread. Preserve the
                # terminal report and enforce the bound if a C operation does not
                # yield within the documented grace interval.
                if not self.stop_event.wait(5.0):  # pragma: no cover - hard fail-safe
                    os._exit(124)
                return

    def __enter__(self):
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("resource watchdog must be installed on the main thread")
        self.old_handler = signal.signal(signal.SIGUSR1, self._signal_failure)
        self.thread = threading.Thread(target=self._monitor, name="pko-resource-watchdog",
                                       daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=max(1.0, 2 * self.interval_s))
        signal.signal(signal.SIGUSR1, self.old_handler)
        if exc is None and self.failure:
            raise ResourceExceeded(self.failure)
        return False

    @contextlib.contextmanager
    def operation(self, name: str, *, update: bool = False):
        if self.operation_started is not None:
            raise RuntimeError("resource watchdog operations may not nest")
        self.operation_name = name
        self.operation_started = time.monotonic() if update else None
        append_jsonl(self.trace, self._sample("operation_start"), self.lock)
        try:
            yield
            append_jsonl(self.trace, self._sample("operation_end"), self.lock)
        finally:
            self.operation_started = None
            self.operation_name = None


def p95(values) -> float:
    values = sorted(float(value) for value in values)
    if not values or not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("p95 requires finite nonnegative values")
    return values[math.ceil(0.95 * len(values)) - 1]


def schedule_estimate(update_times, generation_seconds: float, generation_rows: int,
                      checkpoint_seconds: float) -> dict:
    if len(update_times) != UPDATES or generation_rows <= 0:
        raise ValueError("schedule estimate requires 16 updates and measured generation")
    update_p95 = p95(update_times)
    update_count = FIT_ROWS * 3
    training = update_count * update_p95 * 1.25
    generation = DEV_ROWS * (generation_seconds / generation_rows) * 1.25
    overhead = generation + checkpoint_seconds
    available = max(0.0, 3600.0 - overhead)
    pilot_updates = min(FIT_ROWS, math.floor(available / max(update_p95 * 1.25, 1e-12)))
    pilot_estimate = pilot_updates * update_p95 * 1.25 + overhead
    return {
        "update_p95_s": update_p95,
        "fit_rows": FIT_ROWS,
        "heldout_inner_validation_rows": TRAIN_ROWS - FIT_ROWS,
        "three_epoch_fit_updates": update_count,
        "estimated_three_epoch_training_s": training,
        "estimated_dev_generation_s": generation,
        "measured_checkpoint_reload_s": checkpoint_seconds,
        "estimated_full_schedule_s": training + overhead,
        "pilot_budget_s": 3600.0,
        "bounded_pilot_updates": pilot_updates,
        "estimated_bounded_pilot_s": pilot_estimate,
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_bytes().splitlines() if line]


def load_rows() -> tuple[list[dict], list[dict]]:
    for path, digest in EXPECTED_FILES.items():
        if sha256_file(path) != digest:
            raise ValueError(f"pinned input differs: {path_key(path)}")
    base = _read_jsonl(BASE / "train.jsonl")
    extra = _read_jsonl(EXTRA / "train.jsonl")
    dev_raw = _read_jsonl(BASE / "dev.jsonl")
    if len(base) != 5656 or len(extra) != 2048 or len(dev_raw) != DEV_ROWS:
        raise ValueError("pinned TRAIN/DEV row count differs")
    train = []
    for row in base + extra:
        if (row.get("split") != "train" or row.get("input_contract") != "roman_sentence_only"
                or not isinstance(row.get("id"), str) or not isinstance(row.get("input"), str)
                or not isinstance(row.get("expected"), list) or not row["expected"]
                or not isinstance(row["expected"][0], str)):
            raise ValueError("TRAIN row contract differs")
        train.append({"id": row["id"], "source": row["input"],
                      "target": row["expected"][0]})
    # Deliberately project only source fields; no DEV target value is retained or used.
    dev = []
    for row in dev_raw:
        if (row.get("split") != "dev" or row.get("input_contract") != "roman_sentence_only"
                or not isinstance(row.get("id"), str) or not isinstance(row.get("input"), str)):
            raise ValueError("DEV source contract differs")
        dev.append({"id": row["id"], "source": row["input"]})
    if len(train) != TRAIN_ROWS or len({row["id"] for row in train}) != TRAIN_ROWS:
        raise ValueError("TRAIN union identity differs")
    return train, dev


def fit_partition_ids(train: list[dict]) -> set[str]:
    partition = json.loads(PARTITION.read_text())
    verification_path = ROOT / ".omx/artifacts/pko-training-partition-verification-v1.json"
    verification = json.loads(verification_path.read_text())
    if (verification.get("verified") is not True
            or verification.get("artifact", {}).get("sha256") != EXPECTED_FILES[PARTITION]
            or verification.get("all_partition_leakage_keys_overlap") != 0
            or verification.get("counts") != {"fit": 6916, "inner_validation": 788}):
        raise ValueError("independent pko TRAIN partition verification differs")
    if (partition.get("completed") is not True or partition.get("rows") != TRAIN_ROWS
            or partition.get("counts") != {"fit": 6916, "inner_validation": 788}
            or partition.get("group_count") != 6274 or partition.get("largest_group") != 12):
        raise ValueError("pinned pko TRAIN partition contract differs")
    assignments = {}
    for group in partition.get("groups", []):
        split, ids = group.get("split"), group.get("ids")
        if split not in {"fit", "inner_validation"} or not isinstance(ids, list):
            raise ValueError("pinned pko TRAIN partition group differs")
        for row_id in ids:
            if row_id in assignments:
                raise ValueError("pko TRAIN partition repeats an ID")
            assignments[row_id] = split
    train_ids = {row["id"] for row in train}
    if set(assignments) != train_ids:
        raise ValueError("pko TRAIN partition does not cover the canonical union")
    fit = {row_id for row_id, split in assignments.items() if split == "fit"}
    if len(fit) != 6916:
        raise ValueError("pko TRAIN FIT count differs")
    return fit


def verify_model_dir(model_dir: Path) -> tuple[dict, dict]:
    model_dir = Path(model_dir)
    manifest_path = model_dir / "source-manifest.json"
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != MODEL_MANIFEST_SHA256:
        raise ValueError("pinned model source manifest differs")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("repo_id") != MODEL_REPO or manifest.get("revision") != MODEL_REVISION:
        raise ValueError("pinned model repository or revision differs")
    files = manifest.get("files")
    if not isinstance(files, dict) or any(name not in files for name in REQUIRED_MODEL_FILES):
        raise ValueError("pinned model manifest is incomplete")
    if set(files) != set(MODEL_FILE_SHA256):
        raise ValueError("pinned model file set differs")
    observed = {path_key(manifest_path): manifest_sha}
    for name, record in files.items():
        if Path(name).name != name or not isinstance(record, dict):
            raise ValueError("model manifest contains a nonlocal file")
        path = model_dir / name
        digest, size = sha256_file(path), path.stat().st_size
        if record.get("sha256") != digest or record.get("bytes") != size:
            raise ValueError(f"downloaded model artifact differs: {name}")
        if MODEL_FILE_SHA256.get(name) != digest:
            raise ValueError(f"pinned model artifact identity differs: {name}")
        observed[path_key(path)] = digest
    weight = files["pytorch_model.bin"]
    if weight.get("sha256") != WEIGHT_SHA256 or weight.get("bytes") != WEIGHT_BYTES:
        raise ValueError("pinned pko-T5 weight identity differs")
    return manifest, dict(sorted(observed.items()))


def tokenizer_fingerprint(tokenizer) -> str:
    return hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()


def adapt_lossless_tokenizer(tokenizer):
    """Disable NFKC and split literal special-token spellings into ordinary text."""
    tokenizer.backend_tokenizer.normalizer = None
    # Fast tokenizer calls can otherwise leave mutable batch padding/truncation
    # state that is intentionally omitted by save_pretrained().  Make those
    # inactive states part of the frozen policy before hashing or serializing.
    tokenizer.backend_tokenizer.no_padding()
    tokenizer.backend_tokenizer.no_truncation()
    tokenizer.split_special_tokens = True
    return tokenizer


def _token_ids(tokenizer, text: str, *, split_special_tokens: bool = True) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=True, truncation=False,
                        split_special_tokens=split_special_tokens)
    ids = encoded["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(value) for value in ids]


def audit_tokenizer(tokenizer, train: list[dict], dev: list[dict], *,
                    split_special_tokens: bool = True) -> tuple[dict, list[dict]]:
    records, issues = [], []
    collections = (
        ("train_source", ((row["id"], row["source"]) for row in train)),
        ("train_target", ((row["id"], row["target"]) for row in train)),
        ("dev_source", ((row["id"], row["source"]) for row in dev)),
    )
    special_tokens = tuple(token for token in tokenizer.all_special_tokens
                           if isinstance(token, str) and token)
    for scope, values in collections:
        for row_id, text in values:
            ids = _token_ids(tokenizer, text, split_special_tokens=split_special_tokens)
            decoded = tokenizer.decode(ids, skip_special_tokens=True,
                                       clean_up_tokenization_spaces=False)
            exact = decoded == text
            nfkc_equal = decoded == unicodedata.normalize("NFKC", text)
            collisions = [token for token in special_tokens if token in text]
            record = {"scope": scope, "id": row_id, "text": text,
                      "tokens": len(ids), "roundtrip_exact": exact,
                      "roundtrip_nfkc_equal": nfkc_equal,
                      "special_token_literals": collisions,
                      "special_token_loss": bool(collisions and not exact),
                      "token_ids": ids}
            records.append(record)
            if not exact or len(ids) > MAX_TOKENS:
                issues.append({key: record[key] for key in (
                    "scope", "id", "tokens", "roundtrip_exact",
                    "roundtrip_nfkc_equal", "special_token_literals", "special_token_loss")})
    stats = {}
    for scope in ("train_source", "train_target", "dev_source"):
        values = [row for row in records if row["scope"] == scope]
        lengths = [row["tokens"] for row in values]
        stats[scope] = {
            "rows": len(values), "max_tokens": max(lengths), "p95_tokens": p95(lengths),
            "over_512": sum(value > MAX_TOKENS for value in lengths),
            "roundtrip_exact": sum(row["roundtrip_exact"] for row in values),
            "roundtrip_nfkc_equal": sum(row["roundtrip_nfkc_equal"] for row in values),
            "special_token_literals": sum(bool(row["special_token_literals"])
                                          for row in values),
            "special_token_losses": sum(row["special_token_loss"] for row in values),
        }
    return stats, records


def select_probe_rows(records: list[dict]) -> list[dict]:
    if len(records) < 8:
        raise ValueError("at least eight TRAIN records are required")
    ordered = sorted(records, key=lambda row: (row["source_tokens"] + row["target_tokens"],
                                               row["id"]))
    selected, used = [], set()

    def take(label, predicate, *, key=None, reverse=False, median=False):
        pool = [row for row in ordered if row["id"] not in used and predicate(row)]
        if not pool:
            pool = [row for row in ordered if row["id"] not in used]
            label += "_fallback"
        if key is not None:
            pool = sorted(pool, key=lambda row: (key(row), row["id"]))
        row = pool[len(pool) // 2] if median else (pool[-1] if reverse else pool[0])
        used.add(row["id"])
        selected.append({**row, "selection_reason": label})

    take("longest_total", lambda row: True, reverse=True)
    take("longest_source", lambda row: True, key=lambda row: row["source_tokens"], reverse=True)
    take("longest_target", lambda row: True, key=lambda row: row["target_tokens"], reverse=True)
    take("median_total", lambda row: True, median=True)
    take("mixed_ascii_target", lambda row: bool(re.search(r"[A-Za-z]", row["target"])), reverse=True)
    take("url_or_email", lambda row: bool(re.search(r"https?://|www\.|@", row["source"])), reverse=True)
    take("numeric", lambda row: bool(re.search(r"\d", row["source"])), reverse=True)
    take("punctuated", lambda row: bool(re.search(r"[,.;:!?“”()\[\]/_+%-]", row["source"])),
         reverse=True)
    if len(selected) != 8 or len({row["id"] for row in selected}) != 8:
        raise AssertionError("deterministic TRAIN8 selection differs")
    return selected


def parameter_fingerprint(model) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def validate_logit_parity(expected, actual, *, atol: float = 1e-5,
                          rtol: float = 1e-6) -> dict:
    import torch
    if expected.shape != actual.shape:
        raise RuntimeError("reloaded logit shape differs")
    max_delta = float((expected - actual).abs().max())
    bitwise_equal = torch.equal(expected, actual)
    allclose = torch.allclose(expected, actual, atol=atol, rtol=rtol)
    argmax_same = torch.equal(expected.argmax(dim=-1), actual.argmax(dim=-1))
    result = {
        "max_logit_delta": max_delta,
        "logits_bitwise_equal": bitwise_equal,
        "logits_allclose": allclose,
        "logit_atol": atol,
        "logit_rtol": rtol,
        "argmax_same": argmax_same,
    }
    if not allclose or not argmax_same:
        raise RuntimeError(
            f"safe checkpoint output parity differs: max_delta={max_delta}, "
            f"allclose={allclose}, argmax_same={argmax_same}")
    return result


def load_model(model_dir: Path):
    from transformers import T5ForConditionalGeneration
    return T5ForConditionalGeneration.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False,
        weights_only=True, torch_dtype="float32")


def run(model_dir: Path, output: Path) -> dict:
    import torch
    from transformers import T5TokenizerFast

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source_snapshot = output / "preflight-source.py"
    shutil.copyfile(Path(__file__), source_snapshot)
    with source_snapshot.open("rb") as stream:
        os.fsync(stream.fileno())
    progress = output / "progress.jsonl"
    source_sha = sha256_file(Path(__file__))
    started = time.monotonic()
    baseline_swap = system_swap_bytes()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    random.seed(SEED)
    manifest, model_pins = verify_model_dir(model_dir)
    train, dev = load_rows()
    fit_ids = fit_partition_ids(train)
    pins = {path_key(path): digest for path, digest in EXPECTED_FILES.items()}
    pins[path_key(Path(__file__))] = source_sha
    pins.update(model_pins)

    with ResourceWatchdog(started, baseline_swap, progress) as watchdog:
        official_tokenizer = T5TokenizerFast.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=False)
        official_fingerprint = tokenizer_fingerprint(official_tokenizer)
        official_stats, official_records = audit_tokenizer(
            official_tokenizer, train, dev, split_special_tokens=False)
        tokenizer = T5TokenizerFast.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=False)
        adapt_lossless_tokenizer(tokenizer)
        adapted_fingerprint = tokenizer_fingerprint(tokenizer)
        tokenizer_stats, token_records = audit_tokenizer(tokenizer, train, dev)
        oversized = sum(stat["over_512"] for stat in tokenizer_stats.values())
        special_losses = sum(stat["special_token_losses"]
                             for stat in tokenizer_stats.values())
        roundtrip_failures = sum(stat["rows"] - stat["roundtrip_exact"]
                                 for stat in tokenizer_stats.values())
        official_by_key = {(row["scope"], row["id"]): row for row in official_records}
        adapted_by_key = {(row["scope"], row["id"]): row for row in token_records}
        normal_keys = [key for key, row in official_by_key.items()
                       if row["roundtrip_exact"] and not row["special_token_literals"]]
        changed_normal_ids = [key for key in normal_keys
                              if official_by_key[key]["token_ids"]
                              != adapted_by_key[key]["token_ids"]]
        train_by_id = {row["id"]: row for row in train}
        train_records = []
        for record in token_records:
            if record["scope"] == "train_source":
                row = train_by_id[record["id"]]
                train_records.append({**row, "source_tokens": record["tokens"]})
        targets = {record["id"]: record["tokens"] for record in token_records
                   if record["scope"] == "train_target"}
        train_records = [{**row, "target_tokens": targets[row["id"]]}
                         for row in train_records]
        selected = select_probe_rows([row for row in train_records if row["id"] in fit_ids])
        audit = {
            "policy": {
                "official_normalizer": "NFKC",
                "adapted_normalizer": None,
                "adapted_split_special_tokens": True,
                "vocabulary_changed": False,
                "official_backend_sha256": official_fingerprint,
                "adapted_backend_sha256": adapted_fingerprint,
            },
            "official_stats": official_stats,
            "adapted_stats": tokenizer_stats,
            "oversized_sequences": oversized,
            "adapted_roundtrip_failures": roundtrip_failures,
            "adapted_special_token_losses": special_losses,
            "normal_text_token_id_comparisons": len(normal_keys),
            "normal_text_token_id_mismatches": len(changed_normal_ids),
            "normal_text_token_id_mismatch_examples": [list(key)
                                                       for key in changed_normal_ids[:20]],
            "issues": [{key: row[key] for key in row
                        if key not in {"text", "token_ids"}}
                       for row in token_records
                       if (not row["roundtrip_exact"] or row["tokens"] > MAX_TOKENS)],
        }
        if oversized or special_losses or roundtrip_failures or changed_normal_ids:
            raise RuntimeError("tokenizer audit did not qualify without truncation or repair")
        del official_tokenizer, official_records, official_by_key, adapted_by_key

        with watchdog.operation("load_model"):
            model = load_model(model_dir)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
        before = parameter_fingerprint(model)
        updates = []
        for epoch in range(2):
            for row in selected:
                source = tokenizer(row["source"], return_tensors="pt",
                                   add_special_tokens=True, truncation=False,
                                   split_special_tokens=True)
                target = tokenizer(row["target"], return_tensors="pt",
                                   add_special_tokens=True, truncation=False,
                                   split_special_tokens=True)
                if source["input_ids"].shape[1] > MAX_TOKENS or target["input_ids"].shape[1] > MAX_TOKENS:
                    raise RuntimeError("selected update would require truncation")
                optimizer.zero_grad(set_to_none=True)
                begin = time.monotonic()
                with watchdog.operation(f"update_{len(updates) + 1}", update=True):
                    result = model(**source, labels=target["input_ids"])
                    loss = result.loss
                    if not torch.isfinite(loss):
                        raise RuntimeError("nonfinite seq2seq loss")
                    loss.backward()
                    gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if not torch.isfinite(gradient):
                        raise RuntimeError("nonfinite seq2seq gradient")
                    optimizer.step()
                updates.append({
                    "step": len(updates) + 1, "pass": epoch + 1, "id": row["id"],
                    "source_tokens": int(source["input_ids"].shape[1]),
                    "target_tokens": int(target["input_ids"].shape[1]),
                    "loss": float(loss.detach()), "gradient_norm": float(gradient),
                    "elapsed_s": time.monotonic() - begin,
                })
                append_jsonl(output / "updates.jsonl", updates[-1])
        after = parameter_fingerprint(model)
        if before == after:
            raise RuntimeError("optimizer updates did not change model parameters")
        del result, loss, gradient, source, target

        model.eval()
        parity_source = tokenizer(selected[0]["source"], return_tensors="pt",
                                  add_special_tokens=True, truncation=False,
                                  split_special_tokens=True)
        with torch.inference_mode():
            expected_logits = model(**parity_source, decoder_input_ids=torch.tensor([[0]])).logits.cpu()
        checkpoint = output / "checkpoint"
        checkpoint_started = time.monotonic()
        with watchdog.operation("save_checkpoint"):
            model.save_pretrained(checkpoint, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint)
        checkpoint_pins = {
            path.name: sha256_file(path) for path in sorted(checkpoint.iterdir())
            if path.is_file()
        }
        del optimizer, model
        gc.collect()
        with watchdog.operation("reload_checkpoint"):
            restored = load_model(checkpoint)
            reloaded_tokenizer = T5TokenizerFast.from_pretrained(
                str(checkpoint), local_files_only=True, trust_remote_code=False)
            adapt_lossless_tokenizer(reloaded_tokenizer)
        if tokenizer_fingerprint(reloaded_tokenizer) != adapted_fingerprint:
            raise RuntimeError("adapted tokenizer policy did not survive safe reload")
        restored_fingerprint = parameter_fingerprint(restored)
        if restored_fingerprint != after:
            raise RuntimeError("restored parameter fingerprint differs")
        for probe in (selected[0]["source"], selected[0]["target"],
                      "㈜3.3㎡14㎝", "literal<extra_id_0>tag", "<pad>", "한글"):
            probe_ids = _token_ids(reloaded_tokenizer, probe)
            if reloaded_tokenizer.decode(probe_ids, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False) != probe:
                raise RuntimeError("adapted tokenizer reload is not lossless")
        restored.eval()
        with torch.inference_mode():
            actual_logits = restored(**parity_source,
                                     decoder_input_ids=torch.tensor([[0]])).logits.cpu()
        logit_parity = validate_logit_parity(expected_logits, actual_logits)
        checkpoint_seconds = time.monotonic() - checkpoint_started

        target_max = max(row["target_tokens"] for row in selected)
        max_new_tokens = min(MAX_TOKENS, target_max + 32)
        generated = []
        generation_started = time.monotonic()
        restored.eval()
        for row in selected:
            encoded = reloaded_tokenizer(row["source"], return_tensors="pt",
                                         add_special_tokens=True, truncation=False,
                                         split_special_tokens=True)
            with watchdog.operation(f"generation_{row['id']}"):
                with torch.inference_mode():
                    output_ids = restored.generate(
                        **encoded, max_new_tokens=max_new_tokens, num_beams=1,
                        do_sample=False, early_stopping=False)
            generated.append({"id": row["id"], "source": row["source"],
                              "prediction": reloaded_tokenizer.decode(
                                  output_ids[0], skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False),
                              "output_tokens": int(output_ids.shape[1])})
        generation_seconds = time.monotonic() - generation_started
        estimate = schedule_estimate([row["elapsed_s"] for row in updates],
                                     generation_seconds, len(generated), checkpoint_seconds)
        final_observation = watchdog._sample("final")

    if sha256_file(Path(__file__)) != source_sha:
        raise RuntimeError("preflight source changed during execution")
    for raw_path, digest in pins.items():
        path = ROOT / raw_path if not Path(raw_path).is_absolute() else Path(raw_path)
        if sha256_file(path) != digest:
            raise RuntimeError(f"pinned input changed during execution: {raw_path}")
    qualifies = (
        len(updates) == UPDATES
        and all(math.isfinite(row["loss"]) and math.isfinite(row["gradient_norm"])
                and row["elapsed_s"] <= LIMITS["update_seconds"] for row in updates)
        and final_observation["rss_bytes"] <= LIMITS["rss_bytes"]
        and final_observation["swap_growth_bytes"] <= LIMITS["swap_growth_bytes"]
        and final_observation["elapsed_s"] <= LIMITS["whole_seconds"]
        and estimate["bounded_pilot_updates"] > 0
        and estimate["estimated_bounded_pilot_s"] <= estimate["pilot_budget_s"]
    )
    report = {
        "schema": "pko-t5-small-resource-preflight-v1", "completed": True,
        "qualifies": qualifies, "model": {"repo_id": MODEL_REPO,
            "revision": MODEL_REVISION, "manifest_sha256": model_pins[path_key(Path(model_dir) / "source-manifest.json")],
            "license_note": manifest.get("license_note")},
        "scope": {"train_rows": len(train), "dev_source_rows": len(dev),
                  "probe_training_partition": "fit", "probe_fit_rows": len(fit_ids),
                  "dev_file_parsed_including_label_fields": True,
                  "dev_labels_file_read": True,
                  "dev_targets_consumed": False,
                  "dev_labels_accessed_or_retained": False,
                  "dev_labels_not_passed_to_model": True,
                  "dev_inference_performed": False,
                  "test_opened": False, "production_changed": False},
        "limits": LIMITS, "pins": dict(sorted(pins.items())),
        "source_snapshot": {"path": source_snapshot.name,
                            "sha256": sha256_file(source_snapshot)},
        "tokenizer_audit": audit,
        "selection": [{key: row[key] for key in ("id", "selection_reason", "source_tokens",
                                                   "target_tokens")}
                      for row in selected],
        "training": {"precision": "float32", "device": "cpu", "threads": 2,
                     "batch_size": 1, "learning_rate": 1e-4, "weight_decay": .01,
                     "clip_norm": 1.0, "updates": updates,
                     "parameter_count": parameter_count,
                     "before_parameter_sha256": before,
                     "after_parameter_sha256": after,
                     "parameters_changed": before != after},
        "checkpoint": {"format": "safetensors",
                       "restored_parameters_exact": True,
                       "output_parity": logit_parity,
                       "elapsed_s": checkpoint_seconds,
                       "restored_parameter_sha256": restored_fingerprint,
                       "files": checkpoint_pins},
        "generation": {"scope": "selected TRAIN sources only", "num_beams": 1,
                       "max_new_tokens": max_new_tokens, "elapsed_s": generation_seconds,
                       "rows": generated},
        "schedule_estimate": estimate,
        "resources": {"baseline_swap_bytes": baseline_swap,
                      "watchdog_sample_interval_s": watchdog.interval_s,
                      "watchdog_hard_exit_grace_s": 5.0,
                      "final": final_observation,
                      "trace_sha256": sha256_file(progress)},
        "elapsed_s": time.monotonic() - started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if not qualifies:
        raise RuntimeError("measured pko-T5 resource preflight did not qualify")
    write_json(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output_preexisted = args.output.exists()
    try:
        report = run(args.model_dir, args.output)
    except BaseException as error:
        if output_preexisted:
            raise
        args.output.mkdir(parents=True, exist_ok=True)
        failure = {
            "schema": "pko-t5-small-resource-preflight-v1", "completed": False,
            "qualifies": False, "error_type": type(error).__name__, "error": str(error),
            "test_opened": False, "production_changed": False,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "partial_evidence": partial_evidence(args.output),
        }
        write_json(args.output / "report.json", failure)
        raise
    print(json.dumps({key: report[key] for key in
          ("completed", "qualifies", "elapsed_s")}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
