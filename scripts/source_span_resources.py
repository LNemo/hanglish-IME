"""Atomic resource watchdog for source-span training jobs.

The original preflight watchdog remains hash-pinned by completed experiments.
This module keeps its limit and signal semantics while making an observation and
its operation boundary one serialized event.  A slow RSS/swap query therefore
cannot be labelled with an operation that started or ended midway through the
query.
"""
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import signal
import threading
import time
from datetime import datetime, timezone

from scripts.preflight_pko_seq2seq import (
    LIMITS,
    ResourceExceeded,
    ResourceWatchdog as _LegacyResourceWatchdog,
    evaluate_observation,
    rss_bytes,
    system_swap_bytes,
    write_json,
)


FAILURE_SCHEMA = "source-span-resource-watchdog-failure-v1"


def _append_jsonl_unlocked(path: Path, value: dict) -> None:
    """Append one durable record; the caller owns the watchdog lock."""
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _last_jsonl_record(path: Path) -> dict | None:
    if not Path(path).is_file():
        return None
    for line in reversed(Path(path).read_bytes().splitlines()):
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        return value if isinstance(value, dict) else None
    return None


class AtomicResourceWatchdog(_LegacyResourceWatchdog):
    """Resource watchdog with operation-consistent observations.

    Sampling, durable trace append, and operation start/end transitions all use
    one re-entrant lock.  The re-entrant lock also lets callers request an
    explicit sample while already inside an operation-boundary critical section
    without deadlocking.

    Background failures are written to ``resource-watchdog-failure.json`` by
    default.  Training owns ``report.json`` and this watchdog never overwrites
    it.
    """

    def __init__(self, started: float, baseline_swap: int, trace: Path,
                 *, interval_s: float = 0.25, limits: dict = LIMITS,
                 failure_report: Path | None = None):
        super().__init__(started, baseline_swap, trace,
                         interval_s=interval_s, limits=limits)
        self.lock = threading.RLock()
        self.failure_report = (Path(failure_report) if failure_report is not None
                               else self.trace.parent / "resource-watchdog-failure.json")
        if self.failure_report == self.trace.parent / "report.json":
            raise ValueError("resource watchdog may not overwrite the training report")

    def _sample_locked(self, stage: str) -> dict:
        """Read and classify resources while ``self.lock`` is held."""
        now = time.monotonic()
        operation_name = self.operation_name
        operation_started = self.operation_started
        observed = evaluate_observation(
            elapsed_s=now - self.started,
            operation_elapsed_s=(None if operation_started is None
                                 else now - operation_started),
            rss=rss_bytes(),
            swap=system_swap_bytes(),
            baseline_swap=self.baseline_swap,
            limits=self.limits,
        )
        return observed | {
            "at": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "operation": operation_name,
        }

    def _sample(self, stage: str) -> dict:
        """Return an operation-consistent observation without appending it."""
        with self.lock:
            return self._sample_locked(stage)

    def sample_and_append(self, stage: str) -> dict:
        """Atomically sample resources and durably append the observation."""
        with self.lock:
            observed = self._sample_locked(stage)
            _append_jsonl_unlocked(self.trace, observed)
            return observed

    def _failure_evidence(self) -> dict:
        update_files = {"root": self.trace.parent / "updates.jsonl"}
        update_files.update({arm: self.trace.parent / arm / "updates.jsonl"
                             for arm in ("whole_source", "source_local")})
        per_arm = {name: sum(bool(line) for line in path.read_bytes().splitlines())
                   for name, path in update_files.items() if path.is_file()}
        return {
            "completed_updates": sum(per_arm.values()),
            "completed_updates_by_arm": per_arm,
            "last_resource_observation": _last_jsonl_record(self.trace),
        }

    def _record_failure(self, error: BaseException) -> None:
        self.failure = f"{type(error).__name__}: {error}"
        write_json(self.failure_report, {
            "schema": FAILURE_SCHEMA,
            "completed": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "watchdog_failure": self.failure,
            "partial_evidence": self._failure_evidence(),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        })

    def _monitor(self):
        while not self.stop_event.wait(self.interval_s):
            try:
                with self.lock:
                    # __exit__ may set the event while this thread is waiting for
                    # a long foreground operation to release the boundary lock.
                    if self.stop_event.is_set():
                        return
                    observed = self._sample_locked("watchdog")
                    _append_jsonl_unlocked(self.trace, observed)
            except BaseException as error:
                try:
                    self._record_failure(error)
                except BaseException as report_error:
                    # Disk/report failures must not disable the hard-stop path.
                    self.failure = (f"{type(error).__name__}: {error}; "
                                    f"failure report: {report_error}")
                os.kill(os.getpid(), signal.SIGUSR1)
                # Match the legacy watchdog's hard fail-safe when the main
                # thread remains stuck inside a C operation after signal delivery.
                if not self.stop_event.wait(5.0):  # pragma: no cover - hard fail-safe
                    os._exit(124)
                return

    def __exit__(self, exc_type, exc, tb):
        self.stop_event.set()
        # A sysctl query has a10s timeout, longer than the legacy4s join.
        # Do not restore SIGUSR1 while a live monitor can still send it. A stuck
        # filesystem write is not covered by sysctl's timeout: bound shutdown
        # too, without attempting another potentially stuck report write.
        try:
            if self.thread is not None:
                self.thread.join(timeout=15.)
                if self.thread.is_alive():
                    self.failure = 'resource monitor did not stop within15s'
                    os._exit(124)
        finally:
            signal.signal(signal.SIGUSR1, self.old_handler)
        if exc is None and self.failure:
            raise ResourceExceeded(self.failure)
        return False

    @contextlib.contextmanager
    def operation(self, name: str, *, update: bool = False):
        with self.lock:
            if self.operation_started is not None or self.operation_name is not None:
                raise RuntimeError("resource watchdog operations may not nest")
            self.operation_name = name
            self.operation_started = time.monotonic() if update else None
            try:
                observed = self._sample_locked("operation_start")
                _append_jsonl_unlocked(self.trace, observed)
            except BaseException:
                self.operation_started = None
                self.operation_name = None
                raise
        try:
            yield
        except BaseException:
            with self.lock:
                self.operation_started = None
                self.operation_name = None
            raise
        else:
            with self.lock:
                try:
                    observed = self._sample_locked("operation_end")
                    _append_jsonl_unlocked(self.trace, observed)
                finally:
                    self.operation_started = None
                    self.operation_name = None


# The matched-training runner can use the familiar name while completed pilots
# continue importing the untouched legacy implementation directly.
ResourceWatchdog = AtomicResourceWatchdog


__all__ = [
    "AtomicResourceWatchdog",
    "FAILURE_SCHEMA",
    "ResourceExceeded",
    "ResourceWatchdog",
]
