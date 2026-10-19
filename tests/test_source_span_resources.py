import json
import threading
import time

import pytest

from scripts import source_span_resources as resources


@pytest.fixture
def stable_resources(monkeypatch):
    monkeypatch.setattr(resources, "rss_bytes", lambda: 100)
    monkeypatch.setattr(resources, "system_swap_bytes", lambda: 200)


def watchdog(tmp_path, **kwargs):
    limits = {
        "whole_seconds": 60.0,
        "update_seconds": 60.0,
        "rss_bytes": 1_000,
        "swap_growth_bytes": 1_000,
    }
    return resources.ResourceWatchdog(
        time.monotonic(), 200, tmp_path / "resources.jsonl",
        limits=limits, **kwargs,
    )


def read_trace(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_operation_boundaries_and_sampling_share_one_reentrant_lock(
        tmp_path, monkeypatch):
    entered_swap = threading.Event()
    release_swap = threading.Event()
    calls = 0

    def slow_swap():
        nonlocal calls
        calls += 1
        if calls == 2:  # operation_start is first; watchdog sample is second
            entered_swap.set()
            assert release_swap.wait(2)
        return 200

    monkeypatch.setattr(resources, "rss_bytes", lambda: 100)
    monkeypatch.setattr(resources, "system_swap_bytes", slow_swap)
    guard = watchdog(tmp_path)

    with guard.operation("fit-update", update=True):
        sampler = threading.Thread(
            target=lambda: guard.sample_and_append("watchdog"), daemon=True)
        sampler.start()
        assert entered_swap.wait(1)

        # Leaving the operation must wait until the in-flight observation has
        # been appended with the old operation identity.
        releaser = threading.Thread(target=lambda: (time.sleep(0.05), release_swap.set()),
                                    daemon=True)
        releaser.start()
    sampler.join(1)
    releaser.join(1)

    trace = read_trace(guard.trace)
    assert [(row["stage"], row["operation"]) for row in trace] == [
        ("operation_start", "fit-update"),
        ("watchdog", "fit-update"),
        ("operation_end", "fit-update"),
    ]
    assert all(row["operation_elapsed_s"] is not None for row in trace)


def test_idle_sample_finishes_before_new_operation_starts(tmp_path, monkeypatch):
    entered_swap = threading.Event()
    release_swap = threading.Event()

    def slow_swap():
        entered_swap.set()
        assert release_swap.wait(2)
        return 200

    monkeypatch.setattr(resources, "rss_bytes", lambda: 100)
    monkeypatch.setattr(resources, "system_swap_bytes", slow_swap)
    guard = watchdog(tmp_path)
    sampler = threading.Thread(
        target=lambda: guard.sample_and_append("watchdog"), daemon=True)
    sampler.start()
    assert entered_swap.wait(1)

    timer = threading.Timer(0.05, release_swap.set)
    timer.start()
    with guard.operation("cache-features"):
        pass
    sampler.join(1)
    timer.join(1)

    trace = read_trace(guard.trace)
    assert [(row["stage"], row["operation"]) for row in trace] == [
        ("watchdog", None),
        ("operation_start", "cache-features"),
        ("operation_end", "cache-features"),
    ]
    assert trace[0]["operation_elapsed_s"] is None
    assert trace[1]["operation_elapsed_s"] is None


def test_operation_start_rolls_back_state_when_hard_limit_fails(
        tmp_path, stable_resources):
    guard = watchdog(tmp_path)
    guard.limits["rss_bytes"] = 99

    with pytest.raises(resources.ResourceExceeded, match="RSS bound exceeded"):
        with guard.operation("too-large", update=True):
            raise AssertionError("body must not run")

    assert guard.operation_name is None
    assert guard.operation_started is None


def test_generic_failure_report_does_not_replace_training_report(
        tmp_path, stable_resources, monkeypatch):
    training_report = tmp_path / "report.json"
    training_report.write_text('{"schema":"training-owned"}\n')
    guard = watchdog(tmp_path, interval_s=0.001)
    guard.limits["rss_bytes"] = 99

    def fake_kill(pid, sig):
        assert pid > 0 and sig == resources.signal.SIGUSR1
        guard.stop_event.set()

    monkeypatch.setattr(resources.os, "kill", fake_kill)
    guard._monitor()

    failure = json.loads(guard.failure_report.read_text())
    assert failure["schema"] == resources.FAILURE_SCHEMA
    assert failure["completed"] is False
    assert failure["error_type"] == "ResourceExceeded"
    assert json.loads(training_report.read_text()) == {"schema": "training-owned"}


def test_training_report_cannot_be_selected_as_failure_path(tmp_path):
    with pytest.raises(ValueError, match="may not overwrite"):
        watchdog(tmp_path, failure_report=tmp_path / "report.json")


def test_monitor_still_signals_if_failure_report_cannot_be_written(tmp_path, monkeypatch):
    guard = watchdog(tmp_path, interval_s=.001)
    monkeypatch.setattr(guard, '_sample_locked', lambda stage: (_ for _ in ()).throw(
        resources.ResourceExceeded('test bound')))
    monkeypatch.setattr(guard, '_record_failure', lambda error: (_ for _ in ()).throw(
        OSError('disk error')))
    signals = []
    def kill(pid, sig):
        signals.append(sig)
        guard.stop_event.set()
    monkeypatch.setattr(resources.os, 'kill', kill)
    guard._monitor()
    assert signals == [resources.signal.SIGUSR1]
    assert 'test bound' in guard.failure and 'disk error' in guard.failure


def test_exit_joins_monitor_before_restoring_handler(tmp_path, monkeypatch):
    guard = watchdog(tmp_path)
    events = []
    class Monitor:
        def join(self, timeout):
            assert guard.stop_event.is_set()
            assert timeout == 15.
            events.append('joined')
        def is_alive(self):
            return False
    guard.thread = Monitor()
    monkeypatch.setattr(resources.signal, 'signal', lambda *args: events.append('restored'))
    assert guard.__exit__(None, None, None) is False
    assert events == ['joined', 'restored']


def test_exit_hard_stops_monitor_that_does_not_join(tmp_path, monkeypatch):
    guard = watchdog(tmp_path)
    class Monitor:
        def join(self, timeout):
            assert timeout == 15.
        def is_alive(self):
            return True
    guard.thread = Monitor()
    calls = []
    monkeypatch.setattr(resources.os, '_exit', lambda code: calls.append(code))
    monkeypatch.setattr(resources.signal, 'signal', lambda *args: None)
    with pytest.raises(resources.ResourceExceeded, match='did not stop'):
        guard.__exit__(None, None, None)
    assert calls == [124]


def test_sample_after_operation_end_is_idle(tmp_path, stable_resources):
    guard = watchdog(tmp_path)
    with guard.operation('one', update=True):
        pass
    observed = guard.sample_and_append('watchdog')
    assert observed['operation'] is None and observed['operation_elapsed_s'] is None


def test_failure_evidence_counts_both_arm_update_logs(tmp_path):
    guard = watchdog(tmp_path)
    for arm, count in [('whole_source', 3), ('source_local', 2)]:
        directory = tmp_path / arm
        directory.mkdir()
        (directory / 'updates.jsonl').write_text('{}\n' * count)
    evidence = guard._failure_evidence()
    assert evidence['completed_updates'] == 5
    assert evidence['completed_updates_by_arm'] == {'whole_source': 3, 'source_local': 2}
