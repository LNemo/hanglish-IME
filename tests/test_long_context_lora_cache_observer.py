from collections import deque

import pytest

from scripts.long_context_lora_resources import ResourceObservationError, observe_resources
from scripts import probe_long_context_lora_resources_v2 as probe_v2


LIMITS = {
    "rss_bytes": 600,
    "mlx_peak_bytes": 600,
    "cache_bytes": 256,
    "whole_seconds": 900.0,
    "minimum_system_free_percent": 10.0,
    "maximum_swap_growth_bytes": 512,
}
BASELINE = {"swap_used_bytes": 1000}


class FakeMX:
    def __init__(self, *, active=(100,), cache=(200,), peak=(500,), clear_error=None):
        self.active = deque(active)
        self.cache = deque(cache)
        self.peak = deque(peak)
        self.clear_error = clear_error
        self.clear_calls = 0
        self.sync_calls = 0
        self.reset_calls = 0

    @staticmethod
    def _read(values):
        return values[0] if len(values) == 1 else values.popleft()

    def synchronize(self):
        self.sync_calls += 1

    def get_active_memory(self):
        return self._read(self.active)

    def get_cache_memory(self):
        return self._read(self.cache)

    def get_peak_memory(self):
        return self._read(self.peak)

    def clear_cache(self):
        self.clear_calls += 1
        if self.clear_error:
            raise self.clear_error

    def reset_peak_memory(self):
        self.reset_calls += 1


class Clock:
    def __init__(self, values):
        self.values = iter(values)

    def __call__(self):
        return next(self.values)


def system(free=40, swap=1100):
    return lambda: {"free_percent": free, "swap_used_bytes": swap}


def observe(mx, *, system_reader=None, rss_reader=lambda: 400, clock=None):
    return observe_resources(
        stage="fixture", mx=mx, started=0.0, baseline=BASELINE, limits=LIMITS,
        system_reader=system_reader or system(), rss_reader=rss_reader,
        monotonic=clock or Clock([1, 2, 3]),
    )


def test_cache_below_limit_does_not_clear_and_preserves_standard_observation():
    mx = FakeMX(cache=(200,))
    settled, record = observe(mx)
    assert settled == record["pre"]
    assert settled == {"elapsed_seconds": 2, "rss_peak_bytes": 400,
                       "mlx_active_bytes": 100, "mlx_cache_bytes": 200,
                       "mlx_peak_bytes": 500, "free_percent": 40,
                       "swap_used_bytes": 1100, "swap_growth_bytes": 100}
    assert record["action"] == "none" and record["post"] is None
    assert mx.clear_calls == 0 and mx.sync_calls == 1 and mx.reset_calls == 0


def test_excess_cache_is_cleared_and_pre_post_are_preserved():
    mx = FakeMX(active=(100, 110), cache=(300, 120), peak=(500, 500))
    settled, record = observe(
        mx, clock=Clock([1, 2, 3, 4, 5, 6]))
    assert record["pre"]["mlx_cache_bytes"] == 300
    assert record["post"]["mlx_cache_bytes"] == 120
    assert settled is record["post"]
    assert record["action"] == "clear_unused_mlx_cache"
    assert record["cleanup_seconds"] == 2
    assert record["checkpoint_seconds"] == 5
    assert mx.clear_calls == 1 and mx.sync_calls == 2 and mx.reset_calls == 0


def test_excess_cache_remaining_after_clear_fails_with_exact_record():
    mx = FakeMX(cache=(300, 280), peak=(500, 500), active=(100, 100))
    with pytest.raises(ResourceObservationError, match="residual") as caught:
        observe(mx, clock=Clock([1, 2, 3, 4, 5, 6]))
    record = caught.value.record
    assert record["pre"]["mlx_cache_bytes"] == 300
    assert record["post"]["mlx_cache_bytes"] == 280
    assert record["errors"] == ["MLX residual cache bound exceeded"]


@pytest.mark.parametrize(
    "mx,system_reader,rss_reader,match,post",
    [
        (FakeMX(peak=(601,)), system(), lambda: 400, "peak-memory", False),
        (FakeMX(), system(free=9), lambda: 400, "free-memory", False),
        (FakeMX(cache=(300, 100), peak=(500, 601), active=(100, 100)),
         system(), lambda: 400, "peak-memory", True),
    ],
)
def test_noncache_breaches_fail_before_and_after_reclamation(
        mx, system_reader, rss_reader, match, post):
    with pytest.raises(ResourceObservationError, match=match) as caught:
        observe(mx, system_reader=system_reader, rss_reader=rss_reader,
                clock=Clock([1, 2, 3, 4, 5, 6]))
    assert (caught.value.record["post"] is not None) is post
    if not post:
        assert mx.clear_calls == 0


def test_missing_metrics_fail_without_synthesizing_zero():
    mx = FakeMX()
    with pytest.raises(ResourceObservationError, match="metrics unavailable") as caught:
        observe(mx, system_reader=lambda: {"free_percent": 40},
                clock=Clock([1, 2, 3]))
    assert "swap_used_bytes" not in caught.value.record["pre"]
    assert any("swap_used_bytes unavailable" in item for item in caught.value.record["errors"])


def test_clear_failure_retains_pre_observation_and_cleanup_time():
    mx = FakeMX(cache=(300,), clear_error=RuntimeError("allocator refused"))
    with pytest.raises(ResourceObservationError, match="reclamation failed") as caught:
        observe(mx, clock=Clock([1, 2, 3, 8, 9]))
    record = caught.value.record
    assert record["pre"]["mlx_cache_bytes"] == 300
    assert record["post"] is None
    assert record["cleanup_seconds"] == 5
    assert record["checkpoint_seconds"] == 8
    assert "allocator refused" in record["errors"][0]


def test_cleanup_time_remains_in_outer_update_clock():
    times = iter([10.0, 11.0, 12.0, 15.0, 16.0, 18.0, 20.0, 22.0])
    monotonic = lambda: next(times)
    update_started = monotonic()
    mx = FakeMX(cache=(300, 100), active=(100, 100), peak=(500, 500))
    _, record = observe_resources(
        stage="update", mx=mx, started=0.0, baseline=BASELINE, limits=LIMITS,
        system_reader=system(), rss_reader=lambda: 400, monotonic=monotonic)
    update_elapsed = monotonic() - update_started
    assert record["cleanup_seconds"] == 3
    assert record["checkpoint_seconds"] == 9
    assert update_elapsed == 12


def test_system_query_time_is_included_before_whole_time_enforcement():
    now = [0.0]

    def monotonic():
        return now[0]

    def slow_system():
        now[0] = 901.0
        return {"free_percent": 40, "swap_used_bytes": 1100}

    with pytest.raises(ResourceObservationError, match="whole resource time") as caught:
        observe_resources(
            stage="slow-system-query", mx=FakeMX(), started=0.0,
            baseline=BASELINE, limits=LIMITS, system_reader=slow_system,
            rss_reader=lambda: 400, monotonic=monotonic)
    assert caught.value.record["pre"]["elapsed_seconds"] == 901.0


def test_v2_probe_imports_shared_observer_and_uses_distinct_artifact_schema():
    source = open(probe_v2.__file__, encoding="utf-8").read()
    assert "from scripts.long_context_lora_resources import" in source
    assert probe_v2.DEFAULT_OUTPUT.name == "long-context-lora-resource-v2"
    assert '"schema": "long-context-lora-resource-v2"' in source
    assert "observe_resources(" in source
    assert "v1.check_resources" not in source
