from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import probe_long_context_lora_resources as probe


def test_resource_recipe_has_exact_four_shapes_and_limits():
    assert probe.SHAPES == ((256, 2, (0,)), (512, 3, (0, 2)),
                            (768, 4, (1, 3)), (1024, 5, (0, 2, 4)))
    assert probe.UPDATES == 20
    assert probe.ACCUMULATION == 4
    assert probe.INFERENCE_REPETITIONS == 10
    assert probe.LIMITS == {
        "rss_bytes": 6 * 1024**3,
        "mlx_peak_bytes": 6 * 1024**3,
        "cache_bytes": 256 * 1024**2,
        "whole_seconds": 900.0,
        "update_seconds": 120.0,
        "minimum_system_free_percent": 10.0,
        "maximum_swap_growth_bytes": 512 * 1024**2,
    }


def test_memory_pressure_and_swap_parsers_require_real_metrics():
    assert probe.parse_memory_pressure("System-wide memory free percentage: 59%") == 59
    assert probe.parse_swap_usage("vm.swapusage: total = 8192.00M  used = 7550.88M free = 641.12M") == round(7550.88 * 1024**2)
    with pytest.raises(ValueError, match="unavailable"):
        probe.parse_memory_pressure("no percentage")
    with pytest.raises(ValueError, match="unavailable"):
        probe.parse_swap_usage("vm.swapusage: unavailable")


def test_system_resources_uses_separate_read_only_commands():
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        output = ("System-wide memory free percentage: 41%" if command == ["memory_pressure", "-Q"]
                  else "vm.swapusage: total = 2.00G used = 1.50G free = 0.50G")
        return SimpleNamespace(stdout=output)

    assert probe.system_resources(fake_run) == {
        "free_percent": 41.0, "swap_used_bytes": round(1.5 * 1024**3)}
    assert [call[0] for call in calls] == [["memory_pressure", "-Q"], ["sysctl", "vm.swapusage"]]
    assert all(call[1] == {"check": True, "capture_output": True, "text": True,
                           "timeout": 10}
               for call in calls)


def test_p95_uses_fixed_ceil_index_and_rejects_invalid_values():
    assert probe.p95(range(1, 21)) == 19
    assert probe.p95(range(1, 11)) == 10
    for values in ([], [1, -1], [float("nan")]):
        with pytest.raises(ValueError):
            probe.p95(values)


def test_runtime_estimate_requires_all_timings_and_exact_formula():
    result = probe.runtime_estimate([2.0] * 20, [0.25] * 40)
    assert result["update_p95_seconds"] == 2.0
    assert result["probe_scoring_p95_seconds"] == 0.25
    assert result["estimated_training_seconds"] == pytest.approx(
        1.5 * (693 * 2.0 + 1280 * 0.25) + 300)
    with pytest.raises(ValueError):
        probe.runtime_estimate([2.0] * 19, [0.25] * 40)


def test_default_artifact_is_isolated_and_existing_output_fails_before_importing_mlx(tmp_path):
    assert probe.DEFAULT_OUTPUT == (probe.ROOT / ".omx/artifacts/long-context-lora-resource-v1")
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        probe.run(existing)
    assert not (existing / "report.json").exists()


def test_approved_lifecycle_and_tokenizer_artifact_hashes_are_explicit():
    assert probe.PINNED_ARTIFACT_SHA256 == {
        probe.PLAN: "cfe51fa6ccf6f408cc763a654170ae01b9cb81bfb7fa01f948f12ebfdb39449e",
        probe.TEST_SPEC: "4799276085e525dc66a94dce7610f0edfbb168eb4c1b2ce28aa9f1d5ece90189",
        probe.HANDOFF: "e1d3644a51514b7bcda899b6f27820de60b660f9520016143a01c1e26a99872d",
        probe.REFERENCES: "9b3f25c4c765d716f87b4778c4d63c9c47178251a8cd06ea063bc79ec7033d57",
        probe.TOKENIZER_LINEAGE: "0bae533c74ec1ffcf2afa674df0becb1c95119049202c732f44daa525c8cbd5d",
        probe.MODEL_SOURCE: "67928c796b7e4562c072b2bbbdd0d52569e5ab9118ade1302f8d8d1bb4bae819",
    }
    assert probe.OFFICIAL_TEMPLATE_SHA256 == "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"


def test_probe_source_has_no_fit_or_old_adapter_execution_path():
    source = Path(probe.__file__).read_text()
    assert "torch" not in source
    assert "train_final_choice_selector" not in source
    assert 'old_adapter_loaded": False' in source
    assert 'real_fit_rows_read": False' in source
