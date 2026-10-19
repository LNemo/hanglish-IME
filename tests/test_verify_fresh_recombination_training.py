import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".omx/artifacts/verify-fresh-recombination-training-v1.py"
SPEC = importlib.util.spec_from_file_location("fresh_capture_verifier", SCRIPT)
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def resource_row(stage, operation, elapsed, operation_elapsed):
    return {"stage": stage, "operation": operation,
            "elapsed_s": elapsed, "operation_elapsed_s": operation_elapsed,
            "rss_bytes": 1024, "swap_used_bytes": 0,
            "swap_growth_bytes": 0}


def write_trace(path, operations):
    trace = path / "resources.jsonl"
    elapsed = 0
    with trace.open("w", encoding="utf-8") as stream:
        for operation in operations:
            stream.write(json.dumps(resource_row(
                "operation_start", operation, elapsed, 0)) + "\n")
            elapsed += 1
            stream.write(json.dumps(resource_row(
                "operation_end", operation, elapsed, 1)) + "\n")


def report():
    return {"elapsed_s": 5, "resources": {
        "limits": verifier.LIMITS, "baseline_swap_bytes": 0,
        "final": {"elapsed_s": 4, "rss_bytes": 2048,
                  "swap_used_bytes": 0, "swap_growth_bytes": 0}}}


def test_resource_verification_counts_exact_capture_operations(tmp_path):
    operations = ["load_extended_converter", "convert:row-1",
                  "recombine:row-1"]
    write_trace(tmp_path, operations)
    result = verifier.verify_resources(tmp_path, report(), ["row-1"])
    assert result == {"peak_rss_bytes": 2048, "swap_growth_bytes": 0,
                      "operation_pairs": 3, "trace_rows": 6}


def test_resource_verification_rejects_missing_recombine_operation(tmp_path):
    write_trace(tmp_path, ["load_extended_converter", "convert:row-1"])
    with pytest.raises(ValueError, match="operation pairs"):
        verifier.verify_resources(tmp_path, report(), ["row-1"])


def test_proof_schema_self_test_is_model_free():
    assert verifier.self_test() == {
        "schema": "fresh-recombination-training-independent-verification-v1",
        "self_test": True}
