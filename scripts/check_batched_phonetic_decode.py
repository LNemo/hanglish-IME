#!/usr/bin/env python3
"""Root-owned bounded numerical regression for experimental beam batching."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import train_observed_phonetic_pair as contract


def run(output: Path) -> dict:
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    prepared, _, _ = contract.load_prepared(
        ROOT / ".omx/artifacts/observed-phonetic-pair-prepared-v1")
    inputs = prepared["arms"]["control"]["probe_beam_inputs"]
    checkpoint = ROOT / ".omx/artifacts/supported-phonetic-training-v1/phonetic_transducer.pt"
    expected = "409f94578163a59815f5b80e52846e7ab8cb6b534f2a5b6225719e31232d3725"
    contract.require(contract.sha256_file(checkpoint) == expected, "checkpoint differs")
    paths = [Path(__file__), ROOT / "hanglish/models/batched_phonetic_decode.py",
             ROOT / "hanglish/models/phonetic_transducer.py", checkpoint,
             checkpoint.with_suffix(".json")]
    pins = {contract.path_key(p): contract.sha256_file(p) for p in paths}
    modules = contract.torch_modules()
    from hanglish.models.batched_phonetic_decode import generate_batched
    contract.configure_torch(modules["torch"])
    baseline = modules["system_swap_bytes"]()
    rows = []
    with modules["AtomicResourceWatchdog"](
            started, baseline, output / "resources.jsonl", limits=contract.LIMITS) as watchdog:
        with watchdog.operation("load_checkpoint", update=True):
            model = modules["PhoneticTransducer"].load(checkpoint)
        initial_tensor = contract.tensor_hash(model.network.state_dict())
        for index, source in enumerate(inputs):
            values, timings = {}, {}
            for method in ("reference", "batched"):
                before = time.monotonic()
                with watchdog.operation(f"{method}_{index}", update=True):
                    values[method] = (model.generate(source["input"], 8)
                                      if method == "reference" else
                                      generate_batched(model, source["input"], 8))
                timings[method] = time.monotonic() - before
            same = [x[0] for x in values["reference"]] == [x[0] for x in values["batched"]]
            difference = (max((abs(a[1] - b[1]) for a, b in zip(
                values["reference"], values["batched"])), default=0.0) if same else None)
            row = dict(source, outputs=values, seconds=timings,
                       ordered_candidates_equal=same, max_score_difference=difference,
                       passed=same and difference <= 0.0001)
            rows.append(row)
            contract.append_jsonl(output / "comparisons.jsonl", row)
        final_tensor = contract.tensor_hash(model.network.state_dict())
        contract.verify_pins(pins)
        final_resource = watchdog.sample_and_append("final_before_report")
    report = {
        "schema": "batched-phonetic-numerical-check-v1", "completed": True,
        "passed": all(row["passed"] for row in rows) and initial_tensor == final_tensor,
        "inputs": len(rows), "nonempty_reference_inputs": sum(bool(x["outputs"]["reference"]) for x in rows),
        "mismatches": sum(not row["passed"] for row in rows),
        "reference_seconds": sum(x["seconds"]["reference"] for x in rows),
        "batched_seconds": sum(x["seconds"]["batched"] for x in rows),
        "max_score_difference": max((x["max_score_difference"] or 0 for x in rows), default=0),
        "initial_tensor_sha256": initial_tensor, "final_tensor_sha256": final_tensor,
        "comparisons_sha256": contract.sha256_file(output / "comparisons.jsonl"),
        "source_hashes": pins, "final_resource": final_resource,
        "limits": contract.LIMITS, "quality_evaluation": False,
        "scope": "TRAIN length quantiles and synthetic resource input; bounded numerical regression only",
    }
    contract.finalize_report(output, report, modules, started, baseline)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    existed = args.output.exists()
    try:
        result = run(args.output)
    except BaseException as error:
        if not existed and args.output.is_dir():
            contract.write_json(args.output / "failure-report.json", {
                "completed": False, "error": f"{type(error).__name__}: {error}"})
        raise
    print(json.dumps({k: result[k] for k in ("completed", "passed", "mismatches")}))
