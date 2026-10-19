"""Capture one frozen TRAIN role for final-choice learning.

The partition is fixed before inference. This runner neither reads development
or final data nor trains a model. A role manifest is written only after every
source row has been captured and all inputs pass their post-run hash checks.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from hanglish.core.neural_converter import NeuralConverter, file_sha256
from hanglish.core.phonology import romanize_word
from scripts.final_choice_capture import capture_case
from scripts.partition_final_choice_data import INPUTS, read_inputs, validate_partition


OUTPUT = ROOT / "data/natural_context/final-choice-v1"
PARTITION = ROOT / ".omx/artifacts/final-choice-partition-v1/manifest.json"
PARTITION_SHA256 = "e997f06558bdb5633446171d35912140fb34b55066c13fa13b668939d6347846"
ROLE_SPECS = {
    "natural_fit": ("natural", "fit_ids", 2048),
    "natural_calibration": ("natural", "calibration_ids", 1024),
    "natural_holdout": ("natural", "holdout_ids", 512),
    "challenge_fit": ("challenge", "fit_ids", 1444),
    "challenge_preservation": ("challenge", "preservation_ids", 376),
}
PROFILE = {
    "threads": 2,
    "rss_limit_bytes": 3 * 1024**3,
    "time_limit_seconds": 10800,
    "progress_interval": 32,
}


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _valid_hashes(values):
    hexdigits = set("0123456789abcdef")
    return (isinstance(values, dict) and bool(values)
            and all(isinstance(path, str) and path
                    and isinstance(digest, str) and len(digest) == 64
                    and set(digest) <= hexdigits for path, digest in values.items()))


def read_partition(path=PARTITION, expected_sha=PARTITION_SHA256,
                   input_loader=read_inputs):
    raw = path.read_bytes()
    if sha256_bytes(raw) != expected_sha:
        raise ValueError("final-choice partition manifest hash differs")
    manifest = json.loads(raw)
    rows, snapshots = input_loader()
    if manifest.get("inputs") != snapshots:
        raise ValueError("partition TRAIN input snapshots differ")
    validate_partition(rows, manifest)
    if (manifest.get("model_inference_run") is not False
            or manifest.get("training_run") is not False
            or manifest.get("development_data_read") is not False
            or manifest.get("test_data_read") is not False
            or manifest.get("deployed") is not False):
        raise ValueError("partition manifest scope differs")
    return raw, manifest, rows, snapshots


def read_runtime_evidence(path, explicit_sha):
    if (not isinstance(explicit_sha, str) or len(explicit_sha) != 64
            or set(explicit_sha) - set("0123456789abcdef")):
        raise ValueError("an explicit runtime evidence SHA-256 is required")
    raw = path.read_bytes()
    if sha256_bytes(raw) != explicit_sha:
        raise ValueError("runtime evidence hash differs")
    report = json.loads(raw)
    sources = report.get("integrated_runtime_source_hashes")
    sources_after = report.get("runtime_source_hashes_after")
    weights = report.get("accepted_weight_hashes")
    version = report.get("integrated_runtime_version")
    if (report.get("passed") is not True
            or not isinstance(version, str) or not version
            or not _valid_hashes(sources) or sources_after != sources
            or not _valid_hashes(weights) or len(weights) != 16):
        raise ValueError("runtime evidence does not satisfy the integration schema")
    return raw, report


def _indexed(rows, name):
    result = {}
    for row in rows:
        source_id = row.get("id")
        if not isinstance(source_id, str) or not source_id or source_id in result:
            raise ValueError(f"{name} source IDs are missing or duplicated")
        result[source_id] = row
    return result


def select_sources(rows, partition, role):
    if role not in ROLE_SPECS:
        raise ValueError("unknown final-choice capture role")
    section, id_key, expected_count = ROLE_SPECS[role]
    ordered_ids = partition.get(section, {}).get(id_key)
    if (not isinstance(ordered_ids, list) or len(ordered_ids) != expected_count
            or len(set(ordered_ids)) != len(ordered_ids)):
        raise ValueError("partition role IDs or count differ")
    source_rows = rows["natural_raw_train" if section == "natural" else "challenge_train"]
    by_id = _indexed(source_rows, section)
    selected = []
    for source_id in ordered_ids:
        if source_id not in by_id:
            raise ValueError("partition role ID is absent from its TRAIN source")
        row = by_id[source_id]
        if section == "natural":
            text, context, group = row.get("text"), row.get("left_context"), row.get("group_id")
            if not all(isinstance(value, str) for value in (text, context, group)) or not text or not group:
                raise ValueError("natural TRAIN capture source is malformed")
            selected.append({
                "id": source_id, "group_id": group, "origin": "natural",
                "input": romanize_word(text), "context": context, "expected": [text],
            })
        else:
            source, context, family, expected = (
                row.get("input"), row.get("context"), row.get("family"), row.get("expected")
            )
            if (not all(isinstance(value, str) for value in (source, context, family))
                    or not source or not family or not isinstance(expected, list)
                    or not expected):
                raise ValueError("challenge TRAIN capture source is malformed")
            selected.append({
                "id": source_id, "group_id": family, "origin": "challenge",
                "input": source, "context": context, "expected": list(expected),
            })
    if [row["id"] for row in selected] != ordered_ids:
        raise ValueError("capture source order differs from frozen partition")
    return selected


def merge_hashes(target, additions, label):
    for relative, digest in additions.items():
        if relative in target and target[relative] != digest:
            raise ValueError(f"conflicting {label} hash: {relative}")
        target[relative] = digest


def expected_hashes(partition_manifest, runtime_path, runtime_sha, runtime_report,
                    runner_sha):
    hashes = {}
    merge_hashes(hashes, runtime_report["integrated_runtime_source_hashes"], "runtime source")
    merge_hashes(hashes, runtime_report["accepted_weight_hashes"], "runtime model")
    for snapshot in partition_manifest["inputs"].values():
        hashes[snapshot["path"]] = snapshot["sha256"]
    dependencies = {
        str(PARTITION.relative_to(ROOT)): PARTITION_SHA256,
        str(Path(__file__).resolve().relative_to(ROOT)): runner_sha,
        "scripts/final_choice_capture.py": file_sha256(ROOT / "scripts/final_choice_capture.py"),
        "scripts/final_choice_policy.py": file_sha256(ROOT / "scripts/final_choice_policy.py"),
        "scripts/partition_final_choice_data.py": partition_manifest["source_sha256"],
    }
    merge_hashes(hashes, dependencies, "capture dependency")
    try:
        runtime_relative = str(runtime_path.resolve().relative_to(ROOT))
    except ValueError:
        runtime_relative = str(runtime_path.resolve())
    if runtime_relative in hashes and hashes[runtime_relative] != runtime_sha:
        raise ValueError("runtime evidence conflicts with another pinned input")
    hashes[runtime_relative] = runtime_sha
    return dict(sorted(hashes.items()))


def verify_hashes(hashes):
    observed = {}
    for name, expected in hashes.items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        observed[name] = file_sha256(path)
        if observed[name] != expected:
            raise ValueError(f"capture input changed: {name}")
    return observed


def rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def require_resources(started, stage):
    if (rss_bytes() > PROFILE["rss_limit_bytes"]
            or time.monotonic() - started > PROFILE["time_limit_seconds"]):
        raise RuntimeError(f"final-choice capture resource bound exceeded during {stage}")


def capture_to_directory(output, converter, sources, *, runtime_version,
                         model_artifacts, manifest_base, capture=capture_case,
                         started=None):
    """Stream cases and write a completed manifest only after full success."""
    started = time.monotonic() if started is None else started
    output.mkdir(parents=True)
    selection_raw = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for row in sources
    )
    (output / "selection.jsonl").write_bytes(selection_raw)
    cases_path = output / "cases.jsonl"
    captured = []
    with cases_path.open("x") as stream:
        for index, source in enumerate(sources):
            row = capture(
                converter, source, expected_runtime_version=runtime_version,
                expected_model_artifacts=model_artifacts,
            )
            if row.get("id") != source["id"] or row.get("input") != source["input"]:
                raise ValueError("captured row identity differs from frozen selection")
            captured.append(row)
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            require_resources(started, "row capture")
            if (index + 1) % PROFILE["progress_interval"] == 0:
                print(json.dumps({"event": "final_choice_capture", "rows": index + 1,
                                  "elapsed_seconds": round(time.monotonic() - started, 2)}), flush=True)
    manifest = {
        **manifest_base,
        "completed": True,
        "rows": len(captured),
        "selection_sha256": sha256_bytes(selection_raw),
        "cases_sha256": file_sha256(cases_path),
        "diagnostics": {
            "top1": sum(row["result"] in row["expected"] for row in captured),
            "top5": sum(any(candidate["text"] in row["expected"]
                            for candidate in row["candidates"]) for row in captured),
            "label_aligned": sum(row["label_aligned"] for row in captured),
            "selector_eligible": sum(row["selector_eligible"] for row in captured),
            "all_candidates_can_score": sum(row["all_candidates_can_score"]
                                            for row in captured),
        },
        "baseline_accuracy_is_diagnostic_only": True,
        "split_or_threshold_selected_from_capture_outcomes": False,
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": rss_bytes(),
        "training_run": False, "new_selector_inference_run": False,
        "development_data_read": False, "final_data_read": False,
        "deployed": False,
    }
    return manifest


def run(args):
    started = time.monotonic()
    output = args.output / args.role
    if output.exists():
        raise FileExistsError(output)
    runner_source = Path(__file__).read_bytes()
    runner_sha = sha256_bytes(runner_source)
    partition_raw, partition, rows, input_snapshots = read_partition()
    runtime_raw, runtime = read_runtime_evidence(
        args.runtime_evidence, args.runtime_evidence_sha256
    )
    sources = select_sources(rows, partition, args.role)
    selection_sha = sha256_bytes(b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for row in sources
    ))
    hashes = expected_hashes(
        partition, args.runtime_evidence, sha256_bytes(runtime_raw), runtime, runner_sha
    )
    before = verify_hashes(hashes)
    require_resources(started, "pre-inference validation")

    torch.set_num_threads(PROFILE["threads"])
    os.environ["OMP_NUM_THREADS"] = str(PROFILE["threads"])
    converter = NeuralConverter(ranker_kind="hybrid")
    converter.load_models()
    runtime_version = runtime["integrated_runtime_version"]
    weights = runtime["accepted_weight_hashes"]
    if converter.version != runtime_version or converter.model_artifacts != weights:
        raise ValueError("actual converter differs from accepted runtime evidence")
    manifest_base = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "fresh actual parent TRAIN capture for final-choice learning",
        "role": args.role, "expected_rows": ROLE_SPECS[args.role][2],
        "partition_manifest_sha256": sha256_bytes(partition_raw),
        "partition_ordered_ids": [row["id"] for row in sources],
        "selection_sha256_before_inference": selection_sha,
        "train_inputs": input_snapshots,
        "runtime_evidence_path": str(args.runtime_evidence),
        "runtime_evidence_sha256": sha256_bytes(runtime_raw),
        "runtime_version": runtime_version,
        "runtime_source_hashes": runtime["integrated_runtime_source_hashes"],
        "model_artifact_hashes": weights,
        "source_hashes": before,
        "profile": PROFILE,
    }
    manifest = capture_to_directory(
        output, converter, sources, runtime_version=runtime_version,
        model_artifacts=weights, manifest_base=manifest_base, started=started,
    )
    after = verify_hashes(hashes)
    if (sha256_bytes(partition_raw) != PARTITION_SHA256
            or sha256_bytes(runtime_raw) != args.runtime_evidence_sha256
            or file_sha256(Path(__file__)) != runner_sha
            or file_sha256(output / "selection.jsonl") != selection_sha):
        raise ValueError("capture snapshots changed")
    manifest["source_hashes_after"] = after
    source_path = output / "source.py"
    source_path.write_bytes(runner_source)
    if file_sha256(source_path) != runner_sha:
        raise ValueError("capture runner snapshot differs")
    require_resources(started, "completed manifest")
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({"role": args.role, "rows": manifest["rows"],
                      "diagnostics": manifest["diagnostics"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=tuple(ROLE_SPECS), required=True)
    parser.add_argument("--runtime-evidence", type=Path, required=True)
    parser.add_argument("--runtime-evidence-sha256", required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    run(parser.parse_args())
