"""Conditionally capture fresh parent cases after an accepted long-context TRAIN gate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUTPUT = ROOT / ".omx/artifacts/long-context-lora-fresh-capture-v1"
DATA = ROOT / ".omx/artifacts/long-context-lora-data-v1-review2"
DATA_MANIFEST = DATA / "manifest.json"
DATA_MANIFEST_SHA256 = "cb4558249aa9acefb246abca9275ba8020f39cee27eb9d4143dfb94f266ecd06"
FRESH_SELECTION_SHA256 = "fb1257833c99acc406a831336f535d9ef3a786fc38bb08a84902598e5fe2287d"
FRESH_RAW = ROOT / "data/natural_context/klue-v1.1-pilot-v2/train.jsonl"
FRESH_RAW_SHA256 = "904aab04033ac758d5ccda7d8cdd75d9bd14872ad31017a14d884633e5de4d3b"
PRESERVATION = ROOT / "data/natural_context/final-choice-v1/challenge_preservation"
PRESERVATION_MANIFEST_SHA256 = "c4f587d42ac0838ce7eb2f31881bc182d8f1e0615bab47965f35da032cbaa133"
RUNTIME_REPORT = ROOT / ".omx/artifacts/numeric-structure-integration-v1/report.json"
RUNTIME_REPORT_SHA256 = "77c04334af817e15bc4c5ccc9d27b1d6ab9f9f6a759b4207e752762ae49d3aab"
TRAINING_REPORT = ROOT / ".omx/artifacts/long-context-lora-train-v1/report.json"
TRAINING_VERIFIER = ROOT / ".omx/artifacts/long-context-lora-work-v1/verify_training.py"
TRAINING_VERIFIER_SHA256 = "688484006b92e6496c79c6f4ace8b889613480f043bbb0cfb2f9019dee8b8a22"
PRESERVATION_LOADER_SHA256 = "77b6743184cdc7a4ea23cee4a18e1c86f6899267954d9d34892a502163d13a33"
RUNTIME_VERSION = "neural-1-867f13596ed5"
ROWS = 888
FRESH_ROWS = 512
FRESH_GROUPS = 387
PRESERVATION_ROWS = 376
PROFILE = {"cpu_threads": 2, "rss_limit_bytes": 3 * 1024**3,
           "time_limit_seconds": 1800.0, "progress_interval": 32}


class CaptureResourceError(RuntimeError):
    def __init__(self, message: str, record: dict):
        super().__init__(message)
        self.record = record


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_sha(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and not (
        set(value) - set("0123456789abcdef"))


def path_key(path: Path) -> str:
    return str(Path(path).resolve().relative_to(ROOT))


def merge_hashes(target: dict, values: dict, label: str) -> None:
    for name, digest in values.items():
        if name in target and target[name] != digest:
            raise ValueError(f"{label} source hash conflicts: {name}")
        target[name] = digest


def verify_source_hashes(values: dict) -> None:
    if (not isinstance(values, dict) or not values
            or any(not isinstance(name, str) or not name or not valid_sha(digest)
                   for name, digest in values.items())):
        raise ValueError("source hash inventory is malformed")
    for name, digest in values.items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError(f"source changed: {name}")


def validate_training_proof(proof_path: Path, proof_sha256: str) -> dict:
    """Validate the accepted TRAIN proof and all 137 sources before fresh access."""
    proof_path = Path(proof_path)
    path_key(proof_path)
    if not valid_sha(proof_sha256) or sha256_file(proof_path) != proof_sha256:
        raise ValueError("explicit training proof SHA-256 differs")
    if sha256_file(TRAINING_VERIFIER) != TRAINING_VERIFIER_SHA256:
        raise ValueError("accepted training verifier source differs")
    proof = json.loads(proof_path.read_text())
    if (proof.get("passed") is not True
            or proof.get("training_gate_passed") is not True
            or proof.get("verifier_sha256") != TRAINING_VERIFIER_SHA256
            or proof.get("source_hashes_checked") != 137
            or proof.get("training_records") != 2771
            or proof.get("updates") != 693
            or proof.get("final_update_microbatches") != 3
            or proof.get("diagnostic_rows") != 640
            or proof.get("reload_probes") != 4
            or proof.get("model_loaded_by_verifier") is not False
            or proof.get("fresh_labels_read") is not False
            or proof.get("development_final_labels_read") is not False
            or proof.get("train_labels_read") is not True):
        raise ValueError("training proof did not pass the accepted verifier contract")
    report_path = TRAINING_REPORT
    path_key(report_path)
    report_sha = proof.get("report_sha256")
    if not valid_sha(report_sha) or sha256_file(report_path) != report_sha:
        raise ValueError("training report hash differs from proof")
    report = json.loads(report_path.read_text())
    train_gate = report.get("train_gate", {})
    if (report.get("schema") != "long-context-lora-train-v1"
            or report.get("completed") is not True
            or report.get("terminal_status") != "success"
            or report.get("training_gate_passed") is not True
            or train_gate.get("passed") is not True
            or train_gate.get("required_net_gain") != 5
            or type(train_gate.get("net_gain")) is not int
            or train_gate["net_gain"] < 5
            or proof.get("net_top1_gain") != train_gate["net_gain"]
            or report.get("source_hashes") != report.get("source_hashes_after")
            or len(report.get("source_hashes", {})) != 137):
        raise ValueError("training report did not pass the fixed TRAIN gate")
    source_hashes = report["source_hashes"]
    verify_source_hashes(source_hashes)
    return {"report": report, "report_path": report_path, "proof": proof,
            "source_hashes": dict(source_hashes)}


def read_jsonl_bytes(raw: bytes, label: str) -> list[dict]:
    try:
        rows = [json.loads(line) for line in raw.splitlines() if line]
    except Exception as error:
        raise ValueError(f"{label} JSONL is malformed") from error
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{label} rows must be objects")
    return rows


def select_fresh_sources(selection_rows: list[dict], raw_rows: list[dict], romanizer) -> list[dict]:
    if (len(selection_rows) != FRESH_ROWS
            or any(set(row) != {"id", "group_id"} for row in selection_rows)
            or len({row["id"] for row in selection_rows}) != FRESH_ROWS
            or len({row["group_id"] for row in selection_rows}) != FRESH_GROUPS):
        raise ValueError("fresh selection identity/group accounting differs")
    by_id = {}
    for row in raw_rows:
        identity = row.get("id")
        if not isinstance(identity, str) or not identity or identity in by_id:
            raise ValueError("fresh raw source IDs are missing or duplicated")
        by_id[identity] = row
    selected_ids = {row["id"] for row in selection_rows}
    selected_groups = {row["group_id"] for row in selection_rows}
    whole_group_ids = {row["id"] for row in raw_rows if row.get("group_id") in selected_groups}
    if whole_group_ids != selected_ids:
        raise ValueError("fresh selection does not contain complete groups")
    sources = []
    for selected in selection_rows:
        row = by_id.get(selected["id"])
        if (row is None or row.get("group_id") != selected["group_id"]
                or row.get("split") != "train"
                or not isinstance(row.get("text"), str) or not row["text"]
                or not isinstance(row.get("left_context"), str)):
            raise ValueError("fresh selected raw row differs")
        source = romanizer(row["text"])
        if not isinstance(source, str) or not source:
            raise ValueError("fresh romanized input is empty")
        sources.append({"id": row["id"], "group_id": row["group_id"],
                        "origin": "fresh_natural", "input": source,
                        "context": row["left_context"], "expected": [row["text"]]})
    return sources


def wrap_items(fresh_rows: list[dict], preservation_rows: list[dict]) -> list[dict]:
    if len(fresh_rows) != FRESH_ROWS or len(preservation_rows) != PRESERVATION_ROWS:
        raise ValueError("fresh/preservation capture counts differ")
    if any(not isinstance(row.get("group_id"), str) or not row["group_id"]
           for row in fresh_rows):
        raise ValueError("fresh capture group identity is missing")
    return ([{"role": "fresh_natural", "stratum": "natural", "row": row}
             for row in fresh_rows]
            + [{"role": "challenge_preservation", "stratum": "preservation", "row": row}
               for row in preservation_rows])


def rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(usage if sys.platform == "darwin" else usage * 1024)


def require_resources(started: float, stage: str) -> dict:
    observed = {"stage": stage, "elapsed_seconds": time.monotonic() - started,
                "rss_peak_bytes": rss_bytes()}
    if observed["rss_peak_bytes"] > PROFILE["rss_limit_bytes"]:
        raise CaptureResourceError("fresh capture exceeded 3GiB RSS", observed)
    if observed["elapsed_seconds"] > PROFILE["time_limit_seconds"]:
        raise CaptureResourceError("fresh capture exceeded 1800 seconds", observed)
    return observed


def canonical_jsonl(values: list[dict]) -> bytes:
    return b"".join((json.dumps(value, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":")) + "\n").encode()
                    for value in values)


def access_target(report: dict, flag: str, reader):
    """Record target-bearing access before invoking a reader that may fail."""
    report[flag] = True
    return reader()


def write_failed_report(report_path: Path, report: dict) -> None:
    """Best-effort failure evidence that never replaces the active exception."""
    try:
        report_path.write_text(json.dumps(report, ensure_ascii=False,
                                          sort_keys=True, indent=2) + "\n")
    except Exception as error:
        try:
            sys.stderr.write(json.dumps({
                "event": "fresh_capture_failure_report_write_failed",
                "error": f"{type(error).__name__}: {error}",
            }, sort_keys=True) + "\n")
            sys.stderr.flush()
        except Exception:
            pass


def run(args) -> dict:
    started = time.monotonic()
    if args.output.exists():
        raise FileExistsError(f"fresh capture output already exists: {args.output}")

    source_raw = Path(__file__).read_bytes()
    source_sha = hashlib.sha256(source_raw).hexdigest()
    # This complete proof/source gate precedes every target-bearing fresh or
    # preservation read and every CPU model import/load.
    training = validate_training_proof(args.training_proof, args.training_proof_sha256)
    if sha256_file(DATA_MANIFEST) != DATA_MANIFEST_SHA256:
        raise ValueError("reviewed long-context data manifest differs")
    data_manifest = json.loads(DATA_MANIFEST.read_text())
    selection_path = DATA / "fresh_selection.jsonl"
    selection_raw = selection_path.read_bytes()
    if (sha256_file(selection_path) != FRESH_SELECTION_SHA256
            or data_manifest.get("file_hashes", {}).get("fresh_selection.jsonl")
            != FRESH_SELECTION_SHA256
            or data_manifest.get("fresh", {}).get("rows") != FRESH_ROWS
            or data_manifest.get("fresh", {}).get("groups") != FRESH_GROUPS
            or data_manifest.get("fresh", {}).get("whole_group_selection") is not True
            or data_manifest.get("fresh", {}).get("target_fields_copied") is not False):
        raise ValueError("reviewed fresh selection contract differs")
    selection_ids = read_jsonl_bytes(selection_raw, "fresh selection")

    from scripts.prepare_final_choice_data import read_runtime_evidence
    runtime_raw, runtime = read_runtime_evidence(RUNTIME_REPORT, RUNTIME_REPORT_SHA256)
    if runtime.get("integrated_runtime_version") != RUNTIME_VERSION:
        raise ValueError("accepted runtime version differs")
    require_resources(started, "pre-label gate")

    args.output.mkdir(parents=True)
    report_path = args.output / "manifest.json"
    report = {"schema": "long-context-lora-fresh-capture-v1", "completed": False,
              "created_at": datetime.now(timezone.utc).isoformat(),
              "terminal_status": "running", "deployed": False,
              "model_inference_run": False, "fresh_labels_read": False,
              "preservation_labels_read": False,
              "training_report_sha256": training["proof"]["report_sha256"],
              "training_proof_sha256": args.training_proof_sha256,
              "capture_profile": PROFILE}
    converter = None
    try:
        # Target-bearing data begins only after the accepted proof gate above.
        fresh_raw = access_target(report, "fresh_labels_read", FRESH_RAW.read_bytes)
        if hashlib.sha256(fresh_raw).hexdigest() != FRESH_RAW_SHA256:
            raise ValueError("pinned fresh raw source differs")
        from hanglish.core.phonology import romanize_word
        fresh_sources = select_fresh_sources(
            selection_ids, read_jsonl_bytes(fresh_raw, "fresh raw source"), romanize_word)
        preservation_manifest_path = PRESERVATION / "manifest.json"
        if (sha256_file(preservation_manifest_path) != PRESERVATION_MANIFEST_SHA256
                or sha256_file(ROOT / "scripts/evaluate_final_choice_holdout.py")
                != PRESERVATION_LOADER_SHA256):
            raise ValueError("pinned preservation loader or manifest differs")
        preservation_manifest = json.loads(preservation_manifest_path.read_text())
        verify_source_hashes(preservation_manifest.get("source_hashes"))
        from scripts.evaluate_final_choice_holdout import load_evaluation_capture
        preservation = access_target(
            report, "preservation_labels_read",
            lambda: load_evaluation_capture(
                PRESERVATION, PRESERVATION_MANIFEST_SHA256, "challenge_preservation"))
        if (len(preservation["rows"]) != PRESERVATION_ROWS
                or preservation["manifest"].get("runtime_version") != RUNTIME_VERSION
                or preservation["manifest"].get("model_artifact_hashes")
                != runtime["accepted_weight_hashes"]):
            raise ValueError("pinned preservation capture differs")

        source_hashes = dict(training["source_hashes"])
        merge_hashes(source_hashes, runtime["integrated_runtime_source_hashes"], "runtime")
        merge_hashes(source_hashes, runtime["accepted_weight_hashes"], "runtime weights")
        merge_hashes(source_hashes, preservation["manifest"]["source_hashes"], "preservation")
        fixed_paths = [Path(__file__), TRAINING_VERIFIER, args.training_proof,
                       training["report_path"], DATA_MANIFEST, selection_path,
                       FRESH_RAW, RUNTIME_REPORT,
                       ROOT / "scripts/prepare_final_choice_data.py",
                       ROOT / "scripts/evaluate_final_choice_holdout.py",
                       ROOT / "scripts/final_choice_capture.py",
                       ROOT / "hanglish/core/neural_converter.py",
                       ROOT / "hanglish/core/phonology.py"]
        fixed_paths += [PRESERVATION / name for name in
                        ("manifest.json", "selection.jsonl", "cases.jsonl")]
        merge_hashes(source_hashes, {path_key(path): sha256_file(path) for path in fixed_paths},
                     "capture inputs")
        verify_source_hashes(source_hashes)
        before = dict(sorted(source_hashes.items()))
        if before[path_key(Path(__file__))] != source_sha:
            raise RuntimeError("fresh capture source snapshot differs")

        selection_output = args.output / "selection.jsonl"
        selection_output.write_bytes(canonical_jsonl(fresh_sources))
        os.environ["OMP_NUM_THREADS"] = str(PROFILE["cpu_threads"])
        import torch
        torch.set_num_threads(PROFILE["cpu_threads"])
        from hanglish.core.neural_converter import NeuralConverter
        from scripts.final_choice_capture import capture_case
        converter = NeuralConverter(ranker_kind="hybrid", external_lexicon=True)
        converter.load_models()
        if (converter.version != RUNTIME_VERSION
                or converter.model_artifacts != runtime["accepted_weight_hashes"]):
            raise ValueError("loaded CPU parent runtime differs")
        report["model_inference_run"] = True
        captured = []
        for index, source in enumerate(fresh_sources, 1):
            captured.append(capture_case(
                converter, source, expected_runtime_version=RUNTIME_VERSION,
                expected_model_artifacts=runtime["accepted_weight_hashes"]))
            require_resources(started, f"fresh_capture_{index}")
            if index % PROFILE["progress_interval"] == 0:
                print(json.dumps({"event": "long_context_fresh_capture",
                                  "rows": index}), flush=True)
        items = wrap_items(captured, preservation["rows"])
        items_path = args.output / "items.jsonl"
        items_path.write_bytes(canonical_jsonl(items))
        del converter
        converter = None
        gc.collect()
        terminal = require_resources(started, "terminal")
        verify_source_hashes(before)
        if Path(__file__).read_bytes() != source_raw:
            raise RuntimeError("fresh capture source changed during execution")
        after = dict(before)
        if after != before:
            raise RuntimeError("fresh capture sources changed")
        file_hashes = {"items.jsonl": sha256_file(items_path),
                       "selection.jsonl": sha256_file(selection_output)}
        report.update({
            "completed": True, "terminal_status": "success", "rows": ROWS,
            "natural_rows": FRESH_ROWS, "natural_groups": FRESH_GROUPS,
            "preservation_rows": PRESERVATION_ROWS, "runtime_version": RUNTIME_VERSION,
            "model_artifact_hashes": runtime["accepted_weight_hashes"],
            "file_hashes": file_hashes, "source_hashes": before,
            "source_hashes_after": after, "capture_resources": terminal,
            "fresh_source_schema": {
                "input": "romanize_word(text)", "context": "left_context",
                "expected": "singleton original text", "text_consulted_after_training_gate": True,
                "whole_groups_revalidated": True,
            },
            "preservation_reused_without_new_parent_prediction": True,
            "full_candidate_objects_preserved": True, "raw_bypass_preserved": True,
            "elapsed_seconds": time.monotonic() - started,
        })
        report_path.write_text(json.dumps(report, ensure_ascii=False,
                                          sort_keys=True, indent=2) + "\n")
        require_resources(started, "terminal_report_written")
        return report
    except BaseException as error:
        report.update({"completed": False, "terminal_status": "failed",
                       "failure_type": type(error).__name__, "failure": str(error),
                       "elapsed_seconds": time.monotonic() - started})
        if isinstance(error, CaptureResourceError):
            report["failure_resources"] = error.record
        raise
    finally:
        if report.get("terminal_status") != "success":
            converter = None
            gc.collect()
            write_failed_report(report_path, report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-proof", type=Path, required=True)
    parser.add_argument("--training-proof-sha256", required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"completed": report["completed"], "rows": report["rows"],
                      "output": str(args.output)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
