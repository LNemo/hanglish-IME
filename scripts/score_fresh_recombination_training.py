"""Score fresh latest-runtime TRAIN recombination pools with a frozen parent ranker."""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import zip_longest
import json
import os
from pathlib import Path
import signal
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import score_recombination_training_pools as base


SCHEMA = "recombination-fresh-training-parent-census-v1"
CAPTURE_SCHEMA = "recombination-fresh-runtime-capture-v1"
PROOF_SCHEMA = "fresh-recombination-training-independent-verification-v1"
EXPECTED_ROWS = 2048
EXPECTED_PARTITION = {"fit": 1642, "inner_hold": 406}
PLAN = ROOT / ".omx/plans/recombination-fresh-census-v1.md"
SELECTION = ROOT / ".omx/artifacts/recombination-fresh-train-sources-v1"
SELECTION_HASHES = {
    "manifest.json": "5680aa160f99d9656b02e52e06d4218f0ca386848d7b733fdf03d2570acfa696",
    "selection.jsonl": "15a0617e95d5c2fb0ec7ce895230794a9ac9d27f185b288eb3d138e0f91968d2",
    "independent-verification.json": "5aa1c088cfe84c30581325b934098e41db568fd14824153c56c61d9e0519a492",
}
RUNTIME_REFERENCE = ROOT / ".omx/artifacts/extended-runtime-dev-v2/report.json"
RUNTIME_REFERENCE_SHA = "64fa054990b0dc78bde0bb20e5de0ee89fa7ae910ff8fc281cb16bf607387227"
EXPECTED_RUNTIME_VERSION = "experimental-expanded-pool-256-1-1b1ff433b3cf"
CAPTURE_PRODUCER = ROOT / "scripts/capture_fresh_recombination_training.py"
CAPTURE_VERIFIER = ROOT / ".omx/artifacts/verify-fresh-recombination-training-v1.py"
CAPTURE_VERIFIER_SHA = "f8040bbc699205d3336eab8e4861853ffaf1bcb1897847e2cdae2e44ed002678"
# Filled only after the latest-runtime capture and independent verification finish.
EXPECTED_CAPTURE_HASHES = {
    "report.json": "d890bad55cbd52f3168e8a8fa7f9caefc88be448dc35af011e4e386e7871212d",
    "runtime-capture.jsonl": "92f7ca44ae9b1302ffe87fec30fda3ae6849284c560530d3c25de21fd66e5a69",
    "source-capture.jsonl": "df9250b8f422f7aecdf255127e677bb30471957a888c74b6e713aab92a7d07ad",
    "capture.complete.json": "9a291245e3e43d8c44dcef70952c9e41e09a7f0174d3e3b55acf055e57d5c511",
    "prepared.jsonl": "340f7611ff9f247e80b06dcbd48593b0c391572a0435f6ce7db477e789f6809e",
    "pins.json": "dad031203a43cb80ef902e416560803065a348a76aa4c8111237d75ea2f718d7",
    "independent-verification.json": "d01213d92b20523eb29646cbb4b460342e91bd7c10d782b54ce03ff1ef054ca6"
}
LIMITS = {"whole_seconds": 1800.0, "update_seconds": 60.0,
          "rss_bytes": 3 * 1024**3, "swap_growth_bytes": 512 * 1024**2}


def selection_authorization() -> dict:
    paths = {name: SELECTION / name for name in SELECTION_HASHES}
    for name, digest in SELECTION_HASHES.items():
        base.require(paths[name].is_file() and base.sha256_file(paths[name]) == digest,
                     f"fresh TRAIN selection differs: {name}")
    manifest = json.loads(paths["manifest.json"].read_text(encoding="utf-8"))
    proof = json.loads(paths["independent-verification.json"].read_text(encoding="utf-8"))
    partition = manifest.get("partition")
    base.require(manifest.get("schema") == "recombination-fresh-train-sources-v1"
                 and manifest.get("completed") is True
                 and manifest.get("counts", {}).get("rows") == EXPECTED_ROWS
                 and isinstance(partition, list) and len(partition) == EXPECTED_ROWS
                 and dict(Counter(row.get("split") for row in partition))
                    == EXPECTED_PARTITION
                 and manifest.get("model_loaded") is False
                 and manifest.get("model_inference") is False
                 and manifest.get("dev_accessed") is False
                 and manifest.get("test_accessed") is False
                 and manifest.get("production_changed") is False,
                 "fresh TRAIN selection manifest contract differs")
    selection_rows = list(base.iter_jsonl(paths["selection.jsonl"]))
    base.require(len(selection_rows) == EXPECTED_ROWS
                 and [row.get("id") for row in selection_rows]
                    == [row.get("id") for row in partition]
                 and all(set(row) == {"id", "input"}
                         and isinstance(row["id"], str)
                         and isinstance(row["input"], str) for row in selection_rows),
                 "fresh TRAIN selection source projection differs")
    contracts = proof.get("contracts", {})
    base.require(proof.get("schema")
                    == "recombination-fresh-train-sources-independent-verification-v1"
                 and proof.get("verified") is True
                 and proof.get("manifest_sha256") == SELECTION_HASHES["manifest.json"]
                 and proof.get("selection_sha256") == SELECTION_HASHES["selection.jsonl"]
                 and proof.get("rows") == EXPECTED_ROWS
                 and proof.get("partition_counts") == EXPECTED_PARTITION
                 and contracts.get("prediction_input_fields") == ["id", "input"]
                 and contracts.get("old_components_excluded") is True
                 and contracts.get("selection_uses_accuracy") is False
                 and contracts.get("model_loaded") is False
                 and contracts.get("dev_opened") is False
                 and contracts.get("test_opened") is False
                 and contracts.get("production_changed") is False,
                 "fresh TRAIN selection proof differs")
    return {"paths": paths, "manifest": manifest, "proof": proof,
            "selection": selection_rows,
            "partition": {row["id"]: (row["component_id"], row["split"])
                          for row in partition}}


def validate_inputs(data_dir: Path, proof_path: Path) -> dict:
    base.require(all(value != "PENDING" for value in EXPECTED_CAPTURE_HASHES.values()),
                 "fresh runtime capture is not frozen and independently verified")
    selection = selection_authorization()
    expected_model_pins = {
        base.path_key(base.CHECKPOINT / name): digest
        for name, digest in base.EXPECTED_MODEL.items()}
    data_dir = Path(data_dir)
    paths = {name: data_dir / name for name in EXPECTED_CAPTURE_HASHES
             if name != "independent-verification.json"}
    paths["independent-verification.json"] = Path(proof_path)
    for name, digest in EXPECTED_CAPTURE_HASHES.items():
        base.require(paths[name].is_file() and base.sha256_file(paths[name]) == digest,
                     f"fresh runtime capture identity differs: {name}")
    report = json.loads(paths["report.json"].read_text(encoding="utf-8"))
    proof = json.loads(paths["independent-verification.json"].read_text(encoding="utf-8"))
    outputs = report.get("outputs", {})
    base.require(report.get("schema") == CAPTURE_SCHEMA
                 and report.get("completed") is True
                 and report.get("input_contract") == "roman_sentence_only"
                 and report.get("rows") == EXPECTED_ROWS
                 and report.get("prepared_rows") == EXPECTED_ROWS
                 and report.get("partition_counts") == EXPECTED_PARTITION
                 and report.get("runtime_version") == EXPECTED_RUNTIME_VERSION
                 and report.get("source_only_capture") is True
                 and report.get("targets_joined_after_closed_capture") is True
                 and report.get("labels_opened_after_capture") is True
                 and report.get("single_request_conversion") is True
                 and report.get("typed_pipeline_captured") is True
                 and report.get("model_loaded") is True
                 and report.get("model_inference") is True
                 and report.get("dev_accessed") is False
                 and report.get("test_accessed") is False
                 and report.get("production_changed") is False,
                 "fresh runtime capture report contract differs")
    base.require(report.get("runtime_reference") == {
                     "path": base.path_key(RUNTIME_REFERENCE),
                     "sha256": RUNTIME_REFERENCE_SHA}
                 and base.sha256_file(RUNTIME_REFERENCE) == RUNTIME_REFERENCE_SHA,
                 "fresh runtime reference differs")
    for name in ("runtime-capture.jsonl", "source-capture.jsonl",
                 "capture.complete.json", "prepared.jsonl"):
        detail = outputs.get(name, {})
        base.require(detail.get("rows") == EXPECTED_ROWS
                     and detail.get("sha256") == EXPECTED_CAPTURE_HASHES[name],
                     f"fresh runtime output binding differs: {name}")
    contracts = proof.get("contracts", {})
    base.require(proof.get("schema") == PROOF_SCHEMA and proof.get("verified") is True
                 and proof.get("report_sha256") == EXPECTED_CAPTURE_HASHES["report.json"]
                 and proof.get("runtime_capture_sha256")
                    == EXPECTED_CAPTURE_HASHES["runtime-capture.jsonl"]
                 and proof.get("source_capture_sha256")
                    == EXPECTED_CAPTURE_HASHES["source-capture.jsonl"]
                 and proof.get("closure_sha256")
                    == EXPECTED_CAPTURE_HASHES["capture.complete.json"]
                 and proof.get("prepared_sha256")
                    == EXPECTED_CAPTURE_HASHES["prepared.jsonl"]
                 and proof.get("source_selection_manifest_sha256")
                    == SELECTION_HASHES["manifest.json"]
                 and proof.get("source_selection_sha256")
                    == SELECTION_HASHES["selection.jsonl"]
                 and proof.get("source_selection_proof_sha256")
                    == SELECTION_HASHES["independent-verification.json"]
                 and proof.get("partition_counts") == EXPECTED_PARTITION
                 and proof.get("counts", {}).get("rows") == EXPECTED_ROWS
                 and proof.get("model_pins") == report.get("model_pins")
                 and all(proof["model_pins"].get(path) == digest
                         for path, digest in expected_model_pins.items())
                 and proof.get("reference_report_sha256") == RUNTIME_REFERENCE_SHA
                 and proof.get("producer_sha256") == base.sha256_file(CAPTURE_PRODUCER)
                 and proof.get("verifier_sha256") == CAPTURE_VERIFIER_SHA
                 and base.sha256_file(CAPTURE_VERIFIER) == CAPTURE_VERIFIER_SHA
                 and contracts == {
                     "source_only_capture": True,
                     "capture_closed_before_labels": True,
                     "model_loaded_by_verifier": False,
                     "model_inference_by_verifier": False,
                     "dev_opened": False, "test_opened": False,
                     "production_changed": False,
                 }, "fresh runtime independent verification differs")
    source_rows = validate_prepared_sources(
        paths["prepared.jsonl"], selection, expected_rows=EXPECTED_ROWS,
        expected_partition=EXPECTED_PARTITION)
    pins = {base.path_key(path): EXPECTED_CAPTURE_HASHES[name]
            for name, path in paths.items()}
    pins.update({base.path_key(path): SELECTION_HASHES[name]
                 for name, path in selection["paths"].items()})
    pins[base.path_key(RUNTIME_REFERENCE)] = RUNTIME_REFERENCE_SHA
    pins[base.path_key(CAPTURE_PRODUCER)] = proof["producer_sha256"]
    pins[base.path_key(CAPTURE_VERIFIER)] = CAPTURE_VERIFIER_SHA
    for raw, digest in report.get("pins", {}).items():
        path = Path(raw)
        path = path if path.is_absolute() else ROOT / path
        base.require(path.is_file() and base.sha256_file(path) == digest,
                     f"fresh runtime source pin differs: {raw}")
        pins[base.path_key(path)] = digest
    return {"paths": paths, "report": report, "proof": proof,
            "selection": selection, "source_rows": source_rows, "pins": pins}


def validate_prepared_sources(prepared_path: Path, selection: dict,
                              *, expected_rows: int = EXPECTED_ROWS,
                              expected_partition: dict = EXPECTED_PARTITION) -> list[dict]:
    """Project and bind source fields without retaining TRAIN labels."""
    source_rows = []
    prepared_rows = base.iter_jsonl(prepared_path)
    for prepared, selected in zip_longest(prepared_rows, selection["selection"]):
        base.require(prepared is not None and selected is not None,
                     "fresh prepared denominator differs")
        source = base.source_projection(prepared)
        row_partition = selection["partition"].get(source["id"])
        base.require({"id": source["id"], "input": source["input"]} == selected
                     and row_partition
                        == (source["component_id"], source["split"]),
                     "fresh prepared global partition differs")
        source_rows.append(source)
    base.require(len(source_rows) == expected_rows
                 and dict(Counter(row["split"] for row in source_rows))
                    == expected_partition,
                 "fresh prepared split count differs")
    return source_rows


def run(data_dir: Path, proof_path: Path, output: Path, *, ranker_factory=None) -> dict:
    authorization = validate_inputs(data_dir, proof_path)  # before Torch/model import
    from scripts import preflight_pko_seq2seq as preflight
    from hanglish.models.sentence_ranker import SentenceRanker
    import torch

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    baseline_swap = preflight.system_swap_bytes()
    captures = []
    pins = dict(authorization["pins"])
    tracked = [Path(__file__).resolve(), PLAN,
               ROOT / "scripts/score_recombination_training_pools.py",
               ROOT / "scripts/preflight_pko_seq2seq.py",
               ROOT / "hanglish/models/sentence_ranker.py"]
    pins.update({base.path_key(path): base.sha256_file(path) for path in tracked})
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)

    class FreshWatchdog(preflight.ResourceWatchdog):
        def _monitor(self):
            while not self.stop_event.wait(self.interval_s):
                try:
                    preflight.append_jsonl(self.trace, self._sample("watchdog"), self.lock)
                except BaseException as error:
                    self.failure = f"{type(error).__name__}: {error}"
                    if not (self.trace.parent / "report.json").exists():
                        preflight.write_json(self.trace.parent / "report.json", {
                            "schema": SCHEMA, "completed": False,
                            "error": self.failure, "captured_rows": len(captures),
                            "dev_accessed": False, "test_accessed": False,
                            "production_changed": False})
                    os.kill(os.getpid(), signal.SIGUSR1)
                    if not self.stop_event.wait(5.0):  # pragma: no cover
                        os._exit(124)
                    return

    try:
        with FreshWatchdog(started, baseline_swap, output / "resources.jsonl",
                           limits=LIMITS, interval_s=5.0) as watchdog:
            model_pins = base.frozen_model_pins()
            pins.update(model_pins)
            config = json.loads((base.CHECKPOINT / "config.json").read_text())
            metadata = json.loads((base.CHECKPOINT / "metadata.json").read_text())
            base.require(config.get("max_position_embeddings", 0) >= base.MAX_LENGTH
                         and metadata.get("max_length") == 96,
                         "sentence ranker length contract differs")
            with watchdog.operation("load_sentence_ranker", update=True):
                original_ranker = (ranker_factory or SentenceRanker.load)(base.CHECKPOINT)
                ranker = SentenceRanker(
                    original_ranker.model, original_ranker.tokenizer,
                    max_length=base.MAX_LENGTH, batch_size=original_ranker.batch_size,
                    metadata=original_ranker.metadata)
            base.require(ranker.model is original_ranker.model
                         and ranker.tokenizer is original_ranker.tokenizer
                         and ranker.max_length == base.MAX_LENGTH,
                         "256 wrapper does not share the frozen ranker")
            with (output / "capture.jsonl").open("x", encoding="utf-8") as stream:
                for source in authorization["source_rows"]:
                    with watchdog.operation(f"score:{source['id']}", update=True):
                        capture = base.score_row(source, ranker)
                    captures.append(capture)
                    stream.write(json.dumps(capture, ensure_ascii=False, sort_keys=True,
                                            allow_nan=False) + "\n")
                    stream.flush()
                    if len(captures) % 25 == 0:
                        print(json.dumps({"captured": len(captures),
                                          "rows": EXPECTED_ROWS,
                                          "elapsed_s": time.monotonic() - started}),
                              flush=True)
                os.fsync(stream.fileno())
            base.require(len(captures) == EXPECTED_ROWS, "fresh census denominator differs")
            capture_sha = base.sha256_file(output / "capture.jsonl")
            closure = {"schema": SCHEMA + "-source-closure",
                       "rows": EXPECTED_ROWS, "capture_sha256": capture_sha,
                       "source_only": True, "targets_joined": False,
                       "score_context": "", "input_truncation": False}
            base.write_json(output / "capture.complete.json", closure)
            prepared = authorization["paths"]["prepared.jsonl"]
            base.require(base.sha256_file(prepared)
                         == EXPECTED_CAPTURE_HASHES["prepared.jsonl"],
                         "fresh prepared labels changed before late join")
            label_rows = list(base.iter_jsonl(prepared))
            base.require(len(label_rows) == EXPECTED_ROWS,
                         "fresh TRAIN label denominator differs")
            metrics = {split: base.summarize(
                captures, authorization["source_rows"], label_rows, split)
                       for split in base.SPLITS}
            gate = base.adequacy_gate(metrics["inner_hold"])
            for path, digest in pins.items():
                base.require(Path(path).is_file() and base.sha256_file(Path(path)) == digest,
                             f"fresh census pin changed: {path}")
            final_resources = watchdog._sample("final")
        report = {
            "schema": SCHEMA, "completed": True, "scope": "TRAIN-only",
            "counts": {"rows": EXPECTED_ROWS, **EXPECTED_PARTITION},
            "metrics": metrics, "adequacy_gate": gate,
            "capture_sha256": capture_sha,
            "closure_sha256": base.sha256_file(output / "capture.complete.json"),
            "pins": dict(sorted(pins.items())), "model_pins": model_pins,
            "model": {"checkpoint": base.path_key(base.CHECKPOINT),
                      "tuned_max_length": 96,
                      "diagnostic_max_length": base.MAX_LENGTH,
                      "max_position_embeddings": config["max_position_embeddings"],
                      "shared_model_and_tokenizer": True,
                      "score_context": "", "input_truncation": False},
            "resources": {"limits": LIMITS,
                          "baseline_swap_bytes": baseline_swap,
                          "final": final_resources,
                          "trace_sha256": base.sha256_file(output / "resources.jsonl")},
            "targets_joined_after_closed_capture": True,
            "labels_never_passed_to_model": True,
            "capture_generation_model_inference": True,
            "ranker_training_run": False, "dev_accessed": False,
            "test_accessed": False, "quality_or_promotion_gate": False,
            "production_changed": False, "elapsed_s": time.monotonic() - started,
        }
        base.write_json(output / "report.json", report)
        return report
    except BaseException as error:
        if output.exists() and not (output / "report.json").exists():
            base.write_json(output / "report.json", {
                "schema": SCHEMA, "completed": False,
                "error": f"{type(error).__name__}: {error}",
                "captured_rows": len(captures), "dev_accessed": False,
                "test_accessed": False, "ranker_training_run": False,
                "production_changed": False})
        raise


def self_test() -> dict:
    base.require(EXPECTED_ROWS == sum(EXPECTED_PARTITION.values()),
                 "fresh partition total differs")
    return {"schema": SCHEMA, "self_test": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path)
    parser.add_argument("--proof", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), sort_keys=True))
    else:
        base.require(args.data is not None and args.proof is not None
                     and args.output is not None,
                     "--data, --proof, and --output are required")
        result = run(args.data, args.proof, args.output)
        print(json.dumps({"completed": result["completed"],
                          "adequacy_gate": result["adequacy_gate"],
                          "elapsed_s": result["elapsed_s"]}, sort_keys=True), flush=True)
