#!/usr/bin/env python3
"""Compare a verified supported transducer with frozen production TRAIN diagnostics."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import resource
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hanglish.core.neural_converter import PROTECTED, TOKENS, file_sha256
from hanglish.models.phonetic_transducer import PhoneticTransducer
from scripts.audit_final_choice_candidate_stages import StageConverter
from scripts.final_choice_capture import _response_flags, validate_response


SOURCE = ROOT / ".omx/artifacts/semantic-pair-training-input-v2"
STAGES = ROOT / ".omx/artifacts/final-choice-candidate-stages-v1"
TARGETS = ROOT / "data/natural_context/final-choice-v1/natural_fit"
SOURCE_ROWS_SHA = "6a1f7f61a7f7ffeb9209772559412931137d71ef0173d7429264d9cb53aed396"
SOURCE_MANIFEST_SHA = "3ab561d1a767e7a65b4706c2cd0f668366693536849e17f2f3e620fbcf5550a6"
STAGE_CASES_SHA = "860da6746310eee896008280a8ad6eac72736b35db16a289aa02a9a5f45264c6"
STAGE_REPORT_SHA = "528f4fbc1ebb2bbc9e22e18b85b4af6e0a205a9a5fe1a64bbe266b52ba46b615"
TARGET_CASES_SHA = "789e45a3b6028e7416ca3b82739c906c558ec414320796c53b0b76ed251b9a18"
TARGET_MANIFEST_SHA = "fc22b1635cc496ed70aa757750944726d583d640f2d9f3a598e380f3e262a1c9"
LEGACY_TRAINING_REPORT_SHA = "7e001e0c405fa5afb2714c0db1c769acb9427c736b48061f292f4ccfebd57389"
ROWS = 512
PRODUCTION_RUNTIME_VERSION = "neural-1-867f13596ed5"
TIME_LIMIT = 900
RSS_LIMIT = 3 * 1024**3
RUNTIME_FILES = (
    "hanglish/core/lattice.py", "hanglish/core/lexical_data.py",
    "hanglish/core/neural_converter.py", "hanglish/core/phonology.py",
    "hanglish/models/phonetic_transducer.py",
    "scripts/audit_final_choice_candidate_stages.py",
    "scripts/final_choice_capture.py", "scripts/probe_supported_phonetic_runtime.py",
)
FORBIDDEN_SOURCE_FIELDS = frozenset({"expected", "accepted", "answers", "labels", "target"})


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def path_key(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(Path(path).resolve())


def checked_bytes(path: Path, expected: str) -> bytes:
    raw = Path(path).read_bytes()
    if sha256_bytes(raw) != expected:
        raise ValueError(f"pinned file differs: {path}")
    return raw


def parse_jsonl(raw: bytes) -> list[dict]:
    return [json.loads(line) for line in raw.splitlines() if line]


def rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def check_resources(started, clock=time.monotonic, rss_reader=rss_bytes):
    elapsed, peak = clock() - started, rss_reader()
    if elapsed > TIME_LIMIT:
        raise RuntimeError("supported runtime probe exceeded 900 seconds")
    if peak > RSS_LIMIT:
        raise MemoryError("supported runtime probe exceeded 3 GiB RSS")
    return elapsed, peak


def verify_pins(pins):
    for name, expected in pins.items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"runtime probe source changed: {name}")


def validate_independent_training(checkpoint: Path, verification: Path,
                                  verification_sha: str) -> dict:
    if not isinstance(verification_sha, str) or len(verification_sha) != 64:
        raise ValueError("independent verification SHA-256 is required")
    verification_raw = checked_bytes(verification, verification_sha)
    proof = json.loads(verification_raw)
    report_path = checkpoint.parent / "report.json"
    metadata_path = checkpoint.with_suffix(".json")
    report_raw = checked_bytes(report_path, proof.get("report_sha256"))
    report = json.loads(report_raw)
    report_sha = sha256_bytes(report_raw)
    accepted_report_schema = (
        report.get("schema") == "supported-phonetic-transducer-training-v1"
        or (report_sha == LEGACY_TRAINING_REPORT_SHA
            and report.get("schema") == "supported-phonetic-transducer-checkpoint-v1")
    )
    if (proof.get("schema") not in {
                "supported-phonetic-transducer-independent-v1",
                "supported-phonetic-transducer-independent-v2",
            }
            or proof.get("completed") is not True
            or proof.get("production_changed") is not False
            or proof.get("fresh_replayed_inputs") != 3292
            or not accepted_report_schema
            or report.get("completed") is not True
            or report.get("training_run") is not True
            or report.get("checkpoint") != checkpoint.name
            or proof.get("checkpoint_sha256") != report.get("checkpoint_sha256")
            or proof.get("predictions_sha256") != report.get("predictions_sha256")):
        raise ValueError("independent supported-checkpoint verification is incomplete")
    checked_bytes(checkpoint, report["checkpoint_sha256"])
    checked_bytes(metadata_path, report["metadata_sha256"])
    checked_bytes(checkpoint.parent / "held_predictions.jsonl", report["predictions_sha256"])
    verifier_path = ROOT / "scripts/verify_supported_phonetic_transducer.py"
    if proof.get("verifier_sha256") != file_sha256(verifier_path):
        raise ValueError("independent verifier source differs")
    return {
        "proof": proof, "training_report": report,
        "training_report_schema": report.get("schema"),
        "legacy_training_report_schema_collision": report_sha == LEGACY_TRAINING_REPORT_SHA,
        "pins": {
            path_key(checkpoint): report["checkpoint_sha256"],
            path_key(metadata_path): report["metadata_sha256"],
            path_key(report_path): proof["report_sha256"],
            path_key(verification): verification_sha,
            path_key(verifier_path): proof["verifier_sha256"],
            path_key(checkpoint.parent / "held_predictions.jsonl"): report["predictions_sha256"],
        },
    }


def load_source_rows() -> tuple[list[dict], dict]:
    manifest_raw = checked_bytes(SOURCE / "manifest.json", SOURCE_MANIFEST_SHA)
    rows_raw = checked_bytes(SOURCE / "rows.jsonl", SOURCE_ROWS_SHA)
    manifest = json.loads(manifest_raw)
    rows = []
    for row in parse_jsonl(rows_raw):
        if row.get("source_role") != "natural_fit":
            continue
        if FORBIDDEN_SOURCE_FIELDS.intersection(row):
            raise ValueError("source-only row contains labels")
        rows.append(row)
        if len(rows) == ROWS:
            break
    if (manifest.get("completed") is not True
            or manifest.get("rows_sha256") != SOURCE_ROWS_SHA
            or manifest.get("labels_or_expected_saved") is not False
            or len(rows) != ROWS
            or [row.get("index") for row in rows] != list(range(ROWS))):
        raise ValueError("source-only first512 differs")
    for row in rows:
        finite = row.get("finite_candidates")
        if (row.get("finite_candidate_indices") != [0, 1, 2, 3]
                or not isinstance(finite, list) or len(finite) != 4
                or [item.get("source_index") for item in finite] != [0, 1, 2, 3]):
            raise ValueError("baseline source candidates differ")
    return rows, manifest


def baseline_candidates(row):
    candidates = [deepcopy(value["object"]) for value in row["finite_candidates"]]
    source = row["input"]
    candidates.append({
        "text": source, "score": None,
        "segments": [{"input_start": 0, "input_end": len(source), "source": source,
                      "text": source, "kind": "raw"}],
    })
    return candidates


def experiment_identity(base_version, artifacts):
    digest = hashlib.sha256(json.dumps(
        {"base_version": base_version, "artifacts": artifacts},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    return "supported-phonetic-runtime-1-" + digest[:12]


def install_supported_transducer(converter, model, checkpoint, metadata):
    base_version = converter.version
    artifacts = dict(converter.model_artifacts)
    artifacts.pop("checkpoints/phonetic_transducer.pt", None)
    artifacts.pop("checkpoints/phonetic_transducer.json", None)
    artifacts[path_key(checkpoint)] = file_sha256(checkpoint)
    artifacts[path_key(metadata)] = file_sha256(metadata)
    converter.transducer = model
    converter.word_options.cache_clear()
    converter.model_artifacts = dict(sorted(artifacts.items()))
    converter.version = experiment_identity(base_version, converter.model_artifacts)
    return base_version


class SupportedPhoneticRuntime(StageConverter):
    def configure_supported_checkpoint(self, checkpoint, proof_path):
        self.supported_checkpoint = Path(checkpoint)
        self.supported_proof_path = Path(proof_path)
        self.supported_installed = False

    def load_models(self):
        if getattr(self, "supported_installed", False):
            return
        super().load_models()
        self.production_model_artifacts = deepcopy(self.model_artifacts)
        model = PhoneticTransducer.load(self.supported_checkpoint, device="cpu")
        self.production_runtime_version = install_supported_transducer(
            self, model, self.supported_checkpoint,
            self.supported_checkpoint.with_suffix(".json"),
        )
        self.supported_installed = True


def serialized_pieces(pieces):
    return [[source, offset, [asdict(option) for option in options]]
            for source, offset, options in pieces]


def piece_paths(pieces, limit=24):
    paths = [("", 0.0)]
    for _, _, options in pieces:
        expanded = []
        for prefix, score in paths:
            for option in options[:8]:
                text = option["text"] if isinstance(option, dict) else option.text
                option_score = option.get("score", 0.0) if isinstance(option, dict) else option.score
                expanded.append((prefix + text, score + option_score))
        unique = {}
        for item in sorted(expanded, key=lambda item: (-item[1], item[0])):
            unique.setdefault(item[0], item)
        paths = list(unique.values())[:limit]
    return [text for text, _ in paths]


def protected_input(text):
    return any(not match.group().isspace() and PROTECTED.fullmatch(match.group())
               for match in TOKENS.finditer(text))


def preserved_segments(candidate):
    for segment in candidate["segments"]:
        source = segment["source"]
        if (segment["kind"] in {"english", "raw"} or PROTECTED.fullmatch(source)) \
                and segment["text"] != source:
            return False
    return True


def has_preserved_span(candidates):
    return any(
        segment["kind"] == "english" or PROTECTED.fullmatch(segment["source"])
        for candidate in candidates for segment in candidate["segments"]
    )


def capture_sources(converter, rows, capture_path, runtime_version, model_artifacts,
                    *, guard=lambda: None):
    captures = []
    with Path(capture_path).open("x") as stream:
        for index, source in enumerate(rows):
            response = converter.convert(source["input"], source["context"])
            pieces = serialized_pieces(converter.audit_pieces)
            learned_beam = piece_paths(pieces)
            raw = converter.convert(source["input"], source["context"], raw=True)
            validate_response(response, source["input"], source["context"],
                              runtime_version, model_artifacts, raw=False)
            validate_response(raw, source["input"], source["context"],
                              runtime_version, model_artifacts, raw=True)
            if not all(preserved_segments(candidate) for candidate in response["candidates"]):
                raise ValueError("new runtime changed a protected/English/raw segment")
            captured = {
                "index": index, "id": source["id"], "group_id": source["group_id"],
                "input": source["input"], "context": source["context"],
                "baseline_result": source["current_result"],
                "baseline_candidates": baseline_candidates(source),
                "new_result": response["result"],
                "new_candidates": deepcopy(response["candidates"]),
                "new_response_flags": _response_flags(response),
                "new_engine_ms": response["engine_ms"],
                "new_learned_pieces": pieces, "new_learned_beam": learned_beam,
                "raw_response": deepcopy(raw),
                "protected_input": protected_input(source["input"])
                    or has_preserved_span(baseline_candidates(source)),
                "protected_segments_preserved": True,
            }
            captures.append(captured)
            stream.write(json.dumps(captured, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")) + "\n")
            stream.flush()
            guard()
    return captures


def percentile(values, quantile):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = math.ceil(quantile * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def summarize(captures, targets, stage_rows, stage_report):
    outcomes = []
    for captured, target, stage in zip(captures, targets, stage_rows):
        identity = (captured["id"], captured["group_id"], captured["input"], captured["context"])
        if identity != (target["id"], target["group_id"], target["input"], target["context"]):
            raise ValueError("target identity differs from source-only capture")
        if identity != (stage["id"], stage["group_id"], stage["input"], stage["context"]):
            raise ValueError("frozen stage identity differs")
        if (captured["baseline_result"] != target["result"]
                or captured["baseline_candidates"] != target["candidates"]
                or captured["baseline_result"] != stage["result"]):
            raise ValueError("label-free baseline differs from frozen production capture")
        expected = target["expected"]
        before_texts = [value["text"] for value in captured["baseline_candidates"]]
        new_texts = [value["text"] for value in captured["new_candidates"]]
        before_top1 = captured["baseline_result"] in expected
        top1 = captured["new_result"] in expected
        before_top5 = any(value in expected for value in before_texts)
        top5 = any(value in expected for value in new_texts)
        before_beam = any(value in expected for value in piece_paths(stage["learned_pieces"]))
        beam = any(value in expected for value in captured["new_learned_beam"])
        outcomes.append({
            "id": captured["id"], "expected": expected,
            "before_result": captured["baseline_result"], "result": captured["new_result"],
            "before_top1": before_top1, "top1": top1,
            "before_top5": before_top5, "top5": top5,
            "before_beam_covered": before_beam, "beam_covered": beam,
            "protected_input": captured["protected_input"],
        })
    before_top1 = sum(row["before_top1"] for row in outcomes)
    top1 = sum(row["top1"] for row in outcomes)
    before_top5 = sum(row["before_top5"] for row in outcomes)
    top5 = sum(row["top5"] for row in outcomes)
    before_beam = sum(row["before_beam_covered"] for row in outcomes)
    beam = sum(row["beam_covered"] for row in outcomes)
    if before_top1 != stage_report["top1"] or before_top5 != stage_report["top5"]:
        raise ValueError("recomputed production metrics differ from frozen stage report")
    protected = [row for row in captures if row["protected_input"]]
    return {
        "counts": {
            "rows": len(outcomes), "before_top1": before_top1, "top1": top1,
            "before_top5": before_top5, "top5": top5,
            "before_beam_coverage": before_beam, "beam_coverage": beam,
        },
        "correction_ids": [row["id"] for row in outcomes if not row["before_top1"] and row["top1"]],
        "regression_ids": [row["id"] for row in outcomes if row["before_top1"] and not row["top1"]],
        "top5_gain_ids": [row["id"] for row in outcomes if not row["before_top5"] and row["top5"]],
        "top5_loss_ids": [row["id"] for row in outcomes if row["before_top5"] and not row["top5"]],
        "beam_gain_ids": [row["id"] for row in outcomes if not row["before_beam_covered"] and row["beam_covered"]],
        "beam_loss_ids": [row["id"] for row in outcomes if row["before_beam_covered"] and not row["beam_covered"]],
        "protected": {
            "rows": len(protected),
            "classification_limit": (
                "Rows include mixed inputs with English, raw, URL, code, or numeric protected "
                "segments. Result/candidate changes can come from other convertible segments."
            ),
            "result_changed_ids": [row["id"] for row in protected
                                   if row["baseline_result"] != row["new_result"]],
            "candidate_changed_ids": [row["id"] for row in protected
                                      if row["baseline_candidates"] != row["new_candidates"]],
            "candidate_text_changed_ids": [row["id"] for row in protected
                                           if [value["text"] for value in row["baseline_candidates"]]
                                           != [value["text"] for value in row["new_candidates"]]],
            "segments_preserved_all": all(row["protected_segments_preserved"]
                                          for row in protected),
        },
        "outcomes": outcomes,
    }


def merge_pins(*mappings):
    pins = {}
    for mapping in mappings:
        for name, digest in mapping.items():
            if name in pins and pins[name] != digest:
                raise ValueError(f"conflicting source pin: {name}")
            pins[name] = digest
    return dict(sorted(pins.items()))


def source_pins(proof_pins, source_manifest, stage_report=None):
    direct = {
        path_key(SOURCE / "rows.jsonl"): SOURCE_ROWS_SHA,
        path_key(SOURCE / "manifest.json"): SOURCE_MANIFEST_SHA,
        **{name: file_sha256(ROOT / name) for name in RUNTIME_FILES},
    }
    inherited = source_manifest.get("source_hashes", {})
    if stage_report is None:
        # The semantic manifest inherits FIT label artifacts. Its own pinned
        # rows/manifest are sufficient for the source-only phase; inherited
        # closure is verified only after the capture file has been closed.
        return merge_pins(proof_pins, direct)
    label_pins = {
        path_key(STAGES / "cases.jsonl"): STAGE_CASES_SHA,
        path_key(STAGES / "report.json"): STAGE_REPORT_SHA,
        path_key(TARGETS / "cases.jsonl"): TARGET_CASES_SHA,
        path_key(TARGETS / "manifest.json"): TARGET_MANIFEST_SHA,
    }
    return merge_pins(proof_pins, direct, inherited, label_pins,
                      stage_report["source_hashes"])


def finalize_success(output, report, guard):
    report_path = Path(output) / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    )
    try:
        guard()
    except Exception:
        report_path.unlink(missing_ok=True)
        raise
    (Path(output) / "status.json").write_text(json.dumps({
        "state": "complete", "phase": "complete",
        "report_sha256": file_sha256(report_path),
    }, ensure_ascii=False, sort_keys=True) + "\n")


def run(output: Path, checkpoint: Path, verification: Path, verification_sha: str,
        *, converter_factory=SupportedPhoneticRuntime, clock=time.monotonic,
        rss_reader=rss_bytes):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = clock()
    inference_started = False
    try:
        proof = validate_independent_training(Path(checkpoint), Path(verification), verification_sha)
        sources, source_manifest = load_source_rows()
        pre_capture_pins = source_pins(proof["pins"], source_manifest)
        verify_pins(pre_capture_pins)
        import torch
        torch.set_num_threads(2)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            if torch.get_num_interop_threads() != 1:
                raise
        converter = converter_factory(ranker_kind="hybrid", external_lexicon=True)
        converter.configure_supported_checkpoint(checkpoint, verification)
        converter.load_models()
        check_resources(started, clock, rss_reader)
        if (converter.production_runtime_version != PRODUCTION_RUNTIME_VERSION
                or converter.version == converter.production_runtime_version):
            raise ValueError("experimental runtime identity differs")
        write_status = lambda state, **values: (output / "status.json").write_text(
            json.dumps({"state": state, **values}, ensure_ascii=False, sort_keys=True) + "\n")
        write_status("running", phase="capture")
        capture_path = output / "capture.jsonl"
        inference_started = True
        captures = capture_sources(
            converter, sources, capture_path, converter.version, converter.model_artifacts,
            guard=lambda: check_resources(started, clock, rss_reader),
        )
        capture_sha = file_sha256(capture_path)
        # Labels and frozen label-bearing stage rows are opened only after capture is closed.
        stage_report = json.loads(checked_bytes(STAGES / "report.json", STAGE_REPORT_SHA))
        target_manifest = json.loads(checked_bytes(TARGETS / "manifest.json", TARGET_MANIFEST_SHA))
        if (stage_report.get("completed") is not True
                or stage_report.get("runtime_version") != PRODUCTION_RUNTIME_VERSION
                or stage_report.get("profile", {}).get("rows") != ROWS
                or target_manifest.get("runtime_version") != PRODUCTION_RUNTIME_VERSION
                or target_manifest.get("model_artifact_hashes")
                != converter.production_model_artifacts):
            raise ValueError("frozen production runtime identity differs")
        pins = source_pins(proof["pins"], source_manifest, stage_report)
        verify_pins(pins)
        targets = parse_jsonl(checked_bytes(TARGETS / "cases.jsonl", TARGET_CASES_SHA))[:ROWS]
        stage_rows = parse_jsonl(checked_bytes(STAGES / "cases.jsonl", STAGE_CASES_SHA))
        if len(targets) != ROWS or len(stage_rows) != ROWS:
            raise ValueError("label join denominator differs")
        summary = summarize(captures, targets, stage_rows, stage_report)
        verify_pins(pins)
        elapsed, peak = check_resources(started, clock, rss_reader)
        baseline_ms = [row["engine_ms"] for row in targets]
        new_ms = [row["new_engine_ms"] for row in captures]
        report = {
            "schema": "supported-phonetic-runtime-probe-v1", "completed": True,
            "scope": "natural FIT TRAIN first512 diagnostic; no new DEV/holdout data",
            "runtime_version": converter.version,
            "production_runtime_version": converter.production_runtime_version,
            "model_artifacts": converter.model_artifacts,
            "source_hashes": pins, "source_hashes_after": pins,
            "capture_sha256": capture_sha, **{key: value for key, value in summary.items()
                                                if key != "outcomes"},
            "latency_ms": {
                "baseline_p50": statistics.median(baseline_ms),
                "baseline_p95": percentile(baseline_ms, .95),
                "new_p50": statistics.median(new_ms), "new_p95": percentile(new_ms, .95),
                "comparison_limit": "Baseline and new timings were captured in different runs; no speed claim is made.",
            },
            "raw_preserved_all": all(
                row["raw_response"]["result"] == row["input"]
                and row["raw_response"]["candidates"][0]["text"] == row["input"]
                for row in captures
            ),
            "labels_parsed_after_closed_capture": True,
            "independent_training_verification": {
                "path": path_key(verification), "sha256": verification_sha,
                "fresh_replayed_inputs": proof["proof"]["fresh_replayed_inputs"],
                "original_training_report_schema": proof["training_report_schema"],
                "legacy_training_report_schema_collision":
                    proof["legacy_training_report_schema_collision"],
            },
            "actual_model_inference_run": True, "production_changed": False,
            "development_data_read": False, "reserve_data_read": False,
            "final_data_read": False, "elapsed_seconds": elapsed, "peak_rss_bytes": peak,
        }
        (output / "outcomes.jsonl").write_bytes(b"".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")).encode() + b"\n"
            for row in summary["outcomes"]
        ))
        report["outcomes_sha256"] = file_sha256(output / "outcomes.jsonl")
        finalize_success(
            output, report, lambda: check_resources(started, clock, rss_reader)
        )
        return report
    except Exception as error:
        (output / "report.json").unlink(missing_ok=True)
        (output / "status.json").unlink(missing_ok=True)
        (output / "failure.json").write_text(json.dumps({
            "schema": "supported-phonetic-runtime-probe-failure-v1",
            "error_type": type(error).__name__, "error": str(error),
            "actual_model_inference_run": inference_started,
            "elapsed_seconds": clock() - started, "peak_rss_bytes": rss_reader(),
        }, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--verification-sha", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.checkpoint, args.verification,
                         args.verification_sha), ensure_ascii=False, sort_keys=True))
