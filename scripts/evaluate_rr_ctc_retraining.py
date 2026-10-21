#!/usr/bin/env python3
"""Compare original, phonetic-only, and fully RR-retrained CTC runtimes.

Every arm runs in a fresh process and receives only source inputs. Expected
answers are joined after all three captures close. DEV and TEST are known
regression suites and cannot support a generalization claim or automatic model
promotion.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing
from pathlib import Path
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import evaluate_rr_retraining as rr  # noqa: E402
from scripts.source_span_resources import ResourceExceeded  # noqa: E402


OLD_PHONETIC_CHECKPOINT = rr.BASELINE_CHECKPOINT
OLD_WORD_CHECKPOINT = ROOT / ".omx/artifacts/sentence-ctc-pilot-v1/model.pt"
OLD_GROWTH_CHECKPOINT = ROOT / ".omx/artifacts/sentence-ctc-growth-pilot-v1/model.pt"
SCORE_SCALE_CHECKPOINT = rr.SCORE_SCALE_CHECKPOINT
ARM_ORDER = ("original", "phonetic_only", "all_retrained")
LIMITS = {
    "whole_seconds": 5400.0,
    "update_seconds": 60.0,
    "rss_bytes": 3 * 1024**3,
    "swap_growth_bytes": 1536 * 1024**2,
}
RESOURCE_RATIONALE = (
    "Each arm runs sequentially in a separate process with two Torch CPU threads.",
    "The 3 GiB RSS limit applies to each process; the 1.5 GiB swap-growth limit "
    "allows unrelated host-wide variation without increasing process memory.",
    "The 90-minute shared wall limit covers three full-engine arms without "
    "changing cases, scoring, checkpoints, or selection behavior.",
)
ADDITIONAL_RUNTIME_SOURCES = (
    "hanglish/core/ctc_word_proposals.py",
    "hanglish/core/mixed_span_converter.py",
    "hanglish/core/mixed_span_proposals.py",
    "hanglish/models/sentence_ctc.py",
    "hanglish/models/sentence_transducer.py",
)
CTC_RESPONSE_FIELDS = {
    "word_ctc": "ctc_word",
    "growth_ctc": "ctc_union",
}
CTC_COUNTERS = (
    "generation_requests", "completed_generations",
    "generated_proposals", "accepted_proposals",
)


def checkpoint_bundle(phonetic: Path, word: Path, growth: Path) -> dict:
    """Freeze the three independently replaceable model groups."""
    return {
        "phonetic": rr._checkpoint_files(Path(phonetic)),
        "word_ctc": rr._checkpoint_files(Path(word)),
        "growth_ctc": rr._checkpoint_files(Path(growth)),
    }


def flatten_bundle(bundle: dict) -> dict[str, str]:
    result = {}
    for group in ("phonetic", "word_ctc", "growth_ctc"):
        files = bundle.get(group)
        if not isinstance(files, dict) or not files:
            raise ValueError(f"missing checkpoint group: {group}")
        for path, digest in files.items():
            if path in result and result[path] != digest:
                raise ValueError(f"conflicting checkpoint identity: {path}")
            result[path] = digest
    return result


def production_converter_kwargs(phonetic: Path, word: Path, growth: Path) -> dict:
    """Mirror Workbench production defaults while overriding only checkpoints."""
    return {
        "score_scale_checkpoint": SCORE_SCALE_CHECKPOINT,
        "phonetic_checkpoint": Path(phonetic),
        "word_checkpoint": Path(word),
        "growth_checkpoint": Path(growth),
    }


def make_converter(phonetic: Path, word: Path, growth: Path, converter_class):
    # external_lexicon intentionally remains omitted: production defaults to True.
    return converter_class(**production_converter_kwargs(phonetic, word, growth))


def source_pins() -> dict[str, str]:
    paths = [Path(__file__), *(ROOT / name for name in rr.RUNTIME_SOURCES),
             *(ROOT / name for name in ADDITIONAL_RUNTIME_SOURCES)]
    unique = list(dict.fromkeys(path.resolve() for path in paths))
    missing = [str(path) for path in unique if not path.is_file()]
    if missing:
        raise ValueError("runtime source missing: " + ", ".join(missing))
    return {str(path): rr.sha256(path) for path in unique}


def _artifact_key(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT.resolve()))
    except ValueError:
        return "external/" + path.name


def intentional_artifact_keys(bundle: dict, groups: set[str]) -> set[str]:
    keys = set()
    for group in groups:
        for path in bundle[group]:
            keys.add(_artifact_key(Path(path)))
    return keys


def fixed_artifacts(identity: dict, ignored_keys: set[str]) -> dict:
    artifacts = identity.get("model_artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("runtime model artifact identity missing")
    return {key: value for key, value in artifacts.items()
            if key not in ignored_keys}


def _ctc_capture(response: dict, field: str) -> dict:
    value = response.get(field)
    if not isinstance(value, dict):
        raise ValueError(f"runtime response is missing {field}")
    counters = {name: value.get(name) for name in CTC_COUNTERS}
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0
           for item in counters.values()):
        raise ValueError(f"runtime response has invalid {field} counters")
    prediction = value.get("prediction")
    if prediction is not None and not isinstance(prediction, dict):
        raise ValueError(f"runtime response has invalid {field} prediction")
    return {**counters, "prediction": dict(prediction) if prediction is not None else None}


def capture_request(converter, item: dict, guard, *, timer=time.perf_counter) -> dict:
    """Capture runtime output plus both candidate-generator audit records."""
    started = timer()
    record = {"id": item["id"], "request": item}
    try:
        with guard.operation(f"predict:{item['id']}", update=True):
            response = converter.convert(
                item["input"], context=item["context"], engine="neural",
                mode="batch", raw=item["raw"])
            rr._validate_response(response, item)
            ctc = {name: _ctc_capture(response, field)
                   for name, field in CTC_RESPONSE_FIELDS.items()}
        record["response"] = {
            "result": response["result"],
            "candidates": response["candidates"],
            "version": response.get("version"),
            "engine_ms": response["engine_ms"],
            "raw": response.get("raw", False),
            **ctc,
        }
    except ResourceExceeded:
        raise
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
    record["elapsed_ms"] = (timer() - started) * 1000
    return record


def proposal_activity(captures: list[dict]) -> dict:
    """Aggregate auditable generator activity without consulting labels."""
    result = {}
    for group in CTC_RESPONSE_FIELDS:
        totals = {name: 0 for name in CTC_COUNTERS}
        reasons = {}
        predictions = 0
        requests_with_accepted_proposals = 0
        for row in captures:
            if "error" in row:
                continue
            value = row.get("response", {}).get(group)
            if not isinstance(value, dict):
                raise ValueError(f"capture is missing {group} audit metadata")
            for name in CTC_COUNTERS:
                item = value.get(name)
                if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                    raise ValueError(f"capture has invalid {group} counter: {name}")
                totals[name] += item
            prediction = value.get("prediction")
            if prediction is not None:
                if not isinstance(prediction, dict):
                    raise ValueError(f"capture has invalid {group} prediction")
                predictions += 1
                reason = prediction.get("reason", "missing_reason")
                if not isinstance(reason, str):
                    raise ValueError(f"capture has invalid {group} prediction reason")
                reasons[reason] = reasons.get(reason, 0) + 1
            requests_with_accepted_proposals += int(value["accepted_proposals"] > 0)
        result[group] = {
            **totals,
            "predictions_recorded": predictions,
            "prediction_reasons": dict(sorted(reasons.items())),
            "requests_with_accepted_proposals": requests_with_accepted_proposals,
        }
    return result


def proposal_difference(before: list[dict], after: list[dict]) -> dict:
    before_activity = proposal_activity(before)
    after_activity = proposal_activity(after)
    if len(before) != len(after):
        raise ValueError("proposal comparison row count mismatch")
    result = {}
    for group in CTC_RESPONSE_FIELDS:
        deltas = {
            name: after_activity[group][name] - before_activity[group][name]
            for name in (*CTC_COUNTERS, "predictions_recorded",
                         "requests_with_accepted_proposals")
        }
        changed_predictions = sum(
            before_row.get("response", {}).get(group, {}).get("prediction")
            != after_row.get("response", {}).get(group, {}).get("prediction")
            for before_row, after_row in zip(before, after)
        )
        result[group] = {"deltas": deltas,
                         "requests_with_changed_prediction": changed_predictions}
    return result


def verify_arm_pair(before_name: str, after_name: str, before_identity: dict,
                    after_identity: dict, before_rows: list[dict],
                    after_rows: list[dict], *, before_capture_sha256: str,
                    after_capture_sha256: str, allowed_groups: set[str]) -> None:
    if before_identity.get("input_sha256") != after_identity.get("input_sha256"):
        raise ValueError(f"{before_name}/{after_name} input hash mismatch")
    if before_identity.get("source_hashes") != after_identity.get("source_hashes"):
        raise ValueError(f"{before_name}/{after_name} runtime source hash mismatch")
    for name, identity, rows, digest in (
        (before_name, before_identity, before_rows, before_capture_sha256),
        (after_name, after_identity, after_rows, after_capture_sha256),
    ):
        if identity.get("rows") != len(rows):
            raise ValueError(f"{name} capture row count differs from identity")
        if identity.get("capture_sha256") != digest:
            raise ValueError(f"{name} capture hash differs from identity")
        flatten_bundle(identity.get("checkpoint_files", {}))
    if len(before_rows) != len(after_rows):
        raise ValueError(f"{before_name}/{after_name} row count mismatch")
    for before, after in zip(before_rows, after_rows):
        if before.get("id") != after.get("id") or before.get("request") != after.get("request"):
            raise ValueError(f"{before_name}/{after_name} cases differ")

    before_bundle = before_identity["checkpoint_files"]
    after_bundle = after_identity["checkpoint_files"]
    for group in {"phonetic", "word_ctc", "growth_ctc"} - allowed_groups:
        if before_bundle[group] != after_bundle[group]:
            raise ValueError(f"unexpected checkpoint change in {group}")
    ignored = (intentional_artifact_keys(before_bundle, allowed_groups)
               | intentional_artifact_keys(after_bundle, allowed_groups))
    if fixed_artifacts(before_identity, ignored) != fixed_artifacts(after_identity, ignored):
        raise ValueError("non-target runtime model artifacts differ between arms")


def _arm_capture(arm: str, phonetic: str, word: str, growth: str,
                 input_path: str, arm_dir: str, global_started: float,
                 baseline_swap: int) -> None:
    arm_path = Path(arm_dir)
    arm_path.mkdir(parents=False, exist_ok=False)
    try:
        import torch
        from hanglish.core.score_scale_converter import ScoreScaleConverter
        from scripts.source_span_resources import AtomicResourceWatchdog

        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        inputs = json.loads(Path(input_path).read_text())
        input_sha = rr.sha256(Path(input_path))
        sources = source_pins()
        checkpoints = checkpoint_bundle(Path(phonetic), Path(word), Path(growth))
        with AtomicResourceWatchdog(
                global_started, baseline_swap, arm_path / "resources.jsonl",
                limits=LIMITS, interval_s=0.5,
                failure_report=arm_path / "resource-failure.json") as guard:
            with guard.operation("load_models", update=True):
                converter = make_converter(
                    Path(phonetic), Path(word), Path(growth), ScoreScaleConverter)
                converter.load_models()
                converter.load_phonetic_model()
                converter.load_word_model()
                converter.load_growth_model()
                model_artifacts = dict(converter.model_artifacts)
                version = converter.version
            captures = []
            with (arm_path / "capture.jsonl").open("x") as stream:
                for index, item in enumerate(inputs):
                    record = capture_request(converter, item, guard)
                    captures.append(record)
                    rr.append_jsonl(stream, record)
                    if (index + 1) % 25 == 0 or index + 1 == len(inputs):
                        print(json.dumps({"arm": arm, "captured": index + 1,
                                          "total": len(inputs)}), flush=True)
            with guard.operation("verify_identity", update=True):
                if rr.sha256(Path(input_path)) != input_sha:
                    raise ValueError("label-free input projection changed during inference")
                for path, digest in {**sources, **flatten_bundle(checkpoints)}.items():
                    if rr.sha256(Path(path)) != digest:
                        raise ValueError(f"pinned source/model changed: {path}")
                rr.write_json(arm_path / "identity.json", {
                    "arm": arm, "rows": len(captures), "input_sha256": input_sha,
                    "source_hashes": sources, "checkpoint_files": checkpoints,
                    "runtime_version": version, "model_artifacts": model_artifacts,
                    "capture_sha256": rr.sha256(arm_path / "capture.jsonl"),
                })
            guard.sample_and_append("complete")
    except BaseException as error:
        rr.write_json(arm_path / "failure.json", {
            "completed": False, "error_type": type(error).__name__,
            "error": str(error), "traceback": traceback.format_exc(),
        })
        raise


def _run_arm(arm: str, checkpoints: tuple[Path, Path, Path], input_path: Path,
             output: Path, global_started: float, baseline_swap: int) -> None:
    remaining = LIMITS["whole_seconds"] - (time.monotonic() - global_started)
    if remaining <= 0:
        raise RuntimeError("evaluation wall limit expired before arm launch")
    process = multiprocessing.get_context("spawn").Process(
        target=_arm_capture,
        args=(arm, *(str(path.resolve()) for path in checkpoints),
              str(input_path.resolve()), str((output / arm).resolve()),
              global_started, baseline_swap),
    )
    process.start()
    process.join(remaining)
    if process.is_alive():
        process.terminate()
        process.join(15)
        raise RuntimeError(f"{arm} exceeded the shared wall limit")
    if process.exitcode != 0:
        failure = output / arm / "failure.json"
        detail = json.loads(failure.read_text()).get("error") if failure.is_file() else None
        raise RuntimeError(f"{arm} evaluation failed" + (f": {detail}" if detail else ""))


def comparison(before_name: str, after_name: str, captures: dict[str, list[dict]],
               cases: list[dict]) -> dict:
    result = rr.compare_captures(captures[before_name], captures[after_name], cases)
    result[before_name] = result.pop("baseline")
    result[after_name] = result.pop("proposed")
    return result


def run(phonetic_checkpoint: Path, word_checkpoint: Path,
        growth_checkpoint: Path, output: Path, split: str) -> dict:
    from scripts.source_span_resources import system_swap_bytes

    started = time.monotonic()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        cases, source_identity = rr.load_cases(split)
        inputs = rr.project_inputs(cases)
        input_path = output / "inputs.json"
        rr.write_json(input_path, inputs)
        input_hash = rr.sha256(input_path)
        sources = source_pins()
        arm_paths = {
            "original": (OLD_PHONETIC_CHECKPOINT, OLD_WORD_CHECKPOINT,
                         OLD_GROWTH_CHECKPOINT),
            "phonetic_only": (Path(phonetic_checkpoint), OLD_WORD_CHECKPOINT,
                              OLD_GROWTH_CHECKPOINT),
            "all_retrained": (Path(phonetic_checkpoint), Path(word_checkpoint),
                              Path(growth_checkpoint)),
        }
        arm_bundles = {name: checkpoint_bundle(*paths)
                       for name, paths in arm_paths.items()}
        rr.write_json(output / "frozen-input.json", {
            "split": split, "rows": len(inputs), "input_sha256": input_hash,
            "source_identity": source_identity, "runtime_source_hashes": sources,
            "arm_checkpoint_files": arm_bundles,
            "rules_hash": rr.canonical_hash([row.get("rule") for row in cases]),
        })
        baseline_swap = system_swap_bytes()
        for arm in ARM_ORDER:
            _run_arm(arm, arm_paths[arm], input_path, output, started, baseline_swap)

        if rr.sha256(input_path) != input_hash:
            raise ValueError("label-free input projection changed")
        for path, digest in sources.items():
            if rr.sha256(Path(path)) != digest:
                raise ValueError(f"pinned source changed: {path}")
        identities = {
            arm: json.loads((output / arm / "identity.json").read_text())
            for arm in ARM_ORDER
        }
        capture_paths = {arm: output / arm / "capture.jsonl" for arm in ARM_ORDER}
        captures = {arm: rr._load_jsonl(path) for arm, path in capture_paths.items()}
        verify_arm_pair(
            "original", "phonetic_only", identities["original"],
            identities["phonetic_only"], captures["original"],
            captures["phonetic_only"],
            before_capture_sha256=rr.sha256(capture_paths["original"]),
            after_capture_sha256=rr.sha256(capture_paths["phonetic_only"]),
            allowed_groups={"phonetic"})
        verify_arm_pair(
            "phonetic_only", "all_retrained", identities["phonetic_only"],
            identities["all_retrained"], captures["phonetic_only"],
            captures["all_retrained"],
            before_capture_sha256=rr.sha256(capture_paths["phonetic_only"]),
            after_capture_sha256=rr.sha256(capture_paths["all_retrained"]),
            allowed_groups={"word_ctc", "growth_ctc"})

        original_to_all = comparison("original", "all_retrained", captures, cases)
        phonetic_to_all = comparison("phonetic_only", "all_retrained", captures, cases)
        outcomes = {
            "original_to_all": original_to_all.pop("outcomes"),
            "phonetic_only_to_all": phonetic_to_all.pop("outcomes"),
        }
        for name, rows in outcomes.items():
            with (output / f"{name}-outcomes.jsonl").open("x") as stream:
                for row in rows:
                    rr.append_jsonl(stream, row)
        raw_ids = {row["id"] for row in cases if row.get("raw")}
        raw_protected = all(
            rr.topk_outcome(capture, case["expected"])["top1"]
            for arm in ARM_ORDER
            for case, capture in zip(cases, captures[arm])
            if case["id"] in raw_ids
        )
        errors = sum(
            1 for arm in ARM_ORDER for row in captures[arm] if row.get("error") is not None)
        report = {
            "schema": "rr-ctc-retraining-full-engine-comparison-v1",
            "completed": True, "split": split, "rows": len(cases),
            "input_contract": "source_only_until_all_three_captures_close",
            "labels_passed_to_engine": False,
            "expected_joined_after_all_arms": True,
            "separate_process_per_arm": True,
            "arms": {
                arm: comparison(arm, arm, captures, cases)[arm]
                for arm in ARM_ORDER
            },
            "comparisons": {
                "original_to_all_retrained": original_to_all,
                "phonetic_only_to_all_retrained": phonetic_to_all,
            },
            "raw_ascii_protected_all": raw_protected,
            "capture_errors": errors,
            "latency": {arm: rr._latency(captures[arm]) for arm in ARM_ORDER},
            "proposal_activity": {
                arm: proposal_activity(captures[arm]) for arm in ARM_ORDER
            },
            "proposal_differences": {
                "original_to_all_retrained": proposal_difference(
                    captures["original"], captures["all_retrained"]),
                "phonetic_only_to_all_retrained": proposal_difference(
                    captures["phonetic_only"], captures["all_retrained"]),
            },
            "identities": identities,
            "frozen_input_sha256": input_hash,
            "frozen_input_record_sha256": rr.sha256(output / "frozen-input.json"),
            "outcomes_sha256": {
                name: rr.sha256(output / f"{name}-outcomes.jsonl") for name in outcomes
            },
            "limits": LIMITS,
            "resource_limit_rationale": list(RESOURCE_RATIONALE),
            "automatic_promotion": False,
            "quality_summary_only": True,
            "generalization_claim_allowed": False,
            "limitations": ([
                "DEV and TEST are known regression suites, not blind generalization data.",
                "No TEST result may be used to tune or select a retrained checkpoint.",
            ] if split in rr.EXPECTED_SUITES else [
                "Official RR diagnostics are known and can be ambiguous in reverse.",
                "Historical diagnostics overlap known training history.",
                "Raw ASCII checks cover explicit raw mode only.",
            ]),
            "elapsed_seconds": time.monotonic() - started,
        }
        rr.write_json(output / "report.json", report)
        rr.write_json(output / "status.json", {
            "state": "complete", "report_sha256": rr.sha256(output / "report.json"),
            "capture_errors": errors,
        })
        return report
    except BaseException as error:
        rr.write_json(output / "failure.json", {
            "completed": False, "error_type": type(error).__name__,
            "error": str(error), "traceback": traceback.format_exc(),
            "elapsed_seconds": time.monotonic() - started,
        })
        raise


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phonetic-checkpoint", type=Path, required=True)
    parser.add_argument("--word-checkpoint", type=Path, required=True)
    parser.add_argument("--growth-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test", "diagnostics"), required=True)
    args = parser.parse_args(argv)
    report = run(args.phonetic_checkpoint, args.word_checkpoint,
                 args.growth_checkpoint, args.output, args.split)
    print(json.dumps({
        "completed": report["completed"], "split": report["split"],
        "arms": {name: value["counts"] for name, value in report["arms"].items()},
        "capture_errors": report["capture_errors"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
