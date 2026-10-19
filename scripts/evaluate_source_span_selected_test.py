"""Closed 1,000-case regression for the DEV-selected source-span arm.

The existing TEST runtime capture is the only generator input.  Fresh
source-edge proposals and every selected-arm prediction are durably closed
before the canonical TEST targets are opened.  This is an offline qualification
gate; it never edits or instantiates the production converter.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_source_span_pair import (
    collect_candidate_groups,
    compact_public,
    protected_fixtures,
    response_protects_literals,
    support_reason,
    validate_training_artifacts,
    verify_paths_protect_literals,
)
from scripts.preflight_source_span_residual import PARENT, PARENT_HASHES, require, sha, write
from scripts.score_source_edge_recombination import apply_policy
from scripts.train_source_span_pair import install_arm


SCHEMA = "source-span-selected-test-evaluation-v1"
SELECTED_ARM = "whole_source"
EXPECTED_ROWS = 1_000
BASELINE = {"top1": 953, "top5": 972}
PLAN = ROOT / ".omx/plans/source-span-selected-test-v1.md"
DEFAULT_TRAINING = ROOT / ".omx/artifacts/source-span-paired-training-v1"
DEFAULT_DEV = ROOT / ".omx/artifacts/source-span-paired-dev-v1"
DEFAULT_RUNTIME = ROOT / ".omx/artifacts/extended-runtime-test-v1"
DEV_REPORT_SHA256 = "05b6fe0924bbbaeba17ec0e55a8a566f3c42d8a3c51b20bc84475fde25c1b9de"
DEV_PROOF_SHA256 = "5cb8e7ec046878e5176597de42438b6c0e96d538f1c05ab751a0b1f7d4653893"
RUNTIME_HASHES = {
    "report.json": "a4a1bb33ec2756fee2ed9833eda126fdf7cd27caa591414c92250af35439a402",
    "capture.jsonl": "3d7b12d458411bd8caa6b34ea47eb024df15d5bbaa0467b58ac7d3b9a30ce010",
    "capture.complete.json": "c6ee6af86852544945aad63066bd1b4b52ecb44e0c87772b76960b6d22c4a62d",
    "independent-verification.json": "8384213f9faf07283ce54b0abe64cdda04328e2a7936dd065ad6a3ae896327e5",
    "pins.json": "2590fe9bd42a5e7658f443ca96f7b2bf509db1534ac990590e245836803212b0",
}
TEST_SHA256 = "1dcf804460b40c445abf688dc47b452937d8911ac156b8520141fad25b973d06"
SOURCE_GENERATOR = ROOT / "hanglish/core/source_edge_recombination.py"
SOURCE_GENERATOR_SHA256 = "c4e7d7c59c3edb8623b28dc933f429b220b4d17ff58416d47449eeaa4d78d119"
SELECTED_RESIDUAL_SHA256 = "80c49d898af37edbc22fbe39b66efc8bcda130cfcd130040d5e05421ba9fb5cf"
EVAL_LIMITS = {"whole_seconds": 2400.0, "update_seconds": 60.0,
               "rss_bytes": 3 * 1024**3, "swap_growth_bytes": 512 * 1024**2}
DEV_RELIED_PATHS = (
    ROOT / "scripts/evaluate_source_span_pair.py",
    ROOT / "scripts/score_source_edge_recombination.py",
    ROOT / "scripts/train_source_span_pair.py",
    ROOT / "scripts/source_span_resources.py",
    ROOT / "hanglish/models/source_span_residual.py",
    ROOT / "hanglish/models/span_token_bridge.py",
    ROOT / "hanglish/core/source_correspondence.py",
    SOURCE_GENERATOR,
)


def iter_jsonl(path: Path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{number}") from error
            require(isinstance(value, dict), f"JSONL row is not an object at {path}:{number}")
            yield value


def append_jsonl(stream, value) -> None:
    stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True,
                            allow_nan=False) + "\n")
    stream.flush()


def validate_dev_selection_documents(report: dict, proof: dict) -> None:
    expected_counts = {
        "captured_runtime": {"top1": 264, "top5": 284},
        "parent": {"top1": 263, "top5": 280},
        "whole_source": {"top1": 271, "top5": 288},
        "source_local": {"top1": 271, "top5": 288},
    }
    decision = report.get("decision")
    require(report.get("schema") == "source-span-paired-dev-evaluation-v1"
            and report.get("completed") is True and report.get("rows") == 316
            and report.get("split") == "dev"
            and report.get("scope", {}).get("test_opened") is False
            and report.get("scope", {}).get("production_changed") is False
            and isinstance(decision, dict)
            and decision.get("selected_for_1000_case_regression") == SELECTED_ARM
            and decision.get("arms", {}).get(SELECTED_ARM, {}).get(
                "eligible_for_1000_case_regression") is True,
            "DEV selected-arm gate differs")
    for arm, counts in expected_counts.items():
        actual = report.get("metrics", {}).get(arm, {}).get("counts", {})
        require(all(actual.get(key) == value for key, value in counts.items()),
                f"DEV metric differs: {arm}")
    require(proof.get("schema")
            == "source-span-paired-dev-independent-verification-v1"
            and proof.get("verified") is True
            and proof.get("evaluation_report_sha256") == DEV_REPORT_SHA256
            and proof.get("decision") == decision
            and proof.get("contracts", {}).get("metrics_categories_gain_loss_and_decision_recomputed")
            is True,
            "DEV independent selection proof differs")


def validate_dev_reliance_pins(report: dict, paths=DEV_RELIED_PATHS) -> dict:
    """Require every reused DEV policy/model dependency to remain byte-identical."""
    report_pins = report.get("pins")
    require(isinstance(report_pins, dict), "DEV pin manifest is absent")
    verified = {}
    for path in map(Path, paths):
        expected = report_pins.get(str(path))
        require(isinstance(expected, str) and path.is_file() and sha(path) == expected,
                "DEV-evaluated dependency differs: " + str(path))
        verified[str(path)] = expected
    return verified


def validate_dev_selection(directory: Path) -> dict:
    directory = Path(directory)
    report_path = directory / "report.json"
    proof_path = directory / "independent-verification.json"
    require(report_path.is_file() and proof_path.is_file()
            and sha(report_path) == DEV_REPORT_SHA256
            and sha(proof_path) == DEV_PROOF_SHA256,
            "DEV selected-arm artifacts differ")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    validate_dev_selection_documents(report, proof)
    reliance_pins = validate_dev_reliance_pins(report)
    return {"report": report, "proof": proof,
            "pins": {str(report_path): DEV_REPORT_SHA256,
                     str(proof_path): DEV_PROOF_SHA256, **reliance_pins}}


def _target_pin(pins: dict) -> tuple[Path, str]:
    values = [(Path(raw), digest) for raw, digest in pins.items()
              if Path(raw).name == "test.jsonl"
              and "data/challenge/sentence-v3" in Path(raw).as_posix()]
    require(len(values) == 1 and values[0][1] == TEST_SHA256,
            "canonical TEST target pin differs")
    return values[0]


def validate_runtime_documents(proof: dict, closure: dict,
                               pins: dict) -> tuple[Path, str]:
    # Do not JSON-parse the historical report here: it contains per-row TEST
    # outcomes.  Its exact hash is authenticated above and by this label-free
    # independent proof, which is the aggregate metric authority.
    require(proof.get("schema") == "extended-separated-runtime-independent-verification-v1"
            and proof.get("verified") is True and proof.get("split") == "test"
            and proof.get("identity", {}).get("report_sha256")
            == RUNTIME_HASHES["report.json"]
            and proof.get("identity", {}).get("capture_sha256")
            == RUNTIME_HASHES["capture.jsonl"]
            and proof.get("metrics", {}).get("proposed")
            == {"errors": 0, "rows": EXPECTED_ROWS, **BASELINE}
            and proof.get("contracts", {}).get("source_only_runtime") is True
            and proof.get("contracts", {}).get("capture_closed_before_target_join") is True,
            "captured TEST runtime proof differs")
    require(closure == {"capture_sha256": RUNTIME_HASHES["capture.jsonl"],
                        "rows": EXPECTED_ROWS,
                        "schema": "extended-separated-runtime-source-closure-v1",
                        "source_only": True, "split": "test",
                        "targets_joined": False, "test_opened": True},
            "captured TEST source-only closure differs")
    model_pins = proof.get("identity", {}).get("model_pins")
    require(isinstance(model_pins, dict) and model_pins
            and all(pins.get(raw) == digest for raw, digest in model_pins.items()),
            "captured TEST model pins differ from independent proof")
    return _target_pin(pins)


def validate_runtime_artifacts(directory: Path) -> dict:
    """Validate source/runtime identity without opening the TEST target file."""
    directory = Path(directory)
    for name, digest in RUNTIME_HASHES.items():
        path = directory / name
        require(path.is_file() and sha(path) == digest,
                "captured TEST artifact differs: " + name)
    proof = json.loads((directory / "independent-verification.json").read_text(encoding="utf-8"))
    closure = json.loads((directory / "capture.complete.json").read_text(encoding="utf-8"))
    pins = json.loads((directory / "pins.json").read_text(encoding="utf-8"))
    target_path, target_sha = validate_runtime_documents(proof, closure, pins)
    for raw, digest in pins.items():
        path = Path(raw)
        if path == target_path:
            continue
        require(path.is_file() and sha(path) == digest,
                "captured TEST runtime pin differs: " + raw)
    for name, digest in PARENT_HASHES.items():
        require(pins.get(str(PARENT / name)) == digest
                and sha(PARENT / name) == digest,
                "captured TEST parent identity differs: " + name)
    require(SOURCE_GENERATOR.is_file() and sha(SOURCE_GENERATOR) == SOURCE_GENERATOR_SHA256,
            "source-edge generator differs from DEV-evaluated implementation")
    artifact_pins = {str(directory / name): digest
                     for name, digest in RUNTIME_HASHES.items()}
    return {"proof": proof, "pins": pins,
            "artifact_pins": artifact_pins, "target_path": target_path,
            "target_sha256": target_sha}


def validate_selected_training_binding(dev_report: dict, training_auth: dict) -> dict:
    """Bind the selected checkpoint to the exact training run evaluated on DEV."""
    report_path = Path(training_auth["report_path"])
    checkpoint = training_auth.get("checkpoints", {}).get(SELECTED_ARM, {})
    residual_path = Path(checkpoint.get("residual_path", ""))
    report_digest = sha(report_path) if report_path.is_file() else None
    residual_digest = sha(residual_path) if residual_path.is_file() else None
    require(report_digest == dev_report.get("training_report_sha256"),
            "DEV selection is not bound to the supplied matched training report")
    require(residual_digest == SELECTED_RESIDUAL_SHA256,
            "DEV-selected whole-source residual identity differs")
    return {"report_sha256": report_digest, "residual_sha256": residual_digest,
            "checkpoint": checkpoint}


def make_proposal_row(runtime_row: dict) -> dict:
    from hanglish.core.source_edge_recombination import propose

    identifier = runtime_row.get("id")
    request = runtime_row.get("request")
    source = request.get("input") if isinstance(request, dict) else None
    response = runtime_row.get("proposed")
    require(isinstance(identifier, str) and isinstance(source, str)
            and request == {"input": source, "context": ""}
            and isinstance(response, dict)
            and response.get("input") == source and response.get("context") == "",
            "captured TEST source-only row differs")
    generated = propose(source, response["pipeline"]["pool"], response["candidates"])
    return {"id": identifier, "request": deepcopy(request), **generated}


def build_prediction(runtime_row: dict, proposal_row: dict, texts: list[str],
                     representatives: list[dict], groups, scores: list[float] | None,
                     reason: str | None, latency_ms: dict[str, float]) -> dict:
    source = runtime_row["request"]["input"]
    original = runtime_row["proposed"]
    require(set(latency_ms) == {"captured_runtime", "source_edge_proposal",
                                "parent_features", SELECTED_ARM}
            and all(type(value) in (int, float) and math.isfinite(value) and value >= 0
                    for value in latency_ms.values()), "TEST latency record differs")
    if reason is None:
        require(scores is not None and len(scores) == len(texts)
                and all(type(value) in (int, float) and math.isfinite(value)
                        for value in scores), "selected-arm scores differ")
        applied = apply_policy(original, representatives, dict(zip(texts, scores)),
            supported=True, source=source, model_id=SELECTED_ARM)
        require(applied["applied"] is True, "selected-arm policy was not applied")
        selected = applied["response"]
        score_values = list(map(float, scores))
    else:
        require(scores is None, "fallback prediction may not contain scores")
        selected = deepcopy(original)
        score_values = []
    path_contract = verify_paths_protect_literals(source, groups)
    proposal_ids = [row.get("proposal_id") for row in proposal_row.get("proposals", [])]
    candidate_ids = [row.get("candidate_id") or row.get("proposal_id")
                     for row in representatives]
    path_ids = [[path.path_id for path in group] for group in groups]
    require(all(isinstance(value, str) for value in proposal_ids + candidate_ids)
            and len(candidate_ids) == len(texts)
            and all(group and len(group) == len(set(group)) for group in path_ids),
            "selected TEST candidate/path identities differ")
    return {
        "id": runtime_row["id"], "request": deepcopy(runtime_row["request"]),
        "proposal_schema_version": proposal_row.get("schema_version"),
        "proposal_ids": proposal_ids,
        "candidate_texts": list(texts),
        "candidate_ids": candidate_ids, "path_ids": path_ids,
        "support": {"supported": reason is None, "reason": reason},
        "scores": score_values, "latency_ms": latency_ms,
        "responses": {"captured_runtime": compact_public(original),
                      SELECTED_ARM: compact_public(selected)},
        "protected_paths": path_contract,
        "protected_responses": {
            "captured_runtime": response_protects_literals(source, original),
            SELECTED_ARM: response_protects_literals(source, selected),
        },
    }


def _response_result(response: dict, expected: list[str]) -> tuple[list[str], bool, bool]:
    texts = [row.get("text") for row in response.get("candidates", [])[:5]
             if isinstance(row, dict) and isinstance(row.get("text"), str)]
    return texts, bool(texts and texts[0] in expected), any(text in expected for text in texts)


def summarize(predictions: list[dict], labels: list[dict], arm: str) -> dict:
    require(arm in ("captured_runtime", SELECTED_ARM)
            and len(predictions) == len(labels) == EXPECTED_ROWS
            and [row["id"] for row in predictions] == [row.get("id") for row in labels],
            "TEST metric denominator or identity differs")
    counts = Counter(); categories = defaultdict(Counter); outcomes = []; latency = []
    for row, label in zip(predictions, labels):
        require(row["request"] == {"input": label.get("input"), "context": ""},
                "late TEST source identity differs")
        expected = label.get("expected")
        require(isinstance(expected, list) and expected
                and all(isinstance(value, str) for value in expected),
                "TEST expected values differ")
        texts, top1, top5 = _response_result(row["responses"][arm], expected)
        covered = bool(set(row["candidate_texts"] + [label["input"]]).intersection(expected))
        # Pool-wide path preservation is reported separately.  The arm metric
        # concerns the response the user would actually receive.
        protected = row["protected_responses"][arm]["verified"]
        supported = row["support"]["supported"]
        values = {"rows": 1, "errors": 0, "top1": int(top1), "top5": int(top5),
                  "full_pool_covered": int(covered), "supported": int(supported),
                  "fallback": int(not supported), "protected_exact": int(protected)}
        counts.update(values)
        categories[label.get("category", "uncategorized")].update(values)
        outcomes.append({"id": row["id"], "actual": texts, "expected": expected,
                         "top1": top1, "top5": top5, "full_pool_covered": covered,
                         "supported": supported, "protected_exact": protected})
        if arm == "captured_runtime":
            latency.append(row["latency_ms"]["captured_runtime"])
        else:
            latency.append(sum(row["latency_ms"][key] for key in
                               ("source_edge_proposal", "parent_features", SELECTED_ARM)))
    latency.sort()
    return {"counts": dict(counts),
            "categories": {key: dict(value) for key, value in sorted(categories.items())},
            "latency": {"measured_rows": len(latency), "total_ms": sum(latency),
                        "mean_ms": sum(latency) / len(latency), "max_ms": max(latency),
                        "p95_ms": latency[math.ceil(.95 * len(latency)) - 1]},
            "outcomes": outcomes}


def paired_delta(before: dict, after: dict) -> dict:
    left = {row["id"]: row for row in before["outcomes"]}
    right = {row["id"]: row for row in after["outcomes"]}
    require(set(left) == set(right), "TEST paired metric IDs differ")
    return {
        "top1_gain_ids": [key for key in left if right[key]["top1"] and not left[key]["top1"]],
        "top1_loss_ids": [key for key in left if left[key]["top1"] and not right[key]["top1"]],
        "top5_gain_ids": [key for key in left if right[key]["top5"] and not left[key]["top5"]],
        "top5_loss_ids": [key for key in left if left[key]["top5"] and not right[key]["top5"]],
        "protected_regression_ids": [key for key in left
            if left[key]["protected_exact"] and not right[key]["protected_exact"]],
    }


def decide(metrics: dict[str, dict]) -> dict:
    baseline = metrics["captured_runtime"]
    selected = metrics[SELECTED_ARM]
    delta = paired_delta(baseline, selected)
    base_counts, counts = baseline["counts"], selected["counts"]
    require(base_counts["top1"] == BASELINE["top1"]
            and base_counts["top5"] == BASELINE["top5"],
            "captured 1,000-case baseline differs")
    offline_gate = (counts["top1"] > BASELINE["top1"]
                    and counts["top5"] >= BASELINE["top5"]
                    and not delta["top1_loss_ids"]
                    and not delta["top5_loss_ids"]
                    and not delta["protected_regression_ids"]
                    and counts["protected_exact"] == EXPECTED_ROWS
                    and counts["errors"] == 0)
    return {
        "selected_arm": SELECTED_ARM, "versus_captured_runtime": delta,
        "offline_1000_case_gate": offline_gate,
        "gate_contract": {"top1_strictly_above": BASELINE["top1"],
                          "top5_at_least": BASELINE["top5"],
                          "no_top1_losses": True, "no_top5_losses": True,
                          "no_protected_literal_failures": True,
                          "zero_execution_errors": True},
        "production_qualified": False,
        "production_qualification_reason": (
            "offline cached-pipeline evaluation still requires fresh actual-runtime "
            "integration, exact parity, resource, localhost, and Tailscale verification"),
    }


def run(training: Path, dev: Path, runtime: Path, output: Path) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    hook = None
    try:
        require(PLAN.is_file(), "frozen selected-TEST plan is absent")
        dev_auth = validate_dev_selection(dev)  # mandatory before any model load
        training_auth = validate_training_artifacts(training)
        runtime_auth = validate_runtime_artifacts(runtime)
        selected_binding = validate_selected_training_binding(
            dev_auth["report"], training_auth)
        selected_checkpoint = selected_binding["checkpoint"]

        from safetensors.torch import load_file
        import torch
        from hanglish.models.sentence_ranker import SentenceRanker
        from hanglish.models.source_span_residual import SourceSpanResidualRanker
        from hanglish.models.span_token_bridge import encode_aligned_candidates
        from scripts.preflight_pko_seq2seq import system_swap_bytes
        from scripts.source_span_resources import AtomicResourceWatchdog

        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        pins = dict(training_auth["pins"])
        pins.update(training_auth["artifact_pins"])
        pins.update(runtime_auth["pins"])
        pins.update(runtime_auth["artifact_pins"])
        pins.update(dev_auth["pins"])
        tracked = [Path(__file__).resolve(), PLAN, SOURCE_GENERATOR,
                   ROOT / "scripts/evaluate_source_span_pair.py",
                   ROOT / "scripts/score_source_edge_recombination.py",
                   ROOT / "scripts/train_source_span_pair.py",
                   ROOT / "scripts/source_span_resources.py",
                   ROOT / "hanglish/models/source_span_residual.py",
                   ROOT / "hanglish/models/span_token_bridge.py",
                   ROOT / "hanglish/core/source_correspondence.py"]
        pins.update({str(path): sha(path) for path in tracked})
        write(output / "pins.json", dict(sorted(pins.items())))

        baseline_swap = system_swap_bytes()
        watchdog = AtomicResourceWatchdog(started, baseline_swap,
            output / "resources.jsonl", interval_s=2.0, limits=EVAL_LIMITS)
        predictions = []
        with watchdog:
            with watchdog.operation("load_parent_and_selected_residual", update=True):
                parent = SentenceRanker.load(PARENT)
                model = SourceSpanResidualRanker(parent)
                require(model.architecture == training_auth["architecture"],
                        "selected TEST architecture differs from checkpoint")
                model.load_residual_state_dict(load_file(
                    selected_checkpoint["residual_path"]))
                hook = install_arm(model, SELECTED_ARM)
                require(hook is not None, "whole-source control hook was not installed")
                model.eval()

            proposal_path = output / "proposals.jsonl"
            prediction_path = output / "predictions.jsonl"
            seen = set()
            with (proposal_path.open("x", encoding="utf-8") as proposal_stream,
                  prediction_path.open("x", encoding="utf-8") as prediction_stream):
                for runtime_row in iter_jsonl(Path(runtime) / "capture.jsonl"):
                    identifier = runtime_row.get("id")
                    require(isinstance(identifier, str) and identifier not in seen,
                            "captured TEST ID is missing or duplicated")
                    seen.add(identifier)
                    before = time.monotonic()
                    with watchdog.operation("source_edge_proposal:" + identifier, update=True):
                        proposal_row = make_proposal_row(runtime_row)
                    proposal_ms = (time.monotonic() - before) * 1000
                    append_jsonl(proposal_stream, proposal_row)

                    texts, representatives, groups = collect_candidate_groups(
                        runtime_row, proposal_row)
                    source = runtime_row["request"]["input"]
                    reason = support_reason(source, texts, lambda values: (
                        encode_aligned_candidates(parent.tokenizer, values, 256), True)[1])
                    captured_ms = runtime_row.get("elapsed_ms")
                    require(type(captured_ms) in (int, float) and math.isfinite(captured_ms)
                            and captured_ms >= 0, "captured TEST latency differs")
                    latency = {"captured_runtime": float(captured_ms),
                               "source_edge_proposal": proposal_ms,
                               "parent_features": 0.0, SELECTED_ARM: 0.0}
                    scores = None
                    if reason is None:
                        before = time.monotonic()
                        with watchdog.operation("parent_features:" + identifier, update=True):
                            feature = model.extract_features(texts)
                        latency["parent_features"] = (time.monotonic() - before) * 1000
                        before = time.monotonic()
                        with watchdog.operation(SELECTED_ARM + ":" + identifier,
                                                update=True), torch.no_grad():
                            values = model(source, groups, features=feature)["scores"]
                        latency[SELECTED_ARM] = (time.monotonic() - before) * 1000
                        scores = values.tolist()
                    prediction = build_prediction(runtime_row, proposal_row, texts,
                        representatives, groups, scores, reason, latency)
                    append_jsonl(prediction_stream, prediction)
                    predictions.append(prediction)
                    if len(predictions) % 25 == 0:
                        print(json.dumps({"captured": len(predictions),
                                          "rows": EXPECTED_ROWS,
                                          "elapsed_s": time.monotonic() - started}), flush=True)
                require(len(predictions) == EXPECTED_ROWS,
                        "selected TEST prediction denominator differs")
                os.fsync(proposal_stream.fileno())
                os.fsync(prediction_stream.fileno())

            proposal_sha = sha(proposal_path)
            prediction_sha = sha(prediction_path)
            closure = {"schema": SCHEMA + "-source-only-closure",
                       "rows": EXPECTED_ROWS, "proposals_sha256": proposal_sha,
                       "predictions_sha256": prediction_sha, "source_only": True,
                       "targets_joined": False, "test_opened": False,
                       "selected_arm": SELECTED_ARM,
                       "candidate_policy": "fresh_source_edge_recombination_then_top4_plus_raw",
                       "input_truncation": False}
            write(output / "source-only.complete.json", closure)
            closure_sha = sha(output / "source-only.complete.json")

            # First access to canonical TEST target bytes occurs after both
            # source-only streams and their closure record are durable.
            require(sha(runtime_auth["target_path"]) == runtime_auth["target_sha256"],
                    "canonical TEST target changed before late join")
            labels = list(iter_jsonl(runtime_auth["target_path"]))
            metrics = {arm: summarize(predictions, labels, arm)
                       for arm in ("captured_runtime", SELECTED_ARM)}
            decision = decide(metrics)
            protected_rows = [row for row in predictions
                              if protected_fixtures(row["request"]["input"])]
            protected_summary = {
                "rows": len(protected_rows),
                "fixture_counts": dict(Counter(fixture["kind"] for row in protected_rows
                    for fixture in protected_fixtures(row["request"]["input"]))),
                "all_observed_candidate_paths_verified": all(
                    row["protected_paths"]["verified"] for row in protected_rows),
                "response_failures": {arm: [row["id"] for row in protected_rows
                    if not row["protected_responses"][arm]["verified"]]
                    for arm in ("captured_runtime", SELECTED_ARM)},
                "production_runtime_preservation_verified": False,
                "not_verified_reason": "selected checkpoint is not wired into production runtime",
            }
            final_resources = watchdog._sample("final")
        if hook is not None:
            hook.remove()
            hook = None
        for raw, digest in pins.items():
            require(Path(raw).is_file() and sha(Path(raw)) == digest,
                    "selected TEST evaluation pin changed: " + raw)
        outputs = {name: sha(output / name) for name in
                   ("pins.json", "proposals.jsonl", "predictions.jsonl",
                    "source-only.complete.json", "resources.jsonl")}
        report = {
            "schema": SCHEMA, "completed": True, "split": "test",
            "rows": EXPECTED_ROWS, "selected_arm": SELECTED_ARM,
            "scope": {"source_only_streams_closed_before_target_join": True,
                      "existing_exposed_1000_case_regression": True,
                      "fresh_original_generators_run": False,
                      "cached_pipeline_reused": True,
                      "fresh_source_edge_proposals": True,
                      "production_changed": False, "input_truncation": False},
            "metrics": metrics, "decision": decision,
            "protected_literal_contract": protected_summary,
            "proposals_sha256": proposal_sha, "predictions_sha256": prediction_sha,
            "closure_sha256": closure_sha,
            "training_report_sha256": sha(training_auth["report_path"]),
            "dev_report_sha256": DEV_REPORT_SHA256,
            "dev_proof_sha256": DEV_PROOF_SHA256,
            "runtime_report_sha256": RUNTIME_HASHES["report.json"],
            "runtime_proof_sha256": RUNTIME_HASHES["independent-verification.json"],
            "test_labels_sha256": runtime_auth["target_sha256"],
            "pins": dict(sorted(pins.items())), "outputs": outputs,
            "limits": EVAL_LIMITS, "baseline_swap_bytes": baseline_swap,
            "final_resources": final_resources, "elapsed_s": time.monotonic() - started,
            "limitations": [
                "The 1,000-case suite is an existing exposed regression, not blind generalization.",
                "Original candidate generators are not rerun; the verified cached pipeline is reused.",
                "Fresh source-edge proposals and selected residual scores are offline only.",
                "A passing offline gate still requires actual-runtime integration, parity, resource, localhost, and Tailscale verification before promotion.",
            ],
        }
        write(output / "report.json", report)
        print(json.dumps({"completed": True,
                          "metrics": {arm: value["counts"] for arm, value in metrics.items()},
                          "decision": decision}, ensure_ascii=False), flush=True)
        return report
    except BaseException as error:
        if hook is not None:
            hook.remove()
        write(output / "failure.json", {"schema": SCHEMA, "completed": False,
              "error_type": type(error).__name__, "error": str(error),
              "elapsed_s": time.monotonic() - started})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--dev", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.training, arguments.dev, arguments.runtime, arguments.output)
