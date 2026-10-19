"""Closed DEV comparison for the matched source-span residual arms.

The source-only candidate domain and all model predictions are durably closed
before DEV targets are opened.  This runner evaluates development eligibility
only; it does not update production or claim TEST/1,000-case qualification.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
from itertools import zip_longest
import json
import math
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hanglish.core.source_correspondence import Path as AlignmentPath, align
from scripts.preflight_source_span_residual import PARENT, PARENT_HASHES, require, sha, write
from scripts.score_source_edge_recombination import (
    DEFAULT_INPUT,
    DEFAULT_PROOF,
    DEV,
    EXPECTED_ROWS,
    RUNTIME,
    apply_policy,
    compact_public,
    policy_pools,
    validate_inputs,
)
from scripts.train_source_span_pair import ARMS, LIMITS, RECIPE, SCHEMA as TRAIN_SCHEMA, install_arm


SCHEMA = "source-span-paired-dev-evaluation-v1"
PLAN = ROOT / ".omx/plans/source-span-paired-training-v1.md"
DEFAULT_TRAINING = ROOT / ".omx/artifacts/source-span-paired-training-v1"
ARM_NAMES = ("captured_runtime", "parent", *ARMS)
DEV_SHA256 = "593e1a032b54bf4d2c541da6754cfc68daf759cdcee7f8fa1f0a746459910ac9"
EVAL_LIMITS = {"whole_seconds": 1800.0, "update_seconds": 60.0,
               "rss_bytes": 3 * 1024**3, "swap_growth_bytes": 512 * 1024**2}
URL_RE = re.compile(r"https?://[^\s`]+")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
EXPLICIT_COPY_RE = re.compile(r"`[^`]+`")


def iter_jsonl(path: Path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{number}") from error


def canonical_hash(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def append_jsonl(stream, value) -> None:
    stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True,
                            allow_nan=False) + "\n")


def _candidate_path(source: str, candidate: dict) -> AlignmentPath:
    require(isinstance(candidate, dict) and isinstance(candidate.get("text"), str)
            and isinstance(candidate.get("segments"), list),
            "candidate path is malformed")
    return align(source, candidate["text"], candidate["segments"])


def collect_candidate_groups(runtime_row: dict, proposal_row: dict):
    """Return the fixed text domain and every observed path for each text."""
    source = runtime_row.get("request", {}).get("input")
    pools = policy_pools(runtime_row, proposal_row)
    representatives = pools["recombined_pool_256"]
    texts = [candidate["text"] for candidate in representatives]
    require(source and len(texts) == len(set(texts))
            and all(text != source for text in texts),
            "recombined nonraw text domain differs")
    domain = set(texts)
    observed = [*runtime_row["proposed"]["pipeline"]["pool"],
                *proposal_row.get("proposals", []),
                *runtime_row["proposed"].get("candidates", [])]
    grouped = defaultdict(dict)
    for candidate in observed:
        if not isinstance(candidate, dict) or candidate.get("text") not in domain:
            continue
        path = _candidate_path(source, candidate)
        grouped[path.text][path.path_id] = path
    require(set(grouped) == domain, "an evaluated candidate has no observed structural path")
    groups = [[grouped[text][key] for key in sorted(grouped[text])] for text in texts]
    return texts, representatives, groups


def protected_fixtures(source: str) -> list[dict]:
    fixtures = []
    for kind, pattern in (("url", URL_RE), ("email", EMAIL_RE),
                          ("explicit_copy", EXPLICIT_COPY_RE)):
        fixtures.extend({"kind": kind, "start": match.start(), "end": match.end(),
                         "text": match.group()} for match in pattern.finditer(source))
    return sorted(fixtures, key=lambda item: (item["start"], item["end"], item["kind"]))


def path_preserves_fixture(path: AlignmentPath, fixture: dict) -> bool:
    """Prove the fixture's complete source interval is copied byte-for-byte."""
    start, end = fixture["start"], fixture["end"]
    if not (0 <= start < end <= len(path.source)
            and path.source[start:end] == fixture["text"]):
        return False
    cursor = start
    output = []
    for span in path.spans:
        left, right = max(start, span.input_start), min(end, span.input_end)
        if left >= right:
            continue
        if left != cursor or span.action != "copy":
            return False
        relative_left = left - span.input_start
        relative_right = right - span.input_start
        output.append(span.text[relative_left:relative_right])
        cursor = right
    return cursor == end and "".join(output) == fixture["text"]


def verify_paths_protect_literals(source: str, groups: list[list[AlignmentPath]]) -> dict:
    fixtures = protected_fixtures(source)
    failures = []
    checked = 0
    for candidate_index, group in enumerate(groups):
        for path in group:
            for fixture in fixtures:
                checked += 1
                if not path_preserves_fixture(path, fixture):
                    failures.append({"candidate_index": candidate_index,
                                     "path_id": path.path_id, **fixture})
    return {"fixture_count": len(fixtures), "path_fixture_checks": checked,
            "verified": not failures, "failures": failures}


def response_protects_literals(source: str, response: dict) -> dict:
    fixtures = protected_fixtures(source)
    if not fixtures:
        return {"fixture_count": 0, "candidate_fixture_checks": 0,
                "verified": True, "failures": []}
    failures = []
    checks = 0
    candidates = response.get("candidates") if isinstance(response, dict) else None
    if not isinstance(candidates, list) or not candidates:
        return {"fixture_count": len(fixtures), "candidate_fixture_checks": 0,
                "verified": False, "failures": [{"reason": "response_candidates_absent"}]}
    for index, candidate in enumerate(candidates):
        try:
            path = _candidate_path(source, candidate)
        except (TypeError, ValueError) as error:
            failures.append({"candidate_index": index, "reason": str(error)})
            continue
        for fixture in fixtures:
            checks += 1
            if not path_preserves_fixture(path, fixture):
                failures.append({"candidate_index": index, "path_id": path.path_id,
                                 **fixture})
    return {"fixture_count": len(fixtures), "candidate_fixture_checks": checks,
            "verified": not failures, "failures": failures}


def support_reason(source: str, texts: list[str], can_encode) -> str | None:
    if len(source.encode("utf-8")) > 2048:
        return "source_exceeds_2048_bytes"
    if not texts:
        return "empty_nonraw_pool"
    try:
        fits = can_encode(texts)
    except ValueError:
        return "candidate_exceeds_256_paired_tokens"
    require(fits is True, "candidate fit callback must return True or raise ValueError")
    return None


def build_prediction(runtime_row: dict, texts: list[str], representatives: list[dict],
                     groups: list[list[AlignmentPath]], scores: dict[str, list[float]] | None,
                     latencies_ms: dict[str, float], reason: str | None) -> dict:
    """Build one source-only result, enforcing a single shared support decision."""
    source = runtime_row["request"]["input"]
    original = runtime_row["proposed"]
    require(set(latencies_ms) == {"captured_runtime", "parent_features", *ARMS}
            and all(type(v) in (int, float) and math.isfinite(v) and v >= 0
                    for v in latencies_ms.values()), "latency record differs")
    responses = {"captured_runtime": deepcopy(original)}
    full_hashes = {"captured_runtime": canonical_hash(original)}
    score_rows = {}
    if reason is None:
        require(scores is not None and set(scores) == {"parent", *ARMS},
                "complete supported score map is absent")
        for arm in ("parent", *ARMS):
            values = scores[arm]
            require(len(values) == len(texts)
                    and all(type(value) in (int, float) and math.isfinite(value)
                            for value in values), "model returned malformed scores")
            score_rows[arm] = list(map(float, values))
            applied = apply_policy(original, representatives,
                dict(zip(texts, score_rows[arm])), supported=True, source=source,
                model_id=arm)
            require(applied["applied"] is True, "supported policy was not applied")
            responses[arm] = applied["response"]
            full_hashes[arm] = canonical_hash(applied["response"])
    else:
        require(scores is None, "fallback row may not contain model scores")
        for arm in ("parent", *ARMS):
            responses[arm] = deepcopy(original)
            full_hashes[arm] = canonical_hash(responses[arm])
            require(responses[arm] == original, "fallback changed captured runtime response")
    path_contract = verify_paths_protect_literals(source, groups)
    response_contracts = {arm: response_protects_literals(source, response)
                          for arm, response in responses.items()}
    return {
        "id": runtime_row["id"], "request": deepcopy(runtime_row["request"]),
        "candidate_texts": texts,
        "path_ids": [[path.path_id for path in group] for group in groups],
        "support": {"supported": reason is None, "reason": reason},
        "scores": score_rows,
        "latency_ms": latencies_ms,
        "responses": {arm: compact_public(response) for arm, response in responses.items()},
        "full_response_sha256": full_hashes,
        "protected_paths": path_contract,
        "protected_responses": response_contracts,
    }


def validate_training_artifacts(training: Path) -> dict:
    training = Path(training)
    report_path = training / "report.json"
    proof_path = training / "independent-verification.json"
    require(report_path.is_file() and proof_path.is_file(),
            "matched training report or independent proof is absent")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    require(proof.get("schema")
            == "source-span-paired-training-independent-verification-v1"
            and proof.get("verified") is True
            and proof.get("run_report_sha256") == sha(report_path),
            "matched training independent proof differs")
    require(report.get("schema") == TRAIN_SCHEMA and report.get("completed") is True
            and report.get("rows") == 1483 and report.get("recipe") == RECIPE
            and report.get("limits") == LIMITS
            and report.get("contracts") == {
                "fit_only": True, "hold_opened": False, "dev_opened": False,
                "test_opened": False, "production_changed": False,
                "matched_initialization_data_order_recipe": True}
            and report.get("parent_before") == report.get("parent_after"),
            "matched training report contract differs")
    pins = report.get("pins")
    require(isinstance(pins, dict) and pins, "matched training pins are absent")
    require(json.loads((training / "pins.json").read_text(encoding="utf-8")) == pins,
            "matched training pins manifest differs")
    for raw, digest in pins.items():
        path = Path(raw)
        require(path.is_file() and sha(path) == digest, f"matched training pin differs: {raw}")
    for name, digest in PARENT_HASHES.items():
        require(pins.get(str(PARENT / name)) == digest
                and sha(PARENT / name) == digest, "frozen parent identity differs: " + name)
    checkpoints = {}
    architecture = None
    initial_hashes = None
    gradient_keys = {"source_embedding.", "source_encoder.",
                     "segment_head.0.", "segment_head.2."}
    for arm in ARMS:
        arm_report = report.get("arms", {}).get(arm, {})
        gradients = arm_report.get("upstream_gradient_maxima")
        arm_initial = arm_report.get("initial_hashes")
        directory = training / arm
        checkpoint_path = directory / "checkpoint.json"
        residual_path = directory / "residual.safetensors"
        require(checkpoint_path.is_file() and residual_path.is_file()
                and arm_report.get("updates") == 2966
                and arm_report.get("checkpoint_sha256") == sha(checkpoint_path)
                and isinstance(gradients, dict) and set(gradients) == gradient_keys
                and all(type(value) in (int, float) and math.isfinite(value) and value > 0
                        for value in gradients.values())
                and isinstance(arm_initial, dict) and arm_initial
                and arm_report.get("reload_rows") == 16
                and isinstance(arm_report.get("epoch_mean_loss"), list)
                and len(arm_report["epoch_mean_loss"]) == RECIPE["epochs"]
                and all(math.isfinite(value) for value in arm_report["epoch_mean_loss"]),
                f"matched training arm report differs: {arm}")
        initial_hashes = initial_hashes or arm_initial
        require(arm_initial == initial_hashes, "matched arm initial tensors differ")
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        require(checkpoint.get("schema") == TRAIN_SCHEMA
                and checkpoint.get("arm") == arm
                and checkpoint.get("recipe") == RECIPE
                and checkpoint.get("pins") == pins
                and checkpoint.get("residual_sha256") == sha(residual_path)
                and checkpoint.get("selection_sha256") == sha(training / "selection.json")
                and checkpoint.get("initial_sha256") == sha(training / "initial.safetensors"),
                f"matched residual checkpoint identity differs: {arm}")
        architecture = architecture or checkpoint.get("architecture")
        require(checkpoint.get("architecture") == architecture,
                "matched arm architectures differ")
        checkpoints[arm] = {"metadata": checkpoint, "metadata_path": checkpoint_path,
                            "residual_path": residual_path}
    outputs = report.get("outputs")
    required_outputs = {"pins.json", "selection.json", "initial.safetensors",
                        "feature-manifest.json", "before.jsonl"} | {
        f"{arm}/{name}" for arm in ARMS for name in
        ("updates.jsonl", "after.jsonl", "residual.safetensors",
         "checkpoint.json", "reload.json")}
    require(isinstance(outputs, dict) and outputs
            and required_outputs <= set(outputs),
            "matched training output manifest is incomplete")
    for relative, digest in outputs.items():
        path = training / relative
        require(path.is_file() and sha(path) == digest,
                "matched training output differs: " + relative)
    artifact_pins = {str(report_path): sha(report_path), str(proof_path): sha(proof_path)}
    artifact_pins.update({str(training / relative): digest
                          for relative, digest in outputs.items()})
    return {"report": report, "report_path": report_path, "proof": proof,
            "proof_path": proof_path, "pins": pins, "artifact_pins": artifact_pins,
            "architecture": architecture, "checkpoints": checkpoints}


def _counts_for_response(response: dict, expected: list[str]):
    texts = [candidate.get("text") for candidate in response.get("candidates", [])[:5]
             if isinstance(candidate, dict)]
    return texts, bool(texts and texts[0] in expected), any(text in expected for text in texts)


def summarize(predictions: list[dict], labels: list[dict], arm: str) -> dict:
    require(arm in ARM_NAMES and len(predictions) == len(labels) == EXPECTED_ROWS
            and [row["id"] for row in predictions] == [row.get("id") for row in labels],
            "DEV metric identity differs")
    counts = Counter(); categories = defaultdict(Counter); outcomes = []
    latency = []
    for row, label in zip(predictions, labels):
        require(row["request"] == {"input": label.get("input"), "context": ""},
                "late DEV source identity differs")
        expected = label.get("expected")
        require(isinstance(expected, list) and expected, "DEV expected values differ")
        texts, top1, top5 = _counts_for_response(row["responses"][arm], expected)
        covered = bool(set(row["candidate_texts"] + [label["input"]]).intersection(expected))
        protected = row["protected_responses"][arm]["verified"]
        supported = row["support"]["supported"]
        values = {"rows": 1, "top1": int(top1), "top5": int(top5),
                  "full_pool_covered": int(covered), "supported": int(supported),
                  "fallback": int(not supported), "protected_exact": int(protected)}
        counts.update(values); categories[label.get("category", "uncategorized")].update(values)
        if arm == "captured_runtime":
            latency.append(row["latency_ms"]["captured_runtime"])
        elif arm == "parent":
            latency.append(row["latency_ms"]["parent_features"])
        elif arm in ARMS:
            latency.append(row["latency_ms"]["parent_features"]
                           + row["latency_ms"][arm])
        outcomes.append({"id": row["id"], "actual": texts, "expected": expected,
                         "top1": top1, "top5": top5, "full_pool_covered": covered,
                         "supported": supported, "protected_exact": protected})
    latency.sort()
    latency_summary = ({"measured_rows": len(latency), "total_ms": sum(latency),
                        "mean_ms": sum(latency) / len(latency), "max_ms": max(latency),
                        "p95_ms": latency[min(len(latency) - 1,
                                              math.ceil(.95 * len(latency)) - 1)]}
                       if latency else {"measured_rows": 0, "total_ms": 0.0,
                                        "mean_ms": None, "max_ms": None, "p95_ms": None})
    return {"counts": dict(counts),
            "categories": {key: dict(value) for key, value in sorted(categories.items())},
            "latency": latency_summary, "outcomes": outcomes}


def paired_delta(before: dict, after: dict) -> dict:
    left = {row["id"]: row for row in before["outcomes"]}
    right = {row["id"]: row for row in after["outcomes"]}
    require(set(left) == set(right), "paired metric IDs differ")
    return {
        "top1_gain_ids": [key for key in left if right[key]["top1"] and not left[key]["top1"]],
        "top1_loss_ids": [key for key in left if left[key]["top1"] and not right[key]["top1"]],
        "top5_gain_ids": [key for key in left if right[key]["top5"] and not left[key]["top5"]],
        "top5_loss_ids": [key for key in left if left[key]["top5"] and not right[key]["top5"]],
        "protected_regression_ids": [key for key in left
            if left[key]["protected_exact"] and not right[key]["protected_exact"]],
    }


def decide(metrics: dict[str, dict]) -> dict:
    comparisons = {}
    qualified = []
    for arm in ARMS:
        against = {baseline: paired_delta(metrics[baseline], metrics[arm])
                   for baseline in ("parent", "captured_runtime")}
        counts = metrics[arm]["counts"]
        eligible = (all(counts["top1"] > metrics[baseline]["counts"]["top1"]
                        and counts["top5"] >= metrics[baseline]["counts"]["top5"]
                        and not against[baseline]["protected_regression_ids"]
                        for baseline in ("parent", "captured_runtime")))
        comparisons[arm] = {"versus": against,
                            "eligible_for_1000_case_regression": eligible}
        if eligible:
            qualified.append(arm)
    selected = None
    if qualified:
        # The plan's exact tie break prefers the control arm.
        selected = sorted(qualified, key=lambda arm: (
            -metrics[arm]["counts"]["top1"], -metrics[arm]["counts"]["top5"],
            0 if arm == "whole_source" else 1))[0]
    localization = (metrics["source_local"]["counts"]["top1"]
                    > metrics["whole_source"]["counts"]["top1"]
                    and metrics["source_local"]["counts"]["top5"]
                    >= metrics["whole_source"]["counts"]["top5"])
    return {"arms": comparisons, "selected_for_1000_case_regression": selected,
            "source_localization_claim_supported": localization,
            "production_qualified": False,
            "production_qualification_reason":
                "DEV eligibility still requires the full 1,000-case regression, input contracts, runtime resources, model identity wiring, and localhost/Tailscale verification"}


def run(training: Path, output: Path) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        training_auth = validate_training_artifacts(training)  # before Torch/model load
        source_auth = validate_inputs(DEFAULT_INPUT, DEFAULT_PROOF)
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
        pins.update(source_auth["pins"])
        tracked = [Path(__file__).resolve(), PLAN,
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
            with watchdog.operation("load_parent_and_residuals", update=True):
                parent = SentenceRanker.load(PARENT)
                models = {}
                hooks = {}
                for arm in ARMS:
                    model = SourceSpanResidualRanker(parent)
                    require(model.architecture == training_auth["architecture"],
                            "evaluation architecture differs from checkpoint")
                    state = load_file(training_auth["checkpoints"][arm]["residual_path"])
                    model.load_residual_state_dict(state)
                    hooks[arm] = install_arm(model, arm)
                    model.eval()
                    models[arm] = model
            capture_path = output / "predictions.jsonl"
            with capture_path.open("x", encoding="utf-8") as stream:
                runtime_rows = iter_jsonl(RUNTIME / "capture.jsonl")
                proposal_rows = iter_jsonl(DEFAULT_INPUT / "capture.jsonl")
                for runtime_row, proposal_row in zip_longest(runtime_rows, proposal_rows):
                    require(runtime_row is not None and proposal_row is not None,
                            "runtime/proposal denominator differs")
                    texts, representatives, groups = collect_candidate_groups(
                        runtime_row, proposal_row)
                    source = runtime_row["request"]["input"]
                    reason = support_reason(source, texts, lambda values: (
                        encode_aligned_candidates(parent.tokenizer, values, 256), True)[1])
                    captured_ms = runtime_row.get("elapsed_ms")
                    require(type(captured_ms) in (int, float) and math.isfinite(captured_ms)
                            and captured_ms >= 0, "captured runtime latency differs")
                    latency = {"captured_runtime": float(captured_ms), "parent_features": 0.0,
                               "whole_source": 0.0, "source_local": 0.0}
                    scores = None
                    if reason is None:
                        before = time.monotonic()
                        with watchdog.operation("parent_features:" + runtime_row["id"], update=True):
                            feature = models["source_local"].extract_features(texts)
                        latency["parent_features"] = (time.monotonic() - before) * 1000
                        scores = {"parent": feature.logits.tolist()}
                        for arm in ARMS:
                            before = time.monotonic()
                            with watchdog.operation(arm + ":" + runtime_row["id"], update=True), torch.no_grad():
                                values = models[arm](source, groups, features=feature)["scores"]
                            latency[arm] = (time.monotonic() - before) * 1000
                            scores[arm] = values.tolist()
                    prediction = build_prediction(runtime_row, texts, representatives,
                                                  groups, scores, latency, reason)
                    predictions.append(prediction)
                    append_jsonl(stream, prediction)
                    stream.flush()
                    if len(predictions) % 25 == 0:
                        print(json.dumps({"captured": len(predictions),
                                          "rows": EXPECTED_ROWS,
                                          "elapsed_s": time.monotonic() - started}), flush=True)
                os.fsync(stream.fileno())
            require(len(predictions) == EXPECTED_ROWS, "DEV prediction denominator differs")
            capture_sha = sha(capture_path)
            closure = {"schema": SCHEMA + "-source-only-closure",
                       "rows": EXPECTED_ROWS, "predictions_sha256": capture_sha,
                       "source_only": True, "targets_joined": False,
                       "candidate_policy": "recombined_pool_256",
                       "common_support_rule": True, "input_truncation": False}
            write(output / "predictions.complete.json", closure)
            closure_sha = sha(output / "predictions.complete.json")
            require(sha(DEV) == DEV_SHA256, "DEV targets changed before late join")
            labels = list(iter_jsonl(DEV))  # first target parse, after durable closure
            metrics = {arm: summarize(predictions, labels, arm) for arm in ARM_NAMES}
            require(metrics["captured_runtime"]["counts"]["top1"] == 264
                    and metrics["captured_runtime"]["counts"]["top5"] == 284,
                    "captured extended-runtime baseline differs")
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
                    for arm in ARM_NAMES},
                "production_runtime_preservation_verified": False,
                "not_verified_reason":
                    "offline DEV paths only; selected checkpoint is not wired into production runtime",
            }
            final_resources = watchdog._sample("final")
            for handle in hooks.values():
                if handle is not None:
                    handle.remove()
        for raw, digest in pins.items():
            require(Path(raw).is_file() and sha(Path(raw)) == digest,
                    "evaluation pin changed: " + raw)
        report = {
            "schema": SCHEMA, "completed": True, "split": "dev", "rows": EXPECTED_ROWS,
            "scope": {"source_only_predictions_closed_before_target_join": True,
                      "dev_is_exposed_development": True, "test_opened": False,
                      "production_changed": False, "input_truncation": False},
            "metrics": metrics, "decision": decision,
            "protected_literal_contract": protected_summary,
            "capture_sha256": capture_sha, "closure_sha256": closure_sha,
            "training_report_sha256": sha(training_auth["report_path"]),
            "dev_labels_sha256": DEV_SHA256,
            "pins": dict(sorted(pins.items())), "limits": EVAL_LIMITS,
            "baseline_swap_bytes": baseline_swap, "final_resources": final_resources,
            "elapsed_s": time.monotonic() - started,
            "limitations": [
                "DEV metrics do not establish blind generalization",
                "protected literals are verified on offline candidate paths, not a production runtime",
                "rescore latency excludes candidate generation while captured-runtime latency is historical end-to-end time, so they are not an apples-to-apples speed benchmark",
                "no checkpoint is production-qualified until the full 1,000-case and runtime gates pass",
            ],
        }
        write(output / "report.json", report)
        print(json.dumps({"completed": True, "metrics": {
            arm: value["counts"] for arm, value in metrics.items()},
            "decision": decision}, ensure_ascii=False), flush=True)
        return report
    except BaseException as error:
        write(output / "failure.json", {"schema": SCHEMA, "completed": False,
              "error_type": type(error).__name__, "error": str(error),
              "elapsed_s": time.monotonic() - started})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.training, args.output)
