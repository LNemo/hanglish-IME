"""Closed DEV comparison for the candidate-independent shared-edge residual."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from itertools import zip_longest
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_local_span_objective import (
    CONTROL_CLOSURE_SHA256,
    CONTROL_DEV,
    CONTROL_PREDICTIONS_SHA256,
    DEV_SHA256,
    append_jsonl,
    canonical_hash,
    collect_candidate_groups,
    iter_jsonl,
    paired_delta,
    protected_fixtures,
    response_protects_literals,
    validate_control_artifacts,
    validate_control_dev_capture,
    verify_paths_protect_literals,
)
from scripts.preflight_source_span_residual import PARENT, PARENT_HASHES, require, sha, write
from scripts.score_source_edge_recombination import (
    DEFAULT_INPUT, DEFAULT_PROOF, DEV, EXPECTED_ROWS, RUNTIME,
    apply_policy, compact_public, validate_inputs,
)
from scripts.train_shared_edge_residual import (
    AUTH_SCHEMA, LIMITS as TRAIN_LIMITS, RECIPE, SCHEMA as TRAIN_SCHEMA,
)


SCHEMA = "shared-edge-residual-dev-evaluation-v1"
AUTHORIZATION_SCHEMA = "shared-edge-residual-dev-prelaunch-authorization-v1"
PROOF_SCHEMA = "shared-edge-residual-training-independent-verification-v1"
PLAN = ROOT / ".omx/plans/shared-edge-residual-v1.md"
DEFAULT_CONTROL = ROOT / ".omx/artifacts/source-span-paired-training-v1"
DEFAULT_TRAINING = ROOT / ".omx/artifacts/shared-edge-residual-training-v1"
ARM_NAMES = ("captured_runtime", "parent", "source_local_control", "shared_edge")
MODEL_ARMS = ("source_local_control", "shared_edge")
EVAL_LIMITS = {"whole_seconds": 1800.0, "update_seconds": 60.0,
               "rss_bytes": 3 * 1024**3, "swap_growth_bytes": 512 * 1024**2}
TARGET_CACHE_BYTES = 256 * 1024**2


def finite(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def finite_nonnegative(value) -> bool:
    return finite(value) and value >= 0


def isolated_target_texts(groups: list[list]) -> list[str]:
    values = {span.text for group in groups for path in group for span in path.spans}
    require(values, "isolated target domain is empty")
    return sorted(values)


def extend_target_cache(model, missing: list[str], vectors: dict, flags: dict,
                        *, chunk_size: int = 256) -> int:
    """Extract missing text vectors in bounded chunks and return tensor bytes."""
    require(type(chunk_size) is int and chunk_size > 0
            and len(missing) == len(set(missing))
            and not set(missing).intersection(vectors)
            and set(vectors) == set(flags), "isolated target cache input differs")
    for start in range(0, len(missing), chunk_size):
        chunk = missing[start:start + chunk_size]
        extracted = model.extract_edge_features(chunk)
        require(extracted.texts == tuple(chunk), "isolated target extraction order differs")
        for index, text in enumerate(extracted.texts):
            require(text not in vectors, "isolated target was extracted twice")
            vectors[text] = extracted.vectors[index].detach().cpu().contiguous().clone()
            flags[text] = extracted.has_tokens[index].detach().cpu().clone()
    cache_bytes = sum(value.numel() * value.element_size()
                      for value in [*vectors.values(), *flags.values()])
    require(cache_bytes <= TARGET_CACHE_BYTES,
            "isolated target cache memory bound exceeded")
    return cache_bytes


def common_support_reason(source: str, candidates: list[str], targets: list[str], can_encode):
    """One fallback decision covers source, full candidates, and isolated targets."""
    if len(source.encode("utf-8")) > 2048:
        return "source_exceeds_2048_bytes"
    if not candidates:
        return "empty_nonraw_pool"
    try:
        require(can_encode(candidates) is True,
                "full-candidate fit callback must return True")
    except ValueError:
        return "full_candidate_exceeds_256_tokens"
    try:
        require(can_encode(targets) is True,
                "isolated-target fit callback must return True")
    except ValueError:
        return "isolated_target_exceeds_256_tokens"
    return None


def build_prediction(runtime_row: dict, texts: list[str], representatives: list[dict],
                     groups: list[list], targets: list[str],
                     scores: dict[str, list[float]] | None,
                     latencies_ms: dict[str, float], reason: str | None,
                     shared_audit: dict | None = None, target_cache: dict | None = None) -> dict:
    source = runtime_row["request"]["input"]
    original = runtime_row["proposed"]
    require(set(latencies_ms) == {"captured_runtime", "parent_features",
            "target_extraction", *MODEL_ARMS}
            and all(finite_nonnegative(value) for value in latencies_ms.values()),
            "latency record differs")
    responses = {"captured_runtime": deepcopy(original)}
    hashes = {"captured_runtime": canonical_hash(original)}
    score_rows = {}
    if reason is None:
        require(scores is not None and set(scores) == {"parent", *MODEL_ARMS},
                "complete supported score map is absent")
        require(isinstance(shared_audit, dict)
                and set(shared_audit) == {"edge_ids", "edge_scores", "residuals",
                                          "winning_path_ids"},
                "shared-edge audit is absent")
        for arm in ("parent", *MODEL_ARMS):
            values = scores[arm]
            require(len(values) == len(texts) and all(finite(value) for value in values),
                    "model returned malformed scores")
            score_rows[arm] = list(map(float, values))
            model_id = "source_local" if arm == "source_local_control" else arm
            applied = apply_policy(original, representatives,
                dict(zip(texts, score_rows[arm])), supported=True,
                source=source, model_id=model_id)
            require(applied["applied"] is True, "supported policy was not applied")
            responses[arm] = applied["response"]
            hashes[arm] = canonical_hash(applied["response"])
    else:
        require(scores is None and shared_audit is None,
                "fallback row may not contain model outputs")
        for arm in ("parent", *MODEL_ARMS):
            responses[arm] = deepcopy(original)
            hashes[arm] = canonical_hash(responses[arm])
    cache = target_cache or {"requested": len(targets), "hits": 0, "misses": 0}
    require(set(cache) == {"requested", "hits", "misses"}
            and all(type(cache[key]) is int and cache[key] >= 0 for key in cache)
            and cache["requested"] == len(targets)
            and cache["hits"] + cache["misses"] == len(targets),
            "isolated target cache accounting differs")
    return {
        "id": runtime_row["id"], "request": deepcopy(runtime_row["request"]),
        "candidate_texts": list(texts),
        "path_ids": [[path.path_id for path in group] for group in groups],
        "isolated_target_texts": list(targets), "target_cache": dict(cache),
        "support": {"supported": reason is None, "reason": reason},
        "scores": score_rows, "latency_ms": dict(latencies_ms),
        "shared_edge_audit": deepcopy(shared_audit),
        "responses": {arm: compact_public(response) for arm, response in responses.items()},
        "full_response_sha256": hashes,
        "protected_paths": verify_paths_protect_literals(source, groups),
        "protected_responses": {arm: response_protects_literals(source, response)
                                for arm, response in responses.items()},
    }


def verify_control_replay(prediction: dict, historical: dict) -> None:
    """Exact historical control replay is required on common-supported rows."""
    require(prediction["id"] == historical.get("id")
            and prediction["request"] == historical.get("request")
            and prediction["candidate_texts"] == historical.get("candidate_texts")
            and prediction["path_ids"] == historical.get("path_ids"),
            "historical control source/domain differs")
    if not prediction["support"]["supported"]:
        return
    require(historical.get("support", {}).get("supported") is True
            and prediction["scores"]["parent"] == historical.get("scores", {}).get("parent")
            and prediction["scores"]["source_local_control"]
                == historical.get("scores", {}).get("source_local")
            and prediction["responses"]["source_local_control"]
                == historical.get("responses", {}).get("source_local")
            and prediction["full_response_sha256"]["source_local_control"]
                == historical.get("full_response_sha256", {}).get("source_local"),
            "historical source-local supported score replay differs")


def public_counts(response: dict, expected: list[str]):
    texts = [candidate.get("text") for candidate in response.get("candidates", [])[:5]
             if isinstance(candidate, dict)]
    return texts, bool(texts and texts[0] in expected), any(text in expected for text in texts)


def summarize(predictions: list[dict], labels: list[dict], arm: str) -> dict:
    require(arm in ARM_NAMES and len(predictions) == len(labels) == EXPECTED_ROWS
            and [row["id"] for row in predictions] == [row.get("id") for row in labels],
            "DEV metric identity differs")
    counts = Counter(); categories = defaultdict(Counter); outcomes = []; latency = []
    for row, label in zip(predictions, labels):
        require(row["request"] == {"input": label.get("input"), "context": ""},
                "late DEV source identity differs")
        expected = label.get("expected")
        require(isinstance(expected, list) and expected, "DEV expected values differ")
        texts, top1, top5 = public_counts(row["responses"][arm], expected)
        covered = bool(set(row["candidate_texts"] + [label["input"]]).intersection(expected))
        protected = row["protected_responses"][arm]["verified"]
        supported = row["support"]["supported"]
        values = {"rows": 1, "top1": int(top1), "top5": int(top5),
                  "full_pool_covered": int(covered), "supported": int(supported),
                  "fallback": int(not supported), "protected_exact": int(protected)}
        counts.update(values); categories[label.get("category", "uncategorized")].update(values)
        if arm == "captured_runtime": measured = row["latency_ms"]["captured_runtime"]
        elif arm == "parent": measured = row["latency_ms"]["parent_features"]
        elif arm == "source_local_control":
            measured = row["latency_ms"]["parent_features"] + row["latency_ms"][arm]
        else:
            measured = (row["latency_ms"]["parent_features"]
                        + row["latency_ms"]["target_extraction"]
                        + row["latency_ms"][arm])
        latency.append(measured)
        outcomes.append({"id": row["id"], "actual": texts, "expected": expected,
                         "top1": top1, "top5": top5, "full_pool_covered": covered,
                         "supported": supported, "protected_exact": protected})
    latency.sort()
    return {"counts": dict(counts),
        "categories": {key: dict(value) for key, value in sorted(categories.items())},
        "latency": {"measured_rows": len(latency), "total_ms": sum(latency),
                    "mean_ms": sum(latency) / len(latency), "max_ms": max(latency),
                    "p95_ms": latency[min(len(latency) - 1,
                                           math.ceil(.95 * len(latency)) - 1)]},
        "outcomes": outcomes}


def decide(metrics: dict[str, dict]) -> dict:
    require(set(metrics) == set(ARM_NAMES), "evaluation metric arms differ")
    baselines = ("source_local_control", "captured_runtime")
    comparisons = {baseline: paired_delta(metrics[baseline], metrics["shared_edge"])
                   for baseline in baselines}
    values = metrics["shared_edge"]["counts"]
    eligible = all(values["top1"] > metrics[baseline]["counts"]["top1"]
                   and values["top5"] >= metrics[baseline]["counts"]["top5"]
                   and not comparisons[baseline]["protected_regression_ids"]
                   for baseline in baselines)
    return {"evaluated_arm": "shared_edge", "historical_control_eligible": False,
            "versus": comparisons, "eligible_for_1000_case_regression": eligible,
            "selected_for_1000_case_regression": "shared_edge" if eligible else None,
            "production_qualified": False,
            "production_qualification_reason":
                "DEV eligibility still requires the fixed 1,000-case and runtime gates"}


def validate_training_artifacts(training: Path, control: Path = DEFAULT_CONTROL) -> dict:
    control_auth = validate_control_artifacts(control)
    training = Path(training); report_path = training / "report.json"
    proof_path = training / "independent-verification.json"
    require(report_path.is_file() and proof_path.is_file(),
            "shared-edge report or independent proof is absent")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    require(proof.get("schema") == PROOF_SCHEMA and proof.get("verified") is True
            and proof.get("report_sha256") == sha(report_path)
            and proof.get("producer_sha256") == sha(ROOT / "scripts/train_shared_edge_residual.py")
            and proof.get("counts") == {"rows": 1483, "targets": 16814,
                "edges": 47282, "candidate_paths": 11793,
                "updates": 2966, "reload_rows": 16}
            and proof.get("tensors", {}).get("trainable_parameters") == 71073
            and proof.get("tensors", {}).get("initial_matches_historical") is True
            and proof.get("tensors", {}).get("parent_unchanged") is True
            and proof.get("resources", {}).get("atomic_operation_attribution_verified") is True,
            "shared-edge independent proof differs")
    proof_contracts = proof.get("contracts", {})
    required_proof_contracts = {
        "authorized_fit_inventory_independently_reconstructed": True,
        "all_target_cache_shapes_flags_metadata_and_hashes_verified": True,
        "historical_initial_and_parent_frozen": True,
        "exact_CE_only_update_order_and_losses_verified": True,
        "all_final_edges_paths_weights_contributions_and_winners_replayed": True,
        "shared_edge_contributions_equal_across_paths": True,
        "fp32_base_plus_residual_and_fit_metrics_replayed": True,
        "reload_scores_and_edge_scores_verified": True,
        "closed_output_and_pin_hashes_verified": True,
        "fit_only": True, "hold_opened": False, "dev_opened": False,
        "test_opened": False, "production_changed": False,
        "neural_forward_rerun": False}
    require(all(proof_contracts.get(key) == value
                for key, value in required_proof_contracts.items()),
            "shared-edge independent proof contract differs")
    contracts = {"fit_only": True, "hold_opened": False, "dev_opened": False,
        "test_opened": False, "production_changed": False,
        "candidate_hidden_states_used_in_residual": False,
        "all_fit_initial_parent_scores_exact": True,
        "all_fit_shared_edge_contributions_equal": True,
        "source_codepoint_weight_mass_verified": True}
    require(report.get("schema") == TRAIN_SCHEMA and report.get("completed") is True
            and report.get("rows") == 1483 and report.get("updates") == 2966
            and report.get("recipe") == RECIPE and report.get("limits") == TRAIN_LIMITS
            and all(report.get("contracts", {}).get(key) == value
                    for key, value in contracts.items())
            and report.get("parent_before") == report.get("parent_after")
            and report.get("reload_rows") == 16,
            "shared-edge training contract differs")
    pins = report.get("pins"); require(isinstance(pins, dict) and pins,
                                       "shared-edge pins are absent")
    for raw, digest in pins.items():
        require(Path(raw).is_file() and sha(Path(raw)) == digest,
                "shared-edge input pin differs: " + raw)
    for name, digest in PARENT_HASHES.items():
        require(pins.get(str(PARENT / name)) == digest and sha(PARENT / name) == digest,
                "frozen parent differs: " + name)
    authorizations = []
    for raw, digest in pins.items():
        path = Path(raw)
        if path.suffix == ".json":
            try: value = json.loads(path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError): continue
            if isinstance(value, dict) and value.get("schema") == AUTH_SCHEMA:
                authorizations.append((path, digest, value))
    require(len(authorizations) == 1, "consumed prelaunch authorization differs")
    authorization_path, _, authorization = authorizations[0]
    require(authorization.get("rows") == 1483 and authorization.get("targets") == 16814
            and authorization.get("recipe") == RECIPE
            and {raw: digest for raw, digest in pins.items()
                 if raw != str(authorization_path.resolve())} == authorization.get("pins"),
            "prelaunch authorization contract differs")
    checkpoint_path = training / "checkpoint.json"; residual_path = training / "residual.safetensors"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    architecture = report.get("architecture")
    require(checkpoint == {"schema": TRAIN_SCHEMA, "arm": "shared_edge",
            "architecture": architecture, "recipe": RECIPE, "pins": pins,
            "residual_sha256": sha(residual_path),
            "initial_sha256": sha(training / "initial.safetensors"),
            "selection_sha256": sha(training / "selection.json")}
            and architecture.get("schema") == "shared-source-edge-residual-v1"
            and architecture.get("target_context") == "isolated_exact_output_text"
            and architecture.get("path_reduction") == "distinct_path_weighted_sum_max"
            and report.get("initial_hashes")
                == control_auth["report"]["arms"]["source_local"]["initial_hashes"]
            and sha(training / "initial.safetensors")
                == control_auth["checkpoints"]["source_local"]["metadata"]["initial_sha256"],
            "shared-edge checkpoint/control identity differs")
    required = {"pins.json", "selection.json", "targets.json", "resources.jsonl",
        "initial.safetensors", "tokens.json", "target-features.safetensors",
        "target-features.json", "before.jsonl", "updates.jsonl", "after.jsonl",
        "residual.safetensors", "reload.json", "checkpoint.json"}
    outputs = report.get("outputs")
    require(isinstance(outputs, dict) and required == set(outputs),
            "shared-edge output manifest differs")
    for relative, digest in outputs.items():
        require(sha(training / relative) == digest,
                "shared-edge output changed: " + relative)
    artifact_pins = {str(report_path): sha(report_path), str(proof_path): sha(proof_path)}
    artifact_pins.update({str(training / name): digest for name, digest in outputs.items()})
    return {"report": report, "report_path": report_path, "proof": proof,
        "proof_path": proof_path, "pins": pins, "artifact_pins": artifact_pins,
        "architecture": architecture, "residual_path": residual_path,
        "control_auth": control_auth}


def authorization_payload(training: Path, control: Path = DEFAULT_CONTROL) -> dict:
    """Build a model-free, target-free authorization after every source freezes."""
    training_auth = validate_training_artifacts(training, control)
    source_auth = validate_inputs(DEFAULT_INPUT, DEFAULT_PROOF)
    control_dev = validate_control_dev_capture()
    return {"schema": AUTHORIZATION_SCHEMA, "rows": EXPECTED_ROWS,
        "limits": EVAL_LIMITS, "target_cache_bytes": TARGET_CACHE_BYTES,
        "evaluator_sha256": sha(Path(__file__).resolve()),
        "training_report_sha256": sha(training_auth["report_path"]),
        "training_proof_sha256": sha(training_auth["proof_path"]),
        "control_training_report_sha256": sha(training_auth["control_auth"]["report_path"]),
        "source_authorization_sha256": canonical_hash(source_auth["pins"]),
        "historical_predictions_sha256": CONTROL_PREDICTIONS_SHA256,
        "historical_closure_sha256": CONTROL_CLOSURE_SHA256,
        "model_sha256": sha(ROOT / "hanglish/models/shared_edge_residual.py")}


def consume_authorization(path: Path, expected: dict) -> dict:
    path = Path(path).resolve()
    require(path.is_file(), "DEV prelaunch authorization is absent")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(value == expected, "DEV prelaunch authorization differs")
    return {str(path): sha(path)}


def run(training: Path, output: Path, authorization: Path,
        control: Path = DEFAULT_CONTROL) -> dict:
    output = Path(output); output.mkdir(parents=True, exist_ok=False); started = time.monotonic()
    try:
        training_auth = validate_training_artifacts(training, control)
        source_auth = validate_inputs(DEFAULT_INPUT, DEFAULT_PROOF)
        control_dev_auth = validate_control_dev_capture()
        authorization_pins = consume_authorization(
            authorization, authorization_payload(training, control))
        from safetensors.torch import load_file
        import torch
        from hanglish.models.sentence_ranker import SentenceRanker
        from hanglish.models.shared_edge_residual import FrozenEdgeFeatures, SharedEdgeResidualRanker
        from hanglish.models.source_span_residual import SourceSpanResidualRanker
        from hanglish.models.span_token_bridge import encode_aligned_candidates
        from scripts.preflight_pko_seq2seq import system_swap_bytes
        from scripts.source_span_resources import AtomicResourceWatchdog

        torch.set_num_threads(2); torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        pins = dict(training_auth["pins"]); pins.update(training_auth["artifact_pins"])
        pins.update(training_auth["control_auth"]["artifact_pins"])
        pins.update(source_auth["pins"]); pins.update(control_dev_auth["pins"])
        pins.update(authorization_pins)
        tracked = [Path(__file__).resolve(), PLAN,
            ROOT / "scripts/evaluate_local_span_objective.py",
            ROOT / "scripts/score_source_edge_recombination.py",
            ROOT / "scripts/train_shared_edge_residual.py",
            ROOT / "scripts/source_span_resources.py",
            ROOT / "hanglish/models/shared_edge_residual.py",
            ROOT / "hanglish/models/source_span_residual.py",
            ROOT / "hanglish/models/span_token_bridge.py",
            ROOT / "hanglish/core/source_correspondence.py"]
        pins.update({str(path): sha(path) for path in tracked})
        write(output / "pins.json", dict(sorted(pins.items())))
        baseline_swap = system_swap_bytes()
        watchdog = AtomicResourceWatchdog(started, baseline_swap, output / "resources.jsonl",
            interval_s=2.0, limits=EVAL_LIMITS)
        predictions = []; target_vectors = {}; target_flags = {}
        with watchdog:
            with watchdog.operation("load_parent_and_residuals", update=True):
                parent = SentenceRanker.load(PARENT)
                control_model = SourceSpanResidualRanker(parent)
                control_artifact = training_auth["control_auth"]["checkpoints"]["source_local"]
                control_model.load_residual_state_dict(load_file(control_artifact["residual_path"]))
                shared_model = SharedEdgeResidualRanker(parent)
                require(shared_model.architecture == training_auth["architecture"],
                        "shared-edge evaluation architecture differs")
                shared_model.load_residual_state_dict(load_file(training_auth["residual_path"]))
                control_model.eval(); shared_model.eval()
            capture_path = output / "predictions.jsonl"
            with capture_path.open("x", encoding="utf-8") as stream:
                rows = zip_longest(iter_jsonl(RUNTIME / "capture.jsonl"),
                    iter_jsonl(DEFAULT_INPUT / "capture.jsonl"),
                    iter_jsonl(control_dev_auth["predictions"]))
                for runtime_row, proposal_row, historical in rows:
                    require(runtime_row is not None and proposal_row is not None
                            and historical is not None, "DEV source denominator differs")
                    require(runtime_row.get("id") == proposal_row.get("id") == historical.get("id")
                            and runtime_row.get("request") == proposal_row.get("request")
                            == historical.get("request"), "DEV source identity differs")
                    texts, representatives, groups = collect_candidate_groups(runtime_row, proposal_row)
                    targets = isolated_target_texts(groups); source = runtime_row["request"]["input"]
                    reason = common_support_reason(source, texts, targets, lambda values: (
                        encode_aligned_candidates(parent.tokenizer, values, 256), True)[1])
                    captured_ms = runtime_row.get("elapsed_ms")
                    require(finite_nonnegative(captured_ms), "captured runtime latency differs")
                    latency = {"captured_runtime": float(captured_ms), "parent_features": 0.0,
                        "target_extraction": 0.0, "source_local_control": 0.0, "shared_edge": 0.0}
                    scores = shared_audit = None
                    misses = [text for text in targets if text not in target_vectors]
                    cache_counts = {"requested": len(targets),
                                    "hits": len(targets) - len(misses), "misses": len(misses)}
                    if reason is None:
                        before = time.monotonic()
                        with watchdog.operation("parent_features:" + runtime_row["id"], update=True):
                            base = control_model.extract_features(texts)
                        latency["parent_features"] = (time.monotonic() - before) * 1000
                        if misses:
                            before = time.monotonic()
                            with watchdog.operation("target_extraction:" + runtime_row["id"], update=True):
                                extend_target_cache(shared_model, misses,
                                                    target_vectors, target_flags)
                            latency["target_extraction"] = (time.monotonic() - before) * 1000
                        edge_features = FrozenEdgeFeatures(tuple(targets),
                            torch.stack([target_vectors[text] for text in targets]),
                            torch.stack([target_flags[text] for text in targets]))
                        scores = {"parent": base.logits.tolist()}
                        before = time.monotonic()
                        with watchdog.operation("source_local_control:" + runtime_row["id"], update=True), torch.no_grad():
                            control_values = control_model(source, groups, features=base)["scores"]
                        latency["source_local_control"] = (time.monotonic() - before) * 1000
                        scores["source_local_control"] = control_values.tolist()
                        before = time.monotonic()
                        with watchdog.operation("shared_edge:" + runtime_row["id"], update=True), torch.no_grad():
                            result = shared_model(source, groups, base_features=base,
                                                  edge_features=edge_features)
                        latency["shared_edge"] = (time.monotonic() - before) * 1000
                        scores["shared_edge"] = result["scores"].tolist()
                        shared_audit = {"edge_ids": [list(edge) for edge in result["edge_ids"]],
                            "edge_scores": result["edge_scores"].tolist(),
                            "residuals": result["residuals"].tolist(),
                            "winning_path_ids": [list(value) for value in result["winning_path_ids"]]}
                    prediction = build_prediction(runtime_row, texts, representatives, groups,
                        targets, scores, latency, reason, shared_audit, cache_counts)
                    verify_control_replay(prediction, historical)
                    predictions.append(prediction); append_jsonl(stream, prediction); stream.flush()
                    if len(predictions) % 25 == 0:
                        print(json.dumps({"captured": len(predictions), "rows": EXPECTED_ROWS,
                                          "elapsed_s": time.monotonic() - started}), flush=True)
                os.fsync(stream.fileno())
            require(len(predictions) == EXPECTED_ROWS, "DEV prediction denominator differs")
            capture_sha = sha(capture_path)
            closure = {"schema": SCHEMA + "-source-only-closure", "rows": EXPECTED_ROWS,
                "predictions_sha256": capture_sha, "source_only": True, "targets_joined": False,
                "candidate_policy": "recombined_pool_256", "common_support_rule": True,
                "full_candidate_limit": 256, "isolated_target_limit": 256,
                "source_byte_limit": 2048, "input_truncation": False}
            write(output / "predictions.complete.json", closure); closure_sha = sha(output / "predictions.complete.json")
            require(sha(DEV) == DEV_SHA256, "DEV targets changed before late join")
            labels = list(iter_jsonl(DEV))
            metrics = {arm: summarize(predictions, labels, arm) for arm in ARM_NAMES}
            require(metrics["captured_runtime"]["counts"]["top1"] == 264
                    and metrics["captured_runtime"]["counts"]["top5"] == 284,
                    "captured runtime baseline differs")
            decision = decide(metrics)
            protected_rows = [row for row in predictions if protected_fixtures(row["request"]["input"])]
            protected_summary = {"rows": len(protected_rows),
                "fixture_counts": dict(Counter(fixture["kind"] for row in protected_rows
                    for fixture in protected_fixtures(row["request"]["input"]))),
                "all_observed_candidate_paths_verified": all(row["protected_paths"]["verified"]
                                                               for row in protected_rows),
                "production_runtime_preservation_verified": False}
            protected_summary["response_failures"] = {arm: [row["id"] for row in protected_rows
                if not row["protected_responses"][arm]["verified"]] for arm in ARM_NAMES}
            final_resources = watchdog.sample_and_append("final")
            require(final_resources.get("stage") == "final"
                    and final_resources.get("operation") is None
                    and final_resources.get("operation_elapsed_s") is None
                    and finite_nonnegative(final_resources.get("elapsed_s"))
                    and type(final_resources.get("rss_bytes")) is int
                    and 0 <= final_resources["rss_bytes"] <= EVAL_LIMITS["rss_bytes"]
                    and type(final_resources.get("swap_used_bytes")) is int
                    and type(final_resources.get("swap_growth_bytes")) is int
                    and final_resources["swap_used_bytes"] - baseline_swap
                        == final_resources["swap_growth_bytes"]
                    and final_resources["swap_growth_bytes"]
                        <= EVAL_LIMITS["swap_growth_bytes"],
                    "final evaluation resource observation differs")
        for raw, digest in pins.items():
            require(sha(Path(raw)) == digest, "evaluation pin changed: " + raw)
        elapsed = time.monotonic() - started
        require(elapsed <= EVAL_LIMITS["whole_seconds"], "whole evaluation time exceeded")
        report = {"schema": SCHEMA, "completed": True, "split": "dev", "rows": EXPECTED_ROWS,
            "scope": {"source_only_predictions_closed_before_target_join": True,
                "dev_is_exposed_development": True, "test_opened": False,
                "production_changed": False, "input_truncation": False,
                "isolated_target_vectors_cached_by_text_only": True,
                "edge_scores_not_cached_across_requests": True},
            "metrics": metrics, "decision": decision,
            "protected_literal_contract": protected_summary,
            "target_cache": {"unique_texts": len(target_vectors),
                "limit_bytes": TARGET_CACHE_BYTES,
                "tensor_bytes": sum(value.numel() * value.element_size()
                    for value in [*target_vectors.values(), *target_flags.values()]),
                "requests": sum(row["target_cache"]["requested"] for row in predictions),
                "hits": sum(row["target_cache"]["hits"] for row in predictions),
                "misses": sum(row["target_cache"]["misses"] for row in predictions)},
            "capture_sha256": capture_sha, "closure_sha256": closure_sha,
            "training_report_sha256": sha(training_auth["report_path"]),
            "control_training_report_sha256": sha(training_auth["control_auth"]["report_path"]),
            "historical_control_predictions_sha256": CONTROL_PREDICTIONS_SHA256,
            "historical_control_closure_sha256": CONTROL_CLOSURE_SHA256,
            "dev_labels_sha256": DEV_SHA256, "pins": dict(sorted(pins.items())),
            "limits": EVAL_LIMITS, "baseline_swap_bytes": baseline_swap,
            "final_resources": final_resources, "elapsed_s": elapsed,
            "limitations": ["DEV metrics do not establish blind generalization",
                "the shared-edge bundle changes target context, position features, and normalization",
                "captured and rescored latency are not an apples-to-apples benchmark",
                "the historical control is descriptive and cannot be selected"]}
        write(output / "report.json", report)
        print(json.dumps({"completed": True,
            "metrics": {arm: value["counts"] for arm, value in metrics.items()},
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
    parser.add_argument("--control", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        require(args.authorization is None, "prepare-only does not consume authorization")
        write(args.output, authorization_payload(args.training, args.control))
    else:
        require(args.authorization is not None, "DEV prelaunch authorization required")
        run(args.training, args.output, args.authorization, args.control)
