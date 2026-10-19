"""Train the frozen-candidate selector on the two fixed FIT captures only."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version as package_version
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from hanglish.core.neural_converter import file_sha256
from hanglish.models.sentence_ranker import SentenceRanker, encode_sentence_pairs
from scripts.final_choice_capture import _validate_segments, validate_response
from scripts.final_choice_policy import finite_candidate_indices
from scripts.prepare_final_choice_data import PARTITION_SHA256
from scripts.train_natural_sentence_ranker import rss_bytes
from scripts.train_sentence_ranker import _optimizer, _set_trainable_layers


PLAN = ROOT / ".omx/artifacts/final-choice-learning-plan-v1.md"
PLAN_SHA256 = "9adcb7c6d0fbbe596e44cf734d5b3d7b269c7709ca084e82fd2e36783dc71c49"
PARENT = ROOT / "checkpoints/sentence_ranker"
PARENT_WEIGHT = "a68f3bd26733e9bc897a182f78b1decafefa3582fa82b6b2cfcaa34672412ffa"
OUTPUT = ROOT / ".omx/artifacts/final-choice-selector-v1"
ROLE_SPECS = {"natural_fit": 2048, "challenge_fit": 1444}
RECIPE = {
    "name": "final-choice-selector-fit-v1",
    "seed": 23,
    "epochs": 1,
    "top_layers": 2,
    "encoder_lr": 1e-5,
    "head_lr": 5e-5,
    "weight_decay": .01,
    "max_length": 96,
    "inference_batch_size": 8,
    "group_batch": 4,
    "gradient_clip_norm": 1.,
    "parent_error_weight": 3.,
    "other_weight": 1.,
    "distillation_weight": 1.,
    "temperature": 2.,
    "threads": 2,
    "rss_limit_bytes": 3 * 1024**3,
    "training_seconds_limit": 1800,
}


@dataclass(frozen=True)
class SelectorExample:
    case_id: str
    group_id: str
    role: str
    context: str
    candidates: tuple[str, ...]
    positives: tuple[int, ...]
    teacher_scores: tuple[float, ...]
    finite_source_indices: tuple[int, ...]
    parent_correct: bool
    loss_weight: float


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _valid_hashes(values):
    return (isinstance(values, dict) and bool(values)
            and all(isinstance(name, str) and name
                    and isinstance(digest, str) and len(digest) == 64
                    and not set(digest) - set("0123456789abcdef")
                    for name, digest in values.items()))


def _read_json_bytes(path, expected_sha):
    raw = path.read_bytes()
    if sha256_bytes(raw) != expected_sha:
        raise ValueError(f"final-choice capture hash differs: {path}")
    return raw, json.loads(raw)


def _read_jsonl_bytes(path, expected_sha):
    raw = path.read_bytes()
    if sha256_bytes(raw) != expected_sha:
        raise ValueError(f"final-choice capture hash differs: {path}")
    return raw, [json.loads(line) for line in raw.splitlines() if line]


def _validate_row(row, selection, role, runtime_version, model_artifacts):
    identity = {key: row.get(key) for key in
                ("id", "group_id", "origin", "input", "context", "expected")}
    if identity != {key: selection.get(key) for key in identity}:
        raise ValueError("FIT captured row differs from its frozen selection")
    expected_origin = "natural" if role == "natural_fit" else "challenge"
    candidates = row.get("candidates")
    if (not isinstance(candidates, list) or not 1 <= len(candidates) <= 5
            or row.get("origin") != expected_origin
            or row.get("runtime_version") != runtime_version
            or row.get("result") != candidates[0].get("text")
            or not isinstance(row.get("expected"), list) or not row["expected"]
            or any(not isinstance(value, str) or not value for value in row["expected"])
            or type(row.get("engine_ms")) not in (int, float)
            or not math.isfinite(row["engine_ms"]) or row["engine_ms"] < 0
            or type(row.get("raw_engine_ms")) not in (int, float)
            or not math.isfinite(row["raw_engine_ms"]) or row["raw_engine_ms"] < 0):
        raise ValueError("FIT captured row identity or candidates differ")
    texts = []
    for candidate in candidates:
        if not isinstance(candidate, dict) or not isinstance(candidate.get("text"), str):
            raise ValueError("FIT candidate object is malformed")
        score = candidate.get("score")
        if (score is not None and (type(score) not in (int, float)
                or not math.isfinite(score) or round(score, 4) != score)):
            raise ValueError("FIT candidate score differs from returned finite score")
        _validate_segments(candidate, row["input"])
        if score is None and candidate != {
                "text": row["input"], "score": None,
                "segments": [{"input_start": 0, "input_end": len(row["input"]),
                              "source": row["input"], "text": row["input"],
                              "kind": "raw"}]}:
            raise ValueError("FIT None score is not the complete raw candidate")
        texts.append(candidate["text"])
    if len(texts) != len(set(texts)):
        raise ValueError("FIT candidate texts are duplicated")
    finite = finite_candidate_indices(candidates)
    positives = [index for index in finite if texts[index] in row["expected"]]
    can_score = row.get("candidate_can_score")
    if (not isinstance(can_score, list) or len(can_score) != len(candidates)
            or any(type(value) is not bool for value in can_score)):
        raise ValueError("FIT candidate budget flags are malformed")
    reasons = []
    if not all(can_score):
        reasons.append("candidate_exceeds_sentence_model_budget")
    if len(finite) < 2 or 0 not in finite:
        reasons.append("insufficient_finite_candidates")
    if (row.get("finite_indices") != finite
            or row.get("accepted_positive_indices") != positives
            or row.get("label_aligned") is not bool(positives)
            or row.get("all_candidates_can_score") is not all(can_score)
            or row.get("selector_eligible") is not (not reasons)
            or row.get("ineligible_reasons") != reasons):
        raise ValueError("FIT candidate eligibility or labels differ")
    flags = row.get("response_flags", {})
    artifact_flags = row.get("model_artifact_flags", {})
    if (flags.get("version") != runtime_version
            or flags.get("model_artifacts") != model_artifacts
            or flags.get("raw") is not False
            or artifact_flags != {"expected_count": len(model_artifacts),
                                  "response_matches_expected": True}):
        raise ValueError("FIT runtime model identity differs")
    validate_response(
        row.get("raw_bypass_response"), row["input"], row["context"],
        runtime_version, model_artifacts, raw=True,
    )


def load_fit_role(directory, expected_manifest_sha, expected_role):
    if expected_role not in ROLE_SPECS:
        raise ValueError("selector trainer accepts FIT roles only")
    manifest_raw, manifest = _read_json_bytes(
        directory / "manifest.json", expected_manifest_sha
    )
    cases_raw, rows = _read_jsonl_bytes(
        directory / "cases.jsonl", manifest.get("cases_sha256")
    )
    selection_raw, selection = _read_jsonl_bytes(
        directory / "selection.jsonl", manifest.get("selection_sha256")
    )
    count = ROLE_SPECS[expected_role]
    ordered_ids = manifest.get("partition_ordered_ids")
    if (manifest.get("completed") is not True or manifest.get("role") != expected_role
            or manifest.get("scope") != "fresh actual parent TRAIN capture for final-choice learning"
            or manifest.get("partition_manifest_sha256") != PARTITION_SHA256
            or manifest.get("expected_rows") != count or manifest.get("rows") != count
            or len(rows) != count or len(selection) != count
            or not isinstance(ordered_ids, list) or len(ordered_ids) != count
            or [row.get("id") for row in rows] != ordered_ids
            or [row.get("id") for row in selection] != ordered_ids
            or len(set(ordered_ids)) != count
            or manifest.get("selection_sha256_before_inference")
               != manifest.get("selection_sha256")
            or manifest.get("source_hashes") != manifest.get("source_hashes_after")
            or manifest.get("training_run") is not False
            or manifest.get("new_selector_inference_run") is not False
            or manifest.get("development_data_read") is not False
            or manifest.get("final_data_read") is not False
            or manifest.get("deployed") is not False):
        raise ValueError("completed exact FIT capture manifest is required")
    runtime_version = manifest.get("runtime_version")
    model_artifacts = manifest.get("model_artifact_hashes")
    runtime_sources = manifest.get("runtime_source_hashes")
    source_hashes = manifest.get("source_hashes")
    train_inputs = manifest.get("train_inputs")
    runtime_path = manifest.get("runtime_evidence_path")
    runtime_sha = manifest.get("runtime_evidence_sha256")
    if (not isinstance(runtime_version, str) or not runtime_version
            or not _valid_hashes(model_artifacts) or len(model_artifacts) != 16
            or not _valid_hashes(runtime_sources)
            or not _valid_hashes(source_hashes)
            or any(source_hashes.get(name) != digest
                   for name, digest in {**runtime_sources, **model_artifacts}.items())
            or not isinstance(train_inputs, dict) or not train_inputs
            or any(not isinstance(snapshot, dict)
                   or source_hashes.get(snapshot.get("path")) != snapshot.get("sha256")
                   for snapshot in train_inputs.values())
            or not isinstance(runtime_path, str) or not runtime_path
            or not isinstance(runtime_sha, str) or len(runtime_sha) != 64):
        raise ValueError("FIT capture runtime provenance is malformed")
    evidence_path = Path(runtime_path)
    if evidence_path.is_absolute():
        try:
            evidence_key = str(evidence_path.resolve().relative_to(ROOT))
        except ValueError:
            evidence_key = str(evidence_path.resolve())
    else:
        evidence_key = str(evidence_path)
    if source_hashes.get(evidence_key) != runtime_sha:
        raise ValueError("FIT runtime evidence is absent from its source hashes")
    for row, selected in zip(rows, selection):
        _validate_row(row, selected, expected_role, runtime_version, model_artifacts)
    diagnostics = {
        "top1": sum(row["result"] in row["expected"] for row in rows),
        "top5": sum(any(candidate["text"] in row["expected"]
                        for candidate in row["candidates"]) for row in rows),
        "label_aligned": sum(row["label_aligned"] for row in rows),
        "selector_eligible": sum(row["selector_eligible"] for row in rows),
        "all_candidates_can_score": sum(row["all_candidates_can_score"] for row in rows),
    }
    if (manifest.get("diagnostics") != diagnostics
            or manifest.get("baseline_accuracy_is_diagnostic_only") is not True
            or manifest.get("split_or_threshold_selected_from_capture_outcomes") is not False):
        raise ValueError("FIT capture diagnostics do not reproduce from cases")
    return {
        "directory": directory,
        "manifest_raw": manifest_raw,
        "cases_raw": cases_raw,
        "selection_raw": selection_raw,
        "manifest": manifest,
        "rows": rows,
    }


def validate_fit_pair(natural, challenge):
    left, right = natural["manifest"], challenge["manifest"]
    if (left["role"] != "natural_fit" or right["role"] != "challenge_fit"
            or left["partition_manifest_sha256"] != right["partition_manifest_sha256"]
            or left["runtime_version"] != right["runtime_version"]
            or left["runtime_source_hashes"] != right["runtime_source_hashes"]
            or left["model_artifact_hashes"] != right["model_artifact_hashes"]
            or left["source_hashes"] != right["source_hashes"]
            or left["train_inputs"] != right["train_inputs"]
            or set(left["partition_ordered_ids"]) & set(right["partition_ordered_ids"])):
        raise ValueError("natural and challenge FIT captures have incompatible lineage")


def build_example(row, role):
    finite = tuple(row["finite_indices"])
    positives_source = tuple(row["accepted_positive_indices"])
    if (not row["selector_eligible"] or not row["all_candidates_can_score"]
            or len(finite) < 2 or 0 not in finite or not positives_source):
        return None
    position = {source_index: index for index, source_index in enumerate(finite)}
    if any(index not in position for index in positives_source):
        raise ValueError("positive candidate is absent from the finite selector set")
    candidates = tuple(row["candidates"][index]["text"] for index in finite)
    teacher = tuple(float(row["candidates"][index]["score"]) for index in finite)
    positives = tuple(position[index] for index in positives_source)
    parent_correct = 0 in positives_source
    return SelectorExample(
        row["id"], row["group_id"], role, row["context"], candidates,
        positives, teacher, finite, parent_correct,
        RECIPE["other_weight"] if parent_correct else RECIPE["parent_error_weight"],
    )


def selector_loss(scores, examples, recipe=RECIPE):
    cursor, weighted_losses, weights = 0, [], []
    for example in examples:
        group = scores[cursor:cursor + len(example.candidates)]
        cursor += len(example.candidates)
        if len(group) != len(example.candidates):
            raise ValueError("selector logits do not cover the complete finite candidate set")
        positives = torch.tensor(example.positives, device=group.device)
        loss = torch.logsumexp(group, 0) - torch.logsumexp(group[positives], 0)
        if example.parent_correct:
            temperature = recipe["temperature"]
            teacher = group.new_tensor(example.teacher_scores)
            kd = torch.nn.functional.kl_div(
                torch.log_softmax(group / temperature, dim=0),
                torch.softmax(teacher / temperature, dim=0), reduction="sum",
            )
            loss = loss + recipe["distillation_weight"] * temperature**2 * kd
        weighted_losses.append(example.loss_weight * loss)
        weights.append(example.loss_weight)
    if cursor != len(scores) or not weighted_losses:
        raise ValueError("selector groups must consume every logit")
    return torch.stack(weighted_losses).sum() / scores.new_tensor(weights).sum()


def training_order(examples, seed=RECIPE["seed"]):
    order = list(examples)
    random.Random(seed).shuffle(order)
    return order


def validate_scores(scores, expected_count):
    if (not isinstance(scores, (list, tuple)) or len(scores) != expected_count
            or not scores or any(type(value) not in (int, float)
                                 or not math.isfinite(value) for value in scores)):
        raise ValueError("selector scores must be finite and cover every candidate")
    return scores


def score_parity_delta(memory, saved, examples):
    if len(memory) != len(examples) or len(saved) != len(examples) or not examples:
        raise ValueError("selector parity groups differ")
    deltas = []
    for left, right, example in zip(memory, saved, examples):
        validate_scores(left, len(example.candidates))
        validate_scores(right, len(example.candidates))
        deltas.extend(abs(a - b) for a, b in zip(left, right))
    delta = max(deltas)
    if not math.isfinite(delta) or delta > 1e-4:
        raise RuntimeError("saved final-choice selector differs from memory")
    return delta


def require_finite_parameters(model):
    if any(not bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
        raise RuntimeError("nonfinite final-choice selector parameters")


@torch.inference_mode()
def evaluate_examples(ranker, examples):
    ranker.model.eval()
    outcomes = []
    for example in examples:
        scores = ranker.score_pairs(
            [example.context] * len(example.candidates), list(example.candidates)
        )
        validate_scores(scores, len(example.candidates))
        chosen = min(range(len(scores)), key=lambda index: (-scores[index], index))
        outcomes.append({"id": example.case_id, "role": example.role,
                         "parent_correct": example.parent_correct,
                         "chosen_finite_index": chosen,
                         "chosen_source_index": example.finite_source_indices[chosen],
                         "correct": chosen in example.positives})
    return {"rows": len(outcomes), "correct": sum(row["correct"] for row in outcomes),
            "outcomes": outcomes}


def configure_trainable_model(model):
    return _set_trainable_layers(model, RECIPE["top_layers"])


def merge_hashes(target, additions, label):
    for name, digest in additions.items():
        if name in target and target[name] != digest:
            raise ValueError(f"conflicting {label} hash: {name}")
        target[name] = digest


def provenance_hashes(natural, challenge, trainer_sha):
    hashes = {}
    for capture in (natural, challenge):
        merge_hashes(hashes, capture["manifest"]["source_hashes"], "capture source")
        relative = str(capture["directory"].relative_to(ROOT))
        hashes[f"{relative}/manifest.json"] = sha256_bytes(capture["manifest_raw"])
        hashes[f"{relative}/cases.jsonl"] = sha256_bytes(capture["cases_raw"])
        hashes[f"{relative}/selection.jsonl"] = sha256_bytes(capture["selection_raw"])
    dependencies = {
        str(PLAN.relative_to(ROOT)): PLAN_SHA256,
        "scripts/train_final_choice_selector.py": trainer_sha,
        "scripts/final_choice_capture.py": file_sha256(ROOT / "scripts/final_choice_capture.py"),
        "scripts/final_choice_policy.py": file_sha256(ROOT / "scripts/final_choice_policy.py"),
        "scripts/prepare_final_choice_data.py": file_sha256(ROOT / "scripts/prepare_final_choice_data.py"),
        "scripts/train_sentence_ranker.py": file_sha256(ROOT / "scripts/train_sentence_ranker.py"),
        "hanglish/models/sentence_ranker.py": file_sha256(ROOT / "hanglish/models/sentence_ranker.py"),
    }
    merge_hashes(hashes, dependencies, "trainer dependency")
    return dict(sorted(hashes.items()))


def verify_hashes(hashes):
    observed = {}
    for name, expected in hashes.items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        observed[name] = file_sha256(path)
        if observed[name] != expected:
            raise ValueError(f"selector training input changed: {name}")
    return observed


def require_resources(started, stage):
    if (rss_bytes() > RECIPE["rss_limit_bytes"]
            or time.monotonic() - started > RECIPE["training_seconds_limit"]):
        raise RuntimeError(f"final-choice selector resource bound exceeded during {stage}")


def train(args):
    started = time.monotonic()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = Path(__file__).read_bytes()
    source_sha = sha256_bytes(source)
    if file_sha256(PLAN) != PLAN_SHA256:
        raise ValueError("final-choice learning plan differs")
    natural = load_fit_role(args.natural_fit, args.natural_manifest_sha256, "natural_fit")
    challenge = load_fit_role(
        args.challenge_fit, args.challenge_manifest_sha256, "challenge_fit"
    )
    validate_fit_pair(natural, challenge)
    before = provenance_hashes(natural, challenge, source_sha)
    verify_hashes(before)
    if file_sha256(PARENT / "model.safetensors") != PARENT_WEIGHT:
        raise ValueError("final-choice selector parent checkpoint differs")
    require_resources(started, "input validation")

    examples, eligibility = [], {}
    for role, capture in (("natural_fit", natural), ("challenge_fit", challenge)):
        role_examples = [example for row in capture["rows"]
                         if (example := build_example(row, role)) is not None]
        examples.extend(role_examples)
        eligibility[role] = {
            "rows": len(capture["rows"]), "eligible_with_positive": len(role_examples),
            "missing_eligible_positive": sum(
                row["selector_eligible"] and not row["accepted_positive_indices"]
                for row in capture["rows"]
            ),
            "ineligible": sum(not row["selector_eligible"] for row in capture["rows"]),
            "parent_correct": sum(example.parent_correct for example in role_examples),
            "parent_error": sum(not example.parent_correct for example in role_examples),
            "distilled": sum(example.parent_correct for example in role_examples),
        }
    if not examples:
        raise ValueError("no eligible FIT selector examples")

    torch.set_num_threads(RECIPE["threads"])
    os.environ["OMP_NUM_THREADS"] = str(RECIPE["threads"])
    random.seed(RECIPE["seed"])
    torch.manual_seed(RECIPE["seed"])
    ranker = SentenceRanker.load(PARENT)
    if ranker.max_length != RECIPE["max_length"]:
        raise ValueError("selector parent sentence length differs")
    ranker.batch_size = RECIPE["inference_batch_size"]
    parent = evaluate_examples(ranker, examples)
    require_resources(started, "parent FIT evaluation")

    args.output.mkdir(parents=True)
    layers = configure_trainable_model(ranker.model)
    optimizer = _optimizer(ranker.model, SimpleNamespace(**RECIPE), head_only=False)
    order = training_order(examples)
    losses = []
    ranker.model.train()
    training_started = time.monotonic()
    for start in range(0, len(order), RECIPE["group_batch"]):
        batch = order[start:start + RECIPE["group_batch"]]
        candidates = [candidate for example in batch for candidate in example.candidates]
        contexts = [example.context for example in batch for _ in example.candidates]
        encoded = encode_sentence_pairs(
            ranker.tokenizer, candidates, contexts, RECIPE["max_length"]
        )
        optimizer.zero_grad(set_to_none=True)
        scores = ranker.model(**encoded).logits.reshape(-1)
        loss = selector_loss(scores, batch)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonfinite final-choice selector loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            ranker.model.parameters(), RECIPE["gradient_clip_norm"]
        )
        if not bool(torch.isfinite(norm)):
            raise RuntimeError("nonfinite final-choice selector gradients")
        optimizer.step()
        require_finite_parameters(ranker.model)
        losses.append(float(loss.detach()))
        require_resources(started, "training")
    training_seconds = time.monotonic() - training_started
    ranker.model.eval()

    bundle = args.output / "checkpoint"
    ranker.model.save_pretrained(bundle, safe_serialization=True)
    ranker.tokenizer.save_pretrained(bundle)
    artifacts = {path.name: file_sha256(path) for path in bundle.iterdir() if path.is_file()}
    parent_metadata = ranker.metadata
    metadata = {key: parent_metadata[key] for key in
                ("base_model", "base_revision", "base_source", "license", "candidate_generator")
                if key in parent_metadata}
    metadata.update(
        format_version=1, created_at=datetime.now(timezone.utc).isoformat(),
        artifact_hashes=artifacts, max_length=RECIPE["max_length"],
        inference_batch_size=RECIPE["inference_batch_size"],
        parent_checkpoint_sha256=PARENT_WEIGHT, parent_metadata=parent_metadata,
        parent_runtime_version=natural["manifest"]["runtime_version"],
        fit_captures={role: {"path": str(capture["directory"]),
                             "manifest_sha256": sha256_bytes(capture["manifest_raw"])}
                      for role, capture in (("natural_fit", natural),
                                            ("challenge_fit", challenge))},
        frozen_source_hashes=before, training=RECIPE,
        training_eligibility=eligibility, trainable_layers=layers,
        selector_only=True, calibration_data_read=False, holdout_data_read=False,
        preservation_data_read=False, development_data_read=False,
        final_data_read=False, deployed=False,
        library_versions={name: package_version(name) for name in
                          ("torch", "transformers", "tokenizers", "safetensors")},
    )
    (bundle / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    loaded = SentenceRanker.load(bundle)
    parity_examples = examples[:4]
    memory = [ranker.score_pairs([example.context] * len(example.candidates),
                                 list(example.candidates)) for example in parity_examples]
    saved = [loaded.score_pairs([example.context] * len(example.candidates),
                                list(example.candidates)) for example in parity_examples]
    parity_delta = score_parity_delta(memory, saved, parity_examples)
    final = evaluate_examples(loaded, examples)
    require_resources(started, "saved FIT evaluation")

    load_fit_role(args.natural_fit, args.natural_manifest_sha256, "natural_fit")
    load_fit_role(args.challenge_fit, args.challenge_manifest_sha256, "challenge_fit")
    after = provenance_hashes(natural, challenge, source_sha)
    verify_hashes(after)
    if (after != before or file_sha256(PARENT / "model.safetensors") != PARENT_WEIGHT
            or file_sha256(Path(__file__)) != source_sha):
        raise RuntimeError("final-choice selector sources or parent changed")
    source_path = args.output / "trainer-source.py"
    source_path.write_bytes(source)
    if file_sha256(source_path) != source_sha:
        raise RuntimeError("final-choice selector source snapshot differs")
    require_resources(started, "post-validation")
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "FIT-only final-candidate selector training; no deployment decision",
        "recipe": RECIPE, "runtime_version": natural["manifest"]["runtime_version"],
        "partition_manifest_sha256": PARTITION_SHA256,
        "fit_manifest_sha256": {
            "natural_fit": sha256_bytes(natural["manifest_raw"]),
            "challenge_fit": sha256_bytes(challenge["manifest_raw"]),
        },
        "source_hashes": before, "source_hashes_after": after,
        "parent_checkpoint_sha256": PARENT_WEIGHT,
        "checkpoint_sha256": artifacts["model.safetensors"],
        "eligibility": eligibility, "parent_fit": parent, "trained_fit": final,
        "optimizer_steps": len(losses), "mean_loss": sum(losses) / len(losses),
        "training_seconds": training_seconds,
        "save_load_score_max_delta": parity_delta,
        "peak_rss_bytes": rss_bytes(), "elapsed_seconds": time.monotonic() - started,
        "selector_only": True, "calibration_data_read": False,
        "holdout_data_read": False, "preservation_data_read": False,
        "development_data_read": False, "final_data_read": False,
        "eligible_for_deployment": False, "deployed": False,
        "next_gate": "Freeze the selector, then train the separate accept/stay gate on CAL only.",
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({"optimizer_steps": len(losses), "eligibility": eligibility,
                      "parent_fit_correct": parent["correct"],
                      "trained_fit_correct": final["correct"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--natural-fit", type=Path, required=True)
    parser.add_argument("--natural-manifest-sha256", required=True)
    parser.add_argument("--challenge-fit", type=Path, required=True)
    parser.add_argument("--challenge-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    train(parser.parse_args())
