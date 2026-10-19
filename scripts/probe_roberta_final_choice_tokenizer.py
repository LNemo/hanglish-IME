"""Measure RoBERTa token-budget eligibility on the frozen FIT captures.

This diagnostic loads a tokenizer only. It does not load model weights, score
candidates, optimize parameters, or inspect evaluation/development/test data.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hanglish.core.neural_converter import file_sha256
from hanglish.models.sentence_ranker import REQUIRED_ARTIFACTS
from scripts.final_choice_policy import finite_candidate_indices
from scripts.train_final_choice_selector import load_fit_role, validate_fit_pair


CHECKPOINT = ROOT / ".omx/artifacts/roberta-context-ranker-v1/checkpoint"
METADATA_SHA256 = "29ad7b22c2d01e0caba34b3489aa909778359ea7025926ef0935e88bd0caed9e"
MODEL_SHA256 = "7e687917ba496d2cec61907c64bafdf88857d471fac2f7f43bfab86c687c6f0a"
BASE_HELPER_SHA256 = "e66d7fdef1461b39a2874654aca6be12487b8dc241405ab9694efbf934f92cc8"
FIT = {
    "natural_fit": (ROOT / "data/natural_context/final-choice-v1/natural_fit",
                    "fc22b1635cc496ed70aa757750944726d583d640f2d9f3a598e380f3e262a1c9"),
    "challenge_fit": (ROOT / "data/natural_context/final-choice-v1/challenge_fit",
                      "2b8a6349e0ad091e797d86694c2a5165b81fa906a91c3d540ece9042b1b66194"),
}
OUTPUT = ROOT / ".omx/artifacts/roberta-final-choice-tokenizer-v1"
PROFILE = {
    "name": "roberta-final-choice-fit-tokenizer-budget-v1",
    "rows": 3492,
    "max_length": 96,
    "threads": 2,
    "rss_limit_bytes": 3 * 1024**3,
    "time_limit_seconds": 300,
    "progress_interval": 128,
}


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def require_resources(started, stage):
    if (rss_bytes() > PROFILE["rss_limit_bytes"]
            or time.monotonic() - started > PROFILE["time_limit_seconds"]):
        raise RuntimeError(f"RoBERTa tokenizer probe resource bound exceeded during {stage}")


def load_tokenizer_checkpoint(checkpoint=CHECKPOINT, tokenizer_factory=None):
    """Validate the complete bundle, then instantiate only its tokenizer."""
    metadata_raw = (checkpoint / "metadata.json").read_bytes()
    if sha256_bytes(metadata_raw) != METADATA_SHA256:
        raise ValueError("RoBERTa checkpoint metadata hash differs")
    metadata = json.loads(metadata_raw)
    artifacts = metadata.get("artifact_hashes")
    if (metadata.get("format_version") != 1
            or metadata.get("base_model") != "klue/roberta-small"
            or metadata.get("max_length") != PROFILE["max_length"]
            or not isinstance(artifacts, dict)
            or set(artifacts) != REQUIRED_ARTIFACTS
            or artifacts.get("model.safetensors") != MODEL_SHA256):
        raise ValueError("RoBERTa checkpoint metadata contract differs")
    observed = {}
    for name, expected in artifacts.items():
        observed[name] = file_sha256(checkpoint / name)
        if observed[name] != expected:
            raise ValueError(f"RoBERTa checkpoint artifact differs: {name}")
    if tokenizer_factory is None:
        from transformers import AutoTokenizer
        tokenizer_factory = AutoTokenizer.from_pretrained
    tokenizer = tokenizer_factory(
        checkpoint, local_files_only=True, trust_remote_code=False,
    )
    return tokenizer, metadata_raw, metadata, observed


def normalized_token_length(tokenizer, text):
    if not isinstance(text, str):
        raise ValueError("candidate text must be a string")
    normalized = " ".join(text.strip().split())
    encoded = tokenizer(normalized, add_special_tokens=False, truncation=False)
    input_ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else None
    if (not isinstance(input_ids, list)
            or (input_ids and isinstance(input_ids[0], list))
            or any(type(value) is not int for value in input_ids)):
        raise ValueError("tokenizer returned malformed candidate token IDs")
    return len(input_ids)


def candidate_budget(row, tokenizer, max_length=PROFILE["max_length"]):
    """Compute complete candidate lengths without consulting stored labels."""
    candidates = row.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("FIT row candidates are missing")
    reserved = tokenizer.num_special_tokens_to_add(pair=True)
    if type(reserved) is not int or reserved < 1 or reserved >= max_length:
        raise ValueError("tokenizer pair-special-token budget is invalid")
    lengths = []
    for candidate in candidates:
        if not isinstance(candidate, dict) or not isinstance(candidate.get("text"), str):
            raise ValueError("FIT candidate object is malformed")
        lengths.append(normalized_token_length(tokenizer, candidate["text"]))
    return lengths, reserved, all(length + reserved <= max_length for length in lengths)


def diagnose_row(row, tokenizer, role):
    lengths, reserved, all_fit = candidate_budget(row, tokenizer)
    # Stored alignment is consulted only after the source-only token budget.
    finite = finite_candidate_indices(row["candidates"])
    stored_finite = row.get("finite_indices")
    accepted = row.get("accepted_positive_indices")
    if (stored_finite != finite
            or not isinstance(accepted, list)
            or any(type(index) is not int or index not in finite for index in accepted)
            or len(accepted) != len(set(accepted))):
        raise ValueError("FIT finite or accepted candidate indices differ")
    new_eligible = all_fit and len(finite) >= 2 and 0 in finite
    none_indices = [index for index, candidate in enumerate(row["candidates"])
                    if candidate.get("score") is None]
    return {
        "id": row["id"], "group_id": row["group_id"], "role": role,
        "candidate_count": len(lengths), "candidate_token_lengths": lengths,
        "pair_special_tokens": reserved,
        "none_score_indices": none_indices,
        "none_score_token_lengths": [lengths[index] for index in none_indices],
        "finite_indices": list(finite),
        "accepted_positive_indices": list(accepted),
        "accepted_positive_count": len(accepted),
        "old_eligible": row["selector_eligible"],
        "old_all_candidates_can_score": row["all_candidates_can_score"],
        "new_all_candidates_fit": all_fit,
        "new_eligible": new_eligible,
        "parent_correct": 0 in accepted,
    }


def summarize(rows):
    return {
        "rows": len(rows),
        "groups": len({(row["role"], row["group_id"]) for row in rows}),
        "old_eligible": sum(row["old_eligible"] for row in rows),
        "new_eligible": sum(row["new_eligible"] for row in rows),
        "old_to_new_eligible": sum(not row["old_eligible"] and row["new_eligible"]
                                   for row in rows),
        "old_to_new_ineligible": sum(row["old_eligible"] and not row["new_eligible"]
                                     for row in rows),
        "new_parent_correct": sum(row["new_eligible"] and row["parent_correct"]
                                  for row in rows),
        "new_parent_error": sum(row["new_eligible"] and not row["parent_correct"]
                                and row["accepted_positive_count"] > 0 for row in rows),
        "new_eligible_without_positive": sum(row["new_eligible"]
                                             and row["accepted_positive_count"] == 0
                                             for row in rows),
        "rows_with_none_original": sum(bool(row["none_score_indices"]) for row in rows),
        "max_candidate_tokens": max((max(row["candidate_token_lengths"])
                                     for row in rows), default=0),
    }


def source_hashes(checkpoint, loaded, runner_sha, fit_data):
    hashes = dict(fit_data["natural_fit"]["manifest"]["source_hashes"])
    additions = {
        "scripts/train_final_choice_selector.py": BASE_HELPER_SHA256,
        "scripts/probe_roberta_final_choice_tokenizer.py": runner_sha,
    }
    for role, value in fit_data.items():
        directory = value["directory"]
        additions[str((directory / "manifest.json").relative_to(ROOT))] = FIT[role][1]
        additions[str((directory / "cases.jsonl").relative_to(ROOT))] = sha256_bytes(value["cases_raw"])
        additions[str((directory / "selection.jsonl").relative_to(ROOT))] = sha256_bytes(value["selection_raw"])
    additions[str((checkpoint / "metadata.json").relative_to(ROOT))] = METADATA_SHA256
    for name, digest in loaded.items():
        additions[str((checkpoint / name).relative_to(ROOT))] = digest
    for name, digest in additions.items():
        if name in hashes and hashes[name] != digest:
            raise ValueError(f"conflicting RoBERTa tokenizer probe source hash: {name}")
        hashes[name] = digest
    return dict(sorted(hashes.items()))


def verify_hashes(hashes):
    for name, expected in hashes.items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        if file_sha256(path) != expected:
            raise ValueError(f"RoBERTa tokenizer probe input changed: {name}")


def run(args, *, fit_loader=load_fit_role, tokenizer_loader=load_tokenizer_checkpoint):
    started = time.monotonic()
    if args.output.exists():
        raise FileExistsError(args.output)
    os.environ["OMP_NUM_THREADS"] = str(PROFILE["threads"])
    runner_source = Path(__file__).read_bytes()
    runner_sha = sha256_bytes(runner_source)
    fit_data = {role: fit_loader(directory, digest, role)
                for role, (directory, digest) in FIT.items()}
    validate_fit_pair(fit_data["natural_fit"], fit_data["challenge_fit"])
    tokenizer, metadata_raw, metadata, artifact_hashes = tokenizer_loader(args.checkpoint)
    rows = []
    for role in FIT:
        for index, row in enumerate(fit_data[role]["rows"], 1):
            rows.append(diagnose_row(row, tokenizer, role))
            if index % PROFILE["progress_interval"] == 0:
                require_resources(started, "token length diagnostics")
    if len(rows) != PROFILE["rows"]:
        raise ValueError("frozen FIT row count differs")
    hashes = source_hashes(args.checkpoint, artifact_hashes, runner_sha, fit_data)
    verify_hashes(hashes)
    summary = summarize(rows)
    args.output.mkdir(parents=True)
    diagnostic_raw = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode()
        for row in rows
    )
    (args.output / "diagnostics.jsonl").write_bytes(diagnostic_raw)
    (args.output / "source.py").write_bytes(runner_source)
    verify_hashes(hashes)
    if (Path(__file__).read_bytes() != runner_source
            or (args.checkpoint / "metadata.json").read_bytes() != metadata_raw):
        raise RuntimeError("RoBERTa tokenizer probe sources changed")
    require_resources(started, "terminal report")
    report = {
        "schema": "roberta-final-choice-fit-tokenizer-budget-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed": True, "profile": PROFILE, "summary": summary,
        "fit_manifest_sha256": {role: digest for role, (_, digest) in FIT.items()},
        "checkpoint": str(args.checkpoint),
        "checkpoint_metadata_sha256": METADATA_SHA256,
        "checkpoint_model_sha256": MODEL_SHA256,
        "checkpoint_artifact_hashes": artifact_hashes,
        "tokenizer_base_model": metadata["base_model"],
        "diagnostics_sha256": sha256_bytes(diagnostic_raw),
        "source_snapshot_sha256": runner_sha,
        "source_hashes": hashes, "source_hashes_after": hashes,
        "model_weights_loaded": False, "model_inference_run": False,
        "optimization_run": False, "evaluation_data_read": False,
        "development_data_read": False, "final_data_read": False,
        "selection_uses_labels": False,
        "purpose": "resource and recipe planning only; not a model-quality measurement",
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": rss_bytes(),
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    run(parser.parse_args())
