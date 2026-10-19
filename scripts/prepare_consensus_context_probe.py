"""Prepare the fixed TRAIN-only data slice for the context consensus pilot."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_final_choice_data import read_runtime_evidence
from scripts.train_final_choice_selector import load_fit_role, validate_fit_pair


OUTPUT = ROOT / ".omx/artifacts/consensus-context-pilot-data-v1"
PLAN_SHA256 = "8329bc741f5c69cba6f8e779c9b5b2b0c2ef15b32b29f64776729ffbb077778e"
NATURAL_FIT = ROOT / "data/natural_context/final-choice-v1/natural_fit"
CHALLENGE_FIT = ROOT / "data/natural_context/final-choice-v1/challenge_fit"
NATURAL_MANIFEST_SHA256 = "fc22b1635cc496ed70aa757750944726d583d640f2d9f3a598e380f3e262a1c9"
CHALLENGE_MANIFEST_SHA256 = "2b8a6349e0ad091e797d86694c2a5165b81fa906a91c3d540ece9042b1b66194"
RUNTIME_EVIDENCE = ROOT / ".omx/artifacts/numeric-structure-integration-v1/report.json"
RUNTIME_EVIDENCE_SHA256 = "77c04334af817e15bc4c5ccc9d27b1d6ab9f9f6a759b4207e752762ae49d3aab"
SALT = "consensus-context-v1-2026-09-15"
LOWER_WORD = re.compile(r"[a-z]+\Z")
SELECTION_COUNTS = {"natural": 64, "challenge_single": 32,
                    "challenge_sentence": 32}
DEPENDENCIES = (
    "scripts/prepare_final_choice_data.py",
    "scripts/train_final_choice_selector.py",
    "scripts/consensus_context_policy.py",
)


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def _path_key(path):
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _valid_explicit_sha(value, label):
    if (not isinstance(value, str) or len(value) != 64
            or set(value) - set("0123456789abcdef")):
        raise ValueError(f"explicit {label} SHA-256 is required")


def selection_key(role, source_id, salt=SALT):
    if not all(isinstance(value, str) and value
               for value in (role, source_id, salt)):
        raise ValueError("selection identity must be nonempty strings")
    value = f"{salt}\0{role}\0{source_id}".encode()
    return hashlib.sha256(value).hexdigest(), source_id


def _source_identity(row):
    if not isinstance(row, dict):
        raise ValueError("pilot source row must be an object")
    for key in ("id", "group_id", "input", "context"):
        value = row.get(key)
        if not isinstance(value, str) or (key != "context" and not value):
            raise ValueError(f"pilot source has invalid {key}")
    return row["id"], row["input"], row["context"]


def select_pilot(natural_rows, challenge_rows, salt=SALT):
    """Select fixed rows using only IDs, group IDs, input, and context."""
    natural = []
    for row in natural_rows:
        source_id, _, _ = _source_identity(row)
        natural.append((selection_key("natural_fit", source_id, salt), row))
    singles, sentences = [], []
    for row in challenge_rows:
        source_id, source, context = _source_identity(row)
        item = (selection_key("challenge_fit", source_id, salt), row)
        if LOWER_WORD.fullmatch(source) and context.strip():
            singles.append(item)
        if any(character.isspace() for character in source):
            sentences.append(item)
    groups = {
        "natural": sorted(natural)[:SELECTION_COUNTS["natural"]],
        "challenge_single": sorted(singles)[:SELECTION_COUNTS["challenge_single"]],
        "challenge_sentence": sorted(sentences)[:SELECTION_COUNTS["challenge_sentence"]],
    }
    if any(len(groups[name]) != count for name, count in SELECTION_COUNTS.items()):
        raise ValueError("FIT captures do not contain the fixed pilot shapes")
    selected = []
    for name, role in (("natural", "natural_fit"),
                       ("challenge_single", "challenge_fit"),
                       ("challenge_sentence", "challenge_fit")):
        selected.extend((role, row) for _, row in groups[name])
    identities = [(role, row["id"]) for role, row in selected]
    if len(identities) != len(set(identities)):
        raise ValueError("pilot selection contains duplicate role IDs")
    return selected


def validate_runtime(natural, challenge, runtime):
    validate_fit_pair(natural, challenge)
    left, right = natural["manifest"], challenge["manifest"]
    sources = runtime.get("integrated_runtime_source_hashes")
    weights = runtime.get("accepted_weight_hashes")
    version = runtime.get("integrated_runtime_version")
    if (runtime.get("passed") is not True
            or not isinstance(sources, dict) or not sources
            or runtime.get("runtime_source_hashes_after") != sources
            or not isinstance(weights, dict) or len(weights) != 16
            or left.get("runtime_version") != version
            or right.get("runtime_version") != version
            or left.get("runtime_source_hashes") != sources
            or right.get("runtime_source_hashes") != sources
            or left.get("model_artifact_hashes") != weights
            or right.get("model_artifact_hashes") != weights):
        raise ValueError("FIT captures differ from the current runtime evidence")


def _merge_hashes(target, values, label):
    for name, value in values.items():
        if name in target and target[name] != value:
            raise ValueError(f"{label} source hash conflicts: {name}")
        target[name] = value


def provenance(natural, challenge, runtime, *, plan_path, plan_sha,
               runtime_path, runtime_sha, runner_sha):
    hashes = {}
    _merge_hashes(hashes, natural["manifest"]["source_hashes"], "natural FIT")
    _merge_hashes(hashes, challenge["manifest"]["source_hashes"], "challenge FIT")
    _merge_hashes(hashes, runtime["integrated_runtime_source_hashes"], "runtime")
    _merge_hashes(hashes, runtime["accepted_weight_hashes"], "runtime model")
    additions = {
        _path_key(plan_path): plan_sha,
        _path_key(runtime_path): runtime_sha,
        _path_key(NATURAL_FIT / "manifest.json"): NATURAL_MANIFEST_SHA256,
        _path_key(CHALLENGE_FIT / "manifest.json"): CHALLENGE_MANIFEST_SHA256,
        _path_key(NATURAL_FIT / "cases.jsonl"):
            natural["manifest"]["cases_sha256"],
        _path_key(NATURAL_FIT / "selection.jsonl"):
            natural["manifest"]["selection_sha256"],
        _path_key(CHALLENGE_FIT / "cases.jsonl"):
            challenge["manifest"]["cases_sha256"],
        _path_key(CHALLENGE_FIT / "selection.jsonl"):
            challenge["manifest"]["selection_sha256"],
        "scripts/prepare_consensus_context_probe.py": runner_sha,
    }
    additions.update({name: sha256_bytes((ROOT / name).read_bytes())
                      for name in DEPENDENCIES})
    _merge_hashes(hashes, additions, "pilot preparation")
    return dict(sorted(hashes.items()))


def verify_hashes(hashes):
    for name, expected in hashes.items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        if sha256_bytes(path.read_bytes()) != expected:
            raise ValueError(f"pilot preparation input changed: {name}")


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    _valid_explicit_sha(args.plan_sha256, "plan")
    if args.plan_sha256 != PLAN_SHA256:
        raise ValueError("plan hash is not the frozen consensus context plan")
    plan_raw = args.plan.read_bytes()
    if sha256_bytes(plan_raw) != args.plan_sha256:
        raise ValueError("consensus context plan hash differs")
    runner_raw = Path(__file__).read_bytes()
    runner_sha = sha256_bytes(runner_raw)
    runtime_raw, runtime = read_runtime_evidence(
        args.runtime_evidence, RUNTIME_EVIDENCE_SHA256)
    natural = load_fit_role(NATURAL_FIT, NATURAL_MANIFEST_SHA256, "natural_fit")
    challenge = load_fit_role(
        CHALLENGE_FIT, CHALLENGE_MANIFEST_SHA256, "challenge_fit")
    validate_runtime(natural, challenge, runtime)
    selected = select_pilot(natural["rows"], challenge["rows"])
    before = provenance(
        natural, challenge, runtime, plan_path=args.plan,
        plan_sha=args.plan_sha256, runtime_path=args.runtime_evidence,
        runtime_sha=sha256_bytes(runtime_raw), runner_sha=runner_sha)
    verify_hashes(before)

    cases_raw = b"".join((json.dumps(
        {"role": role, "row": row}, ensure_ascii=False, sort_keys=True
    ) + "\n").encode() for role, row in selected)
    sections = {
        "natural": selected[:64],
        "challenge_single": selected[64:96],
        "challenge_sentence": selected[96:128],
    }
    manifest = {
        "schema": "consensus-context-train-pilot-data-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed": True,
        "scope": "fixed source-only TRAIN pilot preparation; no model inference",
        "salt": SALT, "rows": len(selected), "cases_sha256": sha256_bytes(cases_raw),
        "selection": {
            name: {"role": rows[0][0], "shape": name,
                   "ids": [row["id"] for _, row in rows],
                   "group_ids": [row["group_id"] for _, row in rows],
                   "rows": len(rows)}
            for name, rows in sections.items()
        },
        "shape_counts": {name: len(rows) for name, rows in sections.items()},
        "natural_fit_manifest_sha256": NATURAL_MANIFEST_SHA256,
        "challenge_fit_manifest_sha256": CHALLENGE_MANIFEST_SHA256,
        "plan_path": _path_key(args.plan), "plan_sha256": args.plan_sha256,
        "runtime_evidence_path": _path_key(args.runtime_evidence),
        "runtime_evidence_sha256": sha256_bytes(runtime_raw),
        "runtime_version": runtime["integrated_runtime_version"],
        "runtime_source_hashes": runtime["integrated_runtime_source_hashes"],
        "model_artifact_hashes": runtime["accepted_weight_hashes"],
        "source_hashes": before,
        "selection_uses_labels_categories_eligibility_or_outcomes": False,
        "rows_preserved_without_candidate_or_label_mutation": True,
        "mlx_model_inference_run": False, "development_data_read": False,
        "final_data_read": False, "deployed": False,
    }
    args.output.mkdir(parents=True)
    (args.output / "cases.jsonl").write_bytes(cases_raw)
    after = provenance(
        natural, challenge, runtime, plan_path=args.plan,
        plan_sha=args.plan_sha256, runtime_path=args.runtime_evidence,
        runtime_sha=sha256_bytes(runtime_raw), runner_sha=runner_sha)
    verify_hashes(after)
    if before != after or sha256_bytes(Path(__file__).read_bytes()) != runner_sha:
        raise RuntimeError("pilot preparation inputs changed during execution")
    manifest["source_hashes_after"] = after
    (args.output / "manifest.json").write_text(json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--runtime-evidence", type=Path,
                        default=RUNTIME_EVIDENCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
