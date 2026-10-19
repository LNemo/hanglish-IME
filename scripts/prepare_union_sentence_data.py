"""Prepare sentence-ranker data from a gated parent-preserving proposal union."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hanglish.core.neural_converter import NeuralConverter, file_sha256
from hanglish.models.semantic_ranker import SemanticRanker
from scripts.context_proposal_union import propose
from scripts.prepare_context_training import (
    OUTPUT as CONTEXT_DATA,
    RUNTIME_VERSION,
    source_hashes as context_source_hashes,
    validate_context_data,
)
from scripts.probe_context_proposal_union import (
    PROFILE as PROBE_PROFILE,
    STUDENT,
    STUDENT_WEIGHT,
    hashes as probe_source_hashes,
)


CONTEXT_MANIFEST = "7c1c03205dbc9c8983c2b3dee37e4a6bcabb04d45a265e9e1699460595275b10"
PROBE = ROOT / ".omx/artifacts/context-proposal-union-v1"
OUTPUT = ROOT / "data/natural_context/union-sentence-v1"
ROLES = ("natural_train", "natural_dev", "retention_train")
PROFILE = {
    "name": "union-sentence-source-only-v1",
    "natural_train": 512,
    "natural_dev": 128,
    "retention_train": 1820,
    "candidate_limit": 72,
    "parent_candidate_limit": 48,
    "threads": 2,
    "rss_limit_bytes": 3 * 1024**3,
    "time_limit_seconds": 1800,
    "progress_interval": 64,
    "natural_train_proposals": "validated completed probe reuse",
    "other_proposals": "fresh source-only union",
}
SOURCE_KEYS = ("id", "origin", "split", "group_id", "input", "context", "expected")


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def source_hashes():
    hashes = context_source_hashes()
    paths = [
        Path(__file__).resolve(),
        ROOT / "scripts/context_proposal_union.py",
        ROOT / "scripts/probe_context_proposal_union.py",
        PROBE / "cases.jsonl",
        PROBE / "report.json",
    ]
    paths.extend(path for path in STUDENT.iterdir() if path.is_file())
    paths.extend(CONTEXT_DATA / f"{role}.jsonl" for role in ROLES)
    paths.append(CONTEXT_DATA / "manifest.json")
    for directory in (ROOT / "checkpoints/semantic_ranker", ROOT / "checkpoints/sentence_ranker"):
        paths.extend(path for path in directory.iterdir() if path.is_file())
    for path in sorted(set(paths)):
        hashes[str(path.relative_to(ROOT))] = file_sha256(path)
    return dict(sorted(hashes.items()))


def load_probe(probe_report_sha256, joint_data):
    report_path = PROBE / "report.json"
    if (not isinstance(probe_report_sha256, str)
            or file_sha256(report_path) != probe_report_sha256):
        raise ValueError("proposal probe report differs from explicit trusted hash")
    report = json.loads(report_path.read_text())
    if (report.get("profile") != PROBE_PROFILE
            or report.get("parent_version") != RUNTIME_VERSION
            or report.get("student_weight_sha256") != STUDENT_WEIGHT
            or report.get("preparation_manifest_sha256") != CONTEXT_MANIFEST
            or report.get("source_hashes") != probe_source_hashes()
            or report.get("eligible_for_joint_training") is not True
            or report.get("all_parent_candidates_preserved") is not True
            or report.get("test_data_read") is not False
            or report.get("development_inference_run") is not False
            or not report.get("new_coverage")):
        raise ValueError("proposal probe did not pass the fixed training gate")
    cases_path = PROBE / "cases.jsonl"
    if file_sha256(cases_path) != report.get("cases_sha256"):
        raise ValueError("proposal probe cases differ from report")
    cases = read_rows(cases_path)
    sources = joint_data["natural_train"]
    if len(cases) != PROFILE["natural_train"] or len(cases) != len(sources):
        raise ValueError("proposal probe is incomplete")
    for case, source in zip(cases, sources):
        validate_reused_case(case, source)
    return report, cases


def validate_reused_case(case, source, *, include_labels=True):
    keys = ("id", "input", "context", "expected") if include_labels else ("id", "input", "context")
    for key in keys:
        if case.get(key) != source[key]:
            raise ValueError("proposal probe source order or labels differ from joint data")
    parent = case.get("parent_candidates")
    union = case.get("union_candidates")
    if parent != source["candidates"] or not isinstance(union, list) or union[:len(parent)] != parent:
        raise ValueError("proposal probe parent candidates differ from joint data")
    if (not 1 <= len(union) <= PROFILE["candidate_limit"]
            or len(set(union)) != len(union)
            or any(not isinstance(value, str) or not value for value in union)
            or type(case.get("sentence_eligible")) is not bool):
        raise ValueError("invalid reused union candidate pool")


def score_teacher(sentence_ranker, candidates, context, sentence_eligible):
    if not sentence_eligible:
        return []
    scores = [float(value) for value in sentence_ranker.score_pairs(
        [context] * len(candidates), candidates
    )]
    if len(scores) != len(candidates) or any(not math.isfinite(value) for value in scores):
        raise ValueError("invalid parent sentence teacher scores")
    return scores


def align_row(source, parent_candidates, candidates, sentence_eligible, teacher_scores):
    row = {key: source[key] for key in SOURCE_KEYS}
    positives = [index for index, candidate in enumerate(candidates)
                 if candidate in source["expected"]]
    row.update(
        parent_candidates=parent_candidates,
        candidates=candidates,
        positive_indices=positives,
        sentence_eligible=sentence_eligible,
        training_eligible=sentence_eligible and len(candidates) > 1 and bool(positives),
        teacher_scores=teacher_scores,
    )
    validate_row(row, source)
    return row


def prepare_reused_row(source, probe_case, sentence_ranker):
    validate_reused_case(probe_case, source, include_labels=False)
    parent = list(probe_case["parent_candidates"])
    candidates = list(probe_case["union_candidates"])
    eligible = probe_case["sentence_eligible"]
    scores = score_teacher(sentence_ranker, candidates, source["context"], eligible)
    validate_reused_case(probe_case, source)
    return align_row(source, parent, candidates, eligible, scores)


def prepare_fresh_row(source, converter, student):
    proposal = propose(converter, student, source["input"], source["context"])
    parent = [path[0] for path in proposal["parent_paths"]]
    candidates = [path[0] for path in proposal["union_paths"]]
    scores = score_teacher(
        converter.sentence_ranker, candidates, source["context"],
        proposal["sentence_eligible"],
    )
    # Targets are consulted only after proposal generation and teacher scoring.
    return align_row(source, parent, candidates, proposal["sentence_eligible"], scores)


def validate_row(row, source):
    if any(row.get(key) != source[key] for key in SOURCE_KEYS):
        raise ValueError("union sentence row differs from selected source or labels")
    parent = row.get("parent_candidates")
    candidates = row.get("candidates")
    if (parent != source["candidates"]
            or not isinstance(candidates, list)
            or not 1 <= len(candidates) <= PROFILE["candidate_limit"]
            or candidates[:len(parent)] != parent
            or len(set(candidates)) != len(candidates)
            or any(not isinstance(value, str) or not value for value in candidates)):
        raise ValueError("invalid parent-preserving sentence candidate pool")
    positives = row.get("positive_indices")
    expected_positives = [index for index, value in enumerate(candidates)
                          if value in source["expected"]]
    if (not isinstance(positives, list)
            or any(type(index) is not int for index in positives)
            or positives != expected_positives):
        raise ValueError("invalid union sentence positive indices")
    eligible = row.get("sentence_eligible")
    training_eligible = row.get("training_eligible")
    scores = row.get("teacher_scores")
    expected_training_eligible = eligible and len(candidates) > 1 and bool(positives)
    if (type(eligible) is not bool or type(training_eligible) is not bool
            or training_eligible != expected_training_eligible or not isinstance(scores, list)):
        raise ValueError("invalid sentence eligibility or teacher scores")
    if eligible:
        if len(scores) != len(candidates) or any(
                type(value) not in (int, float) or not math.isfinite(value)
                for value in scores):
            raise ValueError("invalid eligible teacher score array")
    elif scores != []:
        raise ValueError("fallback rows cannot contain sentence teacher scores")


def source_digest(data):
    payload = {role: [{key: row[key] for key in SOURCE_KEYS} for row in data[role]]
               for role in ROLES}
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def validate_union_data(directory, manifest_sha256):
    manifest_path = directory / "manifest.json"
    if not isinstance(manifest_sha256, str) or file_sha256(manifest_path) != manifest_sha256:
        raise ValueError("union manifest differs from explicitly pinned hash")
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("schema_version") != 1
            or manifest.get("profile") != PROFILE
            or manifest.get("parent_runtime_version") != RUNTIME_VERSION
            or manifest.get("student_weight_sha256") != STUDENT_WEIGHT
            or manifest.get("context_manifest_sha256") != CONTEXT_MANIFEST
            or manifest.get("test_data_read") is not False
            or manifest.get("uncovered_rows_retained") is not True
            or manifest.get("fallback_rows_excluded_from_training") is not True):
        raise ValueError("invalid union sentence preparation policy")
    _, joint_data = validate_context_data(CONTEXT_DATA, CONTEXT_MANIFEST)
    probe_report, _ = load_probe(manifest.get("probe_report_sha256"), joint_data)
    if (manifest.get("probe_report") != probe_report
            or manifest.get("source_hashes") != source_hashes()
            or manifest.get("source_digest") != source_digest(joint_data)
            or set(manifest.get("outputs", {})) != set(ROLES)):
        raise ValueError("union sentence sources changed")
    result = {}
    for role in ROLES:
        path = directory / f"{role}.jsonl"
        spec = manifest["outputs"][role]
        if file_sha256(path) != spec["sha256"] or path.stat().st_size != spec["bytes"]:
            raise ValueError("union sentence output integrity mismatch")
        rows = read_rows(path)
        if len(rows) != PROFILE[role] or len(rows) != spec["rows"]:
            raise ValueError("union sentence output row count mismatch")
        for row, source in zip(rows, joint_data[role]):
            validate_row(row, source)
        result[role] = rows
    return manifest, result


def rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def prepare(directory, probe_report_sha256):
    if directory.exists():
        raise FileExistsError(directory)
    _, joint_data = validate_context_data(CONTEXT_DATA, CONTEXT_MANIFEST)
    probe_report, probe_cases = load_probe(probe_report_sha256, joint_data)
    before = source_hashes()
    converter = NeuralConverter(ranker_kind="hybrid")
    converter.load_models()
    if converter.version != RUNTIME_VERSION:
        raise ValueError("runtime differs from pinned union parent")
    if file_sha256(STUDENT / "model.safetensors") != STUDENT_WEIGHT:
        raise ValueError("preserved student differs from pinned checkpoint")
    student = SemanticRanker.load(STUDENT)
    import torch
    torch.set_num_threads(PROFILE["threads"])
    directory.mkdir(parents=True)
    started = time.monotonic()
    completed = 0
    outputs = {}
    for role in ROLES:
        counts = {"rows": len(joint_data[role]), "eligible": 0, "covered": 0,
                  "training_eligible": 0, "parent_covered": 0, "added_candidates": 0}
        path = directory / f"{role}.jsonl"
        with path.open("x") as stream:
            for index, source in enumerate(joint_data[role]):
                row = (prepare_reused_row(source, probe_cases[index], converter.sentence_ranker)
                       if role == "natural_train" else
                       prepare_fresh_row(source, converter, student))
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                counts["eligible"] += row["sentence_eligible"]
                counts["training_eligible"] += row["training_eligible"]
                counts["covered"] += bool(row["positive_indices"])
                counts["parent_covered"] += any(
                    candidate in row["expected"] for candidate in row["parent_candidates"]
                )
                counts["added_candidates"] += len(row["candidates"]) - len(row["parent_candidates"])
                completed += 1
                elapsed = time.monotonic() - started
                if rss_bytes() > PROFILE["rss_limit_bytes"] or elapsed > PROFILE["time_limit_seconds"]:
                    raise RuntimeError("union sentence preparation resource bound exceeded")
                if completed % PROFILE["progress_interval"] == 0:
                    print(json.dumps({"event": "prepare_union_sentence", "completed": completed,
                                      "elapsed_seconds": round(elapsed, 2)}), flush=True)
        outputs[role] = {**counts, "bytes": path.stat().st_size,
                         "sha256": file_sha256(path)}
    validate_context_data(CONTEXT_DATA, CONTEXT_MANIFEST)
    load_probe(probe_report_sha256, joint_data)
    if before != source_hashes():
        raise ValueError("union sentence sources changed during preparation")
    manifest = {
        "schema_version": 1,
        "profile": PROFILE,
        "parent_runtime_version": RUNTIME_VERSION,
        "student_weight_sha256": STUDENT_WEIGHT,
        "context_manifest_sha256": CONTEXT_MANIFEST,
        "probe_report_sha256": probe_report_sha256,
        "probe_report": probe_report,
        "source_hashes": before,
        "source_digest": source_digest(joint_data),
        "outputs": outputs,
        "test_data_read": False,
        "uncovered_rows_retained": True,
        "fallback_rows_excluded_from_training": True,
        "teacher_policy": "Current parent sentence logits over eligible union pools; empty for exact runtime fallback routes",
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": rss_bytes(),
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    digest = file_sha256(manifest_path)
    validate_union_data(directory, digest)
    print(json.dumps({"manifest_sha256": digest, "outputs": outputs,
                      "peak_rss_bytes": manifest["peak_rss_bytes"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-report-sha256", required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    prepare(args.output, args.probe_report_sha256)
