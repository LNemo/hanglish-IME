"""Development-only gate for a parent-preserving union sentence ranker."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from hanglish.core.neural_converter import NeuralConverter, file_sha256
from hanglish.models.semantic_ranker import SemanticRanker
from hanglish.models.sentence_ranker import SentenceRanker
from scripts.context_proposal_union import choose, propose
from scripts.evaluate_natural_ranker import summarize
from scripts.evaluate_roberta_context_ranker import DEV_SUITES
from scripts.prepare_context_training import RUNTIME_VERSION
from scripts.prepare_union_sentence_data import source_hashes, validate_union_data
from scripts.probe_context_proposal_union import STUDENT, STUDENT_WEIGHT
from scripts.train_natural_sentence_ranker import rss_bytes
from scripts.train_union_sentence_ranker import PARENT_WEIGHT, training_hashes


DEV_COUNTS = {"natural_dev": 128, "original_dev": 140,
              "selection_dev": 60, "reasoning_dev": 40}
PROFILE = {"threads": 2, "rss_limit_bytes": 3 * 1024**3,
           "time_limit_seconds": 1800, "progress_interval": 32}


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def visible_candidates(paths, original):
    selected = [path[0] for path in paths[:5]]
    if original not in selected:
        selected = selected[:4] + [original]
    return selected


def validate_training_gate(training, checkpoint_hashes):
    parent = training.get("parent", {})
    final = training.get("final", {})
    if (training.get("checkpoint_sha256") != checkpoint_hashes.get("model.safetensors")
            or training.get("eligible_for_runtime_dev") is not True
            or training.get("regressions") != []
            or training.get("test_data_read") is not False
            or parent.get("rows") != DEV_COUNTS["natural_dev"]
            or final.get("rows") != DEV_COUNTS["natural_dev"]
            or final.get("correct", 0) <= parent.get("correct", 0)):
        raise ValueError("complete improved union sentence training gate required")


def runtime_development_passed(results):
    return (
        set(results) == set(DEV_COUNTS)
        and all(results[name]["rows"] == count for name, count in DEV_COUNTS.items())
        and all(not result["regressions"] for result in results.values())
        and results["natural_dev"]["top1"] > results["natural_dev"]["before_top1"]
        and all(result["parent_runtime_parity_rows"] == result["rows"]
                for result in results.values())
    )


def evaluate_case(converter, student, candidate_sentence_ranker, case, prepared=None):
    """Generate and rank every pool before consulting expected labels."""
    parent_ranker = converter.ranker
    parent_actual = converter.convert(case["input"], case["context"])
    proposal = propose(converter, student, case["input"], case["context"])
    if converter.ranker is not parent_ranker:
        raise ValueError("proposal evaluation did not restore the parent word ranker")
    parent_choice = choose(
        {**proposal, "union_paths": proposal["parent_paths"]},
        converter.sentence_ranker,
        case["context"],
    )
    final_choice = choose(proposal, candidate_sentence_ranker, case["context"])
    parent_visible = visible_candidates(parent_choice["paths"], case["input"])
    final_visible = visible_candidates(final_choice["paths"], case["input"])
    actual_visible = [candidate["text"] for candidate in parent_actual["candidates"]]
    if (parent_visible != actual_visible
            or parent_choice["sentence_ranking_applied"] != parent_actual["sentence_ranking_applied"]):
        raise ValueError("parent proposal replay differs from actual runtime")
    parent_pool = [path[0] for path in proposal["parent_paths"]]
    union_pool = [path[0] for path in proposal["union_paths"]]
    if prepared is not None:
        if (prepared["id"] != case["id"]
                or prepared["input"] != case["input"]
                or prepared["context"] != case["context"]
                or prepared["parent_candidates"] != parent_pool
                or prepared["candidates"] != union_pool
                or prepared["sentence_eligible"] != proposal["sentence_eligible"]):
            raise ValueError("generated natural development pool differs from prepared data")

    # Proposal generation, both rankings, and parity checks are label-independent.
    expected = case["expected"]
    sentence_eligible = proposal["sentence_eligible"]
    return {
        "id": case["id"],
        "input": case["input"],
        "context": case["context"],
        "expected": expected,
        "before_actual": parent_visible[0],
        "actual": final_visible,
        "before_top1": parent_visible[0] in expected,
        "top1": final_visible[0] in expected,
        "top5": any(candidate in expected for candidate in final_visible),
        "before_top5": any(candidate in expected for candidate in parent_visible),
        "before_sentence_pool_covered": (
            any(candidate in expected for candidate in parent_pool) if sentence_eligible else None
        ),
        "sentence_pool_covered": (
            any(candidate in expected for candidate in union_pool) if sentence_eligible else None
        ),
        "before_sentence_pool_size": len(parent_pool) if sentence_eligible else None,
        "sentence_pool_size": len(union_pool) if sentence_eligible else None,
        "sentence_ranking_applied": final_choice["sentence_ranking_applied"],
        "parent_runtime_parity": True,
    }


def evaluation_hashes():
    paths = [
        Path(__file__).resolve(),
        ROOT / "scripts/context_proposal_union.py",
        ROOT / "scripts/prepare_union_sentence_data.py",
        ROOT / "scripts/train_union_sentence_ranker.py",
        ROOT / "scripts/evaluate_roberta_context_ranker.py",
        ROOT / "scripts/evaluate_natural_ranker.py",
    ]
    return dict(sorted((str(path.relative_to(ROOT)), file_sha256(path)) for path in paths))


def capture_state(checkpoint, training_path):
    return {
        "source_hashes": source_hashes(),
        "trainer_hashes": training_hashes(),
        "evaluation_hashes": evaluation_hashes(),
        "checkpoint_hashes": dict(sorted(
            (path.name, file_sha256(path)) for path in checkpoint.iterdir() if path.is_file()
        )),
        "training_report_sha256": file_sha256(training_path),
        "student_weight_sha256": file_sha256(STUDENT / "model.safetensors"),
    }


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    checkpoint = args.checkpoint.resolve()
    ranker = SentenceRanker.load(checkpoint)
    provenance = ranker.metadata.get("union_preparation", {})
    _, data = validate_union_data(
        Path(provenance.get("path", "")), provenance.get("manifest_sha256")
    )
    training_path = checkpoint.parent / "report.json"
    before = capture_state(checkpoint, training_path)
    training = json.loads(training_path.read_text())
    validate_training_gate(training, before["checkpoint_hashes"])
    if (before["student_weight_sha256"] != STUDENT_WEIGHT
            or ranker.metadata.get("parent_runtime_version") != RUNTIME_VERSION
            or ranker.metadata.get("parent_checkpoint_sha256") != PARENT_WEIGHT
            or ranker.metadata.get("frozen_source_hashes") != before["trainer_hashes"]):
        raise ValueError("candidate checkpoint provenance differs from current training sources")

    torch.set_num_threads(PROFILE["threads"])
    converter = NeuralConverter(ranker_kind="hybrid")
    converter.load_models()
    if converter.version != RUNTIME_VERSION:
        raise ValueError("runtime differs from union training parent")
    student = SemanticRanker.load(STUDENT)
    parent_ranker = converter.ranker
    started = time.monotonic()
    results = {}
    outcomes = []
    sources = {}

    suites = [("natural_dev", data["natural_dev"], data["natural_dev"], None)]
    for name, relative, identity_relative in DEV_SUITES:
        path = ROOT / relative
        identity_path = ROOT / ".omx/artifacts" / identity_relative
        identity = json.loads(identity_path.read_text())
        expected_hash = identity.get("sha256") or identity["corpus_identity"]["sha256"]
        if file_sha256(path) != expected_hash:
            raise ValueError("development suite differs from frozen identity")
        suites.append((name, read_rows(path), None, (path, identity_path, expected_hash)))

    completed = 0
    for name, cases, prepared_rows, identity_spec in suites:
        current = []
        for index, case in enumerate(cases):
            converter.ranker = parent_ranker
            current.append(evaluate_case(
                converter, student, ranker, case,
                None if prepared_rows is None else prepared_rows[index],
            ))
            completed += 1
            elapsed = time.monotonic() - started
            if rss_bytes() > PROFILE["rss_limit_bytes"] or elapsed > PROFILE["time_limit_seconds"]:
                raise RuntimeError("union sentence development resource bound exceeded")
            if completed % PROFILE["progress_interval"] == 0:
                print(json.dumps({"suite": name, "completed": completed,
                                  "elapsed_seconds": round(elapsed, 2)}), flush=True)
        summary = summarize(current)
        summary["before_top5"] = sum(row["before_top5"] for row in current)
        summary["parent_runtime_parity_rows"] = sum(row["parent_runtime_parity"] for row in current)
        summary["before_sentence_pool_covered"] = sum(
            row["before_sentence_pool_covered"] is True for row in current
        )
        summary["sentence_pool_covered"] = sum(
            row["sentence_pool_covered"] is True for row in current
        )
        summary["before_sentence_scored_rows"] = sum(
            row["before_sentence_pool_covered"] is not None for row in current
        )
        summary["sentence_scored_rows"] = sum(
            row["sentence_pool_covered"] is not None for row in current
        )
        summary["sentence_pool_fallbacks"] = sum(
            row["sentence_pool_covered"] is None for row in current
        )
        results[name] = summary
        outcomes.extend({"suite": name, **row} for row in current)
        if identity_spec is None:
            sources[name] = {"union_manifest_sha256": provenance["manifest_sha256"]}
        else:
            path, identity_path, expected_hash = identity_spec
            if file_sha256(path) != expected_hash:
                raise ValueError("development suite changed during evaluation")
            sources[name] = {"sha256": expected_hash,
                             "identity_report_sha256": file_sha256(identity_path)}
        print(json.dumps({"suite": name, **summary}, ensure_ascii=False), flush=True)

    validate_union_data(Path(provenance["path"]), provenance["manifest_sha256"])
    after = capture_state(checkpoint, training_path)
    if before != after or converter.ranker is not parent_ranker:
        raise ValueError("evaluation inputs or parent ranker changed")
    passed = runtime_development_passed(results)
    report = {
        "eligible_for_runtime_dev": bool(passed),
        "passed": bool(passed),
        "scope": "actual union sentence development",
        "parent_version": RUNTIME_VERSION,
        "experimental_version": RUNTIME_VERSION + "-union-sentence-"
                                + before["checkpoint_hashes"]["model.safetensors"][:12],
        "checkpoint": str(checkpoint),
        "checkpoint_hashes": before["checkpoint_hashes"],
        "training_report_sha256": before["training_report_sha256"],
        "union_manifest_sha256": provenance["manifest_sha256"],
        "source_hashes": before["source_hashes"],
        "trainer_hashes": before["trainer_hashes"],
        "evaluation_code_hashes": before["evaluation_hashes"],
        "source_state": before,
        "sources": sources,
        "suites": results,
        "outcomes": outcomes,
        "profile": PROFILE,
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": rss_bytes(),
        "test_data_read": False,
        "deployed": False,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"eligible_for_runtime_dev": bool(passed), "suites": results},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    evaluate(parser.parse_args())
