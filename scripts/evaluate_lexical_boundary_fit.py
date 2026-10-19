"""Actual word/sentence ranking of boundary proposals on frozen natural FIT512.

All predictions are persisted before accuracy is calculated. This is an
isolated candidate-generation experiment, not a production installation or
evidence of new-document generalization.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_numeric_composition_fit import canonical, without_timing

OUTPUT = ROOT / ".omx/artifacts/lexical-boundary-runtime-fit-v1"
AUDIT = ROOT / ".omx/artifacts/final-choice-candidate-stages-v1"
AUDIT_REPORT_SHA = "528f4fbc1ebb2bbc9e22e18b85b4af6e0a205a9a5fe1a64bbe266b52ba46b615"
AUDIT_CASES_SHA = "860da6746310eee896008280a8ad6eac72736b35db16a289aa02a9a5f45264c6"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_record(path):
    return {"text": path[0], "score": path[1], "segments": deepcopy(path[2])}


def display(pool, scores, source):
    if (not isinstance(scores, list) or len(scores) != len(pool)
            or not pool or len({p["text"] for p in pool}) != len(pool)
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in scores)):
        raise ValueError("actual sentence score vector differs from complete pool")
    ranked = sorted(zip(pool, scores), key=lambda pair: (-pair[1], pair[0]["text"]))
    result = [{"text": p["text"], "score": round(float(score), 4),
               "segments": deepcopy(p["segments"])} for p, score in ranked[:5]]
    if not any(c["text"] == source for c in result):
        result = result[:4] + [{"text": source, "score": None, "segments": [{
            "input_start": 0, "input_end": len(source), "source": source,
            "text": source, "kind": "raw"}]}]
    return result


def summarize(rows, captures):
    if len(rows) != 512 or len(captures) != 512:
        raise ValueError("complete fixed FIT512 captures required")
    seen = set()
    # Replay all non-label invariants first, including the actual score vectors.
    for row, capture in zip(rows, captures):
        if (capture.get("id") != row["id"] or row["id"] in seen
                or any(capture.get(k) != row[k] for k in ("input", "context"))):
            raise ValueError("FIT source identity or ordering differs")
        seen.add(row["id"])
        parent, child, trace = capture["parent"], capture["child"], capture["trace"]
        if parent["result"] != row["result"] or parent["candidates"] != row["candidates"]:
            raise ValueError("actual parent differs from frozen FIT response")
        if capture["parent_raw"] != capture["child_raw"]:
            raise ValueError("raw bypass changed")
        if capture["parent_sentence_scored"]:
            if not capture["child_sentence_scored"]:
                raise ValueError("parent sentence scoring was disabled")
            preserved = trace["preserved_pool"]
            extras = trace["accepted_new_extras"]
            if preserved != capture["parent_pool"] or len(extras) > 48:
                raise ValueError("preserved parent pool objects or addition limit changed")
            complete = preserved + extras
            if (capture["child_score_inputs"] != [p["text"] for p in complete]
                    or capture["parent_score_inputs"] != [p["text"] for p in preserved]):
                raise ValueError("trace differs from actual sentence scorer input")
            if parent["candidates"] != display(preserved, capture["parent_scores"], row["input"]):
                raise ValueError("parent output does not replay from actual scores")
            if child["candidates"] != display(complete, capture["child_scores"], row["input"]):
                raise ValueError("child output does not replay from actual scores")
        else:
            if (child != parent or capture["child_sentence_scored"]
                    or capture["child_score_inputs"] is not None
                    or capture["child_scores"] is not None
                    or trace["accepted_new_extras"]):
                raise ValueError("unsupported parent branch changed")
        for response in (parent, child):
            if response["result"] != response["candidates"][0]["text"]:
                raise ValueError("first result differs from displayed candidate")
    names = ("rows", "before_top1", "top1", "before_top5", "top5",
             "before_pool", "pool", "rows_with_extras", "new_paths")
    counts = dict.fromkeys(names, 0)
    corrections, regressions, top5_gains, top5_losses = [], [], [], []
    for row, capture in zip(rows, captures):
        expected = row["expected"]
        if (not isinstance(expected, list) or not expected
                or any(not isinstance(v, str) or not v for v in expected)):
            raise ValueError("accepted FIT answers missing")
        parent, child, trace = capture["parent"], capture["child"], capture["trace"]
        before, after = parent["result"] in expected, child["result"] in expected
        before5 = any(c["text"] in expected for c in parent["candidates"])
        after5 = any(c["text"] in expected for c in child["candidates"])
        pool = (trace["preserved_pool"] + trace["accepted_new_extras"]
                if capture["child_sentence_scored"] else child["candidates"])
        values = (1, before, after, before5, after5,
                  any(p["text"] in expected for p in capture["parent_pool"]),
                  any(p["text"] in expected for p in pool),
                  bool(trace["accepted_new_extras"]), len(trace["accepted_new_extras"]))
        for name, value in zip(names, values):
            counts[name] += int(value)
        for condition, collection in ((after and not before, corrections),
                (before and not after, regressions), (after5 and not before5, top5_gains),
                (before5 and not after5, top5_losses)):
            if condition:
                collection.append(row["id"])
    return {"counts": counts, "correction_ids": corrections, "regression_ids": regressions,
            "top5_gain_ids": top5_gains, "top5_loss_ids": top5_losses,
            "gate": {"passed": bool(corrections) and not regressions and not top5_losses,
                     "first_choice_gain_required": True, "zero_first_choice_losses": not regressions,
                     "zero_top5_losses": not top5_losses, "parent_pool_preserved": True,
                     "raw_preserved": True}, "promotion_allowed": False}


def run(output=OUTPUT):
    import torch
    from scripts import audit_final_choice_candidate_stages as prior
    from scripts.final_choice_capture import _response_flags, validate_response
    from scripts.lexical_boundary_runtime import BoundaryCompositionRuntime
    from scripts.train_final_choice_selector import load_fit_role

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if sha(AUDIT / "report.json") != AUDIT_REPORT_SHA or sha(AUDIT / "cases.jsonl") != AUDIT_CASES_SHA:
        raise ValueError("fixed FIT source audit changed")
    loaded = load_fit_role(prior.FIT, prior.FIT_MANIFEST_SHA256, "natural_fit")
    rows = loaded["rows"][:512]
    audit_ids = [json.loads(line)["id"] for line in (AUDIT / "cases.jsonl").read_bytes().splitlines()]
    if len(rows) != 512 or [r["id"] for r in rows] != audit_ids:
        raise ValueError("complete fixed FIT cohort differs")
    hashes = dict(json.loads((AUDIT / "report.json").read_bytes())["source_hashes"])
    for name in ("scripts/lexical_boundary_composition.py", "scripts/lexical_boundary_runtime.py",
                 "scripts/evaluate_lexical_boundary_fit.py", "scripts/evaluate_numeric_composition_fit.py"):
        hashes[name] = sha(ROOT / name)
    prior.verify_hashes(hashes)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"schema": "lexical-boundary-runtime-fit-v1", "completed": False,
              "scope": "fixed natural FIT512 only", "deployed": False,
              "development_data_read": False, "fresh_data_read": False, "final_data_read": False,
              "source_hashes_before": hashes, "promotion_allowed": False}
    try:
        torch.set_num_threads(2)
        parent = prior.StageConverter(ranker_kind="hybrid", external_lexicon=True)
        parent.load_models()
        report["model_load_seconds"] = time.monotonic() - started
        if parent.version != prior.RUNTIME_VERSION or parent.model_artifacts != loaded["manifest"]["model_artifact_hashes"]:
            raise ValueError("parent model identity differs")
        recorder = prior.SentenceRecorder(parent.sentence_ranker)
        parent.sentence_ranker = recorder
        child = BoundaryCompositionRuntime.__new__(BoundaryCompositionRuntime)
        child.__dict__ = parent.__dict__.copy()
        child.initialize_boundary_overlay()
        sources = [{k: row[k] for k in ("id", "input", "context")} for row in rows]
        (output / "sources.jsonl").write_bytes(b"".join(canonical(row) + b"\n" for row in sources))
        with (output / "captures.jsonl").open("xb") as stream:
            for index, source in enumerate(sources):
                text, context = source["input"], source["context"]
                recorder.reset()
                before = parent.convert(text, context)
                parent_inputs, parent_scores = deepcopy(recorder.candidates), deepcopy(recorder.scores)
                if before["sentence_ranking_applied"]:
                    _, _, full, _, _ = prior.reconstruct_pool(parent, parent.audit_pieces,
                        parent.audit_static_pieces, sentence_possible=True)
                    parent_pool = [path_record(path) for path in full]
                    if parent_inputs != [p["text"] for p in parent_pool]:
                        raise ValueError("actual parent scorer input differs from reconstructed pool")
                else:
                    parent_pool = [path_record(p) for p in prior.beam_paths(parent.audit_pieces)]
                recorder.reset()
                after = child.convert(text, context)
                trace = deepcopy(child.boundary_trace)
                child_inputs, child_scores = deepcopy(recorder.candidates), deepcopy(recorder.scores)
                parent_raw = parent.convert(text, context, raw=True)
                child_raw = child.convert(text, context, raw=True)
                for response in (before, after, parent_raw, child_raw):
                    validate_response(response, text, context, prior.RUNTIME_VERSION,
                                      parent.model_artifacts, raw=response["raw"])
                capture = {**source, "parent": without_timing(before), "child": without_timing(after),
                    "parent_raw": without_timing(parent_raw), "child_raw": without_timing(child_raw),
                    "parent_pool": parent_pool, "trace": trace,
                    "parent_sentence_scored": before["sentence_ranking_applied"],
                    "child_sentence_scored": after["sentence_ranking_applied"],
                    "parent_score_inputs": parent_inputs, "parent_scores": parent_scores,
                    "child_score_inputs": child_inputs, "child_scores": child_scores,
                    "parent_ms": before["engine_ms"], "child_ms": after["engine_ms"]}
                stream.write(canonical(capture) + b"\n")
                stream.flush()
                prior.require_resources(started)
                if (index + 1) % 32 == 0:
                    print(json.dumps({"stage": "source_capture", "rows": index + 1,
                                      "seconds": time.monotonic() - started}), flush=True)
        # Reopen the closed prediction stream; only now calculate label metrics.
        captures = [json.loads(line) for line in (output / "captures.jsonl").read_bytes().splitlines()]
        for row, capture in zip(rows, captures):
            if _response_flags(capture["parent"]) != row["response_flags"]:
                raise ValueError("parent response flags differ from frozen FIT")
        report.update(summarize(rows, captures))
        prior.verify_hashes(hashes)
        prior.require_resources(started)
        timings = sorted(c["child_ms"] for c in captures[1:])
        report.update({"completed": True, "source_hashes_after": hashes,
            "rows_captured_before_accuracy": len(captures), "runtime_version": parent.version,
            "model_artifacts": parent.model_artifacts,
            "warm_child_p95_ms": timings[math.ceil(.95 * len(timings)) - 1],
            "first_child_ms": captures[0]["child_ms"],
            "latency_scope": "actual engine after model load; not browser or cold model loading",
            "captures_sha256": sha(output / "captures.jsonl"),
            "sources_sha256": sha(output / "sources.jsonl")})
    except Exception as error:
        report["failure"] = f"{type(error).__name__}: {error}"
    report["elapsed_seconds"] = time.monotonic() - started
    report["peak_rss_bytes"] = prior.rss_bytes()
    report["created_at"] = datetime.now(timezone.utc).isoformat()
    (output / "report.json").write_bytes(canonical(report) + b"\n")
    print(json.dumps({k: report[k] for k in ("completed", "counts", "gate", "failure", "elapsed_seconds")
                      if k in report}), flush=True)
    return report


if __name__ == "__main__":
    result = run()
    raise SystemExit(0 if result["completed"] else 1)
