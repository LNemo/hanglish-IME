"""Actual provenance-boundary evaluation on a source-only natural reserve."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hanglish.core.phonology import romanize_word
from scripts import audit_final_choice_candidate_stages as audit
from scripts import evaluate_lexical_boundary_fit as legacy
from scripts.boundary_proposal_policy import boundary_proposal_allowed
from scripts.boundary_provenance_runtime import ProvenanceBoundaryRuntime
from scripts.evaluate_boundary_provenance_fit import TwoPassRecorder
from scripts.final_choice_capture import validate_response

OUTPUT = ROOT / ".omx/artifacts/boundary-reserve-runtime-v1"
SELECTION = ROOT / ".omx/artifacts/boundary-reserve-v1"
RAW = ROOT / "data/natural_context/klue-v1.1-pilot-v2/train.jsonl"
FIT_REPORT = ROOT / ".omx/artifacts/boundary-provenance-runtime-fit-v1/report.json"
FIT_INDEPENDENT = ROOT / ".omx/artifacts/boundary-provenance-runtime-fit-v1/independent-replay.json"
FIT_REPORT_SHA256 = "b75826e0d205c9294169ccbb854a7d91c672d3b19d01a8a2e3144d86361c29af"
FIT_INDEPENDENT_SHA256 = "68e94276cf798af080cc5d222cdc05ac27312595550fca486896235e9c9c77aa"
SELECTION_MANIFEST_SHA256 = "aa76dd1e2bbb210e0401ac745a58fe54dfbe1e9e3dc124ea24053b4dd8cfea61"
SELECTION_SHA256 = "3db429739cea0834d7a6bb284944bf3107060d2cf4f0a2fd252ed48c094771d0"

PINNED_INPUTS = {
    ".omx/artifacts/final-choice-partition-v1/manifest.json":
        "e997f06558bdb5633446171d35912140fb34b55066c13fa13b668939d6347846",
    ".omx/artifacts/balanced-final-choice-partition-v2/manifest.json":
        "57170914d4f14d5cd725ea0fd52a96b68b0859ba83041b93f1ecc09daa7cc8ca",
    ".omx/artifacts/finite-final-choice-partition-v3/manifest.json":
        "8f699052b3e8d6e07dcaf59dc8696cc13a8ffb15b4e1bb5df62571e906ae3e61",
    "data/natural_context/klue-v1.1-pilot-v2/train.jsonl":
        "904aab04033ac758d5ccda7d8cdd75d9bd14872ad31017a14d884633e5de4d3b",
    ".omx/artifacts/long-context-lora-data-v1-review2/fresh_selection.jsonl":
        "fb1257833c99acc406a831336f535d9ef3a786fc38bb08a84902598e5fe2287d",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hashes(hashes):
    for name, expected in hashes.items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"source pin differs: {name}")


def require_fit_gate(path=FIT_REPORT, independent_path=FIT_INDEPENDENT):
    """Require the completed actual FIT gate before reserve data is opened."""
    path = Path(path)
    independent_path = Path(independent_path)
    if (sha(path) != FIT_REPORT_SHA256
            or sha(independent_path) != FIT_INDEPENDENT_SHA256):
        raise ValueError("fixed FIT proof artifacts changed")
    report = json.loads(path.read_bytes())
    independent = json.loads(independent_path.read_bytes())
    if (report.get("schema") != "boundary-provenance-runtime-fit-v1"
            or report.get("completed") is not True
            or report.get("gate", {}).get("passed") is not True
            or not report.get("correction_ids")
            or report.get("regression_ids")
            or report.get("top5_loss_ids")
            or report.get("source_hashes_before") != report.get("source_hashes_after")):
        raise ValueError("completed regression-free provenance FIT gate required")
    if (independent.get("verified") is not True
            or independent.get("quality_gate_passed") is not True
            or independent.get("report_sha256") != FIT_REPORT_SHA256
            or independent.get("rows") != 512
            or independent.get("raw_and_parent_pool_preserved") is not True
            or independent.get("all_extras_pass_source_policy") is not True):
        raise ValueError("independent provenance FIT replay required")
    verify_hashes(report["source_hashes_after"])
    return report


def _manifest_count(manifest, *names):
    for name in names:
        if name in manifest:
            return manifest[name]
    counts = manifest.get("counts", {})
    for name in names:
        if name in counts:
            return counts[name]
    return None


def load_reserve(directory=SELECTION, raw_path=RAW, *, target_access=None):
    """Validate source-only selection, then derive inference source and target."""
    directory, raw_path = Path(directory), Path(raw_path)
    manifest_raw = (directory / "manifest.json").read_bytes()
    selection_raw = (directory / "selection.jsonl").read_bytes()
    if (hashlib.sha256(manifest_raw).hexdigest() != SELECTION_MANIFEST_SHA256
            or hashlib.sha256(selection_raw).hexdigest() != SELECTION_SHA256):
        raise ValueError("fixed boundary reserve artifacts changed")
    manifest = json.loads(manifest_raw)
    selected = [json.loads(line) for line in selection_raw.splitlines() if line]
    if manifest.get("schema") != "boundary-reserve-selection-v1":
        raise ValueError("boundary reserve manifest schema differs")
    if (len(selected) != 512 or _manifest_count(manifest, "rows", "selected_rows") != 512
            or _manifest_count(manifest, "groups", "selected_groups") != 394
            or manifest.get("selection_sha256") != hashlib.sha256(selection_raw).hexdigest()
            or manifest.get("raw_target_bytes_read") is not True
            or manifest.get("raw_json_rows_parsed") is not True
            or manifest.get("raw_target_fields_parsed") is not True
            or manifest.get("selection_uses_target_text") is not False
            or manifest.get("target_fields_used_for_selection") is not False
            or manifest.get("target_text_copied") is not False
            or manifest.get("selection_fields_used") != ["id", "group_id"]
            or manifest.get("whole_groups_preserved") is not True
            or manifest.get("model_inference_run") is not False
            or manifest.get("training_run") is not False
            or manifest.get("source_rows_modified") is not False):
        raise ValueError("boundary reserve shape or selection binding differs")
    identities, groups = set(), set()
    for item in selected:
        if (set(item) != {"id", "group_id"}
                or not isinstance(item["id"], str) or not item["id"]
                or not isinstance(item["group_id"], str) or not item["group_id"]
                or item["id"] in identities):
            raise ValueError("boundary reserve selection row is malformed")
        identities.add(item["id"])
        groups.add(item["group_id"])
    if len(groups) != 394:
        raise ValueError("boundary reserve group count differs")
    manifest_hashes = manifest.get("source_hashes")
    if not isinstance(manifest_hashes, dict):
        raise ValueError("boundary reserve source hashes missing")
    for name, expected in PINNED_INPUTS.items():
        if manifest_hashes.get(name) != expected:
            raise ValueError(f"boundary reserve source pin differs: {name}")
    verify_hashes({name: digest for name, digest in PINNED_INPUTS.items()
                   if name != str(raw_path.relative_to(ROOT))})

    if target_access is not None:
        target_access.append({
            "reason": "target text is required to derive romanized inference input",
            "path": str(raw_path.relative_to(ROOT)),
            "sha256": PINNED_INPUTS[str(raw_path.relative_to(ROOT))],
            "before_raw_read": True,
        })
    raw_bytes = raw_path.read_bytes()
    if hashlib.sha256(raw_bytes).hexdigest() != PINNED_INPUTS[str(raw_path.relative_to(ROOT))]:
        raise ValueError("raw natural reserve changed")
    raw_rows = [json.loads(line) for line in raw_bytes.splitlines() if line]
    by_id = {}
    group_members = {}
    for row in raw_rows:
        identity, group = row.get("id"), row.get("group_id")
        if not isinstance(identity, str) or identity in by_id:
            raise ValueError("raw reserve IDs are malformed or duplicated")
        by_id[identity] = row
        group_members.setdefault(group, set()).add(identity)
    if identities != set().union(*(group_members.get(group, set()) for group in groups)):
        raise ValueError("boundary reserve selection is not whole-group closed")

    sources = []
    for item in selected:
        row = by_id.get(item["id"])
        if (row is None or row.get("group_id") != item["group_id"]
                or row.get("split") != "train"):
            raise ValueError("boundary reserve raw join differs")
        text, context = row.get("text"), row.get("left_context")
        if not isinstance(text, str) or not text or not isinstance(context, str):
            raise ValueError("boundary reserve target source is malformed")
        sources.append({
            "id": item["id"], "group_id": item["group_id"],
            "input": romanize_word(text), "context": context,
        })
    lineage = {
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "selection_sha256": hashlib.sha256(selection_raw).hexdigest(),
        "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "source_hashes": manifest_hashes,
        "target_bytes_needed_for_input_derivation": True,
    }
    selected_pairs = tuple((item["id"], item["group_id"]) for item in selected)
    return sources, (lambda: _load_targets_late(
        raw_path, selected_pairs,
        PINNED_INPUTS[str(raw_path.relative_to(ROOT))]
    )), lineage


def _load_targets_late(raw_path, selected_pairs, expected_sha256):
    """Reopen target bytes only after captures close and models release."""
    raw_path = Path(raw_path)
    raw_bytes = raw_path.read_bytes()
    if hashlib.sha256(raw_bytes).hexdigest() != expected_sha256:
        raise ValueError("raw reserve changed before target join")
    wanted = dict(selected_pairs)
    targets = {}
    for row in (json.loads(line) for line in raw_bytes.splitlines() if line):
        identity = row.get("id")
        if identity not in wanted:
            continue
        text = row.get("text")
        if (row.get("group_id") != wanted[identity]
                or row.get("split") != "train"
                or not isinstance(text, str) or not text):
            raise ValueError("late reserve target join differs")
        targets[identity] = [text]
    if set(targets) != set(wanted):
        raise ValueError("late reserve target join is incomplete")
    return targets


def verify_capture(capture, source=None):
    """Pure replay of one source-bound capture and its score invariants."""
    if source is None:
        source = {key: capture.get(key) for key in
                  ("id", "group_id", "input", "context")}
    if any(capture.get(key) != source[key] for key in source):
        raise ValueError("reserve capture source identity differs")
    parent, child, trace = capture["parent"], capture["child"], capture["trace"]
    if capture["parent_raw"] != capture["child_raw"]:
        raise ValueError("reserve raw response changed")
    if (capture["parent_raw"].get("result") != source["input"]
            or capture["child_raw"].get("result") != source["input"]):
        raise ValueError("reserve raw result did not preserve input")
    if capture["parent_sentence_scored"]:
        preserved, extras = trace["preserved_pool"], trace["accepted_new_extras"]
        if preserved != capture["parent_pool"] or len(extras) > 48:
            raise ValueError("reserve parent pool or addition bound changed")
        if (capture.get("child_sentence_scored") is not True
                or capture["parent_score_inputs"] != [path["text"] for path in preserved]
                or capture["child_score_inputs"] != [path["text"] for path in preserved + extras]
                or parent["candidates"] != legacy.display(
                    preserved, capture["parent_scores"], source["input"])
                or child["candidates"] != legacy.display(
                    preserved + extras, capture["child_scores"], source["input"])):
            raise ValueError("reserve score vector or display replay differs")
        for extra in extras:
            if not boundary_proposal_allowed(
                    source["input"], parent["candidates"][0], extra,
                    trace["derived_options"]):
                raise ValueError("reserve trace contains a disallowed boundary extra")
        calls = capture.get("child_score_calls")
        if (not isinstance(calls, list) or len(calls) != 2
                or calls[0]["candidates"] != capture["parent_score_inputs"]
                or calls[0]["scores"] != capture["parent_scores"]
                or calls[1]["candidates"] != capture["child_score_inputs"]
                or calls[1]["scores"] != capture["child_scores"]):
            raise ValueError("reserve two-pass score calls differ")
    elif (child != parent or capture["child_sentence_scored"]
            or trace["accepted_new_extras"]
            or capture["child_score_inputs"] is not None
            or capture["child_scores"] is not None
            or capture.get("child_score_calls")):
        raise ValueError("unsupported reserve branch changed")
    if parent["result"] != parent["candidates"][0]["text"] or child["result"] != child["candidates"][0]["text"]:
        raise ValueError("reserve first candidate/result differs")
    return True


def summarize_closed(capture_path, sources, load_targets, *, capture_stream,
                     models_released):
    """Join targets only after the capture stream is durably closed."""
    if not capture_stream.closed:
        raise ValueError("capture stream must close before target join")
    if models_released is not True:
        raise ValueError("model references must release before target join")
    captures = [json.loads(line) for line in Path(capture_path).read_bytes().splitlines() if line]
    if len(captures) != 512 or len(sources) != 512:
        raise ValueError("complete reserve512 captures required")
    for source, capture in zip(sources, captures):
        verify_capture(capture, source)
    targets = load_targets()
    if set(targets) != {source["id"] for source in sources}:
        raise ValueError("reserve target join differs")
    counts = dict.fromkeys(("rows", "before_top1", "top1", "before_top5", "top5",
                           "before_pool", "pool"), 0)
    corrections, regressions, top5_gains, top5_losses = [], [], [], []
    for source, capture in zip(sources, captures):
        accepted = targets[source["id"]]
        before = capture["parent"]["result"] in accepted
        after = capture["child"]["result"] in accepted
        before5 = any(c["text"] in accepted for c in capture["parent"]["candidates"])
        after5 = any(c["text"] in accepted for c in capture["child"]["candidates"])
        before_pool = any(c["text"] in accepted for c in capture["parent_pool"])
        paths = (capture["trace"]["preserved_pool"]
                 + capture["trace"]["accepted_new_extras"]
                 if capture["parent_sentence_scored"] else capture["parent_pool"])
        after_pool = any(c["text"] in accepted for c in paths)
        for key, value in zip(counts, (1, before, after, before5, after5,
                                       before_pool, after_pool)):
            counts[key] += int(value)
        for condition, collection in (
                (after and not before, corrections),
                (before and not after, regressions),
                (after5 and not before5, top5_gains),
                (before5 and not after5, top5_losses)):
            if condition:
                collection.append(source["id"])
    gate = {
        "passed": bool(corrections) and not regressions and not top5_losses,
        "first_choice_gain_required": True,
        "zero_first_choice_losses": not regressions,
        "zero_top5_losses": not top5_losses,
    }
    return {"counts": counts, "correction_ids": corrections,
            "regression_ids": regressions, "top5_gain_ids": top5_gains,
            "top5_loss_ids": top5_losses, "gate": gate}


def _resources(started):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss = rss if sys.platform == "darwin" else rss * 1024
    if rss > 3 * 1024**3 or time.monotonic() - started > 1800:
        raise RuntimeError("boundary reserve resource cap exceeded")
    return rss


def capture_raw_pair(parent, child, text, context):
    """Snapshot normal trace before raw bypass resets the child trace."""
    trace = deepcopy(child.boundary_trace)
    parent_raw = parent.convert(text, context, raw=True)
    child_raw = child.convert(text, context, raw=True)
    return trace, parent_raw, child_raw


def run(output=OUTPUT, selection=SELECTION, fit_report=FIT_REPORT):
    import torch

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    fit = require_fit_gate(fit_report)
    target_access = []
    sources, load_targets, lineage = load_reserve(selection, target_access=target_access)
    source_hashes = dict(fit["source_hashes_after"])
    for path in (Path(__file__), ROOT / "scripts/boundary_provenance_runtime.py",
                 ROOT / "scripts/boundary_proposal_policy.py",
                 ROOT / ".omx/plans/boundary-reserve-v1.md",
                 Path(fit_report), FIT_INDEPENDENT):
        source_hashes[str(path.relative_to(ROOT))] = sha(path)
    source_hashes[".omx/artifacts/boundary-reserve-v1/manifest.json"] = sha(Path(selection) / "manifest.json")
    source_hashes[".omx/artifacts/boundary-reserve-v1/selection.jsonl"] = sha(Path(selection) / "selection.jsonl")
    verify_hashes(source_hashes)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {
        "schema": "boundary-reserve-runtime-v1", "completed": False,
        "scope": "new source-only natural reserve512 / 394 complete groups",
        "development_data_read": False, "final_data_read": False,
        "target_access": target_access, "lineage": lineage,
        "source_hashes_before": source_hashes, "promotion_allowed": False,
    }
    stream = None
    try:
        torch.set_num_threads(2)
        parent = audit.StageConverter(ranker_kind="hybrid", external_lexicon=True)
        parent.load_models()
        if parent.version != fit["runtime_version"] or parent.model_artifacts != fit["model_artifacts"]:
            raise ValueError("reserve parent model identity differs from passed FIT")
        recorder = TwoPassRecorder(parent.sentence_ranker)
        parent.sentence_ranker = recorder
        child = ProvenanceBoundaryRuntime.__new__(ProvenanceBoundaryRuntime)
        child.__dict__ = parent.__dict__.copy()
        child.initialize_boundary_overlay()
        (output / "sources.jsonl").write_bytes(b"".join(
            legacy.canonical(source) + b"\n" for source in sources
        ))
        with (output / "captures.jsonl").open("xb") as stream:
            for index, source in enumerate(sources):
                text, context = source["input"], source["context"]
                recorder.reset()
                before = parent.convert(text, context)
                parent_inputs, parent_scores = deepcopy(recorder.candidates), deepcopy(recorder.scores)
                if before["sentence_ranking_applied"]:
                    _, _, full, _, _ = audit.reconstruct_pool(
                        parent, parent.audit_pieces, parent.audit_static_pieces,
                        sentence_possible=True)
                    parent_pool = [legacy.path_record(path) for path in full]
                else:
                    parent_pool = [legacy.path_record(path) for path in audit.beam_paths(parent.audit_pieces)]
                recorder.reset()
                after = child.convert(text, context)
                calls = deepcopy(recorder.calls)
                child_inputs, child_scores = deepcopy(recorder.candidates), deepcopy(recorder.scores)
                trace, parent_raw, child_raw = capture_raw_pair(
                    parent, child, text, context)
                for response in (before, after, parent_raw, child_raw):
                    validate_response(response, text, context, parent.version,
                                      parent.model_artifacts, raw=response["raw"])
                capture = {**source, "parent": legacy.without_timing(before),
                    "child": legacy.without_timing(after),
                    "parent_raw": legacy.without_timing(parent_raw),
                    "child_raw": legacy.without_timing(child_raw),
                    "parent_pool": parent_pool, "trace": trace,
                    "parent_sentence_scored": before["sentence_ranking_applied"],
                    "child_sentence_scored": after["sentence_ranking_applied"],
                    "parent_score_inputs": parent_inputs, "parent_scores": parent_scores,
                    "child_score_inputs": child_inputs, "child_scores": child_scores,
                    "child_score_calls": calls, "parent_ms": before["engine_ms"],
                    "child_ms": after["engine_ms"]}
                stream.write(legacy.canonical(capture) + b"\n")
                stream.flush()
                _resources(started)
                if (index + 1) % 32 == 0:
                    print(json.dumps({"stage": "reserve_capture", "rows": index + 1,
                                      "seconds": time.monotonic() - started}), flush=True)
        runtime_version = parent.version
        model_artifacts = deepcopy(parent.model_artifacts)
        del parent, child, recorder
        gc.collect()
        models_released = True
        report.update(summarize_closed(
            output / "captures.jsonl", sources, load_targets,
            capture_stream=stream, models_released=models_released))
        _resources(started)
        verify_hashes(source_hashes)
        parent_timings, child_timings = [], []
        with (output / "captures.jsonl").open("rb") as timing_stream:
            for index, line in enumerate(timing_stream):
                if index == 0:
                    continue
                item = json.loads(line)
                parent_timings.append(item["parent_ms"])
                child_timings.append(item["child_ms"])
        parent_timings.sort()
        child_timings.sort()
        _resources(started)
        middle = len(parent_timings) // 2
        p95 = math.ceil(.95 * len(parent_timings)) - 1
        report.update({"completed": True, "source_hashes_after": source_hashes,
            "rows_captured_before_accuracy": 512,
            "warm_parent_p50_ms": parent_timings[middle],
            "warm_parent_p95_ms": parent_timings[p95],
            "warm_child_p50_ms": child_timings[middle],
            "warm_child_p95_ms": child_timings[p95],
            "latency_comparison": (
                "fixed parent-then-child order with shared model state and caches; "
                "do not infer relative speed, and benchmark absolute cold/warm "
                "engine latency separately"
            ),
            "latency_scope": "both parent-anchor and filtered-overlay calls; model load excluded",
            "captures_sha256": sha(output / "captures.jsonl"),
            "sources_sha256": sha(output / "sources.jsonl"),
            "runtime_version": runtime_version, "model_artifacts": model_artifacts,
            "models_released_before_target_join": models_released,
            "target_join_reopened_raw": True})
    except Exception as error:
        report["failure"] = f"{type(error).__name__}: {error}"
    report["elapsed_seconds"] = time.monotonic() - started
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["peak_rss_bytes"] = rss if sys.platform == "darwin" else rss * 1024
    report["created_at"] = datetime.now(timezone.utc).isoformat()
    (output / "report.json").write_bytes(legacy.canonical(report) + b"\n")
    print(json.dumps({key: report[key] for key in
        ("completed", "counts", "gate", "failure") if key in report}), flush=True)
    return report


if __name__ == "__main__":
    result = run()
    raise SystemExit(0 if result["completed"] else 1)
