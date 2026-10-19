"""Audit sentence-v3 dev failures across the bounded conversion stages.

The converter sees only a Roman sentence and an empty context.  Expected
sentences are joined only after every source-only trace has been captured.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.trace_conversion_candidates import TraceConverter, trace_conversion


DATA = ROOT / "data/challenge/sentence-v3/dev.jsonl"
MANIFEST = ROOT / "data/challenge/sentence-v3/manifest.json"
BASELINE = ROOT / ".omx/artifacts/sentence-only-dev-baseline-v1"
OUTPUT = ROOT / ".omx/artifacts/sentence-only-stage-audit-v1"
PROFILE = {
    "rows": 316,
    "threads": 2,
    "rss_limit_bytes": 3 * 1024**3,
    "time_limit_seconds": 1800,
}
CATEGORY_NAMES = (
    "correct", "final-selection", "raw-slot-displacement",
    "option_cap", "beam_pruning", "missing_piece",
)
STAGE_NAMES = (
    "unlimited_piece_reachable", "top8_piece_reachable",
    "final_pool_reachable", "displayed_reachable", "top1_correct",
    "word_ranker_called", "sentence_ranker_called", "length_bypass",
)
HANGUL = re.compile(r"[\u3130-\u318f\uac00-\ud7a3]")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def require_resources(started, stage):
    elapsed = time.monotonic() - started
    peak = rss_bytes()
    if elapsed > PROFILE["time_limit_seconds"] or peak > PROFILE["rss_limit_bytes"]:
        raise RuntimeError(
            f"sentence stage audit resource bound exceeded during {stage}: "
            f"elapsed={elapsed:.3f}s, rss={peak}"
        )


def source_inputs(path, manifest_path):
    """Validate dev identity and return only fields allowed across inference."""
    identity = json.loads(Path(manifest_path).read_text())
    source_digest = sha256(path)
    if (identity.get("input_contract") != "roman_sentence_only"
            or identity.get("model_input_fields", ["input"]) != ["input"]
            or identity.get("sha256", {}).get(Path(path).name) != source_digest):
        raise ValueError("sentence-v3 manifest or dev hash is invalid")
    values = []
    seen = set()
    for row in read_jsonl(path):
        source = row.get("input")
        if (row.get("split") != "dev" or not isinstance(row.get("id"), str)
                or row["id"] in seen or not isinstance(source, str)
                or len(source.split()) < 2 or HANGUL.search(source)
                or row.get("context", "") != "" or row.get("left_context")
                or row.get("context_role", "none") != "none"):
            raise ValueError("dev row violates the Roman-sentence-only contract")
        seen.add(row["id"])
        values.append({"id": row["id"], "input": source})
    if len(values) != identity.get("counts", {}).get("dev"):
        raise ValueError("sentence-v3 dev count differs from manifest")
    return values, {
        "source_sha256": source_digest,
        "manifest_sha256": sha256(manifest_path),
        "corpus_version": identity.get("corpus_version"),
    }


def baseline_evidence(directory, inputs, identity):
    """Validate the frozen runtime baseline without consulting its outcomes."""
    report = json.loads((Path(directory) / "report.json").read_text())
    capture_path = Path(directory) / "capture.jsonl"
    if (report.get("source_sha256") != identity["source_sha256"]
            or report.get("manifest_sha256") != identity["manifest_sha256"]
            or report.get("corpus_version") != identity["corpus_version"]
            or report.get("capture_sha256") != sha256(capture_path)
            or report.get("counts", {}).get("rows") != len(inputs)):
        raise ValueError("baseline data or capture hashes do not match sentence-v3 dev")
    captures = read_jsonl(capture_path)
    if len(captures) != len(inputs):
        raise ValueError("baseline capture count differs from dev inputs")
    indexed = {}
    for expected_input, capture in zip(inputs, captures):
        request = capture.get("request", {})
        if (capture.get("id") != expected_input["id"]
                or request != {"input": expected_input["input"], "context": "",
                               "engine": "neural", "mode": "batch", "raw": False}
                or "response" not in capture or capture.get("error") is not None):
            raise ValueError("baseline capture violates the source-only request contract")
        indexed[capture["id"]] = capture["response"]
    return indexed, report.get("model_artifacts", {})


def can_compose(target, pieces, option_limit=None):
    """Return exact finite-lattice reachability without target-led generation."""
    positions = {0}
    for piece in pieces:
        options = piece["options"]
        if option_limit is not None:
            options = options[:option_limit]
        positions = {
            position + len(option["text"])
            for position in positions
            for option in options
            if target.startswith(option["text"], position)
        }
        if not positions:
            return False
    return len(target) in positions


def compact_trace(case_id, trace):
    if (trace.get("source_only") is not True
            or trace.get("target_fields_read") is not False
            or trace.get("context") != ""):
        raise ValueError("trace is not source-only with empty context")
    runtime = trace.get("runtime", {})
    return {
        "id": case_id,
        "source": trace["source"],
        "source_only": True,
        "target_fields_read": False,
        "pieces_before": trace["pieces_before_word_ranking"],
        "pieces_after": trace["pieces_after_word_ranking"],
        "final_pre_display_pool": [
            {"text": path["text"], "score": path["score"]}
            for path in trace["final_pre_display_paths"]
        ],
        "ranker_called": {
            "word": bool(trace["word_ranker_score_pairs"]),
            "sentence": bool(trace["sentence_score_pairs"]),
        },
        "displayed_candidates": trace["displayed_candidates"],
        "result": trace["result"],
        "version": runtime.get("version"),
        "runtime_flags": {
            "ranking_applied": runtime.get("ranking_applied"),
            "sentence_ranking_applied": runtime.get("sentence_ranking_applied"),
            "external_context_used": runtime.get("external_context_used"),
            "warnings": runtime.get("warnings", []),
        },
    }


def parity_check(capture, baseline_response):
    if capture["version"] != baseline_response.get("version"):
        raise ValueError("runtime version differs from frozen sentence-only baseline")
    if capture["displayed_candidates"] != baseline_response.get("candidates"):
        raise ValueError("displayed candidates differ from frozen sentence-only baseline")
    if capture["result"] != baseline_response.get("result"):
        raise ValueError("result differs from frozen sentence-only baseline")
    if capture["runtime_flags"]["external_context_used"] is not False:
        raise ValueError("converter used external context in sentence-only audit")


def _texts(records):
    return [record["text"] for record in records]


def classify(expected, capture):
    """Classify the earliest bounded stage that excludes every valid answer."""
    result = capture["result"]
    if result in expected:
        return "correct"
    pool = _texts(capture["final_pre_display_pool"])
    displayed = _texts(capture["displayed_candidates"])
    source = capture["source"]
    displaced = (len(pool) >= 5 and source not in pool[:5]
                 and pool[4] in expected
                 and not any(answer in displayed for answer in expected))
    if displaced:
        return "raw-slot-displacement"
    if any(answer in expected for answer in pool):
        return "final-selection"
    before = capture["pieces_before"]
    after = capture["pieces_after"]
    unlimited = any(can_compose(answer, before) or can_compose(answer, after)
                    for answer in expected)
    if not unlimited:
        return "missing_piece"
    top8 = any(can_compose(answer, before, 8) or can_compose(answer, after, 8)
               for answer in expected)
    return "beam_pruning" if top8 else "option_cap"


def length_bypass(capture):
    warnings = capture["runtime_flags"].get("warnings", [])
    return (not capture["ranker_called"]["sentence"]
            and any("길이 범위" in warning for warning in warnings))


def stage_flags(expected, capture):
    before, after = capture["pieces_before"], capture["pieces_after"]
    pool = _texts(capture["final_pre_display_pool"])
    displayed = _texts(capture["displayed_candidates"])
    return {
        "unlimited_piece_reachable": any(
            can_compose(answer, before) or can_compose(answer, after)
            for answer in expected
        ),
        "top8_piece_reachable": any(
            can_compose(answer, before, 8) or can_compose(answer, after, 8)
            for answer in expected
        ),
        "final_pool_reachable": any(answer in pool for answer in expected),
        "displayed_reachable": any(answer in displayed for answer in expected),
        "top1_correct": capture["result"] in expected,
        "word_ranker_called": capture["ranker_called"]["word"],
        "sentence_ranker_called": capture["ranker_called"]["sentence"],
        "length_bypass": length_bypass(capture),
    }


def summarize(captures, labels):
    if [capture["id"] for capture in captures] != [row["id"] for row in labels]:
        raise ValueError("capture and label identities differ")
    categories = Counter()
    reachability = Counter()
    outcomes = []
    for capture, row in zip(captures, labels):
        expected = row.get("expected")
        if not isinstance(expected, list) or not expected or any(
                not isinstance(value, str) or not value for value in expected):
            raise ValueError("whole-sentence expected labels are required")
        category = classify(expected, capture)
        flags = stage_flags(expected, capture)
        categories[category] += 1
        reachability.update(key for key, value in flags.items() if value)
        outcomes.append({
            "id": capture["id"], "input": capture["source"],
            "expected": expected, "category": category, **flags,
        })
    rows = len(captures)
    return {
        "counts": {"rows": rows, "denominator": rows},
        "categories": {key: categories[key] for key in CATEGORY_NAMES},
        "stage_reachability": {
            key: {"rows": reachability[key], "denominator": rows}
            for key in STAGE_NAMES
        },
        "outcomes": outcomes,
    }


def run(data=DATA, manifest=MANIFEST, baseline=BASELINE, output=OUTPUT,
        converter=None, trace_fn=trace_conversion):
    started = time.monotonic()
    inputs, identity = source_inputs(data, manifest)
    if len(inputs) != PROFILE["rows"] and Path(data).resolve() == DATA.resolve():
        raise ValueError("sentence-v3 dev must contain exactly 316 rows")
    baselines, pinned_models = baseline_evidence(baseline, inputs, identity)
    if output.exists():
        raise FileExistsError(output)
    if converter is None:
        import torch
        torch.set_num_threads(PROFILE["threads"])
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        converter = TraceConverter(ranker_kind="hybrid")
    converter.load_models()
    if getattr(converter, "model_artifacts", {}) != pinned_models:
        raise ValueError("loaded model artifact hashes differ from frozen baseline")
    output.mkdir(parents=True)
    captures = []
    with (output / "capture.jsonl").open("x") as stream:
        for item in inputs:
            trace = trace_fn(converter, item["input"], "", engine="neural", raw=False)
            capture = compact_trace(item["id"], trace)
            parity_check(capture, baselines[item["id"]])
            captures.append(capture)
            stream.write(json.dumps(capture, ensure_ascii=False) + "\n")
            stream.flush()
            require_resources(started, f"capture {item['id']}")
            if len(captures) % 25 == 0:
                print(json.dumps({"captured": len(captures), "total": len(inputs)}),
                      flush=True)
    # No expected label is joined until every trace is durably captured above.
    labels = read_jsonl(data)
    metrics = summarize(captures, labels)
    if (sha256(data) != identity["source_sha256"]
            or sha256(manifest) != identity["manifest_sha256"]):
        raise ValueError("sentence-v3 data changed during audit")
    for relative, expected in pinned_models.items():
        path = Path(relative)
        path = path if path.is_absolute() else ROOT / path
        if sha256(path) != expected:
            raise ValueError("model artifact changed during audit")
    require_resources(started, "report")
    report = {
        "completed": True,
        "schema": "sentence-only-stage-audit-v1",
        "input_contract": "roman_sentence_only",
        "external_context_allowed": False,
        "labels_joined_after_all_captures": True,
        "source": str(Path(data)), "manifest": str(Path(manifest)), **identity,
        "baseline_capture_sha256": sha256(Path(baseline) / "capture.jsonl"),
        "capture_sha256": sha256(output / "capture.jsonl"),
        "model_artifacts": pinned_models,
        "profile": PROFILE,
        **metrics,
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": rss_bytes(),
        "test_data_read": False,
        "production_changed": False,
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    report = run(args.data, args.manifest, args.baseline, args.output)
    print(json.dumps({"completed": report["completed"],
                      "counts": report["counts"],
                      "categories": report["categories"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
