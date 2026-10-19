#!/usr/bin/env python3
"""Capture source-only DEV runtime recovery for parent-unranked sentences.

The capture never opens DEV labels.  It replays the closed production response
for already-ranked rows (with model access blocked), and runs fresh parent,
phonetic and sentence inference only for the four source-selected rows that the
production sentence ranker skipped.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SOURCE = ROOT / ".omx/artifacts/phonetic-choice-runtime-dev-v1"
COMPARATOR = ROOT / ".omx/artifacts/score-scale-runtime-dev-v1"
INVENTORY = ROOT / ".omx/artifacts/ungated-phonetic-fit-v1"
MODEL = ROOT / ".omx/artifacts/observed-phonetic-augmented-training-v1/phonetic_transducer.pt"
PLAN = ROOT / ".omx/plans/ungated-joint-runtime-v1.md"
ROWS = 316
SELECTED_ROWS = 4
PARENT_VERSION = "experimental-score-scale-1-f6976e1b269d"
LIMITS = {"whole_seconds": 900.0, "row_seconds": 120.0,
          "rss_bytes": 3 * 1024**3, "swap_growth_bytes": 512 * 1024**2,
          "output_bytes": 128 * 1024**2}


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2,
                  allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def verify_pins(pins: dict[str, str]) -> None:
    for name, expected in pins.items():
        path = Path(name)
        need(path.is_file() and sha(path) == expected, "frozen input differs: " + name)


def authenticate_sources() -> tuple[dict[str, str], list[str]]:
    source_report = read(SOURCE / "report.json")
    source_proof = read(SOURCE / "independent-verification.json")
    comparator_report = read(COMPARATOR / "report.json")
    comparator_proof = read(COMPARATOR / "independent-verification.json")
    inventory_report = read(INVENTORY / "report.json")
    need(source_report.get("completed") is True and source_proof.get("verified") is True,
         "DEV source is not closed and verified")
    need(comparator_report.get("completed") is True
         and comparator_report.get("rows") == ROWS
         and comparator_report.get("labels_opened") is False
         and comparator_report.get("version") == PARENT_VERSION
         and comparator_proof.get("verified") is True,
         "production comparator is not closed and verified")
    inventory = inventory_report.get("known_english")
    need(isinstance(inventory, list) and inventory == sorted(set(inventory)),
         "known-English inventory differs")
    pins = {}
    for report in (source_report, comparator_report, inventory_report):
        need(isinstance(report.get("pins"), dict), "source pins missing")
        pins.update(report["pins"])
    for directory in (SOURCE, COMPARATOR, INVENTORY):
        for name in ("report.json", "capture.jsonl"):
            path = directory / name
            if path.is_file():
                pins[str(path.resolve())] = sha(path)
    for path in (SOURCE / "independent-verification.json",
                 COMPARATOR / "independent-verification.json", PLAN,
                 Path(__file__).resolve()):
        pins[str(path.resolve())] = sha(path)
    verify_pins(pins)
    return pins, inventory


class _Blocked:
    def __getattr__(self, name):
        raise AssertionError("passthrough attempted model access: " + name)


class CachedParent:
    """Return one captured response and make every model access fail."""
    def __init__(self, response):
        self.response = deepcopy(response)
        self.lock = threading.RLock()
        self.sentence_ranker = _Blocked()

    def convert(self, text, context="", engine="neural", mode="batch", raw=False):
        need(text == self.response["input"] and context == self.response.get("context", ""),
             "cached request differs")
        return deepcopy(self.response)

    def load_phonetic_model(self):
        raise AssertionError("passthrough attempted phonetic inference")


class FreshBaselineParent:
    """Supply one fresh baseline while delegating only recovery model access."""
    def __init__(self, engine, baseline):
        self.engine = engine
        self.baseline = deepcopy(baseline)
        self.lock = engine.lock
        self.sentence_ranker = engine.sentence_ranker
        self.convert_calls = 0

    def convert(self, text, context="", engine="neural", mode="batch", raw=False):
        self.convert_calls += 1
        need(self.convert_calls == 1 and text == self.baseline["input"],
             "wrapper parent dispatch differs")
        return deepcopy(self.baseline)

    def load_phonetic_model(self):
        return self.engine.load_phonetic_model()


def selected_by_source(source_row: dict) -> bool:
    proposed = source_row.get("proposed")
    value = proposed.get("sentence_ranking_applied") if isinstance(proposed, dict) else None
    need(type(value) is bool, "source ranking flag differs")
    return not value


def capture_row(source_row, comparator_row, *, engine=None, inventory=()):
    """Capture one row.  Injectable engine keeps unit tests model-free."""
    from hanglish.core.ungated_joint_converter import UngatedJointConverter

    need(source_row.get("id") == comparator_row.get("id")
         and source_row.get("request") == comparator_row.get("request"),
         "DEV source/comparator identity differs")
    request = source_row["request"]
    need(request == {"input": request.get("input"), "context": ""},
         "DEV request contract differs")
    cached = comparator_row.get("response")
    need(isinstance(cached, dict) and cached.get("version") == PARENT_VERSION,
         "captured production response differs")
    selected = selected_by_source(source_row)
    started = time.monotonic()
    if not selected:
        wrapper = UngatedJointConverter(CachedParent(cached), known_english=inventory)
        response = wrapper.convert(request["input"])
        need(response == cached and wrapper.last_trace == {
            "applied": False, "reason": "parent_passthrough"},
            "already-ranked response did not pass through exactly")
        return {"id": source_row["id"], "request": deepcopy(request),
                "selected_unranked": False, "passthrough_exact": True,
                "cached_response_sha256": hashlib.sha256(json.dumps(
                    cached, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
                "elapsed_seconds": time.monotonic() - started}

    need(engine is not None, "fresh engine required for selected row")
    baseline = engine.convert(request["input"])
    need(baseline.get("version") == PARENT_VERSION
         and baseline.get("sentence_ranking_applied") is False,
         "fresh selected parent response differs")
    proxy = FreshBaselineParent(engine, baseline)
    wrapper = UngatedJointConverter(proxy, known_english=inventory)
    response = wrapper.convert(request["input"])
    trace = wrapper.last_trace
    need(proxy.convert_calls == 1 and trace.get("baseline") == baseline,
         "fresh baseline was not preserved")
    need(isinstance(trace.get("pool"), list), "recovery pool missing")
    for row in trace["pool"]:
        need(type(row.get("supported")) is bool
             and ((row["supported"] and isinstance(row.get("score"), float))
                  or (not row["supported"] and row.get("score") is None)),
             "support/score contract differs")
    return {"id": source_row["id"], "request": deepcopy(request),
            "selected_unranked": True, "cached_production": deepcopy(cached),
            "fresh_baseline": baseline, "response": response, "trace": trace,
            "controls": {"parent_version": baseline["version"],
                         "parent_ranked": False,
                         "parent_candidates_exact_cached":
                             baseline.get("candidates") == cached.get("candidates"),
                         "wrapper_applied": bool(trace.get("applied")),
                         "pool_size": len(trace["pool"]),
                         "supported": sum(r["supported"] for r in trace["pool"]),
                         "unsupported": sum(not r["supported"] for r in trace["pool"])},
            "elapsed_seconds": time.monotonic() - started}


def run(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    from scripts.train_character_edge_full_fit import ResourceGuard
    import torch
    from hanglish.core.score_scale_converter import ScoreScaleConverter
    from hanglish.core.ungated_joint_converter import UngatedJointConverter

    guard = ResourceGuard(output)
    def observe(stage):
        value = guard.observe(stage)
        need(value["seconds"] <= LIMITS["whole_seconds"]
             and value["rss_bytes"] <= LIMITS["rss_bytes"]
             and value["swap_growth_bytes"] <= LIMITS["swap_growth_bytes"],
             "runtime resources exceeded")
        need(sum(p.stat().st_size for p in output.iterdir() if p.is_file())
             <= LIMITS["output_bytes"], "runtime output exceeded")
        return value

    try:
        pins, inventory = authenticate_sources()
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        engine = ScoreScaleConverter(phonetic_checkpoint=MODEL)
        engine.load_models(); engine.load_word_model(); engine.load_growth_model()
        engine.load_phonetic_model()
        need(engine.version == PARENT_VERSION, "production parent version differs")
        wrapper = UngatedJointConverter(engine, known_english=inventory)
        for name, expected in {**engine.model_artifacts, **wrapper.artifacts}.items():
            path = Path(name) if Path(name).is_absolute() else ROOT / name
            need(sha(path) == expected, "runtime dependency differs: " + str(path))
            pins[str(path.resolve())] = expected
        verify_pins(pins); observe("models-loaded")
        selected_ids, timings = [], []
        with (SOURCE / "capture.jsonl").open(encoding="utf-8") as source, \
                (COMPARATOR / "capture.jsonl").open(encoding="utf-8") as comparator, \
                (output / "capture.jsonl").open("x", encoding="utf-8") as target:
            count = 0
            for count, (left, right) in enumerate(zip(source, comparator), 1):
                row = capture_row(json.loads(left), json.loads(right),
                                  engine=engine, inventory=inventory)
                need(row["elapsed_seconds"] <= LIMITS["row_seconds"], "row time exceeded")
                target.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                target.flush(); timings.append(row["elapsed_seconds"])
                if row["selected_unranked"]: selected_ids.append(row["id"])
                observe("row-" + row["id"])
            need(next(source, None) is None and next(comparator, None) is None,
                 "DEV source lengths differ")
        need(count == ROWS and len(selected_ids) == SELECTED_ROWS,
             "DEV row/selection count differs")
        verify_pins(pins)
        final = observe("complete")
        report = {"schema": "ungated-joint-runtime-dev-capture-v1",
                  "completed": True, "rows": count, "selected_rows": len(selected_ids),
                  "selected_ids": selected_ids, "parent_version": PARENT_VERSION,
                  "known_english": inventory, "pins": pins, "limits": LIMITS,
                  "resources": final, "max_rss_bytes": guard.peak_rss,
                  "max_swap_growth_bytes": guard.peak_swap_growth,
                  "timings_seconds": timings,
                  "capture_sha256": sha(output / "capture.jsonl"),
                  "resources_sha256": sha(output / "resources.jsonl"),
                  "labels_opened": False, "test_opened": False,
                  "production_changed": False, "fresh_selected_inference": True,
                  "cached_ranked_passthrough": True}
        write(output / "report.json", report)
        write(output / "capture.complete.json", {
            "completed": True, "rows": count, "selected_rows": len(selected_ids),
            "report_sha256": sha(output / "report.json"),
            "capture_sha256": report["capture_sha256"], "labels_opened": False})
    except Exception as error:
        write(output / "failure.json", {"type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args().output)
