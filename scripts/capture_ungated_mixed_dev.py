#!/usr/bin/env python3
"""Capture frozen ungated mixed runtime behavior on the closed 316-row DEV source."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.capture_ungated_joint_dev import (
    COMPARATOR, INVENTORY, MODEL, PARENT_VERSION, SOURCE,
    authenticate_sources as authenticate_dev_sources,
    need, read, selected_by_source, sha, verify_pins, write,
)

FIT = ROOT / ".omx/artifacts/ungated-mixed-runtime-fit-v1"
PLAN = ROOT / ".omx/plans/ungated-mixed-dev-v1.md"
ROWS = 316
SELECTED_ROWS = 4
LIMITS = {"whole_seconds": 900.0, "row_seconds": 120.0,
          "rss_bytes": 3 * 1024**3, "swap_growth_bytes": 512 * 1024**2,
          "output_bytes": 128 * 1024**2}


def authenticate():
    pins, inventory = authenticate_dev_sources()
    report, closure, transfer = (read(FIT / "report.json"),
                                 read(FIT / "capture.complete.json"),
                                 read(FIT / "metric-transfer.json"))
    need(report.get("completed") is True and report.get("rows") == 13
         and report.get("labels_opened") is False
         and report.get("production_changed") is False
         and report.get("fresh_parent_inference") is True
         and report.get("fresh_mixed_word_inference") is True
         and closure == {"completed": True, "rows": 13,
             "report_sha256": sha(FIT / "report.json"),
             "capture_sha256": sha(FIT / "capture.jsonl"), "labels_opened": False}
         and transfer.get("verified") is True and transfer.get("rows") == 13
         and transfer.get("runtime_report_sha256") == sha(FIT / "report.json")
         and transfer.get("runtime_capture_sha256") == sha(FIT / "capture.jsonl"),
         "successful closed mixed FIT runtime required")
    verify_pins(report["pins"])
    pins.update(report["pins"])
    for path in (FIT / "report.json", FIT / "capture.complete.json",
                 FIT / "metric-transfer.json", FIT / "capture.jsonl",
                 PLAN, Path(__file__).resolve()):
        pins[str(path.resolve())] = sha(path)
    verify_pins(pins)
    return pins, inventory


class _Blocked:
    def __getattr__(self, name):
        raise AssertionError("ranked passthrough attempted model access: " + name)
    def __call__(self, *args, **kwargs):
        raise AssertionError("ranked passthrough attempted callback")


class CachedParent:
    def __init__(self, response):
        self.response = deepcopy(response); self.lock = threading.RLock()
        self.sentence_ranker = _Blocked(); self.word_options = _Blocked()
        self.version = response.get("version")
    def convert(self, text, context="", engine="neural", mode="batch", raw=False):
        need(text == self.response["input"] and context == self.response.get("context", ""),
             "cached request differs")
        return deepcopy(self.response)
    def load_phonetic_model(self):
        raise AssertionError("ranked passthrough attempted phonetic inference")


class RecordingRanker:
    def __init__(self, ranker):
        self.ranker = ranker; self.calls = []
    def can_score(self, text): return self.ranker.can_score(text)
    def score_pairs(self, contexts, texts):
        scores = self.ranker.score_pairs(contexts, texts)
        self.calls.append({"contexts": list(contexts), "texts": list(texts),
                           "scores": [float(value) for value in scores]})
        return scores


class FreshBaselineParent:
    def __init__(self, engine, baseline):
        self.engine = engine; self.baseline = deepcopy(baseline); self.lock = engine.lock
        self.version = engine.version; self.sentence_ranker = RecordingRanker(engine.sentence_ranker)
        self.convert_calls = 0
    def convert(self, text, context="", engine="neural", mode="batch", raw=False):
        self.convert_calls += 1
        need(self.convert_calls == 1 and text == self.baseline["input"],
             "fresh baseline dispatch differs")
        return deepcopy(self.baseline)
    def load_phonetic_model(self): return self.engine.load_phonetic_model()
    def word_options(self, source, learned=True):
        return self.engine.word_options(source, learned=learned)


def _validate_segments(source, text, segments):
    need(isinstance(segments, list) and segments, "candidate segments missing")
    cursor = 0
    for segment in segments:
        start, end = segment.get("input_start"), segment.get("input_end")
        need(type(start) is int and type(end) is int and start == cursor and end > start
             and source[start:end] == segment.get("source"), "candidate source offsets differ")
        if segment.get("kind") == "raw":
            need(segment.get("text") == segment.get("source"), "raw source changed")
        cursor = end
    need(cursor == len(source) and text == "".join(s["text"] for s in segments),
         "candidate render differs")


def validate_selected(source, response, trace, calls):
    need(trace.get("baseline", {}).get("input") == source
         and isinstance(trace.get("joint_trace"), dict)
         and isinstance(trace.get("mixed_proposals"), list)
         and isinstance(trace.get("union_pool"), list), "mixed trace differs")
    joint_pool = trace["joint_trace"].get("pool", [])
    need(len(trace["union_pool"]) >= len(joint_pool), "joint pool was dropped")
    for old, union in zip(joint_pool, trace["union_pool"]):
        need(old.get("text") == union.get("text")
             and old.get("segments") == union.get("segments")
             and old.get("supported") == union.get("supported")
             and old.get("score") == union.get("score")
             and union.get("origins", [])[:len(old.get("origins", []))]
                 == old.get("origins", []),
             "joint pool candidate was not preserved")
    seen = set()
    for row in trace["union_pool"]:
        need(row.get("text") not in seen and isinstance(row.get("origins"), list)
             and row["origins"] and type(row.get("supported")) is bool,
             "union candidate identity/support differs")
        seen.add(row["text"]); _validate_segments(source, row["text"], row["segments"])
        need((row["supported"] and isinstance(row.get("score"), float)
              and math.isfinite(row["score"]))
             or (not row["supported"] and row.get("score") is None),
             "union score differs")
    for row in trace["mixed_proposals"]:
        _validate_segments(source, row["text"], row["segments"])
    joint_supported = [row["text"] for row in trace["joint_trace"].get("pool", [])
                       if row.get("supported")]
    new_mixed = [row["text"] for row in trace["union_pool"]
                 if row.get("origins") == ["mixed"] and row.get("supported")]
    expected_calls = [group for group in (joint_supported, new_mixed) if group]
    need([call["texts"] for call in calls] == expected_calls
         and all(call["contexts"] == [""] * len(call["texts"]) for call in calls),
         "sentence score call sequence differs")
    ranked = sorted((row for row in trace["union_pool"] if row["supported"]),
                    key=lambda row: (-row["score"], row["text"]))
    if ranked:
        baseline_raw = [row for row in trace["baseline"].get("candidates", [])
                        if row.get("text") == source and row.get("segments")
                        and all(segment.get("kind") == "raw"
                                for segment in row["segments"])]
        need(trace["supported_order"] == [row["text"] for row in ranked]
             and [row["text"] for row in response["candidates"][:-1]]
                 == [row["text"] for row in ranked[:4]]
             and len(baseline_raw) == 1
             and response["candidates"][-1] == baseline_raw[0],
             "union public ordering/raw reserve differs")
    return {"pool_size": len(trace["union_pool"]),
            "supported": len(ranked), "unsupported": len(trace["union_pool"]) - len(ranked),
            "mixed_proposals": len(trace["mixed_proposals"]),
            "sentence_score_calls": len(calls)}


def capture_row(source_row, comparator_row, *, engine=None, inventory=()):
    from hanglish.core.ungated_mixed_converter import UngatedMixedConverter
    need(source_row.get("id") == comparator_row.get("id")
         and source_row.get("request") == comparator_row.get("request"),
         "DEV source/comparator identity differs")
    request, cached = source_row["request"], comparator_row.get("response")
    need(request == {"input": request.get("input"), "context": ""}
         and isinstance(cached, dict) and cached.get("version") == PARENT_VERSION,
         "DEV request/production baseline differs")
    started = time.monotonic()
    if not selected_by_source(source_row):
        wrapper = UngatedMixedConverter(CachedParent(cached), known_english=inventory)
        response = wrapper.convert(request["input"])
        need(response == cached and wrapper.last_trace.get("reason") == "joint_passthrough",
             "ranked response did not pass through exactly")
        return {"id": source_row["id"], "request": deepcopy(request),
                "selected_unranked": False, "passthrough_exact": True,
                "cached_response_sha256": hashlib.sha256(json.dumps(
                    cached, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
                "elapsed_seconds": time.monotonic() - started}
    need(engine is not None, "fresh engine required")
    baseline = engine.convert(request["input"])
    need(baseline.get("version") == PARENT_VERSION
         and baseline.get("sentence_ranking_applied") is False
         and baseline.get("candidates") == cached.get("candidates"),
         "fresh parent candidate identity differs")
    proxy = FreshBaselineParent(engine, baseline)
    wrapper = UngatedMixedConverter(proxy, known_english=inventory)
    response = wrapper.convert(request["input"]); trace = wrapper.last_trace
    need(proxy.convert_calls == 1 and trace.get("baseline") == baseline,
         "fresh original baseline differs")
    controls = validate_selected(request["input"], response, trace,
                                 proxy.sentence_ranker.calls)
    controls.update(parent_version=baseline["version"], parent_ranked=False,
                    parent_candidates_exact_cached=True,
                    wrapper_applied=bool(trace.get("applied")))
    return {"id": source_row["id"], "request": deepcopy(request),
            "selected_unranked": True, "cached_production": deepcopy(cached),
            "fresh_baseline": baseline, "response": response, "trace": trace,
            "sentence_score_calls": proxy.sentence_ranker.calls,
            "controls": controls, "elapsed_seconds": time.monotonic() - started}


def run(output: Path):
    output.mkdir(parents=True, exist_ok=False)
    from scripts.train_character_edge_full_fit import ResourceGuard
    import torch
    from hanglish.core.score_scale_converter import ScoreScaleConverter
    from hanglish.core.ungated_mixed_converter import UngatedMixedConverter
    guard = ResourceGuard(output)
    def observe(stage):
        value = guard.observe(stage)
        need(value["seconds"] <= LIMITS["whole_seconds"]
             and value["rss_bytes"] <= LIMITS["rss_bytes"]
             and value["swap_growth_bytes"] <= LIMITS["swap_growth_bytes"],
             "resource limit exceeded")
        need(sum(p.stat().st_size for p in output.iterdir() if p.is_file())
             <= LIMITS["output_bytes"], "output limit exceeded")
        return value
    try:
        pins, inventory = authenticate()
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        engine = ScoreScaleConverter(phonetic_checkpoint=MODEL)
        engine.load_models(); engine.load_word_model(); engine.load_growth_model()
        engine.load_phonetic_model()
        need(engine.version == PARENT_VERSION, "production parent version differs")
        wrapper = UngatedMixedConverter(engine, known_english=inventory)
        for name, expected in {**engine.model_artifacts, **wrapper.artifacts}.items():
            path = Path(name) if Path(name).is_absolute() else ROOT / name
            need(sha(path) == expected, "runtime dependency differs: " + str(path))
            pins[str(path.resolve())] = expected
        verify_pins(pins); observe("models-loaded")
        selected_ids=[]; timings=[]
        with (SOURCE/"capture.jsonl").open() as left, (COMPARATOR/"capture.jsonl").open() as right, \
                (output/"capture.jsonl").open("x") as target:
            count=0
            for count,(a,b) in enumerate(zip(left,right),1):
                row=capture_row(json.loads(a),json.loads(b),engine=engine,inventory=inventory)
                need(row["elapsed_seconds"] <= LIMITS["row_seconds"], "row time exceeded")
                target.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n");target.flush()
                timings.append(row["elapsed_seconds"])
                if row["selected_unranked"]: selected_ids.append(row["id"])
                observe("row-"+row["id"])
            need(next(left,None) is None and next(right,None) is None,"DEV source lengths differ")
        need(count==ROWS and len(selected_ids)==SELECTED_ROWS,"DEV row/selection count differs")
        verify_pins(pins); final=observe("complete")
        report={"schema":"ungated-mixed-runtime-dev-capture-v1","completed":True,
                "rows":count,"selected_rows":len(selected_ids),"selected_ids":selected_ids,
                "parent_version":PARENT_VERSION,"pins":pins,"limits":LIMITS,
                "resources":final,"max_rss_bytes":guard.peak_rss,
                "max_swap_growth_bytes":guard.peak_swap_growth,"timings_seconds":timings,
                "capture_sha256":sha(output/"capture.jsonl"),
                "resources_sha256":sha(output/"resources.jsonl"),
                "labels_opened":False,"test_opened":False,"production_changed":False,
                "fresh_selected_inference":True,"cached_ranked_passthrough":True}
        write(output/"report.json",report)
        write(output/"capture.complete.json",{"completed":True,"rows":count,
              "selected_rows":len(selected_ids),"report_sha256":sha(output/"report.json"),
              "capture_sha256":report["capture_sha256"],"labels_opened":False})
    except Exception as error:
        write(output/"failure.json",{"type":type(error).__name__,"error":str(error)});raise


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",required=True,type=Path)
    run(parser.parse_args().output)
