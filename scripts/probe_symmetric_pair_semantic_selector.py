"""Run the fixed comparison-head recipe with qualified semantic-only views.

This isolated process temporarily supplies a new source-view builder to the
frozen v1 experiment. It restores both functions on exit and changes no runtime.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import probe_symmetric_pair_selector as experiment
from scripts import symmetric_pair_text_layout as layout

OUTPUT = ROOT / ".omx/artifacts/symmetric-pair-semantic-selector-fit-v1"
LAYOUT_REPORT_SHA = "de4186dad93e2ceb26abfcab9bb8aa95e3b2fba3190e7b872c438bd7af2dcc4f"


def run(output=OUTPUT):
    report_path = layout.OUTPUT / "report.json"
    if experiment.sha(report_path) != LAYOUT_REPORT_SHA:
        raise ValueError("fixed source-only layout qualification changed")
    qualification = json.loads(report_path.read_bytes())
    if (qualification["eligible_pairs"] != qualification["pairs"]
            or qualification["pairs"] != 1608
            or not qualification["complete_roman_and_candidate_text_roundtrip"]):
        raise ValueError("every source pair must fit without text truncation")
    for name, expected in qualification["source_hashes"].items():
        if experiment.sha(ROOT / name) != expected:
            raise ValueError(f"layout source changed: {name}")
    if experiment.sha(layout.OUTPUT / "views.jsonl") != qualification["views_sha256"]:
        raise ValueError("qualified source views changed")
    original_prepare, original_views = experiment.prepare, experiment.pair_views

    def prepare(directory):
        rows, pairs, pins = original_prepare(directory)
        qualified = [json.loads(line) for line in (layout.OUTPUT / "views.jsonl").read_bytes().splitlines()]
        if len(qualified) != len(pairs):
            raise ValueError("qualified pair count differs")
        for pair, view in zip(pairs, qualified):
            if (pair["index"] != view["index"] or pair["row_index"] != view["row_index"]
                    or pair["views"] != view["views"]):
                raise ValueError("regenerated semantic pair differs from qualified view")
        old_sources = experiment.OUTPUT
        if experiment.sha(directory / "sources.json") != experiment.sha(old_sources / "sources.json"):
            raise ValueError("fixed source rows or folds changed")
        for path in (Path(__file__), Path(layout.__file__), report_path, layout.OUTPUT / "views.jsonl"):
            pins[str(path.relative_to(ROOT))] = experiment.sha(path)
        config = json.loads((directory / "configuration.json").read_bytes())
        config.update({"pins": pins, "representation": "semantic-text-only-v1",
            "removed_neural_fields": qualification["removed_model_fields"],
            "layout_qualification_sha256": LAYOUT_REPORT_SHA,
            "previous_run_preserved": str(experiment.OUTPUT.relative_to(ROOT))})
        experiment.write_json(directory / "configuration.json", config)
        return rows, pairs, pins

    experiment.prepare, experiment.pair_views = prepare, layout.semantic_views
    try:
        return experiment.run(output)
    finally:
        experiment.prepare, experiment.pair_views = original_prepare, original_views


if __name__ == "__main__":
    result = run()
    raise SystemExit(0 if result["completed"] else 1)
