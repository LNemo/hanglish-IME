"""Join frozen FIT paths and phonetic scores without loading any model."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hanglish.models.phonetic_choice_features import FEATURE_NAMES, project_row

FIT = ROOT / ".omx/artifacts/phonetic-likelihood-fit-v1"
CAPTURE = ROOT / ".omx/artifacts/phonetic-likelihood-fit-capture-v1"
PLAN = ROOT / ".omx/plans/phonetic-choice-calibration-v1.md"
FROZEN = {
    FIT / "rows.jsonl": "5eee232c9bba02a071c1b7856f99dd20e8619528b1caca4b045f82314dc27617",
    FIT / "manifest.json": "f93ec4333f994717fdbebcea539227f293bebdaf5fdf5bf6545645f2f8dd75dd",
    FIT / "independent-verification.json": "9b33a03469a6b85e6dc346e74231f87e9e935ce607e5cd895b18a3dd448fe85f",
    CAPTURE / "scores.jsonl": "8f6627450cdce0c8c566ad3197af16bc3700d9880f67ee503d8ddc7eab475933",
    CAPTURE / "report.json": "667f2e34ce360a82a9aeb6e1aae69147d99b6919d5e6cecbfb62eb911ef239c2",
    CAPTURE / "independent-verification.json": "7c05dcc2e93d3b6597e059133009d60c51fe448e029c59352026c7e2cb21ddb9",
}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def records(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True,
                  indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path, expected in FROZEN.items():
        require(digest(path) == expected, f"frozen input changed: {path}")
    pins = {str(path.relative_to(ROOT)): value for path, value in FROZEN.items()}
    for path in (Path(__file__), ROOT / "hanglish/models/phonetic_choice_features.py", PLAN):
        pins[str(path.relative_to(ROOT))] = digest(path)

    scores = {}
    for value in records(CAPTURE / "scores.jsonl"):
        require(value["pair_id"] not in scores, "duplicate phonetic score ID")
        scores[value["pair_id"]] = value
    require(len(scores) == 26544, "phonetic score denominator changed")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "pins.json", pins)
    seen = set()
    joined = set()
    candidate_count = path_count = positive_count = 0
    minimum = [math.inf] * len(FEATURE_NAMES)
    maximum = [-math.inf] * len(FEATURE_NAMES)
    sums = [0.0] * len(FEATURE_NAMES)
    with (args.output / "rows.jsonl").open("x") as stream:
        for row in records(FIT / "rows.jsonl"):
            require(row["id"] not in seen, "duplicate FIT row ID")
            seen.add(row["id"])
            projected = project_row(row, scores)
            require(all(projected[key] == row[key] for key in (
                "id", "input", "component_id", "selected_indices", "candidates", "base_scores", "positive_indices")),
                "candidate or target identity changed")
            require(len(projected["features"]) == len(row["candidates"]),
                    "feature denominator differs")
            require(projected["alternative_path_counts"] == list(map(len, row["candidate_paths"]))
                    and projected["total_paths"] == sum(map(len, row["candidate_paths"])),
                    "alternative path denominator differs")
            for vector in projected["features"]:
                require(len(vector) == len(FEATURE_NAMES)
                        and all(math.isfinite(v) for v in vector),
                        "invalid feature vector")
                for i, v in enumerate(vector):
                    minimum[i] = min(minimum[i], v)
                    maximum[i] = max(maximum[i], v)
                    sums[i] += v
            joined.update(edge["pair_id"] for edge in row["edges"]
                          if edge["status"] == "supported")
            candidate_count += len(row["candidates"])
            path_count += sum(map(len, row["candidate_paths"]))
            positive_count += len(row["positive_indices"])
            stream.write(json.dumps(projected, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    require(len(seen) == 1483 and candidate_count == 202603
            and path_count == 202607 and positive_count == 1483, "full FIT domain changed")
    require(joined == set(scores), "score coverage is not exact")
    for path, expected in pins.items():
        require(digest(ROOT / path) == expected, f"input changed while projecting: {path}")
    write_json(args.output / "manifest.json", {
        "schema": "phonetic-choice-features-v1", "completed": True,
        "pins": pins, "feature_names": list(FEATURE_NAMES),
        "rows_sha256": digest(args.output / "rows.jsonl"),
        "counts": {"rows": len(seen), "candidates": candidate_count,
                   "paths": path_count, "positive_indices": positive_count,
                   "joined_pairs": len(joined)},
        "feature_census": {"minimum": minimum, "maximum": maximum,
                           "mean": [v / candidate_count for v in sums]},
        "contracts": {"all_candidates_preserved": True,
                      "base_scores_and_positive_indices_preserved": True,
                      "all_alternative_paths_used": True,
                      "alternative_feature_aggregation": "arithmetic_mean",
                      "copy_is_reference_category": True,
                      "neural_model_loaded": False, "training_performed": False,
                      "quality_evaluated": False, "production_changed": False},
        "limitations": [
            "Normalized edge compatibility is not a sentence log probability.",
            "Uniform feature averaging is not probabilistic path marginalization.",
            "The frozen FIT pool predates the latest phonetic candidate union.",
        ],
    })
    print(json.dumps({"output": str(args.output), "rows": len(seen),
                      "candidates": candidate_count, "pairs": len(joined)}))


if __name__ == "__main__":
    main()
