"""Evaluate frozen authored cases; never modifies expectations or trains models."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hanglish.core.workbench import WorkbenchEngine


def distance(a, b):
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def evaluate(path, engine, output, *, context_ranking=True, external_lexicon=True,
             ranker_kind="auto", manifest=None):
    if output.exists():
        raise FileExistsError(f"Preserve the previous exposure report; choose a new output: {output}")
    raw = path.read_bytes()
    cases = [json.loads(line) for line in raw.splitlines() if line]
    source_hash = hashlib.sha256(raw).hexdigest()
    manifest_path = Path(manifest) if manifest is not None else path.parent / "manifest.json"
    manifest_bytes = None
    corpus_identity = {"verified": False, "corpus_version": None, "manifest_sha256": None}
    if manifest is not None or manifest_path.exists():
        manifest_bytes = manifest_path.read_bytes()
        identity = json.loads(manifest_bytes)
        if (identity.get("schema_version") != 2 or not identity.get("corpus_version")
                or identity.get("sha256", {}).get(path.name) != source_hash
                or identity.get("counts", {}).get(path.stem) != len(cases)
                or any(case.get("split") != path.stem for case in cases)):
            raise ValueError("evaluation source does not match the versioned corpus manifest")
        corpus_identity = {"verified": True, "corpus_version": identity["corpus_version"],
                           "manifest": str(manifest_path),
                           "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()}
    initialization_start = time.perf_counter()
    converter = WorkbenchEngine()
    if engine in ("neural", "morphology"):
        from hanglish.core.neural_converter import NeuralConverter
        converter._neural = NeuralConverter(context_ranking=context_ranking, external_lexicon=external_lexicon, ranker_kind=ranker_kind)
    initialization_ms = (time.perf_counter() - initialization_start) * 1000
    outcomes, timings, totals = [], [], Counter()
    categories = defaultdict(Counter)
    for case in cases:
        start = time.perf_counter()
        error = None
        try:
            result = converter.convert(case["input"], case["context"], engine=engine)
            candidates = [item["text"] for item in result["candidates"]]
            version = result["version"]
            resolved_ranker = result.get("ranker")
            model_artifacts = result.get("model_artifacts", {})
        except Exception as exception:
            error = str(exception)
            candidates = []
            version = None
            resolved_ranker = None
            model_artifacts = {}
        elapsed = (time.perf_counter() - start) * 1000
        timings.append(elapsed)
        answers = case["expected"]
        exact = bool(candidates) and candidates[0] in answers
        top5 = any(candidate in answers for candidate in candidates)
        char_errors, reference_length = min(
            (distance(candidates[0] if candidates else "", answer), len(answer)) for answer in answers)
        counts = {"total": 1, "top1": int(exact), "top5": int(top5), "errors": int(error is not None),
                  "char_errors": char_errors, "characters": reference_length}
        totals.update(counts)
        categories[case["category"]].update(counts)
        outcomes.append({"id": case["id"], "input": case["input"], "context": case["context"],
                         "expected": answers, "actual": candidates[:5], "top1": exact, "top5": top5,
                         "error": error, "elapsed_ms": round(elapsed, 3), "version": version,
                         "ranker": resolved_ranker, "model_artifacts": model_artifacts})
    resolved_rankers = sorted({row["ranker"] for row in outcomes if row["ranker"] is not None})
    report = {"engine": engine, "requested_ranker": ranker_kind, "resolved_rankers": resolved_rankers,
              "context_ranking": context_ranking, "external_lexicon": external_lexicon,
              "passed": totals["total"] > 0 and totals["top1"] == totals["total"] and totals["errors"] == 0,
              "source": str(path), "sha256": source_hash, "corpus_identity": corpus_identity,
              "counts": dict(totals), "categories": {key: dict(value) for key, value in categories.items()},
              "top1_rate": totals["top1"] / max(1, totals["total"]),
              "top5_rate": totals["top5"] / max(1, totals["total"]),
              "cer": totals["char_errors"] / max(1, totals["characters"]),
              "latency_ms": {"initialization": initialization_ms, "first": timings[0] if timings else None,
                             "median": statistics.median(timings) if timings else None,
                             "p95": sorted(timings)[min(len(timings) - 1, int(len(timings) * .95))] if timings else None},
              "outcomes": outcomes}
    output.parent.mkdir(parents=True, exist_ok=True)
    if path.read_bytes() != raw or (manifest_bytes is not None and manifest_path.read_bytes() != manifest_bytes):
        raise ValueError("evaluation source or manifest changed during inference")
    with output.open("x") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "outcomes"}, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/challenge"))
    parser.add_argument("--engine", choices=("baseline", "rules", "morphology", "neural"), default="neural")
    parser.add_argument("--final", action="store_true", help="Acknowledge opening the held-out test set")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-context-ranking", action="store_true", help="Ablation: same candidates, static order")
    parser.add_argument("--ranker", choices=("auto", "mlp", "semantic", "sentence", "hybrid"), default="auto")
    parser.add_argument("--no-external-lexicon", action="store_true", help="Ablation: omit external runtime word frequencies")
    parser.add_argument("--require-pass", action="store_true", help="Exit nonzero unless all top-1 answers are accepted")
    args = parser.parse_args()
    if args.split == "test" and not args.final:
        parser.error("Use --final only after development evaluation; preserve this exposure report.")
    report = evaluate(args.corpus_dir / f"{args.split}.jsonl", args.engine,
                      args.output or Path(".omx/artifacts") / f"challenge-{args.split}-{args.engine}.json",
                      context_ranking=not args.no_context_ranking, external_lexicon=not args.no_external_lexicon,
                      ranker_kind=args.ranker, manifest=args.corpus_dir / "manifest.json")
    if args.require_pass and not report["passed"]:
        sys.exit(1)
