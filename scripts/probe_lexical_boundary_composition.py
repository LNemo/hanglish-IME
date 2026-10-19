"""Model-free reachability probe on fixed natural FIT512, never DEV/final."""
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hanglish.core.lattice import CandidateGenerator
from scripts.lexical_boundary_composition import BoundaryComposer

CACHED = ROOT / ".omx/artifacts/final-choice-candidate-stages-v1"
OUTPUT = ROOT / ".omx/artifacts/lexical-boundary-composition-fit-v1"
REPORT_SHA = "528f4fbc1ebb2bbc9e22e18b85b4af6e0a205a9a5fe1a64bbe266b52ba46b615"
CASES_SHA = "860da6746310eee896008280a8ad6eac72736b35db16a289aa02a9a5f45264c6"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def composable(target, pieces):
    positions = {0}
    for texts in pieces:
        positions = {offset + len(text) for offset in positions for text in texts
                     if target.startswith(text, offset)}
        if not positions:
            return False
    return len(target) in positions


def _resources(started):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss = rss if sys.platform == "darwin" else rss * 1024
    if rss > 3 * 1024**3 or time.monotonic() - started > 600:
        raise RuntimeError("model-free candidate probe resource cap exceeded")
    return rss


def main():
    started = time.monotonic()
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(CACHED / "report.json") != REPORT_SHA or sha(CACHED / "cases.jsonl") != CASES_SHA:
        raise ValueError("fixed production FIT candidate audit changed")
    audit = json.loads((CACHED / "report.json").read_bytes())
    source_hashes = dict(audit["source_hashes"])
    for name in ("scripts/lexical_boundary_composition.py", "scripts/probe_lexical_boundary_composition.py"):
        source_hashes[name] = sha(ROOT / name)
    for name, expected in source_hashes.items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"source identity changed: {name}")
    rows = [json.loads(line) for line in (CACHED / "cases.jsonl").read_bytes().splitlines()]
    if len(rows) != 512 or len({row["id"] for row in rows}) != 512:
        raise ValueError("complete fixed FIT512 required")
    generator = CandidateGenerator(external_lexicon=True)
    composer = BoundaryComposer(generator)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    proposal_rows = []
    # No expected field is used for generating, restricting or ranking proposals.
    with (OUTPUT / "proposals.jsonl").open("x") as stream:
        for index, row in enumerate(rows):
            additions = []
            for source, offset, options in row["static_pieces"]:
                existing = {option["text"] for option in options}
                proposed = [asdict(option) for option in composer.propose(source)
                            if option.text not in existing]
                additions.append({"source": source, "offset": offset, "options": proposed})
            projected = {"id": row["id"], "additions": additions}
            stream.write(json.dumps(projected, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            proposal_rows.append(projected)
            _resources(started)
            if (index + 1) % 64 == 0:
                print(json.dumps({"source_rows": index + 1, "seconds": time.monotonic() - started}), flush=True)
    counts = Counter()
    outcomes = []
    for row, proposal in zip(rows, proposal_rows):
        before = [{o["text"] for o in p[2]} for p in row["static_pieces"]]
        after = [old | {o["text"] for o in extra["options"]}
                 for old, extra in zip(before, proposal["additions"])]
        baseline = any(composable(target, before) for target in row["expected"])
        added = any(composable(target, after) for target in row["expected"])
        counts["before_composable"] += baseline
        counts["after_composable"] += added
        counts["newly_composable"] += added and not baseline
        counts["lost_composable"] += baseline and not added
        counts["rows_with_proposals"] += any(a["options"] for a in proposal["additions"])
        counts["proposed_pieces"] += sum(bool(a["options"]) for a in proposal["additions"])
        counts["proposed_options"] += sum(len(a["options"]) for a in proposal["additions"])
        outcomes.append({"id": row["id"], "before_composable": baseline, "after_composable": added})
    for name, expected in source_hashes.items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"source identity changed during probe: {name}")
    report = {"schema": "lexical-boundary-composition-fit-probe-v1", "completed": True,
              "scope": "fixed natural FIT512 candidate reachability only", "rows": 512,
              "counts": dict(counts), "cached_report_sha256": REPORT_SHA,
              "cached_cases_sha256": CASES_SHA, "source_hashes": source_hashes,
              "external_lexicon_fingerprint": generator.external_fingerprint,
              "proposals_sha256": sha(OUTPUT / "proposals.jsonl"),
              "model_inference_calls": 0, "development_data_read": False,
              "final_data_read": False, "fresh_data_read": False,
              "first_choice_accuracy_measured": False, "production_changed": False,
              "elapsed_seconds": time.monotonic() - started,
              "peak_rss_bytes": _resources(started), "outcomes": outcomes}
    with (OUTPUT / "report.json").open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({k: report[k] for k in ("counts", "elapsed_seconds", "peak_rss_bytes")}), flush=True)


if __name__ == "__main__":
    main()
