"""Actual isolated FIT512 replay with provenance-filtered boundary extras."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import evaluate_lexical_boundary_fit as evaluator
from scripts import lexical_boundary_runtime as runtime
from scripts import audit_final_choice_candidate_stages as audit
from scripts.boundary_provenance_runtime import ProvenanceBoundaryRuntime

OUTPUT = ROOT / ".omx/artifacts/boundary-provenance-runtime-fit-v1"
REPLAY = ROOT / ".omx/artifacts/boundary-provenance-policy-fit-v1.json"
REPLAY_SHA = "c0d8af05f65abeadb3be7a9dbb13dd73031b8e4018bd86ccaf7ee7c9c5546020"


class TwoPassRecorder(audit.SentenceRecorder):
    """Keep both calls while exposing the final overlay to the old evaluator."""

    def __init__(self, delegate):
        super().__init__(delegate)
        self.calls = []

    def reset(self):
        super().reset()
        self.calls = []

    def score_pairs(self, contexts, candidates):
        if len(self.calls) >= 2:
            raise RuntimeError("more than baseline and overlay scoring calls")
        if self.calls:
            baseline = self.calls[0]
            size = len(baseline["candidates"])
            if (candidates[:size] != baseline["candidates"]
                    or contexts[:size] != baseline["contexts"]):
                raise ValueError("overlay must preserve exact baseline scoring prefix")
            super().reset()
        result = super().score_pairs(contexts, candidates)
        self.calls.append({"contexts": list(contexts), "candidates": list(self.candidates),
                           "scores": list(self.scores)})
        return result


def run(output=OUTPUT):
    if evaluator.sha(REPLAY) != REPLAY_SHA:
        raise ValueError("fixed source-policy cached replay changed")
    replay = json.loads(REPLAY.read_bytes())
    if (not replay["counterfactual"]["gate"]["passed"]
            or evaluator.sha(ROOT / "scripts/boundary_proposal_policy.py")
            != replay["bindings"]["policy_sha256"]
            or evaluator.sha(ROOT / "scripts/lexical_boundary_runtime.py")
            != replay["bindings"]["runtime_sha256"]):
        raise ValueError("qualified source policy differs")
    additions = {str(p.relative_to(ROOT)): evaluator.sha(p) for p in (
        Path(__file__), ROOT / "scripts/boundary_provenance_runtime.py",
        ROOT / "scripts/boundary_proposal_policy.py", REPLAY)}
    original_class = runtime.BoundaryCompositionRuntime
    original_recorder = audit.SentenceRecorder
    runtime.BoundaryCompositionRuntime = ProvenanceBoundaryRuntime
    audit.SentenceRecorder = TwoPassRecorder
    try:
        report = evaluator.run(output)
    finally:
        runtime.BoundaryCompositionRuntime = original_class
        audit.SentenceRecorder = original_recorder
    changed = [name for name, digest in additions.items() if evaluator.sha(ROOT / name) != digest]
    if changed:
        report.update({"completed": False, "failure": f"new policy sources changed: {changed}"})
    report["schema"] = "boundary-provenance-runtime-fit-v1"
    report["source_hashes_before"].update(additions)
    if "source_hashes_after" in report:
        report["source_hashes_after"].update(additions)
    report["policy"] = "new extras only; direct provenance; preserve parent whole lexemes"
    report["latency_scope"] = "both baseline-anchor and filtered-overlay engine calls; model load excluded"
    report["cached_replay_is_not_generalization"] = True
    report["recorder"] = "two-pass, baseline prefix verified, final overlay vector captured"
    (output / "report.json").write_bytes(evaluator.canonical(report) + b"\n")
    print(json.dumps({k: report[k] for k in ("completed", "counts", "gate", "failure") if k in report}), flush=True)
    return report


if __name__ == "__main__":
    result = run()
    raise SystemExit(0 if result["completed"] else 1)
