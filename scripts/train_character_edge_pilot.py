"""Fixed, FIT-only resource/learning pilot; no production model changes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from scripts.audit_recombination_tokenization import digest, require
from scripts.current_recombination_selection import calibrated_score, render_public


def select_rows(rows):
    return sorted(rows, key=lambda row: (
        hashlib.sha256(("character-edge-pilot-v1\0" + row["input"]).encode()).hexdigest(),
        row["id"]))[:32]


def swap_bytes():
    result = subprocess.run(["/usr/sbin/sysctl", "-n", "vm.swapusage"],
                            capture_output=True, text=True, check=True)
    match = re.search(r"used\s*=\s*([0-9.]+)M", result.stdout)
    require(match is not None, "swap telemetry unavailable")
    return float(match.group(1)) * 1024**2


def metrics(row, values):
    if not row["candidates"]:
        public = row["fallback"]
    else:
        public = render_public(row["input"], [dict(candidate, score=float(score))
                                             for candidate, score in zip(row["candidates"], values)])
    texts = [value["text"] for value in public]
    return {"id": row["id"], "first": texts[0] in row["expected"],
            "top5": bool(set(texts) & set(row["expected"])), "public": texts,
            "pool_covered": bool(row["positive_indices"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    proof_root = ROOT / ".omx/artifacts/current-fit-recombination-selection-v1"
    report = json.loads((proof_root / "report.json").read_text())
    proof = json.loads((proof_root / "independent-verification.json").read_text())
    intake_path = ROOT / ".omx/artifacts/current-fit-recombination-selection-intake-v1.json"
    labels_path = ROOT / ".omx/artifacts/current-phonetic-train-features-v1/rows.jsonl"
    requests_path = ROOT / ".omx/artifacts/current-phonetic-train-inputs-v1/requests.json"
    calibration_path = ROOT / ".omx/artifacts/phonetic-score-scale-training-v1/model.json"
    require(proof["verified"] and digest(proof_root / "report.json") == proof["report_sha256"],
            "selection proof mismatch")
    require(digest(intake_path) == report["intake_sha256"]
            and digest(labels_path) == report["features_sha256"]
            and digest(calibration_path) == report["model_sha256"], "FIT inputs differ")
    intake = json.loads(intake_path.read_text())
    for shard in intake["shards"]:
        for name, expected in shard["hashes"].items():
            require(digest(Path(shard["directory"]) / name) == expected, "closed shard differs")
    requests = json.loads(requests_path.read_text())["rows"]
    require(digest(requests_path) == "97b416a805b9da8dc15d4ff9178b3a1db2a98f57f961a0e369c51bc771e1762d",
            "source request manifest differs")
    selected = select_rows(requests)
    args.output.mkdir(parents=True, exist_ok=False)
    selection = {"rule": "SHA256(character-edge-pilot-v1 NUL + input), then ID; first32",
                 "rows": selected, "requests_sha256": digest(requests_path)}
    (args.output / "selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2))
    ids = {row["id"] for row in selected}
    labels = {}
    for line in labels_path.open():
        row = json.loads(line)
        if row["id"] in ids:
            labels[row["id"]] = row
    calibration = json.loads(calibration_path.read_text())
    captures = {}
    for shard in intake["shards"]:
        for line in (Path(shard["directory"]) / "capture.jsonl").open():
            row = json.loads(line)
            if row["id"] in ids:
                captures[row["id"]] = row["scoring"]
    require(set(labels) == ids == set(captures), "pilot denominator differs")
    rows = []
    for request in selected:
        identity = request["id"]; scored = captures[identity]; label = labels[identity]
        require(request["input"] == label["input"] == scored["request"]["input"],
                "pilot source differs")
        candidates = scored["original"] + [c for c in scored["novel"] if c["status"] == "scored"]
        rows.append({"id": identity, "input": request["input"],
                     "component_id": request["component_id"], "expected": label["expected"],
                     "candidates": [{"text": c["text"], "segments": c["segments"]} for c in candidates],
                     "base_scores": [calibrated_score(c, calibration) for c in candidates],
                     "positive_indices": [i for i, c in enumerate(candidates) if c["text"] in label["expected"]],
                     "unsupported": sum(c["status"] == "unsupported" for c in scored["novel"]),
                     "fallback": scored["fallback_public"]})
    del captures, labels
    with (args.output / "data.jsonl").open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    started = time.monotonic()
    swap_start = swap_bytes()
    resources = []

    def observe(stage):
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        item = {"stage": stage, "seconds": time.monotonic() - started,
                "rss_bytes": rss, "swap_growth_bytes": swap_bytes() - swap_start}
        resources.append(item)
        with (args.output / "resources.jsonl").open("a") as stream:
            stream.write(json.dumps(item) + "\n")
        require(item["seconds"] <= 600 and rss <= 2 * 1024**3
                and item["swap_growth_bytes"] <= 512 * 1024**2, "pilot resource limit exceeded")

    observe("start")
    import torch
    from hanglish.models.character_edge_residual import CharacterEdgeResidualRanker

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260923)
    torch.use_deterministic_algorithms(True)
    model = CharacterEdgeResidualRanker(byte_dim=16, hidden=32, head_hidden=32)
    prepared = [model.prepare(row["input"], row["candidates"], row["base_scores"])
                if row["candidates"] else None for row in rows]

    def evaluate():
        records, losses = [], []
        model.eval()
        with torch.no_grad():
            for row, inputs in zip(rows, prepared):
                scores = model(inputs)["scores"] if inputs is not None else []
                records.append(metrics(row, scores))
                if row["positive_indices"]:
                    losses.append(float(torch.logsumexp(scores, 0)
                                        - torch.logsumexp(scores[row["positive_indices"]], 0)))
        return records, sum(losses) / len(losses) if losses else None

    for row, inputs in zip(rows, prepared):
        if inputs is not None:
            with torch.no_grad():
                scores = model(inputs)["scores"]
            require(torch.equal(scores, torch.tensor(row["base_scores"], dtype=torch.float64)),
                    "initial baseline not exact")
    before, initial_loss = evaluate()
    observe("initialized")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    history, updates = [], 0
    for epoch in range(4):
        model.train()
        for row, inputs in zip(rows, prepared):
            if not row["positive_indices"]:
                continue
            optimizer.zero_grad(set_to_none=True)
            scores = model(inputs)["scores"]
            loss = torch.logsumexp(scores, 0) - torch.logsumexp(scores[row["positive_indices"]], 0)
            require(bool(torch.isfinite(loss)), "nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step(); updates += 1
            observe(f"epoch-{epoch+1}-update-{updates}")
        after, current_loss = evaluate()
        history.append({"epoch": epoch + 1, "mean_loss": current_loss,
                        "first": sum(r["first"] for r in after), "top5": sum(r["top5"] for r in after)})
        observe(f"epoch-{epoch+1}-complete")
        print(json.dumps(history[-1]), flush=True)
    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), "nonfinite checkpoint")
    checkpoint = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    torch.save(checkpoint, args.output / "model.pt")
    observe("complete")
    result = {"completed": True, "rows": len(rows), "trainable_rows": sum(bool(r["positive_indices"]) for r in rows),
              "updates": updates, "parameter_count": sum(p.numel() for p in model.parameters()),
              "initial_loss": initial_loss, "final_loss": current_loss, "history": history,
              "initial_baseline_exact": True,
              "before": {k: sum(r[k] for r in before) for k in ("first", "top5", "pool_covered")},
              "after": {k: sum(r[k] for r in after) for k in ("first", "top5", "pool_covered")},
              "corrections": sum(not a["first"] and b["first"] for a,b in zip(before,after)),
              "regressions": sum(a["first"] and not b["first"] for a,b in zip(before,after)),
              "top5_losses": sum(a["top5"] and not b["top5"] for a,b in zip(before,after)),
              "unsupported_candidates": sum(r["unsupported"] for r in rows),
              "resources": {"seconds": resources[-1]["seconds"], "max_rss_bytes": max(r["rss_bytes"] for r in resources),
                            "max_swap_growth_bytes": max(r["swap_growth_bytes"] for r in resources)},
              "hashes": {name: digest(args.output/name) for name in ("selection.json", "data.jsonl", "model.pt", "resources.jsonl")},
              "runner_sha256": digest(__file__), "model_source_sha256": digest(ROOT/"hanglish/models/character_edge_residual.py"),
              "plan_sha256": digest(ROOT/".omx/plans/character-edge-residual-pilot-v1.md"),
              "selection_report_sha256": proof["report_sha256"],
              "scope": "32 source-hash-selected FIT rows; resource and learning feasibility only",
              "dev_opened": False, "test_opened": False, "production_changed": False}
    (args.output/"outcomes.json").write_text(json.dumps({"before":before,"after":after}, ensure_ascii=False, indent=2))
    result["hashes"]["outcomes.json"] = digest(args.output/"outcomes.json")
    (args.output/"report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
