"""Independently replay a contextual-character pilot or full FIT checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import train_character_edge_full_fit as fullfit


PILOT_RUNNER = ROOT / "scripts/train_contextual_character.py"
FULL_RUNNER = ROOT / "scripts/train_contextual_character_full.py"
MODEL_SOURCE = ROOT / "hanglish/models/contextual_character_residual.py"
PILOT_DATA = ROOT / ".omx/artifacts/character-edge-residual-pilot-v1/data.jsonl"
EXPECTED_RECIPE = {
    "epochs": 4, "learning_rate": .001, "weight_decay": .01,
    "clip_norm": 1., "seed": 20260923, "threads": 2,
    "interop_threads": 1, "candidate_chunk": 32,
}
OUTPUTS = ("before.jsonl", "after.jsonl", "model.pt", "resources.jsonl")
FULL_LIMITS = {"whole_seconds": 7200., "row_seconds": 30.,
               "rss_bytes": 2 * 1024**3,
               "swap_growth_bytes": 512 * 1024**2}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def read_jsonl(path: Path):
    with path.open() as stream:
        for line in stream:
            require(bool(line.strip()), f"blank JSONL row in {path.name}")
            yield json.loads(line)


def public_texts(source: str, candidates: list[dict], scores) -> list[str]:
    """Reproduce the public five slots without importing selection rendering."""
    require(isinstance(source, str) and source and candidates,
            "ranked public inputs differ")
    values = [float(value) for value in scores]
    require(len(candidates) == len(values)
            and all(math.isfinite(value) for value in values),
            "ranked public scores differ")
    texts = [candidate.get("text") for candidate in candidates]
    require(all(isinstance(text, str) and text for text in texts)
            and len(texts) == len(set(texts)), "candidate texts differ")
    ordered = [text for _, text in sorted(
        zip(values, texts), key=lambda item: (-item[0], item[1]))[:5]]
    if source not in ordered:
        ordered = ordered[:4] + [source]
    return ordered


def multi_positive_loss(scores, positive_indices: list[int]) -> float:
    """Stable listwise loss, independently implemented with scalar math."""
    values = [float(value) for value in scores]
    require(values and positive_indices
            and all(type(index) is int and 0 <= index < len(values)
                    for index in positive_indices)
            and len(positive_indices) == len(set(positive_indices))
            and all(math.isfinite(value) for value in values),
            "multi-positive loss inputs differ")

    def logsumexp(items):
        peak = max(items)
        return peak + math.log(math.fsum(math.exp(item - peak) for item in items))

    return logsumexp(values) - logsumexp([values[index] for index in positive_indices])


def infer_chunks(model, row: dict, chunk_size: int = 32) -> list[float]:
    import torch

    require(type(chunk_size) is int and chunk_size == 32,
            "checkpoint replay requires candidate chunks of 32")
    scores = []
    with torch.no_grad():
        for start in range(0, len(row["candidates"]), chunk_size):
            stop = start + chunk_size
            prepared = model.prepare(
                row["input"], row["candidates"][start:stop],
                row["base_scores"][start:stop])
            result = model(prepared)
            values = result.get("scores") if isinstance(result, dict) else None
            require(isinstance(values, torch.Tensor) and values.ndim == 1,
                    "checkpoint score tensor differs")
            scores.extend(float(value) for value in values.detach().cpu().tolist())
    require(len(scores) == len(row["candidates"])
            and all(math.isfinite(value) for value in scores),
            "checkpoint score count differs")
    return scores


def metric(row: dict, scores) -> dict:
    if row["no_pool"]:
        require(not row["candidates"] and isinstance(row["fallback"], list)
                and row["fallback"], "no-pool fallback differs")
        texts = [candidate["text"] for candidate in row["fallback"]]
    else:
        texts = public_texts(row["input"], row["candidates"], scores)
    expected = set(row["expected"])
    return {
        "id": row["id"], "first": texts[0] in expected,
        "top5": bool(expected.intersection(texts)),
        "pool_covered": bool(row["positive_indices"]), "public": texts,
        "no_pool": row["no_pool"], "unsupported": row["unsupported"],
    }


def summarize(records: list[dict]) -> dict:
    return {"rows": len(records), **{
        key: sum(int(row[key]) for row in records)
        for key in ("first", "top5", "pool_covered", "no_pool", "unsupported")
    }}


def pilot_rows():
    rows = list(read_jsonl(PILOT_DATA))
    require(len(rows) == len({row.get("id") for row in rows}) == 32,
            "pilot denominator differs")
    for row in rows:
        value = dict(row)
        value["no_pool"] = not bool(value.get("candidates"))
        yield value


def full_rows(authenticated):
    for identity, scored in fullfit.iter_captures(authenticated["shards"]):
        yield fullfit.build_row(
            identity, scored, authenticated["labels"][identity],
            authenticated["calibration"])


def authenticate(directory: Path):
    require(directory.is_dir(), "training directory is missing")
    proof_path = directory / "independent-verification.json"
    require(not proof_path.exists(), "independent proof already exists")
    report_path = directory / "report.json"
    report = read_json(report_path)
    report_sha256 = digest(report_path)
    scope = report.get("scope")
    require(report.get("completed") is True and scope in ("pilot", "full")
            and report.get("rows") == (32 if scope == "pilot" else 1642)
            and report.get("recipe") == EXPECTED_RECIPE
            and report.get("parameter_count") == 31681
            and report.get("dev_opened") is False
            and report.get("test_opened") is False
            and report.get("production_changed") is False,
            "completed contextual training report differs")
    runner = PILOT_RUNNER if scope == "pilot" else FULL_RUNNER
    require(report.get("model_source_sha256") == digest(MODEL_SOURCE)
            and runner.is_file()
            and report.get("runner_sha256") == digest(runner),
            "reported source identity differs")
    if scope == "full":
        require(report.get("limits") == FULL_LIMITS,
                "full training resource contract differs")
    hashes = report.get("hashes")
    require(isinstance(hashes, dict) and set(hashes) == set(OUTPUTS),
            "training output hash inventory differs")
    require(all(digest(directory / name) == hashes[name] for name in OUTPUTS),
            "training output changed")
    pins = report.get("pins")
    require(isinstance(pins, dict) and pins
            and all(isinstance(path, str) and isinstance(expected, str)
                    and len(expected) == 64 and digest(Path(path)) == expected
                    for path, expected in pins.items()), "training pins changed")
    resources = list(read_jsonl(directory / "resources.jsonl"))
    require(resources and all(
        isinstance(row.get("stage"), str)
        and type(row.get("seconds")) in (int, float)
        and math.isfinite(row["seconds"]) and row["seconds"] >= 0
        and type(row.get("rss_bytes")) is int
        and 0 <= row["rss_bytes"] <= 2 * 1024**3
        and type(row.get("swap_growth_bytes")) is int
        and row["swap_growth_bytes"] <= 512 * 1024**2
        for row in resources), "training resource trace differs")
    limit = 600 if scope == "pilot" else 7200
    require([row["seconds"] for row in resources]
            == sorted(row["seconds"] for row in resources)
            and all(row["seconds"] <= limit for row in resources),
            "training resource time trace differs")
    reported_resources = report.get("resources")
    require(isinstance(reported_resources, dict)
            and type(reported_resources.get("seconds")) in (int, float)
            and math.isfinite(reported_resources["seconds"])
            and 0 <= reported_resources["seconds"] <= limit
            and type(reported_resources.get("max_rss_bytes")) is int
            and 0 <= reported_resources["max_rss_bytes"] <= 2 * 1024**3
            and type(reported_resources.get("max_swap_growth_bytes")) is int
            and 0 <= reported_resources["max_swap_growth_bytes"] <= 512 * 1024**2
            and reported_resources.get("max_rss_bytes")
                == max(row["rss_bytes"] for row in resources)
            and reported_resources.get("max_swap_growth_bytes")
                == max(0, max(row["swap_growth_bytes"] for row in resources)),
            "reported training resources differ")
    if scope == "pilot":
        require(str(PILOT_DATA) in pins and pins[str(PILOT_DATA)] == digest(PILOT_DATA),
                "authenticated pilot data differs")
        authenticated = None
    else:
        authenticated = fullfit.authenticate_inputs()
        require(all(pins.get(path) == expected
                    for path, expected in authenticated["pins"].items()),
                "full FIT authentication pins differ")
    frozen = {"report_sha256": report_sha256, "hashes": dict(hashes),
              "pins": dict(pins)}
    return report, scope, authenticated, frozen


def verify(directory: Path) -> dict:
    started = time.monotonic()
    report, scope, authenticated, frozen = authenticate(directory)
    import torch
    from hanglish.models.contextual_character_residual import ContextualCharacterResidualRanker

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(EXPECTED_RECIPE["seed"])
    torch.use_deterministic_algorithms(True)
    model = ContextualCharacterResidualRanker()
    require(sum(parameter.numel() for parameter in model.parameters()) == 31681,
            "checkpoint model parameter count differs")
    state = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
    require(isinstance(state, dict), "checkpoint state differs")
    model.load_state_dict(state, strict=True)
    require(all(parameter.device.type == "cpu"
                and bool(torch.isfinite(parameter).all())
                for parameter in model.parameters()), "checkpoint weights differ")
    model.eval()

    saved_before = iter(read_jsonl(directory / "before.jsonl"))
    saved_after = iter(read_jsonl(directory / "after.jsonl"))
    rows = pilot_rows() if scope == "pilot" else full_rows(authenticated)
    before_records, after_records, before_losses, after_losses = [], [], [], []
    row_limit = 30.
    whole_limit = 600 if scope == "pilot" else 7200
    for row in rows:
        row_started = time.monotonic()
        before_scores = [float(value) for value in row["base_scores"]]
        after_scores = infer_chunks(model, row) if row["candidates"] else []
        before = metric(row, before_scores)
        after = metric(row, after_scores)
        require(next(saved_before, None) == before, "saved before row differs")
        require(next(saved_after, None) == after, "saved after row differs")
        before_records.append(before)
        after_records.append(after)
        if row["positive_indices"]:
            before_losses.append(multi_positive_loss(
                before_scores, row["positive_indices"]))
            after_losses.append(multi_positive_loss(
                after_scores, row["positive_indices"]))
        require(time.monotonic() - row_started <= row_limit,
                "independent replay row time limit exceeded")
        require(time.monotonic() - started <= whole_limit,
                "independent replay time limit exceeded")
    require(next(saved_before, None) is None and next(saved_after, None) is None,
            "saved outcome row count differs")
    before_summary, after_summary = summarize(before_records), summarize(after_records)
    require(before_summary == report.get("before")
            and after_summary == report.get("after"),
            "fresh checkpoint metrics differ")
    if scope == "full":
        counts = {
            "ranked_rows": before_summary["rows"] - before_summary["no_pool"],
            "no_pool_rows": before_summary["no_pool"],
            "unsupported_candidates": before_summary["unsupported"],
            "gold_present_rows": before_summary["pool_covered"],
            "missing_gold_rows": (
                before_summary["rows"] - before_summary["pool_covered"]),
        }
        require(counts == report.get("counts")
                and report.get("updates") == 4 * counts["gold_present_rows"],
                "full FIT denominators or update count differ")
    require(before_losses and len(before_losses) == len(after_losses)
            and math.isclose(math.fsum(before_losses) / len(before_losses),
                             report.get("initial_loss", math.nan), abs_tol=1e-6)
            and math.isclose(math.fsum(after_losses) / len(after_losses),
                             report.get("final_loss", math.nan), abs_tol=1e-6),
            "fresh checkpoint mean loss differs")
    corrections = sum(not old["first"] and new["first"]
                      for old, new in zip(before_records, after_records))
    regressions = sum(old["first"] and not new["first"]
                      for old, new in zip(before_records, after_records))
    top5_losses = sum(old["top5"] and not new["top5"]
                      for old, new in zip(before_records, after_records))
    require((corrections, regressions, top5_losses) == (
        report.get("corrections"), report.get("regressions"),
        report.get("top5_losses")), "fresh transition counts differ")
    elapsed = time.monotonic() - started
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss *= 1 if sys.platform == "darwin" else 1024
    require(elapsed <= whole_limit and rss <= 2 * 1024**3,
            "independent replay resource limit exceeded")
    require(digest(directory / "report.json") == frozen["report_sha256"]
            and all(digest(directory / name) == expected
                    for name, expected in frozen["hashes"].items())
            and all(digest(Path(path)) == expected
                    for path, expected in frozen["pins"].items()),
            "training report, outputs, or pins changed during replay")
    result = {
        "schema": "contextual-character-fit-independent-verification-v1",
        "verified": True, "scope": scope, "rows": len(before_records),
        "before": before_summary, "after": after_summary,
        "corrections": corrections, "regressions": regressions,
        "top5_losses": top5_losses,
        "initial_loss": math.fsum(before_losses) / len(before_losses),
        "final_loss": math.fsum(after_losses) / len(after_losses),
        "report_sha256": frozen["report_sha256"],
        "checkpoint_sha256": frozen["hashes"]["model.pt"],
        "before_sha256": frozen["hashes"]["before.jsonl"],
        "after_sha256": frozen["hashes"]["after.jsonl"],
        "resources_sha256": frozen["hashes"]["resources.jsonl"],
        "verifier_sha256": digest(Path(__file__).resolve()),
        "resources": {"seconds": elapsed, "max_rss_bytes": rss,
                      "seconds_limit": whole_limit,
                      "rss_bytes_limit": 2 * 1024**3},
        "candidate_chunk": 32, "dev_opened": False, "test_opened": False,
        "scope_note": (
            "Independent baseline/public rendering, multi-positive loss, "
            "transition metrics, and fresh CPU checkpoint replay over every "
            "authenticated row; no DEV or TEST data opened."
        ),
    }
    with (directory / "independent-verification.json").open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.directory.resolve()), ensure_ascii=False))
