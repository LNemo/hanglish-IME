"""Resource-qualified likelihood capture for the fixed, full TRAIN candidate set."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hanglish.models.phonetic_likelihood import score_pairs, validate_pairs
from scripts.source_span_resources import AtomicResourceWatchdog, system_swap_bytes

PREPARED = ROOT / ".omx/artifacts/phonetic-likelihood-fit-v1"
MANIFEST_SHA = "f93ec4333f994717fdbebcea539227f293bebdaf5fdf5bf6545645f2f8dd75dd"
NUMERICAL = ROOT / ".omx/artifacts/phonetic-likelihood-numerical-v1"
MODEL = ROOT / ".omx/artifacts/observed-phonetic-augmented-training-v1/phonetic_transducer.pt"
MODEL_SHA = "f06b0a7cd6c5ccbdfaa376cc29f59e90b3fcf0ae3a157fd2c4f7efaa73b4180d"
LIMITS = {"whole_seconds": 1800., "update_seconds": 60.,
          "rss_bytes": 2 * 1024**3, "swap_growth_bytes": 512 * 1024**2}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


def groups_for(requests):
    require(bool(requests), "empty likelihood requests")
    ordered = sorted(requests, key=lambda r: (len(r["target"]), len(r["source"]), r["pair_id"]))
    count = math.ceil(len(ordered) / 64); small, extra = divmod(len(ordered), count)
    groups, offset = [], 0
    for index in range(count):
        size = small + int(index < extra)
        groups.append(ordered[offset:offset + size]); offset += size
    return groups


def work_units(group):
    return len(group) * (max(len(r["source"]) for r in group) + 2
                         + 3 * max(len(r["target"]) for r in group) + 1)


def probe_indices(groups):
    # Fixed length spread plus the largest padded batch; selection uses no outcomes.
    selected = {i * (len(groups) - 1) // 15 for i in range(16)}
    selected.add(max(range(len(groups)), key=lambda i: (work_units(groups[i]), i)))
    selected.add(max(range(len(groups)), key=lambda i: (max(len(r["source"]) for r in groups[i]), i)))
    selected.add(max(range(len(groups)), key=lambda i: (max(len(r["target"]) for r in groups[i]), i)))
    return sorted(selected)


def validate_scores(group, scores):
    require(len(group) == len(scores), "likelihood output count differs")
    result = []
    for request, score in zip(group, scores):
        require(score["source"] == request["source"] and score["target"] == request["target"],
                "likelihood source/target order differs")
        tokens = 3 * len(request["target"]) + 1
        value = score["log_probability"]
        require(type(value) in (float, int) and math.isfinite(value) and value <= 0
                and score["token_count"] == tokens
                and score["mean_log_probability"] == value / tokens,
                "likelihood probability or token count differs")
        result.append({"pair_id": request["pair_id"], **score})
    return result


def authenticate():
    require(sha(PREPARED / "manifest.json") == MANIFEST_SHA, "FIT manifest differs")
    manifest = read(PREPARED / "manifest.json"); proof = read(PREPARED / "independent-verification.json")
    require(proof.get("verified") is True and proof.get("manifest_sha256") == MANIFEST_SHA,
            "FIT independent proof differs")
    require(manifest["counts"]["rows"] == 1483
            and manifest["counts"]["candidates"] == 202603
            and manifest["counts"]["unique_supported_pairs"] == 26544,
            "FIT domain differs")
    for name, digest in manifest["outputs"].items():
        require(sha(PREPARED / name) == digest, "FIT output hash differs")
    for name, digest in manifest["pins"].items():
        require(sha(name) == digest, "FIT source hash differs")
    requests = [json.loads(line) for line in (PREPARED / "requests.jsonl").read_text().splitlines()]
    require(len(requests) == len({r["pair_id"] for r in requests}) == 26544,
            "FIT request identity count differs")
    validate_pairs([(r["source"], r["target"]) for r in requests])
    numerical = read(NUMERICAL / "report.json"); numerical_proof = read(NUMERICAL / "independent-verification.json")
    require(numerical_proof["verified"] is True
            and numerical_proof["report_sha256"] == sha(NUMERICAL / "report.json")
            and numerical["completed"] is True and numerical["tolerance"] == .0001
            and numerical["max_batch_scalar_difference"] <= .0001
            and numerical["max_decoder_difference"] <= .0001
            and numerical["tensor_hash_before"] == numerical["tensor_hash_after"],
            "numerical qualification differs")
    require(sha(MODEL) == MODEL_SHA, "trained checkpoint differs")
    files = [Path(__file__), PREPARED / "manifest.json", PREPARED / "independent-verification.json",
             PREPARED / "requests.jsonl", PREPARED / "rows.jsonl", MODEL, MODEL.with_suffix(".json"),
             NUMERICAL / "report.json", NUMERICAL / "completion.json", NUMERICAL / "independent-verification.json",
             ROOT / "hanglish/models/phonetic_likelihood.py", ROOT / "hanglish/models/phonetic_transducer.py",
             ROOT / "scripts/source_span_resources.py", ROOT / "scripts/preflight_pko_seq2seq.py"]
    pins = {str(p.relative_to(ROOT)): sha(p) for p in files}
    for name, digest in numerical["pins"].items():
        require(sha(ROOT / name) == digest, "numerical source changed")
        pins[name] = digest
    return requests, pins


def tensor_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.network.state_dict().items()):
        digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def capture_group(model, group, watchdog, stream, operation):
    started = time.monotonic()
    with watchdog.operation(operation, update=True):
        values = validate_scores(group, score_pairs(model, [(r["source"], r["target"]) for r in group]))
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        stream.flush(); os.fsync(stream.fileno())
    return values, time.monotonic() - started


def run(output):
    started = time.monotonic(); baseline = system_swap_bytes()
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    try:
        with AtomicResourceWatchdog(started, baseline, output / "resources.jsonl",
                                    limits=LIMITS, interval_s=.5) as watchdog:
            with watchdog.operation("authenticate", update=True):
                requests, pins = authenticate(); groups = groups_for(requests); chosen = probe_indices(groups)
                write(output / "pins.json", pins)
                write(output / "schedule.json", {"group_sizes": list(map(len, groups)), "probe_indices": chosen,
                    "order": "target length, source length, pair ID", "batch_maximum": 64,
                    "group_work_units": list(map(work_units, groups))})
            with watchdog.operation("load_model", update=True):
                import torch
                from hanglish.models.phonetic_transducer import PhoneticTransducer
                torch.set_num_threads(2); torch.set_num_interop_threads(1)
                torch.use_deterministic_algorithms(True)
                model = PhoneticTransducer.load(MODEL); initial = tensor_hash(model)
            cached = {}; timings = []
            with (output / "probe-scores.jsonl").open("x") as stream:
                for i in chosen:
                    values, elapsed = capture_group(model, groups[i], watchdog, stream, f"probe:{i}")
                    cached[i] = values
                    timings.append({"group": i, "work_units": work_units(groups[i]), "seconds": elapsed})
            projected = 1.3 * (time.monotonic() - started + max(r["seconds"] / r["work_units"] for r in timings)
                               * sum(map(work_units, groups))) + 60
            qualification = {"qualified": projected <= LIMITS["whole_seconds"], "projected_seconds": projected,
                             "group_timings": timings, "projection": "1.3*(elapsed+max_seconds_per_work*full_work)+60"}
            write(output / "qualification.json", qualification)
            require(qualification["qualified"], "full FIT likelihood resource projection failed")
            count = 0
            with (output / "scores.jsonl").open("x") as stream:
                for i, group in enumerate(groups):
                    values, _ = capture_group(model, group, watchdog, stream, f"capture:{i}")
                    if i in cached:
                        require(values == cached[i], "repeated frozen group scores changed")
                    count += len(values)
                    if (i + 1) % 50 == 0:
                        print(json.dumps({"captured": count, "total": len(requests),
                                          "elapsed_s": time.monotonic() - started}), flush=True)
            with watchdog.operation("finalize", update=True):
                require(count == 26544 and tensor_hash(model) == initial, "count or frozen model changed")
                require(all(sha(ROOT / k) == v for k, v in pins.items()), "pinned input changed")
                report = {"schema": "phonetic-likelihood-fit-capture-v1", "completed": True,
                    "requests": count, "groups": len(groups), "probe_groups": len(chosen),
                    "pins": pins, "scores_sha256": sha(output / "scores.jsonl"),
                    "probe_scores_sha256": sha(output / "probe-scores.jsonl"),
                    "qualification_sha256": sha(output / "qualification.json"),
                    "schedule_sha256": sha(output / "schedule.json"),
                    "tensor_hash_before": initial, "tensor_hash_after": tensor_hash(model),
                    "limits": LIMITS, "train_only": True, "gold_used_for_scoring": False,
                    "ranking_or_quality_evaluated": False, "production_changed": False,
                    "unsupported_edges_scored": False, "unsupported_edges_removed_from_paths": False}
                write(output / "report.json", report)
            terminal = watchdog.sample_and_append("after_report")
            with watchdog.operation("write_completion", update=True):
                write(output / "completion.json", {"completed": True,
                    "report_sha256": sha(output / "report.json"), "resource_after_report": terminal})
            watchdog.sample_and_append("after_completion")
        return report
    except BaseException as error:
        write(output / "failure.json", {"completed": False, "error": f"{type(error).__name__}: {error}"})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    result = run(parser.parse_args().output)
    print(json.dumps({k: result[k] for k in ("completed", "requests", "groups")}), flush=True)
