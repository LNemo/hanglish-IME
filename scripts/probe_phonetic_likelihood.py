"""TRAIN-only numerical verification of frozen teacher-forced likelihood."""
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

MODEL = ROOT / ".omx/artifacts/observed-phonetic-augmented-training-v1/phonetic_transducer.pt"
MODEL_SHA = "f06b0a7cd6c5ccbdfaa376cc29f59e90b3fcf0ae3a157fd2c4f7efaa73b4180d"
DATA = ROOT / ".omx/artifacts/phonetic-supported-data-v1/pairs.jsonl"
DATA_SHA = "cf3e43f75420c17fbe8e1ec118a29a0f6a71c0e55cedbdbae5d7ef3668c84a84"
LIMITS = {"whole_seconds": 600., "update_seconds": 60.,
          "rss_bytes": 2 * 1024**3, "swap_growth_bytes": 512 * 1024**2}
TOLERANCE = 0.0001


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


def cohort(rows):
    pairs = sorted({(r["roman"], r["surface"]) for r in rows if r["split"] == "train"},
                   key=lambda p: (len(p[0]), len(p[1]), p))
    if len(pairs) != 51361:
        raise ValueError("frozen TRAIN denominator differs")
    chosen = [pairs[i * (len(pairs) - 1) // 31] for i in range(32)]
    longest_target = max(pairs, key=lambda p: (len(p[1]), len(p[0]), p))
    if longest_target not in chosen:
        chosen.append(longest_target)
    chosen.append(chosen[0])  # Identity/order and padding regression for duplicate requests.
    if (len(chosen) != 34 or len(set(chosen)) != 33 or chosen[-1] != chosen[0]
            or max(len(p[0]) for p in chosen) != 39
            or max(len(p[1]) for p in chosen) != 13):
        raise ValueError("fixed TRAIN numerical cohort shape differs")
    return validate_pairs(chosen)


def scalar_reference(model, source, target):
    """Independent explicit decoder loop; does not call network.forward/scorer."""
    import torch
    from hanglish.models.phonetic_transducer import (
        INPUT_TO_ID, OUTPUT_TO_ID, BOS, EOS, hangul_to_jamo,
    )
    source_ids = torch.tensor([[INPUT_TO_ID[BOS], *(INPUT_TO_ID[c] for c in source),
                                INPUT_TO_ID[EOS]]], device=model.device)
    tokens = [OUTPUT_TO_ID[t] for t in hangul_to_jamo(target)] + [OUTPUT_TO_ID[EOS]]
    total = 0.; previous = OUTPUT_TO_ID[BOS]
    with torch.inference_mode():
        encoded, hidden, mask = model.network.encode(source_ids)
        for token in tokens:
            logits, hidden = model.network.decode_step(
                torch.tensor([previous], device=model.device), hidden, encoded, mask)
            total += float(torch.log_softmax(logits[0], dim=-1)[token])
            previous = token
    return total


def tensor_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.network.state_dict().items()):
        digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run(output):
    started = time.monotonic(); baseline = system_swap_bytes()
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    try:
        with AtomicResourceWatchdog(started, baseline, output / "resources.jsonl",
                                    limits=LIMITS, interval_s=.5) as watchdog:
            with watchdog.operation("authenticate", update=True):
                if sha(MODEL) != MODEL_SHA or sha(DATA) != DATA_SHA:
                    raise ValueError("frozen model or TRAIN data differs")
                pins = {str(p.relative_to(ROOT)): sha(p) for p in (
                    Path(__file__), MODEL, MODEL.with_suffix(".json"), DATA,
                    ROOT / "hanglish/models/phonetic_likelihood.py",
                    ROOT / "hanglish/models/phonetic_transducer.py",
                    ROOT / "scripts/source_span_resources.py",
                    ROOT / "scripts/preflight_pko_seq2seq.py")}
                pairs = cohort([json.loads(line) for line in DATA.read_text().splitlines()])
                write(output / "pins.json", pins)
                write(output / "requests.json", pairs)
                beam_sources = list(dict.fromkeys(p[0] for p in pairs))[:8]
                if len(beam_sources) != 8:
                    raise ValueError("eight unique TRAIN decoder sources required")
                write(output / "beam-sources.json", beam_sources)
            with watchdog.operation("load_model", update=True):
                import torch
                from hanglish.models.phonetic_transducer import PhoneticTransducer
                torch.set_num_threads(2); torch.set_num_interop_threads(1)
                torch.use_deterministic_algorithms(True)
                model = PhoneticTransducer.load(MODEL); before = tensor_hash(model)
            with watchdog.operation("batch_score", update=True):
                batch = score_pairs(model, pairs, batch_size=64)
            with watchdog.operation("single_score", update=True):
                singles = score_pairs(model, pairs, batch_size=1)
            with watchdog.operation("independent_scalar", update=True):
                reference = [scalar_reference(model, *p) for p in pairs]
            evidence = []
            for p, actual, single, expected in zip(pairs, batch, singles, reference):
                if (actual["source"], actual["target"]) != p or actual["token_count"] != 3 * len(p[1]) + 1:
                    raise ValueError("score order/target token count differs")
                for score in (actual["log_probability"], single["log_probability"], expected):
                    if not math.isfinite(score) or score > 0:
                        raise ValueError("invalid log probability")
                if max(abs(actual["log_probability"] - expected),
                       abs(single["log_probability"] - expected)) > TOLERANCE:
                    raise ValueError("batch/scalar numerical mismatch")
                evidence.append({**actual, "single_log_probability": single["log_probability"],
                                 "reference_log_probability": expected})
            beam_evidence = []; attempts = []
            with watchdog.operation("decoder_equivalence", update=True):
                for source in beam_sources:
                    outputs = model.generate(source, beam_size=4)
                    attempts.append({"source": source, "generated_count": len(outputs)})
                    values = score_pairs(model, [(source, target) for target, _ in outputs])
                    for (target, expected), value in zip(outputs, values):
                        delta = abs(expected - value["log_probability"])
                        if delta > TOLERANCE:
                            raise ValueError("teacher forcing/beam score mismatch")
                        beam_evidence.append({"source": source, "target": target,
                                              "decoder": expected, "teacher_forced": value["log_probability"],
                                              "absolute_difference": delta})
            with watchdog.operation("finalize", update=True):
                after = tensor_hash(model)
                if before != after or not beam_evidence:
                    raise ValueError("weights changed or no beam scores verified")
                if any(sha(ROOT / key) != digest for key, digest in pins.items()):
                    raise ValueError("source/model/data changed")
                write(output / "scores.json", evidence); write(output / "decoder-scores.json", beam_evidence)
                write(output / "decoder-attempts.json", attempts)
                report = {"schema": "phonetic-likelihood-numerical-v1", "completed": True,
                          "pairs": len(pairs), "decoder_candidates": len(beam_evidence),
                          "decoder_sources_attempted": len(beam_sources),
                          "decoder_sources_nonempty": sum(r["generated_count"] > 0 for r in attempts),
                          "decoder_contract": "eight unique TRAIN sources attempted; every emitted candidate compared",
                          "max_batch_scalar_difference": max(abs(r["log_probability"] - r["reference_log_probability"]) for r in evidence),
                          "max_decoder_difference": max(r["absolute_difference"] for r in beam_evidence),
                          "tolerance": TOLERANCE, "tensor_hash_before": before, "tensor_hash_after": after,
                          "pins": pins, "limits": LIMITS, "train_only": True,
                          "quality_evaluation": False, "production_changed": False,
                          "requests_sha256": sha(output / "requests.json"),
                          "beam_sources_sha256": sha(output / "beam-sources.json"),
                          "decoder_attempts_sha256": sha(output / "decoder-attempts.json"),
                          "score_sha256": sha(output / "scores.json"),
                          "decoder_scores_sha256": sha(output / "decoder-scores.json")}
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
    print(json.dumps({k: result[k] for k in ("completed", "pairs", "decoder_candidates",
                    "max_batch_scalar_difference", "max_decoder_difference")}), flush=True)
