"""Synthetic-only resource/serialization probe for the local warm RoBERTa."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version as package_version
import hashlib
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from hanglish.models.sentence_ranker import SentenceRanker, REQUIRED_ARTIFACTS, encode_sentence_pairs
from scripts.train_final_choice_selector import (
    SelectorExample, selector_loss, require_finite_parameters, validate_scores,
)
from scripts.train_natural_sentence_ranker import rss_bytes
from scripts.train_sentence_ranker import _set_trainable_layers, _optimizer

WARM = ROOT / ".omx/artifacts/roberta-context-ranker-v1/checkpoint"
WARM_METADATA_SHA = "29ad7b22c2d01e0caba34b3489aa909778359ea7025926ef0935e88bd0caed9e"
WARM_WEIGHT_SHA = "7e687917ba496d2cec61907c64bafdf88857d471fac2f7f43bfab86c687c6f0a"
WARM_REPORT_SHA = "0c1400b5af87a172c5e1825ef82dfb1c589b8fde4263874149cd3bdd06e1bb91"
PROFILE = {"threads": 2, "device": "cpu", "seed": 23, "top_layers": 2,
           "pairs_per_batch": 20, "group_batch": 4, "candidates_per_group": 5,
           "max_length": 96, "inference_batch_size": 8,
           "optimizer_steps": 11, "warmup_steps": 3,
           "encoder_lr": 1e-5, "head_lr": 5e-5, "weight_decay": .01,
           "temperature": 2., "distillation_weight": 1.,
           "rss_limit_bytes": 3 * 1024**3, "time_limit_seconds": 240}
FROZEN_HELPERS = {
    "hanglish/models/sentence_ranker.py": "72ed3e7223aac215d692f96890ab642cf9c1da82d25c4485c5b901ff832bd1ca",
    "scripts/train_final_choice_selector.py": "e66d7fdef1461b39a2874654aca6be12487b8dc241405ab9694efbf934f92cc8",
    "scripts/train_sentence_ranker.py": "dfc1009b3877ae9f1889d3cdaf7a00531508a73a37bdb8490226ff33d6eb7fce",
    "scripts/train_natural_sentence_ranker.py": "06f0299190b4831c54852fa3cb73a6fe4bcf697f4e6ffe3326d7ef745e3c4a61",
}
OUTPUT = ROOT / ".omx/artifacts/roberta-final-choice-resources-v1/report.json"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def verify_hashes(hashes):
    for name, expected in hashes.items():
        path = ROOT / name
        if digest(path.read_bytes()) != expected:
            raise ValueError(f"resource probe input changed: {name}")


def warm_checkpoint_hashes():
    metadata_path = WARM / "metadata.json"
    metadata_raw = metadata_path.read_bytes()
    report_path = WARM.parent / "report.json"
    report_raw = report_path.read_bytes()
    if digest(metadata_raw) != WARM_METADATA_SHA or digest(report_raw) != WARM_REPORT_SHA:
        raise ValueError("historical warm initializer identity differs")
    metadata, report = json.loads(metadata_raw), json.loads(report_raw)
    artifacts = metadata.get("artifact_hashes", {})
    if (set(artifacts) != REQUIRED_ARTIFACTS
            or artifacts.get("model.safetensors") != WARM_WEIGHT_SHA
            or report.get("checkpoint_sha256") != WARM_WEIGHT_SHA
            or report.get("deployed") is not False
            or metadata.get("base_model") != "klue/roberta-small"
            or metadata.get("max_length") != 96):
        raise ValueError("warm RoBERTa artifact metadata differs")
    hashes = {str((WARM / name).relative_to(ROOT)): h for name, h in artifacts.items()}
    hashes[str(metadata_path.relative_to(ROOT))] = WARM_METADATA_SHA
    hashes[str(report_path.relative_to(ROOT))] = WARM_REPORT_SHA
    verify_hashes(hashes)
    return hashes


def synthetic_examples():
    texts = tuple("오늘 회의에서 새로운 " + word + "에 필요한 자료를 함께 검토했다."
                  for word in ("사업", "작업", "산업", "학업", "기업"))
    context = "이것은 모델 자원만 측정하는 합성 문장이다. " * 80
    return [SelectorExample(f"synthetic-{i}", f"synthetic-group-{i}", "synthetic",
                context, texts, (1,) if i == 0 else (0,), (2., 1., 0., -1., -2.),
                (0, 1, 2, 3, 4), i != 0, 1.) for i in range(4)]


def validate_encoded(encoded):
    shape = tuple(encoded["input_ids"].shape)
    if shape != (PROFILE["pairs_per_batch"], PROFILE["max_length"]):
        raise ValueError("resource probe must exercise twenty full 96-token pairs")
    if not bool(encoded["attention_mask"].all()):
        raise ValueError("maximum resource workload must not contain padding")
    if "token_type_ids" in encoded and bool(encoded["token_type_ids"].any()):
        raise ValueError("RoBERTa pair token-type IDs must be zero")


def require_resources(started):
    if rss_bytes() > PROFILE["rss_limit_bytes"] or time.monotonic() - started > PROFILE["time_limit_seconds"]:
        raise RuntimeError("RoBERTa final-choice resource limit exceeded")


def run(output):
    started = time.monotonic()
    if output.exists():
        raise FileExistsError(output)
    source = Path(__file__).read_bytes()
    before = {**warm_checkpoint_hashes(), **FROZEN_HELPERS,
              str(Path(__file__).relative_to(ROOT)): digest(source)}
    verify_hashes(before)
    torch.set_num_threads(PROFILE["threads"])
    torch.manual_seed(PROFILE["seed"])
    ranker = SentenceRanker.load(WARM)
    ranker.batch_size = PROFILE["inference_batch_size"]
    if ranker.max_length != PROFILE["max_length"] or next(ranker.model.parameters()).device.type != "cpu":
        raise ValueError("loaded warm ranker has unexpected resource bounds")
    layers = _set_trainable_layers(ranker.model, PROFILE["top_layers"])
    examples = synthetic_examples()
    texts = [text for e in examples for text in e.candidates]
    contexts = [e.context for e in examples for _ in e.candidates]
    if not all(ranker.can_score(text) for text in texts):
        raise ValueError("synthetic complete candidate does not fit")
    encoded = encode_sentence_pairs(ranker.tokenizer, texts, contexts, PROFILE["max_length"])
    validate_encoded(encoded)
    require_resources(started)
    optimizer = _optimizer(ranker.model, SimpleNamespace(**PROFILE), head_only=False)
    ranker.model.train()
    train_times, losses = [], []
    for index in range(PROFILE["optimizer_steps"]):
        tick = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        scores = ranker.model(**encoded).logits.reshape(-1)
        loss = selector_loss(scores, examples, PROFILE)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonfinite synthetic loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(ranker.model.parameters(), 1.)
        if not bool(torch.isfinite(norm)):
            raise RuntimeError("nonfinite synthetic gradient")
        optimizer.step()
        require_finite_parameters(ranker.model)
        require_resources(started)
        if index >= PROFILE["warmup_steps"]:
            train_times.append((time.perf_counter() - tick) * 1000)
            losses.append(float(loss.detach()))
    del optimizer, loss, scores, encoded
    ranker.model.eval()
    inference_times = []
    for index in range(PROFILE["optimizer_steps"]):
        tick = time.perf_counter()
        scores = ranker.score_pairs(contexts[:8], texts[:8])
        validate_scores(scores, 8)
        if index >= PROFILE["warmup_steps"]:
            inference_times.append((time.perf_counter() - tick) * 1000)
        require_resources(started)
    with tempfile.TemporaryDirectory(prefix="hanglish-roberta-probe-") as temp:
        bundle = Path(temp)
        ranker.model.save_pretrained(bundle, safe_serialization=True)
        ranker.tokenizer.save_pretrained(bundle)
        artifacts = {p.name: digest(p.read_bytes()) for p in bundle.iterdir() if p.is_file()}
        metadata = {"format_version": 1, "artifact_hashes": artifacts,
                    "max_length": PROFILE["max_length"],
                    "inference_batch_size": PROFILE["inference_batch_size"]}
        (bundle / "metadata.json").write_text(json.dumps(metadata))
        require_resources(started)
        restored = SentenceRanker.load(bundle)
        require_finite_parameters(restored.model)
        saved = restored.score_pairs(contexts[:8], texts[:8])
        validate_scores(saved, 8)
        delta = max(abs(a - b) for a, b in zip(scores, saved))
        if delta > 1e-4:
            raise RuntimeError("temporary checkpoint score roundtrip differs")
        require_resources(started)
    verify_hashes(before)
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "passed": True,
        "scope": "synthetic-only full-candidate RoBERTa training/resource/serialization probe",
        "profile": PROFILE, "warm_checkpoint_sha256": WARM_WEIGHT_SHA,
        "trainable_layers": layers,
        "attention_implementation": ranker.model.config._attn_implementation,
        "train_step_ms": train_times, "train_median_ms": statistics.median(train_times),
        "inference_batch_ms": inference_times, "inference_median_ms": statistics.median(inference_times),
        "synthetic_loss_values": losses, "save_load_score_max_delta": delta,
        "peak_rss_bytes": rss_bytes(), "elapsed_seconds": time.monotonic() - started,
        "source_hashes": before, "source_hashes_after": dict(before),
        "library_versions": {name: package_version(name) for name in ("torch", "transformers", "safetensors")},
        "real_fit_examples_read": False, "evaluation_examples_read": False,
        "accuracy_measured": False, "synthetic_optimizer_steps": PROFILE["optimizer_steps"],
        "persistent_checkpoint_saved": False, "temporary_checkpoint_deleted": True,
        "model_training_recipe_selected": False, "deployed": False}
    require_resources(started)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("passed", "train_median_ms", "inference_median_ms",
                                           "peak_rss_bytes", "save_load_score_max_delta")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    run(parser.parse_args().output)
