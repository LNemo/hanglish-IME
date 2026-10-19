"""Pinned 7,704-row growth entry point for the frozen SentenceCTC trainer."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import random
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from hanglish.models.sentence_ctc import SentenceCTCConfig, training_batch
from scripts import train_sentence_ctc as engine


EXTRA = ROOT / "data/challenge/sentence-train-extra-v3"
PROOF = EXTRA / "independent-verification.json"
EXPECTED_BASE_ROWS = 5656
EXPECTED_EXTRA_ROWS = 2048
EXPECTED_TRAINING_ROWS = 7704
EXPECTED_TRAIN_SHA256 = "e0729e728c0c5bf2827c83b666380c4f8c4757ee204cdb248578327667a0539b"
EXPECTED_MANIFEST_SHA256 = "27d3e2d7afa5c76b5db82ffa8dc269054d58747b67f3a352fd4c9b0e8280d504"
EXPECTED_PROOF_SHA256 = "f3cbd142282c90115c3b98ad550e60e9940127e517a9fa96509cab67aba43926"
EXPECTED_CONFIG = {
    "width": 192, "heads": 4, "encoder_layers": 2, "feedforward": 768,
    "dropout": 0.1, "max_source_tokens": 512, "embedding_scale": 1.0,
    "beam_size": 8, "token_shortlist": 32,
}


def verify_control():
    if (engine.SEED != 20260921 or engine.BATCH != 16 or engine.EPOCHS != 8
            or asdict(SentenceCTCConfig()) != EXPECTED_CONFIG
            or engine.MAX_TRAIN_SECONDS != 1800
            or engine.MAX_EVAL_SECONDS != 900
            or engine.MAX_RSS != 3 * 1024**3):
        raise ValueError("frozen CTC growth control differs")


def verify_growth_inputs(extra=EXTRA, proof_path=PROOF):
    """Validate the immutable supplement and its independent pre-existing proof."""
    extra, proof_path = Path(extra), Path(proof_path)
    train_path, manifest_path = extra / "train.jsonl", extra / "manifest.json"
    if (engine.sha(train_path) != EXPECTED_TRAIN_SHA256
            or engine.sha(manifest_path) != EXPECTED_MANIFEST_SHA256
            or engine.sha(proof_path) != EXPECTED_PROOF_SHA256):
        raise ValueError("sentence CTC growth corpus or proof identity differs")
    manifest = json.loads(manifest_path.read_text())
    proof = json.loads(proof_path.read_text())
    if (manifest.get("completed") is not True
            or manifest.get("corpus_version") != "sentence-train-extra-v3"
            or manifest.get("input_contract") != "roman_sentence_only"
            or manifest.get("split") != "train"
            or manifest.get("rows") != EXPECTED_EXTRA_ROWS
            or manifest.get("counts") != {"train": EXPECTED_EXTRA_ROWS}
            or manifest.get("sha256", {}).get("train.jsonl") != EXPECTED_TRAIN_SHA256):
        raise ValueError("sentence CTC growth manifest contract differs")
    if (proof.get("verified") is not True
            or proof.get("rows") != EXPECTED_EXTRA_ROWS
            or proof.get("retained_rows") != 1024
            or proof.get("new_rows") != 1024
            or proof.get("train_sha256") != EXPECTED_TRAIN_SHA256
            or proof.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256
            or proof.get("source_token_cap") != 512
            or proof.get("target_token_cap") != 512
            or proof.get("model_inference_performed") is not False):
        raise ValueError("sentence CTC growth independent proof contract differs")
    return proof


def growth_pins(extra=EXTRA, proof_path=PROOF):
    """Extend the engine's immutable input identity without changing its globals."""
    pins = engine.input_pins(Path(extra))
    pins[str(Path(__file__))] = engine.sha(Path(__file__))
    pins[str(Path(proof_path))] = engine.sha(Path(proof_path))
    return pins


def validate_rows(rows):
    if len(rows) != EXPECTED_TRAINING_ROWS:
        raise ValueError("CTC growth control requires the pinned 7,704-row TRAIN union")
    # Apply the CTC-specific complete-sequence/frame audit independently of the
    # AR-compatible validation performed by the shared read_training helper.
    for row in rows:
        training_batch([(row["input"], row["expected"][0])])


def prepare_output(output, resume):
    output = Path(output)
    if resume:
        if (not (output / "recovery.pt").is_file()
                or (output / "report.json").exists()
                or (output / "failure.json").exists()
                or (output / "model.pt").exists()
                or (output / "dev-predictions.jsonl").exists()):
            raise ValueError("resume requires an interrupted training-phase CTC checkpoint")
    else:
        output.mkdir(parents=True, exist_ok=False)
    return output


def run(output, qualification, resume=False, *, train_fn=engine.train):
    verify_control()
    verify_growth_inputs()
    rows = engine.read_training(EXTRA)
    validate_rows(rows)
    pins = growth_pins()
    output = prepare_output(output, resume)
    lock = (output / ".run.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError("CTC growth output is already owned by a live process")
    try:
        train_fn(output, rows, pins, SentenceCTCConfig(), Path(qualification),
                 engine.EPOCHS, resume, EXTRA)
    except Exception as error:
        engine.write_json(output / "failure.json", {
            "completed": False, "error_type": type(error).__name__,
            "error": str(error), "pins": pins,
            "recovery_scope": engine.RECOVERY_SCOPE,
            "production_changed": False,
        })
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(engine.SEED)
    random.seed(engine.SEED)
    run(args.output, args.qualification, args.resume)


if __name__ == "__main__":
    main()
