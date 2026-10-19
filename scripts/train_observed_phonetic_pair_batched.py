#!/usr/bin/env python3
"""Execution-only wrapper for the immutable paired phonetic training runner."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import train_observed_phonetic_pair as base


def configure() -> None:
    base.RECIPE = dict(base.RECIPE, decoding_execution="batched-active-beams-v1")
    for path in (
        Path(__file__),
        ROOT / "hanglish/models/batched_phonetic_decode.py",
        ROOT / ".omx/plans/observed-phonetic-pair-execution-v2.md",
    ):
        base.FIXED_HASHES[path] = base.sha256_file(path)
    original = base.torch_modules

    def modules():
        result = original()
        from hanglish.models.batched_phonetic_decode import generate_batched

        class BatchedPhoneticTransducer(result["PhoneticTransducer"]):
            def generate(self, roman, beam_size=4):
                return generate_batched(self, roman, beam_size)

        result["PhoneticTransducer"] = BatchedPhoneticTransducer
        return result

    base.torch_modules = modules


if __name__ == "__main__":
    configure()
    base.main()
