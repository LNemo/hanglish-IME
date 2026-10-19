#!/usr/bin/env python3
"""Correct Unicode stable IDs in the frozen FIT census verifier v1."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts import verify_current_fit_recombination_census as base

BASE_VERIFIER = Path(base.__file__).resolve()
BASE_VERIFIER_SHA256 = "93772051d01bf9213cc7bc540c4c86c1428f2910c66017282e0f1d9b0afdf2d9"
CORRECTION = "Use ensure_ascii=False when reproducing generator stable IDs containing Hangul edge text."


def stable_id(prefix, value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"))
    return prefix + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def verify(output, producer_sha256):
    if base.sha256_file(BASE_VERIFIER) != BASE_VERIFIER_SHA256:
        raise ValueError("frozen v1 verifier identity differs")
    original_stable_id, original_file, original_dump = base.stable_id, base.__file__, base.json.dump

    def dump_with_correction(value, *args, **kwargs):
        if (isinstance(value, dict)
                and value.get("schema") == "current-fit-recombination-census-independent-verification-v1"):
            value["base_verifier_path"] = str(BASE_VERIFIER)
            value["base_verifier_sha256"] = BASE_VERIFIER_SHA256
            value["correction"] = CORRECTION
        return original_dump(value, *args, **kwargs)

    base.stable_id = stable_id
    base.__file__ = __file__
    base.json.dump = dump_with_correction
    try:
        return base.verify(output, producer_sha256)
    finally:
        base.stable_id = original_stable_id
        base.__file__ = original_file
        base.json.dump = original_dump


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--producer-sha256", required=True)
    arguments = parser.parse_args()
    result = verify(arguments.output, arguments.producer_sha256)
    print(json.dumps({key: result[key] for key in
                      ("verified", "mode", "shard", "rows", "qualified")},
                     ensure_ascii=False))
