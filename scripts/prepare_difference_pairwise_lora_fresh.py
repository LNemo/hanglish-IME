"""Project verified parent snapshots after the pairwise TRAIN gate.

This bridge prepares inputs only. It reads no fresh target file, loads no model,
and confers no inference, quality, or deployment approval. The numeric child is
not part of the projection. The parent snapshots are historical observations of
the unchanged production engine, not predictions of the new pairwise adapter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.difference_pairwise_lora_evaluation import canonical

SCHEMA = "difference-pairwise-lora-fresh-inputs-v1"
VERSION = "neural-1-867f13596ed5"
SOURCE_FIELDS = ("id", "group_id", "origin", "input", "context")
PARENT = ".omx/artifacts/numeric-composition-downstream-v1"
TRAIN_PROOF = ROOT / ".omx/artifacts/difference-pairwise-lora-train-v2-proof.json"
OUTPUT = ROOT / ".omx/artifacts/difference-pairwise-lora-fresh-inputs-v1"
PINS = {
    PARENT + "-proof.json": "b7777ebe42950dc5e264e728c0f47c894f06b9024fa82ac954a3c2c58b25fd7c",
    PARENT + "/report.json": "209df79c60e01f50855ddaeb9ca245f06b27649bbabbb9d722988994343ccbd9",
    PARENT + "/sources.jsonl": "e8fefa187dd3b87809ac4d8a2b88add86b6f34cd87c5c552580b922d00b55ff6",
    PARENT + "/captures.jsonl": "d4639c041330262ac8444a047e6d841f2512c8ef3f893222969750a5e5b2ddae",
    ".omx/artifacts/long-context-lora-data-v1-review2/fresh_selection.jsonl":
        "fb1257833c99acc406a831336f535d9ef3a786fc38bb08a84902598e5fe2287d",
    ".omx/artifacts/numeric-structure-integration-v1/report.json":
        "77c04334af817e15bc4c5ccc9d27b1d6ab9f9f6a759b4207e752762ae49d3aab",
}


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_bytes().splitlines() if line]


def verify_pins(pins):
    for name, expected in pins.items():
        path = ROOT / name
        if not path.resolve().is_relative_to(ROOT) or sha(path) != expected:
            raise ValueError(f"source binding differs: {name}")


def require_training_proof():
    # This check precedes even reading the parent capture/source bytes. Replaying
    # the independent verifier prevents a hand-written success envelope from
    # opening this gate. Its label access is confined to the existing FIT data.
    from scripts import verify_difference_pairwise_lora_training as verifier

    proof_hash = sha(TRAIN_PROOF)
    envelope = json.loads(TRAIN_PROOF.read_bytes())
    payload = envelope.get("payload", {})
    if (envelope.get("schema") != verifier.SCHEMA
            or envelope.get("digest") != hashlib.sha256(canonical(payload)).hexdigest()
            or payload.get("verification_passed") is not True
            or payload.get("quality_gate_passed") is not True
            or payload.get("downstream_evaluation_allowed") is not True):
        raise ValueError("accepted pairwise TRAIN proof required")
    if payload != verifier.verify_training() or sha(TRAIN_PROOF) != proof_hash:
        raise ValueError("pairwise TRAIN proof replay differs")
    return {"path": str(TRAIN_PROOF.relative_to(ROOT)), "sha256": proof_hash,
            "adapter": payload["adapter"], "logical_fit_id": payload["logical_fit_id"]}


def project_parent_rows(sources, captures, fresh_selection, model_artifacts):
    """Keep every fixed parent row; never inspect child outcomes or targets."""
    if len(sources) != 888 or len(captures) != 888 or len(fresh_selection) != 512:
        raise ValueError("complete 512 fresh + 376 preservation rows required")
    if not isinstance(model_artifacts, dict) or not model_artifacts:
        raise ValueError("parent model identity required")
    if [{"id": s["id"], "group_id": s["group_id"]} for s in sources[:512]] != fresh_selection:
        raise ValueError("frozen fresh order or document selection differs")
    if len({s["group_id"] for s in sources[:512]}) != 387:
        raise ValueError("fresh document group count differs")
    rows, seen = [], set()
    for index, (source, capture) in enumerate(zip(sources, captures)):
        fields = {key: source[key] for key in SOURCE_FIELDS}
        if (any(not isinstance(value, str) for value in fields.values())
                or any(not fields[key] for key in ("id", "group_id", "input"))
                or fields["id"] in seen
                or any(capture[key] != fields[key] for key in SOURCE_FIELDS)):
            raise ValueError("parent source identity or order differs")
        seen.add(fields["id"])
        parent, raw = capture["parent"], capture["parent_raw"]
        for response, raw_mode in ((parent, False), (raw, True)):
            if (not isinstance(response, dict)
                    or response.get("input") != fields["input"]
                    or response.get("context") != fields["context"]
                    or response.get("version") != VERSION
                    or response.get("raw") is not raw_mode):
                raise ValueError("parent response source, version, or raw mode differs")
        candidates = parent.get("candidates")
        if (not isinstance(candidates, list) or not candidates
                or parent.get("result") != candidates[0].get("text")
                or parent.get("model_artifacts") != model_artifacts
                or raw.get("result") != fields["input"]):
            raise ValueError("parent result or model identity differs")
        canonical(parent)
        canonical(raw)
        rows.append({**fields,
            "role": "fresh_natural" if index < 512 else "challenge_preservation",
            "parent_response": parent, "candidates": candidates,
            "raw_bypass_response": raw})
    return rows


def prepare(output=OUTPUT):
    training = require_training_proof()
    output = Path(output)
    if output.exists():
        raise ValueError("fresh inputs output already exists")
    verify_pins(PINS)
    proof = json.loads((ROOT / (PARENT + "-proof.json")).read_bytes())["payload"]
    runtime = json.loads((ROOT / ".omx/artifacts/numeric-structure-integration-v1/report.json").read_bytes())
    if (proof.get("verification_passed") is not True
            or proof.get("runtime_version") != VERSION
            or runtime.get("integrated_runtime_version") != VERSION
            or runtime.get("accepted_weight_hashes") != proof["model_artifacts"]):
        raise ValueError("verified parent provenance differs")
    runtime_pins = {**runtime["integrated_runtime_source_hashes"], **proof["model_artifacts"]}
    verify_pins(runtime_pins)
    rows = project_parent_rows(
        read_jsonl(ROOT / PARENT / "sources.jsonl"),
        read_jsonl(ROOT / PARENT / "captures.jsonl"),
        read_jsonl(ROOT / ".omx/artifacts/long-context-lora-data-v1-review2/fresh_selection.jsonl"),
        proof["model_artifacts"],
    )
    body = b"".join(canonical(row) + b"\n" for row in rows)
    verify_pins({**PINS, **runtime_pins})
    if sha(TRAIN_PROOF) != training["sha256"]:
        raise ValueError("TRAIN proof changed during projection")
    manifest = {"schema": SCHEMA, "training": training, "runtime_version": VERSION,
        "rows": 888, "fresh_rows": 512, "fresh_groups": 387, "preservation_rows": 376,
        "inputs_sha256": hashlib.sha256(body).hexdigest(),
        "source_hashes": {**PINS, **runtime_pins, str(Path(__file__).relative_to(ROOT)): sha(__file__)},
        "target_files_read": False, "model_loaded": False, "child_predictions_used": False,
        "historical_targets_exposed_in_numeric_lane": True,
        "historical_semantic_family_independence": "unestablished",
        "upstream_pretraining_overlap": "unknown",
        "inference_authorized": False, "promotion_allowed": False}
    output.mkdir(parents=True, exist_ok=False)
    with (output / "inputs.jsonl").open("xb") as stream:
        stream.write(body)
    with (output / "manifest.json").open("xb") as stream:
        stream.write(canonical(manifest) + b"\n")
    return manifest


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    try:
        manifest = prepare()
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"prepared": False, "error": str(error)}), file=sys.stderr)
        return 1
    print(json.dumps({"prepared": True, "rows": manifest["rows"], "inference_authorized": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
