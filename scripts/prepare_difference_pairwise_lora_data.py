"""Build the frozen difference-local KEEP/SWITCH LoRA data artifact.

Run ``fit-bridge`` with ``.venv`` (the strict FIT loader imports Torch), then
run ``prepare`` with ``.venv-mlx`` so the pinned Qwen tokenizer is exercised
without loading model weights.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.final_choice_policy import finite_candidate_indices, proposal_allowed


SCHEMA = "difference-pairwise-lora-data-v1"
NATURAL_FIT = ROOT / "data/natural_context/final-choice-v1/natural_fit"
CHALLENGE_FIT = ROOT / "data/natural_context/final-choice-v1/challenge_fit"
NATURAL_MANIFEST_SHA256 = "fc22b1635cc496ed70aa757750944726d583d640f2d9f3a598e380f3e262a1c9"
CHALLENGE_MANIFEST_SHA256 = "2b8a6349e0ad091e797d86694c2a5165b81fa906a91c3d540ece9042b1b66194"
OLD_DATA = ROOT / ".omx/artifacts/long-context-lora-data-v1-review2"
OLD_DATA_MANIFEST_SHA256 = "cb4558249aa9acefb246abca9275ba8020f39cee27eb9d4143dfb94f266ecd06"
OLD_DIAGNOSTIC_SHA256 = "15f7c6414cc17e70b37da2466ec3e180f7c842c1f635a862465793fa04a096f6"
MODEL_DIR = ROOT / "data/external/qwen3-4b-mlx-4bit"
MODEL_REVISION = "52a5ab34fa604bc8af6d3ce0cac0cab10b7eb495"
MODEL_SOURCE_SHA256 = "67928c796b7e4562c072b2bbbdd0d52569e5ab9118ade1302f8d8d1bb4bae819"
MODEL_WEIGHTS_SHA256 = "e56a94b846a2dce7d1cc154ce58b9cfc4b8d99e8ffb74bfe256a265f4cbc7fb9"
CHAT_TEMPLATE_SHA256 = "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
OLD_TRAINING_EXAMPLES_SHA256 = "fa19eb34f9a31b50b896ee0c19ccbb58e86ec2aee8c9ca5bc23bf74f50ba4b1b"
PLAN_SHA256 = "340a53f28d09b76598942898c6b8084e6e05550e79ba837e0f28c5585925e18d"
TEST_SPEC_SHA256 = "7dde72e45a80176a0a0f29447f9a87e0cdadfcdac946f07cff578d8f9b2b88e8"
HANDOFF_SESSION_ID = "01a09d52-067f-78b1-8f7a-1b081335ac90"
PLAN = ROOT / ".omx/plans/lora-successor-v1.md"
TEST_SPEC = ROOT / ".omx/plans/lora-successor-test-spec-v1.md"
HANDOFF = ROOT / ".omx/plans/lora-successor-handoff-v1.json"
COMPACT_PLAN = ROOT / ".omx/plans/lora-successor-compact-kind-v1.md"
COMPACT_TEST_SPEC = ROOT / ".omx/plans/lora-successor-compact-kind-test-spec-v1.md"
COMPACT_HANDOFF = ROOT / ".omx/plans/lora-successor-compact-kind-handoff-v1.json"
COMPACT_INTENT = ROOT / ".omx/artifacts/difference-pairwise-lora-compact-kind-intent-v1.json"
COMPACT_PREFLIGHT = ROOT / ".omx/artifacts/difference-pairwise-compact-preflight-v1.json"
COMPACT_PREFLIGHT_PROOF = ROOT / ".omx/artifacts/difference-pairwise-compact-preflight-v1-proof.json"
PRECOMPACT_ARCHIVE = ROOT / ".omx/artifacts/difference-pairwise-lora-data-v1-precompact-archive/manifest.json"
VERIFIER = ROOT / "scripts/verify_difference_pairwise_lora.py"
COMPACT_BRIDGE = ROOT / ".omx/artifacts/difference-pairwise-lora-compact-kind-bridge-v1.json"
OUTPUT = ROOT / ".omx/artifacts/difference-pairwise-lora-data-v1"
REPRESENTATION_VERSION = "shared-alpha-kind-edits-v1"
KIND_CODEBOOK = {"L": "lexicon", "X": "mixed", "M": "morphology", "N": "neural",
                 "D": "numeric", "R": "raw", "S": "syllable"}
SYSTEM_TEXT = (
    "로마자 원문과 문맥을 보고 CURRENT를 유지하면 1, ALTERNATIVE로 바꾸면 2를 답하세요. "
    "K.c는 구간별 kind 코드열(L=lexicon,X=mixed,M=morphology,N=neural,D=numeric,R=raw,S=syllable), "
    "K.a는 CURRENT와 다른 ALTERNATIVE의 0기준 index=code 목록입니다. 숫자 하나만 답하세요."
)
SYSTEM_TEXT_SHA256 = "a31b624a44eb340970ac74ae9d3b768c8033ecc823d0039e35e37a839732cdc2"
COMPACT_PLAN_SHA256 = "0376b670313c015a7c13fa2f984c6abd1a3272e94f5208443a359d633a5784be"
COMPACT_TEST_SPEC_SHA256 = "44e1766a89375bd56e6836823dd6d045ada2c8276f81da3961922396bdce42da"
COMPACT_HANDOFF_SHA256 = "90f3b6382af349bf61b96862b658bb81bb87caaf7acb613db400c0737291f4c4"
BASE_HANDOFF_SHA256 = "785ef5ce678c684563a22f4c21c9dfcf9350a791256e0fe51de8b19e0da71092"
COMPACT_INTENT_SHA256 = "e2009c3c323e523270dde8a6705ff3d754e950a2daea095d57cf0a5435457922"
COMPACT_PREFLIGHT_SHA256 = "f50a6e4ba3e0ad4883591c64dd2b36f31955bc9ba36b32487360c051ea7b1b7a"
COMPACT_PREFLIGHT_PROOF_SHA256 = "e09a70f528f23560f578a28755063f5c1c1f54425538abc2c210810b8d177123"
PRECOMPACT_ARCHIVE_SHA256 = "823f1bc6ecd01cc4bb20a991de3d69c58b4fbff7b127893440fc00f83aacb07a"
PAIR_FIELDS = ("role", "row_id", "current_source_index", "alternative_source_index")
MATCH_FIELDS = ("shape", "context", "span_count", "roman_length", "kind_signature", "neighbor_class")
DROP_FIELDS = ("context", "roman_length", "neighbor_class", "shape", "kind_signature")
GAP_MARKER = "[…중략…]"
MAX_CONTEXT_CHARS = 256
MAX_PROMPT_TOKENS = 1024


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def encode_compact_prompt_payload(semantic: dict):
    """Losslessly replace the two full kind arrays with shared alpha edits."""
    if not isinstance(semantic, dict) or "K" in semantic:
        raise ValueError("reserved K collision")
    compact = deepcopy(semantic)
    try:
        current = compact["CURRENT"].pop("kinds")
        alternative = compact["ALTERNATIVE"].pop("kinds")
    except (KeyError, AttributeError) as error:
        raise ValueError("semantic kind arrays are missing") from error
    if (not isinstance(current, list) or not current or not isinstance(alternative, list)
            or len(current) != len(alternative)):
        raise ValueError("kind sequence lengths differ or are empty")
    reverse = {kind: code for code, kind in KIND_CODEBOOK.items()}
    if any(not isinstance(kind, str) or kind not in reverse for kind in current + alternative):
        raise ValueError("unknown candidate kind")
    compact["K"] = {
        "c": "".join(reverse[kind] for kind in current),
        "a": ",".join(f"{index}={reverse[right]}" for index, (left, right)
                      in enumerate(zip(current, alternative)) if left != right),
    }
    return compact


def decode_compact_prompt_payload(compact: dict):
    """Strict inverse used by producer tests and construction self-checks."""
    if not isinstance(compact, dict):
        raise ValueError("compact payload must be an object")
    semantic = deepcopy(compact)
    try:
        current_object = semantic["CURRENT"]
        alternative_object = semantic["ALTERNATIVE"]
        encoded = semantic.pop("K")
    except (KeyError, AttributeError) as error:
        raise ValueError("compact payload fields are missing") from error
    if (not isinstance(current_object, dict) or not isinstance(alternative_object, dict)
            or "kinds" in current_object or "kinds" in alternative_object):
        raise ValueError("compact payload contains full kind arrays")
    if not isinstance(encoded, dict) or set(encoded) != {"c", "a"}:
        raise ValueError("malformed K object")
    codes, edits = encoded["c"], encoded["a"]
    if (not isinstance(codes, str) or not codes
            or any(code not in KIND_CODEBOOK for code in codes)):
        raise ValueError("invalid current kind codes")
    if not isinstance(edits, str):
        raise ValueError("alternative edits must be a string")
    current = [KIND_CODEBOOK[code] for code in codes]
    alternative = current.copy()
    previous = -1
    for edit in edits.split(",") if edits else []:
        if re.fullmatch(r"(0|[1-9][0-9]*)=[LXMNDRS]", edit) is None:
            raise ValueError("malformed alternative kind edit")
        raw_index, code = edit.split("=")
        index = int(raw_index)
        if not previous < index < len(current):
            raise ValueError("alternative kind edit is out of order or range")
        if current[index] == KIND_CODEBOOK[code]:
            raise ValueError("alternative kind edit is redundant")
        alternative[index] = KIND_CODEBOOK[code]
        previous = index
    current_object["kinds"] = current
    alternative_object["kinds"] = alternative
    return semantic


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def file_digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def path_key(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_jsonl(path: Path):
    raw = path.read_bytes()
    return raw, [json.loads(line) for line in raw.splitlines() if line]


def write_jsonl(path: Path, rows) -> str:
    raw = b"".join(canonical(row) + b"\n" for row in rows)
    path.write_bytes(raw)
    return digest(raw)


def verify_hashes(hashes: dict):
    for name, expected in hashes.items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        if not path.is_file() or file_digest(path) != expected:
            raise ValueError(f"source pin differs: {name}")


def validate_compact_authority():
    expected = {
        COMPACT_PLAN: COMPACT_PLAN_SHA256,
        COMPACT_TEST_SPEC: COMPACT_TEST_SPEC_SHA256,
        COMPACT_HANDOFF: COMPACT_HANDOFF_SHA256,
        HANDOFF: BASE_HANDOFF_SHA256,
        COMPACT_INTENT: COMPACT_INTENT_SHA256,
        COMPACT_PREFLIGHT: COMPACT_PREFLIGHT_SHA256,
        COMPACT_PREFLIGHT_PROOF: COMPACT_PREFLIGHT_PROOF_SHA256,
        PRECOMPACT_ARCHIVE: PRECOMPACT_ARCHIVE_SHA256,
    }
    for path, sha in expected.items():
        if file_digest(path) != sha:
            raise ValueError(f"compact amendment pin differs: {path_key(path)}")
    intent = json.loads(COMPACT_INTENT.read_bytes())
    if (intent.get("schema") != "difference-pairwise-lora-compact-kind-intent-v1"
            or intent.get("representation_version") != REPRESENTATION_VERSION
            or intent.get("codebook") != KIND_CODEBOOK
            or intent.get("system_text") != SYSTEM_TEXT
            or intent.get("system_text_sha256") != SYSTEM_TEXT_SHA256):
        raise ValueError("compact intent codec differs")
    for name, sha in intent.get("files", {}).items():
        if file_digest(ROOT / name) != sha:
            raise ValueError(f"compact intent source differs: {name}")
    handoff = json.loads(COMPACT_HANDOFF.read_bytes())
    if (handoff.get("terminal_planning_state") != "complete"
            or handoff.get("authorized") is not True
            or handoff.get("session_id") != HANDOFF_SESSION_ID
            or handoff.get("architect") != "APPROVE"
            or handoff.get("critic") != "APPROVE"
            or handoff.get("sequential_reviews") is not True):
        raise ValueError("approved compact amendment handoff is required")
    archive = json.loads(PRECOMPACT_ARCHIVE.read_bytes())
    if archive.get("schema") != "pairwise-precompact-archive-v1" or archive.get("immutable") is not True:
        raise ValueError("precompact archive manifest is malformed")
    for entry in archive.get("files", []):
        archive_path = ROOT / entry["archive"]
        if file_digest(archive_path) != entry["sha256"] or archive_path.stat().st_size != entry["bytes"]:
            raise ValueError(f"precompact archive entry differs: {entry['archive']}")
    preflight = json.loads(COMPACT_PREFLIGHT.read_bytes())
    expected_counts = {"pairs": 1928, "units": 59, "references": 232, "updates": 15,
                       "final_update_units": 3, "mechanism_rows": 175,
                       "E_mechanism": 481, "E_train640": 1556}
    if (preflight.get("schema") != "compact-kind-preflight-v1"
            or preflight.get("representation_version") != REPRESENTATION_VERSION
            or preflight.get("representation_passed") is not True
            or preflight.get("data_qualified") is not False
            or preflight.get("counts") != expected_counts
            or preflight.get("statistics") != {"min": 270, "median": 576.5, "p95": 811,
                                                "max": 986, "count": 1928,
                                                "ceiling": 1024, "overflow": 0}
            or len(preflight.get("observations", [])) != 1928):
        raise ValueError("successful compact preflight differs")
    proof = json.loads(COMPACT_PREFLIGHT_PROOF.read_bytes())
    payload = proof.get("payload")
    if (proof.get("schema") != "compact-kind-preflight-proof-v1"
            or not isinstance(payload, dict)
            or proof.get("digest") != digest(canonical(payload))
            or payload.get("passed") is not True
            or payload.get("preflight_sha256") != COMPACT_PREFLIGHT_SHA256
            or payload.get("counts") != expected_counts):
        raise ValueError("independent compact preflight proof differs")
    return intent, preflight, proof, archive


def fit_bridge(output: Path):
    if output.exists():
        raise FileExistsError(output)
    from scripts.train_final_choice_selector import load_fit_role, validate_fit_pair

    intent, preflight, proof, archive = validate_compact_authority()
    loader = ROOT / "scripts/train_final_choice_selector.py"
    inputs = [loader, ROOT / "scripts/final_choice_policy.py", Path(__file__), VERIFIER,
              COMPACT_INTENT, COMPACT_PREFLIGHT, COMPACT_PREFLIGHT_PROOF,
              PRECOMPACT_ARCHIVE, COMPACT_PLAN, COMPACT_TEST_SPEC, COMPACT_HANDOFF, HANDOFF]
    for directory in (NATURAL_FIT, CHALLENGE_FIT):
        inputs.extend(directory / name for name in ("manifest.json", "cases.jsonl", "selection.jsonl"))
    before = {path_key(path): file_digest(path) for path in inputs}
    natural = load_fit_role(NATURAL_FIT, NATURAL_MANIFEST_SHA256, "natural_fit")
    challenge = load_fit_role(CHALLENGE_FIT, CHALLENGE_MANIFEST_SHA256, "challenge_fit")
    validate_fit_pair(natural, challenge)
    report = {
        "schema": "difference-pairwise-lora-compact-kind-bridge-v1",
        "completed": True,
        "representation_version": REPRESENTATION_VERSION,
        "intent": {"path": path_key(COMPACT_INTENT), "sha256": COMPACT_INTENT_SHA256},
        "preflight": {"path": path_key(COMPACT_PREFLIGHT), "sha256": COMPACT_PREFLIGHT_SHA256,
                      "ordered_pair_ids_sha256": preflight["ordered_pair_ids_sha256"],
                      "ordered_token_digest": preflight["ordered_token_digest"]},
        "preflight_proof": {"path": path_key(COMPACT_PREFLIGHT_PROOF),
                            "sha256": COMPACT_PREFLIGHT_PROOF_SHA256,
                            "digest": proof["digest"]},
        "archive_manifest": {"path": path_key(PRECOMPACT_ARCHIVE),
                             "sha256": PRECOMPACT_ARCHIVE_SHA256,
                             "entries": len(archive["files"])},
        "implementation_sources": {path_key(Path(__file__)): file_digest(Path(__file__)),
                                   path_key(VERIFIER): file_digest(VERIFIER)},
        "strict_loader": "scripts.train_final_choice_selector.load_fit_role",
        "strict_pair_validator": "scripts.train_final_choice_selector.validate_fit_pair",
        "torch_environment_required": True,
        "torch_version": importlib.metadata.version("torch"),
        "captures": {
            role: {
                "rows": len(loaded["rows"]),
                "ordered_ids_sha256": digest(canonical([row["id"] for row in loaded["rows"]])),
                "manifest_sha256": digest(loaded["manifest_raw"]),
                "cases_sha256": digest(loaded["cases_raw"]),
                "selection_sha256": digest(loaded["selection_raw"]),
            }
            for role, loaded in (("natural_fit", natural), ("challenge_fit", challenge))
        },
        "input_hashes_before": dict(sorted(before.items())),
    }
    after = {path_key(path): file_digest(path) for path in inputs}
    if before != after:
        raise RuntimeError("strict FIT inputs changed during validation")
    report["input_hashes_after"] = dict(sorted(after.items()))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2,
                                  allow_nan=False).encode("utf-8") + b"\n")
    return report


def validate_bridge(path: Path, role_rows: dict):
    if path.resolve() != COMPACT_BRIDGE.resolve():
        raise ValueError("mandatory compact bridge path differs")
    bridge = json.loads(path.read_bytes())
    if (bridge.get("schema") != "difference-pairwise-lora-compact-kind-bridge-v1"
            or bridge.get("completed") is not True
            or bridge.get("representation_version") != REPRESENTATION_VERSION
            or bridge.get("input_hashes_before") != bridge.get("input_hashes_after")):
        raise ValueError("completed compact-kind FIT bridge is required")
    verify_hashes(bridge["input_hashes_before"])
    if bridge.get("implementation_sources") != {
            path_key(Path(__file__)): file_digest(Path(__file__)),
            path_key(VERIFIER): file_digest(VERIFIER)}:
        raise ValueError("compact bridge implementation binding differs")
    intent, preflight, proof, archive = validate_compact_authority()
    if (bridge.get("intent") != {"path": path_key(COMPACT_INTENT), "sha256": COMPACT_INTENT_SHA256}
            or bridge.get("preflight", {}).get("sha256") != COMPACT_PREFLIGHT_SHA256
            or bridge.get("preflight", {}).get("ordered_pair_ids_sha256") != preflight["ordered_pair_ids_sha256"]
            or bridge.get("preflight", {}).get("ordered_token_digest") != preflight["ordered_token_digest"]
            or bridge.get("preflight_proof", {}).get("sha256") != COMPACT_PREFLIGHT_PROOF_SHA256
            or bridge.get("preflight_proof", {}).get("digest") != proof["digest"]
            or bridge.get("archive_manifest", {}).get("sha256") != PRECOMPACT_ARCHIVE_SHA256
            or bridge.get("archive_manifest", {}).get("entries") != len(archive["files"])):
        raise ValueError("compact bridge evidence binding differs")
    for role, (directory, rows) in role_rows.items():
        expected = {
            "rows": len(rows),
            "ordered_ids_sha256": digest(canonical([row["id"] for row in rows])),
            "manifest_sha256": file_digest(directory / "manifest.json"),
            "cases_sha256": file_digest(directory / "cases.jsonl"),
            "selection_sha256": file_digest(directory / "selection.jsonl"),
        }
        if bridge.get("captures", {}).get(role) != expected:
            raise ValueError(f"strict FIT bridge differs for {role}")
    return bridge


def segment_changes(source: str, parent: dict, alternative: dict):
    left, right = parent.get("segments"), alternative.get("segments")
    if not isinstance(left, list) or not left or not isinstance(right, list) or len(left) != len(right):
        raise ValueError("pair segment lists differ")
    cursor, changes = 0, []
    for index, (current, other) in enumerate(zip(left, right)):
        for segment in (current, other):
            if (not isinstance(segment, dict) or segment.get("input_start") != cursor
                    or type(segment.get("input_end")) is not int
                    or not cursor < segment["input_end"] <= len(source)
                    or segment.get("source") != source[cursor:segment["input_end"]]
                    or not isinstance(segment.get("text"), str)
                    or not isinstance(segment.get("kind"), str)):
                raise ValueError("candidate segment alignment is malformed")
        if (current["input_end"], current["source"]) != (other["input_end"], other["source"]):
            raise ValueError("candidate segment sources differ")
        end = current["input_end"]
        if current["text"] != other["text"]:
            changes.append({
                "segment_index": index,
                "input_start": cursor,
                "input_end": end,
                "roman": source[cursor:end],
                "left_context": source[max(0, cursor - 12):cursor],
                "right_context": source[end:end + 12],
                "current_text": current["text"],
                "current_kind": current["kind"],
                "alternative_text": other["text"],
                "alternative_kind": other["kind"],
            })
        cursor = end
    if cursor != len(source):
        raise ValueError("candidate segments do not cover the full source")
    if parent.get("text") != "".join(x["text"] for x in left):
        raise ValueError("parent text differs from segments")
    if alternative.get("text") != "".join(x["text"] for x in right):
        raise ValueError("alternative text differs from segments")
    if not changes:
        raise ValueError("pair has no changed segment")
    return changes


def external_context_window(context: str, source: str):
    if not isinstance(context, str):
        raise ValueError("external context must be a string")
    if len(context) <= MAX_CONTEXT_CHARS:
        return {"text": context, "strategy": "complete", "original_chars": len(context)}
    found = context.find(source)
    if found >= 0:
        center = found + len(source) // 2
        start = max(0, min(len(context) - MAX_CONTEXT_CHARS, center - MAX_CONTEXT_CHARS // 2))
        return {"text": context[start:start + MAX_CONTEXT_CHARS], "strategy": "centered_source",
                "original_chars": len(context), "window_start": start}
    return {"head": context[:128], "tail": context[-128:],
            "strategy": "head_tail_gap", "original_chars": len(context),
            "source_chars": 256, "gap_marker": GAP_MARKER}


def _source_class(value: str):
    classes = set()
    if any("a" <= char.lower() <= "z" for char in value):
        classes.add("latin")
    if any(char.isdigit() for char in value):
        classes.add("numeric")
    if any(not ("a" <= char.lower() <= "z") and not char.isdigit() for char in value):
        classes.add("symbol")
    if len(classes) > 1:
        return "mixed"
    return next(iter(classes), "none")


def matching_key(source: str, parent: dict, alternative: dict, changes: list):
    changed_indices = {item["segment_index"] for item in changes}
    neighbor_classes = set()
    for index in changed_indices:
        for neighbor in (index - 1, index + 1):
            if 0 <= neighbor < len(parent["segments"]) and neighbor not in changed_indices:
                category = _source_class(parent["segments"][neighbor]["source"])
                if category != "none":
                    neighbor_classes.add(category)
    neighbor = "none" if not neighbor_classes else (
        next(iter(neighbor_classes)) if len(neighbor_classes) == 1 else "mixed")
    length = sum(item["input_end"] - item["input_start"] for item in changes)
    bucket = "1_2" if length <= 2 else "3_4" if length <= 4 else "5_8" if length <= 8 else "9_plus"
    return {
        "shape": "sentence" if any(char.isspace() for char in source) else "single",
        "context": None,
        "span_count": "one" if len(changes) == 1 else "multi",
        "roman_length": bucket,
        "kind_signature": "+".join(
            f"{item['current_kind']}>{item['alternative_kind']}" for item in changes),
        "neighbor_class": neighbor,
    }


def pair_identity(role: str, row_id: str, alternative_index: int):
    identity = {"role": role, "row_id": row_id, "current_source_index": 0,
                "alternative_source_index": alternative_index}
    return digest(canonical(identity))


def build_pair(role: str, row: dict, alternative_index: int):
    candidates = row["candidates"]
    if alternative_index == 0 or alternative_index not in finite_candidate_indices(candidates):
        raise ValueError("pair alternative must be finite and non-parent")
    parent, alternative = candidates[0], candidates[alternative_index]
    if not proposal_allowed(row["input"], parent, alternative):
        raise ValueError("pair alternative is not source-aligned and permitted")
    changes = segment_changes(row["input"], parent, alternative)
    key = matching_key(row["input"], parent, alternative, changes)
    key["context"] = "present" if row["context"].strip() else "empty"
    ranked = sorted(finite_candidate_indices(candidates), key=lambda i: (-candidates[i]["score"], i))
    record = {
        "pair_id": pair_identity(role, row["id"], alternative_index),
        "role": role,
        "row_id": row["id"],
        "group_id": row["group_id"],
        "current_source_index": 0,
        "alternative_source_index": alternative_index,
        "source": row["input"],
        "external_context": external_context_window(row["context"], row["input"]),
        "raw_mode": False,
        "current": {"text": parent["text"], "score": parent["score"],
                    "rank": ranked.index(0) + 1, "kinds": [x["kind"] for x in parent["segments"]]},
        "alternative": {"text": alternative["text"], "score": alternative["score"],
                        "rank": ranked.index(alternative_index) + 1,
                        "kinds": [x["kind"] for x in alternative["segments"]]},
        "changed_spans": changes,
        "match_key": key,
        "candidate_count": len(candidates),
        "current_candidate_sha256": digest(canonical(parent)),
        "alternative_candidate_sha256": digest(canonical(alternative)),
        "candidate_objects_sha256": digest(canonical(candidates)),
        "raw_response_sha256": digest(canonical(row["raw_bypass_response"])),
    }
    target = "KEEP" if parent["text"] in row["expected"] else (
        "SWITCH" if alternative["text"] in row["expected"] else "KEEP")
    return record, target


def permitted_pairs(role: str, row: dict):
    result = []
    for index in finite_candidate_indices(row["candidates"]):
        if index and proposal_allowed(row["input"], row["candidates"][0], row["candidates"][index]):
            result.append(build_pair(role, row, index))
    return result


def _relaxed_equal(left: dict, right: dict, depth: int):
    if not 0 <= depth <= len(DROP_FIELDS):
        raise ValueError("matching relaxation depth is out of range")
    dropped = set(DROP_FIELDS[:depth])
    return all(left[field] == right[field] for field in MATCH_FIELDS if field not in dropped)


def choose_control(unit_id: str, repair_key: dict, pool: list, use_counts: Counter):
    for reused in (False, True):
        candidates = [pair for pair in pool if (use_counts[pair["pair_id"]] > 0) is reused]
        for depth in range(len(DROP_FIELDS) + 1):
            matches = [pair for pair in candidates if _relaxed_equal(repair_key, pair["match_key"], depth)]
            if matches:
                chosen = min(matches, key=lambda pair: (
                    digest(("lora-successor-v1-control\0" + unit_id + "\0" + pair["pair_id"]).encode()),
                    pair["pair_id"]))
                prior = use_counts[chosen["pair_id"]]
                use_counts[chosen["pair_id"]] += 1
                return chosen, {"relaxation_depth": depth, "prior_use_count": prior,
                                "reuse_after_unused_exhaustion": reused}
    raise ValueError("control pool is empty")


def prompt_payload(pair: dict):
    return {
        "roman_input": pair["source"],
        "external_context": pair["external_context"],
        "raw_mode": pair["raw_mode"],
        "mapping": {"1": "KEEP", "2": "SWITCH"},
        "CURRENT": {"source_index": pair["current_source_index"], **pair["current"]},
        "ALTERNATIVE": {"source_index": pair["alternative_source_index"], **pair["alternative"]},
        "changed_spans": pair["changed_spans"],
    }


def make_messages(pair: dict):
    compact = encode_compact_prompt_payload(prompt_payload(pair))
    return [{"role": "system", "content": SYSTEM_TEXT},
            {"role": "user", "content": canonical(compact).decode("utf-8")}]


def tokenize_pair(tokenizer, pair: dict):
    messages = make_messages(pair)
    tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           enable_thinking=False, return_dict=False)
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
    if (not isinstance(tokens, list) or not tokens or any(type(value) is not int for value in tokens)
            or tokenizer.encode(rendered, add_special_tokens=False) != tokens):
        raise ValueError("tokenizer generation prefix differs")
    digit_ids = {}
    for digit in ("1", "2"):
        completed = tokenizer.encode(rendered + digit, add_special_tokens=False)
        if completed[:-1] != tokens or len(completed) != len(tokens) + 1:
            raise ValueError(f"{digit} is not one continuation token")
        digit_ids[digit] = completed[-1]
    if digit_ids["1"] == digit_ids["2"]:
        raise ValueError("answer digit continuation IDs are not distinct")
    return messages, tokens, digit_ids, rendered


def _source_pins(bridge_path: Path):
    paths = [Path(__file__), ROOT / "scripts/final_choice_policy.py",
             ROOT / "scripts/train_final_choice_selector.py", VERIFIER, PLAN, TEST_SPEC, HANDOFF,
             COMPACT_PLAN, COMPACT_TEST_SPEC, COMPACT_HANDOFF, COMPACT_INTENT,
             COMPACT_PREFLIGHT, COMPACT_PREFLIGHT_PROOF, PRECOMPACT_ARCHIVE,
             bridge_path, OLD_DATA / "manifest.json", OLD_DATA / "diagnostic.jsonl",
             OLD_DATA / "training_examples.jsonl"]
    for directory in (NATURAL_FIT, CHALLENGE_FIT):
        paths.extend(directory / name for name in ("manifest.json", "cases.jsonl", "selection.jsonl"))
    paths.extend(MODEL_DIR / name for name in (
        "SOURCE.json", "config.json", "model.safetensors", "model.safetensors.index.json",
        "tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json"))
    return {path_key(path): file_digest(path) for path in paths}


def prepare(output: Path, bridge_path: Path, tokenizer_loader=None):
    if output.exists():
        raise FileExistsError(output)
    if file_digest(OLD_DATA / "manifest.json") != OLD_DATA_MANIFEST_SHA256:
        raise ValueError("old data manifest pin differs")
    if file_digest(OLD_DATA / "diagnostic.jsonl") != OLD_DIAGNOSTIC_SHA256:
        raise ValueError("TRAIN640 diagnostic pin differs")
    compact_intent, compact_preflight, compact_proof, compact_archive = validate_compact_authority()
    handoff = json.loads(HANDOFF.read_bytes())
    if (handoff.get("terminal_planning_state") != "complete"
            or handoff.get("ralplan_execution_handoff", {}).get("authorized") is not True
            or handoff.get("ralplan_architect_review", {}).get("verdict") != "APPROVE"
            or handoff.get("ralplan_critic_review", {}).get("verdict") != "APPROVE"
            or handoff.get("ralplan_architect_review", {}).get("review_cycle") != 2
            or handoff.get("ralplan_critic_review", {}).get("review_cycle") != 2
            or handoff.get("ralplan_execution_handoff", {}).get("review_cycle") != 2
            or handoff.get("ralplan_execution_handoff", {}).get("session_id") != HANDOFF_SESSION_ID
            or handoff.get("ralplan_execution_handoff", {}).get("source") != "user"
            or file_digest(PLAN) != PLAN_SHA256 or file_digest(TEST_SPEC) != TEST_SPEC_SHA256
            or handoff.get("planning_artifacts", {}).get("prd_sha256") != PLAN_SHA256
            or handoff.get("planning_artifacts", {}).get("test_spec_sha256") != TEST_SPEC_SHA256):
        raise ValueError("approved successor handoff is required")

    role_rows = {}
    for role, directory in (("natural_fit", NATURAL_FIT), ("challenge_fit", CHALLENGE_FIT)):
        _, role_rows[role] = read_jsonl(directory / "cases.jsonl")
    validate_bridge(bridge_path, {"natural_fit": (NATURAL_FIT, role_rows["natural_fit"]),
                                  "challenge_fit": (CHALLENGE_FIT, role_rows["challenge_fit"])})
    before = _source_pins(bridge_path)
    verify_hashes(before)

    old_examples_raw, old_examples = read_jsonl(OLD_DATA / "training_examples.jsonl")
    old_manifest = json.loads((OLD_DATA / "manifest.json").read_bytes())
    if (digest(old_examples_raw) != OLD_TRAINING_EXAMPLES_SHA256
            or old_manifest.get("file_hashes", {}).get("training_examples.jsonl") != OLD_TRAINING_EXAMPLES_SHA256):
        raise ValueError("old eligible training-example pin differs")
    old_eligible = {(row["role"], row["id"]) for row in old_examples}
    if len(old_examples) != 2771 or len(old_eligible) != 2771:
        raise ValueError("old eligible FIT inventory differs")

    row_by_identity = {}
    inventory, repairs, natural_pool, challenge_pool = [], [], [], []
    accounting = Counter()
    for role in ("natural_fit", "challenge_fit"):
        for order, row in enumerate(role_rows[role]):
            identity = (role, row["id"])
            if identity in row_by_identity:
                raise ValueError("duplicate FIT row identity")
            row_by_identity[identity] = row
            parent_accepted = row["candidates"][0]["text"] in row["expected"]
            pairs = permitted_pairs(role, row)
            switches = [(pair, target) for pair, target in pairs if target == "SWITCH"]
            wrong_keeps = [(pair, target) for pair, target in pairs if target == "KEEP"]
            if parent_accepted:
                (natural_pool if role == "natural_fit" else challenge_pool).extend(pair for pair, _ in pairs)
                status = "parent_accepted"
            elif len(switches) == 1:
                status = "repair"
                repairs.append({"role": role, "row": row, "switch": switches[0][0],
                                "same_row_keep": min((pair for pair, _ in wrong_keeps),
                                                     key=lambda pair: pair["alternative_source_index"],
                                                     default=None)})
            else:
                status = "parent_wrong_unrepairable"
            accounting[(role, status)] += 1
            inventory.append({
                "role": role, "row_id": row["id"], "row_order": order, "status": status,
                "parent_accepted": parent_accepted, "allowed_pair_count": len(pairs),
                "allowed_switch_count": len(switches), "old_eligible": identity in old_eligible,
                "source_sha256": digest(row["input"].encode()),
                "context_sha256": digest(row["context"].encode()),
                "expected_sha256": digest(canonical(row["expected"])),
                "candidate_objects_sha256": digest(canonical(row["candidates"])),
                "raw_response_sha256": digest(canonical(row["raw_bypass_response"])),
            })
    if len(inventory) != 3492 or len(repairs) != 59:
        raise ValueError("FIT 3492/repair 59 construction facts differ")
    if any(item["role"] != "natural_fit" or (item["role"], item["row"]["id"]) not in old_eligible
           for item in repairs):
        raise ValueError("repair rows are not the frozen old eligible natural set")
    same_count = sum(item["same_row_keep"] is not None for item in repairs)
    if same_count != 55:
        raise ValueError("same-row KEEP 55/4 construction fact differs")

    units, selected_pairs, use_counts = [], {}, {"natural_fit": Counter(), "challenge_fit": Counter()}
    for repair in repairs:
        switch = repair["switch"]
        unit_id = digest(("lora-successor-v1-unit\0" + switch["pair_id"]).encode())
        repair["unit_id"] = unit_id
    repairs.sort(key=lambda item: (
        digest(("lora-successor-v1-repair-order\0" + item["unit_id"]).encode()), item["unit_id"]))
    for repair in repairs:
        unit_id, switch = repair["unit_id"], repair["switch"]
        natural, natural_match = choose_control(unit_id, switch["match_key"], natural_pool,
                                                use_counts["natural_fit"])
        challenge, challenge_match = choose_control(unit_id, switch["match_key"], challenge_pool,
                                                    use_counts["challenge_fit"])
        refs = [
            {"component": "repair_switch", "pair_id": switch["pair_id"], "target": "SWITCH", "weight": 0.5},
        ]
        keeps = [("natural_keep", natural, natural_match),
                 ("challenge_keep", challenge, challenge_match)]
        if repair["same_row_keep"] is not None:
            keeps.append(("same_row_keep", repair["same_row_keep"], {
                "same_row": True, "relaxation_depth": None, "prior_use_count": 0,
                "reuse_after_unused_exhaustion": False,
            }))
        keep_weight = 0.5 / len(keeps)
        for component, pair, match in keeps:
            refs.append({"component": component, "pair_id": pair["pair_id"], "target": "KEEP",
                         "weight": keep_weight, "match": match})
        if not math.isclose(sum(ref["weight"] for ref in refs), 1.0, abs_tol=1e-12):
            raise ValueError("unit weights do not sum to one")
        for pair in [switch, *(pair for _, pair, _ in keeps)]:
            selected_pairs[pair["pair_id"]] = pair
        units.append({"unit_id": unit_id, "order": len(units) + 1,
                      "switch_pair_id": switch["pair_id"], "references": refs})
    if len(units) != 59 or sum(len(unit["references"]) for unit in units) != 232:
        raise ValueError("unit/reference count differs")

    required = {}
    scopes = defaultdict(set)
    targets = {}
    def include(pair, target, scope):
        pair_id = pair["pair_id"]
        if pair_id in required and required[pair_id] != pair:
            raise ValueError("pair identity collision")
        if pair_id in targets and targets[pair_id] != target:
            raise ValueError("pair target collision")
        required[pair_id], targets[pair_id] = pair, target
        scopes[pair_id].add(scope)

    for pair in selected_pairs.values():
        row = row_by_identity[(pair["role"], pair["row_id"])]
        target = "KEEP" if row["candidates"][0]["text"] in row["expected"] else (
            "SWITCH" if row["candidates"][pair["alternative_source_index"]]["text"] in row["expected"] else "KEEP")
        include(pair, target, "training")
    mechanism_rows = {(pair["role"], pair["row_id"]) for pair in selected_pairs.values()}
    for identity in sorted(mechanism_rows):
        for pair, target in permitted_pairs(identity[0], row_by_identity[identity]):
            include(pair, target, "mechanism")

    _, diagnostic = read_jsonl(OLD_DATA / "diagnostic.jsonl")
    if len(diagnostic) != 640:
        raise ValueError("TRAIN640 row count differs")
    train640_rows = []
    for item in diagnostic:
        role, row = item["role"], item["row"]
        identity = (role, row["id"])
        if identity not in row_by_identity or canonical(row_by_identity[identity]) != canonical(row):
            raise ValueError("TRAIN640 row does not bind to FIT capture")
        pairs = permitted_pairs(role, row)
        for pair, target in pairs:
            include(pair, target, "train640")
        train640_rows.append({
            "role": role, "row_id": row["id"], "stratum": item["stratum"],
            "pair_ids": [pair["pair_id"] for pair, _ in pairs],
            "candidate_objects_sha256": digest(canonical(row["candidates"])),
            "raw_response_sha256": digest(canonical(row["raw_bypass_response"])),
            "fallback_reason": None if pairs else "no_permitted_non_parent_alternative",
        })
    e_mechanism = sum("mechanism" in values for values in scopes.values())
    e_train640 = sum("train640" in values for values in scopes.values())

    ordered_pair_ids = sorted(required)
    pair_ids_sha256 = digest(canonical(ordered_pair_ids))
    units_sha256 = digest(canonical(units))
    if (pair_ids_sha256 != compact_preflight["ordered_pair_ids_sha256"]
            or len(required) != 1928 or len(mechanism_rows) != 175
            or e_mechanism != 481 or e_train640 != 1556):
        raise ValueError(f"constructed identities differ from successful compact preflight: "
                         f"pairs={len(required)}/{pair_ids_sha256}, units={units_sha256}, "
                         f"mechanism_rows={len(mechanism_rows)}, E={e_mechanism}/{e_train640}")

    actual_tokenizer = tokenizer_loader is None
    if tokenizer_loader is None:
        from transformers import AutoTokenizer
        tokenizer_loader = lambda: AutoTokenizer.from_pretrained(
            MODEL_DIR, local_files_only=True, trust_remote_code=False)
    versions = {name: importlib.metadata.version(name) for name in ("mlx", "mlx-lm", "transformers")}
    if versions != {"mlx": "0.31.2", "mlx-lm": "0.31.3", "transformers": "5.12.1"}:
        raise ValueError("tokenizer environment package pins differ")
    tokenizer = tokenizer_loader()
    source = json.loads((MODEL_DIR / "SOURCE.json").read_bytes())
    if (source.get("revision") != MODEL_REVISION
            or file_digest(MODEL_DIR / "SOURCE.json") != MODEL_SOURCE_SHA256
            or file_digest(MODEL_DIR / "model.safetensors") != MODEL_WEIGHTS_SHA256):
        raise ValueError("Qwen source revision differs")
    template = tokenizer.chat_template
    if not isinstance(template, str) or digest(template.encode()) != CHAT_TEMPLATE_SHA256:
        raise ValueError("Qwen chat template pin differs")

    observations = {item["pair_id"]: item for item in compact_preflight["observations"]}
    if len(observations) != 1928:
        raise ValueError("compact preflight observations contain duplicate identities")
    pair_rows, prompt_rows, token_rows, target_rows, lengths = [], [], [], [], []
    semantic_hashes, compact_hashes, decoded_hashes, token_hashes = [], [], [], []
    prefix_suffix = None
    digit_ids_seen = set()
    for pair_id in ordered_pair_ids:
        pair = deepcopy(required[pair_id])
        pair["scopes"] = sorted(scopes[pair_id])
        messages, tokens, digit_ids, rendered = tokenize_pair(tokenizer, pair)
        if len(tokens) > MAX_PROMPT_TOKENS:
            raise ValueError(json.dumps({"error": "prompt_overflow", "pair_id": pair_id,
                                         "prompt_tokens": len(tokens), "ceiling": MAX_PROMPT_TOKENS},
                                        sort_keys=True))
        suffix = tokens[-4:]
        if prefix_suffix is None:
            prefix_suffix = suffix
        elif suffix != prefix_suffix:
            raise ValueError("generation-prefix suffix differs across real pairs")
        digit_ids_seen.add((digit_ids["1"], digit_ids["2"]))
        semantic = prompt_payload(pair)
        compact = encode_compact_prompt_payload(semantic)
        decoded = decode_compact_prompt_payload(compact)
        if canonical(decoded) != canonical(semantic):
            raise ValueError("compact prompt does not round-trip to semantic bytes")
        forbidden = {"expected", "accepted", "target", "answer", "category", "outcome"}
        if forbidden & set(compact):
            raise ValueError("prompt contains a target-side field")
        semantic_sha = digest(canonical(semantic))
        compact_sha = digest(canonical(compact))
        decoded_sha = digest(canonical(decoded))
        tokens_sha = digest(canonical(tokens))
        observation = {"pair_id": pair_id, "semantic_sha256": semantic_sha,
                       "compact_sha256": compact_sha, "decoded_sha256": decoded_sha,
                       "tokens_sha256": tokens_sha, "token_count": len(tokens),
                       "digit_continuation_ids": digit_ids}
        if actual_tokenizer and observations.get(pair_id) != observation:
            raise ValueError(f"pair evidence differs from successful compact preflight: {pair_id}")
        semantic_hashes.append(semantic_sha)
        compact_hashes.append(compact_sha)
        decoded_hashes.append(decoded_sha)
        token_hashes.append(tokens_sha)
        lengths.append(len(tokens))
        pair_rows.append(pair)
        prompt_rows.append({"pair_id": pair_id, "messages": messages,
                            "payload_sha256": compact_sha,
                            "semantic_payload_sha256": semantic_sha,
                            "compact_payload_sha256": compact_sha,
                            "decoded_payload_sha256": decoded_sha,
                            "rendered_prefix_sha256": digest(rendered.encode())})
        token_rows.append({"pair_id": pair_id, "tokens": tokens,
                           "tokens_sha256": tokens_sha, "prompt_tokens": len(tokens),
                           "digit_continuation_ids": digit_ids})
        target_rows.append({"pair_id": pair_id, "target": targets[pair_id],
                            "target_digit": 1 if targets[pair_id] == "KEEP" else 2})
    if len(digit_ids_seen) != 1:
        raise ValueError("digit continuation IDs differ by prompt boundary")

    schedule = [{"unit_id": unit["unit_id"], "unit_order": index + 1,
                 "update": index // 4 + 1, "position_in_update": index % 4 + 1}
                for index, unit in enumerate(units)]
    if len(schedule) != 59 or schedule[-1]["update"] != 15 or schedule[-1]["position_in_update"] != 3:
        raise ValueError("59-unit/15-update/tail-three schedule differs")
    sorted_lengths = sorted(lengths)
    token_stats = {
        "count": len(lengths), "min": sorted_lengths[0],
        "median": statistics.median(sorted_lengths),
        "p95": sorted_lengths[math.ceil(0.95 * len(sorted_lengths)) - 1],
        "max": sorted_lengths[-1], "ceiling": MAX_PROMPT_TOKENS,
        "overflow": sum(length > MAX_PROMPT_TOKENS for length in sorted_lengths),
    }
    payload_digests = {
        "ordered_semantic_sha256": digest(canonical(semantic_hashes)),
        "ordered_compact_sha256": digest(canonical(compact_hashes)),
        "ordered_decoded_sha256": digest(canonical(decoded_hashes)),
        "ordered_token_sha256": digest(canonical(token_hashes)),
    }
    if actual_tokenizer and (token_stats != compact_preflight["statistics"]
            or payload_digests["ordered_token_sha256"] != compact_preflight["ordered_token_digest"]):
        raise ValueError("aggregate token evidence differs from successful compact preflight")
    component_counts = Counter(ref["component"] for unit in units for ref in unit["references"])
    source_after = _source_pins(bridge_path)
    verify_hashes(source_after)
    if before != source_after:
        raise RuntimeError("pinned inputs changed during construction")

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=output.name + "-", dir=output.parent))
    try:
        file_hashes = {
            "fit_inventory.jsonl": write_jsonl(tmp / "fit_inventory.jsonl", inventory),
            "pairs.jsonl": write_jsonl(tmp / "pairs.jsonl", pair_rows),
            "units.jsonl": write_jsonl(tmp / "units.jsonl", units),
            "schedule.jsonl": write_jsonl(tmp / "schedule.jsonl", schedule),
            "prompts.jsonl": write_jsonl(tmp / "prompts.jsonl", prompt_rows),
            "tokens.jsonl": write_jsonl(tmp / "tokens.jsonl", token_rows),
            "targets.jsonl": write_jsonl(tmp / "targets.jsonl", target_rows),
            "train640.jsonl": write_jsonl(tmp / "train640.jsonl", train640_rows),
        }
        manifest = {
            "schema": SCHEMA, "completed": True,
            "representation_version": REPRESENTATION_VERSION,
            "created_at": json.loads(COMPACT_HANDOFF.read_bytes())["created_at"],
            "scope": "stage1 FIT construction and tokenizer qualification; no model weights",
            "counts": {
                "fit_rows": len(inventory), "repair_units": len(units),
                "repair_switch_references": component_counts["repair_switch"],
                "natural_keep_references": component_counts["natural_keep"],
                "challenge_keep_references": component_counts["challenge_keep"],
                "same_row_keep_references": component_counts["same_row_keep"],
                "repair_rows_without_same_row_keep": 59 - component_counts["same_row_keep"],
                "component_references": sum(component_counts.values()),
                "unique_training_pairs": len(selected_pairs), "required_unique_pairs": len(pair_rows),
                "mechanism_distinct_rows": len(mechanism_rows), "E_mechanism": e_mechanism,
                "train640_rows": len(train640_rows), "E_train640": e_train640,
                "updates": schedule[-1]["update"], "final_update_units": schedule[-1]["position_in_update"],
            },
            "fit_accounting": {f"{role}:{status}": count
                               for (role, status), count in sorted(accounting.items())},
            "matching": {"fields": list(MATCH_FIELDS),
                         "cumulative_relaxation_drop_order": list(DROP_FIELDS),
                         "unused_before_reuse": True,
                         "natural_unique_controls": len(use_counts["natural_fit"]),
                         "challenge_unique_controls": len(use_counts["challenge_fit"]),
                         "natural_reused_references": sum(max(0, count - 1) for count in use_counts["natural_fit"].values()),
                         "challenge_reused_references": sum(max(0, count - 1) for count in use_counts["challenge_fit"].values())},
            "prompt_contract": {
                "system_text": SYSTEM_TEXT, "system_text_sha256": SYSTEM_TEXT_SHA256,
                "canonical_payload": True,
                "semantic_payload_fields": ["ALTERNATIVE", "CURRENT", "changed_spans",
                                            "external_context", "mapping", "raw_mode", "roman_input"],
                "compact_payload_fields": ["ALTERNATIVE", "CURRENT", "K", "changed_spans",
                                           "external_context", "mapping", "raw_mode", "roman_input"],
                "kind_codec": {"representation_version": REPRESENTATION_VERSION,
                               "codebook": KIND_CODEBOOK,
                               "current_codes_field": "K.c",
                               "alternative_edits_field": "K.a",
                               "alternative_edit_grammar": "index=code comma-separated; zero-based strictly increasing"},
                "codebook": KIND_CODEBOOK,
                "compact_kind_field": "K",
                "current_codes_field": "c",
                "alternative_edits_field": "a",
                "decoded_semantics_byte_equal": True,
                "external_context_max_source_chars": MAX_CONTEXT_CHARS,
                "external_context_gap_marker_is_metadata": True,
                "local_source_context_each_side_chars": 12,
                "mapping": {"1": "KEEP", "2": "SWITCH"},
                "enable_thinking": False, "add_generation_prompt": True,
                "answer_appended": False, "target_fields_separate": True,
            },
            "tokenizer": {
                "model_revision": MODEL_REVISION, "versions": versions,
                "chat_template_sha256": digest(template.encode()),
                "tokenizer_config_sha256": file_digest(MODEL_DIR / "tokenizer_config.json"),
                "generation_prefix_token_suffix": prefix_suffix,
                "digit_continuation_ids": {"1": next(iter(digit_ids_seen))[0],
                                           "2": next(iter(digit_ids_seen))[1]},
                "statistics": token_stats, "padding": False, "truncation": False,
                "model_weights_loaded": False,
            },
            "invariants": {"candidate_objects_preserved_by_sha256": True,
                           "raw_responses_preserved_by_sha256": True,
                           "all_fit_rows_accounted": True,
                           "diagnostic_rows_bound_to_fit_bytes": True,
                           "fresh_dev_final_targets_read": False,
                           "production_modified": False},
            "strict_fit_bridge_sha256": file_digest(bridge_path),
            "strict_bridge": {"path": path_key(bridge_path),
                              "sha256": file_digest(bridge_path),
                              "schema": "difference-pairwise-lora-compact-kind-bridge-v1"},
            "compact_lineage": {
                "intent": {"path": path_key(COMPACT_INTENT), "sha256": COMPACT_INTENT_SHA256},
                "archive_manifest": {"path": path_key(PRECOMPACT_ARCHIVE),
                                     "sha256": PRECOMPACT_ARCHIVE_SHA256,
                                     "entries": len(compact_archive["files"])},
                "preflight": {"path": path_key(COMPACT_PREFLIGHT),
                              "sha256": COMPACT_PREFLIGHT_SHA256},
                "preflight_proof": {"path": path_key(COMPACT_PREFLIGHT_PROOF),
                                    "sha256": COMPACT_PREFLIGHT_PROOF_SHA256,
                                    "digest": compact_proof["digest"]},
                "bridge": {"path": path_key(bridge_path), "sha256": file_digest(bridge_path),
                           "schema": "difference-pairwise-lora-compact-kind-bridge-v1"},
            },
            "payload_digests": payload_digests,
            "identity_digests": {"ordered_pair_ids_sha256": pair_ids_sha256,
                                 "produced_units_sha256": units_sha256},
            "fit_sources": {
                role: {
                    "directory": path_key(directory),
                    "manifest": {"path": path_key(directory / "manifest.json"),
                                 "sha256": file_digest(directory / "manifest.json")},
                    "cases": {"path": path_key(directory / "cases.jsonl"),
                              "sha256": file_digest(directory / "cases.jsonl")},
                    "selection": {"path": path_key(directory / "selection.jsonl"),
                                  "sha256": file_digest(directory / "selection.jsonl")},
                    "rows": len(role_rows[role]),
                }
                for role, directory in (("natural_fit", NATURAL_FIT),
                                        ("challenge_fit", CHALLENGE_FIT))
            },
            "train640_source": {"path": path_key(OLD_DATA / "diagnostic.jsonl"),
                                "sha256": OLD_DIAGNOSTIC_SHA256, "rows": 640},
            "pins": {
                "plan_sha256": PLAN_SHA256, "test_spec_sha256": TEST_SPEC_SHA256,
                "handoff_sha256": BASE_HANDOFF_SHA256,
                "compact_plan_sha256": COMPACT_PLAN_SHA256,
                "compact_test_spec_sha256": COMPACT_TEST_SPEC_SHA256,
                "compact_handoff_sha256": COMPACT_HANDOFF_SHA256,
                "compact_intent_sha256": COMPACT_INTENT_SHA256,
                "compact_preflight_sha256": COMPACT_PREFLIGHT_SHA256,
                "compact_preflight_proof_sha256": COMPACT_PREFLIGHT_PROOF_SHA256,
                "precompact_archive_sha256": PRECOMPACT_ARCHIVE_SHA256,
                "policy_sha256": file_digest(ROOT / "scripts/final_choice_policy.py"),
                "strict_loader_sha256": file_digest(ROOT / "scripts/train_final_choice_selector.py"),
                "producer_sha256": file_digest(Path(__file__)),
                "model_source_sha256": MODEL_SOURCE_SHA256,
                "model_weights_sha256": MODEL_WEIGHTS_SHA256,
                "chat_template_sha256": CHAT_TEMPLATE_SHA256,
                "tokenizer_config_sha256": file_digest(MODEL_DIR / "tokenizer_config.json"),
                "package_versions": versions,
            },
            "old_eligible_examples_sha256": digest(old_examples_raw),
            "old_data_manifest_sha256": OLD_DATA_MANIFEST_SHA256,
            "train640_diagnostic_sha256": OLD_DIAGNOSTIC_SHA256,
            "artifacts": dict(sorted(file_hashes.items())),
            "file_hashes": dict(sorted(file_hashes.items())),
            "source_hashes_before": before, "source_hashes_after": source_after,
            "training_run": False, "resource_probe_run": False, "deployed": False,
        }
        (tmp / "manifest.json").write_bytes(json.dumps(
            manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        ).encode("utf-8") + b"\n")
        tmp.rename(output)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return manifest


def main():
    parser = argparse.ArgumentParser()
    phases = parser.add_subparsers(dest="phase", required=True)
    bridge = phases.add_parser("fit-bridge")
    bridge.add_argument("--output", type=Path, required=True)
    build = phases.add_parser("prepare")
    build.add_argument("--bridge", type=Path, required=True)
    build.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = fit_bridge(args.output) if args.phase == "fit-bridge" else prepare(args.output, args.bridge)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
