"""Independent verifier for difference-local pairwise LoRA evidence.

Stage 1 validates the immutable data artifact without loading model weights.
The verifier deliberately reimplements pair eligibility, matching, targeting,
ordering, prompt construction, and schedule arithmetic instead of importing the
producer.  Later resource/training/evaluation stages fail closed until their
independent verification is implemented.
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
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hanglish.core.phonology import romanize_word


SCHEMA = "difference-pairwise-lora-proof-v1"
DATA_SCHEMA = "difference-pairwise-lora-data-v1"
PLAN = ROOT / ".omx/plans/lora-successor-v1.md"
TEST_SPEC = ROOT / ".omx/plans/lora-successor-test-spec-v1.md"
HANDOFF = ROOT / ".omx/plans/lora-successor-handoff-v1.json"
COMPACT_PLAN = ROOT / ".omx/plans/lora-successor-compact-kind-v1.md"
COMPACT_TEST_SPEC = ROOT / ".omx/plans/lora-successor-compact-kind-test-spec-v1.md"
COMPACT_HANDOFF = ROOT / ".omx/plans/lora-successor-compact-kind-handoff-v1.json"
COMPACT_INTENT = ROOT / ".omx/artifacts/difference-pairwise-lora-compact-kind-intent-v1.json"
PRECOMPACT_ARCHIVE = ROOT / ".omx/artifacts/difference-pairwise-lora-data-v1-precompact-archive/manifest.json"
COMPACT_PREFLIGHT = ROOT / ".omx/artifacts/difference-pairwise-compact-preflight-v1.json"
COMPACT_PREFLIGHT_PROOF = ROOT / ".omx/artifacts/difference-pairwise-compact-preflight-v1-proof.json"
COMPACT_BRIDGE = ROOT / ".omx/artifacts/difference-pairwise-lora-compact-kind-bridge-v1.json"
STRICT_FIT_BRIDGE = ROOT / ".omx/artifacts/difference-pairwise-lora-data-v1-fit-bridge.json"
POLICY = ROOT / "scripts/final_choice_policy.py"
STRICT_LOADER = ROOT / "scripts/train_final_choice_selector.py"
MODEL = ROOT / "data/external/qwen3-4b-mlx-4bit"
NATURAL = ROOT / "data/natural_context/final-choice-v1/natural_fit"
CHALLENGE = ROOT / "data/natural_context/final-choice-v1/challenge_fit"
TRAIN640 = ROOT / ".omx/artifacts/long-context-lora-data-v1-review2/diagnostic.jsonl"
OLD_DATA = ROOT / ".omx/artifacts/long-context-lora-data-v1-review2"

PLAN_SHA256 = "340a53f28d09b76598942898c6b8084e6e05550e79ba837e0f28c5585925e18d"
TEST_SPEC_SHA256 = "7dde72e45a80176a0a0f29447f9a87e0cdadfcdac946f07cff578d8f9b2b88e8"
HANDOFF_SHA256 = "785ef5ce678c684563a22f4c21c9dfcf9350a791256e0fe51de8b19e0da71092"
MODEL_REVISION = "52a5ab34fa604bc8af6d3ce0cac0cab10b7eb495"
OLD_DATA_MANIFEST_SHA256 = "cb4558249aa9acefb246abca9275ba8020f39cee27eb9d4143dfb94f266ecd06"
TRAIN640_SHA256 = "15f7c6414cc17e70b37da2466ec3e180f7c842c1f635a862465793fa04a096f6"
NATURAL_MANIFEST_SHA256 = "fc22b1635cc496ed70aa757750944726d583d640f2d9f3a598e380f3e262a1c9"
CHALLENGE_MANIFEST_SHA256 = "2b8a6349e0ad091e797d86694c2a5165b81fa906a91c3d540ece9042b1b66194"
MODEL_SOURCE_SHA256 = "67928c796b7e4562c072b2bbbdd0d52569e5ab9118ade1302f8d8d1bb4bae819"
MODEL_WEIGHTS_SHA256 = "e56a94b846a2dce7d1cc154ce58b9cfc4b8d99e8ffb74bfe256a265f4cbc7fb9"
CHAT_TEMPLATE_SHA256 = "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
COMPACT_PLAN_SHA256 = "0376b670313c015a7c13fa2f984c6abd1a3272e94f5208443a359d633a5784be"
COMPACT_TEST_SPEC_SHA256 = "44e1766a89375bd56e6836823dd6d045ada2c8276f81da3961922396bdce42da"
COMPACT_HANDOFF_SHA256 = "90f3b6382af349bf61b96862b658bb81bb87caaf7acb613db400c0737291f4c4"
COMPACT_INTENT_SHA256 = "e2009c3c323e523270dde8a6705ff3d754e950a2daea095d57cf0a5435457922"
PRECOMPACT_ARCHIVE_SHA256 = "823f1bc6ecd01cc4bb20a991de3d69c58b4fbff7b127893440fc00f83aacb07a"
COMPACT_PREFLIGHT_SHA256 = "f50a6e4ba3e0ad4883591c64dd2b36f31955bc9ba36b32487360c051ea7b1b7a"
COMPACT_PREFLIGHT_PROOF_SHA256 = "e09a70f528f23560f578a28755063f5c1c1f54425538abc2c210810b8d177123"
COMPACT_PREFLIGHT_VERIFIER_SHA256 = "1b8f7d12d3da55d8768272abc76a5f97544d09c42bc4696579313f08097d2cef"
STRICT_FIT_BRIDGE_SHA256 = "19b02d649cb707133fa72790259c9ebea2c37516cccd1e7c600a0e28191d5967"
SYSTEM_TEXT_SHA256 = "a31b624a44eb340970ac74ae9d3b768c8033ecc823d0039e35e37a839732cdc2"
PRECOMPACT_SOURCE_HASHES = {
    "scripts/prepare_difference_pairwise_lora_data.py":
        "c48c49a65cc547aee3ca35a4fcd27f346782c09151c126636b83ff7f978e43db",
    "scripts/verify_difference_pairwise_lora.py":
        "8bf1609c49a74f17916f19dfb599af87ee36af74217fb380b4f50c53c68d0068",
}
REPRESENTATION_VERSION = "shared-alpha-kind-edits-v1"
CODEBOOK = {"L": "lexicon", "X": "mixed", "M": "morphology",
            "N": "neural", "D": "numeric", "R": "raw", "S": "syllable"}
REVERSE_CODEBOOK = {value: key for key, value in CODEBOOK.items()}
SYSTEM_TEXT = (
    "로마자 원문과 문맥을 보고 CURRENT를 유지하면 1, ALTERNATIVE로 바꾸면 2를 답하세요. "
    "K.c는 구간별 kind 코드열(L=lexicon,X=mixed,M=morphology,N=neural,D=numeric,R=raw,S=syllable), "
    "K.a는 CURRENT와 다른 ALTERNATIVE의 0기준 index=code 목록입니다. 숫자 하나만 답하세요."
)
VERSIONS = {"mlx": "0.31.2", "mlx-lm": "0.31.3", "transformers": "5.12.1"}
ALLOWED_KINDS = frozenset({"lexicon", "morphology", "syllable", "neural"})
ARTIFACT_FILES = (
    "fit_inventory.jsonl", "pairs.jsonl", "units.jsonl", "schedule.jsonl",
    "prompts.jsonl", "tokens.jsonl", "targets.jsonl", "train640.jsonl",
)
EXPECTED_COUNTS = {
    "fit_rows": 3492, "repair_units": 59,
    "repair_rows_without_same_row_keep": 4,
    "component_references": 232, "repair_switch_references": 59,
    "natural_keep_references": 59, "challenge_keep_references": 59,
    "same_row_keep_references": 55, "updates": 15,
    "final_update_units": 3, "train640_rows": 640,
}


class VerificationError(ValueError):
    """Evidence is missing, malformed, hash-mismatched, or inconsistent."""


def need(condition, message):
    if not condition:
        raise VerificationError(message)


def canonical_bytes(value):
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise VerificationError("evidence is not canonical-JSON encodable") from exc


def digest_value(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_digest(path):
    value = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                value.update(chunk)
    except OSError as exc:
        raise VerificationError(f"required file is unavailable: {path}") from exc
    return value.hexdigest()


def path_key(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path):
    try:
        value = json.loads(Path(path).read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid JSON: {path}") from exc
    need(isinstance(value, dict), f"JSON object required: {path}")
    return value


def read_jsonl(path):
    rows = []
    try:
        raw = Path(path).read_bytes()
        for number, line in enumerate(raw.splitlines(), 1):
            if not line.strip():
                raise VerificationError(f"blank JSONL row: {path}:{number}")
            value = json.loads(line)
            need(isinstance(value, dict), f"JSONL object required: {path}:{number}")
            rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid JSONL: {path}") from exc
    return rows


def _finite_score(candidate):
    score = candidate.get("score")
    return type(score) in (int, float) and math.isfinite(score)


def _segments(source, candidate):
    segments = candidate.get("segments")
    if not isinstance(segments, list) or not segments:
        return None
    cursor, spans, output = 0, [], []
    for segment in segments:
        if not isinstance(segment, dict):
            return None
        start, end = segment.get("input_start"), segment.get("input_end")
        if (type(start) is not int or type(end) is not int or start != cursor
                or not start < end <= len(source)
                or segment.get("source") != source[start:end]
                or not isinstance(segment.get("text"), str)
                or not isinstance(segment.get("kind"), str)):
            return None
        spans.append((start, end, segment["source"]))
        output.append(segment["text"])
        cursor = end
    if cursor != len(source) or candidate.get("text") != "".join(output):
        return None
    return spans


def proposal_allowed(source, parent, alternative):
    """Independent copy of the frozen source-only proposal predicate."""
    left_spans, right_spans = _segments(source, parent), _segments(source, alternative)
    if left_spans is None or left_spans != right_spans:
        return False
    changed = False
    for left, right in zip(parent["segments"], alternative["segments"]):
        if left["text"] == right["text"]:
            continue
        changed = True
        if (left.get("kind") not in ALLOWED_KINDS
                or right.get("kind") not in ALLOWED_KINDS
                or re.fullmatch(r"[a-z]+", right["source"]) is None
                or re.fullmatch(r"[가-힣]+", right["text"]) is None
                or romanize_word(right["text"]) != right["source"]):
            return False
    return changed


def changed_spans(source, parent, alternative):
    need(_segments(source, parent) == _segments(source, alternative)
         and _segments(source, parent) is not None, "pair segment alignment differs")
    result = []
    for index, (left, right) in enumerate(zip(parent["segments"], alternative["segments"])):
        if left["text"] == right["text"]:
            continue
        start, end = left["input_start"], left["input_end"]
        result.append({
            "segment_index": index, "input_start": start, "input_end": end,
            "roman": source[start:end], "left_context": source[max(0, start - 12):start],
            "right_context": source[end:end + 12], "current_text": left["text"],
            "current_kind": left["kind"], "alternative_text": right["text"],
            "alternative_kind": right["kind"],
        })
    need(result, "pair has no changed span")
    return result


def _source_classes(value):
    classes = set()
    if re.search(r"[A-Za-z]", value):
        classes.add("latin")
    if re.search(r"[0-9]", value):
        classes.add("numeric")
    if re.search(r"[^A-Za-z0-9]", value):
        classes.add("symbol")
    return classes


def match_key(source, context, parent, alternative):
    spans = changed_spans(source, parent, alternative)
    changed_indices = {span["segment_index"] for span in spans}
    adjacent = set()
    for index in changed_indices:
        for neighbor in (index - 1, index + 1):
            if 0 <= neighbor < len(parent["segments"]) and neighbor not in changed_indices:
                adjacent.update(_source_classes(parent["segments"][neighbor]["source"]))
    neighbor_class = ("none" if not adjacent else next(iter(adjacent))
                      if len(adjacent) == 1 else "mixed")
    total = sum(span["input_end"] - span["input_start"] for span in spans)
    roman_length = ("1_2" if total <= 2 else "3_4" if total <= 4
                    else "5_8" if total <= 8 else "9_plus")
    return {
        "shape": "sentence" if re.search(r"\s", source) else "single",
        "context": "present" if context.strip() else "empty",
        "span_count": "one" if len(spans) == 1 else "multi",
        "roman_length": roman_length,
        "kind_signature": "+".join(
            f'{span["current_kind"]}>{span["alternative_kind"]}' for span in spans),
        "neighbor_class": neighbor_class,
    }


def pair_id(role, row_id, alternative_index):
    return digest_value({"role": role, "row_id": row_id,
                         "current_source_index": 0,
                         "alternative_source_index": alternative_index})


def unit_id(switch_pair_id):
    return hashlib.sha256(("lora-successor-v1-unit\0" + switch_pair_id).encode()).hexdigest()


def repair_order(identity):
    return hashlib.sha256(("lora-successor-v1-repair-order\0" + identity).encode()).hexdigest()


def control_order(unit_identity, pair_identity):
    raw = "lora-successor-v1-control\0" + unit_identity + "\0" + pair_identity
    return hashlib.sha256(raw.encode()).hexdigest()


DROP_ORDER = (None, "context", "roman_length", "neighbor_class", "shape", "kind_signature")


def matches_at_depth(repair, control, depth):
    need(type(depth) is int and 0 <= depth <= 5, "matching depth is outside 0..5")
    dropped = set(DROP_ORDER[1:depth + 1])
    return all(repair[name] == control[name] for name in repair if name not in dropped)


def allocate_control(unit_identity, repair_key, pool, use_counts):
    for reused in (False, True):
        candidates = [pair for pair in pool if (use_counts[pair["pair_id"]] > 0) is reused]
        for depth in range(6):
            available = [pair for pair in candidates
                         if matches_at_depth(repair_key, pair["matching_key"], depth)]
            if available:
                chosen = min(available, key=lambda pair: (
                    control_order(unit_identity, pair["pair_id"]), pair["pair_id"]))
                prior = use_counts[chosen["pair_id"]]
                use_counts[chosen["pair_id"]] += 1
                return chosen, {"relaxation_depth": depth, "prior_use_count": prior,
                    "reuse_after_unused_exhaustion": reused}
    raise VerificationError("control pool cannot match a repair at any depth")


def _load_fit_rows():
    result, hashes = {}, {}
    for role, directory in (("natural_fit", NATURAL), ("challenge_fit", CHALLENGE)):
        expected_manifest_sha = (NATURAL_MANIFEST_SHA256 if role == "natural_fit"
                                 else CHALLENGE_MANIFEST_SHA256)
        need(file_digest(directory / "manifest.json") == expected_manifest_sha,
             f"frozen FIT manifest pin differs: {role}")
        manifest = read_json(directory / "manifest.json")
        cases, selections = read_jsonl(directory / "cases.jsonl"), read_jsonl(directory / "selection.jsonl")
        need(len(cases) == len(selections), f"FIT case/selection count differs: {role}")
        rows = []
        for case, selection in zip(cases, selections):
            for field in ("id", "group_id", "input", "context", "expected"):
                need(case.get(field) == selection.get(field), f"FIT case/selection join differs: {role}/{field}")
            rows.append(case)
        need(manifest.get("completed") is True and manifest.get("role") == role
             and manifest.get("rows") == len(rows), f"FIT manifest differs: {role}")
        result[role] = rows
        for name in ("manifest.json", "cases.jsonl", "selection.jsonl"):
            path = directory / name
            hashes[path_key(path)] = file_digest(path)
    need(len(result["natural_fit"]) == 2048 and len(result["challenge_fit"]) == 1444,
         "frozen FIT row counts differ")
    return result, hashes


def external_context_window(context, source):
    need(isinstance(context, str), "external context must be a string")
    if len(context) <= 256:
        return {"text": context, "strategy": "complete", "original_chars": len(context)}
    found = context.find(source)
    if found >= 0:
        center = found + len(source) // 2
        start = max(0, min(len(context) - 256, center - 128))
        return {"text": context[start:start + 256], "strategy": "centered_source",
                "original_chars": len(context), "window_start": start}
    return {"head": context[:128], "tail": context[-128:],
            "strategy": "head_tail_gap", "original_chars": len(context),
            "source_chars": 256, "gap_marker": "[…중략…]"}


def semantic_payload(pair):
    return {
        "roman_input": pair["source"], "external_context": pair["external_context"],
        "raw_mode": pair["raw_mode"], "mapping": {"1": "KEEP", "2": "SWITCH"},
        "CURRENT": {"source_index": pair["current_source_index"], **pair["current"]},
        "ALTERNATIVE": {"source_index": pair["alternative_source_index"], **pair["alternative"]},
        "changed_spans": pair["changed_spans"],
    }


def encode_compact_payload(semantic):
    need(isinstance(semantic, dict) and "K" not in semantic,
         "semantic payload is malformed or collides with reserved K")
    compact = deepcopy(semantic)
    current, alternative = compact.get("CURRENT"), compact.get("ALTERNATIVE")
    need(isinstance(current, dict) and isinstance(alternative, dict),
         "CURRENT and ALTERNATIVE objects are required")
    left, right = current.pop("kinds", None), alternative.pop("kinds", None)
    need(isinstance(left, list) and left and isinstance(right, list) and len(left) == len(right),
         "kind arrays must be nonempty and have equal length")
    try:
        current_codes = [REVERSE_CODEBOOK[value] for value in left]
        alternative_codes = [REVERSE_CODEBOOK[value] for value in right]
    except (KeyError, TypeError) as exc:
        raise VerificationError("kind array contains an unknown value") from exc
    compact["K"] = {"a": ",".join(f"{index}={other}" for index, (base_code, other)
                                     in enumerate(zip(current_codes, alternative_codes))
                                     if base_code != other),
                    "c": "".join(current_codes)}
    return compact


def decode_compact_payload(compact):
    need(isinstance(compact, dict), "compact payload must be an object")
    semantic = deepcopy(compact)
    encoded = semantic.pop("K", None)
    current, alternative = semantic.get("CURRENT"), semantic.get("ALTERNATIVE")
    need(isinstance(encoded, dict) and set(encoded) == {"a", "c"},
         "K must contain exactly a and c")
    need(isinstance(current, dict) and isinstance(alternative, dict)
         and "kinds" not in current and "kinds" not in alternative,
         "compact candidates must omit full kind arrays")
    codes, edits = encoded["c"], encoded["a"]
    need(isinstance(codes, str) and codes and all(code in CODEBOOK for code in codes),
         "K.c must be a nonempty known uppercase code sequence")
    need(isinstance(edits, str), "K.a must be a string")
    alternative_codes, previous = list(codes), -1
    if edits:
        for edit in edits.split(","):
            match = re.fullmatch(r"(0|[1-9][0-9]*)=([LXMNDRS])", edit)
            need(match is not None, "K.a edit grammar differs")
            index, code = int(match.group(1)), match.group(2)
            need(previous < index < len(codes),
                 "K.a indices must be unique, increasing, and in range")
            need(codes[index] != code, "K.a contains a redundant edit")
            alternative_codes[index] = code
            previous = index
    current["kinds"] = [CODEBOOK[code] for code in codes]
    alternative["kinds"] = [CODEBOOK[code] for code in alternative_codes]
    need(encode_compact_payload(semantic) == compact, "compact payload is noncanonical")
    return semantic


def prompt_payload(pair):
    return encode_compact_payload(semantic_payload(pair))


def expected_messages(pair):
    return [{"role": "system", "content": SYSTEM_TEXT},
            {"role": "user", "content": canonical_bytes(prompt_payload(pair)).decode()}]


def _allowed_pairs(role_rows):
    pairs, repairs, accounting = {}, [], Counter()
    seen_rows = set()
    for role in ("natural_fit", "challenge_fit"):
        for row in role_rows[role]:
            source, candidates, expected = row.get("input"), row.get("candidates"), row.get("expected")
            need(isinstance(source, str) and isinstance(row.get("context"), str)
                 and isinstance(candidates, list) and candidates
                 and isinstance(expected, list) and expected
                 and all(isinstance(value, str) and value for value in expected)
                 and isinstance(row.get("id"), str) and row["id"]
                 and isinstance(row.get("group_id"), str) and row["group_id"]
                 and isinstance(row.get("raw_bypass_response"), dict),
                 "FIT row source fields are malformed")
            identity = (role, row["id"])
            need(identity not in seen_rows, "FIT row identity is duplicated")
            seen_rows.add(identity)
            for candidate in candidates:
                score = candidate.get("score")
                need((score is None or type(score) in (int, float) and math.isfinite(score))
                     and _segments(source, candidate) is not None,
                     "FIT candidate score/segment structure is malformed")
            parent_correct = candidates[0].get("text") in expected
            allowed = []
            for index, alternative in enumerate(candidates[1:], 1):
                if _finite_score(alternative) and proposal_allowed(source, candidates[0], alternative):
                    identity = pair_id(role, row["id"], index)
                    pair = {"pair_id": identity, "role": role, "row_id": row["id"],
                            "current_source_index": 0, "alternative_source_index": index,
                            "parent_accepted": parent_correct,
                            "target": "KEEP" if parent_correct or alternative.get("text") not in expected else "SWITCH",
                            "changes": changed_spans(source, candidates[0], alternative),
                            "matching_key": match_key(source, row["context"], candidates[0], alternative)}
                    pairs[identity] = pair
                    allowed.append(pair)
            accounting[(role, "rows")] += 1
            accounting[(role, "allowed_pairs")] += len(allowed)
            if not parent_correct:
                switches = [pair for pair in allowed if pair["target"] == "SWITCH"]
                if switches:
                    repairs.append((row, switches, [pair for pair in allowed if pair["target"] == "KEEP"]))
    return pairs, repairs, accounting


def _expected_units(all_pairs, repairs):
    need(len(repairs) == 59, "FIT repair count differs from 59")
    need(all(len(switches) == 1 for _, switches, _ in repairs),
         "each repair must have exactly one allowed finite SWITCH")
    need(Counter(bool(wrong) for _, _, wrong in repairs) == {True: 55, False: 4},
         "same-row wrong KEEP accounting differs")
    pools = _control_pools(all_pairs)
    uses = {role: Counter() for role in pools}
    units = []
    ordered = sorted(repairs, key=lambda item: (
        repair_order(unit_id(item[1][0]["pair_id"])), unit_id(item[1][0]["pair_id"])))
    for row, switches, same_row in ordered:
        switch = switches[0]
        identity = unit_id(switch["pair_id"])
        natural, natural_match = allocate_control(identity, switch["matching_key"], pools["natural_fit"], uses["natural_fit"])
        challenge, challenge_match = allocate_control(identity, switch["matching_key"], pools["challenge_fit"], uses["challenge_fit"])
        same = min(same_row, key=lambda pair: pair["alternative_source_index"]) if same_row else None
        refs = [switch["pair_id"], natural["pair_id"], challenge["pair_id"]]
        if same:
            refs.append(same["pair_id"])
        n_keep = len(refs) - 1
        units.append({"unit_id": identity, "switch_pair_id": switch["pair_id"],
            "references": [{"component": "repair_switch", "pair_id": refs[0],
                            "target": "SWITCH", "weight": 0.5}]
                + [{"component": component, "pair_id": value, "target": "KEEP",
                    "weight": 0.5 / n_keep,
                    "match": ({"same_row": True} if component == "same_row_keep" else
                              natural_match if component == "natural_keep" else challenge_match)}
                   for component, value in zip(
                       ["natural_keep", "challenge_keep"] + (["same_row_keep"] if same else []), refs[1:])]})
    return units


def _control_pools(all_pairs):
    return {role: [pair for pair in all_pairs.values()
                   if pair["role"] == role and pair["parent_accepted"]]
            for role in ("natural_fit", "challenge_fit")}


def _artifact_hashes(directory, manifest):
    declared = manifest.get("file_hashes")
    need(isinstance(declared, dict) and set(declared) == set(ARTIFACT_FILES),
         "manifest file_hashes must bind the exact stage-1 files")
    need(manifest.get("artifacts") == declared, "artifacts and file_hashes bindings differ")
    for name in ARTIFACT_FILES:
        need(re.fullmatch(r"[0-9a-f]{64}", declared[name] or "") is not None,
             f"invalid artifact digest: {name}")
        need(file_digest(directory / name) == declared[name], f"artifact digest differs: {name}")
    return declared


def _verify_source_hashes(manifest, expected_fit_hashes):
    before = manifest.get("source_hashes_before")
    after = manifest.get("source_hashes_after")
    need(isinstance(before, dict) and before == after, "source hashes before/after differ")
    required = dict(expected_fit_hashes)
    required.update({path_key(PLAN): PLAN_SHA256, path_key(TEST_SPEC): TEST_SPEC_SHA256,
                     path_key(HANDOFF): HANDOFF_SHA256, path_key(POLICY): file_digest(POLICY),
                     path_key(STRICT_LOADER): file_digest(STRICT_LOADER),
                     path_key(COMPACT_PLAN): COMPACT_PLAN_SHA256,
                     path_key(COMPACT_TEST_SPEC): COMPACT_TEST_SPEC_SHA256,
                     path_key(COMPACT_HANDOFF): COMPACT_HANDOFF_SHA256,
                     path_key(COMPACT_INTENT): COMPACT_INTENT_SHA256,
                     path_key(COMPACT_PREFLIGHT): COMPACT_PREFLIGHT_SHA256,
                     path_key(COMPACT_PREFLIGHT_PROOF): COMPACT_PREFLIGHT_PROOF_SHA256,
                     path_key(PRECOMPACT_ARCHIVE): PRECOMPACT_ARCHIVE_SHA256,
                     path_key(COMPACT_BRIDGE): file_digest(COMPACT_BRIDGE),
                     path_key(TRAIN640): TRAIN640_SHA256,
                     path_key(OLD_DATA / "manifest.json"): OLD_DATA_MANIFEST_SHA256,
                     path_key(OLD_DATA / "training_examples.jsonl"):
                         file_digest(OLD_DATA / "training_examples.jsonl"),
                     path_key(MODEL / "SOURCE.json"): MODEL_SOURCE_SHA256,
                     path_key(MODEL / "model.safetensors"): MODEL_WEIGHTS_SHA256})
    for name, expected in required.items():
        need(before.get(name) == expected, f"required source pin differs: {name}")
    for name, expected in before.items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        need(file_digest(path) == expected, f"source changed after construction: {name}")
    return before


def _by_id(rows, label):
    result = {}
    for row in rows:
        identity = row.get("pair_id") or row.get("unit_id") or row.get("id")
        need(isinstance(identity, str) and identity and identity not in result,
             f"duplicate or missing identity in {label}")
        result[identity] = row
    return result


def _field(record, *names):
    for name in names:
        if name in record:
            return record[name]
    raise VerificationError(f"missing required field: {'/'.join(names)}")


def _verify_inventory(rows, role_rows, old_eligible, all_pairs):
    expected = [(role, order, row) for role in ("natural_fit", "challenge_fit")
                for order, row in enumerate(role_rows[role])]
    need(len(rows) == len(expected) == 3492, "FIT inventory must contain all 3,492 rows")
    repairs = 0
    for record, (role, order, row) in zip(rows, expected):
        need(_field(record, "role") == role and _field(record, "row_id", "id") == row["id"],
             "FIT inventory order or identity differs")
        checks = {
            "source_sha256": hashlib.sha256(row["input"].encode()).hexdigest(),
            "context_sha256": hashlib.sha256(row["context"].encode()).hexdigest(),
            "expected_sha256": digest_value(row["expected"]),
            "candidate_objects_sha256": digest_value(row["candidates"]),
            "raw_response_sha256": digest_value(row["raw_bypass_response"]),
        }
        for name, expected_digest in checks.items():
            need(record.get(name) == expected_digest, f"FIT inventory snapshot differs: {role}/{row['id']}/{name}")
        row_pairs = [pair for pair in all_pairs.values()
                     if pair["role"] == role and pair["row_id"] == row["id"]]
        parent_accepted = row["candidates"][0]["text"] in row["expected"]
        switches = [pair for pair in row_pairs if pair["target"] == "SWITCH"]
        status = ("parent_accepted" if parent_accepted else "repair" if len(switches) == 1
                  else "parent_wrong_unrepairable")
        repairs += status == "repair"
        need(record.get("row_order") == order and record.get("status") == status
             and record.get("parent_accepted") is parent_accepted
             and record.get("allowed_pair_count") == len(row_pairs)
             and record.get("allowed_switch_count") == len(switches)
             and record.get("old_eligible") is ((role, row["id"]) in old_eligible),
             "FIT inventory classification differs")
        if status == "repair":
            need(role == "natural_fit" and (role, row["id"]) in old_eligible,
                 "repair row is outside the pinned old eligible natural set")
    need(repairs == 59, "FIT inventory repair count differs")


def _verify_pairs(pair_rows, expected_pairs, role_rows, expected_scopes):
    actual = _by_id(pair_rows, "pairs")
    need(set(actual) == set(expected_scopes), "stored pair set omits or adds a required pair")
    row_index = {(role, row["id"]): row for role in role_rows for row in role_rows[role]}
    for identity, record in actual.items():
        role, row_id = record.get("role"), _field(record, "row_id", "id")
        alt = record.get("alternative_source_index")
        need(identity == pair_id(role, row_id, alt), "pair identity differs")
        row = row_index.get((role, row_id))
        need(row is not None and type(alt) is int and 0 < alt < len(row["candidates"]),
             "pair source mapping differs")
        expected = expected_pairs.get(identity)
        need(expected is not None, "stored pair is not an allowed finite alternative")
        record["_expected_target"] = expected["target"]
        need(record.get("current_source_index") == 0, "current source index must be zero")
        need(record.get("changed_spans") == expected["changes"], "changed-span reconstruction differs")
        need(record.get("match_key") == expected["matching_key"], "pair matching signature differs")
        need(record.get("source") == row["input"] and record.get("group_id") == row["group_id"],
             "pair source/group snapshot differs")
        need(record.get("external_context") == external_context_window(row["context"], row["input"]),
             "external context window differs")
        candidates = row["candidates"]
        ranked = sorted((i for i, c in enumerate(candidates) if _finite_score(c)),
                        key=lambda i: (-candidates[i]["score"], i))
        current = {"text": candidates[0]["text"], "score": candidates[0]["score"],
                   "rank": ranked.index(0) + 1, "kinds": [x["kind"] for x in candidates[0]["segments"]]}
        alternative = {"text": candidates[alt]["text"], "score": candidates[alt]["score"],
                       "rank": ranked.index(alt) + 1, "kinds": [x["kind"] for x in candidates[alt]["segments"]]}
        need(record.get("current") == current and record.get("alternative") == alternative,
             "pair candidate summaries differ")
        need(record.get("candidate_count") == len(candidates)
             and record.get("current_candidate_sha256") == digest_value(candidates[0])
             and record.get("alternative_candidate_sha256") == digest_value(candidates[alt])
             and record.get("candidate_objects_sha256") == digest_value(candidates)
             and record.get("raw_response_sha256") == digest_value(row["raw_bypass_response"])
             and record.get("raw_mode") is False, "pair candidate/raw invariants differ")
        need(record.get("scopes") == sorted(expected_scopes[identity]), "pair scope binding differs")
    return actual


def _expected_scopes(all_pairs, expected_units, role_rows):
    scopes = defaultdict(set)
    training_ids = {component["pair_id"] for unit in expected_units
                    for component in unit["references"]}
    mechanism_rows = {(all_pairs[identity]["role"], all_pairs[identity]["row_id"])
                      for identity in training_ids}
    source = read_jsonl(TRAIN640)
    fit_index = {(role, row["id"]): row for role in role_rows for row in role_rows[role]}
    train640_rows = set()
    for item in source:
        identity = (item.get("role"), item.get("row", {}).get("id"))
        need(identity in fit_index and item.get("row") == fit_index[identity],
             "TRAIN640 source does not bind byte-equivalent FIT row data")
        train640_rows.add(identity)
    need(len(train640_rows) == 640, "TRAIN640 source identities are duplicated")
    for identity in training_ids:
        scopes[identity].add("training")
    for identity, pair in all_pairs.items():
        row_identity = (pair["role"], pair["row_id"])
        if row_identity in mechanism_rows:
            scopes[identity].add("mechanism")
        if row_identity in train640_rows:
            scopes[identity].add("train640")
    return dict(scopes), mechanism_rows, train640_rows


def _normalize_unit(record):
    components = record.get("references")
    need(isinstance(components, list), "unit components are missing")
    normalized = deepcopy(record)
    normalized["components"] = components
    return normalized


def _verify_same_row_match(value):
    need(value == {"same_row": True, "relaxation_depth": None,
                   "prior_use_count": 0, "reuse_after_unused_exhaustion": False},
         "same-row matching evidence differs")


def _verify_units(unit_rows, expected_units, actual_pairs, targets):
    need(len(unit_rows) == len(expected_units) == 59, "exactly 59 units are required")
    target_map = {(_field(row, "pair_id")): _field(row, "target", "action") for row in targets}
    need(len(target_map) == len(targets), "target identities are duplicated")
    need([row.get("unit_id") for row in unit_rows] == [row["unit_id"] for row in expected_units],
         "ordered unit identities differ or contain duplicates")
    expected_map = {row["unit_id"]: row for row in expected_units}
    references = Counter()
    for position, raw in enumerate(unit_rows):
        record = _normalize_unit(raw)
        expected = expected_map.get(record.get("unit_id"))
        need(expected is not None, "unit identity is not derived from a repair")
        need(record["unit_id"] == unit_id(record.get("switch_pair_id")), "unit identity differs")
        need(record.get("switch_pair_id") == expected["switch_pair_id"], "unit switch pair differs")
        need(record.get("order") == position + 1, "unit order differs")
        components = record["components"]
        need(len(components) == len(expected["references"]), "unit component count differs")
        total = 0.0
        for component, wanted in zip(components, expected["references"]):
            need(component.get("component") == wanted["component"]
                 and component.get("pair_id") == wanted["pair_id"]
                 and component.get("target") == wanted["target"],
                 "unit component order or target differs")
            weight = component.get("weight")
            need(type(weight) in (int, float) and math.isfinite(weight)
                 and math.isclose(weight, wanted["weight"], rel_tol=0, abs_tol=1e-15),
                 "unit component weight differs")
            need(component["pair_id"] in actual_pairs and target_map.get(component["pair_id"]) == component["target"],
                 "unit reference lacks a matching stored pair/target")
            if component["component"] == "same_row_keep":
                _verify_same_row_match(component.get("match"))
            elif component["component"] != "repair_switch":
                need(component.get("match") == wanted["match"], "unit matching evidence differs")
            total += weight
            references[component["target"]] += 1
        need(math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-15), "unit weights do not sum to one")
    need(references == {"SWITCH": 59, "KEEP": 173}, "component reference target counts differ")


def _verify_schedule(rows, expected_units):
    need(len(rows) == 59, "schedule must contain 59 units")
    for position, (record, unit) in enumerate(zip(rows, expected_units), 1):
        expected = {"unit_order": position, "unit_id": unit["unit_id"],
                    "update": (position - 1) // 4 + 1,
                    "position_in_update": (position - 1) % 4 + 1}
        for name, value in expected.items():
            need(record.get(name) == value, f"schedule differs at {position}/{name}")
    need(rows[-1]["update"] == 15 and rows[-1]["position_in_update"] == 3,
         "schedule must end with update 15 tail 3")


def _verify_prompt_and_tokens(pair_rows, prompt_rows, token_rows, target_rows, tokenizer, preflight):
    pairs = _by_id(pair_rows, "pairs")
    prompts, tokens, targets = (_by_id(prompt_rows, "prompts"), _by_id(token_rows, "tokens"),
                                _by_id(target_rows, "targets"))
    need(set(prompts) == set(tokens) == set(targets) == set(pairs),
         "pair/prompt/token/target identity sets differ")
    observations = _by_id(preflight.get("observations", []), "preflight observations")
    need(set(observations) == set(pairs)
         and preflight.get("ordered_pair_ids_sha256") == digest_value(sorted(pairs)),
         "preflight/data pair identity set differs")
    lengths = []
    semantic_digests, compact_digests, decoded_digests, token_digests = [], [], [], []
    for identity in sorted(pairs):
        prompt, token, target = prompts[identity], tokens[identity], targets[identity]
        messages = prompt.get("messages")
        semantic = semantic_payload(pairs[identity])
        expected_payload = encode_compact_payload(semantic)
        decoded = decode_compact_payload(expected_payload)
        need(canonical_bytes(decoded) == canonical_bytes(semantic),
             "compact payload does not round-trip to semantic bytes")
        need(messages == expected_messages(pairs[identity]),
             "prompt messages differ from the source-only whitelist")
        semantic_sha, compact_sha, decoded_sha = (
            digest_value(semantic), digest_value(expected_payload), digest_value(decoded))
        need(prompt.get("semantic_payload_sha256") == semantic_sha
             and prompt.get("compact_payload_sha256") == compact_sha
             and prompt.get("decoded_payload_sha256") == decoded_sha
             and prompt.get("payload_sha256") == compact_sha,
             "prompt semantic, compact, decoded, or payload digest differs")
        action = _field(target, "target", "action")
        need(action == pairs[identity]["_expected_target"]
             and target.get("target_digit") == (1 if action == "KEEP" else 2),
             "target action or digit differs")
        actual = tokenizer.apply_chat_template(messages, tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_dict=False)
        need(isinstance(actual, list) and actual and all(type(value) is int for value in actual),
             "tokenizer returned malformed prompt tokens")
        stored = _field(token, "tokens", "token_ids")
        need(stored == actual and token.get("prompt_tokens") == len(stored)
             and token.get("tokens_sha256") == digest_value(stored),
             "stored token sequence or digest differs")
        rendered = tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        need(prompt.get("rendered_prefix_sha256") == hashlib.sha256(rendered.encode()).hexdigest(),
             "rendered generation prefix digest differs")
        need(tokenizer.encode(rendered, add_special_tokens=False) == stored,
             "rendered generation prefix tokenization differs")
        digit_ids = []
        for digit in ("1", "2"):
            with_answer = tokenizer.encode(rendered + digit, add_special_tokens=False)
            need(len(with_answer) == len(stored) + 1 and with_answer[:-1] == stored,
                 "answer digit is not one continuation token at the actual boundary")
            digit_ids.append(with_answer[-1])
        need(len(set(digit_ids)) == 2
             and token.get("digit_continuation_ids") == {"1": digit_ids[0], "2": digit_ids[1]},
             "answer digit IDs differ or collide")
        need(len(stored) <= 1024, "real pair prompt exceeds 1,024 tokens")
        token_sha = digest_value(stored)
        need(observations[identity] == {
            "pair_id": identity, "semantic_sha256": semantic_sha,
            "compact_sha256": compact_sha, "decoded_sha256": decoded_sha,
            "tokens_sha256": token_sha, "token_count": len(stored),
            "digit_continuation_ids": {"1": digit_ids[0], "2": digit_ids[1]}},
            f"data prompt differs from successful compact preflight: {identity}")
        lengths.append(len(stored)); token_digests.append(token_sha)
        semantic_digests.append(semantic_sha); compact_digests.append(compact_sha)
        decoded_digests.append(decoded_sha)
    ordered = sorted(lengths)
    stats = {"min": ordered[0], "median": statistics.median(ordered),
             "p95": ordered[math.ceil(0.95 * len(ordered)) - 1], "max": ordered[-1],
             "count": len(ordered), "ceiling": 1024,
             "overflow": sum(length > 1024 for length in lengths)}
    payload_digests = {
        "ordered_semantic_sha256": digest_value(semantic_digests),
        "ordered_compact_sha256": digest_value(compact_digests),
        "ordered_decoded_sha256": digest_value(decoded_digests),
        "ordered_token_sha256": digest_value(token_digests),
    }
    return stats, payload_digests


def _verify_train640(rows):
    source = read_jsonl(TRAIN640)
    need(len(source) == len(rows) == 640, "TRAIN640 binding must contain all rows")
    total = 0
    identities = set()
    for stored, item in zip(rows, source):
        row, role = item.get("row"), item.get("role")
        need(isinstance(row, dict) and stored.get("role") == role
             and _field(stored, "row_id", "id") == row.get("id"), "TRAIN640 identity/order differs")
        need(stored.get("stratum") == item.get("stratum")
             and stored.get("candidate_objects_sha256") == digest_value(row["candidates"])
             and stored.get("raw_response_sha256") == digest_value(row["raw_bypass_response"]),
             "TRAIN640 row binding differs")
        identities.add((role, row["id"]))
        pair_ids = []
        for index, alternative in enumerate(row["candidates"][1:], 1):
            if _finite_score(alternative) and proposal_allowed(row["input"], row["candidates"][0], alternative):
                total += 1
                pair_ids.append(pair_id(role, row["id"], index))
        need(stored.get("pair_ids") == pair_ids, "TRAIN640 permitted pair list differs")
        need(stored.get("fallback_reason") == (None if pair_ids else "no_permitted_non_parent_alternative"),
             "TRAIN640 fallback reason differs")
    need(len(identities) == 640, "TRAIN640 identities are duplicated")
    return total, identities


def _expected_compact_lineage(preflight, proof, archive, bridge_sha):
    return {
        "intent": {"path": path_key(COMPACT_INTENT), "sha256": COMPACT_INTENT_SHA256},
        "archive_manifest": {"path": path_key(PRECOMPACT_ARCHIVE),
                             "sha256": PRECOMPACT_ARCHIVE_SHA256,
                             "entries": len(archive["files"])},
        "preflight": {"path": path_key(COMPACT_PREFLIGHT),
                      "sha256": COMPACT_PREFLIGHT_SHA256},
        "preflight_proof": {"path": path_key(COMPACT_PREFLIGHT_PROOF),
                            "sha256": COMPACT_PREFLIGHT_PROOF_SHA256,
                            "digest": proof["digest"]},
        "bridge": {"path": path_key(COMPACT_BRIDGE), "sha256": bridge_sha,
                   "schema": "difference-pairwise-lora-compact-kind-bridge-v1"},
    }


def _verify_lineage_records(manifest, bridge, expected, preflight):
    need(manifest.get("compact_lineage") == expected,
         "manifest compact lineage differs")
    need(bridge.get("intent") == expected["intent"]
         and bridge.get("preflight") == {
             **expected["preflight"],
             "ordered_pair_ids_sha256": preflight["ordered_pair_ids_sha256"],
             "ordered_token_digest": preflight["ordered_token_digest"]}
         and bridge.get("preflight_proof") == expected["preflight_proof"]
         and bridge.get("archive_manifest") == expected["archive_manifest"],
         "compact bridge lineage differs")


def _verify_payload_digest_summary(manifest, actual):
    need(manifest.get("payload_digests") == actual,
         "manifest ordered payload digests differ")


def _verify_identity_digests(manifest, pair_rows, unit_rows):
    expected = {
        "ordered_pair_ids_sha256": digest_value(sorted(row.get("pair_id") for row in pair_rows)),
        "produced_units_sha256": digest_value(unit_rows),
    }
    need(manifest.get("identity_digests") == expected,
         "manifest identity digests differ")
    return expected


def _verify_bridge_torch_provenance(bridge):
    need(file_digest(STRICT_FIT_BRIDGE) == STRICT_FIT_BRIDGE_SHA256,
         "pinned strict FIT bridge differs")
    predecessor = read_json(STRICT_FIT_BRIDGE)
    need(predecessor.get("schema") == "difference-pairwise-lora-strict-fit-bridge-v1"
         and predecessor.get("completed") is True
         and predecessor.get("input_hashes_before") == predecessor.get("input_hashes_after")
         and predecessor.get("strict_loader") ==
             "scripts.train_final_choice_selector.load_fit_role"
         and predecessor.get("strict_pair_validator") ==
             "scripts.train_final_choice_selector.validate_fit_pair"
         and predecessor.get("torch_environment_required") is True
         and isinstance(predecessor.get("torch_version"), str)
         and predecessor["torch_version"]
         and bridge.get("torch_environment_required") is True
         and bridge.get("torch_version") == predecessor["torch_version"],
         "compact bridge Torch provenance differs from pinned strict FIT capture")


def _verify_compact_authority(manifest, role_rows):
    pinned = {
        COMPACT_PLAN: COMPACT_PLAN_SHA256,
        COMPACT_TEST_SPEC: COMPACT_TEST_SPEC_SHA256,
        COMPACT_HANDOFF: COMPACT_HANDOFF_SHA256,
        COMPACT_INTENT: COMPACT_INTENT_SHA256,
        PRECOMPACT_ARCHIVE: PRECOMPACT_ARCHIVE_SHA256,
        COMPACT_PREFLIGHT: COMPACT_PREFLIGHT_SHA256,
        COMPACT_PREFLIGHT_PROOF: COMPACT_PREFLIGHT_PROOF_SHA256,
    }
    for path, expected in pinned.items():
        need(file_digest(path) == expected,
             f"compact authority pin differs: {path_key(path)}")

    intent, archive = read_json(COMPACT_INTENT), read_json(PRECOMPACT_ARCHIVE)
    need(intent.get("schema") == "difference-pairwise-lora-compact-kind-intent-v1"
         and intent.get("representation_version") == REPRESENTATION_VERSION
         and intent.get("codebook") == CODEBOOK
         and intent.get("system_text") == SYSTEM_TEXT
         and intent.get("system_text_sha256") == SYSTEM_TEXT_SHA256
         and intent.get("versions") == VERSIONS,
         "compact intent contract differs")
    for name, expected in intent.get("files", {}).items():
        need(file_digest(ROOT / name) == expected, f"compact intent file differs: {name}")
    need(archive.get("schema") == "pairwise-precompact-archive-v1"
         and archive.get("immutable") is True and isinstance(archive.get("files"), list),
         "precompact archive manifest differs")
    archive_by_source = {}
    for item in archive["files"]:
        need(isinstance(item, dict) and isinstance(item.get("source"), str)
             and item["source"] not in archive_by_source,
             "precompact archive source is missing or duplicated")
        archived = ROOT / item.get("archive", "")
        need(file_digest(archived) == item.get("sha256")
             and archived.stat().st_size == item.get("bytes"),
             f"precompact archive entry differs: {item.get('archive')}")
        archive_by_source[item["source"]] = item
    for name, expected in PRECOMPACT_SOURCE_HASHES.items():
        need(archive_by_source.get(name, {}).get("sha256") == expected,
             f"precompact successor archive differs: {name}")

    preflight = read_json(COMPACT_PREFLIGHT)
    expected_counts = {"pairs": 1928, "units": 59, "references": 232, "updates": 15,
        "final_update_units": 3, "mechanism_rows": 175,
        "E_mechanism": 481, "E_train640": 1556}
    expected_stats = {"min": 270, "median": 576.5, "p95": 811, "max": 986,
                      "count": 1928, "ceiling": 1024, "overflow": 0}
    need(preflight.get("schema") == "compact-kind-preflight-v1"
         and preflight.get("representation_version") == REPRESENTATION_VERSION
         and preflight.get("intent") == {"path": path_key(COMPACT_INTENT),
                                            "sha256": COMPACT_INTENT_SHA256}
         and preflight.get("archive_manifest") == {"path": path_key(PRECOMPACT_ARCHIVE),
                                                       "sha256": PRECOMPACT_ARCHIVE_SHA256}
         and preflight.get("codebook") == CODEBOOK
         and preflight.get("system_text") == SYSTEM_TEXT
         and preflight.get("versions") == VERSIONS
         and preflight.get("counts") == expected_counts
         and preflight.get("statistics") == expected_stats
         and preflight.get("round_trips") == 1928
         and len(preflight.get("observations", [])) == 1928
         and preflight.get("representation_passed") is True
         and preflight.get("model_weights_loaded") is False
         and preflight.get("data_qualified") is False,
         "successful compact preflight contract differs")
    before = preflight.get("source_hashes_before")
    need(isinstance(before, dict) and before == preflight.get("source_hashes_after"),
         "compact preflight source snapshots differ")
    for name, expected in before.items():
        if name in PRECOMPACT_SOURCE_HASHES:
            need(expected == PRECOMPACT_SOURCE_HASHES[name]
                 and archive_by_source.get(name, {}).get("sha256") == expected,
                 f"preflight predecessor source is not archived: {name}")
        else:
            source_path = Path(name) if Path(name).is_absolute() else ROOT / name
            need(file_digest(source_path) == expected,
                 f"compact preflight source changed: {name}")

    proof = read_json(COMPACT_PREFLIGHT_PROOF)
    proof_payload = proof.get("payload")
    need(proof.get("schema") == "compact-kind-preflight-proof-v1"
         and isinstance(proof_payload, dict)
         and proof.get("digest") == digest_value(proof_payload)
         and proof_payload.get("stage") == "compact_kind_preflight"
         and proof_payload.get("passed") is True
         and proof_payload.get("preflight_sha256") == COMPACT_PREFLIGHT_SHA256
         and proof_payload.get("counts") == expected_counts
         and proof_payload.get("statistics") == expected_stats
         and proof_payload.get("ordered_pair_ids_sha256") == preflight.get("ordered_pair_ids_sha256")
         and proof_payload.get("units_sha256") == preflight.get("units_sha256")
         and proof_payload.get("ordered_token_digest") == preflight.get("ordered_token_digest")
         and proof_payload.get("verifier_source") == {
             "path": "scripts/verify_difference_pairwise_compact_preflight.py",
             "sha256": COMPACT_PREFLIGHT_VERIFIER_SHA256}
         and proof_payload.get("model_weights_loaded") is False
         and proof_payload.get("data_qualified") is False,
         "independent compact preflight proof differs")
    need(file_digest(ROOT / proof_payload["verifier_source"]["path"])
         == COMPACT_PREFLIGHT_VERIFIER_SHA256,
         "compact preflight verifier source changed")

    need(COMPACT_BRIDGE.is_file(), "mandatory compact bridge is missing")
    bridge = read_json(COMPACT_BRIDGE)
    bridge_sha = file_digest(COMPACT_BRIDGE)
    need(bridge.get("schema") == "difference-pairwise-lora-compact-kind-bridge-v1"
         and bridge.get("completed") is True
         and bridge.get("representation_version") == REPRESENTATION_VERSION
         and bridge.get("input_hashes_before") == bridge.get("input_hashes_after"),
         "completed compact-kind bridge differs")
    implementation_sources = {
        "scripts/prepare_difference_pairwise_lora_data.py":
            file_digest(ROOT / "scripts/prepare_difference_pairwise_lora_data.py"),
        "scripts/verify_difference_pairwise_lora.py": file_digest(Path(__file__)),
    }
    need(bridge.get("implementation_sources") == implementation_sources,
         "compact bridge implementation source binding differs")
    bridge_inputs = bridge.get("input_hashes_before")
    need(isinstance(bridge_inputs, dict)
         and all(bridge_inputs.get(name) == value
                 for name, value in implementation_sources.items())
         and bridge.get("strict_loader") == "scripts.train_final_choice_selector.load_fit_role"
         and bridge.get("strict_pair_validator") ==
             "scripts.train_final_choice_selector.validate_fit_pair"
         and bridge.get("torch_environment_required") is True,
         "compact bridge loader or input contract differs")
    _verify_bridge_torch_provenance(bridge)
    for name, expected in bridge_inputs.items():
        source_path = Path(name) if Path(name).is_absolute() else ROOT / name
        need(file_digest(source_path) == expected, f"compact bridge input changed: {name}")
    expected_lineage = _expected_compact_lineage(preflight, proof, archive, bridge_sha)
    _verify_lineage_records(manifest, bridge, expected_lineage, preflight)
    for role, directory in (("natural_fit", NATURAL), ("challenge_fit", CHALLENGE)):
        expected = {"rows": len(role_rows[role]),
            "ordered_ids_sha256": digest_value([row["id"] for row in role_rows[role]]),
            "manifest_sha256": file_digest(directory / "manifest.json"),
            "cases_sha256": file_digest(directory / "cases.jsonl"),
            "selection_sha256": file_digest(directory / "selection.jsonl")}
        need(bridge.get("captures", {}).get(role) == expected,
             f"compact bridge FIT capture differs: {role}")
    return preflight, proof, expected_lineage


def _verify_manifest_bindings(manifest, fit_hashes, role_rows):
    fit_sources = manifest.get("fit_sources")
    need(isinstance(fit_sources, dict), "FIT source metadata is missing")
    for role, directory in (("natural_fit", NATURAL), ("challenge_fit", CHALLENGE)):
        item = fit_sources.get(role)
        need(isinstance(item, dict) and item.get("directory") == path_key(directory)
             and item.get("rows") == len(role_rows[role]), f"FIT source metadata differs: {role}")
        for field, name in (("manifest", "manifest.json"), ("cases", "cases.jsonl"),
                            ("selection", "selection.jsonl")):
            path = directory / name
            need(item.get(field) == {"path": path_key(path), "sha256": fit_hashes[path_key(path)]},
                 f"FIT source file metadata differs: {role}/{name}")
    need(manifest.get("train640_source") == {"path": path_key(TRAIN640),
          "sha256": TRAIN640_SHA256, "rows": 640}, "TRAIN640 source metadata differs")

    pins = manifest.get("pins")
    need(isinstance(pins, dict), "manifest pins are missing")
    expected_pins = {
        "plan_sha256": PLAN_SHA256, "test_spec_sha256": TEST_SPEC_SHA256,
        "handoff_sha256": HANDOFF_SHA256, "policy_sha256": file_digest(POLICY),
        "compact_plan_sha256": COMPACT_PLAN_SHA256,
        "compact_test_spec_sha256": COMPACT_TEST_SPEC_SHA256,
        "compact_handoff_sha256": COMPACT_HANDOFF_SHA256,
        "compact_intent_sha256": COMPACT_INTENT_SHA256,
        "compact_preflight_sha256": COMPACT_PREFLIGHT_SHA256,
        "compact_preflight_proof_sha256": COMPACT_PREFLIGHT_PROOF_SHA256,
        "precompact_archive_sha256": PRECOMPACT_ARCHIVE_SHA256,
        "strict_loader_sha256": file_digest(STRICT_LOADER),
        "model_source_sha256": MODEL_SOURCE_SHA256,
        "model_weights_sha256": MODEL_WEIGHTS_SHA256,
        "chat_template_sha256": CHAT_TEMPLATE_SHA256,
        "tokenizer_config_sha256": file_digest(MODEL / "tokenizer_config.json"),
        "package_versions": VERSIONS,
    }
    for name, value in expected_pins.items():
        need(pins.get(name) == value, f"manifest pin differs: {name}")
    producer = ROOT / "scripts/prepare_difference_pairwise_lora_data.py"
    need(pins.get("producer_sha256") == file_digest(producer), "producer source pin differs")
    need(manifest.get("old_data_manifest_sha256") == OLD_DATA_MANIFEST_SHA256
         and manifest.get("train640_diagnostic_sha256") == TRAIN640_SHA256,
         "old data/TRAIN640 fixed pins differ")
    old_examples = OLD_DATA / "training_examples.jsonl"
    need(manifest.get("old_eligible_examples_sha256") == file_digest(old_examples),
         "old 2,771-row eligible membership file differs")

    tokenizer = manifest.get("tokenizer")
    need(isinstance(tokenizer, dict) and tokenizer.get("model_revision") == MODEL_REVISION
         and tokenizer.get("versions") == VERSIONS
         and tokenizer.get("chat_template_sha256") == CHAT_TEMPLATE_SHA256
         and tokenizer.get("tokenizer_config_sha256") == expected_pins["tokenizer_config_sha256"]
         and tokenizer.get("padding") is False and tokenizer.get("truncation") is False
         and tokenizer.get("model_weights_loaded") is False,
         "tokenizer/model declaration differs")
    model_source = read_json(MODEL / "SOURCE.json")
    need(model_source.get("revision") == MODEL_REVISION, "model SOURCE revision differs")

    bridge_meta = manifest.get("strict_bridge")
    need(isinstance(bridge_meta, dict) and isinstance(bridge_meta.get("path"), str)
         and bridge_meta["path"], "strict FIT bridge metadata is missing")
    bridge_path = ROOT / bridge_meta["path"]
    need(bridge_path.resolve() == COMPACT_BRIDGE.resolve()
         and bridge_meta.get("schema") == "difference-pairwise-lora-compact-kind-bridge-v1"
         and bridge_meta.get("sha256") == manifest.get("strict_fit_bridge_sha256")
         and file_digest(bridge_path) == bridge_meta.get("sha256"), "strict FIT bridge binding differs")
    return tokenizer


def verify_data(manifest_path, tokenizer_loader=None):
    manifest_path = Path(manifest_path).resolve()
    directory = manifest_path.parent
    manifest = read_json(manifest_path)
    need(manifest.get("schema") == DATA_SCHEMA and manifest.get("completed") is True
         and manifest.get("representation_version") == REPRESENTATION_VERSION,
         "completed difference-pairwise-lora-data-v1 manifest is required")
    handoff = read_json(HANDOFF)
    need(handoff.get("ralplan_execution_handoff", {}).get("authorized") is True
         and handoff.get("terminal_planning_state") == "complete",
         "approved Ralplan handoff is required")
    need(file_digest(PLAN) == PLAN_SHA256 and file_digest(TEST_SPEC) == TEST_SPEC_SHA256
         and file_digest(HANDOFF) == HANDOFF_SHA256, "approved planning pins differ")
    file_hashes = _artifact_hashes(directory, manifest)
    role_rows, fit_hashes = _load_fit_rows()
    compact_preflight, compact_proof, compact_lineage = _verify_compact_authority(
        manifest, role_rows)
    source_hashes = _verify_source_hashes(manifest, fit_hashes)
    tokenizer_manifest = _verify_manifest_bindings(manifest, fit_hashes, role_rows)
    old_examples = read_jsonl(OLD_DATA / "training_examples.jsonl")
    old_eligible = {(row.get("role"), row.get("id")) for row in old_examples}
    need(len(old_examples) == len(old_eligible) == 2771,
         "old eligible FIT membership must contain exactly 2,771 identities")
    all_pairs, repairs, accounting = _allowed_pairs(role_rows)
    expected_units = _expected_units(all_pairs, repairs)
    expected_scopes, mechanism_ids, train640_source_ids = _expected_scopes(
        all_pairs, expected_units, role_rows)

    inventory = read_jsonl(directory / "fit_inventory.jsonl")
    pair_rows = read_jsonl(directory / "pairs.jsonl")
    unit_rows = read_jsonl(directory / "units.jsonl")
    schedule = read_jsonl(directory / "schedule.jsonl")
    prompts = read_jsonl(directory / "prompts.jsonl")
    tokens = read_jsonl(directory / "tokens.jsonl")
    targets = read_jsonl(directory / "targets.jsonl")
    train640 = read_jsonl(directory / "train640.jsonl")
    identity_digests = _verify_identity_digests(manifest, pair_rows, unit_rows)
    _verify_inventory(inventory, role_rows, old_eligible, all_pairs)
    fit_accounting = dict(Counter(
        f'{row["role"]}:{row["status"]}' for row in inventory))
    need(manifest.get("fit_accounting") == dict(sorted(fit_accounting.items())),
         "FIT accounting summary differs")
    actual_pairs = _verify_pairs(pair_rows, all_pairs, role_rows, expected_scopes)
    _verify_units(unit_rows, expected_units, actual_pairs, targets)
    _verify_schedule(schedule, expected_units)
    e_train640, train640_ids = _verify_train640(train640)
    need(train640_ids == train640_source_ids, "TRAIN640 stored/source identity sets differ")

    e_mechanism = sum("mechanism" in scopes for scopes in expected_scopes.values())
    if tokenizer_loader is None:
        from transformers import AutoTokenizer
        tokenizer_loader = lambda: AutoTokenizer.from_pretrained(
            MODEL, local_files_only=True, trust_remote_code=False)
    versions = {name: importlib.metadata.version(name) for name in VERSIONS}
    need(versions == VERSIONS,
         "pinned tokenizer package versions differ")
    tokenizer = tokenizer_loader()
    need(hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() == CHAT_TEMPLATE_SHA256,
         "actual tokenizer chat template differs")
    token_stats, payload_digests = _verify_prompt_and_tokens(
        pair_rows, prompts, tokens, targets, tokenizer, compact_preflight)
    need(token_stats == compact_preflight.get("statistics")
         and payload_digests["ordered_token_sha256"] == compact_preflight.get("ordered_token_digest"),
         "aggregate tokens differ from successful compact preflight")
    _verify_payload_digest_summary(manifest, payload_digests)

    counts = deepcopy(EXPECTED_COUNTS)
    training_pairs = {component["pair_id"] for unit in expected_units
                      for component in unit["references"]}
    counts.update({"unique_training_pairs": len(training_pairs),
                   "required_unique_pairs": len(actual_pairs),
                   "mechanism_distinct_rows": len(mechanism_ids),
                   "E_mechanism": e_mechanism, "E_train640": e_train640})
    declared_counts = manifest.get("counts")
    need(isinstance(declared_counts, dict), "manifest counts are missing")
    for name, value in counts.items():
        need(declared_counts.get(name) == value, f"manifest count differs: {name}")
    matching = manifest.get("matching")
    natural_refs = [ref["pair_id"] for unit in expected_units for ref in unit["references"]
                    if ref["component"] == "natural_keep"]
    challenge_refs = [ref["pair_id"] for unit in expected_units for ref in unit["references"]
                      if ref["component"] == "challenge_keep"]
    expected_matching = {
        "fields": ["shape", "context", "span_count", "roman_length",
                   "kind_signature", "neighbor_class"],
        "cumulative_relaxation_drop_order": ["context", "roman_length",
            "neighbor_class", "shape", "kind_signature"],
        "unused_before_reuse": True,
        "natural_unique_controls": len(set(natural_refs)),
        "challenge_unique_controls": len(set(challenge_refs)),
        "natural_reused_references": len(natural_refs) - len(set(natural_refs)),
        "challenge_reused_references": len(challenge_refs) - len(set(challenge_refs)),
    }
    need(matching == expected_matching, "matching/exhaustion summary differs")
    prompt_contract = manifest.get("prompt_contract")
    need(prompt_contract == {
        "system_text": SYSTEM_TEXT, "system_text_sha256": SYSTEM_TEXT_SHA256,
        "canonical_payload": True,
        "semantic_payload_fields": ["ALTERNATIVE", "CURRENT", "changed_spans",
                                    "external_context", "mapping", "raw_mode", "roman_input"],
        "compact_payload_fields": ["ALTERNATIVE", "CURRENT", "K", "changed_spans",
                                   "external_context", "mapping", "raw_mode", "roman_input"],
        "kind_codec": {"representation_version": REPRESENTATION_VERSION,
                       "codebook": CODEBOOK, "current_codes_field": "K.c",
                       "alternative_edits_field": "K.a",
                       "alternative_edit_grammar":
                           "index=code comma-separated; zero-based strictly increasing"},
        "codebook": CODEBOOK, "compact_kind_field": "K",
        "current_codes_field": "c", "alternative_edits_field": "a",
        "decoded_semantics_byte_equal": True,
        "external_context_max_source_chars": 256,
        "external_context_gap_marker_is_metadata": True,
        "local_source_context_each_side_chars": 12,
        "mapping": {"1": "KEEP", "2": "SWITCH"},
        "enable_thinking": False, "add_generation_prompt": True,
        "answer_appended": False, "target_fields_separate": True,
    }, "prompt contract differs")
    need(tokenizer_manifest.get("statistics") == token_stats,
         "actual tokenizer statistics differ")
    prefix_suffixes = {tuple(row["tokens"][-4:]) for row in tokens}
    digit_maps = {canonical_bytes(row["digit_continuation_ids"]) for row in tokens}
    need(len(prefix_suffixes) == 1
         and tokenizer_manifest.get("generation_prefix_token_suffix") == list(next(iter(prefix_suffixes)))
         and len(digit_maps) == 1
         and tokenizer_manifest.get("digit_continuation_ids") == json.loads(next(iter(digit_maps))),
         "tokenizer boundary summary differs")
    invariants = manifest.get("invariants")
    need(invariants == {"candidate_objects_preserved_by_sha256": True,
                        "raw_responses_preserved_by_sha256": True,
                        "all_fit_rows_accounted": True,
                        "diagnostic_rows_bound_to_fit_bytes": True,
                        "fresh_dev_final_targets_read": False,
                        "production_modified": False}
         and manifest.get("training_run") is False
         and manifest.get("resource_probe_run") is False
         and manifest.get("deployed") is False, "stage-1 boundary declaration differs")

    return {
        "stage": "data", "passed": True, "data_manifest_sha256": file_digest(manifest_path),
        "file_hashes": file_hashes, "source_hashes_sha256": digest_value(source_hashes),
        "counts": counts, "E_mechanism": e_mechanism, "E_train640": e_train640,
        "representation_version": REPRESENTATION_VERSION,
        "compact_lineage": compact_lineage,
        "compact_preflight_proof_digest": compact_proof["digest"],
        "identity_digests": identity_digests,
        "token_stats": token_stats, "payload_digests": payload_digests,
        "verifier_source": {"path": path_key(Path(__file__)),
                            "sha256": file_digest(Path(__file__))},
        "fit_allowed_pair_accounting": {f"{role}:{name}": value
            for (role, name), value in sorted(accounting.items())},
        "model_weights_loaded": False, "fresh_dev_final_targets_read": False,
    }


def write_proof(output, payload):
    output = Path(output)
    need(not output.exists(), "proof output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"schema": SCHEMA, "payload": payload, "digest": digest_value(payload)}
    output.write_bytes(canonical_bytes(envelope) + b"\n")
    return envelope


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resource", type=Path)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--evaluation", type=Path)
    args = parser.parse_args(argv)
    try:
        need(not any((args.resource, args.train, args.evaluation)),
             "resource/train/evaluation verification is not implemented in stage 1")
        payload = verify_data(args.data)
        write_proof(args.output, payload)
    except VerificationError as exc:
        print(json.dumps({"schema": SCHEMA, "passed": False, "error": str(exc)},
                         ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps({"schema": SCHEMA, "passed": True, "output": str(args.output)},
                     ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
