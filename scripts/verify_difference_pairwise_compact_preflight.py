"""Independently verify the compact-kind pairwise prompt preflight.

This verifier reuses only the already independent Stage-1 source/pair/unit
reconstruction.  It implements the compact codec, prompt construction, and
token replay locally and never imports the preflight producer.
"""
from __future__ import annotations

import argparse
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

from scripts import verify_difference_pairwise_lora as base


REPORT_SCHEMA = "compact-kind-preflight-v1"
PROOF_SCHEMA = "compact-kind-preflight-proof-v1"
REPRESENTATION = "shared-alpha-kind-edits-v1"
INTENT = ROOT / ".omx/artifacts/difference-pairwise-lora-compact-kind-intent-v1.json"
ARCHIVE = ROOT / ".omx/artifacts/difference-pairwise-lora-data-v1-precompact-archive/manifest.json"
AMENDMENT = ROOT / ".omx/plans/lora-successor-compact-kind-v1.md"
TEST_SPEC = ROOT / ".omx/plans/lora-successor-compact-kind-test-spec-v1.md"
HANDOFF = ROOT / ".omx/plans/lora-successor-compact-kind-handoff-v1.json"
MODEL = ROOT / "data/external/qwen3-4b-mlx-4bit"

INTENT_SHA256 = "e2009c3c323e523270dde8a6705ff3d754e950a2daea095d57cf0a5435457922"
ARCHIVE_SHA256 = "823f1bc6ecd01cc4bb20a991de3d69c58b4fbff7b127893440fc00f83aacb07a"
AMENDMENT_SHA256 = "0376b670313c015a7c13fa2f984c6abd1a3272e94f5208443a359d633a5784be"
TEST_SPEC_SHA256 = "44e1766a89375bd56e6836823dd6d045ada2c8276f81da3961922396bdce42da"
HANDOFF_SHA256 = "90f3b6382af349bf61b96862b658bb81bb87caaf7acb613db400c0737291f4c4"
MODEL_SOURCE_SHA256 = "67928c796b7e4562c072b2bbbdd0d52569e5ab9118ade1302f8d8d1bb4bae819"
CHAT_TEMPLATE_SHA256 = "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
CODEBOOK = {"L": "lexicon", "X": "mixed", "M": "morphology",
            "N": "neural", "D": "numeric", "R": "raw", "S": "syllable"}
REVERSE_CODEBOOK = {value: key for key, value in CODEBOOK.items()}
SYSTEM_TEXT = (
    "로마자 원문과 문맥을 보고 CURRENT를 유지하면 1, ALTERNATIVE로 바꾸면 2를 답하세요. "
    "K.c는 구간별 kind 코드열(L=lexicon,X=mixed,M=morphology,N=neural,D=numeric,R=raw,S=syllable), "
    "K.a는 CURRENT와 다른 ALTERNATIVE의 0기준 index=code 목록입니다. 숫자 하나만 답하세요."
)
VERSIONS = {"mlx": "0.31.2", "mlx-lm": "0.31.3", "transformers": "5.12.1"}
COUNTS = {"pairs": 1928, "units": 59, "references": 232, "updates": 15,
          "final_update_units": 3, "mechanism_rows": 175,
          "E_mechanism": 481, "E_train640": 1556}


class VerificationError(ValueError):
    """The preflight evidence is missing, malformed, or inconsistent."""


def need(condition, message):
    if not condition:
        raise VerificationError(message)


def canonical_bytes(value):
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise VerificationError("value is not canonical-JSON encodable") from exc


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


def encode_payload(semantic):
    """Encode a semantic Stage-1 payload under the frozen compact grammar."""
    need(isinstance(semantic, dict) and "K" not in semantic,
         "semantic payload is malformed or collides with reserved K")
    compact = deepcopy(semantic)
    current, alternative = compact.get("CURRENT"), compact.get("ALTERNATIVE")
    need(isinstance(current, dict) and isinstance(alternative, dict),
         "CURRENT and ALTERNATIVE objects are required")
    current_kinds = current.pop("kinds", None)
    alternative_kinds = alternative.pop("kinds", None)
    need(isinstance(current_kinds, list) and current_kinds
         and isinstance(alternative_kinds, list)
         and len(current_kinds) == len(alternative_kinds),
         "kind arrays must be nonempty and have equal length")
    try:
        current_codes = [REVERSE_CODEBOOK[value] for value in current_kinds]
        alternative_codes = [REVERSE_CODEBOOK[value] for value in alternative_kinds]
    except (KeyError, TypeError) as exc:
        raise VerificationError("kind array contains an unknown value") from exc
    edits = [f"{index}={right}" for index, (left, right) in
             enumerate(zip(current_codes, alternative_codes)) if left != right]
    compact["K"] = {"a": ",".join(edits), "c": "".join(current_codes)}
    return compact


def decode_payload(compact):
    """Strictly decode and canonicalize the compact representation."""
    need(isinstance(compact, dict), "compact payload must be an object")
    semantic = deepcopy(compact)
    kind_record = semantic.pop("K", None)
    current, alternative = semantic.get("CURRENT"), semantic.get("ALTERNATIVE")
    need(isinstance(kind_record, dict) and set(kind_record) == {"a", "c"},
         "K must contain exactly a and c")
    need(isinstance(current, dict) and isinstance(alternative, dict)
         and "kinds" not in current and "kinds" not in alternative,
         "compact candidates must omit full kind arrays")
    current_codes, edits_text = kind_record["c"], kind_record["a"]
    need(isinstance(current_codes, str) and current_codes
         and all(code in CODEBOOK for code in current_codes),
         "K.c must be a nonempty sequence of known uppercase codes")
    need(isinstance(edits_text, str), "K.a must be a string")
    alternative_codes = list(current_codes)
    prior = -1
    if edits_text:
        for edit in edits_text.split(","):
            match = re.fullmatch(r"(0|[1-9][0-9]*)=([LXMNDRS])", edit)
            need(match is not None, "K.a edit grammar differs")
            index, code = int(match.group(1)), match.group(2)
            need(index > prior and index < len(alternative_codes),
                 "K.a indices must be unique, increasing, and in range")
            need(alternative_codes[index] != code, "K.a contains a noncanonical unchanged edit")
            alternative_codes[index] = code
            prior = index
    current["kinds"] = [CODEBOOK[code] for code in current_codes]
    alternative["kinds"] = [CODEBOOK[code] for code in alternative_codes]
    need(encode_payload(semantic) == compact, "compact kind edit set is not canonical")
    return semantic


def _pair_payloads():
    role_rows, _ = base._load_fit_rows()
    all_pairs, repairs, _ = base._allowed_pairs(role_rows)
    units = base._expected_units(all_pairs, repairs)
    scopes, mechanism_rows, train640_rows = base._expected_scopes(all_pairs, units, role_rows)
    need(len(scopes) == COUNTS["pairs"] and len(units) == COUNTS["units"]
         and sum(len(unit["references"]) for unit in units) == COUNTS["references"]
         and len(mechanism_rows) == COUNTS["mechanism_rows"]
         and sum("mechanism" in values for values in scopes.values()) == COUNTS["E_mechanism"]
         and sum("train640" in values for values in scopes.values()) == COUNTS["E_train640"],
         "independent Stage-1 counts differ")
    rows = {(role, row["id"]): row for role in role_rows for row in role_rows[role]}
    payloads = {}
    for identity in sorted(scopes):
        pair = all_pairs[identity]
        row = rows[(pair["role"], pair["row_id"])]
        candidates, alternative_index = row["candidates"], pair["alternative_source_index"]
        finite = [index for index, candidate in enumerate(candidates)
                  if type(candidate.get("score")) in (int, float)
                  and math.isfinite(candidate["score"])]
        ranked = sorted(finite, key=lambda index: (-candidates[index]["score"], index))
        semantic = {
            "roman_input": row["input"],
            "external_context": base.external_context_window(row["context"], row["input"]),
            "raw_mode": False, "mapping": {"1": "KEEP", "2": "SWITCH"},
            "CURRENT": {"source_index": 0, "text": candidates[0]["text"],
                        "score": candidates[0]["score"], "rank": ranked.index(0) + 1,
                        "kinds": [segment["kind"] for segment in candidates[0]["segments"]]},
            "ALTERNATIVE": {"source_index": alternative_index,
                "text": candidates[alternative_index]["text"],
                "score": candidates[alternative_index]["score"],
                "rank": ranked.index(alternative_index) + 1,
                "kinds": [segment["kind"] for segment in candidates[alternative_index]["segments"]]},
            "changed_spans": pair["changes"],
        }
        compact = encode_payload(semantic)
        decoded = decode_payload(compact)
        need(canonical_bytes(decoded) == canonical_bytes(semantic),
             "independent compact round trip differs")
        payloads[identity] = (semantic, compact, decoded)
    return payloads, units


def _verify_archive_and_intent(report):
    need(file_digest(INTENT) == INTENT_SHA256 and file_digest(ARCHIVE) == ARCHIVE_SHA256,
         "intent or predecessor archive manifest pin differs")
    intent, archive = read_json(INTENT), read_json(ARCHIVE)
    need(intent.get("schema") == "difference-pairwise-lora-compact-kind-intent-v1"
         and intent.get("representation_version") == REPRESENTATION
         and intent.get("codebook") == CODEBOOK and intent.get("system_text") == SYSTEM_TEXT
         and intent.get("system_text_sha256") == hashlib.sha256(SYSTEM_TEXT.encode()).hexdigest()
         and intent.get("versions") == VERSIONS, "compact intent contract differs")
    for name, expected in intent.get("files", {}).items():
        need(file_digest(ROOT / name) == expected, f"intent file pin differs: {name}")
    need(archive.get("schema") == "pairwise-precompact-archive-v1"
         and archive.get("immutable") is True and isinstance(archive.get("files"), list),
         "predecessor archive manifest differs")
    for item in archive["files"]:
        archived = ROOT / item.get("archive", "")
        need(file_digest(archived) == item.get("sha256")
             and archived.stat().st_size == item.get("bytes"),
             f"archived predecessor differs: {item.get('archive')}")
    need(report.get("intent") == {"path": path_key(INTENT), "sha256": INTENT_SHA256}
         and report.get("archive_manifest") == {"path": path_key(ARCHIVE),
                                                  "sha256": ARCHIVE_SHA256},
         "report intent/archive binding differs")


def _verify_source_pins(report):
    before, after = report.get("source_hashes_before"), report.get("source_hashes_after")
    need(isinstance(before, dict) and before == after, "preflight source hashes differ")
    required = {path_key(INTENT): INTENT_SHA256, path_key(ARCHIVE): ARCHIVE_SHA256,
                path_key(AMENDMENT): AMENDMENT_SHA256, path_key(TEST_SPEC): TEST_SPEC_SHA256,
                path_key(HANDOFF): HANDOFF_SHA256,
                path_key(MODEL / "SOURCE.json"): MODEL_SOURCE_SHA256,
                path_key(MODEL / "tokenizer.json"):
                    "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
                path_key(MODEL / "tokenizer_config.json"):
                    "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
                "scripts/prepare_difference_pairwise_lora_data.py":
                    "c48c49a65cc547aee3ca35a4fcd27f346782c09151c126636b83ff7f978e43db",
                "scripts/verify_difference_pairwise_lora.py":
                    "8bf1609c49a74f17916f19dfb599af87ee36af74217fb380b4f50c53c68d0068",
                path_key(base.TRAIN640): base.TRAIN640_SHA256}
    for directory in (base.NATURAL, base.CHALLENGE):
        for filename in ("manifest.json", "cases.jsonl", "selection.jsonl"):
            path = directory / filename
            required[path_key(path)] = file_digest(path)
    for name, expected in required.items():
        need(before.get(name) == expected, f"required preflight source pin differs: {name}")
    for name, expected in before.items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        need(file_digest(path) == expected, f"preflight source changed: {name}")
    source = report.get("preflight_source")
    need(isinstance(source, dict) and isinstance(source.get("path"), str)
         and before.get(source["path"]) == source.get("sha256"),
         "preflight producer source binding differs")
    return before


def _tokenize(payloads, tokenizer):
    observations, lengths, token_hashes = [], [], []
    need(hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() == CHAT_TEMPLATE_SHA256,
         "actual tokenizer template differs")
    for identity in sorted(payloads):
        semantic, compact, decoded = payloads[identity]
        messages = [{"role": "system", "content": SYSTEM_TEXT},
                    {"role": "user", "content": canonical_bytes(compact).decode()}]
        tokens = tokenizer.apply_chat_template(messages, tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_dict=False)
        rendered = tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        need(isinstance(tokens, list) and tokens and all(type(value) is int for value in tokens)
             and tokenizer.encode(rendered, add_special_tokens=False) == tokens,
             "actual compact prompt tokenization differs")
        digits = {}
        for digit in ("1", "2"):
            completed = tokenizer.encode(rendered + digit, add_special_tokens=False)
            need(len(completed) == len(tokens) + 1 and completed[:-1] == tokens,
                 "answer digit is not one continuation token")
            digits[digit] = completed[-1]
        need(digits == {"1": 16, "2": 17}, "answer digit IDs differ from the pinned boundary")
        token_sha = digest_value(tokens)
        observations.append({"pair_id": identity,
            "semantic_sha256": digest_value(semantic), "compact_sha256": digest_value(compact),
            "decoded_sha256": digest_value(decoded), "tokens_sha256": token_sha,
            "token_count": len(tokens), "digit_continuation_ids": digits})
        lengths.append(len(tokens)); token_hashes.append(token_sha)
    ordered = sorted(lengths)
    statistics_record = {"min": ordered[0], "median": statistics.median(ordered),
        "p95": ordered[math.ceil(0.95 * len(ordered)) - 1], "max": ordered[-1],
        "count": len(ordered), "ceiling": 1024,
        "overflow": sum(length > 1024 for length in lengths)}
    return observations, statistics_record, digest_value(token_hashes)


def verify_preflight(path, tokenizer_loader=None):
    path = Path(path).resolve()
    report_sha256 = file_digest(path)
    report = read_json(path)
    need(report.get("schema") == REPORT_SCHEMA
         and report.get("representation_version") == REPRESENTATION,
         "compact preflight report schema differs")
    _verify_archive_and_intent(report)
    source_hashes = _verify_source_pins(report)
    need(report.get("codebook") == CODEBOOK and report.get("system_text") == SYSTEM_TEXT
         and report.get("versions") == VERSIONS and report.get("counts") == COUNTS,
         "preflight codec/version/count contract differs")
    versions = {name: importlib.metadata.version(name) for name in VERSIONS}
    need(versions == VERSIONS, "actual tokenizer package versions differ")
    if tokenizer_loader is None:
        from transformers import AutoTokenizer
        tokenizer_loader = lambda: AutoTokenizer.from_pretrained(
            MODEL, local_files_only=True, trust_remote_code=False)
    payloads, units = _pair_payloads()
    observations, stats, ordered_token_digest = _tokenize(payloads, tokenizer_loader())
    source_hashes_after_replay = _verify_source_pins(report)
    need(source_hashes_after_replay == source_hashes
         and file_digest(path) == report_sha256
         and file_digest(INTENT) == INTENT_SHA256
         and file_digest(ARCHIVE) == ARCHIVE_SHA256,
         "report, intent, archive, or source pins changed during independent replay")
    ordered_ids = sorted(payloads)
    need(report.get("ordered_pair_ids_sha256") == digest_value(ordered_ids)
         and report.get("units_sha256") == digest_value(units)
         and report.get("statistics") == stats
         and report.get("round_trips") == len(payloads)
         and report.get("ordered_token_digest") == ordered_token_digest
         and report.get("observations") == observations,
         "preflight identities, payloads, tokens, or statistics differ")
    passed = stats["overflow"] == 0 and stats["max"] <= 1024
    need(report.get("representation_passed") is passed
         and report.get("model_weights_loaded") is False
         and report.get("data_qualified") is False,
         "preflight terminal declaration differs")
    return {"stage": "compact_kind_preflight", "passed": passed,
            "preflight_sha256": report_sha256, "counts": COUNTS,
            "statistics": stats, "ordered_pair_ids_sha256": digest_value(ordered_ids),
            "units_sha256": digest_value(units), "ordered_token_digest": ordered_token_digest,
            "source_hashes_sha256": digest_value(source_hashes),
            "verifier_source": {"path": path_key(Path(__file__)),
                                "sha256": file_digest(Path(__file__))},
            "model_weights_loaded": False, "data_qualified": False}


def write_proof(path, payload):
    path = Path(path)
    need(not path.exists(), "proof output already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"schema": PROOF_SCHEMA, "payload": payload, "digest": digest_value(payload)}
    path.write_bytes(canonical_bytes(envelope) + b"\n")
    return envelope


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = verify_preflight(args.preflight)
        if not result["passed"]:
            print(json.dumps({"schema": PROOF_SCHEMA, "passed": False,
                              "reason": "consistent_tokenizer_overflow"},
                             ensure_ascii=False, sort_keys=True), file=sys.stderr)
            return 2
        write_proof(args.output, result)
    except VerificationError as exc:
        print(json.dumps({"schema": PROOF_SCHEMA, "passed": False, "error": str(exc)},
                         ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps({"schema": PROOF_SCHEMA, "passed": True, "output": str(args.output)},
                     ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
