"""Prepare the frozen long-context LoRA data without loading model weights.

Run ``fit-bridge`` with ``.venv`` (the strict FIT loader imports Torch), then
run ``prepare`` with ``.venv-mlx`` (the actual pinned tokenizer environment).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hanglish.core.phonology import romanize_word
from scripts.consensus_context_policy import prompt_choices
from scripts.final_choice_policy import proposal_allowed

NATURAL_FIT = ROOT / "data/natural_context/final-choice-v1/natural_fit"
CHALLENGE_FIT = ROOT / "data/natural_context/final-choice-v1/challenge_fit"
NATURAL_MANIFEST_SHA256 = "fc22b1635cc496ed70aa757750944726d583d640f2d9f3a598e380f3e262a1c9"
CHALLENGE_MANIFEST_SHA256 = "2b8a6349e0ad091e797d86694c2a5165b81fa906a91c3d540ece9042b1b66194"
PLAN = ROOT / ".omx/plans/long-context-lora-v1.md"
TEST_SPEC = ROOT / ".omx/plans/test-spec-long-context-lora-v1.md"
HANDOFF = ROOT / ".omx/plans/long-context-lora-handoff-v1.json"
CONTEXT = ROOT / ".omx/context/long-context-lora-context-v1.md"
PREFLIGHT = ROOT / ".omx/context/long-context-lora-preflight-v1.json"
LINEAGE = ROOT / ".omx/context/long-context-lora-tokenizer-lineage-v1.json"
PARTITION = ROOT / ".omx/artifacts/finite-final-choice-partition-v3/manifest.json"
NATURAL_RAW = ROOT / "data/natural_context/klue-v1.1-pilot-v2/train.jsonl"
MODEL_DIR = ROOT / "data/external/qwen3-4b-mlx-4bit"
OUTPUT = ROOT / ".omx/artifacts/long-context-lora-data-v1"

PREFLIGHT_SHA256 = "040f1d8aaca8321255e7fb9931a9ed1878bef28b204d96d02ca69751924c55d7"
LINEAGE_SHA256 = "0bae533c74ec1ffcf2afa674df0becb1c95119049202c732f44daa525c8cbd5d"
PARTITION_SHA256 = "8f699052b3e8d6e07dcaf59dc8696cc13a8ffb15b4e1bb5df62571e906ae3e61"
PLAN_SHA256 = "cfe51fa6ccf6f408cc763a654170ae01b9cb81bfb7fa01f948f12ebfdb39449e"
TEST_SPEC_SHA256 = "4799276085e525dc66a94dce7610f0edfbb168eb4c1b2ce28aa9f1d5ece90189"
CHAT_TEMPLATE_SHA256 = "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
TRAIN_SALT = "long-context-lora-train-v1-2026-09-15"
DIAGNOSTIC_SALT = "long-context-lora-diagnostic-v1-2026-09-15"
EVAL_SALT = "long-context-lora-eval-v1-2026-09-15"
SYSTEM_SOURCE = ROOT / ".omx/artifacts/lora-smoke-source/train.jsonl"
HISTORICAL_QWEN = (
    ROOT / ".omx/artifacts/lora-smoke-source/train.jsonl",
    ROOT / ".omx/artifacts/lora-smoke-source/valid.jsonl",
    ROOT / ".omx/artifacts/lora-expanded-source/train.jsonl",
    ROOT / ".omx/artifacts/lora-balanced-source/train.jsonl",
    ROOT / ".omx/artifacts/lora-balanced-source/valid.jsonl",
)
PRIOR_SELECTIONS = (
    ROOT / "data/natural_context/final-choice-v1/natural_fit/selection.jsonl",
    ROOT / "data/natural_context/final-choice-v1/natural_calibration/selection.jsonl",
    ROOT / "data/natural_context/final-choice-v1/natural_holdout/selection.jsonl",
    ROOT / "data/natural_context/final-choice-v1/challenge_fit/selection.jsonl",
    ROOT / "data/natural_context/final-choice-v1/challenge_preservation/selection.jsonl",
    ROOT / "data/natural_context/balanced-final-choice-eval-v2/selection.jsonl",
    ROOT / ".omx/artifacts/finite-final-choice-evaluation-v3/selection.jsonl",
)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def file_digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode()


def path_key(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_pinned_json(path: Path, expected: str):
    raw = path.read_bytes()
    if digest(raw) != expected:
        raise ValueError(f"pinned source differs: {path_key(path)}")
    return raw, json.loads(raw)


def read_jsonl(path: Path):
    raw = path.read_bytes()
    return raw, [json.loads(line) for line in raw.splitlines() if line]


def merge_hashes(target: dict, additions: dict, label: str):
    for name, value in additions.items():
        if name in target and target[name] != value:
            raise ValueError(f"{label} hash conflicts: {name}")
        target[name] = value


def verify_hashes(hashes: dict):
    for name, expected in hashes.items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        if not path.is_file() or file_digest(path) != expected:
            raise ValueError(f"preparation source changed: {name}")


def write_jsonl(path: Path, rows) -> str:
    raw = b"".join(canonical(row) + b"\n" for row in rows)
    path.write_bytes(raw)
    return digest(raw)


def fit_bridge(output: Path):
    if output.exists():
        raise FileExistsError(output)
    producer_sha = file_digest(Path(__file__))
    from scripts.train_final_choice_selector import load_fit_role, validate_fit_pair
    trainer_path = ROOT / "scripts/train_final_choice_selector.py"
    input_paths = [trainer_path]
    for directory in (NATURAL_FIT, CHALLENGE_FIT):
        input_paths.extend(directory / name for name in ("manifest.json", "cases.jsonl", "selection.jsonl"))
    before = {path_key(path): file_digest(path) for path in input_paths}
    natural = load_fit_role(NATURAL_FIT, NATURAL_MANIFEST_SHA256, "natural_fit")
    challenge = load_fit_role(CHALLENGE_FIT, CHALLENGE_MANIFEST_SHA256, "challenge_fit")
    validate_fit_pair(natural, challenge)
    captures = {}
    for role, loaded in (("natural_fit", natural), ("challenge_fit", challenge)):
        captures[role] = {
            "rows": len(loaded["rows"]),
            "ordered_ids_sha256": digest(canonical([r["id"] for r in loaded["rows"]])),
            "manifest_sha256": digest(loaded["manifest_raw"]),
            "cases_sha256": digest(loaded["cases_raw"]),
            "selection_sha256": digest(loaded["selection_raw"]),
        }
    report = {
        "schema": "long-context-lora-strict-fit-bridge-v1",
        "completed": True,
        "strict_loader": "scripts.train_final_choice_selector.load_fit_role",
        "strict_pair_validator": "scripts.train_final_choice_selector.validate_fit_pair",
        "torch_environment_required": True,
        "torch_version": importlib.metadata.version("torch"),
        "preparer_source_sha256": producer_sha,
        "strict_loader_source_sha256": before[path_key(trainer_path)],
        "input_hashes_before": dict(sorted(before.items())),
        "captures": captures,
    }
    after = {path_key(path): file_digest(path) for path in input_paths}
    if before != after or file_digest(Path(__file__)) != producer_sha:
        raise RuntimeError("strict FIT bridge sources changed during validation")
    report["input_hashes_after"] = dict(sorted(after.items()))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    return report


def validate_bridge(path: Path, role_rows: dict):
    bridge = json.loads(path.read_bytes())
    if bridge.get("schema") != "long-context-lora-strict-fit-bridge-v1" or bridge.get("completed") is not True:
        raise ValueError("completed strict FIT bridge is required")
    if (bridge.get("input_hashes_before") != bridge.get("input_hashes_after")
            or bridge.get("preparer_source_sha256") != file_digest(Path(__file__))
            or bridge.get("strict_loader_source_sha256") != file_digest(ROOT / "scripts/train_final_choice_selector.py")):
        raise ValueError("strict FIT bridge producer binding differs")
    verify_hashes(bridge["input_hashes_before"])
    for role, (directory, rows) in role_rows.items():
        item = bridge.get("captures", {}).get(role, {})
        expected = {
            "rows": len(rows),
            "ordered_ids_sha256": digest(canonical([r.get("id") for r in rows])),
            "manifest_sha256": digest((directory / "manifest.json").read_bytes()),
            "cases_sha256": digest((directory / "cases.jsonl").read_bytes()),
            "selection_sha256": digest((directory / "selection.jsonl").read_bytes()),
        }
        if item != expected:
            raise ValueError(f"strict FIT bridge differs for {role}")
    return bridge


def system_message():
    first = json.loads(SYSTEM_SOURCE.read_bytes().splitlines()[0])
    message = first.get("messages", [None])[0]
    if not isinstance(message, dict) or message.get("role") != "system" or not isinstance(message.get("content"), str):
        raise ValueError("historical system message is malformed")
    return deepcopy(message)


def make_prompt(row: dict, system: dict):
    source, context, candidates = row.get("input"), row.get("context"), row.get("candidates")
    presented = prompt_choices(source, context, candidates)
    content = f"로마자 입력: {source}\n문맥: {context or '(없음)'}\n후보:\n"
    content += "\n".join(f"{number}. {text}" for number, (_, text) in enumerate(presented, 1))
    return presented, [deepcopy(system), {"role": "user", "content": content}]


def tokenize_prompt(tokenizer, messages):
    tokens = tokenizer.apply_chat_template(messages, tokenize=True,
        add_generation_prompt=True, enable_thinking=False, return_dict=False)
    if not isinstance(tokens, list) or not tokens or any(type(v) is not int for v in tokens):
        raise ValueError("tokenizer returned malformed prompt tokens")
    return tokens


def validate_digit_boundary(tokenizer, messages, tokens, candidate_count):
    rendered = tokenizer.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    if not isinstance(rendered, str):
        raise ValueError("tokenizer did not render the generation prefix")
    rendered_tokens = tokenizer.encode(rendered, add_special_tokens=False)
    if rendered_tokens != tokens:
        raise ValueError("rendered generation prefix tokens differ")
    digit_ids = []
    for digit in range(1, candidate_count + 1):
        with_answer = tokenizer.encode(rendered + str(digit), add_special_tokens=False)
        if with_answer[:-1] != tokens or len(with_answer) != len(tokens) + 1:
            raise ValueError("choice digit is not one continuation token at prompt boundary")
        digit_ids.append(with_answer[-1])
    if len(set(digit_ids)) != candidate_count:
        raise ValueError("choice digits do not have distinct continuation token IDs")
    return digit_ids


def classify(row: dict, presented, prompt_tokens: int):
    if len(presented) < 2:
        return "insufficient_choices", []
    if prompt_tokens > 1024:
        return "prompt_over_1024", []
    expected = row.get("expected")
    visible = [(position, source_index) for position, (source_index, text) in enumerate(presented)
               if text in expected]
    if not visible:
        return "gold_not_in_choices", []
    allowed = [position for position, source_index in visible
               if source_index == 0 or proposal_allowed(row["input"], row["candidates"][0],
                                                         row["candidates"][source_index])]
    if not allowed:
        return "no_allowed_gold", []
    return "eligible", allowed


def salted(role: str, identity: str, salt: str):
    return digest((salt + "\0" + role + "\0" + identity).encode()), role, identity


def training_schedule(records):
    order = sorted(range(len(records)), key=lambda i: salted(
        records[i]["role"], records[i]["id"], TRAIN_SALT))
    return [{"microbatch": n + 1, "update": n // 4 + 1,
             "role": records[i]["role"], "id": records[i]["id"]}
            for n, i in enumerate(order)]


def validate_frozen_records(actual, expected):
    if actual != expected:
        raise ValueError("eligible record identity or gold mapping differs from pinned preflight")


def select_diagnostic(role_rows: dict):
    natural = sorted(role_rows["natural_fit"], key=lambda r: salted("natural_fit", r["id"], DIAGNOSTIC_SALT))[:512]
    single = [r for r in role_rows["challenge_fit"] if r.get("context") and re.fullmatch(r"[a-z]+", r.get("input", ""))]
    sentence = [r for r in role_rows["challenge_fit"] if re.search(r"\s", r.get("input", ""))]
    single = sorted(single, key=lambda r: salted("challenge_fit", r["id"], DIAGNOSTIC_SALT))[:64]
    sentence = sorted(sentence, key=lambda r: salted("challenge_fit", r["id"], DIAGNOSTIC_SALT))[:64]
    if (len(natural), len(single), len(sentence)) != (512, 64, 64):
        raise ValueError("FIT captures do not contain the fixed diagnostic shapes")
    selected = [("natural", "natural_fit", r) for r in natural]
    selected += [("challenge_lower_ascii_context_single", "challenge_fit", r) for r in single]
    selected += [("challenge_whitespace_sentence", "challenge_fit", r) for r in sentence]
    identities = [(role, row["id"]) for _, role, row in selected]
    if len(identities) != len(set(identities)):
        raise ValueError("diagnostic contains duplicate role IDs")
    return selected


def parse_historical_prompt(content: str):
    prefix, middle, marker = "로마자 입력: ", "\n문맥: ", "\n후보:"
    if not isinstance(content, str) or not content.startswith(prefix) or middle not in content or marker not in content:
        raise ValueError("historical Qwen prompt is malformed")
    source, rest = content[len(prefix):].split(middle, 1)
    context, _ = rest.split(marker, 1)
    return source, "" if context == "(없음)" else context


def collision_inventory():
    ids, groups, pairs, hashes = set(), set(), set(), {}
    for path in PRIOR_SELECTIONS:
        raw, rows = read_jsonl(path)
        hashes[path_key(path)] = digest(raw)
        for row in rows:
            source = {key: row.get(key) for key in ("id", "group_id", "input", "context")}
            if any(not isinstance(source[k], str) or not source[k] for k in ("id", "group_id", "input")) or not isinstance(source["context"], str):
                raise ValueError(f"prior source fields malformed: {path_key(path)}")
            ids.add(source["id"]); groups.add(source["group_id"])
            pairs.add((source["input"], source["context"]))
    historical_counts = []
    for path in HISTORICAL_QWEN:
        raw, rows = read_jsonl(path)
        hashes[path_key(path)] = digest(raw)
        historical_counts.append(len(rows))
        for row in rows:
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) != 3:
                raise ValueError("historical Qwen messages malformed")
            pairs.add(parse_historical_prompt(messages[1].get("content")))
    if historical_counts != [58, 8, 298, 588, 140]:
        raise ValueError("historical Qwen inventory counts changed")
    return ids, groups, pairs, hashes


def select_fresh(raw_rows, partition, collision_ids, collision_groups, collision_pairs,
                 target=512, expected_rows=2077, expected_groups=1607):
    remaining_ids = partition.get("remaining_ids")
    remaining_groups = partition.get("remaining_group_ids")
    if len(remaining_ids) != expected_rows or len(remaining_groups) != expected_groups:
        raise ValueError("fresh reserve inventory changed")
    remaining_id_set, remaining_group_set = set(remaining_ids), set(remaining_groups)
    if len(raw_rows) != 9002 and (expected_rows, expected_groups) == (2077, 1607):
        raise ValueError("raw natural TRAIN row count changed")
    by_group, seen = defaultdict(list), set()
    for raw in raw_rows:
        identity, group = raw.get("id"), raw.get("group_id")
        if (not isinstance(identity, str) or not identity or not isinstance(group, str) or not group
                or identity in seen):
            raise ValueError("raw natural TRAIN has malformed or duplicate source IDs")
        seen.add(identity)
        if (identity in remaining_id_set) != (group in remaining_group_set):
            raise ValueError("fresh reserve contains a partial document group")
        if identity in remaining_id_set:
            text, context = raw.get("text"), raw.get("left_context")
            if not isinstance(text, str) or not text or not isinstance(context, str):
                raise ValueError("reserve source fields are malformed")
            source = {"id": identity, "group_id": group,
                      "input": romanize_word(text), "context": context}
            by_group[group].append(source)
    if not remaining_id_set <= seen or set(by_group) != remaining_group_set:
        raise ValueError("reserve source join is incomplete")
    excluded = {}
    for group, rows in by_group.items():
        reasons = set()
        if any(r["id"] in collision_ids for r in rows): reasons.add("id")
        if group in collision_groups: reasons.add("known_group")
        if any((r["input"], r["context"]) in collision_pairs for r in rows): reasons.add("exact_input_context")
        if reasons: excluded[group] = sorted(reasons)
    admissible = set(by_group) - set(excluded)
    ordered = sorted(admissible, key=lambda group: (digest((EVAL_SALT + "\0" + group).encode()), group))
    selected_groups, selected = [], []
    for group in ordered:
        if len(selected) >= target: break
        selected_groups.append(group); selected.extend(by_group[group])
    if len(selected) < target:
        raise ValueError("fewer than 512 admissible fresh rows remain")
    return selected_groups, selected, excluded, len(admissible), sum(len(by_group[g]) for g in admissible)


def base_provenance(preflight, lineage, partition, bridge_path, collision_hashes,
                    producer_sha):
    hashes = {}
    for label, source in (("preflight", preflight), ("tokenizer lineage", lineage),
                          ("fresh partition", partition)):
        merge_hashes(hashes, source.get("source_hashes", {}), label)
    fixed = {path_key(PLAN): PLAN_SHA256, path_key(TEST_SPEC): TEST_SPEC_SHA256,
             path_key(PREFLIGHT): PREFLIGHT_SHA256, path_key(LINEAGE): LINEAGE_SHA256,
             path_key(PARTITION): PARTITION_SHA256,
             path_key(HANDOFF): digest(HANDOFF.read_bytes()), path_key(CONTEXT): digest(CONTEXT.read_bytes()),
             path_key(bridge_path): digest(bridge_path.read_bytes()),
             path_key(Path(__file__)): producer_sha,
             path_key(NATURAL_RAW): file_digest(NATURAL_RAW)}
    merge_hashes(hashes, fixed, "fixed preparation")
    merge_hashes(hashes, collision_hashes, "collision inventory")
    return dict(sorted(hashes.items()))


def prepare(output: Path, bridge_path: Path, tokenizer_loader=None):
    if output.exists():
        raise FileExistsError(output)
    producer_sha = file_digest(Path(__file__))
    preflight_raw, preflight = read_pinned_json(PREFLIGHT, PREFLIGHT_SHA256)
    lineage_raw, lineage = read_pinned_json(LINEAGE, LINEAGE_SHA256)
    _, partition = read_pinned_json(PARTITION, PARTITION_SHA256)
    _, handoff = read_pinned_json(HANDOFF, digest(HANDOFF.read_bytes()))
    if handoff.get("ralplan_execution_handoff", {}).get("authorized") is not True or handoff.get("planning_artifacts", {}).get(path_key(PLAN)) != PLAN_SHA256:
        raise ValueError("approved long-context handoff is required")
    if lineage.get("preflight_sha256") != PREFLIGHT_SHA256 or lineage.get("all_rows_verified") != 3492 or lineage.get("eligible_rows_verified") != 2771:
        raise ValueError("tokenizer lineage contract differs")
    if preflight.get("eligible_rows") != 2771 or len(preflight.get("eligible_records", [])) != 2771:
        raise ValueError("planning preflight contract differs")
    role_rows = {}
    for role, directory in (("natural_fit", NATURAL_FIT), ("challenge_fit", CHALLENGE_FIT)):
        _, rows = read_jsonl(directory / "cases.jsonl")
        role_rows[role] = rows
    validate_bridge(bridge_path, {"natural_fit": (NATURAL_FIT, role_rows["natural_fit"]),
                                  "challenge_fit": (CHALLENGE_FIT, role_rows["challenge_fit"])})
    if tokenizer_loader is None:
        from transformers import AutoTokenizer
        tokenizer_loader = lambda: AutoTokenizer.from_pretrained(
            MODEL_DIR, local_files_only=True, trust_remote_code=False)
    versions = {name: importlib.metadata.version(name) for name in ("mlx", "mlx-lm", "transformers")}
    if versions != {"mlx": "0.31.2", "mlx-lm": "0.31.3", "transformers": "5.12.1"}:
        raise ValueError("actual tokenizer environment versions differ")
    tokenizer = tokenizer_loader()
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str) or digest(chat_template.encode()) != CHAT_TEMPLATE_SHA256:
        raise ValueError("actual tokenizer chat template differs")
    system = system_message()
    lineage_digests = lineage.get("prompt_token_digests")
    if not isinstance(lineage_digests, list) or len(lineage_digests) != 3492:
        raise ValueError("tokenizer digest supplement is incomplete")
    digest_by_identity = {(x.get("role"), x.get("id")): x.get("tokens_sha256") for x in lineage_digests}
    if len(digest_by_identity) != 3492:
        raise ValueError("tokenizer digest supplement has duplicate identities")

    examples, prompts, token_rows, targets, records, accounting = [], [], [], [], [], Counter()
    prefix_tail, boundary_by_count = None, {}
    for role in ("natural_fit", "challenge_fit"):
        for row in role_rows[role]:
            presented, messages = make_prompt(row, system)
            tokens = tokenize_prompt(tokenizer, messages)
            if prefix_tail is None:
                prefix_tail = tokens[-4:]
            elif tokens[-4:] != prefix_tail:
                raise ValueError("assistant generation prefix token suffix differs across rows")
            if len(presented) not in boundary_by_count:
                boundary_by_count[len(presented)] = validate_digit_boundary(
                    tokenizer, messages, tokens, len(presented))
            token_sha = digest(canonical(tokens))
            if digest_by_identity.get((role, row.get("id"))) != token_sha:
                raise ValueError(f"per-row tokenizer digest differs: {role}/{row.get('id')}")
            status, allowed = classify(row, presented, len(tokens))
            accounting[(role, status)] += 1
            if status != "eligible": continue
            record = {"role": role, "id": row["id"], "group_id": row["group_id"],
                      "prompt_tokens": len(tokens), "candidate_count": len(presented),
                      "allowed_gold_positions": allowed}
            records.append(record)
            examples.append({"role": role, "id": row["id"], "group_id": row["group_id"],
                             "source": row["input"], "context": row["context"],
                             "candidates": deepcopy(row["candidates"]),
                             "choices": [{"source_index": i, "text": t} for i, t in presented]})
            prompts.append({"role": role, "id": row["id"], "messages": messages})
            token_rows.append({"role": role, "id": row["id"], "tokens": tokens,
                               "tokens_sha256": token_sha})
            targets.append({"role": role, "id": row["id"], "allowed_gold_positions": allowed})
    validate_frozen_records(records, preflight["eligible_records"])
    expected_accounting = {
        ("natural_fit", "eligible"): 1327, ("natural_fit", "gold_not_in_choices"): 700,
        ("natural_fit", "no_allowed_gold"): 20, ("natural_fit", "prompt_over_1024"): 1,
        ("challenge_fit", "eligible"): 1444,
    }
    if dict(accounting) != expected_accounting or sum(accounting.values()) != 3492:
        raise ValueError("eligibility accounting differs")

    schedule = training_schedule(records)
    if len(schedule) != 2771 or schedule[-1]["update"] != 693:
        raise ValueError("training schedule contract differs")
    diagnostic = [{"stratum": stratum, "role": role, "row": deepcopy(row)}
                  for stratum, role, row in select_diagnostic(role_rows)]

    collision_ids, collision_groups, collision_pairs, collision_hashes = collision_inventory()
    natural_raw_bytes, natural_raw_rows = read_jsonl(NATURAL_RAW)
    groups, fresh, excluded, admissible_groups, admissible_rows = select_fresh(
        natural_raw_rows, partition, collision_ids, collision_groups, collision_pairs)
    fresh_rows = [{"id": row["id"], "group_id": row["group_id"]} for row in fresh]
    before = base_provenance(preflight, lineage, partition, bridge_path, collision_hashes,
                             producer_sha)
    verify_hashes(before)
    if digest(preflight_raw) != PREFLIGHT_SHA256 or digest(lineage_raw) != LINEAGE_SHA256 or digest(natural_raw_bytes) != before[path_key(NATURAL_RAW)]:
        raise ValueError("source snapshot differs during preparation")

    tmp = Path(tempfile.mkdtemp(prefix=output.name + "-", dir=output.parent))
    try:
        file_hashes = {}
        file_hashes["training_examples.jsonl"] = write_jsonl(tmp / "training_examples.jsonl", examples)
        file_hashes["prompts.jsonl"] = write_jsonl(tmp / "prompts.jsonl", prompts)
        file_hashes["tokens.jsonl"] = write_jsonl(tmp / "tokens.jsonl", token_rows)
        file_hashes["targets.jsonl"] = write_jsonl(tmp / "targets.jsonl", targets)
        file_hashes["schedule.jsonl"] = write_jsonl(tmp / "schedule.jsonl", schedule)
        file_hashes["diagnostic.jsonl"] = write_jsonl(tmp / "diagnostic.jsonl", diagnostic)
        file_hashes["fresh_selection.jsonl"] = write_jsonl(tmp / "fresh_selection.jsonl", fresh_rows)
        after = base_provenance(preflight, lineage, partition, bridge_path, collision_hashes,
                                producer_sha)
        verify_hashes(after)
        if before != after or file_digest(Path(__file__)) != producer_sha:
            raise RuntimeError("preparation sources changed during execution")
        manifest = {
            "schema": "long-context-lora-data-v1", "completed": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "scope": "model-free isolated LoRA data preparation",
            "rows_accounted": 3492, "eligible_rows": 2771,
            "eligible_records_sha256": digest(canonical(records)),
            "training_schedule": {"salt": TRAIN_SALT, "microbatches": 2771, "updates": 693,
                                  "final_update_microbatches": 3},
            "eligibility_accounting": {f"{role}:{status}": count for (role, status), count in sorted(accounting.items())},
            "diagnostic": {"salt": DIAGNOSTIC_SALT, "rows": 640,
                           "strata": {"natural": 512, "challenge_lower_ascii_context_single": 64,
                                      "challenge_whitespace_sentence": 64},
                           "parent_fit_rows_retained": True},
            "fresh": {"salt": EVAL_SALT, "rows": len(fresh_rows), "groups": len(groups),
                      "group_ids": groups, "admissible_rows": admissible_rows,
                      "admissible_groups": admissible_groups, "excluded_groups": excluded,
                      "whole_group_selection": True, "target_fields_copied": False,
                      "selection_fields": ["id", "group_id"],
                      "source_boundary": "raw JSON bytes were parsed; text was used only to deterministically derive the inference input for exact-pair exclusion, while no expected/category/outcome was copied, compared, ordered, or used for selection"},
            "historical_qwen_limitations": {"id_collision_check": "unavailable: historical messages have no IDs",
                                             "group_collision_check": "unavailable: historical messages have no group IDs",
                                             "semantic_family_independence": "unestablished",
                                             "upstream_pretraining_overlap": "unknown"},
            "prompt_fields": ["source", "context", "candidates", "choices", "messages"],
            "targets_separate_from_prompts": True, "full_candidate_objects_preserved": True,
            "enable_thinking": False, "max_prompt_tokens": 1024,
            "tokenizer_versions": versions,
            "tokenizer_config_sha256": file_digest(MODEL_DIR / "tokenizer_config.json"),
            "tokenizer_chat_template_sha256": CHAT_TEMPLATE_SHA256,
            "generation_prefix_token_suffix": prefix_tail,
            "digit_continuation_ids_by_candidate_count": {str(k): v for k, v in sorted(boundary_by_count.items())},
            "preflight_sha256": PREFLIGHT_SHA256, "tokenizer_lineage_sha256": LINEAGE_SHA256,
            "strict_fit_bridge_sha256": digest(bridge_path.read_bytes()),
            "file_hashes": dict(sorted(file_hashes.items())),
            "source_hashes": before, "source_hashes_after": after,
            "model_loaded": False, "training_run": False, "fresh_targets_read_for_evaluation": False,
            "development_or_final_read": False, "deployed": False,
        }
        (tmp / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        tmp.rename(output)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return manifest


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="phase", required=True)
    bridge = sub.add_parser("fit-bridge")
    bridge.add_argument("--output", type=Path, required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--bridge", type=Path, required=True)
    prep.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = fit_bridge(args.output) if args.phase == "fit-bridge" else prepare(args.output, args.bridge)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
