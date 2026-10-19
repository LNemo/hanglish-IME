"""Run the fixed four-row MLX resource smoke for context consensus."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import sys
import time


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.consensus_context_policy import decide, prompt_choices


OUTPUT = ROOT / ".omx/artifacts/consensus-context-resource-smoke-v1"
PLAN_SHA256 = "8329bc741f5c69cba6f8e779c9b5b2b0c2ef15b32b29f64776729ffbb077778e"
POLICY_SHA256 = "cd8f3ad1943f96faf7e821f1d9ebd3f24f376d0e03637fb6254949dc5c5a99e1"
EVALUATOR_SHA256 = "d8b71406aa276256cdc85fc9305d91555a00257efb206b779a16508307f5b9a0"
MODEL_SOURCE_SHA256 = "67928c796b7e4562c072b2bbbdd0d52569e5ab9118ade1302f8d8d1bb4bae819"
ADAPTER_SHA256 = "a5e3e00a6641e6328cb3a00baf0601a80fffc63765114276e96d119554da27f0"
ADAPTER_CONFIG_SHA256 = "1975a3d86c22966a14c771f1107a314e29e7180f268e0e92d1509b72fc97e1c6"
TRAINING_REPORT_SHA256 = "d45b619af009304ba798bf211f47c41d0aefe4752e2914bfd6da617388c0f8c5"
TRAINING_SOURCE_SHA256 = "0d176bafdc8d3f6c035c2f9ce98e2a153c8ba5fa2c172d6f29a3ba2094d2e7e8"
RUNTIME_VERSION = "neural-1-867f13596ed5"
MODEL_REVISION = "52a5ab34fa604bc8af6d3ce0cac0cab10b7eb495"
MODEL_WEIGHT_SHA256 = "e56a94b846a2dce7d1cc154ce58b9cfc4b8d99e8ffb74bfe256a265f4cbc7fb9"
THINKING_PREFIX = "{% set enable_thinking = false %}\n"
PROFILE = {"rows": 4, "max_tokens": 256, "time_limit_seconds": 180,
           "rss_limit_bytes": 6 * 1024**3, "mlx_peak_limit_bytes": 6 * 1024**3}

MODEL_DIR = ROOT / "data/external/qwen3-4b-mlx-4bit"
MODEL_SOURCE = MODEL_DIR / "SOURCE.json"
ADAPTER_DIR = ROOT / ".omx/artifacts/mlx-learning-probe-run/adapter"
ADAPTER_FILE = ADAPTER_DIR / "adapters.safetensors"
ADAPTER_CONFIG = ADAPTER_DIR / "adapter_config.json"
TRAINING_REPORT = ROOT / ".omx/artifacts/mlx-learning-probe-run/report.json"
TRAINING_SOURCE = ROOT / ".omx/artifacts/lora-smoke-source/train.jsonl"
EVALUATOR = ROOT / ".omx/artifacts/evaluate_mlx_selector_v2.py"
POLICY = ROOT / "scripts/consensus_context_policy.py"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def file_digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def path_key(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def require_sha(value, label):
    if (not isinstance(value, str) or len(value) != 64
            or set(value) - set("0123456789abcdef")):
        raise ValueError(f"explicit {label} SHA-256 is required")


def rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def require_resources(started, mlx_peak=0):
    if time.monotonic() - started > PROFILE["time_limit_seconds"]:
        raise RuntimeError("consensus resource smoke time bound exceeded")
    if rss_bytes() > PROFILE["rss_limit_bytes"]:
        raise RuntimeError("consensus resource smoke RSS bound exceeded")
    if mlx_peak > PROFILE["mlx_peak_limit_bytes"]:
        raise RuntimeError("consensus resource smoke MLX bound exceeded")


def verify_hashes(values):
    for name, expected in values.items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        if file_digest(path) != expected:
            raise ValueError(f"resource smoke input changed: {name}")


def merge_hashes(target, values, label):
    if not isinstance(values, dict) or not values:
        raise ValueError(f"{label} source hashes missing")
    for name, value in values.items():
        require_sha(value, f"{label} source")
        if name in target and target[name] != value:
            raise ValueError(f"{label} source hash conflict: {name}")
        target[name] = value


def read_json_once(path, expected, label):
    raw = path.read_bytes()
    if digest(raw) != expected:
        raise ValueError(f"{label} hash differs")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return raw, value


def normalize_overlap_identities(entries, pilot_identities):
    if not isinstance(entries, list):
        raise ValueError("pilot lineage overlap record is malformed")
    result = []
    for entry in entries:
        if (not isinstance(entry, dict) or set(entry) != {"role", "id"}
                or entry.get("role") not in {"natural_fit", "challenge_fit"}
                or not isinstance(entry.get("id"), str) or not entry["id"]):
            raise ValueError("pilot lineage overlap identity is malformed")
        identity = (entry["role"], entry["id"])
        if identity not in pilot_identities:
            raise ValueError("pilot lineage overlap identity is outside pilot")
        result.append(identity)
    if len(result) != len(set(result)):
        raise ValueError("pilot lineage overlap identities are duplicated")
    return result


def load_inputs(data_dir, manifest_sha, lineage_path, lineage_sha):
    require_sha(manifest_sha, "pilot manifest")
    require_sha(lineage_sha, "lineage report")
    manifest_raw, manifest = read_json_once(
        data_dir / "manifest.json", manifest_sha, "pilot manifest")
    cases_raw = (data_dir / "cases.jsonl").read_bytes()
    if (manifest.get("schema") != "consensus-context-train-pilot-data-v1"
            or manifest.get("completed") is not True
            or manifest.get("rows") != 128
            or manifest.get("cases_sha256") != digest(cases_raw)
            or manifest.get("runtime_version") != RUNTIME_VERSION
            or manifest.get("plan_sha256") != PLAN_SHA256
            or manifest.get("source_hashes") != manifest.get("source_hashes_after")
            or manifest.get("mlx_model_inference_run") is not False
            or manifest.get("development_data_read") is not False
            or manifest.get("final_data_read") is not False
            or manifest.get("deployed") is not False
            or manifest.get("selection_uses_labels_categories_eligibility_or_outcomes") is not False):
        raise ValueError("pilot manifest identity or TRAIN-only scope differs")
    rows = [json.loads(line) for line in cases_raw.splitlines() if line]
    if len(rows) != 128 or any(
            not isinstance(item, dict) or item.get("role") not in
            {"natural_fit", "challenge_fit"} or not isinstance(item.get("row"), dict)
            for item in rows):
        raise ValueError("pilot cases are malformed")
    sections = manifest.get("selection")
    if (not isinstance(sections, dict)
            or manifest.get("shape_counts") != {
                "natural": 64, "challenge_single": 32, "challenge_sentence": 32}
            or [item["role"] for item in rows] !=
               [*("natural_fit" for _ in range(64)),
                *("challenge_fit" for _ in range(64))]):
        raise ValueError("pilot selection roles or shapes differ")
    slices = {"natural": rows[:64], "challenge_single": rows[64:96],
              "challenge_sentence": rows[96:]}
    for name, selected in slices.items():
        section = sections.get(name)
        if (not isinstance(section, dict) or section.get("rows") != len(selected)
                or section.get("ids") != [item["row"].get("id") for item in selected]
                or section.get("group_ids") !=
                   [item["row"].get("group_id") for item in selected]):
            raise ValueError("pilot selection identity differs")
    lineage_raw, lineage = read_json_once(lineage_path, lineage_sha, "lineage report")
    if (lineage.get("completed") is not True
            or lineage.get("pilot_manifest_sha256") != manifest_sha
            or lineage.get("source_hashes") != lineage.get("source_hashes_after")
            or lineage.get("family_group_independence_established") is not False
            or lineage.get("standalone_development_corpus_read") is not False
            or lineage.get("historical_validation_records_read") != 8
            or lineage.get("final_data_read") is not False):
        raise ValueError("pilot lineage report differs")
    pilot_identities = {(item["role"], item["row"].get("id")) for item in rows}
    for name, historical_rows in (("optimizer_overlap", 58),
                                  ("validation_overlap", 8)):
        overlap = lineage.get(name)
        if not isinstance(overlap, dict):
            raise ValueError("pilot lineage overlap record is missing")
        ids = overlap.get("id_overlap")
        pairs = overlap.get("exact_input_context_overlap")
        id_set = overlap.get("historical_id_set_sha256")
        normalized_ids = normalize_overlap_identities(ids, pilot_identities)
        normalized_pairs = normalize_overlap_identities(pairs, pilot_identities)
        if (overlap.get("historical_rows") != historical_rows
                or overlap.get("id_overlap_count") != len(normalized_ids)
                or overlap.get("exact_input_context_overlap_count") != len(normalized_pairs)
                or not isinstance(id_set, str) or len(id_set) != 64
                or set(id_set) - set("0123456789abcdef")):
            raise ValueError("pilot lineage overlap record is malformed")
    return {"manifest_raw": manifest_raw, "manifest": manifest,
            "cases_raw": cases_raw, "rows": rows, "lineage_raw": lineage_raw,
            "lineage": lineage}


def validate_frozen_sources(inputs, *, data_dir, manifest_sha,
                            lineage_path, lineage_sha, runner_sha, plan_path):
    source_raw, source = read_json_once(MODEL_SOURCE, MODEL_SOURCE_SHA256, "model source")
    if source.get("revision") != MODEL_REVISION or not isinstance(source.get("files"), dict):
        raise ValueError("base model source manifest differs")
    if source["files"].get("model.safetensors") != MODEL_WEIGHT_SHA256:
        raise ValueError("base model weight identity differs")
    training_raw, training = read_json_once(
        TRAINING_REPORT, TRAINING_REPORT_SHA256, "adapter training report")
    if (training.get("adapter", {}).get("sha256") != ADAPTER_SHA256
            or training.get("adapter", {}).get("config_sha256") != ADAPTER_CONFIG_SHA256
            or training.get("source_sha256", {}).get("train.jsonl") != TRAINING_SOURCE_SHA256
            or training.get("model", {}).get("revision") != MODEL_REVISION
            or training.get("model", {}).get("files_sha256") != source["files"]):
        raise ValueError("adapter training provenance differs")
    fixed = {
        path_key(MODEL_SOURCE): MODEL_SOURCE_SHA256,
        path_key(ADAPTER_FILE): ADAPTER_SHA256,
        path_key(ADAPTER_CONFIG): ADAPTER_CONFIG_SHA256,
        path_key(TRAINING_REPORT): TRAINING_REPORT_SHA256,
        path_key(TRAINING_SOURCE): TRAINING_SOURCE_SHA256,
        path_key(EVALUATOR): EVALUATOR_SHA256,
        path_key(POLICY): POLICY_SHA256,
        path_key(plan_path): PLAN_SHA256,
        path_key(data_dir / "manifest.json"): manifest_sha,
        path_key(data_dir / "cases.jsonl"): digest(inputs["cases_raw"]),
        path_key(lineage_path): lineage_sha,
        "scripts/probe_consensus_context_resources.py": runner_sha,
    }
    for name, expected in source["files"].items():
        if Path(name).name != name:
            raise ValueError("model source contains a nonlocal file name")
        fixed[path_key(MODEL_DIR / name)] = expected
    merge_hashes(fixed, inputs["manifest"]["source_hashes"], "pilot")
    merge_hashes(fixed, inputs["lineage"]["source_hashes"], "lineage")
    verify_hashes(fixed)
    return dict(sorted(fixed.items())), source_raw, training_raw


def system_message():
    raw = TRAINING_SOURCE.read_bytes()
    if digest(raw) != TRAINING_SOURCE_SHA256:
        raise ValueError("training prompt source differs")
    first = json.loads(raw.splitlines()[0])
    message = first.get("messages", [None])[0]
    if (not isinstance(message, dict) or message.get("role") != "system"
            or not isinstance(message.get("content"), str)):
        raise ValueError("historical system prompt is malformed")
    return deepcopy(message)


def prompt_record(item, system):
    row = item["row"]
    source, context, candidates = row.get("input"), row.get("context"), row.get("candidates")
    raw_response = row.get("raw_bypass_response")
    if not isinstance(raw_response, dict):
        raise ValueError("pilot raw bypass response is missing")
    if not isinstance(context, str):
        raise ValueError("pilot context is malformed")
    presented = prompt_choices(source, context, candidates)
    prompt = f"로마자 입력: {source}\n문맥: {context or '(없음)'}\n후보:\n"
    prompt += "\n".join(f"{number}. {text}" for number, (_, text) in enumerate(presented, 1))
    return {"id": row.get("id"), "role": item["role"], "source": source,
            "context": context, "candidates": candidates,
            "raw_bypass_response": raw_response, "choices": presented,
            "messages": [deepcopy(system), {"role": "user", "content": prompt}]}


def finite_distribution(logits):
    values = [float(value) for value in logits]
    if not values or len(values) > 5 or not all(math.isfinite(value) for value in values):
        raise ValueError("choice logits must be one to five finite values")
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = sum(weights)
    probabilities = [value / total for value in weights]
    chosen = max(range(len(values)), key=lambda index: (probabilities[index], -index))
    return probabilities, chosen


def preserve_decision(record, base_source_index, trained_source_index):
    before = deepcopy(record["candidates"])
    result = decide(record["source"], record["candidates"],
                    base_source_index, trained_source_index)
    expected = ([record["candidates"][result["chosen_index"]]]
                + record["candidates"][:result["chosen_index"]]
                + record["candidates"][result["chosen_index"] + 1:])
    if (record["candidates"] != before or len(result["candidates"]) != len(before)
            or any(left is not right for left, right in zip(result["candidates"], expected))):
        raise RuntimeError("consensus decision changed candidate objects or content")
    actual_raw = record["raw_bypass_response"]
    raw_before = deepcopy(actual_raw)
    raw_candidates = actual_raw.get("candidates")
    if (not isinstance(raw_candidates, list) or not raw_candidates
            or not isinstance(raw_candidates[0], dict)):
        raise RuntimeError("captured raw bypass candidates are malformed")
    raw = decide(record["source"], raw_candidates, None, None, raw=True)
    if (actual_raw != raw_before or raw["chosen_index"] != 0
            or raw["result"] != record["source"]
            or actual_raw.get("result") != record["source"]
            or raw_candidates[0].get("text") != record["source"]
            or raw["candidates"] is not raw_candidates
            or record["candidates"] != before):
        raise RuntimeError("raw bypass changed candidates or source")
    return result, {"decision": raw, "response": raw_before,
                    "actual_candidate_list_identity_preserved": True}


class MlxBackend:
    def __init__(self):
        import mlx.core as mx
        from mlx_lm.tuner.utils import load_adapters
        from mlx_lm.utils import load
        self.mx, self.load, self.load_adapters = mx, load, load_adapters
        mx.set_memory_limit(PROFILE["mlx_peak_limit_bytes"])
        mx.set_cache_limit(256 * 1024**2)

    def load_model(self, adapter=False):
        model, tokenizer = self.load(str(MODEL_DIR), tokenizer_config={"trust_remote_code": False})
        tokenizer.chat_template = THINKING_PREFIX + tokenizer.chat_template
        if adapter:
            self.load_adapters(model, str(ADAPTER_DIR))
        model.eval()
        return model, tokenizer

    def tokens(self, tokenizer, messages):
        return list(tokenizer.apply_chat_template(messages, add_generation_prompt=True))

    def digit_ids(self, tokenizer):
        result = [tokenizer.encode(str(value), add_special_tokens=False) for value in range(1, 6)]
        if any(len(ids) != 1 for ids in result):
            raise RuntimeError("digits 1..5 are not single tokens")
        return [ids[0] for ids in result]

    def infer(self, model, tokens, digit_ids, count):
        mx = self.mx
        logits = model(mx.array([tokens]))[0, -1].astype(mx.float32)
        selected = logits[mx.array(digit_ids[:count])]
        mx.eval(selected)
        return selected.tolist()

    def peak(self):
        return int(self.mx.get_peak_memory())

    def clear(self):
        gc.collect()
        self.mx.clear_cache()


def infer_stage(backend, records, *, adapter, expected_tokens=None, started=None):
    model, tokenizer = backend.load_model(adapter=adapter)
    try:
        require_resources(started, backend.peak())
        digit_ids = backend.digit_ids(tokenizer)
        outcomes = []
        for index, record in enumerate(records):
            if len(record["choices"]) < 2:
                continue
            tokens = backend.tokens(tokenizer, record["messages"])
            if len(tokens) > PROFILE["max_tokens"]:
                continue
            if expected_tokens is not None:
                if index not in expected_tokens or tokens != expected_tokens[index]:
                    raise RuntimeError("base and adapter prompt tokens differ")
            require_resources(started, backend.peak())
            logits = backend.infer(model, tokens, digit_ids, len(record["choices"]))
            require_resources(started, backend.peak())
            probabilities, selected = finite_distribution(logits)
            outcomes.append({"record_index": index, "tokens": tokens, "logits": logits,
                             "probabilities": probabilities,
                             "prediction_source_index": record["choices"][selected][0]})
            if len(outcomes) == PROFILE["rows"]:
                break
            require_resources(started, backend.peak())
        if len(outcomes) != PROFILE["rows"]:
            raise RuntimeError("fewer than four prompt-eligible pilot rows")
        return outcomes
    finally:
        del model, tokenizer
        backend.clear()


def run(args, backend_factory=MlxBackend):
    started = time.monotonic()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.plan_sha256 != PLAN_SHA256:
        raise ValueError("only the frozen consensus plan may be used")
    plan_raw = args.plan.read_bytes()
    if digest(plan_raw) != PLAN_SHA256:
        raise ValueError("consensus plan hash differs")
    inputs = load_inputs(args.data, args.manifest_sha256,
                         args.lineage_report, args.lineage_sha256)
    runner_raw = Path(__file__).read_bytes()
    before, _, _ = validate_frozen_sources(
        inputs, data_dir=args.data, manifest_sha=args.manifest_sha256,
        lineage_path=args.lineage_report, lineage_sha=args.lineage_sha256,
        runner_sha=digest(runner_raw), plan_path=args.plan)
    require_resources(started)
    system = system_message()
    records = [prompt_record(item, system) for item in inputs["rows"]]
    backend = backend_factory()
    base = infer_stage(backend, records, adapter=False, started=started)
    require_resources(started, backend.peak())
    trained = infer_stage(backend, records, adapter=True,
                          expected_tokens={row["record_index"]: row["tokens"] for row in base},
                          started=started)
    require_resources(started, backend.peak())
    cases = []
    for left, right in zip(base, trained):
        if left["record_index"] != right["record_index"]:
            raise RuntimeError("base and adapter selected different source rows")
        record = records[left["record_index"]]
        decision, raw = preserve_decision(
            record, left["prediction_source_index"], right["prediction_source_index"])
        cases.append({"id": record["id"], "role": record["role"],
                      "choices": record["choices"], "tokens": left["tokens"],
                      "base_logits": left["logits"], "base_probabilities": left["probabilities"],
                      "base_prediction_source_index": left["prediction_source_index"],
                      "trained_logits": right["logits"],
                      "trained_probabilities": right["probabilities"],
                      "trained_prediction_source_index": right["prediction_source_index"],
                      "decision": {"chosen_index": decision["chosen_index"],
                                   "result": decision["result"], "reason": decision["reason"]},
                      "candidate_objects": deepcopy(record["candidates"]),
                      "candidate_objects_and_relative_order_preserved": True,
                      "raw_bypass_response": raw["response"],
                      "raw_bypass_decision": {
                          "result": raw["decision"]["result"],
                          "reason": raw["decision"]["reason"],
                          "chosen_index": raw["decision"]["chosen_index"]},
                      "raw_candidate_list_identity_preserved":
                          raw["actual_candidate_list_identity_preserved"],
                      "labels_read": False})
    after, _, _ = validate_frozen_sources(
        inputs, data_dir=args.data, manifest_sha=args.manifest_sha256,
        lineage_path=args.lineage_report, lineage_sha=args.lineage_sha256,
        runner_sha=digest(runner_raw), plan_path=args.plan)
    if before != after or digest(Path(__file__).read_bytes()) != digest(runner_raw):
        raise RuntimeError("resource smoke inputs changed during execution")
    require_resources(started, backend.peak())
    args.output.mkdir(parents=True)
    (args.output / "source.py").write_bytes(runner_raw)
    report = {
        "schema": "consensus-context-resource-smoke-v1", "completed": True,
        "scope": "four-row TRAIN resource feasibility smoke; no accuracy or promotion gate",
        "profile": PROFILE, "rows": len(cases), "cases": cases,
        "plan_sha256": PLAN_SHA256, "pilot_manifest_sha256": args.manifest_sha256,
        "lineage_report_sha256": args.lineage_sha256,
        "runtime_version": RUNTIME_VERSION, "base_revision": MODEL_REVISION,
        "base_model_sha256": MODEL_WEIGHT_SHA256,
        "adapter_sha256": ADAPTER_SHA256, "source_hashes": before,
        "source_hashes_after": after, "elapsed_seconds": time.monotonic() - started,
        "rss_peak_bytes": rss_bytes(), "mlx_peak_bytes": backend.peak(),
        "versions": {name: importlib.metadata.version(name)
                     for name in ("mlx", "mlx-lm")},
        "base_released_before_adapter_load": True,
        "identical_prompt_tokens_and_choices": True,
        "candidate_objects_and_raw_bypass_preserved": True,
        "expected_or_labels_read": False, "accuracy_measured": False,
        "resource_only": True, "pilot_gate": False, "deployment_eligible": False,
        "development_data_read": False, "final_data_read": False,
    }
    report_raw = (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2)
                  + "\n").encode()
    require_resources(started, backend.peak())
    (args.output / "report.json").write_bytes(report_raw)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--lineage-report", type=Path, required=True)
    parser.add_argument("--lineage-sha256", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
