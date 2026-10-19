"""Run the fixed four-row completed-thinking resource feasibility probe."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.final_choice_policy import proposal_allowed, reorder_candidates
from scripts.probe_consensus_context_resources import (
    MODEL_DIR, MODEL_REVISION, MODEL_WEIGHT_SHA256, RUNTIME_VERSION,
    digest, file_digest, load_inputs, path_key, prompt_record, rss_bytes,
    system_message, validate_frozen_sources, verify_hashes,
)


OUTPUT = ROOT / ".omx/artifacts/thinking-context-resource-v1"
PLAN = ROOT / ".omx/artifacts/thinking-context-plan-v1.md"
PLAN_SHA256 = "b54e1c5ba570442b5dc96c85bf1d1802e846a4193401e7826c55dc3d3decfff2"
SELECTION_PLAN = ROOT / ".omx/artifacts/consensus-context-plan-v1.md"
PILOT_DATA = ROOT / ".omx/artifacts/consensus-context-pilot-data-v1"
PILOT_MANIFEST_SHA256 = "3e9f84fa3c6defc1075824a984e90945895c4ace8c9609521c328172cd013f86"
LINEAGE = ROOT / ".omx/artifacts/consensus-context-work-v1/pilot-lineage.json"
LINEAGE_SHA256 = "1720d99b7e6904eb1289cc35d89bcadf01767be5a08fec24c56b3ded6c40d183"
RESOURCE_HELPER_SHA256 = "eb189141b33a880003b18cb879811c7813d0a41cda0fef4d2059c3772dd50cb9"
FINAL_POLICY_SHA256 = "3acb1480fce1c490892e003a9677b0ad5354d07e971dd3ca73446c08d69cf717"
OFFICIAL_TEMPLATE_SHA256 = "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
GENERATION_SOURCE_SHA256 = "270778ad53eaca55a8533d82e6752660fe5d2605c4aa0879b48a50a91f69345f"
SAMPLING_SOURCE_SHA256 = "c0ce439f8dbf0d4e6d0f37f728a324f3f72878e6a9123df41be520804c596d67"
TOKENIZER_SOURCE_SHA256 = "25784bb03c922d0d7832ce6c66a6cd4eb3a4820b6c5a8e583dedb63a018fb56a"
VERSIONS = {"mlx": "0.31.2", "mlx-lm": "0.31.3", "transformers": "5.12.1"}
SALT = "thinking-context-v1-2026-09-15"
PROFILE = {"rows": 4, "max_prompt_tokens": 1024,
           "max_generated_tokens": 2048, "row_time_limit_seconds": 120,
           "time_limit_seconds": 600, "rss_limit_bytes": 6 * 1024**3,
           "mlx_peak_limit_bytes": 6 * 1024**3, "cache_limit_bytes": 256 * 1024**2}
SAMPLER = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}


def package_sources():
    base = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}"
    package = base / "site-packages/mlx_lm"
    return {path_key(package / "generate.py"): GENERATION_SOURCE_SHA256,
            path_key(package / "sample_utils.py"): SAMPLING_SOURCE_SHA256,
            path_key(package / "tokenizer_utils.py"): TOKENIZER_SOURCE_SHA256}


def require_versions():
    actual = {name: importlib.metadata.version(name) for name in VERSIONS}
    if actual != VERSIONS:
        raise ValueError("installed MLX generation environment differs")
    return actual


def seed_for(source, context):
    if not isinstance(source, str) or not source or not isinstance(context, str):
        raise ValueError("thinking seed identity is malformed")
    payload = (SALT + "\0").encode() + json.dumps(
        [source, context], ensure_ascii=False).encode()
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def inspect_reasoning(tokens, opener, closer, eos, decode):
    """Classify a consumed prefix without adding or repairing any token."""
    if any(type(value) is not int for value in tokens):
        raise ValueError("generated tokens must be integers")
    opened = False
    body_start = None
    for index, token in enumerate(tokens):
        if token == eos:
            return {"terminal": True, "valid": False, "reason": "eos_before_close"}
        if not opened:
            if not decode(tokens[:index + 1]).strip():
                continue
            if token != opener:
                return {"terminal": True, "valid": False, "reason": "missing_opener"}
            opened, body_start = True, index + 1
            continue
        if token == opener:
            return {"terminal": True, "valid": False, "reason": "nested_opener"}
        if token == closer:
            body = decode(tokens[body_start:index])
            if not body.strip():
                return {"terminal": True, "valid": False, "reason": "empty_reasoning"}
            return {"terminal": True, "valid": True, "reason": "completed_reasoning",
                    "body": body}
    return {"terminal": False, "valid": False,
            "reason": "reasoning_open" if opened else "awaiting_opener"}


def snapshot_raw_record(record):
    raw = record.get("raw_bypass_response")
    candidates = record.get("candidates")
    if not isinstance(raw, dict) or not isinstance(candidates, list):
        raise ValueError("thinking probe raw capture is missing")
    canonical = json.dumps(raw, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"))
    return {**record, "_candidate_snapshot": deepcopy(candidates),
            "_raw_snapshot": deepcopy(raw),
            "_raw_canonical_before": canonical,
            "_raw_sha256_before": digest(canonical.encode())}


def single_model_decision(record, predicted_index=None):
    candidates = record["candidates"]
    before = record.get("_candidate_snapshot")
    if not isinstance(before, list):
        raise RuntimeError("thinking probe pre-inference candidate snapshot is missing")
    permitted = {index for index, _ in record["choices"]}
    if predicted_index is None:
        chosen, reason = 0, "reasoning_fallback"
    elif type(predicted_index) is not int or predicted_index not in permitted:
        raise ValueError("thinking prediction is not a permitted source index")
    elif predicted_index == 0:
        chosen, reason = 0, "parent_selected"
    elif proposal_allowed(record["source"], candidates[0], candidates[predicted_index]):
        chosen, reason = predicted_index, "thinking_proposal"
    else:
        chosen, reason = 0, "protected_or_unaligned"
    ordered = candidates if chosen == 0 else reorder_candidates(candidates, chosen)
    expected = [candidates[chosen], *candidates[:chosen], *candidates[chosen + 1:]]
    if (candidates != before or len(ordered) != len(candidates)
            or any(left is not right for left, right in zip(ordered, expected))):
        raise RuntimeError("thinking decision changed candidate objects")
    raw = record["raw_bypass_response"]
    raw_before = record.get("_raw_snapshot")
    raw_canonical_before = record.get("_raw_canonical_before")
    raw_sha_before = record.get("_raw_sha256_before")
    if (not isinstance(raw_before, dict) or not isinstance(raw_canonical_before, str)
            or raw_sha_before != digest(raw_canonical_before.encode())):
        raise RuntimeError("thinking probe pre-inference raw snapshot is missing")
    raw_candidates = raw.get("candidates")
    if (raw != raw_before or raw.get("result") != record["source"]
            or not isinstance(raw_candidates, list) or not raw_candidates
            or raw_candidates[0].get("text") != record["source"]):
        raise RuntimeError("thinking probe raw bypass differs")
    raw_canonical_after = json.dumps(
        raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if raw_canonical_before.encode() != raw_canonical_after.encode():
        raise RuntimeError("thinking probe raw JSON bytes changed")
    return {"chosen_index": chosen, "reason": reason,
            "result": ordered[0]["text"], "candidates": ordered,
            "raw_bypass_response": raw_before,
            "raw_canonical_json_before": raw_canonical_before,
            "raw_canonical_json_after": raw_canonical_after,
            "raw_sha256_before": raw_sha_before,
            "raw_sha256_after": digest(raw_canonical_after.encode())}


def require_resources(started, backend=None, row_started=None):
    if time.monotonic() - started > PROFILE["time_limit_seconds"]:
        raise RuntimeError("thinking resource probe whole time bound exceeded")
    if rss_bytes() > PROFILE["rss_limit_bytes"]:
        raise RuntimeError("thinking resource probe RSS bound exceeded")
    if backend is not None and backend.peak() > PROFILE["mlx_peak_limit_bytes"]:
        raise RuntimeError("thinking resource probe MLX bound exceeded")
    if (row_started is not None
            and time.monotonic() - row_started > PROFILE["row_time_limit_seconds"]):
        return False
    return True


class ThinkingBackend:
    def __init__(self):
        import mlx.core as mx
        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_sampler
        from mlx_lm.utils import load
        self.mx, self.generate_step = mx, generate_step
        self.make_sampler, self.load = make_sampler, load
        mx.set_memory_limit(PROFILE["mlx_peak_limit_bytes"])
        mx.set_cache_limit(PROFILE["cache_limit_bytes"])

    def load_model(self):
        model, tokenizer = self.load(
            str(MODEL_DIR), tokenizer_config={"trust_remote_code": False})
        if digest(tokenizer.chat_template.encode()) != OFFICIAL_TEMPLATE_SHA256:
            raise RuntimeError("official chat template differs")
        model.eval()
        return model, tokenizer

    def tokens(self, tokenizer, messages):
        tokens = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=True,
            return_dict=False)
        if not isinstance(tokens, list) or any(type(token) is not int for token in tokens):
            raise RuntimeError("thinking template did not return an integer token list")
        return tokens

    def decode(self, tokenizer, tokens):
        return tokenizer.decode(tokens)

    def token_contract(self, tokenizer):
        names = ["<think>", "</think>", *(str(index) for index in range(1, 6))]
        encoded = [tokenizer.encode(name, add_special_tokens=False) for name in names]
        if any(len(value) != 1 for value in encoded):
            raise RuntimeError("thinking markers and digits must be single tokens")
        ids = [value[0] for value in encoded]
        if len(ids) != len(set(ids)):
            raise RuntimeError("thinking markers and digits must be distinct")
        eos_ids = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)
        if len(eos_ids) != 1 or eos_ids[0] != tokenizer.eos_token_id:
            raise RuntimeError("tokenizer EOS identity differs")
        newline = tokenizer.encode("\n\n", add_special_tokens=False)
        if not newline:
            raise RuntimeError("newline suffix tokenization is empty")
        return {"opener": ids[0], "closer": ids[1], "digits": ids[2:],
                "eos": tokenizer.eos_token_id, "newline": list(newline)}

    def start_generation(self, model, prompt_tokens, seed):
        mx = self.mx
        mx.random.seed(seed)
        official = self.make_sampler(temp=SAMPLER["temperature"],
                                     top_p=SAMPLER["top_p"],
                                     top_k=SAMPLER["top_k"],
                                     min_p=SAMPLER["min_p"])
        def checked(logprobs):
            mx.eval(logprobs)
            if not bool(mx.all(mx.isfinite(logprobs)).item()):
                raise RuntimeError("generation produced nonfinite pre-sampling logits")
            return official(logprobs)
        return self.generate_step(mx.array(prompt_tokens), model,
                                  max_tokens=PROFILE["max_generated_tokens"],
                                  sampler=checked)

    def score(self, model, full_tokens, digit_ids, count):
        mx = self.mx
        logits = model(mx.array([full_tokens]))[0, -1].astype(mx.float32)
        selected = logits[mx.array(digit_ids[:count])]
        mx.eval(selected)
        return selected.tolist()

    def close_generation(self, generator):
        generator.close()
        self.mx.synchronize()
        self.mx.clear_cache()

    def peak(self):
        return int(self.mx.get_peak_memory())

    def release(self):
        self.mx.synchronize()
        self.mx.clear_cache()


def select_rows(records):
    eligible = [(index, record) for index, record in enumerate(records)
                if 2 <= len(record["choices"]) <= 5
                and len(record["prompt_tokens"]) <= PROFILE["max_prompt_tokens"]]
    natural = [(index, row) for index, row in eligible if index < 64]
    singles = [(index, row) for index, row in eligible if 64 <= index < 96]
    sentences = [(index, row) for index, row in eligible if 96 <= index < 128]
    if len(natural) < 2 or not singles or not sentences:
        raise RuntimeError("fewer than four source-only resource rows qualify")
    longest = max(natural, key=lambda item: (len(item[1]["prompt_tokens"]), -item[0]))
    first_other = next(item for item in natural if item[0] != longest[0])
    selected = [longest, first_other, singles[0], sentences[0]]
    if len({index for index, _ in selected}) != 4:
        raise RuntimeError("thinking resource selection is not distinct")
    return selected


def run_reasoning_row(backend, model, tokenizer, contract, record, started):
    row_started = time.monotonic()
    seed = seed_for(record["source"], record["context"])
    emitted, state, timed_out = [], None, False
    generator = backend.start_generation(model, record["prompt_tokens"], seed)
    try:
        while True:
            if not require_resources(started, backend, row_started):
                timed_out = True
                state = {"terminal": True, "valid": False, "reason": "row_timeout"}
                break
            try:
                token, _ = next(generator)
            except StopIteration:
                break
            require_resources(started, backend)
            emitted.append(int(token))
            if not require_resources(started, backend, row_started):
                timed_out = True
                state = {"terminal": True, "valid": False, "reason": "row_timeout"}
                break
            state = inspect_reasoning(
                emitted, contract["opener"], contract["closer"], contract["eos"],
                lambda values: backend.decode(tokenizer, values))
            if state["terminal"]:
                break
    finally:
        backend.close_generation(generator)
    if not require_resources(started, backend, row_started):
        timed_out = True
        state = {"terminal": True, "valid": False, "reason": "row_timeout"}
    if state is None or not state.get("terminal"):
        state = {"terminal": True, "valid": False, "reason": "token_limit_incomplete"}
    prediction, logits, probabilities = None, [], []
    if state["valid"] and not timed_out:
        full_tokens = record["prompt_tokens"] + emitted + contract["newline"]
        if not require_resources(started, backend, row_started):
            timed_out = True
            state = {"terminal": True, "valid": False, "reason": "row_timeout"}
        else:
            logits = backend.score(
                model, full_tokens, contract["digits"], len(record["choices"]))
        if not require_resources(started, backend, row_started):
            timed_out = True
            state = {"terminal": True, "valid": False, "reason": "row_timeout"}
            logits = []
        if (not timed_out and (len(logits) != len(record["choices"])
                or not all(type(value) in (int, float) and math.isfinite(value)
                           for value in logits))):
            raise RuntimeError("final candidate logits are malformed")
        if not timed_out:
            high = max(logits)
            weights = [math.exp(value - high) for value in logits]
            total = sum(weights)
            probabilities = [value / total for value in weights]
            position = max(range(len(logits)), key=lambda index: (logits[index], -index))
            prediction = record["choices"][position][0]
    decision = single_model_decision(record, prediction)
    return {"seed": seed, "prompt_tokens": record["prompt_tokens"],
            "emitted_tokens": emitted, "prompt_token_count": len(record["prompt_tokens"]),
            "generated_token_count": len(emitted), "reasoning_valid": state["valid"],
            "completion_reason": state["reason"], "reasoning_body": state.get("body"),
            "digit_logits": logits, "digit_probabilities": probabilities,
            "prediction_source_index": prediction, "decision": decision,
            "latency_seconds": time.monotonic() - row_started,
            "upstream_may_prefetch_one_unused_token": True}


def run(args, backend_factory=ThinkingBackend):
    started = time.monotonic()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.plan_sha256 != PLAN_SHA256 or digest(args.plan.read_bytes()) != PLAN_SHA256:
        raise ValueError("thinking context plan differs")
    inputs = load_inputs(args.data, args.manifest_sha256,
                         args.lineage_report, args.lineage_sha256)
    if args.manifest_sha256 != PILOT_MANIFEST_SHA256 or args.lineage_sha256 != LINEAGE_SHA256:
        raise ValueError("thinking resource data identity differs")
    runner_raw = Path(__file__).read_bytes()
    before, _, _ = validate_frozen_sources(
        inputs, data_dir=args.data, manifest_sha=args.manifest_sha256,
        lineage_path=args.lineage_report, lineage_sha=args.lineage_sha256,
        runner_sha=RESOURCE_HELPER_SHA256, plan_path=SELECTION_PLAN)
    before.update({path_key(args.plan): PLAN_SHA256,
                   "scripts/probe_thinking_context_resources.py": digest(runner_raw),
                   "scripts/final_choice_policy.py": FINAL_POLICY_SHA256})
    before.update(package_sources())
    versions = require_versions()
    verify_hashes(before)
    require_resources(started)
    system = system_message()
    source_records = [snapshot_raw_record(prompt_record(item, system))
                      for item in inputs["rows"]]
    backend = backend_factory()
    model, tokenizer = backend.load_model()
    try:
        require_resources(started, backend)
        contract = backend.token_contract(tokenizer)
        records = []
        for record in source_records:
            tokens = backend.tokens(tokenizer, record["messages"])
            decoded = backend.decode(tokenizer, tokens)
            if "<think>\n\n</think>" in decoded or "<think></think>" in decoded:
                raise RuntimeError("thinking template inserted a closed empty block")
            records.append({**record, "prompt_tokens": tokens})
        selected = select_rows(records)
        outcomes = []
        for source_index, record in selected:
            outcome = run_reasoning_row(
                backend, model, tokenizer, contract, record, started)
            decision = outcome.pop("decision")
            outcomes.append({"id": record["id"], "role": record["role"],
                             "source_row_index": source_index,
                             "choices": record["choices"], **outcome,
                             "chosen_index": decision["chosen_index"],
                             "result": decision["result"],
                             "decision_reason": decision["reason"],
                             "candidates": deepcopy(decision["candidates"]),
                             "raw_bypass_response": decision["raw_bypass_response"],
                             "raw_canonical_json_before":
                                 decision["raw_canonical_json_before"],
                             "raw_canonical_json_after":
                                 decision["raw_canonical_json_after"],
                             "raw_sha256_before": decision["raw_sha256_before"],
                             "raw_sha256_after": decision["raw_sha256_after"],
                             "candidate_objects_and_raw_preserved": True,
                             "labels_read": False})
            print(json.dumps({"stage": "thinking_resource_row",
                              "ordinal": len(outcomes),
                              "source_row_index": source_index,
                              "prompt_tokens": outcome["prompt_token_count"],
                              "generated_tokens": outcome["generated_token_count"],
                              "completion_reason": outcome["completion_reason"],
                              "elapsed_seconds": time.monotonic() - started},
                             sort_keys=True), flush=True)
            require_resources(started, backend)
    finally:
        del model, tokenizer
        gc.collect()
        backend.release()
    completed = sum(row["reasoning_valid"] for row in outcomes)
    feasible = completed >= 1 and len(outcomes) == PROFILE["rows"]
    after, _, _ = validate_frozen_sources(
        inputs, data_dir=args.data, manifest_sha=args.manifest_sha256,
        lineage_path=args.lineage_report, lineage_sha=args.lineage_sha256,
        runner_sha=RESOURCE_HELPER_SHA256, plan_path=SELECTION_PLAN)
    after.update({path_key(args.plan): PLAN_SHA256,
                  "scripts/probe_thinking_context_resources.py": digest(runner_raw),
                  "scripts/final_choice_policy.py": FINAL_POLICY_SHA256})
    after.update(package_sources())
    versions_after = require_versions()
    verify_hashes(after)
    if (before != after or versions != versions_after
            or file_digest(Path(__file__)) != digest(runner_raw)):
        raise RuntimeError("thinking resource inputs changed during execution")
    require_resources(started, backend)
    args.output.mkdir(parents=True)
    (args.output / "source.py").write_bytes(runner_raw)
    report = {"schema": "thinking-context-resource-probe-v1",
              "created_at": datetime.now(timezone.utc).isoformat(),
              "completed": True, "resource_feasible": feasible,
              "scope": "four-row TRAIN resource probe; no labels, accuracy, or deployment",
              "profile": PROFILE, "sampler": SAMPLER, "rows": len(outcomes),
              "valid_completed_reasoning_rows": completed, "outcomes": outcomes,
              "plan_sha256": PLAN_SHA256,
              "pilot_manifest_sha256": args.manifest_sha256,
              "lineage_report_sha256": args.lineage_sha256,
              "runtime_version": RUNTIME_VERSION, "base_revision": MODEL_REVISION,
              "base_model_sha256": MODEL_WEIGHT_SHA256,
              "versions": versions, "versions_after": versions_after,
              "source_hashes": before,
              "source_hashes_after": after,
              "elapsed_seconds": time.monotonic() - started,
              "rss_peak_bytes": rss_bytes(), "mlx_peak_bytes": backend.peak(),
              "selection_uses_source_and_actual_tokens_only": True,
              "expected_or_labels_read": False, "accuracy_measured": False,
              "adapter_loaded": False, "training_run": False,
              "development_data_read": False, "final_data_read": False,
              "deployment_eligible": False, "deployed": False}
    report_raw = (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    require_resources(started, backend)
    (args.output / "report.json").write_bytes(report_raw)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=PILOT_DATA)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--lineage-report", type=Path, default=LINEAGE)
    parser.add_argument("--lineage-sha256", required=True)
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
