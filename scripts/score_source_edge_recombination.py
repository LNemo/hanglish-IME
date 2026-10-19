"""DEV-only 256-token scoring of bounded source-edge recombinations."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
from itertools import zip_longest
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCHEMA = "source-edge-recombination-scoring-v1"
POLICIES = ("parent_only_256", "recombined_pool_256")
MAX_LENGTH = 256
EXPECTED_ROWS = 316
RUNTIME = ROOT / ".omx/artifacts/extended-runtime-dev-v2"
DEV = ROOT / "data/challenge/sentence-v3/dev.jsonl"
CHECKPOINT = ROOT / "checkpoints/sentence_ranker"
PLAN = ROOT / ".omx/plans/source-edge-ranking-v1.md"
DEFAULT_INPUT = ROOT / ".omx/artifacts/source-edge-recombination-dev-v1"
DEFAULT_PROOF = DEFAULT_INPUT / "independent-verification.json"
PURE_VERIFIER = ROOT / ".omx/artifacts/verify-source-edge-recombination-v1.py"
LIMITS = {"whole_seconds": 900.0, "update_seconds": 60.0,
          "rss_bytes": 3 * 1024**3, "swap_growth_bytes": 512 * 1024**2}
RUNTIME_HASHES = {
    "report.json": "64fa054990b0dc78bde0bb20e5de0ee89fa7ae910ff8fc281cb16bf607387227",
    "capture.jsonl": "4e329278d18ed4ac0081be2dd2ac41261a1762e68c7a5691f778f07ce897b2f0",
    "capture.complete.json": "d8982c836832e07ba7b2bff07b01841f275154e350adfb4e232419f829ac92e1",
    "independent-verification.json": "42d16c9669af47bc3483c79eb1d171d70ec7d2daee2bbc01a198fb65c0c9ca91",
}
# Filled only after the pure proposal run and independent verification finish.
PURE_HASHES = {
    "report.json": "43bf273157242cad834b97af62b5576dfc5be71f6debfa4691f2f04b6371fb5b",
    "capture.jsonl": "f4d9cab9e9aebc00a788c71fc81dc5e8c2e732e0ec41ac3f858bf4c8f7ad8cc6",
    "capture.complete.json": "9c0e66295e7e43bf0e85212155bc0941010707ac2f110f09a07fa1bdb44398fa",
    "independent-verification.json": "4d2dc1968861b6577f8fa1cc1b2a4d6fbb698b709371d33ad2446cab50a8a30d",
    "pins.json": "45c40f36c520bff05a23d49ba92dea00b54be428f41a7d2e35e6adea3c295cc3",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_key(path: Path) -> str:
    return str(Path(path).resolve())


def iter_jsonl(path: Path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSONL at {path}:{number}") from error


def write_json(path: Path, value: dict) -> None:
    temporary = Path(path).with_name(Path(path).name + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2,
                  allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def canonical_raw(source: str) -> dict:
    return {"text": source, "score": None, "segments": [{
        "input_start": 0, "input_end": len(source), "source": source,
        "text": source, "kind": "raw"}]}


def validate_path(source: str, candidate: dict) -> None:
    require(isinstance(candidate, dict) and isinstance(candidate.get("text"), str)
            and isinstance(candidate.get("segments"), list), "candidate path is malformed")
    cursor = 0; pieces = []
    for segment in candidate["segments"]:
        end = segment.get("input_end")
        require(segment.get("input_start") == cursor and type(end) is int
                and cursor < end <= len(source)
                and segment.get("source") == source[cursor:end]
                and isinstance(segment.get("text"), str),
                "candidate source path is not complete and contiguous")
        cursor = end; pieces.append(segment["text"])
    require(cursor == len(source) and "".join(pieces) == candidate["text"],
            "candidate text differs from its structural path")


def dedupe_original(pool: list[dict], source: str) -> list[dict]:
    """Keep the first original structural path for each non-raw text."""
    require(isinstance(pool, list), "original pool must be a list")
    result = []; seen = set()
    for candidate in pool:
        validate_path(source, candidate)
        text = candidate["text"]
        if text != source and text not in seen:
            result.append(deepcopy(candidate)); seen.add(text)
    return result


def normalize_proposal(proposal: dict, source: str, inventory: dict[str, dict]) -> dict:
    validate_path(source, proposal)
    require(proposal.get("origin") == "source_edge_recombination"
            and proposal.get("selection_eligible") is False
            and proposal.get("score") is None and proposal.get("score_type") is None
            and isinstance(proposal.get("proposal_id"), str)
            and isinstance(proposal.get("edge_ids"), list)
            and len(proposal["edge_ids"]) == len(proposal["segments"]),
            "pure recombination proposal contract differs")
    require(all(edge_id in inventory for edge_id in proposal["edge_ids"])
            and [inventory[edge_id]["structural_segment"]
                 for edge_id in proposal["edge_ids"]] == proposal["segments"],
            "proposal edge references do not bind its structural path")
    value = deepcopy(proposal)
    value.update({"candidate_id": proposal["proposal_id"],
                  "path_id": proposal["proposal_id"]})
    return value


def policy_pools(runtime_row: dict, proposal_row: dict) -> dict[str, list[dict]]:
    source = runtime_row.get("request", {}).get("input")
    require(runtime_row.get("id") == proposal_row.get("id")
            and runtime_row.get("request") == proposal_row.get("request")
            and runtime_row.get("request") == {"input": source, "context": ""},
            "runtime/proposal source identity differs")
    original = dedupe_original(runtime_row["proposed"]["pipeline"]["pool"], source)
    inventory_rows = proposal_row.get("edge_inventory")
    require(isinstance(inventory_rows, list), "proposal edge inventory is absent")
    inventory = {}
    for edge in inventory_rows:
        edge_id = edge.get("edge_id") if isinstance(edge, dict) else None
        require(isinstance(edge_id, str) and edge_id not in inventory
                and isinstance(edge.get("structural_segment"), dict)
                and isinstance(edge.get("provenance"), list),
                "proposal edge inventory is malformed or duplicated")
        inventory[edge_id] = edge
    combined = deepcopy(original); seen = {candidate["text"] for candidate in original}
    for raw in proposal_row.get("proposals", []):
        proposal = normalize_proposal(raw, source, inventory)
        if proposal["text"] != source and proposal["text"] not in seen:
            combined.append(proposal); seen.add(proposal["text"])
    return {"parent_only_256": original, "recombined_pool_256": combined}


def prepare_row(runtime_row: dict, proposal_row: dict, can_score) -> dict:
    pools = policy_pools(runtime_row, proposal_row)
    has_proposals = len(pools["recombined_pool_256"]) > len(pools["parent_only_256"])
    if not has_proposals:
        return {"has_proposals": False, "pools": pools, "supported": {},
                "token_fit": {}, "score_texts": []}
    token_fit = {}
    for policy in POLICIES:
        for candidate in pools[policy]:
            if candidate["text"] not in token_fit:
                token_fit[candidate["text"]] = bool(can_score(candidate["text"]))
    supported = {policy: bool(pools[policy]) and all(
        token_fit[candidate["text"]] for candidate in pools[policy])
                 for policy in POLICIES}
    score_texts = []; seen = set()
    for policy in POLICIES:
        if supported[policy]:
            for candidate in pools[policy]:
                if candidate["text"] not in seen:
                    seen.add(candidate["text"]); score_texts.append(candidate["text"])
    return {"has_proposals": True, "pools": pools, "supported": supported,
            "token_fit": token_fit, "score_texts": score_texts}


def apply_policy(original: dict, pool: list[dict], score_by_text: dict[str, float],
                 *, supported: bool, source: str, model_id: str) -> dict:
    """Use one complete 256 score domain or preserve the exact original response."""
    if not supported:
        reason = "empty_nonraw_pool" if not pool else "candidate_exceeds_256"
        return {"applied": False, "reason": reason,
                "response": deepcopy(original), "structural_pool": deepcopy(pool)}
    require(pool and set(score_by_text) >= {candidate["text"] for candidate in pool},
            "complete policy score map is absent")
    rescored = []
    for candidate in pool:
        score = score_by_text[candidate["text"]]
        require(type(score) in (int, float) and math.isfinite(score),
                "sentence score is malformed")
        value = deepcopy(candidate)
        value.update({"score": float(score), "score_type": "sentence_ranker_256",
                      "model_id": model_id, "evaluated": True,
                      "reason": "scored_at_256", "selection_eligible": True})
        rescored.append(value)
    ranked = sorted(rescored, key=lambda candidate: (-candidate["score"],
                                                       candidate["text"]))
    raw = canonical_raw(source)
    candidates = [{"text": candidate["text"], "score": round(candidate["score"], 4),
                   "segments": deepcopy(candidate["segments"])}
                  for candidate in ranked[:4]] + [raw]
    response = deepcopy(original)
    response.update({"result": candidates[0]["text"], "candidates": candidates,
                     "sentence_ranking_applied": True, "ranking_applied": True})
    return {"applied": True, "reason": "scored_full_nonraw_pool_at_256",
            "response": response, "structural_pool": rescored}


def score_row(runtime_row: dict, proposal_row: dict, ranker, model_id: str) -> dict:
    source = runtime_row["request"]["input"]
    prepared = prepare_row(runtime_row, proposal_row, ranker.can_score)
    if not prepared["has_proposals"]:
        original = runtime_row["proposed"]
        return {"id": runtime_row["id"], "request": deepcopy(runtime_row["request"]),
                "has_proposals": False, "score_texts": [], "scores": [],
                "policies": {policy: {"applied": False, "reason": "no_novel_proposals",
                    "response": deepcopy(original),
                    "structural_pool": deepcopy(prepared["pools"][policy]),
                    "pool_unique_nonraw": len(prepared["pools"][policy]),
                    "unsupported_texts": []} for policy in POLICIES}}
    texts = prepared["score_texts"]
    scores = list(ranker.score_pairs([""] * len(texts), texts)) if texts else []
    require(len(scores) == len(texts)
            and all(type(score) in (int, float) and math.isfinite(score) for score in scores),
            "sentence ranker returned malformed scores")
    score_by_text = dict(zip(texts, map(float, scores)))
    policies = {}
    for policy in POLICIES:
        pool = prepared["pools"][policy]
        policies[policy] = apply_policy(
            runtime_row["proposed"], pool, score_by_text,
            supported=prepared["supported"][policy], source=source, model_id=model_id)
        policies[policy].update({"pool_unique_nonraw": len(pool),
            "unsupported_texts": [candidate["text"] for candidate in pool
                                  if not prepared["token_fit"][candidate["text"]]]})
    return {"id": runtime_row["id"], "request": deepcopy(runtime_row["request"]),
            "has_proposals": True, "score_texts": texts,
            "scores": list(map(float, scores)), "policies": policies}


def compact_public(response: dict) -> dict:
    """Retain only fields consumed by late metrics, never the heavy pipeline."""
    return {"result": response.get("result"),
            "candidates": deepcopy(response.get("candidates", []))}


def compact_metric_row(result: dict) -> dict:
    return {"id": result["id"], "request": deepcopy(result["request"]),
            "has_proposals": result["has_proposals"],
            "policies": {policy: {
                "applied": result["policies"][policy]["applied"],
                "response": compact_public(result["policies"][policy]["response"])}
                for policy in POLICIES}}


def literal_signature(text: str) -> str:
    return "".join(character for character in text
                   if not ("a" <= character <= "z")
                   and not ("가" <= character <= "힣")
                   and not ("ᄀ" <= character <= "ᇿ")
                   and not ("㄰" <= character <= "㆏"))


def summarize(rows: list[dict], labels: list[dict], policy: str,
              structural: dict[str, set[str]]) -> dict:
    require(policy in POLICIES and len(rows) == len(labels) == EXPECTED_ROWS
            and [row["id"] for row in rows] == [row.get("id") for row in labels],
            "metric denominator or identity differs")
    totals = Counter(); categories = defaultdict(Counter); outcomes = []
    for row, label in zip(rows, labels):
        require(row["request"] == {"input": label.get("input"), "context": ""},
                "late DEV source identity differs")
        response = row["policies"][policy]["response"]
        texts = [candidate["text"] for candidate in response.get("candidates", [])[:5]]
        expected = label.get("expected")
        require(isinstance(expected, list) and expected, "DEV expected values differ")
        top1 = bool(texts) and texts[0] in expected
        top5 = any(text in expected for text in texts)
        covered = bool(structural[row["id"]].intersection(expected))
        signature = literal_signature(label["input"]) == literal_signature(
            response.get("result", ""))
        values = {"rows": 1, "top1": int(top1), "top5": int(top5),
                  "full_pool_covered": int(covered),
                  "literal_signature_exact": int(signature),
                  "applied": int(row["policies"][policy]["applied"]),
                  "unsupported": int(row["has_proposals"]
                                     and not row["policies"][policy]["applied"])}
        totals.update(values); categories[label.get("category", "uncategorized")].update(values)
        outcomes.append({"id": row["id"], "top1": top1, "top5": top5,
                         "literal_signature_exact": signature, "actual": texts,
                         "expected": expected, "full_pool_covered": covered})
    return {"counts": dict(totals),
            "categories": {key: dict(value) for key, value in categories.items()},
            "outcomes": outcomes}


def decide(original: dict, parent: dict, recombined: dict) -> dict:
    maps = {name: {row["id"]: row for row in value["outcomes"]}
            for name, value in (("original", original), ("parent", parent),
                                ("recombined", recombined))}
    def deltas(before, after):
        return {
            "correction_ids": [key for key, row in after.items()
                               if row["top1"] and not before[key]["top1"]],
            "regression_ids": [key for key, row in after.items()
                               if not row["top1"] and before[key]["top1"]],
            "top5_loss_ids": [key for key, row in after.items()
                              if not row["top5"] and before[key]["top5"]],
            "literal_signature_regression_ids": [key for key, row in after.items()
                if before[key]["literal_signature_exact"]
                and not row["literal_signature_exact"]],
        }
    versus_original = {"parent_only_256": deltas(maps["original"], maps["parent"]),
                        "recombined_pool_256": deltas(maps["original"], maps["recombined"])}
    incremental = deltas(maps["parent"], maps["recombined"])
    for value in versus_original.values():
        value["quality_positive"] = bool(value["correction_ids"]) and not any(
            value[key] for key in ("regression_ids", "top5_loss_ids",
                                   "literal_signature_regression_ids"))
    exceeds_control = (bool(incremental["correction_ids"])
                       and not any(incremental[key] for key in (
                           "regression_ids", "top5_loss_ids",
                           "literal_signature_regression_ids"))
                       and recombined["counts"]["top1"] > parent["counts"]["top1"]
                       and recombined["counts"]["top5"] >= parent["counts"]["top5"])
    selected = ("recombined_pool_256"
                if versus_original["recombined_pool_256"]["quality_positive"]
                and exceeds_control else None)
    return {"versus_original": versus_original,
            "recombined_versus_parent_only": {**incremental,
                                               "exceeds_parent_only_control": exceeds_control},
            "selected_for_runtime_evaluation": selected,
            "runtime_promotion": False, "selection_is_dev_only": True}


def validate_inputs(input_dir: Path, proof_path: Path) -> dict:
    require(all(value != "PENDING" for value in PURE_HASHES.values()),
            "pure proposal run is not frozen and independently verified")
    pure_paths = {name: Path(input_dir) / name for name in PURE_HASHES
                  if name != "independent-verification.json"}
    pure_paths["independent-verification.json"] = Path(proof_path)
    for name, digest in PURE_HASHES.items():
        require(pure_paths[name].is_file() and sha256_file(pure_paths[name]) == digest,
                f"pure proposal identity differs: {name}")
    for name, digest in RUNTIME_HASHES.items():
        require(sha256_file(RUNTIME / name) == digest,
                f"frozen runtime identity differs: {name}")
    report = json.loads(pure_paths["report.json"].read_text())
    closure = json.loads(pure_paths["capture.complete.json"].read_text())
    proof = json.loads(pure_paths["independent-verification.json"].read_text())
    coverage = report.get("coverage", {})
    require(report.get("schema") == "source-edge-recombination-probe-v1"
            and report.get("completed") is True and report.get("rows") == EXPECTED_ROWS
            and coverage.get("rows") == EXPECTED_ROWS and coverage.get("baseline") == 286
            and coverage.get("union", 0) > coverage.get("baseline", 0)
            and coverage.get("gained_ids") and coverage.get("lost_ids") == []
            and report.get("candidate_improvement_gate") is True
            and report.get("labels_opened_after_capture") is True
            and report.get("model_loaded") is False
            and report.get("model_inference") is False
            and report.get("test_opened") is False
            and report.get("production_changed") is False
            and report.get("capture_sha256") == PURE_HASHES["capture.jsonl"]
            and report.get("closure_sha256") == PURE_HASHES["capture.complete.json"]
            and report.get("dev_labels_sha256")
            == "593e1a032b54bf4d2c541da6754cfc68daf759cdcee7f8fa1f0a746459910ac9",
            "pure proposal coverage gate or contract differs")
    require(closure == {"rows": EXPECTED_ROWS,
                        "capture_sha256": PURE_HASHES["capture.jsonl"],
                        "source_only": True, "targets_joined": False},
            "pure proposal source-only closure differs")
    identity = proof.get("identity", {}); contracts = proof.get("contracts", {})
    producer = path_key(ROOT / "scripts/probe_source_edge_recombination.py")
    module = path_key(ROOT / "hanglish/core/source_edge_recombination.py")
    producer_plan = path_key(ROOT / ".omx/plans/source-edge-recombination-v1.md")
    require(proof.get("schema") == "source-edge-recombination-independent-verification-v1"
            and proof.get("verified") is True
            and identity.get("report_sha256") == PURE_HASHES["report.json"]
            and identity.get("capture_sha256") == PURE_HASHES["capture.jsonl"]
            and identity.get("closure_sha256") == PURE_HASHES["capture.complete.json"]
            and identity.get("producer_sha256") == report.get("pins", {}).get(producer)
            and identity.get("module_sha256") == report.get("pins", {}).get(module)
            and identity.get("plan_sha256") == report.get("pins", {}).get(producer_plan)
            and identity.get("verifier_sha256") == sha256_file(PURE_VERIFIER)
            and identity.get("input") == RUNTIME_HASHES
            and proof.get("coverage") == coverage
            and proof.get("decision") == {"candidate_improvement_gate": True,
                                          "gained_ids": coverage["gained_ids"],
                                          "lost_ids": []}
            and contracts.get("model_loaded") is False
            and contracts.get("model_inference") is False
            and contracts.get("capture_closed_before_target_join") is True
            and contracts.get("source_only") is True
            and contracts.get("test_opened") is False
            and contracts.get("production_changed") is False,
            "pure proposal independent verification differs")
    pins = {}
    for name, path in pure_paths.items():
        pins[path_key(path)] = PURE_HASHES[name]
    for name, digest in RUNTIME_HASHES.items():
        pins[path_key(RUNTIME / name)] = digest
    for raw, digest in report.get("pins", {}).items():
        path = Path(raw); path = path if path.is_absolute() else ROOT / path
        require(path.is_file() and sha256_file(path) == digest,
                f"pure proposal pin differs: {raw}")
        pins[path_key(path)] = digest
    pins[path_key(PURE_VERIFIER)] = identity["verifier_sha256"]
    pins[path_key(DEV)] = report["dev_labels_sha256"]
    return {"report": report, "proof": proof, "pins": pins,
            "paths": pure_paths, "coverage": coverage,
            "dev_labels_sha256": report["dev_labels_sha256"]}


def run(input_dir: Path, proof_path: Path, output: Path, *, ranker_factory=None) -> dict:
    authorization = validate_inputs(input_dir, proof_path)  # before importing Torch
    from scripts import preflight_pko_seq2seq as preflight
    from hanglish.models.sentence_ranker import SentenceRanker
    import torch

    class ScoringWatchdog(preflight.ResourceWatchdog):
        def _monitor(self):
            while not self.stop_event.wait(self.interval_s):
                try:
                    preflight.append_jsonl(self.trace, self._sample("watchdog"), self.lock)
                except BaseException as error:
                    self.failure = f"{type(error).__name__}: {error}"
                    if not (self.trace.parent / "report.json").exists():
                        preflight.write_json(self.trace.parent / "report.json", {
                            "schema": SCHEMA, "completed": False, "error": self.failure,
                            "captured_rows": len(compact), "model_inference": True,
                            "test_opened": False, "production_changed": False})
                    os.kill(os.getpid(), signal.SIGUSR1)
                    if not self.stop_event.wait(5.0):  # pragma: no cover
                        os._exit(124)
                    return

    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); baseline_swap = preflight.system_swap_bytes()
    compact = []; structural = {policy: {} for policy in POLICIES}
    pins = {**authorization["pins"]}
    tracked = [Path(__file__).resolve(), PLAN,
               ROOT / "scripts/preflight_pko_seq2seq.py",
               ROOT / "hanglish/models/sentence_ranker.py"]
    pins.update({path_key(path): sha256_file(path) for path in tracked})
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    try:
        with ScoringWatchdog(started, baseline_swap, output / "resources.jsonl",
                             limits=LIMITS, interval_s=5.0) as watchdog:
            config = json.loads((CHECKPOINT / "config.json").read_text())
            metadata = json.loads((CHECKPOINT / "metadata.json").read_text())
            require(config.get("max_position_embeddings", 0) >= MAX_LENGTH
                    and metadata.get("max_length") == 96,
                    "sentence ranker capacity/tuned-length contract differs")
            with watchdog.operation("load_sentence_ranker", update=True):
                original = (ranker_factory or SentenceRanker.load)(CHECKPOINT)
                ranker = SentenceRanker(original.model, original.tokenizer,
                    max_length=MAX_LENGTH, batch_size=original.batch_size,
                    metadata=original.metadata)
            require(ranker.model is original.model and ranker.tokenizer is original.tokenizer
                    and ranker.max_length == MAX_LENGTH
                    and getattr(ranker.model.config, "max_position_embeddings", 0) >= MAX_LENGTH,
                    "256 wrapper does not share the frozen model/tokenizer")
            model_pins = {path_key(path): sha256_file(path) for path in CHECKPOINT.iterdir()
                          if path.is_file()}
            pins.update(model_pins)
            model_id = "sentence-ranker-256-" + model_pins[
                path_key(CHECKPOINT / "model.safetensors")][:16]
            with (output / "capture.jsonl").open("x", encoding="utf-8") as stream:
                runtime_rows = iter_jsonl(RUNTIME / "capture.jsonl")
                proposal_rows = iter_jsonl(authorization["paths"]["capture.jsonl"])
                for runtime_row, proposal_row in zip_longest(runtime_rows, proposal_rows):
                    require(runtime_row is not None and proposal_row is not None,
                            "runtime/proposal denominator differs")
                    with watchdog.operation(f"score:{runtime_row.get('id')}", update=True):
                        result = score_row(runtime_row, proposal_row, ranker, model_id)
                    for policy in POLICIES:
                        structural[policy][result["id"]] = {
                            candidate["text"]
                            for candidate in result["policies"][policy]["structural_pool"]} | {
                                runtime_row["request"]["input"]}
                    compact.append(compact_metric_row(result))
                    stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True,
                                            allow_nan=False) + "\n")
                    stream.flush()
                    if len(compact) % 25 == 0:
                        print(json.dumps({"captured": len(compact), "rows": EXPECTED_ROWS,
                                          "elapsed_s": time.monotonic() - started}), flush=True)
                os.fsync(stream.fileno())
            require(len(compact) == EXPECTED_ROWS, "scoring denominator differs")
            capture_sha = sha256_file(output / "capture.jsonl")
            closure = {"rows": EXPECTED_ROWS, "capture_sha256": capture_sha,
                       "source_only": True, "targets_joined": False,
                       "score_context": "", "input_truncation": False}
            write_json(output / "capture.complete.json", closure)
            require(sha256_file(DEV) == authorization["dev_labels_sha256"],
                    "DEV targets changed before the late join")
            labels = list(iter_jsonl(DEV))  # first opened after durable closure
            original_rows = []
            for runtime_row in iter_jsonl(RUNTIME / "capture.jsonl"):
                original_rows.append({"id": runtime_row["id"],
                    "request": runtime_row["request"], "has_proposals": False,
                    "policies": {policy: {"applied": False,
                        "response": compact_public(runtime_row["proposed"])}
                        for policy in POLICIES}})
            original_structural = {row["id"]: {
                candidate["text"] for candidate in row["proposed"]["pipeline"]["pool"]}
                | {row["request"]["input"]}
                for row in iter_jsonl(RUNTIME / "capture.jsonl")}
            original = summarize(original_rows, labels, "parent_only_256", original_structural)
            policies = {policy: summarize(compact, labels, policy, structural[policy])
                        for policy in POLICIES}
            require(original["counts"]["top1"] == 264
                    and original["counts"]["top5"] == 284
                    and original["counts"]["full_pool_covered"] == 286,
                    "frozen runtime baseline metrics differ")
            decision = decide(original, policies["parent_only_256"],
                              policies["recombined_pool_256"])
            for path, digest in pins.items():
                require(Path(path).is_file() and sha256_file(Path(path)) == digest,
                        f"scoring pin changed: {path}")
            final_resources = watchdog._sample("final")
        report = {"schema": SCHEMA, "completed": True, "split": "dev",
                  "scope": {"rows": EXPECTED_ROWS, "score_context": "",
                            "input_truncation": False, "test_opened": False,
                            "source_only_capture_before_targets": True},
                  "original": original, "policies": policies, "decision": decision,
                  "pure_coverage": authorization["coverage"],
                  "capture_sha256": capture_sha,
                  "closure_sha256": sha256_file(output / "capture.complete.json"),
                  "pins": dict(sorted(pins.items())), "model_pins": model_pins,
                  "model": {"checkpoint": path_key(CHECKPOINT),
                            "model_id": model_id, "tuned_max_length": 96,
                            "diagnostic_max_length": MAX_LENGTH,
                            "max_position_embeddings": config["max_position_embeddings"],
                            "shared_model_and_tokenizer": True,
                            "interpretation": "extrapolation beyond tuned length; no new training"},
                  "resources": {"limits": LIMITS, "baseline_swap_bytes": baseline_swap,
                                "final": final_resources,
                                "trace_sha256": sha256_file(output / "resources.jsonl")},
                  "targets_joined_after_closed_capture": True,
                  "production_changed": False, "runtime_promotion": False,
                  "elapsed_s": time.monotonic() - started}
        write_json(output / "report.json", report)
        return report
    except BaseException as error:
        if not (output / "report.json").exists():
            write_json(output / "report.json", {"schema": SCHEMA, "completed": False,
                "error": f"{type(error).__name__}: {error}",
                "captured_rows": len(compact), "test_opened": False,
                "production_changed": False, "runtime_promotion": False})
        raise


def self_test() -> dict:
    source = "imnida test"
    raw = canonical_raw(source)
    validate_path(source, raw)
    return {"schema": SCHEMA, "self_test": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--proof", type=Path, default=DEFAULT_PROOF)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), sort_keys=True))
    else:
        require(args.output is not None, "--output is required")
        result = run(args.input, args.proof, args.output)
        print(json.dumps({"completed": result["completed"],
                          "decision": result["decision"],
                          "elapsed_s": result["elapsed_s"]}, ensure_ascii=False), flush=True)
