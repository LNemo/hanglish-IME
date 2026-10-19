"""DEV-only max-length-256 rescoring diagnostic for captured sentence pools.

The diagnostic never regenerates candidates.  It rescales only rows whose
frozen legacy response could not use the sentence ranker, and joins DEV labels
only after every source-only score and response has been durably captured.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCHEMA = "extended-sentence-ranking-dev-v2-20260921"
DEFAULT_INPUT = ROOT / ".omx/artifacts/separated-converter-dev-v1"
DEFAULT_PROOF = DEFAULT_INPUT / "independent-verification.json"
PLAN = ROOT / ".omx/plans/extended-sentence-ranking-v1.md"
CHECKPOINT = ROOT / "checkpoints/sentence_ranker"
DEV = ROOT / "data/challenge/sentence-v3/dev.jsonl"
POLICIES = ("parent_only_256", "expanded_pool_256")
MAX_LENGTH = 256
EXPECTED_ROWS = 316
EXPECTED_TARGET_ROWS = 4
LIMITS = {"whole_seconds": 300.0, "update_seconds": 60.0,
          "rss_bytes": 3 * 1024**3, "swap_growth_bytes": 512 * 1024**2}

# Filled after the upstream run and independent proof are terminal.  prepare()
# fails closed until these immutable identities are available.
INPUT_REPORT_SHA256 = "262d8baf84328659dfb2d9e37275be4d5874ccafdf8110a152e279b982853ff0"
INPUT_CAPTURE_SHA256 = "c2b5d1e3b45b6cee5fff25a9f7f358923fd5bfd2cddc064674c621778919fada"
INPUT_CLOSURE_SHA256 = "72d7640b4d900e3f70e5dfaed4f823e6cf241d1dc60e06040738b7a9be36fe1b"
INPUT_PROOF_SHA256 = "32ea1d283abc29acbf66314744e497a9798ed2c7f2fd930c309c433d3a57b169"
INPUT_PINS_SHA256 = "9e03d1f62f01febd2888f156ff349cde35d4f2cfae9e03f8a5bf8552f37cd28d"
PLAN_SHA256 = "e31fa6cac5d4f0c17598a45f52df5183127829daec0facba6d80a5a577d003f7"
STATIC_PINS = {
    ROOT / "hanglish/models/sentence_ranker.py":
        "72ed3e7223aac215d692f96890ab642cf9c1da82d25c4485c5b901ff832bd1ca",
    ROOT / "scripts/preflight_pko_seq2seq.py":
        "89bc4bd2a9875ace9eb010ca443168df4db84ed34772342dd55923e2ab7518f2",
    ROOT / ".omx/artifacts/verify-separated-converter-v1.py":
        "f479ca1631e805423301126c7cc9926b9f5e8c0c0d2ab34a69d63ca354afc57f",
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
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_json(path: Path, value: dict) -> None:
    temporary = Path(path).with_name(Path(path).name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, value: dict, lock=None) -> None:
    import contextlib
    manager = lock if lock is not None else contextlib.nullcontext()
    with manager:
        with Path(path).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, sort_keys=True) + "\n")
            stream.flush(); os.fsync(stream.fileno())


def canonical_raw(source: str) -> dict:
    return {"text": source, "score": None, "segments": [{
        "input_start": 0, "input_end": len(source), "source": source,
        "text": source, "kind": "raw"}]}


def is_dedicated_raw(candidate: dict, source: str) -> bool:
    """The exact source text is always reserved for the final raw UI slot."""
    return candidate.get("text") == source


def dedupe_pool(pool: list[dict], source: str) -> list[dict]:
    """Keep the first structural path for each non-raw candidate text."""
    require(isinstance(pool, list), "candidate pool must be a list")
    result, seen = [], set()
    for candidate in pool:
        require(isinstance(candidate, dict)
                and isinstance(candidate.get("text"), str)
                and isinstance(candidate.get("segments"), list),
                "candidate pool row is malformed")
        if is_dedicated_raw(candidate, source):
            continue
        text = candidate["text"]
        if text not in seen:
            result.append(deepcopy(candidate)); seen.add(text)
    return result


def policy_pools(row: dict) -> dict[str, list[dict]]:
    source = row.get("request", {}).get("input")
    require(isinstance(source, str) and row.get("request", {}).get("context") == "",
            "captured request is not source-only")
    require(row.get("legacy", {}).get("input") == source
            and row.get("legacy", {}).get("context") == "",
            "legacy response source differs")
    return {
        "parent_only_256": dedupe_pool(row["legacy_pipeline"]["pool"], source),
        "expanded_pool_256": dedupe_pool(row["independent"]["pipeline"]["pool"], source),
    }


def policy_text_sets(row: dict) -> dict[str, set[str]]:
    """Return coverage-only text sets without copying structural pool records."""
    source = row.get("request", {}).get("input")
    require(isinstance(source, str) and row.get("request", {}).get("context") == "",
            "captured request is not source-only")
    result = {}
    for policy, pool in (
            ("parent_only_256", row["legacy_pipeline"]["pool"]),
            ("expanded_pool_256", row["independent"]["pipeline"]["pool"])):
        require(isinstance(pool, list), "candidate pool must be a list")
        texts = set()
        for candidate in pool:
            require(isinstance(candidate, dict)
                    and isinstance(candidate.get("text"), str),
                    "candidate pool row is malformed")
            if candidate["text"] != source:
                texts.add(candidate["text"])
        result[policy] = texts
    return result


def prepare_rescore(row: dict, can_score) -> dict:
    """Determine policy eligibility and one shared per-row scoring request."""
    target = row.get("legacy", {}).get("sentence_ranking_applied") is False
    if not target:
        return {"target": False, "pools": {}, "supported": {}, "score_texts": [],
                "token_fit": {}}
    pools = policy_pools(row)
    all_texts = []
    for policy in POLICIES:
        all_texts.extend(candidate["text"] for candidate in pools[policy])
    token_fit = {}
    for text in all_texts:
        if text not in token_fit:
            token_fit[text] = bool(can_score(text))
    supported = {policy: bool(pools[policy]) and all(
        token_fit[candidate["text"]] for candidate in pools[policy])
                 for policy in POLICIES}
    score_texts, seen = [], set()
    for policy in POLICIES:
        if supported[policy]:
            for candidate in pools[policy]:
                if candidate["text"] not in seen:
                    score_texts.append(candidate["text"]); seen.add(candidate["text"])
    return {"target": True, "pools": pools, "supported": supported,
            "score_texts": score_texts, "token_fit": token_fit}


def apply_policy(legacy_response: dict, pool: list[dict], score_by_text: dict[str, float],
                 *, supported: bool, source: str) -> dict:
    """Apply only comparable 256-token scores, or preserve the legacy response."""
    if not supported:
        return {"applied": False,
                "reason": "empty_nonraw_pool" if not pool else "candidate_exceeds_256",
                "response": deepcopy(legacy_response), "rescored_pool": []}
    require(pool and set(score_by_text) >= {candidate["text"] for candidate in pool},
            "policy score mapping is incomplete")
    rescored = []
    for candidate in pool:
        score = score_by_text[candidate["text"]]
        require(type(score) in (int, float) and math.isfinite(score),
                "sentence score is malformed")
        value = deepcopy(candidate)
        value.update({"score": float(score), "score_type": "sentence_ranker_256",
                      "evaluated": True, "reason": "scored_at_256"})
        rescored.append(value)
    ranked = sorted(rescored, key=lambda candidate: (-candidate["score"], candidate["text"]))
    raw = canonical_raw(source)
    displayed = [{"text": candidate["text"], "score": round(candidate["score"], 4),
                  "segments": deepcopy(candidate["segments"])}
                 for candidate in ranked[:4]] + [raw]
    response = deepcopy(legacy_response)
    response.update({"result": displayed[0]["text"], "candidates": displayed,
                     "sentence_ranking_applied": True, "ranking_applied": True})
    return {"applied": True, "reason": "scored_full_nonraw_pool_at_256",
            "response": response, "rescored_pool": rescored}


def capture_row(row: dict, ranker) -> dict:
    source = row["request"]["input"]
    prepared = prepare_rescore(row, ranker.can_score)
    if not prepared["target"]:
        unchanged = deepcopy(row["legacy"])
        return {"id": row["id"], "request": deepcopy(row["request"]),
                "target_row": False, "score_texts": [], "scores": [],
                "policies": {policy: {"applied": False,
                    "reason": "legacy_sentence_ranking_already_applied",
                    "response": deepcopy(unchanged), "rescored_pool": []}
                    for policy in POLICIES}}
    score_texts = prepared["score_texts"]
    scores = list(ranker.score_pairs([""] * len(score_texts), score_texts))
    require(len(scores) == len(score_texts)
            and all(type(score) in (int, float) and math.isfinite(score)
                    for score in scores), "sentence ranker returned malformed scores")
    score_by_text = dict(zip(score_texts, map(float, scores)))
    policies = {policy: apply_policy(
        row["legacy"], prepared["pools"][policy], score_by_text,
        supported=prepared["supported"][policy], source=source)
        for policy in POLICIES}
    for policy in POLICIES:
        policies[policy].update({
            "pool_unique_nonraw": len(prepared["pools"][policy]),
            "unsupported_texts": [candidate["text"]
                for candidate in prepared["pools"][policy]
                if not prepared["token_fit"][candidate["text"]]],
        })
    return {"id": row["id"], "request": deepcopy(row["request"]),
            "target_row": True, "score_texts": score_texts,
            "scores": list(map(float, scores)), "policies": policies}


def literal_signature(text: str) -> str:
    """Uppercase, numeric, and punctuation diagnostic; not English preservation."""
    return "".join(character for character in text
                   if not ("a" <= character <= "z")
                   and not ("가" <= character <= "힣")
                   and not ("ᄀ" <= character <= "ᇿ")
                   and not ("㄰" <= character <= "㆏"))


def summarize_policy(captures: list[dict], labels: list[dict], policy: str,
                     structural_texts: dict[str, set[str]]) -> dict:
    require(policy in POLICIES and len(captures) == len(labels) == EXPECTED_ROWS,
            "policy summary denominator differs")
    require([row["id"] for row in captures] == [row["id"] for row in labels],
            "late DEV label identity differs")
    totals = Counter(); categories = defaultdict(Counter); outcomes = []
    for capture, label in zip(captures, labels):
        require(capture["request"] == {"input": label["input"], "context": ""},
                "late DEV source identity differs")
        policy_row = capture["policies"][policy]
        response = policy_row["response"]
        candidates = response.get("candidates", [])[:5]
        texts = [candidate["text"] for candidate in candidates]
        expected = label["expected"]
        top1 = bool(texts) and texts[0] in expected
        top5 = any(text in expected for text in texts)
        pool_texts = structural_texts[capture["id"]]
        nonraw_covered = bool(pool_texts.intersection(expected))
        full_covered = bool((pool_texts | {label["input"]}).intersection(expected))
        signature_ok = literal_signature(label["input"]) == literal_signature(
            response.get("result", ""))
        counts = {"rows": 1, "top1": int(top1), "top5": int(top5),
                  "nonraw_pool_covered": int(nonraw_covered),
                  "full_pool_with_raw_covered": int(full_covered),
                  "applied": int(policy_row["applied"]),
                  "unsupported": int(capture["target_row"] and not policy_row["applied"]),
                  "literal_signature_exact": int(signature_ok)}
        totals.update(counts); categories[label.get("category", "uncategorized")].update(counts)
        outcomes.append({"id": capture["id"], "top1": top1, "top5": top5,
                         "nonraw_pool_covered": nonraw_covered,
                         "full_pool_with_raw_covered": full_covered,
                         "applied": policy_row["applied"],
                         "literal_signature_exact": signature_ok,
                         "actual": texts, "expected": expected})
    return {"counts": dict(totals),
            "categories": {name: dict(value) for name, value in categories.items()},
            "outcomes": outcomes,
            "literal_scope": "uppercase/digit/punctuation diagnostic; not English guarantee"}


def compare_and_select(baseline: dict, policies: dict[str, dict]) -> dict:
    base = {row["id"]: row for row in baseline["outcomes"]}
    decisions = {}
    for policy in POLICIES:
        outcomes = policies[policy]["outcomes"]
        corrections = [row["id"] for row in outcomes
                       if not base[row["id"]]["top1"] and row["top1"]]
        regressions = [row["id"] for row in outcomes
                       if base[row["id"]]["top1"] and not row["top1"]]
        top5_losses = [row["id"] for row in outcomes
                       if base[row["id"]]["top5"] and not row["top5"]]
        literal_regressions = [row["id"] for row in outcomes
                               if base[row["id"]]["literal_signature_exact"]
                               and not row["literal_signature_exact"]]
        decisions[policy] = {"correction_ids": corrections,
                             "regression_ids": regressions,
                             "top5_loss_ids": top5_losses,
                             "literal_signature_regression_ids": literal_regressions,
                             "runtime_candidate_gate": bool(corrections)
                                and not regressions and not top5_losses
                                and not literal_regressions}
    passing = [policy for policy in POLICIES
               if decisions[policy]["runtime_candidate_gate"]]
    selected = None
    if passing:
        selected = max(passing, key=lambda policy: (
            policies[policy]["counts"]["top1"],
            policies[policy]["counts"]["top5"],
            policy == "parent_only_256"))
    return {"policies": decisions, "selected_for_runtime_evaluation": selected,
            "runtime_promotion": False,
            "selection_is_dev_only": True}


def validate_input(input_dir: Path, proof_path: Path) -> tuple[list[dict], dict]:
    require(all(value != "PENDING" for value in (
        INPUT_REPORT_SHA256, INPUT_CAPTURE_SHA256, INPUT_CLOSURE_SHA256,
        INPUT_PROOF_SHA256, INPUT_PINS_SHA256)),
        "upstream separated-converter run is not frozen")
    paths = {"report": input_dir / "report.json",
             "capture": input_dir / "capture.jsonl",
             "closure": input_dir / "capture.complete.json", "proof": proof_path,
             "pins": input_dir / "pins.json"}
    expected = {"report": INPUT_REPORT_SHA256, "capture": INPUT_CAPTURE_SHA256,
                "closure": INPUT_CLOSURE_SHA256, "proof": INPUT_PROOF_SHA256,
                "pins": INPUT_PINS_SHA256}
    for name, path in paths.items():
        require(path.is_file() and sha256_file(path) == expected[name],
                f"frozen upstream {name} differs")
    report = json.loads(paths["report"].read_text())
    closure = json.loads(paths["closure"].read_text())
    proof = json.loads(paths["proof"].read_text())
    frozen_pins = json.loads(paths["pins"].read_text())
    require(report.get("schema") == "separated-converter-dev-v1"
            and report.get("completed") is True
            and report.get("split") == "dev"
            and report.get("gold_not_passed_to_predictor") is True
            and report.get("targets_joined_after_closed_capture") is True
            and report.get("test_opened") is False
            and report.get("production_changed") is False
            and report.get("runtime_promotion") is False
            and report.get("capture_sha256") == INPUT_CAPTURE_SHA256
            and report.get("closure_sha256") == INPUT_CLOSURE_SHA256,
            "upstream report does not bind its capture")
    require(closure == {"rows": EXPECTED_ROWS,
                        "capture_sha256": INPUT_CAPTURE_SHA256,
                        "source_only": True, "targets_joined": False},
            "upstream source-only closure differs")
    identity = proof.get("identity", {})
    contracts = proof.get("contracts", {})
    require(proof.get("schema") == "separated-converter-independent-verification-v1"
            and proof.get("verified") is True
            and identity.get("report_sha256") == INPUT_REPORT_SHA256
            and identity.get("capture_sha256") == INPUT_CAPTURE_SHA256
            and identity.get("closure_sha256") == INPUT_CLOSURE_SHA256
            and identity.get("verifier_sha256")
            == STATIC_PINS[ROOT / ".omx/artifacts/verify-separated-converter-v1.py"]
            and proof.get("counts", {}).get("rows") == EXPECTED_ROWS
            and contracts.get("source_only_generation") is True
            and contracts.get("capture_closed_before_gold_join") is True
            and contracts.get("test_opened") is False
            and contracts.get("production_changed") is False,
            "upstream independent proof does not bind report and capture")
    require(frozen_pins == report.get("pins"),
            "upstream pins file differs from terminal report")
    for raw_path, digest in report.get("pins", {}).items():
        path = Path(raw_path); path = path if path.is_absolute() else ROOT / path
        if path.resolve() == DEV.resolve():
            continue
        require(path.is_file() and sha256_file(path) == digest,
                f"upstream pinned input differs: {raw_path}")
    captures = read_jsonl(paths["capture"])
    require(len(captures) == EXPECTED_ROWS
            and len({row.get("id") for row in captures}) == EXPECTED_ROWS,
            "upstream capture denominator differs")
    target_ids = [row["id"] for row in captures
                  if row.get("legacy", {}).get("sentence_ranking_applied") is False]
    require(len(target_ids) == EXPECTED_TARGET_ROWS,
            "predeclared unsupported-row count differs")
    pins = {path_key(path): expected[name] for name, path in paths.items()}
    pins[path_key(PLAN)] = PLAN_SHA256
    for path, digest in STATIC_PINS.items():
        require(path.is_file() and sha256_file(path) == digest,
                f"extended-ranking helper differs: {path}")
        pins[path_key(path)] = digest
    for raw_path, digest in report.get("pins", {}).items():
        source_path = Path(raw_path)
        source_path = source_path if source_path.is_absolute() else ROOT / source_path
        pins[path_key(source_path)] = digest
    require(PLAN.is_file() and sha256_file(PLAN) == PLAN_SHA256,
            "predeclared extended-ranking plan differs")
    return captures, {"report": report, "proof": proof, "pins": dict(sorted(pins.items())),
                      "target_ids": target_ids}


def run(input_dir: Path, proof_path: Path, output: Path, *, ranker_factory=None) -> dict:
    from scripts import preflight_pko_seq2seq as preflight
    from hanglish.models.sentence_ranker import SentenceRanker
    import torch

    class DiagnosticWatchdog(preflight.ResourceWatchdog):
        def _monitor(self):
            while not self.stop_event.wait(self.interval_s):
                try:
                    append_jsonl(self.trace, self._sample("watchdog"), self.lock)
                except BaseException as error:
                    self.failure = f"{type(error).__name__}: {error}"
                    write_json(self.trace.parent / "report.json", {
                        "schema": SCHEMA, "completed": False,
                        "error": self.failure, "captured_rows": len(results),
                        "production_changed": False, "runtime_promotion": False,
                        "test_opened": False})
                    os.kill(os.getpid(), signal.SIGUSR1)
                    if not self.stop_event.wait(5.0):  # pragma: no cover
                        os._exit(124)
                    return

    started = time.monotonic(); baseline_swap = preflight.system_swap_bytes()
    captures, authorization = validate_input(input_dir, proof_path)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    source_sha = sha256_file(Path(__file__))
    pins = {**authorization["pins"], path_key(Path(__file__)): source_sha}
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    results = []
    try:
        with DiagnosticWatchdog(started, baseline_swap, output / "resources.jsonl",
                                limits=LIMITS, interval_s=5.0) as watchdog:
            config = json.loads((CHECKPOINT / "config.json").read_text())
            metadata = json.loads((CHECKPOINT / "metadata.json").read_text())
            require(config.get("max_position_embeddings", 0) >= MAX_LENGTH
                    and metadata.get("max_length") == 96,
                    "sentence ranker position or tuned-length contract differs")
            with watchdog.operation("load_sentence_ranker", update=True):
                original = (ranker_factory or SentenceRanker.load)(CHECKPOINT)
                ranker = SentenceRanker(
                    original.model, original.tokenizer, max_length=MAX_LENGTH,
                    batch_size=original.batch_size, metadata=original.metadata)
            require(getattr(ranker.model.config, "max_position_embeddings", 0) >= MAX_LENGTH
                    and ranker.max_length == MAX_LENGTH,
                    "loaded sentence ranker cannot represent 256 positions")
            capture_path = output / "capture.jsonl"
            with capture_path.open("x", encoding="utf-8") as stream:
                for row in captures:
                    with watchdog.operation(f"score:{row['id']}", update=True):
                        result = capture_row(row, ranker)
                    results.append(result)
                    stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                    stream.flush()
                os.fsync(stream.fileno())
            capture_sha = sha256_file(capture_path)
            closure = {"rows": len(results), "capture_sha256": capture_sha,
                       "source_only": True, "targets_joined": False,
                       "score_context": "", "input_truncation": False}
            write_json(output / "capture.complete.json", closure)

            # The DEV label file is first opened after durable source-only closure.
            labels = read_jsonl(DEV)
            require(len(labels) == EXPECTED_ROWS, "DEV label denominator differs")
            structural = {policy: {} for policy in POLICIES}
            baseline_results = []
            for source_row, result in zip(captures, results):
                text_sets = policy_text_sets(source_row)
                for policy in POLICIES:
                    structural[policy][result["id"]] = text_sets[policy]
                baseline_results.append({"id": result["id"],
                    "request": result["request"], "target_row": result["target_row"],
                    "policies": {policy: {"applied": False,
                        "response": source_row["legacy"]} for policy in POLICIES}})
            baseline = summarize_policy(
                baseline_results, labels, "parent_only_256", structural["parent_only_256"])
            summaries = {policy: summarize_policy(results, labels, policy, structural[policy])
                         for policy in POLICIES}
            decision = compare_and_select(baseline, summaries)
            for raw_path, digest in pins.items():
                path = ROOT / raw_path if not Path(raw_path).is_absolute() else Path(raw_path)
                require(path.is_file() and sha256_file(path) == digest,
                        f"diagnostic input changed: {raw_path}")
            final_resources = watchdog._sample("final")
        report = {"schema": SCHEMA, "completed": True,
                  "scope": {"split": "dev", "rows": EXPECTED_ROWS,
                            "target_rows": EXPECTED_TARGET_ROWS,
                            "target_ids": authorization["target_ids"],
                            "unchanged_rows": EXPECTED_ROWS - EXPECTED_TARGET_ROWS,
                            "source_only_scoring": True, "score_context": "",
                            "input_truncation": False, "test_opened": False},
                  "policies": summaries, "baseline": baseline, "decision": decision,
                  "capture_sha256": capture_sha,
                  "closure_sha256": sha256_file(output / "capture.complete.json"),
                  "pins": pins,
                  "model": {"checkpoint": path_key(CHECKPOINT),
                            "tuned_max_length": 96,
                            "diagnostic_max_length": MAX_LENGTH,
                            "max_position_embeddings": config["max_position_embeddings"],
                            "interpretation": "extrapolation beyond tuned length; no new training"},
                  "literal_scope": "uppercase/digit/punctuation diagnostic; not English guarantee",
                  "resources": {"limits": LIMITS, "baseline_swap_bytes": baseline_swap,
                                "final": final_resources,
                                "trace_sha256": sha256_file(output / "resources.jsonl")},
                  "production_changed": False, "runtime_promotion": False,
                  "elapsed_s": time.monotonic() - started}
        write_json(output / "report.json", report)
        return report
    except BaseException as error:
        if not (output / "report.json").exists():
            write_json(output / "report.json", {"schema": SCHEMA, "completed": False,
                "error": f"{type(error).__name__}: {error}",
                "captured_rows": len(results), "production_changed": False,
                "runtime_promotion": False, "test_opened": False})
        raise


def self_test() -> dict:
    source = "testreul sihaeng"
    first = {"text": "테스트를 시행", "score": -2.0,
             "segments": [{"text": "first", "kind": "neural"}]}
    second = {"text": "테스트를 시행", "score": -3.0,
              "segments": [{"text": "second", "kind": "neural"}]}
    rows = dedupe_pool([first, second, {**canonical_raw(source),
                                       "reason": "appended_for_user",
                                       "origin": "raw_fallback"}], source)
    require(len(rows) == 1 and rows[0]["segments"][0]["text"] == "first",
            "dedupe self-test failed")
    return {"prepared": True, "model_loaded": False, "dev_opened": False,
            "test_opened": False, "max_length": MAX_LENGTH}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--proof", type=Path, default=DEFAULT_PROOF)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), sort_keys=True)); return
    if args.output is None:
        parser.error("--output is required unless --self-test is used")
    report = run(args.input, args.proof, args.output)
    print(json.dumps({"completed": report["completed"],
                      "decision": report["decision"],
                      "elapsed_s": report["elapsed_s"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
