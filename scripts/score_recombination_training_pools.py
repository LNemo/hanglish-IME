"""Score frozen TRAIN-only original and recombined sentence pools."""
from __future__ import annotations

import argparse
from collections import Counter
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

SCHEMA = "recombination-training-parent-census-v1"
DATA_SCHEMA = "recombination-ranker-data-v1"
PROOF_SCHEMA = "recombination-ranker-data-independent-verification-v1"
EXPECTED_ROWS = 512
MAX_LENGTH = 256
CHECKPOINT = ROOT / "checkpoints/sentence_ranker"
EXPECTED_MODEL = {
    "config.json": "8703d582f1e16300ab44e501e4a131b15f0a133b7b74bbcfb51e7e53009fe05a",
    "metadata.json": "8a6a1f9736fab0173003751e70c5e7ef0fa139a0e773444986ede044e0e55d14",
    "model.safetensors": "a68f3bd26733e9bc897a182f78b1decafefa3582fa82b6b2cfcaa34672412ffa",
    "special_tokens_map.json": "5d5b662e421ea9fac075174bb0688ee0d9431699900b90662acd44b2a350503a",
    "tokenizer.json": "5ee39bd824ba876371d68c5e958a8169c29b29060b8db19e6269403f3d3c7999",
    "tokenizer_config.json": "5a85ad076d4ae965ccff1888927079a924e3ffe401be44da855ff08a0ad0c45d",
    "vocab.txt": "6e886927dfcecd22029b1ba80c10a1374740259c1067fc3a28d964b7ae2d55a7",
}
PLAN = ROOT / ".omx/plans/recombination-ranker-training-v1.md"
PREPARATION_VERIFIER = ROOT / ".omx/artifacts/verify-recombination-ranker-data-v1.py"
LIMITS = {"whole_seconds": 900.0, "update_seconds": 60.0,
          "rss_bytes": 3 * 1024**3, "swap_growth_bytes": 512 * 1024**2}
SPLITS = ("fit", "inner_hold")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def path_key(path: Path) -> str:
    return str(Path(path).resolve())


def frozen_model_pins() -> dict[str, str]:
    actual = {path.name for path in CHECKPOINT.iterdir() if path.is_file()}
    require(actual == set(EXPECTED_MODEL), "frozen sentence-ranker file set differs")
    pins = {}
    for name, digest in EXPECTED_MODEL.items():
        path = CHECKPOINT / name
        require(sha256_file(path) == digest,
                f"frozen sentence-ranker artifact differs: {name}")
        pins[path_key(path)] = digest
    return pins


def iter_jsonl(path: Path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{number}") from error


def write_json(path: Path, value: dict) -> None:
    temporary = Path(path).with_name(Path(path).name + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2,
                  allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def dedupe_nonraw(values: list[str], source: str) -> list[str]:
    require(isinstance(values, list)
            and all(isinstance(value, str) and value for value in values),
            "candidate list is malformed")
    result = []
    seen = {source}
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def source_projection(row: dict) -> dict:
    expected_keys = {"id", "input", "expected", "group_id", "source_ids",
                     "component_id", "split", "parent_result",
                     "parent_candidates",
                     "original_candidates", "recombined_candidates",
                     "novel_candidates"}
    require(isinstance(row, dict) and set(row) == expected_keys,
            "prepared row fields differ")
    require(isinstance(row["id"], str) and isinstance(row["input"], str)
            and row["input"] and isinstance(row["group_id"], str)
            and isinstance(row["component_id"], str)
            and row["split"] in SPLITS
            and isinstance(row["source_ids"], list)
            and all(isinstance(value, str) for value in row["source_ids"])
            and isinstance(row["parent_result"], str)
            and isinstance(row["parent_candidates"], list)
            and all(isinstance(value, str) and value
                    for value in row["parent_candidates"])
            and row["parent_candidates"]
            and row["parent_candidates"][0] == row["parent_result"],
            "prepared source identity differs")
    original = dedupe_nonraw(row["original_candidates"], row["input"])
    recombined = dedupe_nonraw(row["recombined_candidates"], row["input"])
    novel = dedupe_nonraw(row["novel_candidates"], row["input"])
    require(set(original).issubset(recombined)
            and set(novel).issubset(recombined)
            and not set(novel).intersection(original)
            and set(recombined) == set(original).union(novel),
            "prepared original/recombined/novel partition differs")
    return {"id": row["id"], "input": row["input"], "split": row["split"],
            "component_id": row["component_id"],
            "parent_result": row["parent_result"],
            "parent_candidates": list(row["parent_candidates"]),
            "original_candidates": original, "recombined_candidates": recombined,
            "novel_candidates": novel}


def prepare_scoring(row: dict, can_score) -> dict:
    fit = {}
    for value in row["recombined_candidates"]:
        if value not in fit:
            fit[value] = bool(can_score(value))
    supported = {
        "original": bool(row["original_candidates"])
                    and all(fit[value] for value in row["original_candidates"]),
        "recombined": bool(row["recombined_candidates"])
                      and all(fit[value] for value in row["recombined_candidates"]),
    }
    score_texts = []
    for policy, key in (("original", "original_candidates"),
                        ("recombined", "recombined_candidates")):
        if supported[policy]:
            for value in row[key]:
                if value not in score_texts:
                    score_texts.append(value)
    return {"fit": fit, "supported": supported, "score_texts": score_texts}


def raw_reserved(candidates: list[str], source: str) -> list[str]:
    return dedupe_nonraw(candidates, source)[:4] + [source]


def apply_scores(row: dict, values: list[str], score_by_text: dict[str, float],
                 supported: bool) -> dict:
    if not supported:
        return {"applied": False, "result": row["parent_result"],
                "candidates": list(row["parent_candidates"])}
    require(values and all(value in score_by_text for value in values),
            "complete score domain is absent")
    ranked = sorted(values, key=lambda value: (-score_by_text[value], value))
    candidates = raw_reserved(ranked, row["input"])
    return {"applied": True, "result": candidates[0], "candidates": candidates}


def score_row(row: dict, ranker) -> dict:
    prepared = prepare_scoring(row, ranker.can_score)
    texts = prepared["score_texts"]
    scores = list(ranker.score_pairs([""] * len(texts), texts)) if texts else []
    require(len(scores) == len(texts)
            and all(type(value) in (int, float) and math.isfinite(value)
                    for value in scores), "sentence ranker scores differ")
    score_by_text = dict(zip(texts, map(float, scores)))
    original = apply_scores(row, row["original_candidates"], score_by_text,
                            prepared["supported"]["original"])
    recombined = apply_scores(row, row["recombined_candidates"], score_by_text,
                              prepared["supported"]["recombined"])
    return {"id": row["id"], "input": row["input"], "split": row["split"],
            "component_id": row["component_id"], "score_texts": texts,
            "scores": list(map(float, scores)), "original": original,
            "recombined": recombined,
            "novel_count": len(row["novel_candidates"])}


def validate_label(row: dict, source: dict) -> list[str]:
    require(row.get("id") == source["id"] and row.get("input") == source["input"]
            and row.get("split") == source["split"]
            and row.get("component_id") == source["component_id"],
            "late TRAIN label identity differs")
    expected = row.get("expected")
    require(isinstance(expected, list) and expected
            and all(isinstance(value, str) and value for value in expected),
            "TRAIN expected labels differ")
    return expected


def summarize(captures: list[dict], source_rows: list[dict], label_rows: list[dict],
              split: str) -> dict:
    selected = [(capture, source, label) for capture, source, label
                in zip(captures, source_rows, label_rows) if source["split"] == split]
    totals = Counter(); outcomes = []
    for capture, source, label in selected:
        expected = validate_label(label, source)
        original_pool = set(source["original_candidates"]) | {source["input"]}
        union_pool = set(source["recombined_candidates"]) | {source["input"]}
        original_top = capture["original"]["candidates"]
        union_top = capture["recombined"]["candidates"]
        outcome = {
            "id": source["id"],
            "has_novel": bool(source["novel_candidates"]),
            "original_supported": capture["original"]["applied"],
            "union_supported": capture["recombined"]["applied"],
            "original_covered": bool(original_pool.intersection(expected)),
            "union_covered": bool(union_pool.intersection(expected)),
            "original_top1": capture["original"]["result"] in expected,
            "original_top5": bool(set(original_top).intersection(expected)),
            "union_top1": capture["recombined"]["result"] in expected,
            "union_top5": bool(set(union_top).intersection(expected)),
        }
        outcome["parent_wrong"] = not outcome["union_top1"]
        outcome["adequacy_eligible"] = (
            outcome["has_novel"] and outcome["union_covered"]
            and outcome["union_supported"])
        for key, value in outcome.items():
            if key != "id":
                totals[key] += int(value)
        totals["rows"] += 1
        outcomes.append(outcome)
    corrections = [row["id"] for row in outcomes
                   if row["union_top1"] and not row["original_top1"]]
    losses = [row["id"] for row in outcomes
              if row["original_top1"] and not row["union_top1"]]
    top5_corrections = [row["id"] for row in outcomes
                        if row["union_top5"] and not row["original_top5"]]
    top5_losses = [row["id"] for row in outcomes
                   if row["original_top5"] and not row["union_top5"]]
    eligible_wrong = [row["id"] for row in outcomes
                      if row["adequacy_eligible"] and row["parent_wrong"]]
    return {"split": split, "counts": dict(totals),
            "correction_ids": corrections, "loss_ids": losses,
            "top5_correction_ids": top5_corrections,
            "top5_loss_ids": top5_losses,
            "adequacy_eligible_parent_wrong_ids": eligible_wrong,
            "outcomes": outcomes}


def adequacy_gate(inner: dict) -> dict:
    eligible = inner["counts"].get("adequacy_eligible", 0)
    wrong = len(inner["adequacy_eligible_parent_wrong_ids"])
    return {"passed": eligible >= 64 and wrong >= 16,
            "inner_hold_novel_union_positive_fully_supported": eligible,
            "inner_hold_parent_wrong_among_eligible": wrong,
            "minimum_rows": 64, "minimum_parent_wrong": 16,
            "interpretation": (
                "Support gate for a later bounded training experiment; it is not a "
                "statistical-power guarantee or a quality/promotion decision.")}


def validate_inputs(data_dir: Path, proof_path: Path) -> dict:
    data_dir = Path(data_dir)
    manifest_path = data_dir / "manifest.json"
    prepared_path = data_dir / "prepared.jsonl"
    capture_path = data_dir / "source-capture.jsonl"
    closure_path = data_dir / "capture.complete.json"
    require(manifest_path.is_file() and prepared_path.is_file()
            and capture_path.is_file() and closure_path.is_file()
            and Path(proof_path).is_file(), "prepared data authorization is incomplete")
    manifest_sha = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    proof = json.loads(Path(proof_path).read_text(encoding="utf-8"))
    outputs = manifest.get("outputs", {})
    require(manifest.get("schema") == DATA_SCHEMA
            and manifest.get("completed") is True
            and manifest.get("rows") == EXPECTED_ROWS
            and manifest.get("counts", {}).get("rows") == EXPECTED_ROWS
            and manifest.get("counts", {}).get("fit", 0)
                + manifest.get("counts", {}).get("inner_hold", 0) == EXPECTED_ROWS
            and manifest.get("source_only_capture_closed_before_labels") is True
            and manifest.get("labels_opened_after_capture") is True
            and manifest.get("model_loaded") is False
            and manifest.get("model_inference") is False
            and manifest.get("dev_accessed") is False
            and manifest.get("test_accessed") is False
            and manifest.get("production_changed") is False,
            "prepared data manifest contract differs")
    for name, path in (("prepared.jsonl", prepared_path),
                       ("source-capture.jsonl", capture_path),
                       ("capture.complete.json", closure_path)):
        record = outputs.get(name, {})
        require(record.get("rows") == EXPECTED_ROWS
                and record.get("sha256") == sha256_file(path),
                f"prepared output differs: {name}")
    contracts = proof.get("contracts", {})
    producer_key = path_key(ROOT / "scripts/prepare_recombination_ranker_data.py")
    edge_verifier_key = path_key(ROOT / ".omx/artifacts/verify-source-edge-recombination-v1.py")
    manifest_pins = {path_key(Path(raw) if Path(raw).is_absolute() else ROOT / raw): digest
                     for raw, digest in manifest.get("pins", {}).items()}
    require(proof.get("schema") == PROOF_SCHEMA and proof.get("verified") is True
            and proof.get("manifest_sha256") == manifest_sha
            and proof.get("prepared_sha256") == outputs["prepared.jsonl"]["sha256"]
            and proof.get("capture_sha256") == outputs["source-capture.jsonl"]["sha256"]
            and proof.get("closure_sha256") == outputs["capture.complete.json"]["sha256"]
            and proof.get("verifier_sha256") == sha256_file(PREPARATION_VERIFIER)
            and proof.get("producer_sha256") == manifest_pins.get(producer_key)
            and proof.get("edge_verifier_sha256")
                == sha256_file(Path(edge_verifier_key))
            and proof.get("counts", {}).get("rows") == EXPECTED_ROWS
            and proof.get("components") == manifest.get("counts", {}).get("components")
            and contracts == {"source_only_capture": True, "model_loaded": False,
                              "dev_opened": False, "test_opened": False},
            "prepared data independent verification differs")
    pins = {path_key(manifest_path): manifest_sha,
            path_key(prepared_path): outputs["prepared.jsonl"]["sha256"],
            path_key(capture_path): outputs["source-capture.jsonl"]["sha256"],
            path_key(closure_path): outputs["capture.complete.json"]["sha256"],
            path_key(Path(proof_path)): sha256_file(Path(proof_path)),
            path_key(PREPARATION_VERIFIER): proof["verifier_sha256"],
            edge_verifier_key: proof["edge_verifier_sha256"]}
    for raw, digest in manifest.get("pins", {}).items():
        path = Path(raw)
        path = path if path.is_absolute() else ROOT / path
        require(path.is_file() and sha256_file(path) == digest,
                f"prepared source pin differs: {raw}")
        pins[path_key(path)] = digest
    return {"manifest": manifest, "proof": proof, "prepared": prepared_path,
            "pins": pins, "counts": manifest["counts"]}


def run(data_dir: Path, proof_path: Path, output: Path, *, ranker_factory=None) -> dict:
    authorization = validate_inputs(data_dir, proof_path)  # model-free authorization
    from scripts import preflight_pko_seq2seq as preflight
    from hanglish.models.sentence_ranker import SentenceRanker
    import torch

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    baseline_swap = preflight.system_swap_bytes()
    captures = []
    source_rows = []
    pins = dict(authorization["pins"])
    tracked = [Path(__file__).resolve(), PLAN,
               ROOT / "scripts/preflight_pko_seq2seq.py",
               ROOT / "hanglish/models/sentence_ranker.py"]
    pins.update({path_key(path): sha256_file(path) for path in tracked})
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)

    class CensusWatchdog(preflight.ResourceWatchdog):
        def _monitor(self):
            while not self.stop_event.wait(self.interval_s):
                try:
                    preflight.append_jsonl(self.trace, self._sample("watchdog"), self.lock)
                except BaseException as error:
                    self.failure = f"{type(error).__name__}: {error}"
                    if not (self.trace.parent / "report.json").exists():
                        preflight.write_json(self.trace.parent / "report.json", {
                            "schema": SCHEMA, "completed": False,
                            "error": self.failure, "captured_rows": len(captures),
                            "dev_accessed": False, "test_accessed": False,
                            "production_changed": False})
                    os.kill(os.getpid(), signal.SIGUSR1)
                    if not self.stop_event.wait(5.0):  # pragma: no cover
                        os._exit(124)
                    return

    try:
        with CensusWatchdog(started, baseline_swap, output / "resources.jsonl",
                            limits=LIMITS, interval_s=5.0) as watchdog:
            model_pins = frozen_model_pins()  # bind the parent before deserialization
            pins.update(model_pins)
            config = json.loads((CHECKPOINT / "config.json").read_text())
            metadata = json.loads((CHECKPOINT / "metadata.json").read_text())
            require(config.get("max_position_embeddings", 0) >= MAX_LENGTH
                    and metadata.get("max_length") == 96,
                    "sentence ranker length contract differs")
            with watchdog.operation("load_sentence_ranker", update=True):
                original_ranker = (ranker_factory or SentenceRanker.load)(CHECKPOINT)
                ranker = SentenceRanker(
                    original_ranker.model, original_ranker.tokenizer,
                    max_length=MAX_LENGTH, batch_size=original_ranker.batch_size,
                    metadata=original_ranker.metadata)
            require(ranker.model is original_ranker.model
                    and ranker.tokenizer is original_ranker.tokenizer
                    and ranker.max_length == MAX_LENGTH,
                    "256 wrapper does not share the frozen ranker")
            with (output / "capture.jsonl").open("x", encoding="utf-8") as stream:
                for raw in iter_jsonl(authorization["prepared"]):
                    source = source_projection(raw)  # expected is never projected
                    with watchdog.operation(f"score:{source['id']}", update=True):
                        capture = score_row(source, ranker)
                    source_rows.append(source)
                    captures.append(capture)
                    stream.write(json.dumps(capture, ensure_ascii=False, sort_keys=True,
                                            allow_nan=False) + "\n")
                    stream.flush()
                    if len(captures) % 25 == 0:
                        print(json.dumps({"captured": len(captures),
                                          "rows": EXPECTED_ROWS,
                                          "elapsed_s": time.monotonic() - started}),
                              flush=True)
                os.fsync(stream.fileno())
            require(len(captures) == EXPECTED_ROWS, "prepared row denominator differs")
            capture_sha = sha256_file(output / "capture.jsonl")
            closure = {"schema": SCHEMA + "-source-closure",
                       "rows": EXPECTED_ROWS, "capture_sha256": capture_sha,
                       "source_only": True, "targets_joined": False,
                       "score_context": "", "input_truncation": False}
            write_json(output / "capture.complete.json", closure)
            require(sha256_file(authorization["prepared"])
                    == authorization["manifest"]["outputs"]["prepared.jsonl"]["sha256"],
                    "prepared TRAIN labels changed before late join")
            label_rows = list(iter_jsonl(authorization["prepared"]))
            require(len(label_rows) == EXPECTED_ROWS, "TRAIN label denominator differs")
            metrics = {split: summarize(captures, source_rows, label_rows, split)
                       for split in SPLITS}
            gate = adequacy_gate(metrics["inner_hold"])
            for path, digest in pins.items():
                require(Path(path).is_file() and sha256_file(Path(path)) == digest,
                        f"census pin changed: {path}")
            final_resources = watchdog._sample("final")
        report = {
            "schema": SCHEMA, "completed": True, "scope": "TRAIN-only",
            "counts": authorization["counts"], "metrics": metrics,
            "adequacy_gate": gate,
            "capture_sha256": capture_sha,
            "closure_sha256": sha256_file(output / "capture.complete.json"),
            "pins": dict(sorted(pins.items())), "model_pins": model_pins,
            "model": {"checkpoint": path_key(CHECKPOINT),
                      "tuned_max_length": 96, "diagnostic_max_length": MAX_LENGTH,
                      "max_position_embeddings": config["max_position_embeddings"],
                      "shared_model_and_tokenizer": True,
                      "score_context": "", "input_truncation": False},
            "resources": {"limits": LIMITS,
                          "baseline_swap_bytes": baseline_swap,
                          "final": final_resources,
                          "trace_sha256": sha256_file(output / "resources.jsonl")},
            "targets_joined_after_closed_capture": True,
            "labels_never_passed_to_model": True,
            "dev_accessed": False, "test_accessed": False,
            "training_run": False, "quality_or_promotion_gate": False,
            "production_changed": False, "elapsed_s": time.monotonic() - started,
        }
        write_json(output / "report.json", report)
        return report
    except BaseException as error:
        if output.exists() and not (output / "report.json").exists():
            write_json(output / "report.json", {
                "schema": SCHEMA, "completed": False,
                "error": f"{type(error).__name__}: {error}",
                "captured_rows": len(captures), "dev_accessed": False,
                "test_accessed": False, "training_run": False,
                "production_changed": False})
        raise


def self_test() -> dict:
    require(raw_reserved(["임니다", "입니다"], "imnida")
            == ["임니다", "입니다", "imnida"], "raw reservation differs")
    return {"schema": SCHEMA, "self_test": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path)
    parser.add_argument("--proof", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), sort_keys=True))
    else:
        require(args.data is not None and args.proof is not None
                and args.output is not None,
                "--data, --proof, and --output are required")
        result = run(args.data, args.proof, args.output)
        print(json.dumps({"completed": result["completed"],
                          "adequacy_gate": result["adequacy_gate"],
                          "elapsed_s": result["elapsed_s"]}, sort_keys=True), flush=True)
