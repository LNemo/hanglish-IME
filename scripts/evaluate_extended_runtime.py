"""Source-only DEV/TEST evaluation of the extended separated runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_ctc_word_runtime import compare as compare_public_metrics

SCHEMA = "extended-separated-runtime-v1"
PLAN = ROOT / ".omx/plans/extended-runtime-v1.md"
EXTENDED_MODULE = ROOT / "hanglish/core/extended_separated_converter.py"
SEPARATED_MODULE = ROOT / "hanglish/core/separated_converter.py"
VERIFIER = ROOT / ".omx/artifacts/verify-extended-runtime-v1.py"
CORPUS = ROOT / "data/challenge/sentence-v3"
OFFLINE_DEV = ROOT / ".omx/artifacts/extended-sentence-ranking-dev-v2"
OFFLINE = {
    "report": "b6d41b44411bac54b384f80fccaadc7972e91bc04412adba8dcb65daac69c99a",
    "capture": "e5261a9948511c0ed882623ad178247f87a347eec75d30cc7c6061e0e8d1207b",
    "closure": "14517f2060cc53c9a175b2460a2749ebab9f7891d7564120cb4aeb104117a3db",
    "proof": "29d1bc1ec9691cca64efd96d706d0ae36e7e83919a500b5858a2e701a8543af0",
}
REFERENCE = {
    "dev": {"report": "05c15eb14ebfd5cad2381d38898fd10f4ac07578789509bf4e14ea528ab6e9f0",
            "capture": "ff8b9e2a8708654637d2129a348cd6cb25745bd47871c4939121664a2db4cada",
            "rows": 316, "top1": 262, "top5": 282},
    "test": {"report": "3800c78273dddddbd2118dceccbf9d91880976a3db993435964d2c3324b5ed3a",
             "capture": "d3f17017848e25212abd0ff3cd0c36eced7480c09667d8ce9e4d5416c552d404",
             "rows": 1000, "top1": 953, "top5": 972},
}
LIMITS = {"whole_seconds": 1800, "update_seconds": 120,
          "rss_bytes": 3 * 1024**3, "swap_growth_bytes": 512 * 1024**2}
HANGUL = re.compile(r"[가-힣ᄀ-ᇿ㄰-㆏]")
SHARED_HELPERS = (
    ROOT / "scripts/preflight_pko_seq2seq.py",
    ROOT / "scripts/evaluate_sentence_only.py",
    ROOT / "scripts/evaluate_ctc_word_runtime.py",
    ROOT / "hanglish/core/ctc_word_proposals.py",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path: Path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSONL at {path}:{number}") from error


def write_json(path: Path, value: dict) -> None:
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        stream.flush(); os.fsync(stream.fileno())


def path_key(path: Path) -> str:
    return str(Path(path).resolve())


def label_path(split: str) -> Path:
    return CORPUS / f"{split}.jsonl"


def reference_paths(split: str) -> dict[str, Path]:
    directory = ROOT / f".omx/artifacts/ctc-union-runtime-{split}-v1"
    return {"report": directory / "report.json", "capture": directory / "capture.jsonl"}


def validate_reference_report(report: dict, split: str) -> None:
    expected = REFERENCE[split]
    require(report.get("completed") is True and report.get("split") == split,
            "frozen production reference is not completed for the requested split")
    require(report.get("input_contract") == "roman_sentence_only"
            and report.get("gold_not_passed_to_predictor") is True,
            "frozen production source-only contract differs")
    require(report.get("capture_sha256") == expected["capture"],
            "frozen production report does not bind its capture")
    require(report.get("proposed", {}).get("counts") == {
        "rows": expected["rows"], "top1": expected["top1"],
        "top5": expected["top5"], "errors": 0},
        "frozen production reference counts differ")
    require(isinstance(report.get("pins"), dict) and report["pins"],
            "frozen production reference has no pins")


def verify_pins(pins: dict[str, str], *, defer: Path | None = None) -> None:
    deferred = defer.resolve() if defer is not None else None
    for raw_path, digest in pins.items():
        path = Path(raw_path)
        path = path if path.is_absolute() else ROOT / path
        if deferred is not None and path.resolve() == deferred:
            continue
        require(path.is_file() and sha256_file(path) == digest,
                f"pinned implementation/model/source differs: {raw_path}")


def load_reference_authorization(split: str) -> dict:
    paths = reference_paths(split); expected = REFERENCE[split]
    require(sha256_file(paths["report"]) == expected["report"]
            and sha256_file(paths["capture"]) == expected["capture"],
            "frozen production reference identity differs")
    report = json.loads(paths["report"].read_text())
    validate_reference_report(report, split)
    # The split corpus contains targets. It is deliberately the only deferred pin.
    deferred = label_path(split)
    require(path_key(deferred) in report["pins"],
            "frozen production report does not pin the split corpus")
    verify_pins(report["pins"], defer=deferred)
    return {"paths": paths, "report": report, "deferred_label": deferred}


def validate_source_reference(row: dict, seen: set[str]) -> tuple[str, str, dict]:
    identifier = row.get("id"); request = row.get("request")
    require(isinstance(identifier, str) and identifier not in seen,
            "reference identity is absent or duplicated")
    require(isinstance(request, dict) and request.get("context") == ""
            and set(request) == {"input", "context"},
            "reference request contract differs")
    source = request.get("input")
    require(isinstance(source, str) and len(source.split()) >= 2
            and HANGUL.search(source) is None,
            "reference source is not a complete Roman sentence")
    parent = row.get("proposed")
    require(isinstance(parent, dict) and parent.get("input") == source
            and isinstance(parent.get("candidates"), list)
            and parent["candidates"],
            "reference proposed response/source differs")
    seen.add(identifier)
    return identifier, source, parent


def verify_candidate(source: str, candidate: dict) -> None:
    require(isinstance(candidate, dict) and isinstance(candidate.get("text"), str),
            "runtime candidate text differs")
    cursor = 0; pieces = []
    for segment in candidate.get("segments", []):
        end = segment.get("input_end")
        require(segment.get("input_start") == cursor and isinstance(end, int)
                and cursor < end <= len(source)
                and segment.get("source") == source[cursor:end]
                and isinstance(segment.get("text"), str),
                "runtime candidate source coverage differs")
        cursor = end; pieces.append(segment["text"])
    require(cursor == len(source) and "".join(pieces) == candidate["text"],
            "runtime candidate text/source coverage differs")


def validate_runtime_response(source: str, response: dict) -> dict:
    require(response.get("input") == source and response.get("context") == ""
            and response.get("external_context_used") is False,
            "extended runtime violated source-only prediction")
    candidates = response.get("candidates")
    require(isinstance(candidates, list) and 1 <= len(candidates) <= 5
            and response.get("result") == candidates[0].get("text"),
            "extended runtime public candidates differ")
    for candidate in candidates:
        verify_candidate(source, candidate)
    require(any(candidate["text"] == source for candidate in candidates),
            "extended runtime omits the original source option")
    pipeline = response.get("pipeline")
    require(isinstance(pipeline, dict)
            and pipeline.get("source_only_generation") is True
            and pipeline.get("source_truncated") is False
            and isinstance(pipeline.get("pool"), list)
            and isinstance(pipeline.get("response_candidates"), list),
            "extended runtime typed pipeline is absent")
    refs = pipeline["response_candidates"]
    require(len(refs) == len(candidates), "typed response candidate count differs")
    for index, (ref, candidate) in enumerate(zip(refs, candidates)):
        require(isinstance(ref, dict) and ref.get("index") == index
                and isinstance(ref.get("candidate_id"), str)
                and ref.get("candidate") == candidate,
                "typed/public response candidate reference differs")
    extended = response.get("extended_sentence_ranking")
    require(isinstance(extended, dict), "extended runtime identity is absent")
    if extended.get("applied") is True:
        raw = candidates[-1]
        require(raw.get("text") == source and raw.get("score") is None
                and raw.get("segments") == [{"input_start": 0, "input_end": len(source),
                                              "source": source, "text": source,
                                              "kind": "raw"}],
                "applied extended ranking omits the canonical final raw option")
    return extended


def compact_record(row: dict) -> dict:
    proposed = row["proposed"]
    return {"id": row["id"], "request": row["request"],
            "parent": {"result": row["parent"].get("result"),
                       "candidates": row["parent"].get("candidates", []),
                       "version": row["parent"].get("version")},
            "proposed": {"result": proposed.get("result"),
                         "candidates": proposed.get("candidates", []),
                         "engine_ms": proposed.get("engine_ms"),
                         "version": proposed.get("version")},
            "pool_texts": sorted({item.get("text") for item in
                                  proposed["pipeline"]["pool"]
                                  if isinstance(item, dict)
                                  and isinstance(item.get("text"), str)}),
            "extended": proposed["extended_sentence_ranking"]}


def validate_extended_identity(extended: dict, model_pins: dict[str, str],
                               pins: dict[str, str]) -> dict:
    identity = extended.get("identity", {})
    shared = identity.get("shared_original_model_and_tokenizer")
    require(extended.get("schema_version") == "extended-separated-ranking-v1"
            and identity.get("experimental") is True
            and identity.get("policy") == "expanded_pool_256"
            and isinstance(shared, bool)
            and (extended.get("applied") is not True or shared is True)
            and identity.get("tuned_max_length") == 96
            and identity.get("runtime_max_length") == 256
            and isinstance(identity.get("max_position_embeddings"), int)
            and identity["max_position_embeddings"] >= 256,
            "extended runtime policy/model identity differs")
    reported_models = {
        path_key(Path(raw) if Path(raw).is_absolute() else ROOT / raw): digest
        for raw, digest in identity.get("model_artifacts", {}).items()}
    require(reported_models == model_pins,
            "response model identity differs from the loaded converter")
    reported_sources = {
        path_key(ROOT / raw): digest for raw, digest in identity.get("artifacts", {}).items()}
    require(reported_sources == {path_key(EXTENDED_MODULE): pins[path_key(EXTENDED_MODULE)],
                                 path_key(SEPARATED_MODULE): pins[path_key(SEPARATED_MODULE)]},
            "response implementation identity differs")
    return identity


def literal_signature(text: str) -> str:
    return "".join(character for character in text
                   if not ("a" <= character <= "z")
                   and HANGUL.fullmatch(character) is None)


def compare_metrics(rows: list[dict], labels: list[dict]) -> dict:
    require(all(row["request"] == {"input": label.get("input"), "context": ""}
                for row, label in zip(rows, labels)),
            "capture/label source identity differs")
    metrics = compare_public_metrics(rows, labels)
    literal_regressions = []
    pool_coverage = 0
    for row, label, before, after in zip(rows, labels,
                                        metrics["parent"]["outcomes"],
                                        metrics["proposed"]["outcomes"]):
        if (literal_signature(row["request"]["input"])
                == literal_signature(before["actual"][0])
                and literal_signature(row["request"]["input"])
                != literal_signature(after["actual"][0])):
            literal_regressions.append(row["id"])
        pool_coverage += int(bool(set(row["pool_texts"]).intersection(label["expected"])))
    return {**metrics, "literal_signature_regressions": literal_regressions,
            "full_pool_coverage": {"proposed": pool_coverage},
            "quality_gate": metrics["improvement_gate"] and not literal_regressions}


def verify_offline_identity() -> dict[str, str]:
    paths = {"report": OFFLINE_DEV / "report.json",
             "capture": OFFLINE_DEV / "capture.jsonl",
             "closure": OFFLINE_DEV / "capture.complete.json",
             "proof": OFFLINE_DEV / "independent-verification.json"}
    for name, path in paths.items():
        require(path.is_file() and sha256_file(path) == OFFLINE[name],
                f"frozen offline DEV {name} differs")
    proof = json.loads(paths["proof"].read_text())
    identity = proof.get("identity", {})
    require(proof.get("verified") is True
            and identity.get("report_sha256") == OFFLINE["report"]
            and identity.get("capture_sha256") == OFFLINE["capture"]
            and identity.get("closure_sha256") == OFFLINE["closure"],
            "offline DEV independent proof differs")
    return {path_key(path): OFFLINE[name] for name, path in paths.items()}


def compare_offline(row: dict, offline: dict) -> None:
    require(row["id"] == offline.get("id")
            and row["request"] == offline.get("request"),
            "offline/runtime source identity differs")
    expected = offline.get("policies", {}).get("expanded_pool_256", {}).get("response", {})
    observed = row["proposed"]
    require(observed.get("result") == expected.get("result")
            and observed.get("candidates") == expected.get("candidates"),
            "runtime result/candidates differ from frozen expanded-pool policy")


def qualification_identity(directory: Path | None, proof_path: Path | None,
                           current_pins: dict[str, str]) -> dict:
    require(directory is not None and proof_path is not None,
            "TEST requires a completed independently verified DEV qualification")
    report_path = Path(directory) / "report.json"
    capture_path = Path(directory) / "capture.jsonl"
    closure_path = Path(directory) / "capture.complete.json"
    report = json.loads(report_path.read_text())
    proof = json.loads(Path(proof_path).read_text())
    identity = proof.get("identity", {})
    contracts = proof.get("contracts", {})
    decision = proof.get("decision", {})
    require(report.get("schema") == SCHEMA and report.get("completed") is True
            and report.get("split") == "dev" and report.get("quality_gate") is True
            and report.get("offline_exact_parity_rows") == REFERENCE["dev"]["rows"]
            and report.get("regressions") == [] and report.get("top5_losses") == []
            and report.get("literal_signature_regressions") == []
            and report.get("capture_sha256") == sha256_file(capture_path)
            and report.get("closure_sha256") == sha256_file(closure_path),
            "runtime DEV quality qualification failed")
    require(proof.get("schema") == "extended-separated-runtime-independent-verification-v1"
            and proof.get("verified") is True and proof.get("split") == "dev"
            and identity.get("report_sha256") == sha256_file(report_path)
            and identity.get("capture_sha256") == report["capture_sha256"]
            and identity.get("closure_sha256") == report["closure_sha256"]
            and identity.get("evaluator_sha256")
            == report["pins"].get(path_key(Path(__file__).resolve()))
            and identity.get("module_sha256")
            == report["pins"].get(path_key(EXTENDED_MODULE))
            and identity.get("separated_sha256")
            == report["pins"].get(path_key(SEPARATED_MODULE))
            and identity.get("plan_sha256") == report["pins"].get(path_key(PLAN))
            and identity.get("verifier_sha256") == sha256_file(VERIFIER)
            and identity.get("model_pins") == report.get("model_pins")
            and contracts.get("model_loaded") is False
            and contracts.get("model_inference") is False
            and contracts.get("capture_closed_before_target_join") is True
            and decision.get("quality_gate") is True
            and decision.get("regressions") == []
            and decision.get("top5_losses") == []
            and decision.get("literal_signature_regressions") == []
            and decision.get("offline_exact_parity_rows") == REFERENCE["dev"]["rows"],
            "runtime DEV independent proof differs")
    for path in (Path(__file__).resolve(), EXTENDED_MODULE, SEPARATED_MODULE,
                 PLAN, VERIFIER, *SHARED_HELPERS):
        key = path_key(path)
        require(report.get("pins", {}).get(key) == current_pins.get(key),
                f"DEV/TEST implementation identity differs: {key}")
    for key, digest in report.get("model_pins", {}).items():
        require(current_pins.get(key) == digest,
                f"DEV/TEST model identity differs: {key}")
    return {"report": path_key(report_path), "report_sha256": sha256_file(report_path),
            "capture_sha256": report["capture_sha256"],
            "closure_sha256": report["closure_sha256"],
            "proof": path_key(Path(proof_path)), "proof_sha256": sha256_file(Path(proof_path))}


def _latency(rows: list[dict]) -> dict:
    values = sorted(row["proposed"].get("engine_ms") for row in rows)
    require(values and all(isinstance(value, (int, float)) and math.isfinite(value)
                           and value >= 0 for value in values),
            "extended runtime timings differ")
    return {name: values[max(0, math.ceil(q * len(values)) - 1)]
            for name, q in (("p50_ms", .5), ("p95_ms", .95), ("max_ms", 1.0))}


def run(split: str, output: Path, *, final: bool = False,
        qualification: Path | None = None, qualification_proof: Path | None = None,
        converter_factory=None) -> dict:
    require(split in REFERENCE, "unsupported split")
    require(split != "test" or final, "TEST requires --final")
    from scripts import preflight_pko_seq2seq as preflight
    import torch
    if converter_factory is None:
        from hanglish.core.extended_separated_converter import ExtendedSeparatedConverter
        converter_factory = ExtendedSeparatedConverter

    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); baseline_swap = preflight.system_swap_bytes()
    compact_rows: list[dict] = []; capture_rows = 0; offline_parity_rows = 0

    class RuntimeWatchdog(preflight.ResourceWatchdog):
        def _monitor(self):
            while not self.stop_event.wait(self.interval_s):
                try:
                    preflight.append_jsonl(self.trace, self._sample("watchdog"), self.lock)
                except BaseException as error:
                    self.failure = f"{type(error).__name__}: {error}"
                    if not (self.trace.parent / "report.json").exists():
                        preflight.write_json(self.trace.parent / "report.json", {
                            "schema": SCHEMA, "completed": False, "split": split,
                            "error": self.failure, "captured_rows": capture_rows,
                            "production_changed": False, "runtime_promotion": False})
                    os.kill(os.getpid(), signal.SIGUSR1)
                    if not self.stop_event.wait(5.0):  # pragma: no cover
                        os._exit(124)
                    return

    try:
        authorization = load_reference_authorization(split)
        offline_pins = verify_offline_identity() if split == "dev" else {}
        tracked = [Path(__file__).resolve(), EXTENDED_MODULE, SEPARATED_MODULE,
                   PLAN, VERIFIER, *SHARED_HELPERS]
        pins = {**authorization["report"]["pins"], **offline_pins,
                **{path_key(path): sha256_file(path) for path in tracked}}
        # TEST qualification is checked before the first TEST source or label is opened.
        dev_qualification = (qualification_identity(qualification, qualification_proof, pins)
                             if split == "test" else None)
        torch.set_num_threads(2); torch.set_num_interop_threads(1)
        with RuntimeWatchdog(started, baseline_swap, output / "resources.jsonl",
                             limits=LIMITS, interval_s=5.0) as watchdog:
            with watchdog.operation("load_extended_converter", update=True):
                converter = converter_factory()
                for method in ("load_models", "load_word_model", "load_growth_model"):
                    loader = getattr(converter, method, None)
                    if loader is not None:
                        loader()
            model_pins = {path_key(Path(raw) if Path(raw).is_absolute() else ROOT / raw): digest
                          for raw, digest in getattr(converter, "model_artifacts", {}).items()}
            require(model_pins, "extended converter exposed no model artifacts")
            for key, digest in model_pins.items():
                require(Path(key).is_file() and sha256_file(Path(key)) == digest,
                        f"extended converter model artifact differs: {key}")
                require(pins.get(key) == digest,
                        f"extended converter used an unpinned model/source: {key}")
            write_json(output / "pins.json", dict(sorted(pins.items())))
            capture_path = output / "capture.jsonl"
            reference_iter = iter_jsonl(authorization["paths"]["capture"])
            offline_iter = (iter_jsonl(OFFLINE_DEV / "capture.jsonl")
                            if split == "dev" else None)
            seen: set[str] = set()
            with capture_path.open("x", encoding="utf-8") as stream:
                for reference in reference_iter:
                    identifier, source, parent = validate_source_reference(reference, seen)
                    with watchdog.operation(f"convert:{identifier}", update=True):
                        request_started = time.monotonic()
                        proposed = converter.convert(source, context="", engine="neural",
                                                     mode="batch", raw=False)
                        elapsed_ms = (time.monotonic() - request_started) * 1000
                    extended = validate_runtime_response(source, proposed)
                    identity = validate_extended_identity(extended, model_pins, pins)
                    require(proposed["pipeline"] == getattr(converter, "last_pipeline", None),
                            "returned and retained typed pipelines differ")
                    record = {"id": identifier,
                              "request": {"input": source, "context": ""},
                              "parent": parent, "proposed": proposed,
                              "extended": extended,
                              "elapsed_ms": elapsed_ms}
                    if offline_iter is not None:
                        try:
                            offline_row = next(offline_iter)
                        except StopIteration as error:
                            raise ValueError("offline DEV capture ended early") from error
                        compare_offline(record, offline_row); offline_parity_rows += 1
                    compact_rows.append(compact_record(record))
                    stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    stream.flush(); capture_rows += 1
                    if capture_rows % 25 == 0:
                        print(json.dumps({"captured": capture_rows,
                                          "total": REFERENCE[split]["rows"],
                                          "elapsed_s": time.monotonic() - started}), flush=True)
                if offline_iter is not None:
                    require(next(offline_iter, None) is None,
                            "offline DEV capture has extra rows")
                os.fsync(stream.fileno())
            require(capture_rows == REFERENCE[split]["rows"],
                    "runtime capture denominator differs")
            capture_sha = sha256_file(capture_path)
            closure = {"schema": "extended-separated-runtime-source-closure-v1",
                       "split": split, "rows": capture_rows,
                       "capture_sha256": capture_sha, "source_only": True,
                       "targets_joined": False, "test_opened": split == "test"}
            write_json(output / "capture.complete.json", closure)
            closure_sha = sha256_file(output / "capture.complete.json")

            # The target-bearing split file is first opened after durable closure.
            labels = list(iter_jsonl(authorization["deferred_label"]))
            require(sha256_file(authorization["deferred_label"])
                    == authorization["report"]["pins"][path_key(authorization["deferred_label"])],
                    "target-bearing split changed before the late join")
            metrics = compare_metrics(compact_rows, labels)
            for key, digest in model_pins.items():
                require(sha256_file(Path(key)) == digest,
                        f"extended converter model artifact differs: {key}")
            verify_pins(pins)
            final_resources = watchdog._sample("final")
        report = {"schema": SCHEMA, "completed": True, "split": split,
                  "input_contract": "roman_sentence_only",
                  "source_reference": {"report_sha256": REFERENCE[split]["report"],
                                       "capture_sha256": REFERENCE[split]["capture"]},
                  "dev_qualification": dev_qualification,
                  **metrics, "capture_sha256": capture_sha,
                  "closure_sha256": closure_sha,
                  "offline_exact_parity_rows": offline_parity_rows,
                  "offline_policy": "expanded_pool_256" if split == "dev" else None,
                  "pins": dict(sorted(pins.items())), "model_pins": model_pins,
                  "parent_version": compact_rows[0]["parent"].get("version"),
                  "proposed_version": compact_rows[0]["extended"].get(
                      "identity", {}).get("runtime_version"),
                  "extended_parent_version": compact_rows[0]["extended"].get(
                      "identity", {}).get("parent_version"),
                  "single_request_conversion": True,
                  "typed_pipeline_captured": True,
                  "gold_not_passed_to_predictor": True,
                  "targets_joined_after_closed_capture": True,
                  "labels_opened_after_capture": True,
                  "latency": {"proposed": _latency(compact_rows)},
                  "literal_scope": "uppercase/digit/punctuation diagnostic; not English guarantee",
                  "resources": {"limits": LIMITS, "baseline_swap_bytes": baseline_swap,
                                "final": final_resources,
                                "trace_sha256": sha256_file(output / "resources.jsonl")},
                  "elapsed_s": time.monotonic() - started,
                  "production_changed": False, "runtime_promotion": False,
                  "test_accessed": split == "test", "fully_blind_generalization": False}
        write_json(output / "report.json", report)
        return report
    except BaseException as error:
        report_path = output / "report.json"
        if not report_path.exists():
            write_json(report_path, {"schema": SCHEMA, "completed": False, "split": split,
                "error": f"{type(error).__name__}: {error}", "captured_rows": capture_rows,
                "production_changed": False, "runtime_promotion": False,
                "test_accessed": split == "test"})
        raise


def self_test() -> dict:
    source = "API2 testreul sihaeng"
    candidate = {"text": "API2 테스트를 시행", "score": 1.0,
                 "segments": [{"input_start": 0, "input_end": len(source),
                               "source": source, "text": "API2 테스트를 시행", "kind": "neural"}]}
    raw = {"text": source, "score": None,
           "segments": [{"input_start": 0, "input_end": len(source),
                         "source": source, "text": source, "kind": "raw"}]}
    response = {"input": source, "context": "", "external_context_used": False,
                "result": candidate["text"], "candidates": [candidate, raw],
                "pipeline": {"source_only_generation": True, "source_truncated": False,
                             "pool": [candidate, raw],
                             "response_candidates": [
                                 {"index": 0, "candidate_id": "a", "candidate": candidate},
                                 {"index": 1, "candidate_id": "b", "candidate": raw}]},
                "extended_sentence_ranking": {"applied": True}}
    validate_runtime_response(source, response)
    return {"schema": SCHEMA, "self_test": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=sorted(REFERENCE))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--qualification-proof", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), sort_keys=True))
    else:
        require(args.split and args.output, "--split and --output are required")
        result = run(args.split, args.output, final=args.final,
                     qualification=args.qualification,
                     qualification_proof=args.qualification_proof)
        print(json.dumps({key: result[key] for key in (
            "completed", "split", "corrections", "regressions", "top5_losses",
            "literal_signature_regressions", "quality_gate", "elapsed_s")},
            ensure_ascii=False, sort_keys=True), flush=True)
