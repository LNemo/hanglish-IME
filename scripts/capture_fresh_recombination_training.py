"""Capture latest runtime and source-edge proposals for fresh TRAIN inputs."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from collections import Counter
import hashlib
from itertools import zip_longest
import json
import os
from pathlib import Path
import re
import signal
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hanglish.core.source_edge_recombination import propose
from scripts.evaluate_extended_runtime import (
    path_key, validate_extended_identity, validate_runtime_response)


DEFAULT_SOURCES = ROOT / ".omx/artifacts/recombination-fresh-train-sources-v1"
TRAIN = ROOT / "data/challenge/sentence-v3/train.jsonl"
REFERENCE_REPORT = ROOT / ".omx/artifacts/extended-runtime-dev-v2/report.json"
PLAN = ROOT / ".omx/plans/recombination-fresh-capture-v1.md"
SELECTION_SHA256 = "15a0617e95d5c2fb0ec7ce895230794a9ac9d27f185b288eb3d138e0f91968d2"
SOURCES_MANIFEST_SHA256 = "5680aa160f99d9656b02e52e06d4218f0ca386848d7b733fdf03d2570acfa696"
SOURCES_PROOF_SHA256 = "5aa1c088cfe84c30581325b934098e41db568fd14824153c56c61d9e0519a492"
REFERENCE_REPORT_SHA256 = "64fa054990b0dc78bde0bb20e5de0ee89fa7ae910ff8fc281cb16bf607387227"
TRAIN_SHA256 = "e73ac3b3baf41075028c3b5629b4412336926faafdb8353b20c61a200bda4c42"
EXPECTED_RUNTIME_VERSION = "experimental-expanded-pool-256-1-1b1ff433b3cf"
EXPECTED_ROWS = 2048
SCHEMA = "recombination-fresh-runtime-capture-v1"
LIMITS = {"whole_seconds": 5400, "update_seconds": 120,
          "rss_bytes": 3 * 1024**3,
          "swap_growth_bytes": 512 * 1024**2}
HANGUL = re.compile(r"[가-힣ᄀ-ᇿ㄰-㆏]")
MODULES = (
    ROOT / "hanglish/core/extended_separated_converter.py",
    ROOT / "hanglish/core/separated_converter.py",
    ROOT / "hanglish/core/source_edge_recombination.py",
    ROOT / "scripts/evaluate_extended_runtime.py",
    ROOT / "scripts/preflight_pko_seq2seq.py",
)
MODULE_HASHES = {
    "hanglish/core/extended_separated_converter.py":
        "21c48b02be21d4444affd22f60d5f7e3ca0fec70e40bf410bf470869f971cb16",
    "hanglish/core/separated_converter.py":
        "bda64e5802a122aa8cc76080a71b90de4655e72fa65e3e3f2f3cf54e775e2672",
    "hanglish/core/source_edge_recombination.py":
        "c4e7d7c59c3edb8623b28dc933f429b220b4d17ff58416d47449eeaa4d78d119",
    "scripts/evaluate_extended_runtime.py":
        "c0a9fe387ed1997afa19cc82a7b9037e9fadc1ff7e6dc05868d00f86049c29f9",
    "scripts/preflight_pko_seq2seq.py":
        "89bc4bd2a9875ace9eb010ca443168df4db84ed34772342dd55923e2ab7518f2",
}
RUNTIME_RESOURCES = (
    ROOT / "data/morphology_lexicon.json",
    ROOT / "data/workbench_lexicon.json",
    ROOT / "data/common_english.txt",
    ROOT / "data/it_terms.txt",
    ROOT / "data/challenge/train_lexicon.json",
    ROOT / "hanglish/core/data/predicate_boundary_inventory.json",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSONL at {path}:{number}") from error


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True,
                  indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def append_jsonl(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True,
                            allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def ordered_unique(values):
    return list(dict.fromkeys(values))


def _relative_or_absolute(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def validate_sources(directory, *, selection_sha=SELECTION_SHA256,
                     manifest_sha=SOURCES_MANIFEST_SHA256,
                     proof_sha=SOURCES_PROOF_SHA256,
                     expected_rows=EXPECTED_ROWS,
                     partition_counts=None):
    directory = Path(directory)
    paths = {"selection": directory / "selection.jsonl",
             "manifest": directory / "manifest.json",
             "proof": directory / "independent-verification.json"}
    require(sha256_file(paths["selection"]) == selection_sha
            and sha256_file(paths["manifest"]) == manifest_sha
            and sha256_file(paths["proof"]) == proof_sha,
            "fresh TRAIN source artifact hash differs")
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    proof = json.loads(paths["proof"].read_text(encoding="utf-8"))
    require(manifest.get("schema") == "recombination-fresh-train-sources-v1"
            and manifest.get("completed") is True
            and manifest.get("input_contract") == "roman_sentence_only"
            and manifest.get("rows") == expected_rows
            and manifest.get("targets_in_selection_artifact") is False
            and manifest.get("outputs", {}).get("selection.jsonl", {}).get(
                "sha256") == selection_sha,
            "fresh TRAIN source manifest differs")
    expected_partition = (partition_counts
                          or {"fit": 1642, "inner_hold": 406})
    require(proof.get("schema")
            == "recombination-fresh-train-sources-independent-verification-v1"
            and proof.get("verified") is True
            and proof.get("manifest_sha256") == manifest_sha
            and proof.get("selection_sha256") == selection_sha
            and proof.get("rows") == expected_rows
            and proof.get("partition_counts") == expected_partition
            and proof.get("contracts", {}).get("prediction_input_fields")
            == ["id", "input"]
            and proof["contracts"].get("model_loaded") is False
            and proof["contracts"].get("dev_opened") is False
            and proof["contracts"].get("test_opened") is False,
            "fresh TRAIN source proof differs")
    selection = list(iter_jsonl(paths["selection"]))
    require(len(selection) == expected_rows
            and all(set(row) == {"id", "input"}
                    and isinstance(row["id"], str)
                    and isinstance(row["input"], str)
                    and len(row["input"].split()) >= 2
                    and HANGUL.search(row["input"]) is None
                    for row in selection)
            and len({row["id"] for row in selection}) == expected_rows,
            "fresh TRAIN source projection differs")
    selected_ids = [row["id"] for row in selection]
    partition = manifest.get("partition")
    require(selected_ids == manifest.get("selected_ids")
            and isinstance(partition, list) and len(partition) == expected_rows
            and [row.get("id") for row in partition] == selected_ids
            and Counter(row.get("split") for row in partition)
            == expected_partition
            and all(isinstance(row.get("component_id"), str)
                    for row in partition),
            "fresh TRAIN global partition differs")
    return paths, manifest, proof, selection


def _runtime_resource_paths():
    return [*RUNTIME_RESOURCES,
            *sorted((ROOT / "data/external/mecab-ko-dic").glob("*.csv"))]


def build_static_pins(reference_report, paths, *, plan_path=PLAN,
                      reference_report_path=REFERENCE_REPORT,
                      reference_sha=REFERENCE_REPORT_SHA256,
                      runtime_resources=None):
    """Validate the signed runtime model/code set before model loading."""
    expected_models = reference_report.get("model_pins")
    require(isinstance(expected_models, dict) and expected_models,
            "production reference model pins are absent")
    pins = {}
    for raw, digest in expected_models.items():
        path = Path(raw) if Path(raw).is_absolute() else ROOT / raw
        require(path.is_file() and sha256_file(path) == digest,
                f"pinned runtime model/module differs: {raw}")
        pins[path_key(path)] = digest
    inherited_sources = reference_report.get("pins", {})
    for raw, digest in inherited_sources.items():
        path = Path(raw) if Path(raw).is_absolute() else ROOT / raw
        try:
            relative = path.resolve().relative_to(ROOT)
        except ValueError:
            continue
        if len(relative.parts) >= 2 and relative.parts[0] == "hanglish" \
                and path.suffix == ".py":
            require(path.is_file() and sha256_file(path) == digest,
                    f"pinned runtime source differs: {raw}")
            pins[path_key(path)] = digest
    require(sha256_file(reference_report_path) == reference_sha,
            "extended runtime reference changed before model loading")
    pins[path_key(reference_report_path)] = reference_sha
    for relative, digest in MODULE_HASHES.items():
        path = ROOT / relative
        require(sha256_file(path) == digest,
                f"capture helper/module differs: {relative}")
        pins[path_key(path)] = digest
    tracked = [Path(__file__).resolve(), Path(plan_path),
               *(runtime_resources if runtime_resources is not None
                 else _runtime_resource_paths()), *paths.values()]
    for path in tracked:
        require(Path(path).is_file(), f"required capture input absent: {path}")
        pins[path_key(path)] = sha256_file(path)
    return dict(sorted(pins.items())), {
        path_key(Path(raw) if Path(raw).is_absolute() else ROOT / raw): digest
        for raw, digest in expected_models.items()}


def capture_runtime_one(converter, row, model_pins, pins,
                        *, response_validator=validate_runtime_response,
                        identity_validator=validate_extended_identity):
    source = row["input"]
    started = time.monotonic()
    response = converter.convert(source, context="", engine="neural",
                                 mode="batch", raw=False)
    elapsed_s = time.monotonic() - started
    extended = response_validator(source, response)
    identity = identity_validator(extended, model_pins, pins)
    require(identity.get("runtime_version") == EXPECTED_RUNTIME_VERSION
            and response.get("version") == EXPECTED_RUNTIME_VERSION,
            "extended runtime version differs")
    require(response.get("pipeline") == getattr(converter, "last_pipeline", None),
            "returned and retained runtime pipelines differ")
    request = {"input": source, "context": ""}
    runtime_record = {"id": row["id"], "request": request,
                      "proposed": response, "extended": extended,
                      "elapsed_ms": elapsed_s * 1000}
    return runtime_record, response, elapsed_s


def recombine_record(row, response):
    source = row["input"]
    request = {"input": source, "context": ""}
    started = time.monotonic()
    generated = propose(source, response["pipeline"]["pool"],
                        response["candidates"])
    return {"id": row["id"], "request": request, **generated,
            "elapsed_s": time.monotonic() - started}


def capture_one(converter, row, model_pins, pins,
                *, response_validator=validate_runtime_response,
                identity_validator=validate_extended_identity):
    runtime, response, elapsed_s = capture_runtime_one(
        converter, row, model_pins, pins,
        response_validator=response_validator,
        identity_validator=identity_validator)
    return runtime, recombine_record(row, response)


def compact_prepared(runtime_rows, source_rows, labels, partition):
    """Join targets only after both source-only captures have closed."""
    label_map = {}
    seen_labels = set()
    wanted = set(partition)
    for row in labels:
        identifier = row.get("id") if isinstance(row, dict) else None
        require(isinstance(identifier, str) and identifier not in seen_labels,
                "canonical TRAIN IDs are missing or duplicate")
        seen_labels.add(identifier)
        if identifier in wanted:
            label_map[identifier] = row
    require(set(label_map) == wanted, "selected canonical TRAIN label is missing")
    for runtime, source in zip_longest(runtime_rows, source_rows):
        require(runtime is not None and source is not None
                and runtime.get("id") == source.get("id")
                and runtime.get("request") == source.get("request"),
                "runtime/source capture identity differs")
        identifier = runtime["id"]
        require(identifier in partition, "capture ID is absent from partition")
        label = label_map[identifier]
        request = runtime["request"]
        require(label.get("split") == "train"
                and label.get("input_contract") == "roman_sentence_only"
                and label.get("input") == request.get("input")
                and request.get("context") == ""
                and isinstance(label.get("expected"), list)
                and label["expected"]
                and all(isinstance(value, str) and value
                        for value in label["expected"])
                and isinstance(label.get("group_id"), str)
                and label["group_id"]
                and isinstance(label.get("source_ids"), list)
                and all(isinstance(value, str) and value
                        for value in label["source_ids"]),
                "canonical TRAIN source identity differs")
        response = runtime["proposed"]
        original = ordered_unique(row["text"]
                                  for row in response["pipeline"]["pool"])
        parent_candidates = [row["text"] for row in response["candidates"]]
        novel = ordered_unique(row["text"] for row in source["proposals"])
        require(all(value not in original for value in novel),
                "source-edge proposal is not novel")
        assignment = partition[identifier]
        yield {"id": identifier, "input": request["input"],
               "expected": list(label["expected"]),
               "group_id": label["group_id"],
               "source_ids": list(label["source_ids"]),
               "component_id": assignment["component_id"],
               "split": assignment["split"],
               "parent_result": response["result"],
               "parent_candidates": parent_candidates,
               "original_candidates": original,
               "recombined_candidates": ordered_unique([*original, *novel]),
               "novel_candidates": novel}


class _NoopWatchdog:
    """Test-only injected watchdog surface."""
    def __enter__(self):
        return self

    def __exit__(self, *unused):
        return False

    @contextmanager
    def operation(self, unused, update=False):
        del unused, update
        yield

    def sample(self):
        return {"rss_bytes": 0, "swap_growth_bytes": 0}


def _default_watchdog(output, started, baseline_swap, captured,
                      targets_opened):
    from scripts import preflight_pko_seq2seq as preflight

    class CaptureWatchdog(preflight.ResourceWatchdog):
        def _monitor(self):
            while not self.stop_event.wait(self.interval_s):
                try:
                    preflight.append_jsonl(
                        self.trace, self._sample("watchdog"), self.lock)
                except BaseException as error:
                    self.failure = f"{type(error).__name__}: {error}"
                    report = self.trace.parent / "report.json"
                    if not report.exists():
                        preflight.write_json(report, {
                            "schema": SCHEMA, "completed": False,
                            "error": self.failure,
                            "captured_rows": captured(),
                            "targets_opened": targets_opened(),
                            "production_changed": False})
                    os.kill(os.getpid(), signal.SIGUSR1)
                    if not self.stop_event.wait(5.0):  # pragma: no cover
                        os._exit(124)
                    return

        def sample(self):
            return self._sample("progress")

    return CaptureWatchdog(started, baseline_swap, output / "resources.jsonl",
                           limits=LIMITS, interval_s=5.0)


def run(sources, output, *, converter_factory=None, watchdog_factory=None,
        runtime_setup=None, train_path=TRAIN, plan_path=PLAN,
        reference_report_path=REFERENCE_REPORT,
        selection_sha=SELECTION_SHA256,
        manifest_sha=SOURCES_MANIFEST_SHA256,
        proof_sha=SOURCES_PROOF_SHA256,
        reference_sha=REFERENCE_REPORT_SHA256,
        train_sha=TRAIN_SHA256, expected_rows=EXPECTED_ROWS,
        partition_counts=None,
        runtime_resources=None, response_validator=validate_runtime_response,
        identity_validator=validate_extended_identity):
    """Run a source-only latest-runtime capture, then late-join TRAIN labels."""
    output = Path(output)
    paths, source_manifest, _, selection = validate_sources(
        sources, selection_sha=selection_sha, manifest_sha=manifest_sha,
        proof_sha=proof_sha, expected_rows=expected_rows,
        partition_counts=partition_counts)
    expected_partition = (partition_counts
                          or {"fit": 1642, "inner_hold": 406})
    require(sha256_file(reference_report_path) == reference_sha,
            "extended runtime production reference hash differs")
    reference = json.loads(Path(reference_report_path).read_text(encoding="utf-8"))
    require(reference.get("completed") is True
            and reference.get("proposed_version") == EXPECTED_RUNTIME_VERSION,
            "extended runtime production reference differs")
    pins, expected_models = build_static_pins(
        reference, paths, plan_path=plan_path,
        reference_report_path=reference_report_path,
        reference_sha=reference_sha,
        runtime_resources=runtime_resources)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "static-pins.json", pins)
    captured = 0
    started = time.monotonic()
    baseline_swap = 0
    target_opened = False
    runtime_path = output / "runtime-capture.jsonl"
    source_path = output / "source-capture.jsonl"
    progress_path = output / "progress.jsonl"
    try:
        if runtime_setup is None:
            import torch
            from scripts.preflight_pko_seq2seq import system_swap_bytes
            torch.set_num_threads(2)
            torch.set_num_interop_threads(1)
            baseline_swap = system_swap_bytes()
        else:
            baseline_swap = runtime_setup()
        if converter_factory is None:
            from hanglish.core.extended_separated_converter import (
                ExtendedSeparatedConverter)
            converter_factory = ExtendedSeparatedConverter
        factory = watchdog_factory or _default_watchdog
        watchdog = (factory(output, started, baseline_swap, lambda: captured,
                            lambda: target_opened)
                    if watchdog_factory is None else factory())
        with watchdog:
            with watchdog.operation("load_extended_converter", update=True):
                converter = converter_factory()
                for method in ("load_models", "load_word_model",
                               "load_growth_model"):
                    loader = getattr(converter, method, None)
                    if loader is not None:
                        loader()
            actual_models = {
                path_key(Path(raw) if Path(raw).is_absolute() else ROOT / raw): digest
                for raw, digest in getattr(converter, "model_artifacts", {}).items()}
            require(actual_models == expected_models,
                    "loaded converter model/source identity differs")
            require(getattr(converter, "version", None)
                    == EXPECTED_RUNTIME_VERSION,
                    "loaded converter runtime version differs")
            with (runtime_path.open("x", encoding="utf-8") as runtime_stream,
                  source_path.open("x", encoding="utf-8") as source_stream,
                  progress_path.open("x", encoding="utf-8") as progress_stream):
                for row in selection:
                    row_started = time.monotonic()
                    with watchdog.operation("convert:" + row["id"], update=True):
                        runtime_record, response, _ = capture_runtime_one(
                            converter, row, actual_models, pins,
                            response_validator=response_validator,
                            identity_validator=identity_validator)
                    with watchdog.operation("recombine:" + row["id"], update=True):
                        source_record = recombine_record(
                            row, response)
                    append_jsonl(runtime_stream, runtime_record)
                    append_jsonl(source_stream, source_record)
                    captured += 1
                    elapsed = time.monotonic() - started
                    observation = watchdog.sample()
                    require(time.monotonic() - row_started
                            <= LIMITS["update_seconds"],
                            "complete row time bound exceeded")
                    append_jsonl(progress_stream, {
                        "captured_rows": captured, "total_rows": expected_rows,
                        "elapsed_s": elapsed,
                        "row_elapsed_ms": (time.monotonic() - row_started) * 1000,
                        "rate_rows_per_s": captured / elapsed if elapsed else None,
                        "rss_bytes": observation.get("rss_bytes"),
                        "swap_growth_bytes": observation.get(
                            "swap_growth_bytes")})
                    del runtime_record, source_record, response
                os.fsync(runtime_stream.fileno())
                os.fsync(source_stream.fileno())
                os.fsync(progress_stream.fileno())
            require(captured == expected_rows,
                    "fresh runtime capture row count differs")
            runtime_sha = sha256_file(runtime_path)
            source_sha = sha256_file(source_path)
            closure = {"schema": SCHEMA + "-source-closure",
                       "rows": captured,
                       "runtime_capture_sha256": runtime_sha,
                       "source_capture_sha256": source_sha,
                       "source_only": True, "targets_joined": False}
            write_json(output / "capture.complete.json", closure)
            closure_sha = sha256_file(output / "capture.complete.json")

            # Canonical TRAIN is first opened after both captures and their
            # durable source-only closure are complete.
            target_opened = True
            require(sha256_file(train_path) == train_sha,
                    "canonical TRAIN hash differs")
            pins[path_key(Path(train_path))] = train_sha
            partition = {row["id"]: {"component_id": row["component_id"],
                                      "split": row["split"]}
                         for row in source_manifest["partition"]}
            prepared_path = output / "prepared.jsonl"
            prepared_rows = 0
            split_counts = Counter()
            with prepared_path.open("x", encoding="utf-8") as stream:
                prepared = compact_prepared(
                    iter_jsonl(runtime_path), iter_jsonl(source_path),
                    iter_jsonl(train_path), partition)
                for row in prepared:
                    append_jsonl(stream, row)
                    prepared_rows += 1
                    split_counts[row["split"]] += 1
                os.fsync(stream.fileno())
            require(prepared_rows == expected_rows
                    and split_counts == expected_partition,
                    "prepared TRAIN partition differs")
            for raw, digest in pins.items():
                require(sha256_file(raw) == digest,
                        f"pinned runtime input changed: {raw}")
            require(sha256_file(train_path) == train_sha,
                    "canonical TRAIN changed during late join")
            write_json(output / "pins.json", dict(sorted(pins.items())))
            final_resources = watchdog.sample()
        outputs = {
            "runtime-capture.jsonl": {"sha256": runtime_sha,
                                      "rows": captured},
            "source-capture.jsonl": {"sha256": source_sha,
                                     "rows": captured},
            "capture.complete.json": {"sha256": closure_sha,
                                      "rows": captured},
            "prepared.jsonl": {"sha256": sha256_file(prepared_path),
                               "rows": prepared_rows},
            "progress.jsonl": {"sha256": sha256_file(progress_path),
                               "rows": captured},
        }
        resources_path = output / "resources.jsonl"
        if resources_path.exists():
            outputs["resources.jsonl"] = {
                "sha256": sha256_file(resources_path),
                "rows": sum(1 for _ in iter_jsonl(resources_path))}
        report = {"schema": SCHEMA, "completed": True,
                  "input_contract": "roman_sentence_only",
                  "rows": captured, "prepared_rows": prepared_rows,
                  "partition_counts": dict(split_counts),
                  "outputs": outputs, "pins": pins,
                  "model_pins": actual_models,
                  "runtime_reference": {
                      "path": path_key(Path(reference_report_path)),
                      "sha256": reference_sha},
                  "runtime_version": EXPECTED_RUNTIME_VERSION,
                  "source_only_capture": True,
                  "targets_joined_after_closed_capture": True,
                  "labels_opened_after_capture": True,
                  "single_request_conversion": True,
                  "typed_pipeline_captured": True,
                  "resources": {"limits": LIMITS,
                                "baseline_swap_bytes": baseline_swap,
                                "final": final_resources},
                  "elapsed_s": time.monotonic() - started,
                  "model_loaded": True, "model_inference": True,
                  "dev_accessed": False, "test_accessed": False,
                  "production_changed": False}
        write_json(output / "report.json", report)
        return report
    except BaseException as error:
        report_path = output / "report.json"
        if not report_path.exists():
            write_json(report_path, {
                "schema": SCHEMA, "completed": False,
                "error": f"{type(error).__name__}: {error}",
                "captured_rows": captured, "targets_opened": target_opened,
                "source_only_prefix_preserved": True,
                "dev_accessed": False, "test_accessed": False,
                "production_changed": False})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.sources, args.output)
    print(json.dumps({key: result[key] for key in
                      ("completed", "rows", "partition_counts",
                       "elapsed_s")}, ensure_ascii=False), flush=True)
