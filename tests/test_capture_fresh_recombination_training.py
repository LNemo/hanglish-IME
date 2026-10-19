from contextlib import contextmanager
import gc
import hashlib
import json
from pathlib import Path

import pytest

from scripts import capture_fresh_recombination_training as capture


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def segment(source, start, end, text, kind="fixture"):
    return {"input_start": start, "input_end": end,
            "source": source[start:end], "text": text, "kind": kind}


def candidate(source, left, right):
    return {"text": left + " " + right, "score": 1.0,
            "segments": [segment(source, 0, 2, left),
                         segment(source, 2, 3, " ", "raw"),
                         segment(source, 3, 5, right)]}


class LiveResponse(dict):
    live = 0
    maximum = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        type(self).live += 1
        type(self).maximum = max(type(self).maximum, type(self).live)

    def __del__(self):
        type(self).live -= 1


class Converter:
    def __init__(self, model_path, fail_at=None):
        self.model_artifacts = {str(model_path): sha(model_path)}
        self.version = capture.EXPECTED_RUNTIME_VERSION
        self.calls = []
        self.loads = []
        self.fail_at = fail_at
        self.last_pipeline = None

    def load_models(self):
        self.loads.append("load_models")

    def load_word_model(self):
        self.loads.append("load_word_model")

    def load_growth_model(self):
        self.loads.append("load_growth_model")

    def convert(self, source, **kwargs):
        self.calls.append((source, kwargs))
        if self.fail_at is not None and len(self.calls) == self.fail_at:
            raise RuntimeError("planned failure")
        first = candidate(source, "가", "나")
        second = candidate(source, "카", "다")
        raw = {"text": source, "score": None,
               "segments": [segment(source, 0, len(source), source, "raw")]}
        pool = []
        for index, value in enumerate((first, second, raw)):
            identity = f"path-{len(self.calls)}-{index}"
            pool.append({"candidate_id": identity, "path_id": identity,
                         "text": value["text"], "segments": value["segments"],
                         "origin": "fixture", "origins": ["fixture"]})
        self.last_pipeline = {"pool": pool}
        extended = {"identity": {
            "runtime_version": capture.EXPECTED_RUNTIME_VERSION}}
        return LiveResponse({
            "input": source, "context": "", "result": first["text"],
            "candidates": [first, second, raw], "pipeline": self.last_pipeline,
            "version": capture.EXPECTED_RUNTIME_VERSION,
            "extended_sentence_ranking": extended})


class Watchdog:
    def __enter__(self):
        return self

    def __exit__(self, *unused):
        return False

    @contextmanager
    def operation(self, name, update=False):
        del update
        yield

    def sample(self):
        return {"rss_bytes": 1, "swap_growth_bytes": 0}


def response_validator(source, response):
    assert response["input"] == source
    return response["extended_sentence_ranking"]


def identity_validator(extended, model_pins, pins):
    assert model_pins and pins
    return extended["identity"]


def fixture(tmp_path, count=2, splits=None):
    sources = tmp_path / "sources"
    sources.mkdir()
    rows = [{"id": f"row-{index}", "input": "ab cd"}
            for index in range(count)]
    selection = sources / "selection.jsonl"
    selection.write_text("".join(json.dumps(row) + "\n" for row in rows))
    splits = splits or (["fit"] * count)
    partition = [{"id": row["id"], "component_id": f"component-{index}",
                  "split": splits[index]} for index, row in enumerate(rows)]
    manifest = {"schema": "recombination-fresh-train-sources-v1",
                "completed": True, "input_contract": "roman_sentence_only",
                "rows": count, "targets_in_selection_artifact": False,
                "selected_ids": [row["id"] for row in rows],
                "partition": partition,
                "outputs": {"selection.jsonl": {"sha256": sha(selection)}}}
    manifest_path = sources / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    proof = {"schema":
             "recombination-fresh-train-sources-independent-verification-v1",
             "verified": True, "manifest_sha256": sha(manifest_path),
             "selection_sha256": sha(selection), "rows": count,
             "partition_counts": dict(__import__("collections").Counter(splits)),
             "contracts": {"prediction_input_fields": ["id", "input"],
                           "model_loaded": False, "dev_opened": False,
                           "test_opened": False}}
    proof_path = sources / "independent-verification.json"
    proof_path.write_text(json.dumps(proof))
    train = tmp_path / "train.jsonl"
    labels = [{"id": row["id"], "input": row["input"],
               "expected": ["가 나"], "group_id": f"group-{index}",
               "source_ids": [f"source-{index}"], "split": "train",
               "input_contract": "roman_sentence_only"}
              for index, row in enumerate(rows)]
    train.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                             for row in labels))
    model = tmp_path / "model.bin"
    model.write_bytes(b"model")
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps({
        "completed": True,
        "proposed_version": capture.EXPECTED_RUNTIME_VERSION,
        "model_pins": {str(model): sha(model)}, "pins": {}}))
    plan = tmp_path / "plan.md"
    plan.write_text("fixed plan\n")
    return {"sources": sources, "selection": selection,
            "manifest": manifest_path, "proof": proof_path,
            "train": train, "model": model, "reference": reference,
            "plan": plan, "rows": rows, "labels": labels,
            "partition_counts": dict(__import__("collections").Counter(splits))}


def run_fixture(tmp_path, data, converter):
    return capture.run(
        data["sources"], tmp_path / "output",
        converter_factory=lambda: converter,
        watchdog_factory=lambda: Watchdog(), runtime_setup=lambda: 0,
        train_path=data["train"], plan_path=data["plan"],
        reference_report_path=data["reference"],
        selection_sha=sha(data["selection"]),
        manifest_sha=sha(data["manifest"]), proof_sha=sha(data["proof"]),
        reference_sha=sha(data["reference"]), train_sha=sha(data["train"]),
        expected_rows=len(data["rows"]),
        partition_counts=data["partition_counts"], runtime_resources=[],
        response_validator=response_validator,
        identity_validator=identity_validator)


def test_capture_uses_exact_source_projection_and_runtime_call_arguments(tmp_path):
    data = fixture(tmp_path, 2, ["fit", "inner_hold"])
    converter = Converter(data["model"])
    report = run_fixture(tmp_path, data, converter)
    assert converter.loads == ["load_models", "load_word_model", "load_growth_model"]
    assert converter.calls == [("ab cd", {"context": "", "engine": "neural",
                                           "mode": "batch", "raw": False})] * 2
    assert report["rows"] == 2
    runtime = list(capture.iter_jsonl(tmp_path / "output/runtime-capture.jsonl"))
    source = list(capture.iter_jsonl(tmp_path / "output/source-capture.jsonl"))
    assert [row["id"] for row in runtime] == ["row-0", "row-1"]
    assert all(set(row) == {"id", "request", "proposed", "extended",
                            "elapsed_ms"} for row in runtime)
    assert all(row["request"] == {"input": "ab cd", "context": ""}
               for row in source)


def test_both_source_streams_close_before_first_target_access(tmp_path,
                                                              monkeypatch):
    data = fixture(tmp_path, 1)
    converter = Converter(data["model"])
    original = capture.sha256_file

    def observing_sha(path):
        if Path(path) == data["train"]:
            closure = tmp_path / "output/capture.complete.json"
            assert closure.is_file()
            closed = json.loads(closure.read_text())
            assert closed["source_only"] is True
            assert closed["targets_joined"] is False
        return original(path)

    monkeypatch.setattr(capture, "sha256_file", observing_sha)
    run_fixture(tmp_path, data, converter)


def test_prepared_rows_preserve_global_partition_and_parent_baseline(tmp_path):
    data = fixture(tmp_path, 2, ["fit", "inner_hold"])
    converter = Converter(data["model"])
    run_fixture(tmp_path, data, converter)
    prepared = list(capture.iter_jsonl(tmp_path / "output/prepared.jsonl"))
    assert [row["split"] for row in prepared] == ["fit", "inner_hold"]
    assert [row["component_id"] for row in prepared] == [
        "component-0", "component-1"]
    assert all(row["parent_result"] == "가 나" for row in prepared)
    assert all(row["parent_candidates"] == ["가 나", "카 다", "ab cd"]
               for row in prepared)
    assert all(row["original_candidates"][:3]
               == ["가 나", "카 다", "ab cd"] for row in prepared)


def test_failure_preserves_prefix_and_never_opens_targets(tmp_path):
    data = fixture(tmp_path, 2)
    data["train"].unlink()
    converter = Converter(data["model"], fail_at=2)
    with pytest.raises(RuntimeError, match="planned failure"):
        capture.run(
            data["sources"], tmp_path / "output",
            converter_factory=lambda: converter,
            watchdog_factory=lambda: Watchdog(), runtime_setup=lambda: 0,
            train_path=data["train"], plan_path=data["plan"],
            reference_report_path=data["reference"],
            selection_sha=sha(data["selection"]),
            manifest_sha=sha(data["manifest"]), proof_sha=sha(data["proof"]),
            reference_sha=sha(data["reference"]), train_sha="never-read",
            expected_rows=2, partition_counts={"fit": 2},
            runtime_resources=[], response_validator=response_validator,
            identity_validator=identity_validator)
    output = tmp_path / "output"
    report = json.loads((output / "report.json").read_text())
    assert report["captured_rows"] == 1 and report["targets_opened"] is False
    assert len(list(capture.iter_jsonl(output / "runtime-capture.jsonl"))) == 1
    assert len(list(capture.iter_jsonl(output / "source-capture.jsonl"))) == 1
    assert not (output / "capture.complete.json").exists()
    assert not (output / "prepared.jsonl").exists()


def test_responses_are_not_retained_across_rows(tmp_path):
    LiveResponse.live = LiveResponse.maximum = 0
    data = fixture(tmp_path, 3)
    converter = Converter(data["model"])
    run_fixture(tmp_path, data, converter)
    gc.collect()
    assert LiveResponse.maximum == 1
    assert LiveResponse.live == 0


def test_real_watchdog_limits_expose_update_seconds_key():
    from scripts.preflight_pko_seq2seq import evaluate_observation

    observed = evaluate_observation(
        elapsed_s=1, operation_elapsed_s=1, rss=1, swap=0,
        baseline_swap=0, limits=capture.LIMITS)
    assert observed["operation_elapsed_s"] == 1


def test_compact_join_rejects_malformed_target_metadata():
    runtime = {"id": "one", "request": {"input": "ab cd", "context": ""},
               "proposed": {"result": "가 나", "candidates": [],
                            "pipeline": {"pool": []}}}
    source = {"id": "one", "request": runtime["request"], "proposals": []}
    target = {"id": "one", "input": "ab cd", "expected": [],
              "group_id": "g", "source_ids": [], "split": "train",
              "input_contract": "roman_sentence_only"}
    with pytest.raises(ValueError, match="identity"):
        list(capture.compact_prepared(
            [runtime], [source], [target],
            {"one": {"component_id": "c", "split": "fit"}}))
