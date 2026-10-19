import hashlib
import json
from types import SimpleNamespace

import pytest

import scripts.evaluate_ensemble_regression as module


def response(source, context, text, *, version="v", artifacts=None, raw=False, ms=1.0):
    return {
        "input": source, "context": context, "raw": raw, "result": text,
        "version": version, "model_artifacts": artifacts or {}, "engine_ms": ms,
        "candidates": [{"text": text, "score": 0.0, "segments": [{
            "input_start": 0, "input_end": len(source), "source": source,
            "text": text, "kind": "raw" if raw else "neural",
        }]}],
    }


class FakeEnsemble:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.last_baseline_response = None
        self.last_supported_response = None
        self.calls = []
        self.baseline = SimpleNamespace(model_artifacts={})
        self.supported = SimpleNamespace(model_artifacts={})

    def load_models(self):
        self.loaded = True

    def convert(self, source, context, raw=False):
        self.calls.append((source, context, raw))
        if raw:
            self.last_supported_response = None
            return response(source, context, source, raw=True)
        values = {
            "ga": ("카", "가", "가"),
            "na": ("나", "너", "나"),
        }[source]
        self.last_baseline_response = response(
            source, context, values[0], version="baseline", artifacts={"base": "a" * 64})
        self.last_supported_response = response(
            source, context, values[1], version="supported", artifacts={"new": "b" * 64})
        return response(source, context, values[2], version="ensemble",
                        artifacts={"ensemble": "c" * 64})


def write_corpus(tmp_path):
    values = {
        "dev": [{"id": "dev-1", "input": "ga", "context": "ctx",
                 "expected": ["가"], "category": "cat", "split": "dev"}],
        "test": [{"id": "test-1", "input": "na", "context": "",
                  "expected": ["나"], "category": "cat", "split": "test"}],
    }
    specs = {}
    for suite, rows in values.items():
        raw = b"".join(json.dumps(row, ensure_ascii=False).encode() + b"\n" for row in rows)
        name = f"{suite}.jsonl"
        (tmp_path / name).write_bytes(raw)
        specs[suite] = {"file": name, "sha256": hashlib.sha256(raw).hexdigest(), "rows": 1}
    manifest = {
        "schema_version": 2, "corpus_version": "2.2",
        "counts": {"dev": 1, "test": 1},
        "category_counts": {"dev": {"cat": 1}, "test": {"cat": 1}},
        "sha256": {spec["file"]: spec["sha256"] for spec in specs.values()},
    }
    raw = json.dumps(manifest).encode()
    (tmp_path / "manifest.json").write_bytes(raw)
    return specs, hashlib.sha256(raw).hexdigest()


def proof_validator(checkpoint, verification, digest):
    return {"training_report": {"checkpoint_sha256": "1" * 64,
                                "metadata_sha256": "2" * 64}, "pins": {}}


def test_run_captures_source_only_before_late_target_join_and_scores_pairs(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    specs, manifest_sha = write_corpus(corpus)
    checkpoint, proof = tmp_path / "model.pt", tmp_path / "proof.json"
    checkpoint.write_bytes(b"model")
    proof.write_bytes(b"proof")
    created = []

    def factory(**kwargs):
        value = FakeEnsemble(**kwargs)
        created.append(value)
        return value

    def late_targets(directory, suites):
        capture = tmp_path / "output/capture.jsonl"
        assert capture.is_file() and len(capture.read_bytes().splitlines()) == 2
        assert b'"expected"' not in capture.read_bytes()
        return module.load_targets(directory, suites)

    result = module.run(
        tmp_path / "output", checkpoint, proof, "0" * 64,
        corpus_dir=corpus, manifest_sha=manifest_sha, suites=specs,
        converter_factory=factory, proof_validator=proof_validator,
        target_loader=late_targets, clock=lambda: 0.0, rss_reader=lambda: 1,
    )
    assert result["rows"] == 2 and result["labels"]["labels_passed_to_converter"] is False
    assert result["suites"]["dev"]["arms"]["baseline"]["top1"] == 0
    assert result["suites"]["dev"]["arms"]["ensemble"]["top1"] == 1
    assert result["suites"]["test"]["versus_baseline"]["supported"]["regression_ids"] == ["test-1"]
    assert result["raw_preserved_all"] is True
    assert created[0].calls == [("ga", "ctx", False), ("ga", "ctx", True),
                                ("na", "", False), ("na", "", True)]
    assert created[0].kwargs["experimental"] is True


def test_source_projection_drops_targets_but_validates_manifest(tmp_path):
    specs, manifest_sha = write_corpus(tmp_path)
    rows, pins = module.load_source_rows(tmp_path, manifest_sha, specs)
    assert rows == [
        {"id": "dev-1", "input": "ga", "context": "ctx", "category": "cat", "suite": "dev"},
        {"id": "test-1", "input": "na", "context": "", "category": "cat", "suite": "test"},
    ]
    assert all("expected" not in row for row in rows) and len(pins) == 3
    with pytest.raises(ValueError, match="pinned corpus differs"):
        module.load_source_rows(tmp_path, "f" * 64, specs)


def test_capture_rejects_changed_protected_segment_and_keeps_partial_file(tmp_path):
    class Bad(FakeEnsemble):
        def convert(self, source, context, raw=False):
            value = super().convert(source, context, raw)
            if not raw:
                self.last_baseline_response["candidates"][0]["segments"][0].update(
                    kind="english", text="changed")
                self.last_baseline_response["candidates"][0]["text"] = "changed"
                self.last_baseline_response["result"] = "changed"
            return value
    path = tmp_path / "capture.jsonl"
    with pytest.raises(ValueError, match="protected"):
        module.capture_rows(Bad(), [{"id": "x", "input": "ga", "context": "",
                                    "category": "c", "suite": "dev"}], path)
    assert path.is_file()


def test_run_resource_failure_preserves_capture_and_writes_only_failure(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    specs, manifest_sha = write_corpus(corpus)
    checkpoint, proof = tmp_path / "model.pt", tmp_path / "proof.json"
    checkpoint.write_bytes(b"model")
    proof.write_bytes(b"proof")
    monkeypatch.setattr(module, "TIME_LIMIT", .5)
    times = iter([0.0, 0.0, 0.0, 1.0, 1.0])
    output = tmp_path / "output"
    with pytest.raises(RuntimeError, match="1200 seconds"):
        module.run(
            output, checkpoint, proof, "0" * 64, corpus_dir=corpus,
            manifest_sha=manifest_sha, suites=specs, converter_factory=FakeEnsemble,
            proof_validator=proof_validator, target_loader=lambda *_: pytest.fail("target read early"),
            clock=lambda: next(times), rss_reader=lambda: 1,
        )
    assert (output / "capture.jsonl").is_file()
    assert (output / "failure.json").is_file()
    assert not (output / "report.json").exists() and not (output / "status.json").exists()


def test_score_reports_quality_without_failing_historical_mismatch():
    rows = []
    targets = {}
    for suite, source, expected in (("dev", "ga", "가"), ("test", "na", "나")):
        converter = FakeEnsemble()
        ensemble = converter.convert(source, "")
        rows.append({"suite": suite, "id": suite, "category": "cat", "input": source,
                     "context": "", "baseline": converter.last_baseline_response,
                     "supported": converter.last_supported_response, "ensemble": ensemble})
        targets[(suite, suite)] = [expected]
    suites, _ = module.score_captures(rows, targets)
    assert suites["dev"]["baseline_matches_historical_reference"] is False
    assert suites["test"]["arms"]["baseline"]["top1"] == 1


@pytest.mark.parametrize("defect", ["empty_segment", "none_score"])
def test_response_rejects_fallback_provenance_not_allowed_by_core(defect):
    value = response("ga", "", "가")
    if defect == "empty_segment":
        value["candidates"][0]["segments"] = [
            {"input_start": 0, "input_end": 0, "source": "", "text": "",
             "kind": "neural"},
            {"input_start": 0, "input_end": 2, "source": "ga", "text": "가",
             "kind": "neural"},
        ]
    else:
        value["candidates"][0]["score"] = None
    with pytest.raises(ValueError):
        module.validate_response(value, "ga", "")


def test_response_accepts_contiguous_multi_segment_original_raw_candidate():
    source = "jigeum jibe ganda"
    value = response(source, "", source)
    value["candidates"][0] = {
        "text": source,
        "score": None,
        "segments": [
            {"input_start": 0, "input_end": 6, "source": "jigeum",
             "text": "jigeum", "kind": "raw"},
            {"input_start": 6, "input_end": 7, "source": " ",
             "text": " ", "kind": "raw"},
            {"input_start": 7, "input_end": 11, "source": "jibe",
             "text": "jibe", "kind": "raw"},
            {"input_start": 11, "input_end": 12, "source": " ",
             "text": " ", "kind": "raw"},
            {"input_start": 12, "input_end": 17, "source": "ganda",
             "text": "ganda", "kind": "raw"},
        ],
    }
    value["result"] = source

    assert module.validate_response(value, source, "") is value
