from dataclasses import dataclass
import json
from pathlib import Path

import pytest

import scripts.probe_supported_phonetic_runtime as module


@dataclass(frozen=True)
class Option:
    text: str
    score: float
    kind: str


def candidate(source, text, score=1.0, kind="neural"):
    return {"text": text, "score": score, "segments": [
        {"input_start": 0, "input_end": len(source), "source": source,
         "text": text, "kind": kind}
    ]}


class Cache:
    def __init__(self):
        self.cleared = 0

    def cache_clear(self):
        self.cleared += 1


class Converter:
    version = "production"
    model_artifacts = {
        "ranker": "a" * 64,
        "checkpoints/phonetic_transducer.pt": "b" * 64,
        "checkpoints/phonetic_transducer.json": "c" * 64,
    }

    def __init__(self):
        self.word_options = Cache()
        self.transducer = "old"


def test_install_replaces_only_transducer_identity_and_clears_cache(tmp_path):
    checkpoint = tmp_path / "phonetic_transducer.pt"
    metadata = checkpoint.with_suffix(".json")
    checkpoint.write_bytes(b"new checkpoint")
    metadata.write_bytes(b"metadata")
    converter = Converter()
    base = module.install_supported_transducer(
        converter, "new", checkpoint, metadata
    )
    assert base == "production" and converter.transducer == "new"
    assert converter.word_options.cleared == 1
    assert "ranker" in converter.model_artifacts
    assert "checkpoints/phonetic_transducer.pt" not in converter.model_artifacts
    assert converter.version.startswith("supported-phonetic-runtime-1-")


class FakeConverter:
    def __init__(self, source):
        self.source = source
        self.audit_pieces = []

    def convert(self, text, context, raw=False):
        assert "expected" not in self.source
        if raw:
            values = [candidate(text, text, 0.0, "raw")]
            return response(text, context, values, raw=True)
        self.audit_pieces = [(text, 0, (Option("가", 1.0, "neural"),))]
        return response(text, context, [candidate(text, "가")])


def response(source, context, candidates, raw=False):
    return {
        "input": source, "context": context, "engine": "neural", "mode": "batch",
        "raw": raw, "version": "experiment", "candidates": candidates,
        "result": candidates[0]["text"], "ranker": None if raw else "hybrid",
        "model_artifacts": {} if raw else {"model": "a" * 64},
        "context_used": not raw, "external_context_used": not raw,
        "ranking_applied": not raw, "sentence_ranking_applied": False,
        "warnings": [], "engine_ms": 1.0,
    }


def source_row():
    return {
        "id": "r", "group_id": "g", "input": "ga", "context": "ctx",
        "current_result": "가", "finite_candidate_indices": [0, 1, 2, 3],
        "finite_candidates": [
            {"source_index": index, "object": candidate("ga", text, 4 - index)}
            for index, text in enumerate(("가", "카", "까", "갸"))
        ],
    }


def test_capture_closes_source_only_rows_and_preserves_raw(tmp_path):
    source = source_row()
    path = tmp_path / "capture.jsonl"
    captured = module.capture_sources(
        FakeConverter(source), [source], path, "experiment", {"model": "a" * 64}
    )
    assert path.is_file() and len(captured) == 1
    assert captured[0]["baseline_candidates"][-1]["text"] == "ga"
    assert captured[0]["new_learned_beam"] == ["가"]
    assert captured[0]["raw_response"]["result"] == "ga"


def test_summary_reports_corrections_regressions_and_beam_changes():
    captures = [
        {"id": "a", "group_id": "g", "input": "ga", "context": "",
         "baseline_result": "카", "baseline_candidates": [candidate("ga", "카")],
         "new_result": "가", "new_candidates": [candidate("ga", "가")],
         "new_learned_beam": ["가"], "new_engine_ms": 2.0,
         "protected_input": False},
        {"id": "b", "group_id": "g", "input": "na", "context": "",
         "baseline_result": "나", "baseline_candidates": [candidate("na", "나")],
         "new_result": "너", "new_candidates": [candidate("na", "너")],
         "new_learned_beam": ["너"], "new_engine_ms": 2.0,
         "protected_input": False},
    ]
    targets = [
        {"id": "a", "group_id": "g", "input": "ga", "context": "",
         "result": "카", "candidates": [candidate("ga", "카")], "expected": ["가"]},
        {"id": "b", "group_id": "g", "input": "na", "context": "",
         "result": "나", "candidates": [candidate("na", "나")], "expected": ["나"]},
    ]
    stages = [
        {"id": "a", "group_id": "g", "input": "ga", "context": "", "result": "카",
         "learned_pieces": [["ga", 0, [{"text": "카"}]]]},
        {"id": "b", "group_id": "g", "input": "na", "context": "", "result": "나",
         "learned_pieces": [["na", 0, [{"text": "나"}]]]},
    ]
    value = module.summarize(captures, targets, stages, {"top1": 1, "top5": 1})
    assert value["correction_ids"] == ["a"]
    assert value["regression_ids"] == ["b"]
    assert value["beam_gain_ids"] == ["a"] and value["beam_loss_ids"] == ["b"]


def test_independent_verification_rejects_unpinned_or_incomplete_proof(tmp_path):
    checkpoint = tmp_path / "phonetic_transducer.pt"
    checkpoint.write_bytes(b"checkpoint")
    proof = tmp_path / "proof.json"
    proof.write_text(json.dumps({"completed": False}))
    digest = module.file_sha256(proof)
    with pytest.raises((ValueError, FileNotFoundError)):
        module.validate_independent_training(checkpoint, proof, digest)


def test_piece_paths_and_protected_detection_are_bounded():
    pieces = [["ga", 0, [{"text": "가"}, {"text": "카"}]],
              ["na", 2, [{"text": "나"}, {"text": "너"}]]]
    assert module.piece_paths(pieces) == ["가나", "가너", "카나", "카너"]
    assert module.protected_input("visit https://example.com") is True
    assert module.protected_input("gachi ganda") is False
    assert module.preserved_segments(candidate("test", "test", kind="english")) is True
    assert module.preserved_segments(candidate("test", "테스트", kind="english")) is False


def test_source_pin_conflicts_fail_closed():
    with pytest.raises(ValueError, match="conflicting source pin"):
        module.merge_pins({"same": "a" * 64}, {"same": "b" * 64})


def test_pre_capture_pins_exclude_every_label_bearing_file(monkeypatch):
    seen = []
    phase = {"verify": False}

    def fake_sha(path):
        key = module.path_key(path)
        seen.append(key)
        if phase["verify"]:
            return pins[key]
        return "f" * 64

    monkeypatch.setattr(module, "file_sha256", fake_sha)
    monkeypatch.setattr(module.Path, "is_file", lambda self: True)
    inherited = {
        module.path_key(module.TARGETS / "cases.jsonl"): module.TARGET_CASES_SHA,
        module.path_key(module.TARGETS / "manifest.json"): module.TARGET_MANIFEST_SHA,
        module.path_key(module.STAGES / "cases.jsonl"): module.STAGE_CASES_SHA,
        module.path_key(module.STAGES / "report.json"): module.STAGE_REPORT_SHA,
        "data/natural_context/final-choice-v1/challenge_fit/cases.jsonl": "e" * 64,
        "source-only.json": "f" * 64,
    }
    pins = module.source_pins({"proof.json": "f" * 64}, {"source_hashes": inherited})
    assert "source-only.json" not in pins and "proof.json" in pins
    assert not any(name.startswith(module.path_key(module.TARGETS)) for name in pins)
    assert not any(name.startswith(module.path_key(module.STAGES)) for name in pins)
    seen.clear()
    phase["verify"] = True
    module.verify_pins(pins)
    assert not any(name.startswith(module.path_key(module.TARGETS)) for name in seen)
    assert not any(name.startswith(module.path_key(module.STAGES)) for name in seen)
    assert not any("final-choice-v1" in name for name in seen)


def test_final_guard_failure_removes_completed_report(tmp_path):
    with pytest.raises(RuntimeError, match="late resource failure"):
        module.finalize_success(
            tmp_path, {"completed": True},
            lambda: (_ for _ in ()).throw(RuntimeError("late resource failure")),
        )
    assert not (tmp_path / "report.json").exists()
    assert not (tmp_path / "status.json").exists()
