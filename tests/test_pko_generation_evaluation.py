import hashlib
import json
import time

import pytest

from scripts import evaluate_pko_generation as evaluation


def jsonl(path, rows):
    raw = b"".join(json.dumps(row, ensure_ascii=False).encode() + b"\n" for row in rows)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def test_source_projection_and_capture_are_independent_of_dev_labels(tmp_path):
    path = tmp_path / "dev.jsonl"
    base = {"id": "d1", "input": "roman sentence", "split": "dev",
            "input_contract": "roman_sentence_only"}
    sha = jsonl(path, [{**base, "expected": ["첫 정답"]}])
    first = evaluation.read_dev_sources(path, expected_sha=sha, expected_rows=1)
    sha = jsonl(path, [{**base, "expected": ["바뀐 정답"]}])
    second = evaluation.read_dev_sources(path, expected_sha=sha, expected_rows=1)
    assert first == second == [{"id": "d1", "source": "roman sentence"}]
    calls = []
    captures = evaluation.capture_sources(
        first, lambda source: calls.append(source) or {
            "generated_text": "생성 문장", "token_ids": [0, 1],
            "reason": "complete", "eos_completed": True, "input_tokens": 3},
        tmp_path / "capture.jsonl")
    assert calls == ["roman sentence"] and "expected" not in captures[0]


def test_scoring_reads_targets_only_after_complete_capture_and_keeps_failures_in_denominator(tmp_path):
    sources = [{"id": "a", "source": "roman one"},
               {"id": "b", "source": "roman 2"}]
    capture = tmp_path / "capture.jsonl"
    jsonl(capture, [
        {"id": "a", "source": "roman one", "generated_text": "정답 하나",
         "token_ids": [0, 1], "reason": "complete", "eos_completed": True,
         "input_tokens": 3, "latency_s": .1},
        {"id": "b", "source": "roman 2", "generated_text": "정답 둘",
         "token_ids": [0] * 513, "reason": "max_tokens_without_eos",
         "eos_completed": False, "input_tokens": 3, "latency_s": .2},
    ])
    observed = []
    def targets():
        observed.append(capture.read_bytes())
        return {"a": ["정답 하나"], "b": ["정답 둘"]}
    metrics, outcomes = evaluation.score_closed_capture(capture, sources, targets)
    assert len(observed[0].splitlines()) == 2
    assert metrics["rows"] == 2 and metrics["whole_exact"] == 1
    assert metrics["raw_text_match"] == 2 and metrics["eos_complete"] == 1
    assert metrics["incomplete_or_invalid"] == 1
    assert [row["correct"] for row in outcomes] == [True, False]


def test_incomplete_or_reordered_capture_blocks_target_access(tmp_path):
    sources = [{"id": "a", "source": "one"}, {"id": "b", "source": "two"}]
    capture = tmp_path / "capture.jsonl"
    jsonl(capture, [{"id": "b", "source": "two"}])
    called = []
    with pytest.raises(ValueError, match="incomplete or reordered"):
        evaluation.score_closed_capture(capture, sources,
                                        lambda: called.append(True) or {})
    assert called == []


def test_late_target_join_cannot_mutate_closed_capture(tmp_path):
    sources = [{"id": "a", "source": "one"}]
    capture = tmp_path / "capture.jsonl"
    row = {"id": "a", "source": "one", "generated_text": "하나",
           "token_ids": [0, 1], "reason": "complete", "eos_completed": True,
           "input_tokens": 2, "latency_s": .1}
    jsonl(capture, [row])
    def mutate_then_load():
        with capture.open("a") as stream:
            stream.write("\n")
        return {"a": ["하나"]}
    with pytest.raises(ValueError, match="changed during late target join"):
        evaluation.score_closed_capture(capture, sources, mutate_then_load)


def test_training_gate_runs_before_runtime_loader(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(evaluation, "validate_training_run",
                        lambda path: (_ for _ in ()).throw(ValueError("gate rejected")))
    with pytest.raises(ValueError, match="gate rejected"):
        evaluation.run(tmp_path / "training", tmp_path / "output",
                       runtime_loader=lambda path: calls.append(path))
    assert calls == []
    failure = json.loads((tmp_path / "output" / "report.json").read_text())
    assert failure["completed"] is False and failure["capture_rows"] == 0


def test_watchdog_terminal_failure_uses_evaluator_schema(tmp_path):
    capture = tmp_path / "capture.jsonl"
    capture.write_text('{"id":"a"}\n')
    watchdog = evaluation.EvaluationWatchdog(
        time.monotonic(), 0, tmp_path / "resources.jsonl", capture)
    watchdog.failure = "ResourceExceeded: RSS bound exceeded"
    watchdog._failure_report(RuntimeError("RSS bound exceeded"))
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["schema"] == evaluation.SCHEMA
    assert report["completed"] is False and report["capture_rows"] == 1
    assert report["scope"] == {"source_only_inference": True,
        "dev_file_parsed_including_label_fields": True,
        "targets_joined": False, "labels_used_as_model_context": False,
        "test_opened": False}
