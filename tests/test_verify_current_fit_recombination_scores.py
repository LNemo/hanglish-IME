import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import verify_current_fit_recombination_scores as verifier
from scripts import current_recombination_scoring as scorer
from scripts import verify_current_fit_recombination_scoring_probe as row_verifier

_spec = importlib.util.spec_from_file_location(
    "full_scoring_fixture", Path(__file__).with_name("test_current_recombination_scoring.py"))
_fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixtures)


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def build_run(monkeypatch, tmp_path, *, summary=None, mutate=None):
    output = tmp_path / "shard"
    output.mkdir()
    runner = tmp_path / "runner.py"
    core = tmp_path / "core.py"
    helper_path = tmp_path / "helper.py"
    for path, text in ((runner, "# runner\n"), (core, "# core\n"),
                       (helper_path, "# helper\n")):
        path.write_text(text, encoding="utf-8")
    runner_sha, core_sha, helper_sha = map(verifier.sha, (runner, core, helper_path))
    probe_report_sha, probe_proof_sha = "a" * 64, "b" * 64
    identifier = "row-1"
    captured = {"id": identifier, "component_id": "component-1",
                "request": {"input": "foo", "context": ""},
                "source_shard": 2, "source_row_index": 0,
                "scoring": {"original_score_parity": None}}
    (output / "capture.jsonl").write_text(json.dumps(captured) + "\n", encoding="utf-8")
    (output / "resources.jsonl").write_text("{}\n", encoding="utf-8")
    pins = {str(runner.resolve()): runner_sha, str(core.resolve()): core_sha}
    dump(output / "pins.json", pins)
    capture_sha = verifier.sha(output / "capture.jsonl")
    closure = {"schema": "current-fit-recombination-scoring-closure-v1",
               "mode": "capture", "shard": 2, "completed": True, "rows": 1,
               "ids": [identifier], "capture_sha256": capture_sha,
               "runner_sha256": runner_sha, "core_sha256": core_sha,
               "labels_opened": False, "model_inference": True,
               "phonetic_model_inference": False}
    dump(output / "capture.complete.json", closure)
    report = {"schema": "current-fit-recombination-scoring-shard-v1",
              "completed": True, "mode": "capture", "shard": 2, "rows": 1,
              "ids": [identifier], "row_counts": {identifier: summary or {
                  "original": 2, "novel": 1, "scored_novel": 1, "unsupported_novel": 0}},
              "timing_seconds": {identifier: 0.5}, "runner_sha256": runner_sha,
              "core_sha256": core_sha, "capture_sha256": capture_sha,
              "capture_output_bytes": (output / "capture.jsonl").stat().st_size,
              "closure_sha256": verifier.sha(output / "capture.complete.json"),
              "pins": pins, "limits": verifier.LIMITS,
              "probe_report_sha256": probe_report_sha,
              "probe_proof_sha256": probe_proof_sha,
              "model_metadata": {"max_length": 96, "batch_size": 8, "device": "cpu",
                                   "threads": 2, "interop_threads": 1},
              "source_only": True, "fit_labels_opened": False, "dev_opened": False,
              "test_opened": False, "model_inference": True,
              "phonetic_model_inference": False, "neural_training": False,
              "production_changed": False}
    dump(output / "report.json", report)
    terminal = {"stage": "after_final_rehash"}
    dump(output / "completion.json", {"completed": True,
         "report_sha256": verifier.sha(output / "report.json"),
         "resource_after_final_rehash": terminal})
    command = [str(verifier.ROOT / ".venv/bin/python"), str(runner.resolve()),
               "--shard", "2", "--runner-sha256", runner_sha,
               "--core-sha256", core_sha, "--probe-report-sha256", probe_report_sha,
               "--probe-proof-sha256", probe_proof_sha]
    launch = {"pid": 99999999, "command": command, "runner_sha256": runner_sha,
              "core_sha256": core_sha, "probe_report_sha256": probe_report_sha,
              "probe_proof_sha256": probe_proof_sha, "output": str(output.resolve())}
    dump(output.with_suffix(".launch.json"), launch)

    actual_summary = summary or {"original": 2, "novel": 1, "scored_novel": 1,
                                 "unsupported_novel": 0}
    helper = SimpleNamespace(
        verify_scoring_row=lambda *args, **kwargs: actual_summary,
        verify_resources=lambda rows, operations: {"after_final_rehash": terminal,
                                                    "operations": len(operations)})
    monkeypatch.setattr(verifier, "CORE", core)
    monkeypatch.setattr(verifier, "HELPER", helper_path)
    monkeypatch.setattr(verifier, "load_helper", lambda value: helper)
    monkeypatch.setattr(verifier, "authenticate_probe", lambda *args: ({"qualified": True}, {}))
    original = {"id": identifier, "component_id": "component-1",
                "request": {"input": "foo", "context": ""}}
    census = {"id": identifier, "component_id": "component-1",
              "request": original["request"],
              "source_reference": {"source_shard": 2, "source_row_index": 0}}
    monkeypatch.setattr(verifier, "authenticate_sources",
                        lambda shard: ([identifier], iter([(original, census, 0)]), {}))
    monkeypatch.setattr(verifier.os, "kill",
                        lambda *args: (_ for _ in ()).throw(ProcessLookupError()))
    if mutate:
        mutate(output)
    return (output.resolve(), runner_sha, core_sha, probe_report_sha, probe_proof_sha, helper_sha)


def test_authenticates_helper_before_import(monkeypatch, tmp_path):
    helper = tmp_path / "helper.py"
    helper.write_text("raise AssertionError('must not execute')\n", encoding="utf-8")
    monkeypatch.setattr(verifier, "HELPER", helper)
    with pytest.raises(ValueError, match="helper identity"):
        verifier.load_helper("0" * 64)


def test_verifies_closed_full_shard_without_model_replay(monkeypatch, tmp_path):
    args = build_run(monkeypatch, tmp_path)
    proof = verifier.verify(*args)
    assert proof["verified"] is True
    assert proof["shard"] == 2 and proof["rows"] == 1
    assert proof["model_loaded"] is False
    assert proof["model_inference_replayed"] is False
    assert "not independently replayed" in proof["limitations"]


def test_rejects_wrong_exact_source_index(monkeypatch, tmp_path):
    def mutate(output):
        row = json.loads((output / "capture.jsonl").read_text())
        row["source_row_index"] = 1
        (output / "capture.jsonl").write_text(json.dumps(row) + "\n")
        report = json.loads((output / "report.json").read_text())
        report["capture_sha256"] = verifier.sha(output / "capture.jsonl")
        report["capture_output_bytes"] = (output / "capture.jsonl").stat().st_size
        closure = json.loads((output / "capture.complete.json").read_text())
        closure["capture_sha256"] = report["capture_sha256"]
        dump(output / "capture.complete.json", closure)
        report["closure_sha256"] = verifier.sha(output / "capture.complete.json")
        dump(output / "report.json", report)
        completion = json.loads((output / "completion.json").read_text())
        completion["report_sha256"] = verifier.sha(output / "report.json")
        dump(output / "completion.json", completion)
    args = build_run(monkeypatch, tmp_path, mutate=mutate)
    with pytest.raises(ValueError, match="shard/order/index"):
        verifier.verify(*args)


def test_accepts_explicit_unsupported_novel_status(monkeypatch, tmp_path):
    summary = {"original": 2, "novel": 1, "scored_novel": 0, "unsupported_novel": 1}
    args = build_run(monkeypatch, tmp_path, summary=summary)
    assert verifier.verify(*args)["counts"]["row-1"] == summary


def test_row_helper_rejects_numeric_score_for_unsupported_novel():
    original, census = _fixtures.fixture()
    scoring = scorer.score_row(original, census, _fixtures.Ranker(), check_original_parity=False)
    scoring["novel"][0].update(status="unsupported", reason="sentence_ranker_can_score_false")
    scoring["score_texts"] = []
    captured = {"id": original["id"], "component_id": original["component_id"],
                "request": original["request"],
                "source_shard": census["source_reference"]["source_shard"],
                "source_row_index": census["source_reference"]["source_row_index"],
                "scoring": scoring}
    with pytest.raises(ValueError, match="score/status"):
        row_verifier.verify_scoring_row(original, census, captured,
                                        original_parity_required=False)


def test_row_helper_accepts_none_score_for_unsupported_novel():
    original, census = _fixtures.fixture()
    scoring = scorer.score_row(original, census, _fixtures.Ranker(), check_original_parity=False)
    scoring["novel"][0].update(status="unsupported", sentence_score=None,
                               reason="sentence_ranker_can_score_false")
    scoring["score_texts"] = []
    captured = {"id": original["id"], "component_id": original["component_id"],
                "request": original["request"],
                "source_shard": census["source_reference"]["source_shard"],
                "source_row_index": census["source_reference"]["source_row_index"],
                "scoring": scoring}
    assert row_verifier.verify_scoring_row(
        original, census, captured, original_parity_required=False)["unsupported_novel"] == 1


def test_rejects_mutated_launch_probe_identity(monkeypatch, tmp_path):
    def mutate(output):
        launch_path = output.with_suffix(".launch.json")
        launch = json.loads(launch_path.read_text())
        launch["command"][-1] = "c" * 64
        dump(launch_path, launch)
    args = build_run(monkeypatch, tmp_path, mutate=mutate)
    with pytest.raises(ValueError, match="launch identity"):
        verifier.verify(*args)


def test_rejects_live_runner_pid(monkeypatch, tmp_path):
    args = build_run(monkeypatch, tmp_path)
    monkeypatch.setattr(verifier.os, "kill", lambda *args: None)
    with pytest.raises(ValueError, match="still present"):
        verifier.verify(*args)
