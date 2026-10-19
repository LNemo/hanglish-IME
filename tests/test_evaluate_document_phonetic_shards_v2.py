import json
from pathlib import Path

import pytest

from scripts import evaluate_document_phonetic_shards_v2 as evaluator


def response(*texts, engine_ms=None, ranked=None):
    value = {"result": texts[0],
             "candidates": [{"text": text} for text in texts]}
    if engine_ms is not None:
        value["engine_ms"] = engine_ms
    if ranked is not None:
        value["sentence_ranking_applied"] = ranked
    return value


def record(identifier, source, baseline, proposed, *, ranked=True, pool=(), ms=1.0):
    return {
        "id": identifier,
        "request": {"input": source, "context": ""},
        "baseline": response(*baseline),
        "proposed": response(*proposed, engine_ms=ms, ranked=ranked),
        "base_paths": [[text, 0.0] for text in pool],
    }


def test_aggregate_uses_literal_exact_reference_without_normalization():
    rows = [record("a", "jeongdap", ["정답"], ["정답"], pool=["정답"])]
    metrics = evaluator.aggregate(rows, {"a": "정답 "})
    assert metrics["baseline"] == {"rows": 1, "top1": 0, "top5": 0}
    assert metrics["proposed"] == {"rows": 1, "top1": 0, "top5": 0}

    metrics = evaluator.aggregate(rows, {"a": "정답"})
    assert metrics["baseline"]["top1"] == metrics["proposed"]["top1"] == 1


def test_ranked_candidate_missing_selection_failure_and_unranked_are_separate():
    labels = {"missing": "정답1", "selection": "정답2", "unranked": "정답3"}
    rows = [
        record("missing", "source1", ["오답"], ["오답"], pool=["오답"]),
        record("selection", "source2", ["오답"], ["오답"], pool=["정답2", "오답"]),
        record("unranked", "source3", ["오답"], ["오답"], ranked=False, pool=[]),
    ]
    metrics = evaluator.aggregate(rows, labels)
    assert metrics["stage_counts"] == {
        "ranked_rows": 2,
        "ranked_pool_gold": 1,
        "ranked_candidate_missing": 1,
        "ranked_selection_failures": 1,
        "unranked_rows": 1,
        "unranked_first_failures": 1,
    }


def test_gain_with_no_regression_passes_and_latency_uses_nearest_rank():
    labels = {"keep": "유지", "fix": "수정"}
    rows = [
        record("keep", "yuji", ["유지"], ["유지"], pool=["유지"], ms=9.0),
        record("fix", "sujeong", ["오답", "수정"], ["수정"], pool=["수정"], ms=1.0),
    ]
    metrics = evaluator.aggregate(rows, labels)
    assert metrics["corrections"] == 1
    assert metrics["regressions"] == metrics["top5_losses"] == 0
    assert metrics["improvement_gate"] is metrics["quality_gate"] is True
    assert metrics["actual_calibrated_latency"] == {
        "p50_ms": 1.0, "p95_ms": 9.0, "max_ms": 9.0,
    }


def test_top5_ascii_and_literal_regressions_each_close_quality_gate():
    top5_rows = [
        record("fix", "gochida", ["오답"], ["고치다"], pool=["고치다"]),
        record("loss", "jeongdap", ["오답", "정답"], ["오답"], pool=["정답", "오답"]),
    ]
    metrics = evaluator.aggregate(top5_rows, {"fix": "고치다", "loss": "정답"})
    assert metrics["corrections"] == 1 and metrics["top5_losses"] == 1
    assert metrics["quality_gate"] is False

    copy_rows = [
        record("fix", "gochida", ["오답"], ["고치다"], pool=["고치다"]),
        record("copy", "testreul A-1!", ["test를 A-1!"], ["테스트를 에이일"],
               pool=["test를 A-1!", "테스트를 에이일"]),
    ]
    metrics = evaluator.aggregate(copy_rows, {"fix": "고치다", "copy": "test를 A-1!"})
    assert metrics["english_ascii_signature"]["signature_regressions"] == 1
    assert metrics["english_ascii_signature"]["literal_signature_regressions"] == 1
    assert metrics["quality_gate"] is False


@pytest.mark.parametrize("labels,records,message", [
    ({}, [record("a", "a", ["가"], ["가"], pool=["가"])], "label"),
    ({"b": "나"}, [record("a", "a", ["가"], ["가"], pool=["가"])], "unknown"),
    ({"a": "가", "b": "나"},
     [record("a", "a", ["가"], ["가"], pool=["가"])], "denominator"),
    ({"a": "가"},
     [record("a", "a", ["가"], ["가"], pool=["가"]),
      record("a", "a", ["가"], ["가"], pool=["가"])], "duplicate"),
    ({"a": ""}, [record("a", "a", [""], [""], pool=[""])], "label"),
])
def test_empty_missing_unknown_and_duplicate_labels_are_rejected(labels, records, message):
    with pytest.raises(ValueError, match=message):
        evaluator.aggregate(records, labels)


@pytest.mark.parametrize("damage,message", [
    ("missing", "incomplete"),
    ("duplicate", "alignment"),
    ("empty", "alignment"),
    ("normalized", "alignment"),
])
def test_late_reference_join_rejects_missing_duplicate_empty_and_normalized_text(
        monkeypatch, damage, message):
    selection = []
    raw = []
    requests = {}
    for index in range(1046):
        identifier = f"id-{index:04d}"
        group = f"group-{min(index, 820):04d}"
        text = f"Source-{index}"
        selection.append({"id": identifier, "group_id": group})
        raw.append({"id": identifier, "group_id": group, "text": text})
        requests[identifier] = text
    if damage == "missing":
        raw.pop()
    elif damage == "duplicate":
        raw.append(dict(raw[0]))
    elif damage == "empty":
        raw[0]["text"] = ""
        requests[raw[0]["id"]] = ""
    else:
        requests[raw[0]["id"]] = raw[0]["text"].lower()

    monkeypatch.setattr(
        evaluator, "sha",
        lambda path: evaluator.RAW_SHA if Path(path) == evaluator.RAW
        else evaluator.SELECTION_SHA,
    )
    monkeypatch.setattr(
        evaluator, "jsonl",
        lambda path: iter(selection if Path(path) == evaluator.SELECTION else raw),
    )
    with pytest.raises(ValueError, match=message):
        evaluator.load_references(requests)


def closed_run_fixture(tmp_path, monkeypatch, *, live=False):
    directories = [tmp_path / "capture-0", tmp_path / "capture-1"]
    for directory in directories:
        directory.mkdir()
    partition_hashes = {
        "manifest.json": "partition-manifest",
        "shard-0.jsonl": "partition-0",
        "shard-1.jsonl": "partition-1",
    }
    inputs = [[], []]
    captures = [[], []]
    for index in range(1046):
        shard = index % 2
        identifier = f"id-{index:04d}"
        source = f"source-{index}"
        inputs[shard].append({"id": identifier, "input": source})
        captures[shard].append({"id": identifier,
                                "request": {"input": source, "context": ""}})

    reports = []
    proofs = []
    completions = []
    resource = {"after_completion": {"elapsed_s": 10.0},
                "peak_rss_bytes": 1024,
                "peak_swap_growth_bytes": 0}
    for index in range(2):
        digests = {"report_sha256": f"report-{index}",
                   "capture_sha256": f"capture-{index}",
                   "closure_sha256": f"closure-{index}",
                   "completion_sha256": f"completion-{index}"}
        pins = {str(evaluator.VERIFIER.resolve()): "verifier",
                str(evaluator.CAPTURE_PRODUCER.resolve()): "producer"}
        if index:
            pins[str((directories[0] / "independent-verification.json").resolve())] = (
                "proof-0")
        reports.append({
            "completed": True, "shard": index,
            "schema": "document-phonetic-shard-capture-v1",
            "accuracy_measured": False, "performance_labels_opened": False,
            "partition_hashes": partition_hashes, "version": "version",
            "weights": [1.0], "rows": len(inputs[index]), "pins": pins,
        })
        proofs.append({
            "verified": True,
            "schema": "document-phonetic-shard-independent-verification-v1",
            "shard": index, "rows": len(inputs[index]),
            "verifier_sha256": "verifier" if index == 0 else "verifier-v2",
            "launch_sha256": f"launch-{index}",
            **digests, "resources": resource,
        })
        completions.append({"completed": True,
                            "report_sha256": digests["report_sha256"]})

    def fake_read(path):
        path = Path(path)
        if path.name.endswith(".launch.json"):
            index = int(path.name.split("-")[-1].split(".")[0])
            command = [str(evaluator.ROOT / ".venv/bin/python"),
                       str(evaluator.CAPTURE_PRODUCER), "--output",
                       str(directories[index].resolve()), "--shard", str(index)]
            if index:
                command += ["--previous", str(directories[0].resolve())]
            return {"pid": index + 100, "runner_sha256": "producer",
                    "output": str(directories[index].resolve()), "command": command}
        index = directories.index(path.parent)
        return {"report.json": reports, "independent-verification.json": proofs,
                "completion.json": completions}[path.name][index]

    def fake_sha(path):
        path = Path(path)
        if path.parent == evaluator.SHARDS:
            return partition_hashes[path.name]
        if path == evaluator.VERIFIER:
            return "verifier"
        if path == evaluator.AUDIT_VERIFIERS[1]:
            return "verifier-v2"
        if path == evaluator.CAPTURE_PRODUCER:
            return "producer"
        if path.name.endswith(".launch.json"):
            index = int(path.name.split("-")[-1].split(".")[0])
            return f"launch-{index}"
        index = directories.index(path.parent)
        return {"report.json": proofs[index]["report_sha256"],
                "capture.jsonl": proofs[index]["capture_sha256"],
                "capture.complete.json": proofs[index]["closure_sha256"],
                "completion.json": proofs[index]["completion_sha256"],
                "independent-verification.json": f"proof-{index}"}[path.name]

    def fake_jsonl(path):
        path = Path(path)
        if path.parent == evaluator.SHARDS:
            return iter(inputs[int(path.stem.split("-")[-1])])
        if path.name == "resources.jsonl":
            return iter([{"resource": True}])
        return iter(captures[directories.index(path.parent)])

    monkeypatch.setattr(evaluator, "read", fake_read)
    monkeypatch.setattr(evaluator, "sha", fake_sha)
    monkeypatch.setattr(evaluator, "jsonl", fake_jsonl)
    monkeypatch.setattr(evaluator, "verify_resources", lambda rows, operations: resource)
    if live:
        monkeypatch.setattr(evaluator.os, "kill", lambda pid, signal: None)
    else:
        def terminal(pid, signal):
            raise ProcessLookupError(pid)
        monkeypatch.setattr(evaluator.os, "kill", terminal)
    return directories


def test_both_closed_shards_authenticate_before_any_label_access(tmp_path, monkeypatch):
    directories = closed_run_fixture(tmp_path, monkeypatch)
    evidence, requests, version, weights = evaluator.authenticate_runs(directories)
    assert len(evidence) == 2 and len(requests) == 1046
    assert version == "version" and weights == [1.0]


def test_wrong_second_audit_verifier_rejected(tmp_path, monkeypatch):
    directories = closed_run_fixture(tmp_path, monkeypatch)
    original = evaluator.sha
    monkeypatch.setattr(evaluator, "sha", lambda path: (
        "verifier" if Path(path) == evaluator.AUDIT_VERIFIERS[1] else original(path)))
    with pytest.raises(ValueError, match="producer/verifier identity"):
        evaluator.authenticate_runs(directories)


def test_live_capture_pid_is_rejected(tmp_path, monkeypatch):
    directories = closed_run_fixture(tmp_path, monkeypatch, live=True)
    with pytest.raises(ValueError, match="still present"):
        evaluator.authenticate_runs(directories)


@pytest.mark.parametrize("name", ["failure.json", "resource-watchdog-failure.json"])
def test_failure_artifact_stops_authentication_before_report_read(
        tmp_path, monkeypatch, name):
    directories = [tmp_path / "capture-0", tmp_path / "capture-1"]
    for directory in directories:
        directory.mkdir()
    (directories[0] / name).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        evaluator, "read",
        lambda path: (_ for _ in ()).throw(AssertionError("report opened after failure")),
    )
    with pytest.raises(ValueError, match="failure evidence"):
        evaluator.authenticate_runs(directories)


def test_run_joins_labels_only_after_both_shards_authenticate(tmp_path, monkeypatch):
    calls = []
    labels = {"a": "정답"}
    metrics = {
        "baseline": {"rows": 1, "top1": 0, "top5": 0},
        "proposed": {"rows": 1, "top1": 1, "top5": 1},
        "corrections": 1, "regressions": 0, "top5_losses": 0,
        "stage_counts": {}, "english_ascii_signature": {},
        "improvement_gate": True, "quality_gate": True,
        "actual_calibrated_latency": {"p50_ms": 1, "p95_ms": 1, "max_ms": 1},
    }

    def authenticate(directories):
        calls.append(("authenticate", tuple(directories)))
        return [{}, {}], {"a": "source"}, "version", [1.0]

    def load(requests):
        assert calls == [("authenticate", ("shard0", "shard1"))]
        calls.append(("labels", dict(requests)))
        return labels

    def aggregate(records, joined):
        assert calls[-1][0] == "labels" and joined is labels
        calls.append(("aggregate", list(records)))
        return metrics

    monkeypatch.setattr(evaluator, "authenticate_runs", authenticate)
    monkeypatch.setattr(evaluator, "load_references", load)
    monkeypatch.setattr(evaluator, "aggregate", aggregate)
    monkeypatch.setattr(evaluator, "jsonl", lambda path: iter(()))
    monkeypatch.setattr(evaluator, "sha", lambda path: "producer")
    output = tmp_path / "report.json"
    evaluator.run(["shard0", "shard1"], output)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert [call[0] for call in calls] == ["authenticate", "labels", "aggregate"]
    assert report["labels_opened_after_both_verified_captures"] is True
    assert report["exact_original_reference_strings"] is True


def test_authentication_failure_prevents_label_access_and_output(tmp_path, monkeypatch):
    def reject(directories):
        raise ValueError("proof rejected")
    monkeypatch.setattr(evaluator, "authenticate_runs", reject)
    monkeypatch.setattr(
        evaluator, "load_references",
        lambda requests: (_ for _ in ()).throw(AssertionError("labels opened")),
    )
    output = tmp_path / "report.json"
    with pytest.raises(ValueError, match="proof rejected"):
        evaluator.run(["shard0", "shard1"], output)
    assert not output.exists()
