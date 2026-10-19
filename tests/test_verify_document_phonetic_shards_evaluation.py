import ast
import copy
import json
from pathlib import Path

import pytest

from scripts import verify_document_phonetic_shards_evaluation as verifier


def response(*texts, ranked=None, latency=None):
    value = {"result": texts[0],
             "candidates": [{"text": text} for text in texts]}
    if ranked is not None:
        value["sentence_ranking_applied"] = ranked
    if latency is not None:
        value["engine_ms"] = latency
    return value


def record(identifier, source, baseline, proposed, *, pool=(), ranked=True,
           latency=1.0):
    return {
        "id": identifier,
        "request": {"input": source, "context": ""},
        "baseline": response(*baseline),
        "proposed": response(*proposed, ranked=ranked, latency=latency),
        "base_paths": [[text, 0.0] for text in pool],
    }


def test_verifier_imports_no_evaluator_model_or_tensor_module():
    tree = ast.parse(Path(verifier.__file__).read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    forbidden = ("evaluate_document_phonetic_shards",
                 "production_phonetic_choice_converter",
                 "phonetic_choice_features", "phonetic_likelihood", "torch")
    assert not any(name == item or name.endswith("." + item)
                   for name in imports for item in forbidden)


def test_unresolved_evaluator_hash_blocks_actual_verification():
    with pytest.raises(ValueError, match="not frozen"):
        verifier.resolved_evaluator(verifier.UNRESOLVED)
    assert verifier.resolved_evaluator("a" * 64) == "a" * 64


def test_independent_aggregate_replays_exact_metrics_and_stage_boundaries():
    labels = {
        "keep": "유지", "fix": "고치다", "missing": "정답1",
        "selection": "정답2", "unranked": "정답3",
    }
    rows = [
        record("keep", "yuji", ["유지"], ["유지"], pool=["유지"], latency=9),
        record("fix", "gochida", ["오답", "고치다"], ["고치다"],
               pool=["고치다"], latency=1),
        record("missing", "source1", ["오답"], ["오답"], pool=["오답"], latency=2),
        record("selection", "source2", ["오답"], ["오답"],
               pool=["정답2", "오답"], latency=3),
        record("unranked", "source3", ["오답"], ["오답"],
               pool=[], ranked=False, latency=4),
    ]
    metrics = verifier.aggregate(rows, labels)
    assert metrics["baseline"] == {"rows": 5, "top1": 1, "top5": 2}
    assert metrics["proposed"] == {"rows": 5, "top1": 2, "top5": 2}
    assert metrics["corrections"] == 1
    assert metrics["regressions"] == metrics["top5_losses"] == 0
    assert metrics["stage_counts"] == {
        "ranked_rows": 4, "ranked_pool_gold": 3,
        "ranked_candidate_missing": 1, "ranked_selection_failures": 1,
        "unranked_rows": 1, "unranked_first_failures": 1,
    }
    assert metrics["quality_gate"] is True
    assert metrics["actual_calibrated_latency"] == {
        "p50_ms": 3, "p95_ms": 9, "max_ms": 9,
    }


def test_exact_string_top5_ascii_and_literal_regressions_close_gate():
    rows = [
        record("fix", "gochida", ["오답"], ["고치다"], pool=["고치다"]),
        record("loss", "jeongdap", ["오답", "정답 "], ["오답"],
               pool=["정답 ", "오답"]),
        record("copy", "testreul A-1!", ["test를 A-1!"], ["테스트를 에이일"],
               pool=["test를 A-1!", "테스트를 에이일"]),
    ]
    labels = {"fix": "고치다", "loss": "정답 ", "copy": "test를 A-1!"}
    metrics = verifier.aggregate(rows, labels)
    assert metrics["corrections"] == 1
    assert metrics["regressions"] == 1
    assert metrics["top5_losses"] == 2
    assert metrics["english_ascii_signature"]["signature_regressions"] == 1
    assert metrics["english_ascii_signature"]["literal_signature_regressions"] == 1
    assert metrics["quality_gate"] is False

    exact = verifier.aggregate(
        [record("a", "source", ["정답"], ["정답"], pool=["정답"])],
        {"a": "정답 "},
    )
    assert exact["baseline"]["top1"] == exact["proposed"]["top1"] == 0


@pytest.mark.parametrize("mutation,message", [
    (lambda rows, labels: labels.clear(), "label"),
    (lambda rows, labels: labels.__setitem__("b", "나"), "denominator"),
    (lambda rows, labels: rows.append(copy.deepcopy(rows[0])), "duplicate"),
    (lambda rows, labels: rows[0]["proposed"].__setitem__("engine_ms", float("nan")),
     "latency"),
    (lambda rows, labels: rows[0]["proposed"].__setitem__(
        "result", "후보와다름"), "selection"),
])
def test_aggregate_rejects_invalid_labels_duplicates_latency_and_public_result(
        mutation, message):
    rows = [record("a", "source", ["가"], ["가"], pool=["가"])]
    labels = {"a": "가"}
    mutation(rows, labels)
    with pytest.raises(ValueError, match=message):
        verifier.aggregate(rows, labels)


def test_join_labels_preserves_exact_text_and_rejects_normalized_missing_duplicate():
    selection = [{"id": "a", "group_id": "g1"},
                 {"id": "b", "group_id": "g2"}]
    raw = [{"id": "a", "group_id": "g1", "text": " Exact "},
           {"id": "b", "group_id": "g2", "text": "다른"}]
    romanize = lambda text: {" Exact ": " Exact ", "다른": "dareun"}[text]
    requests = {"a": " Exact ", "b": "dareun"}
    assert verifier.join_labels(
        selection, raw, requests, expected_rows=2, expected_groups=2,
        romanize=romanize) == {"a": " Exact ", "b": "다른"}

    with pytest.raises(ValueError, match="alignment"):
        verifier.join_labels(selection, raw, {"a": "exact", "b": "dareun"},
                             romanize=romanize)
    with pytest.raises(ValueError, match="incomplete"):
        verifier.join_labels(selection, raw[:1], requests, romanize=romanize)
    with pytest.raises(ValueError, match="alignment"):
        verifier.join_labels(selection, raw + [dict(raw[0])], requests,
                             romanize=romanize)


def test_validate_metrics_rejects_any_report_aggregate_drift():
    rows = [record("a", "source", ["오답"], ["가"], pool=["가"])]
    metrics = verifier.aggregate(rows, {"a": "가"})
    verifier.validate_metrics(dict(metrics), metrics)
    for key in ("baseline", "proposed", "corrections", "regressions",
                "top5_losses", "stage_counts", "english_ascii_signature",
                "improvement_gate", "quality_gate", "actual_calibrated_latency"):
        damaged = copy.deepcopy(metrics)
        damaged[key] = None
        with pytest.raises(ValueError, match="aggregate"):
            verifier.validate_metrics(damaged, metrics)


def test_authenticate_report_binds_producer_report_partition_and_shard_proofs(
        tmp_path, monkeypatch):
    evaluator_path = tmp_path / "evaluator.py"
    evaluator_path.write_text("# frozen\n", encoding="utf-8")
    partition = tmp_path / "partition"
    partition.mkdir()
    for name in verifier.PARTITION_HASHES:
        (partition / name).write_text("{}\n", encoding="utf-8")
    raw_path = tmp_path / "raw.jsonl"
    selection_path = tmp_path / "selection.jsonl"
    raw_path.write_text("", encoding="utf-8")
    selection_path.write_text("", encoding="utf-8")
    report_path = tmp_path / "report.json"
    report_path.write_text("{}\n", encoding="utf-8")
    runs = [tmp_path / "run-0", tmp_path / "run-1"]
    for directory in runs:
        directory.mkdir()

    evaluator_sha = "e" * 64
    resources = {"after_completion": {"elapsed_s": 10.0},
                 "peak_rss_bytes": 1024, "peak_swap_growth_bytes": 0}
    sources = [[], []]
    captures = [[], []]
    for index in range(1046):
        shard = 0 if index < 520 else 1
        identifier = f"id-{index:04d}"
        source = f"source-{index}"
        sources[shard].append({"id": identifier, "input": source})
        captures[shard].append({
            "id": identifier, "request": {"input": source, "context": ""}})

    proofs = []
    shard_reports = []
    evidence = []
    proof_digests = ["0" * 64, "1" * 64]
    for index, directory in enumerate(runs):
        hashes = {"report_sha256": f"report-{index}",
                  "capture_sha256": f"capture-{index}",
                  "closure_sha256": f"closure-{index}",
                  "completion_sha256": f"completion-{index}"}
        proof = {
            "schema": verifier.SHARD_PROOF_SCHEMA, "verified": True,
            "shard": index, "rows": len(captures[index]),
            "partition_hashes": verifier.PARTITION_HASHES,
            "performance_labels_opened": False, "accuracy_measured": False,
            "resources": resources,
            "verifier_sha256": verifier.SHARD_VERIFIER_SHA256,
            "producer_sha256": verifier.CAPTURE_PRODUCER_SHA256,
            "launch_sha256": f"launch-{index}", **hashes,
        }
        proofs.append(proof)
        pins = {
            str(verifier.SHARD_VERIFIER.resolve()): verifier.SHARD_VERIFIER_SHA256,
            str(verifier.CAPTURE_PRODUCER.resolve()): verifier.CAPTURE_PRODUCER_SHA256,
            **{str((partition / name).resolve()): digest
               for name, digest in verifier.PARTITION_HASHES.items()},
        }
        if index:
            pins[str((runs[0] / "independent-verification.json").resolve())] = (
                proof_digests[0])
        shard_reports.append({
            "completed": True, "shard": index, "rows": len(captures[index]),
            "partition_hashes": verifier.PARTITION_HASHES,
            "version": "version", "weights": [1.0],
            "performance_labels_opened": False, "accuracy_measured": False,
            "pins": pins,
        })
        evidence.append({"path": str(directory.resolve()),
                         "proof_sha256": proof_digests[index],
                         "resources": resources, **hashes})
    report = {
        "schema": verifier.REPORT_SCHEMA, "completed": True,
        "rows": 1046, "groups": 821, "producer_sha256": evaluator_sha,
        "reference_sha256": verifier.RAW_SHA256,
        "selection_sha256": verifier.SELECTION_SHA256,
        "labels_opened_after_both_verified_captures": True,
        "exact_original_reference_strings": True,
        "production_changed": False,
        "baseline_is_reconstructed_public_output": True,
        "baseline_latency_measured": False, "fully_blind_generalization": False,
        "version": "version", "weights": [1.0], "shards": evidence,
    }

    monkeypatch.setattr(verifier, "EVALUATOR", evaluator_path)
    monkeypatch.setattr(verifier, "SHARDS", partition)
    monkeypatch.setattr(verifier, "RAW", raw_path)
    monkeypatch.setattr(verifier, "SELECTION", selection_path)

    def fake_sha(path):
        path = Path(path)
        if path == evaluator_path:
            return evaluator_sha
        if path == raw_path:
            return verifier.RAW_SHA256
        if path == selection_path:
            return verifier.SELECTION_SHA256
        if path == verifier.SHARD_VERIFIER:
            return verifier.SHARD_VERIFIER_SHA256
        if path == verifier.CAPTURE_PRODUCER:
            return verifier.CAPTURE_PRODUCER_SHA256
        if path.parent == partition:
            return verifier.PARTITION_HASHES[path.name]
        if path == report_path:
            return "evaluation-report"
        for index, directory in enumerate(runs):
            if path == directory.with_suffix(".launch.json"):
                return f"launch-{index}"
            if path == directory / "independent-verification.json":
                return proof_digests[index]
            if path.parent == directory:
                return {"report.json": f"report-{index}",
                        "capture.jsonl": f"capture-{index}",
                        "capture.complete.json": f"closure-{index}",
                        "completion.json": f"completion-{index}"}[path.name]
        raise AssertionError(path)

    def fake_read(path):
        path = Path(path)
        if path == report_path:
            return report
        for index, directory in enumerate(runs):
            if path == directory.with_suffix(".launch.json"):
                command = [str(verifier.ROOT / ".venv/bin/python"),
                           str(verifier.CAPTURE_PRODUCER), "--output",
                           str(directory.resolve()), "--shard", str(index)]
                if index:
                    command += ["--previous", str(runs[0].resolve())]
                return {"pid": 100 + index,
                        "runner_sha256": verifier.CAPTURE_PRODUCER_SHA256,
                        "output": str(directory.resolve()), "command": command}
        index = runs.index(path.parent)
        return proofs[index] if path.name == "independent-verification.json" \
            else shard_reports[index]

    def fake_jsonl(path, expected):
        path = Path(path)
        if path.parent == partition:
            rows = sources[int(path.stem.split("-")[-1])]
        else:
            rows = captures[runs.index(path.parent)]
        assert len(rows) == expected
        return rows

    monkeypatch.setattr(verifier.pure, "sha", fake_sha)
    monkeypatch.setattr(verifier.pure, "read", fake_read)
    monkeypatch.setattr(verifier.pure, "read_exact_jsonl", fake_jsonl)
    monkeypatch.setattr(verifier.pure, "jsonl", lambda path: iter([{"trace": True}]))
    monkeypatch.setattr(verifier.pure, "verify_resources",
                        lambda rows, operations: resources)
    def terminal(pid, signal):
        raise ProcessLookupError(pid)
    monkeypatch.setattr(verifier.os, "kill", terminal)
    authenticated, records, requests, source_proofs = verifier.authenticate_report(
        report_path, runs, evaluator_sha=evaluator_sha)
    assert authenticated is report and len(records) == len(requests) == 1046
    assert source_proofs == {
        "evaluation_report_sha256": "evaluation-report",
        "evaluation_producer_sha256": evaluator_sha,
        "partition_proof_sha256":
            verifier.PARTITION_HASHES["independent-verification.json"],
        "shard_proof_sha256": proof_digests,
        "raw_reference_sha256": verifier.RAW_SHA256,
        "selection_sha256": verifier.SELECTION_SHA256,
    }

    report["producer_sha256"] = "x" * 64
    with pytest.raises(ValueError, match="identity"):
        verifier.authenticate_report(report_path, runs, evaluator_sha=evaluator_sha)
    report["producer_sha256"] = evaluator_sha

    monkeypatch.setattr(verifier.os, "kill", lambda pid, signal: None)
    with pytest.raises(ValueError, match="still present"):
        verifier.authenticate_report(report_path, runs, evaluator_sha=evaluator_sha)

    def terminal_again(pid, signal):
        raise ProcessLookupError(pid)
    monkeypatch.setattr(verifier.os, "kill", terminal_again)
    (runs[0] / "failure.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="failure evidence"):
        verifier.authenticate_report(report_path, runs, evaluator_sha=evaluator_sha)


def test_verify_authenticates_before_label_join_and_emits_no_failure_rows(
        tmp_path, monkeypatch):
    calls = []
    records = [record("a", "source", ["오답"], ["가"], pool=["가"])]
    metrics = verifier.aggregate(records, {"a": "가"})
    report = {"version": "v", "weights": [1.0], **metrics}

    def authenticate(path):
        calls.append("authenticate")
        return report, records, {"a": "source"}, {"report": "proof"}

    def read_exact(path, rows):
        assert calls == ["authenticate"]
        calls.append("selection")
        return [{"id": "a", "group_id": "g"}]

    def raw_rows(path):
        assert calls == ["authenticate", "selection"]
        calls.append("raw")
        return iter([{"id": "a", "group_id": "g", "text": "가"}])

    def joined(selection, raw, requests, **kwargs):
        assert calls == ["authenticate", "selection", "raw"]
        calls.append("labels")
        return {"a": "가"}

    monkeypatch.setattr(verifier, "authenticate_report", authenticate)
    monkeypatch.setattr(verifier.pure, "read_exact_jsonl", read_exact)
    monkeypatch.setattr(verifier.pure, "jsonl", raw_rows)
    monkeypatch.setattr(verifier, "join_labels", joined)
    monkeypatch.setattr(verifier, "aggregate", lambda captures, labels: metrics)
    monkeypatch.setattr(verifier.pure, "sha", lambda path: "verifier")
    output = tmp_path / "proof.json"
    proof = verifier.verify(tmp_path / "report.json", output)
    assert calls == ["authenticate", "selection", "raw", "labels"]
    assert proof["verified"] is True
    assert "failures" not in proof and "records" not in proof
    encoded = json.loads(output.read_text(encoding="utf-8"))
    assert encoded["metrics"] == metrics
