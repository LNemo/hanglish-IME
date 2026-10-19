import hashlib
import json
from pathlib import Path

import pytest

from scripts import prepare_current_phonetic_train_features as prepare


def _source(identifier, component):
    return {"id": identifier, "input": "abc", "component_id": component}


def _capture(identifier, component, candidate="정답"):
    segment = {"input_start": 0, "input_end": 3, "source": "abc",
               "text": candidate, "kind": "fixture"}
    path = [candidate, 2.5, [segment]]
    return {
        "id": identifier, "component_id": component,
        "request": {"input": "abc", "context": ""}, "elapsed_s": 0.1,
        "base_paths": [path],
        "response": {
            "sentence_ranking_applied": True,
            "candidates": [{"text": candidate, "score": 2.5,
                            "segments": [segment]}],
            "phonetic_word_union": {"production_pool_texts": [candidate]},
            "phonetic_choice": {"rows": [{
                "text": candidate, "base_score": 2.5,
                "segments": [segment], "features": [-0.5, 1.0, 0.0, 0.0],
            }]},
        },
    }


def _line(identifier, split, expected=None, component="component-1"):
    value = {"id": identifier, "split": split, "input": "abc",
             "component_id": component, "expected": expected or ["정답"]}
    return json.dumps(value, ensure_ascii=False, sort_keys=True).encode() + b"\n"


def test_scan_decodes_only_allowlisted_fit_and_skips_invalid_inner_hold_json(tmp_path):
    path = tmp_path / "mixed.jsonl"
    invalid_inner = (b'{"id": "s3-train-2", "split": "inner_hold", '
                     b'"expected": THIS_IS_NOT_JSON}\n')
    path.write_bytes(_line("s3-train-1", "fit") + invalid_inner)
    labels = prepare.scan_fit_labels(
        path, {"s3-train-1": _source("s3-train-1", "component-1")},
        expected_mixed_rows=2, expected_inner_hold_rows=1)
    assert labels == {"s3-train-1": ["정답"]}


def test_scan_rejects_more_than_one_serialized_id_before_json_decode(tmp_path):
    path = tmp_path / "mixed.jsonl"
    path.write_bytes(
        b'{"id":"s3-train-1","id":"s3-train-9","split":"fit",'
        b'"expected":["x"]}\n')
    with pytest.raises(ValueError, match="ambiguous"):
        prepare.scan_fit_labels(
            path, {"s3-train-1": _source("s3-train-1", "component-1")},
            expected_mixed_rows=1, expected_inner_hold_rows=0)


@pytest.mark.parametrize(
    "identifier,split,message",
    [("s3-train-1", "inner_hold", "wrong split"),
     ("s3-train-2", "fit", "not inner-HOLD")],
)
def test_scan_enforces_fit_allowlist_boundary(tmp_path, identifier, split, message):
    path = tmp_path / "mixed.jsonl"
    path.write_bytes(_line(identifier, split))
    allowlist = {"s3-train-1": _source("s3-train-1", "component-1")}
    with pytest.raises(ValueError, match=message):
        prepare.scan_fit_labels(path, allowlist,
                                expected_mixed_rows=1,
                                expected_inner_hold_rows=int(split == "inner_hold"))


def test_scan_rejects_fit_source_or_component_mismatch(tmp_path):
    path = tmp_path / "mixed.jsonl"
    path.write_bytes(_line("s3-train-1", "fit", component="other"))
    with pytest.raises(ValueError, match="target differs"):
        prepare.scan_fit_labels(
            path, {"s3-train-1": _source("s3-train-1", "component-1")},
            expected_mixed_rows=1, expected_inner_hold_rows=0)


def test_project_preserves_full_pool_features_and_derives_indices():
    capture = _capture("s3-train-1", "component-1")
    row = prepare.project_feature_row(capture, ["정답", "허용"], 4)
    assert row == {
        "id": "s3-train-1", "input": "abc", "component_id": "component-1",
        "source_shard": 4, "ranked": True, "training_eligible": True,
        "candidates": ["정답"], "base_scores": [2.5],
        "features": [[-0.5, 1.0, 0.0, 0.0]], "positive_indices": [0],
        "production_indices": [0], "public_candidates": ["정답"],
        "expected": ["정답", "허용"],
    }


def test_project_keeps_uncovered_ranked_row_ineligible_without_injection():
    capture = _capture("s3-train-1", "component-1", candidate="오답")
    row = prepare.project_feature_row(capture, ["정답"], 0)
    assert row["candidates"] == ["오답"]
    assert row["positive_indices"] == []
    assert row["ranked"] is True
    assert row["training_eligible"] is False


def test_project_keeps_unranked_row_with_empty_candidate_domain():
    capture = _capture("s3-train-1", "component-1")
    capture["base_paths"] = []
    capture["response"]["sentence_ranking_applied"] = False
    capture["response"]["phonetic_choice"]["rows"] = []
    capture["response"]["phonetic_word_union"]["production_pool_texts"] = []
    row = prepare.project_feature_row(capture, ["정답"], 2)
    assert row["ranked"] is False
    assert row["training_eligible"] is False
    assert row["candidates"] == row["positive_indices"] == []


def test_process_absent_accepts_only_missing_pid():
    def missing(pid, signal):
        assert pid == 42 and signal == 0
        raise ProcessLookupError

    assert prepare.process_absent(42, missing) is True
    with pytest.raises(ValueError, match="still present"):
        prepare.process_absent(42, lambda pid, signal: None)


def test_capture_row_iterator_is_lazy_and_keeps_only_current_row(monkeypatch):
    events = []
    rows = [_capture("s3-train-1", "component-1"),
            _capture("s3-train-2", "component-2")]
    sources = {row["id"]: _source(row["id"], row["component_id"])
               for row in rows}

    def source_stream(unused):
        events.append("opened")
        for row in rows:
            events.append("yield:" + row["id"])
            yield row
        events.append("closed")

    monkeypatch.setattr(prepare, "jsonl", source_stream)
    iterator = prepare.iter_capture_rows(
        Path("unused"), [row["id"] for row in rows], sources)
    assert events == []
    assert next(iterator)["id"] == "s3-train-1"
    assert events == ["opened", "yield:s3-train-1"]
    assert next(iterator)["id"] == "s3-train-2"
    with pytest.raises(StopIteration):
        next(iterator)
    assert events[-1] == "closed"


def test_expected_capture_launch_is_exact():
    directory = prepare.SHARDS[3].resolve()
    assert prepare.expected_launch(directory, 3) == [
        str(prepare.ROOT / ".venv/bin/python"), "-m",
        "scripts.current_phonetic_train_pool", "--mode", "capture",
        "--output", str(directory), "--shard", "3",
    ]


def _terminal_shard_fixture(tmp_path):
    directory = tmp_path / "shard-0-v1"
    directory.mkdir()
    source = _source("s3-train-1", "component-1")
    capture_path = directory / "capture.jsonl"
    capture_path.write_text(json.dumps(_capture(source["id"], source["component_id"]),
                                       ensure_ascii=False) + "\n")
    capture_sha = hashlib.sha256(capture_path.read_bytes()).hexdigest()
    closure = {"mode": "capture", "shard": 0, "rows": 1,
               "capture_sha256": capture_sha, "labels_opened": False}
    (directory / "capture.complete.json").write_text(json.dumps(closure))
    pins = {str((prepare.PREPARED / name).resolve()): digest
            for name, digest in prepare.PREPARED_HASHES.items()}
    producer = prepare.ROOT / "scripts/current_phonetic_train_pool.py"
    pins[str(producer.resolve())] = prepare.PRODUCER_SHA256
    (directory / "pins.json").write_text(json.dumps(pins))
    report = {"completed": True, "mode": "capture", "shard": 0, "rows": 1,
              "ids": [source["id"]], "pins": pins,
              "capture_sha256": capture_sha,
              "closure_sha256": hashlib.sha256(
                  (directory / "capture.complete.json").read_bytes()).hexdigest(),
              "qualified": True, "labels_opened": False}
    (directory / "report.json").write_text(json.dumps(report))
    report_sha = hashlib.sha256((directory / "report.json").read_bytes()).hexdigest()
    (directory / "completion.json").write_text(json.dumps(
        {"completed": True, "report_sha256": report_sha}))
    proof = {"verified": True, "mode": "capture", "shard": 0,
             "producer_sha256": prepare.PRODUCER_SHA256,
             "verifier_sha256": prepare.VERIFIER_SHA256,
             "report_sha256": report_sha, "capture_sha256": capture_sha,
             "closure_sha256": report["closure_sha256"],
             "completion_sha256": hashlib.sha256(
                 (directory / "completion.json").read_bytes()).hexdigest(),
             "counts": {"rows": 1}}
    log = directory.with_suffix(".log")
    log.write_text("done\n")
    launch = {"pid": 42, "command": prepare.expected_launch(directory, 0),
              "output": str(directory.resolve()), "log": str(log),
              "runner_sha256": prepare.PRODUCER_SHA256,
              "launched_at": "2026-09-23T00:00:00+00:00"}
    directory.with_suffix(".launch.json").write_text(json.dumps(launch))
    proof["launch_sha256"] = hashlib.sha256(
        directory.with_suffix(".launch.json").read_bytes()).hexdigest()
    (directory / "independent-verification.json").write_text(json.dumps(proof))
    return directory, source


def test_authenticate_shard_binds_terminal_proof_and_uses_pid_stub(tmp_path):
    directory, source = _terminal_shard_fixture(tmp_path)

    def missing(pid, signal):
        assert (pid, signal) == (42, 0)
        raise ProcessLookupError

    evidence = prepare.authenticate_shard(
        directory, 0, [source["id"]], {source["id"]: source}, missing)
    assert evidence["rows"] == 1
    assert evidence["completion_sha256"] == hashlib.sha256(
        (directory / "completion.json").read_bytes()).hexdigest()


def test_authenticate_shard_rejects_wrong_fixed_verifier(tmp_path):
    directory, source = _terminal_shard_fixture(tmp_path)
    proof_path = directory / "independent-verification.json"
    proof = json.loads(proof_path.read_text())
    proof["verifier_sha256"] = "0" * 64
    proof_path.write_text(json.dumps(proof))
    with pytest.raises(ValueError, match="proof differs"):
        prepare.authenticate_shard(
            directory, 0, [source["id"]], {source["id"]: source},
            lambda pid, signal: (_ for _ in ()).throw(ProcessLookupError()))


@pytest.mark.parametrize("field", ["launch_sha256", "completion_sha256"])
def test_authenticate_shard_rejects_unlinked_terminal_artifact(tmp_path, field):
    directory, source = _terminal_shard_fixture(tmp_path)
    proof_path = directory / "independent-verification.json"
    proof = json.loads(proof_path.read_text())
    proof[field] = "0" * 64
    proof_path.write_text(json.dumps(proof))
    with pytest.raises(ValueError, match="proof differs"):
        prepare.authenticate_shard(
            directory, 0, [source["id"]], {source["id"]: source},
            lambda pid, signal: (_ for _ in ()).throw(ProcessLookupError()))


def test_run_closes_sources_before_mixed_scan_and_preserves_all_rows(
        tmp_path, monkeypatch):
    ids = ["s3-train-1", "s3-train-2"]
    sources = {ids[0]: _source(ids[0], "component-1"),
               ids[1]: _source(ids[1], "component-2")}
    requests = {"rows": list(sources.values()),
                "shards": [[ids[0]], [ids[1]], [], [], [], []]}
    captures = {ids[0]: _capture(ids[0], "component-1", "정답"),
                ids[1]: _capture(ids[1], "component-2", "오답")}
    mixed = tmp_path / "mixed.jsonl"
    mixed.write_bytes(
        _line(ids[0], "fit", ["정답"], "component-1")
        + _line(ids[1], "fit", ["정답"], "component-2")
        + b'{"id":"s3-train-3","split":"inner_hold",'
          b'"expected":NOT_JSON}\n')
    mixed_sha = hashlib.sha256(mixed.read_bytes()).hexdigest()
    monkeypatch.setattr(prepare, "ROWS", 2)
    monkeypatch.setattr(prepare, "MIXED_ROWS", 3)
    monkeypatch.setattr(prepare, "INNER_HOLD_ROWS", 1)
    monkeypatch.setattr(prepare, "MIXED_SHA256", mixed_sha)
    monkeypatch.setattr(prepare, "authenticate_prepared",
                        lambda: (requests, sources))
    monkeypatch.setattr(prepare, "authenticate_mixed_metadata", lambda: None)

    def fake_shard(directory, shard, expected, source_map, pid_probe):
        del directory, source_map, pid_probe
        return {"shard": shard, "rows": len(expected),
                "capture_sha256": f"capture-{shard}",
                "report_sha256": f"report-{shard}",
                "proof_sha256": f"proof-{shard}",
                "launch_sha256": f"launch-{shard}",
                "completion_sha256": f"completion-{shard}"}

    monkeypatch.setattr(prepare, "authenticate_shard", fake_shard)
    active = {"current": 0, "maximum": 0, "completed": []}

    def streamed(directory, expected, source_map):
        del source_map
        active["current"] += 1
        active["maximum"] = max(active["maximum"], active["current"])
        try:
            for identifier in expected:
                yield captures[identifier]
        finally:
            active["current"] -= 1
            active["completed"].append(Path(directory).name)

    monkeypatch.setattr(prepare, "iter_capture_rows", streamed)
    original_sha = prepare.sha

    def synthetic_sha(path):
        path = Path(path)
        if path.name == "capture.jsonl" and path.parent.name.isdigit():
            return f"capture-{path.parent.name}"
        return original_sha(path)

    monkeypatch.setattr(prepare, "sha", synthetic_sha)
    original_scan = prepare.scan_fit_labels
    output = tmp_path / "output"

    def observing_scan(path, allowlist):
        closure = output / "source-capture.complete.json"
        assert closure.is_file()
        assert json.loads(closure.read_text())["labels_opened"] is False
        return original_scan(path, allowlist)

    monkeypatch.setattr(prepare, "scan_fit_labels", observing_scan)
    report = prepare.run(output, shard_directories=[tmp_path / str(i)
                                                    for i in range(6)],
                         mixed_path=mixed, pid_probe=lambda *unused: None)
    assert report["counts"] == {
        "rows": 2, "ranked": 2, "unranked": 0, "covered": 1,
        "uncovered": 1, "train_eligible": 1, "full_pool_candidates": 2,
    }
    assert report["mixed_container_bytes_read"] is True
    assert report["non_fit_json_decoded"] is False
    assert report["non_fit_target_values_used"] is False
    rows = [json.loads(line) for line in (output / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert rows[1]["positive_indices"] == []
    assert rows[1]["training_eligible"] is False
    assert active["maximum"] == 1
    assert active["current"] == 0
    assert active["completed"] == [str(i) for i in range(6)]
