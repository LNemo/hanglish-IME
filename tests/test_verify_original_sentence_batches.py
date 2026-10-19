import copy
import json
from datetime import datetime, timezone

import pytest

from scripts import verify_original_sentence_batches as verifier


def original(identifier="probe-00"):
    return {
        "id": identifier,
        "component_id": "component-a",
        "request": {"input": "abc", "context": ""},
        "base_paths": [
            ["한bc", 1.25, [
                {"input_start": 0, "input_end": 1, "source": "a", "text": "한", "kind": "lexicon"},
                {"input_start": 1, "input_end": 3, "source": "bc", "text": "bc", "kind": "raw"},
            ]],
            ["abc", -2.0, [
                {"input_start": 0, "input_end": 3, "source": "abc", "text": "abc", "kind": "raw"},
            ]],
        ],
    }


def trace(source=None, shard=2, index=7, capture_sha="c" * 64):
    source = source or original()
    paths = source["base_paths"]
    return {
        "id": source["id"], "component_id": source["component_id"], "request": source["request"],
        "source_shard": shard, "source_row_index": index,
        "original_capture_sha256": capture_sha,
        "original_order": [path[0] for path in paths],
        "original_scores": [path[1] for path in paths],
        "original_paths_sha256": verifier.canonical(paths),
        "calls": [{"contexts": ["", ""], "texts": ["abc", "한bc"],
                   "scores": [-2.0, 1.25], "token_lengths": [3, 4], "batch_size": 8}],
        "replay_max_absolute_difference": 0.0, "tolerance": 1e-5,
    }


def resource_rows(identifiers, replay_seconds=1.0):
    operations = ["authenticate_sources", "load_original_engine"]
    operations += [f"replay:{identifier}" for identifier in identifiers]
    operations += ["final_rehash"]
    rows = []
    elapsed = 0.0
    at = datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat()
    for operation in operations:
        update = operation.startswith("replay:")
        rows.append({"at": at, "elapsed_s": elapsed, "operation": operation,
                     "operation_elapsed_s": 0.0 if update else None, "rss_bytes": 100,
                     "stage": "operation_start", "swap_growth_bytes": 0, "swap_used_bytes": 5})
        elapsed += replay_seconds if update else 0.1
        rows.append({"at": at, "elapsed_s": elapsed, "operation": operation,
                     "operation_elapsed_s": replay_seconds if update else None, "rss_bytes": 110,
                     "stage": "operation_end", "swap_growth_bytes": 0, "swap_used_bytes": 5})
    rows.append({"at": at, "elapsed_s": elapsed, "operation": None,
                 "operation_elapsed_s": None, "rss_bytes": 110,
                 "stage": "after_final_rehash", "swap_growth_bytes": 0, "swap_used_bytes": 5})
    return rows, operations


def test_validate_trace_accepts_reordered_runtime_calls_and_exact_parity():
    summary = verifier.validate_trace(original(), trace(), 2, 7, "c" * 64)
    assert summary == {"originals": 2, "calls": 1, "max_absolute_difference": 0.0}


@pytest.mark.parametrize("mutate,match", [
    (lambda row: row["calls"][0].update(batch_size=True), "call shape"),
    (lambda row: row["calls"][0].update(contexts=["x", ""]), "call shape"),
    (lambda row: row["calls"][0].update(texts=["abc", "abc"]), "duplicate"),
    (lambda row: row["calls"][0].update(token_lengths=[3, 0]), "call shape"),
    (lambda row: row["calls"][0].update(scores=[-2.0, float("inf")]), "call shape"),
    (lambda row: row.update(original_order=["abc", "한bc"]), "pool/order"),
    (lambda row: row.update(replay_max_absolute_difference=1e-6), "runtime parity"),
])
def test_validate_trace_rejects_bad_call_and_pool_contracts(mutate, match):
    row = trace(); mutate(row)
    with pytest.raises(ValueError, match=match):
        verifier.validate_trace(original(), row, 2, 7, "c" * 64)


def test_validate_trace_rejects_segment_source_not_at_claimed_offsets():
    source = original()
    source["base_paths"][0][2][0]["source"] = "b"
    row = trace(source)
    with pytest.raises(ValueError, match="source slice"):
        verifier.validate_trace(source, row, 2, 7, "c" * 64)


def test_validate_trace_rejects_score_difference_over_fixed_tolerance():
    row = trace()
    row["calls"][0]["scores"][1] += 2e-5
    row["replay_max_absolute_difference"] = pytest.approx(2e-5)
    with pytest.raises(ValueError, match="runtime parity"):
        verifier.validate_trace(original(), row, 2, 7, "c" * 64)


def test_resources_accept_unbounded_operations_without_fabricated_clocks():
    rows, operations = resource_rows(["probe"])
    result = verifier.verify_resources(rows, operations)
    assert result["operations"] == 4
    assert result["after_final_rehash"] == rows[-1]


def test_resources_reject_slow_replay_and_clock_on_unbounded_operation():
    rows, operations = resource_rows(["probe"], replay_seconds=30.1)
    with pytest.raises(ValueError, match="30 seconds"):
        verifier.verify_resources(rows, operations)
    rows, operations = resource_rows(["probe"])
    rows[0]["operation_elapsed_s"] = 0.0
    with pytest.raises(ValueError, match="attribution"):
        verifier.verify_resources(rows, operations)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def test_full_verification_writes_scoped_proof_from_synthetic_closed_artifact(tmp_path, monkeypatch):
    output = tmp_path / "output"; output.mkdir()
    runner = tmp_path / "runner.py"; runner.write_text("# frozen runner\n")
    prepared = tmp_path / "prepared.json"; prepared.write_text("{}\n")
    source_files = []
    for shard in range(6):
        directory = tmp_path / f"source-{shard}"; directory.mkdir()
        capture = directory / "capture.jsonl"; capture.write_text("{}\n"); source_files.append(capture)
    identifiers = [f"probe-{index:02d}" for index in range(16)]
    sources = {}
    capture_rows = []
    for index, identifier in enumerate(identifiers):
        item = original(identifier)
        sources[identifier] = (item, index % 6, index)
        capture_rows.append(trace(item, index % 6, index, verifier.sha(source_files[index % 6])))
    (output / "capture.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in capture_rows))
    resources, operations = resource_rows(identifiers)
    (output / "resources.jsonl").write_text("".join(json.dumps(row) + "\n" for row in resources))
    runner_sha = verifier.sha(runner)
    pins = {str(prepared.resolve()): verifier.sha(prepared), str(runner.resolve()): runner_sha}
    pins.update({str(path.resolve()): verifier.sha(path) for path in source_files})
    write_json(output / "pins.json", pins)
    report = {
        "schema": "original-sentence-batches-probe-v1", "completed": True, "rows": 16,
        "ids": identifiers, "capture_sha256": verifier.sha(output / "capture.jsonl"),
        "pins": pins, "limits": verifier.LIMITS, "tolerance": 1e-5,
        "max_original_score_difference": 0.0, "terminal": resources[-1],
        "source_only": True, "labels_opened": False, "phonetic_inference": True,
        "training": False, "production_changed": False, "runner_sha256": runner_sha,
    }
    write_json(output / "report.json", report)
    write_json(output / "completion.json", {"completed": True,
                                              "report_sha256": verifier.sha(output / "report.json")})
    write_json(tmp_path / "output.launch.json", {
        "pid": 999999, "command": [str((verifier.ROOT / ".venv/bin/python").resolve()),
                                    str(runner.resolve()), runner_sha],
        "runner_sha256": runner_sha, "output": str(output.resolve())})
    monkeypatch.setattr(verifier, "PREPARED", prepared)
    monkeypatch.setattr(verifier, "PREPARED_SHA", verifier.sha(prepared))
    monkeypatch.setattr(verifier, "SOURCE_DIRS", tuple(path.parent for path in source_files))
    monkeypatch.setattr(verifier, "SOURCE_HASHES", tuple(verifier.sha(path) for path in source_files))
    monkeypatch.setattr(verifier, "selected_sources", lambda: (identifiers, sources))
    monkeypatch.setattr(verifier.os, "kill", lambda pid, signal: (_ for _ in ()).throw(ProcessLookupError()))
    proof = verifier.verify(output.resolve(), runner_sha)
    assert proof["verified"] is True
    assert proof["model_inference_replayed"] is False
    assert proof["max_original_score_difference"] == 0.0
    assert (output / "independent-verification.json").is_file()


def test_full_verification_rejects_watchdog_failure_artifact(tmp_path):
    output = tmp_path.resolve(); (output / "resource-watchdog-failure.json").write_text("{}")
    with pytest.raises(ValueError, match="failure artifact"):
        verifier.verify(output, "0" * 64)
