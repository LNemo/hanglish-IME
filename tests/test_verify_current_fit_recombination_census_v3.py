from datetime import datetime, timezone
import json

from scripts import current_fit_recombination_census as producer
from scripts import verify_current_fit_recombination_census_v3 as verifier


def segments(source, text, kind="lexicon"):
    return [{"input_start": 0, "input_end": len(source), "source": source,
             "text": text, "kind": kind}]


def no_pool_fixture():
    source = "ab"
    candidates = [{"text": text, "score": float(4 - index),
                   "segments": segments(source, text)}
                  for index, text in enumerate(("아", "애", "어", "오"))]
    expected = {"id": "s3-train-1", "input": source,
                "component_id": "component-1"}
    original = {"id": expected["id"], "component_id": expected["component_id"],
                "request": {"input": source, "context": ""}, "base_paths": [],
                "response": {"input": source, "context": "",
                             "external_context_used": False,
                             "sentence_ranking_applied": False,
                             "candidates": candidates}}
    record = producer.prepare_record(original, expected, 0, 0, "c" * 64)
    return expected, original, record


def test_real_producer_no_pool_nonraw_fallback_seeds_verify_as_unreachable():
    _, original, record = no_pool_fixture()
    assert record["status"] == "no_pool" and record["proposals"] == []
    assert len(record["diagnostics"]["seeds"]) == 4
    assert {seed["termination_reason"] for seed in record["diagnostics"]["seeds"]} == {
        "no_complete_path"}
    result = verifier.verify_row(record, original, 0, 0, "c" * 64)
    assert result == {"proposals": 0, "copy_edges": 0, "pool_rows": 0}
    record["diagnostics"]["seeds"][0]["termination_reason"] = "exhausted"
    try:
        verifier.verify_row(record, original, 0, 0, "c" * 64)
    except ValueError as error:
        assert "no-pool" in str(error)
    else:
        raise AssertionError("invalid no-pool termination was accepted")


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def resource(stage, operation, elapsed):
    return {"at": datetime.now(timezone.utc).isoformat(), "stage": stage,
            "operation": operation, "operation_elapsed_s": .01 if operation else None,
            "elapsed_s": elapsed, "rss_bytes": 100,
            "swap_used_bytes": 1000, "swap_growth_bytes": 0}


def resources(operations):
    result, elapsed = [], 0.0
    for index, operation in enumerate(operations):
        elapsed += .1; result.append(resource("operation_start", operation, elapsed))
        elapsed += .1; result.append(resource("operation_end", operation, elapsed))
        if index == len(operations) - 2:
            elapsed += .1; result.append(resource("after_report", None, elapsed))
    elapsed += .1; result.append(resource("after_completion", None, elapsed))
    return result


def test_full_capture_mode_verification_accepts_closed_no_pool_record(tmp_path, monkeypatch):
    expected, original, record = no_pool_fixture()
    output = (tmp_path / "capture").resolve(); output.mkdir()
    encoded = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    (output / "capture.jsonl").write_text(encoded)
    capture_sha = verifier.sha256_file(output / "capture.jsonl")
    closure = {"schema": "current-fit-recombination-census-closure-v1",
               "completed": True, "mode": "capture", "shard": 0, "rows": 1,
               "capture_sha256": capture_sha, "labels_opened": False,
               "model_loaded": False, "model_inference": False}
    dump(output / "capture.complete.json", closure)
    producer_sha = verifier.sha256_file(verifier.PRODUCER)
    probe_directory = str((tmp_path / "probe").resolve())
    binding = {"directory": probe_directory, "report_sha256": "a" * 64,
               "capture_sha256": "b" * 64, "closure_sha256": "c" * 64,
               "completion_sha256": "d" * 64, "proof_sha256": "e" * 64}
    projected_seconds = [100.0] * 6
    projected_bytes = [1000] * 6
    pins = {str(verifier.PRODUCER.resolve()): producer_sha}
    counts, reasons = verifier._counts([record])
    report = {"schema": "current-fit-recombination-census-v1", "completed": True,
              "mode": "capture", "shard": 0, "rows": 1, "ids": [expected["id"]],
              "counts": counts, "seed_termination_reasons": reasons,
              "timing_seconds": {expected["id"]: .01}, "initialization_seconds": .1,
              "projected_shard_seconds": projected_seconds, "qualified": True,
              "serialized_output_bytes": {expected["id"]: len(encoded.encode())},
              "capture_output_bytes": len(encoded.encode()),
              "projected_shard_output_bytes": projected_bytes,
              "projected_total_output_bytes": sum(projected_bytes),
              "limits": verifier.LIMITS, "disk_limits": verifier.DISK_LIMITS,
              "pins": pins, "capture_sha256": capture_sha,
              "closure_sha256": verifier.sha256_file(output / "capture.complete.json"),
              "producer_sha256": producer_sha,
              "generator_sha256": verifier.STATIC_HASHES[verifier.GENERATOR],
              "generator_schema": "source-edge-recombination-v1",
              "generator_limits": verifier.EXPECTED_LIMITS,
              "probe_binding": binding, "source_only": True, "train_only": True,
              "labels_opened": False, "fit_feature_files_opened": False,
              "dev_opened": False, "test_opened": False, "model_loaded": False,
              "model_inference": False, "neural_training": False,
              "production_changed": False}
    dump(output / "pins.json", pins); dump(output / "report.json", report)
    operations = ["authenticate", "propose:" + expected["id"],
                  "write_report", "write_completion"]
    trace = resources(operations)
    (output / "resources.jsonl").write_text("".join(json.dumps(row) + "\n" for row in trace))
    dump(output / "completion.json", {
        "completed": True, "report_sha256": verifier.sha256_file(output / "report.json"),
        "resource_after_report": next(row for row in trace if row["stage"] == "after_report")})
    dump(output.with_suffix(".launch.json"), {
        "pid": 99999999,
        "command": [str(verifier.ROOT / ".venv/bin/python"), "-m",
                    "scripts.current_fit_recombination_census", "--mode", "capture",
                    "--output", str(output), "--shard", "0", "--probe", probe_directory,
                    "--probe-report-sha256", "a" * 64],
        "output": str(output), "runner_sha256": producer_sha})
    prepared = {"rows": [expected], "groups": [], "shards": [[expected["id"]], [], [], [], [], []]}
    monkeypatch.setattr(verifier, "authenticate_inputs",
                        lambda: (prepared, {expected["id"]: expected}, {}))
    monkeypatch.setattr(verifier, "_authenticate_probe",
                        lambda *_: ({"projected_shard_seconds": projected_seconds,
                                     "projected_shard_output_bytes": projected_bytes,
                                     "projected_total_output_bytes": sum(projected_bytes)}, {}))
    monkeypatch.setattr(verifier, "_selected_source_rows",
                        lambda *_: ([expected["id"]], iter([(original, expected, 0, 0, "c" * 64)])))
    proof = verifier.verify(output, producer_sha)
    assert proof["verified"] is True and proof["mode"] == "capture"
    assert proof["counts"]["no_complete_path_seeds"] == 4
