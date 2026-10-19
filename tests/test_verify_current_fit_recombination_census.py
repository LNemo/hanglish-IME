import ast
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

import pytest

from scripts import verify_current_fit_recombination_census as verifier
from scripts import verify_current_fit_recombination_census_v2 as verifier_v2


def segments(source, text):
    return [{"input_start": 0, "input_end": len(source), "source": source,
             "text": text, "kind": "raw" if source == text else "lexicon"}]


def source_row(source="ab", paths=None, ranked=True):
    paths = paths if paths is not None else [["아", 1.0, segments(source, "아")]]
    public = ([{"text": paths[0][0], "score": paths[0][1], "segments": paths[0][2]}]
              if paths else [])
    return {"id": "s3-train-1", "component_id": "component-1",
            "request": {"input": source, "context": ""}, "base_paths": paths,
            "response": {"input": source, "context": "", "external_context_used": False,
                         "candidates": public, "sentence_ranking_applied": ranked}}


def reference(row, shard=0, index=0, capture="c" * 64):
    return {"source_shard": shard, "source_row_index": index,
            "source_capture_sha256": capture,
            "base_paths_sha256": verifier.canonical_sha(row["base_paths"]),
            "public_candidates_sha256": verifier.canonical_sha(row["response"]["candidates"]),
            "pool_rows": len(row["base_paths"]),
            "ranked": row["response"]["sentence_ranking_applied"]}


def test_verifier_has_no_producer_or_model_imports():
    tree = ast.parse(Path(verifier.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert "scripts.current_fit_recombination_census" not in imported
    assert not any(name.startswith(("torch", "transformers")) for name in imported)


def test_segments_require_exact_contiguous_source_and_output():
    valid = [{"input_start": 0, "input_end": 1, "source": "a", "text": "아", "kind": "x"},
             {"input_start": 1, "input_end": 2, "source": "b", "text": "비", "kind": "x"}]
    verifier.validate_segments("ab", "아비", valid)
    for bad in ([{**valid[0], "input_end": 2}, valid[1]],
                [valid[0], {**valid[1], "source": "x"}],
                [valid[0], {**valid[1], "text": "x"}]):
        with pytest.raises(ValueError, match="segment"):
            verifier.validate_segments("ab", "아비", bad)


def test_proposal_must_be_novel_and_exactly_bound_to_captured_edge_evidence():
    row = source_row()
    inventory, occurrences = verifier.edge_inventory(row["id"], "ab", row["base_paths"])
    assert occurrences == 1
    edge = inventory[0]
    proposal = {"proposal_id": verifier.stable_id("proposal-v1-", {
                    "source": "ab", "edge_ids": [edge["edge_id"]]}),
                "text": "아", "segments": segments("ab", "아"),
                "edge_ids": [edge["edge_id"]],
                "seed_index": 0, "seed_text": "seed", "changed_edges": 1,
                "score": None, "score_type": None,
                "origin": "source_edge_recombination", "selection_eligible": False}
    with pytest.raises(ValueError, match="novelty"):
        verifier.verify_proposal("ab", proposal, inventory, {"아"})
    proposal["text"] = "새"
    proposal["segments"] = segments("ab", "새")
    with pytest.raises(ValueError, match="unauthenticated"):
        verifier.verify_proposal("ab", proposal, inventory, set())
    proposal["text"] = "아"; proposal["segments"] = segments("ab", "아")
    assert verifier.verify_proposal("ab", proposal, inventory, set()) == 0
    forged = [{**edge, "provenance": []}]
    with pytest.raises(ValueError, match="unauthenticated"):
        verifier.verify_proposal("ab", proposal, [], set())
    with pytest.raises(ValueError, match="used edge"):
        verifier.verify_used_inventory(forged, inventory, [edge["edge_id"]])


def test_no_pool_row_preserves_fallback_and_has_no_diagnostics():
    row = source_row(source="123", paths=[], ranked=False)
    raw = {"text": "123", "score": None, "segments": segments("123", "123")}
    row["response"]["candidates"] = [raw]
    record = {"schema": "current-fit-recombination-census-row-v1", "id": row["id"],
              "component_id": row["component_id"], "request": row["request"],
              "source_reference": reference(row), "status": "no_pool", "proposals": [],
              "used_edge_inventory": [],
              "diagnostics": {"limits": verifier.EXPECTED_LIMITS, "source_length": 3,
                              "pool_rows": 0, "edge_occurrences": 0, "edge_count": 0,
                              "seed_count": 0, "seeds": [], "proposal_count": 0,
                              "global_limit_reached": False, "original_text_count": 1,
                              "score_fields_consumed": False,
                              "complete_paths_rejected_as_existing_or_duplicate": 0},
              "fallback_public": [raw]}
    assert verifier.verify_row(record, row, 0, 0, "c" * 64)["pool_rows"] == 0
    record["fallback_public"] = []
    with pytest.raises(ValueError, match="no-pool"):
        verifier.verify_row(record, row, 0, 0, "c" * 64)


def test_ranked_row_recombination_binds_unique_used_edges_and_provenance():
    source = "ab"
    first_segments = [
        {"input_start": 0, "input_end": 1, "source": "a", "text": "아", "kind": "lexicon"},
        {"input_start": 1, "input_end": 2, "source": "b", "text": "비", "kind": "lexicon"},
    ]
    second_segments = [
        {"input_start": 0, "input_end": 1, "source": "a", "text": "애", "kind": "lexicon"},
        {"input_start": 1, "input_end": 2, "source": "b", "text": "베", "kind": "lexicon"},
    ]
    row = source_row(source, [["아비", 2.0, first_segments], ["애베", 1.0, second_segments]])
    inventory, occurrences = verifier.edge_inventory(row["id"], source, row["base_paths"])
    chosen = [next(edge for edge in inventory if edge["structural_segment"]["text"] == text)
              for text in ("아", "베")]
    edge_ids = [edge["edge_id"] for edge in chosen]
    proposal = {"proposal_id": verifier.stable_id("proposal-v1-", {"source": source,
                                                                    "edge_ids": edge_ids}),
                "text": "아베", "segments": [first_segments[0], second_segments[1]],
                "edge_ids": edge_ids, "seed_index": 0, "seed_text": "아비",
                "changed_edges": 1, "score": None, "score_type": None,
                "origin": "source_edge_recombination", "selection_eligible": False}
    diagnostics = {"limits": verifier.EXPECTED_LIMITS, "source_length": 2,
                   "pool_rows": 2, "edge_occurrences": occurrences,
                   "edge_count": len(inventory), "seed_count": 1,
                   "seeds": [{"seed_index": 0, "base_candidate_index": 0,
                              "seed_text": "아비", "pops": 1, "max_frontier": 1,
                              "complete_paths_examined": 1, "accepted_proposals": 1,
                              "termination_reason": "exhausted"}],
                   "proposal_count": 1, "global_limit_reached": False,
                   "original_text_count": 3, "score_fields_consumed": False,
                   "complete_paths_rejected_as_existing_or_duplicate": 0}
    record = {"schema": "current-fit-recombination-census-row-v1", "id": row["id"],
              "component_id": row["component_id"], "request": row["request"],
              "source_reference": reference(row), "status": "generated",
              "proposals": [proposal], "used_edge_inventory": chosen,
              "diagnostics": diagnostics, "fallback_public": None}
    assert verifier.verify_row(record, row, 0, 0, "c" * 64)["proposals"] == 1
    record["used_edge_inventory"][0] = {**chosen[0], "provenance": []}
    with pytest.raises(ValueError, match="used edge"):
        verifier.verify_row(record, row, 0, 0, "c" * 64)


def test_v2_replays_real_producer_record_with_hangul_edge_ids():
    from scripts import current_fit_recombination_census as producer

    source = "ab"
    first = [
        {"input_start": 0, "input_end": 1, "source": "a", "text": "아", "kind": "lexicon"},
        {"input_start": 1, "input_end": 2, "source": "b", "text": "비", "kind": "lexicon"},
    ]
    second = [
        {"input_start": 0, "input_end": 1, "source": "a", "text": "애", "kind": "lexicon"},
        {"input_start": 1, "input_end": 2, "source": "b", "text": "베", "kind": "lexicon"},
    ]
    paths = [["아비", 2.0, first], ["애베", 1.0, second]]
    candidates = [{"text": text, "score": score, "segments": pieces}
                  for text, score, pieces in paths]
    expected = {"id": "s3-train-1", "input": source, "component_id": "component-1"}
    original = {"id": expected["id"], "component_id": expected["component_id"],
                "request": {"input": source, "context": ""}, "base_paths": paths,
                "response": {"input": source, "context": "", "external_context_used": False,
                             "sentence_ranking_applied": True, "candidates": candidates}}
    record = producer.prepare_record(original, expected, 0, 0, "c" * 64)
    assert record["proposals"] and record["used_edge_inventory"]
    old = verifier.stable_id
    try:
        verifier.stable_id = verifier_v2.stable_id
        assert verifier.verify_row(record, original, 0, 0, "c" * 64)["proposals"] > 0
    finally:
        verifier.stable_id = old
    payload = {"source": source, "input_start": 0, "input_end": 1, "text": "아"}
    assert verifier_v2.stable_id("edge-v1-", payload) != old("edge-v1-", payload)


def prepared_rows(count=32):
    rows = [{"id": f"id-{index:02d}", "input": "a" * (index + 1),
             "component_id": f"component-{index // 2}"} for index in range(count)]
    ordered = sorted(rows, key=lambda row: (len(row["input"]), row["id"]))
    small, extra = divmod(count, 16); groups = []; offset = 0
    for index in range(16):
        size = small + int(index < extra); group = ordered[offset:offset + size]
        groups.append({"ids": [row["id"] for row in group], "probe_id": group[-1]["id"],
                       "count": size, "maximum_source_length": len(group[-1]["input"])})
        offset += size
    # Keep each two-row component together.
    shards = [[row["id"] for row in rows if (int(row["id"].split("-")[1]) // 2) % 6 == shard]
              for shard in range(6)]
    return {"rows": rows, "groups": groups, "shards": shards}


def test_probe_schedule_replays_quantiles_components_and_six_projections():
    prepared = prepared_rows()
    timing = {group["probe_id"]: float(index + 1)
              for index, group in enumerate(prepared["groups"])}
    per_id = {identifier: timing[group["probe_id"]]
              for group in prepared["groups"] for identifier in group["ids"]}
    projected = [1.3 * (2.0 + sum(per_id[item] for item in shard)) + 60.0
                 for shard in prepared["shards"]]
    report = {"timing_seconds": timing, "initialization_seconds": 2.0,
              "projected_shard_seconds": projected,
              "qualified": all(value <= 900 for value in projected)}
    assert verifier.verify_schedule(prepared, report) == projected
    report["projected_shard_seconds"][0] += .1
    with pytest.raises(ValueError, match="projection"):
        verifier.verify_schedule(prepared, report)


def resource(stage, operation, elapsed, operation_elapsed=.01):
    return {"at": datetime.now(timezone.utc).isoformat(), "stage": stage,
            "operation": operation,
            "operation_elapsed_s": operation_elapsed if operation else None,
            "elapsed_s": elapsed, "rss_bytes": 100,
            "swap_used_bytes": 1000, "swap_growth_bytes": 0}


def resource_trace(operations):
    result, elapsed = [], 0.0
    for index, operation in enumerate(operations):
        elapsed += .1; result.append(resource("operation_start", operation, elapsed))
        elapsed += .1; result.append(resource("operation_end", operation, elapsed))
        if index == len(operations) - 2:
            elapsed += .1; result.append(resource("after_report", None, elapsed))
    elapsed += .1; result.append(resource("after_completion", None, elapsed))
    return result


def test_resource_replay_enforces_dynamic_operations_and_limits():
    operations = ["authenticate", "propose:id-1", "write_report", "write_completion"]
    assert verifier.verify_resources(resource_trace(operations), operations)["operations"] == 4
    for field, value in (("rss_bytes", verifier.LIMITS["rss_bytes"] + 1),
                         ("swap_growth_bytes", verifier.LIMITS["swap_growth_bytes"] + 1),
                         ("elapsed_s", verifier.LIMITS["whole_seconds"] + 1),
                         ("operation_elapsed_s", verifier.LIMITS["update_seconds"] + 1)):
        rows = resource_trace(operations); rows[0][field] = value
        if field == "swap_growth_bytes": rows[0]["swap_used_bytes"] = 1000 + value
        with pytest.raises(ValueError): verifier.verify_resources(rows, operations)


def test_cli_requires_absolute_output_and_producer_digest_arguments():
    source = Path(verifier.__file__).read_text()
    assert 'parser.add_argument("--output", type=Path, required=True)' in source
    assert 'parser.add_argument("--producer-sha256", required=True)' in source
    assert "records.append" not in source


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def test_full_probe_verification_writes_proof_without_retaining_rows(tmp_path, monkeypatch):
    output = tmp_path / "probe"
    output.mkdir()
    rows = [{"id": f"id-{index:02d}", "input": f"x{index:02d}",
             "component_id": f"component-{index:02d}"} for index in range(16)]
    groups = [{"ids": [row["id"]], "probe_id": row["id"], "count": 1,
               "maximum_source_length": len(row["input"])} for row in rows]
    shards = [[rows[index]["id"] for index in range(shard, 16, 6)] for shard in range(6)]
    prepared = {"rows": rows, "groups": groups, "shards": shards}
    row_map = {row["id"]: row for row in rows}
    source_rows, records = [], []
    empty_diag = {"limits": verifier.EXPECTED_LIMITS, "pool_rows": 0,
                  "edge_occurrences": 0, "edge_count": 0, "seed_count": 0,
                  "seeds": [], "proposal_count": 0, "global_limit_reached": False,
                  "original_text_count": 1, "score_fields_consumed": False,
                  "complete_paths_rejected_as_existing_or_duplicate": 0}
    for index, row in enumerate(rows):
        source = row["input"]
        raw = {"text": source, "score": None, "segments": segments(source, source)}
        original = {"id": row["id"], "component_id": row["component_id"],
                    "request": {"input": source, "context": ""}, "base_paths": [],
                    "response": {"input": source, "context": "", "external_context_used": False,
                                 "candidates": [raw], "sentence_ranking_applied": False}}
        source_rows.append((original, row, 0, index, "c" * 64))
        records.append({"schema": "current-fit-recombination-census-row-v1",
                        "id": row["id"], "component_id": row["component_id"],
                        "request": original["request"],
                        "source_reference": {"source_shard": 0, "source_row_index": index,
                            "source_capture_sha256": "c" * 64,
                            "base_paths_sha256": verifier.canonical_sha([]),
                            "public_candidates_sha256": verifier.canonical_sha([raw]),
                            "pool_rows": 0, "ranked": False},
                        "status": "no_pool", "proposals": [], "used_edge_inventory": [],
                        "diagnostics": dict(empty_diag, source_length=len(source)),
                        "fallback_public": [raw]})
    encoded = [json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records]
    (output / "capture.jsonl").write_text("".join(encoded))
    sizes = {row["id"]: len(text.encode()) for row, text in zip(records, encoded)}
    timing = {row["id"]: .01 for row in rows}
    projections = [1.3 * (.1 + .01 * len(shard)) + 60 for shard in shards]
    storage = [int(__import__("math").ceil(1.3 * sum(sizes[item] for item in shard)))
               for shard in shards]
    counts, reasons = verifier._counts(records)
    capture_sha = verifier.sha256_file(output / "capture.jsonl")
    closure = {"schema": "current-fit-recombination-census-closure-v1", "completed": True,
               "mode": "probe", "shard": None, "rows": 16, "capture_sha256": capture_sha,
               "labels_opened": False, "model_loaded": False, "model_inference": False}
    dump(output / "capture.complete.json", closure)
    producer_sha = verifier.sha256_file(verifier.PRODUCER)
    pins = {str(verifier.PRODUCER.resolve()): producer_sha}
    report = {"schema": "current-fit-recombination-census-v1", "completed": True,
              "mode": "probe", "shard": None, "rows": 16,
              "ids": [row["id"] for row in rows], "counts": counts,
              "seed_termination_reasons": reasons, "timing_seconds": timing,
              "initialization_seconds": .1, "projected_shard_seconds": projections,
              "qualified": True, "serialized_output_bytes": sizes,
              "capture_output_bytes": sum(sizes.values()),
              "projected_shard_output_bytes": storage,
              "projected_total_output_bytes": sum(storage), "limits": verifier.LIMITS,
              "disk_limits": verifier.DISK_LIMITS, "pins": pins,
              "capture_sha256": capture_sha,
              "closure_sha256": verifier.sha256_file(output / "capture.complete.json"),
              "producer_sha256": producer_sha,
              "generator_sha256": verifier.STATIC_HASHES[verifier.GENERATOR],
              "generator_schema": "source-edge-recombination-v1",
              "generator_limits": verifier.EXPECTED_LIMITS, "probe_binding": None,
              "source_only": True, "train_only": True, "labels_opened": False,
              "fit_feature_files_opened": False, "dev_opened": False,
              "test_opened": False, "model_loaded": False, "model_inference": False,
              "neural_training": False, "production_changed": False}
    dump(output / "pins.json", pins); dump(output / "report.json", report)
    operations = ["authenticate"] + ["propose:" + row["id"] for row in rows]
    operations += ["write_report", "write_completion"]
    resources = resource_trace(operations)
    (output / "resources.jsonl").write_text("".join(json.dumps(row) + "\n" for row in resources))
    completion = {"completed": True, "report_sha256": verifier.sha256_file(output / "report.json"),
                  "resource_after_report": next(row for row in resources if row["stage"] == "after_report")}
    dump(output / "completion.json", completion)
    launch = {"pid": 99999999,
              "command": [str(verifier.ROOT / ".venv/bin/python"), "-m",
                          "scripts.current_fit_recombination_census", "--mode", "probe",
                          "--output", str(output.resolve())],
              "output": str(output.resolve()), "runner_sha256": producer_sha}
    dump(output.with_suffix(".launch.json"), launch)
    monkeypatch.setattr(verifier, "authenticate_inputs", lambda: (prepared, row_map, {}))
    monkeypatch.setattr(verifier, "_selected_source_rows",
                        lambda *_: ([row["id"] for row in rows], iter(source_rows)))
    proof = verifier.verify(output.resolve(), producer_sha)
    assert proof["verified"] is True and proof["mode"] == "probe" and proof["rows"] == 16
    assert proof["counts"] == counts
