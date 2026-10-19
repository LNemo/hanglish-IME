from copy import deepcopy
import json

import pytest

from hanglish.core.source_edge_recombination import propose
from scripts import current_fit_recombination_census as census
from scripts import source_span_resources as resource_helper


SOURCE = "abcdef"


def segment(start, end, text, kind="fixture"):
    return {"input_start": start, "input_end": end,
            "source": SOURCE[start:end], "text": text, "kind": kind}


def path(text, segments, score=0.0):
    return [text, score, segments]


def candidate(text, segments, score=0.0):
    return {"text": text, "score": score, "segments": deepcopy(segments)}


def source_fixture(identifier="fit-1", ranked=True):
    first = [segment(0, 2, "A"), segment(2, 4, "B"), segment(4, 6, "C")]
    second = [segment(0, 2, "X"), segment(2, 4, "Y"), segment(4, 6, "Z")]
    raw = [{"input_start": 0, "input_end": len(SOURCE), "source": SOURCE,
            "text": SOURCE, "kind": "raw"}]
    paths = [path("ABC", first, 2.0), path("XYZ", second, 1.0)] if ranked else []
    public = ([candidate("ABC", first, 2.0), candidate("XYZ", second, 1.0),
               candidate(SOURCE, raw, None)] if ranked else [candidate(SOURCE, raw, None)])
    original = {
        "id": identifier, "component_id": "component-1",
        "request": {"input": SOURCE, "context": ""},
        "response": {"input": SOURCE, "context": "", "external_context_used": False,
                     "sentence_ranking_applied": ranked, "candidates": public},
        "base_paths": paths,
    }
    expected = {"id": identifier, "input": SOURCE, "component_id": "component-1"}
    return original, expected


def test_multiple_segment_recombination_is_captured_with_exact_edge_evidence():
    original, expected = source_fixture()
    record = census.prepare_record(original, expected, 2, 17, "a" * 64)

    by_text = {row["text"]: row for row in record["proposals"]}
    assert "AYC" in by_text
    proposal = by_text["AYC"]
    evidence = {row["edge_id"]: row for row in record["used_edge_inventory"]}
    assert len(proposal["edge_ids"]) == len(proposal["segments"]) == 3
    assert all(evidence[edge_id]["provenance"] for edge_id in proposal["edge_ids"])
    assert record["source_reference"]["source_shard"] == 2
    assert "pool" not in record


def test_unranked_no_pool_retains_original_public_candidate_dicts():
    original, expected = source_fixture(ranked=False)
    record = census.prepare_record(original, expected, 0, 3, "b" * 64)

    assert record["status"] == "no_pool"
    assert record["proposals"] == []
    assert record["source_reference"]["pool_rows"] == 0
    assert record["fallback_public"] == original["response"]["candidates"]
    assert isinstance(record["fallback_public"][0], dict)


def test_foreign_proposal_edge_is_rejected():
    original, expected = source_fixture()

    def foreign(source, pool, public):
        generated = propose(source, pool, public)
        generated["proposals"][0]["segments"][0]["text"] = "FOREIGN"
        generated["proposals"][0]["text"] = "FOREIGN" + "".join(
            row["text"] for row in generated["proposals"][0]["segments"][1:])
        return generated

    with pytest.raises(ValueError, match="foreign"):
        census.prepare_record(original, expected, 0, 0, "c" * 64,
                              generator=foreign)


def test_probe_quantiles_and_projection_are_deterministic():
    rows = [{"id": f"id-{index:02d}", "input": "x" * (index + 1),
             "component_id": f"component-{index % 4}"} for index in range(32)]
    groups = census.probe_groups(rows)
    shards = [[row["id"] for row in rows[index::6]] for index in range(6)]
    timings = {group["probe_id"]: (index + 1) / 100
               for index, group in enumerate(groups)}

    assert groups == census.probe_groups(list(reversed(rows)))
    first = census.projected_shard_seconds(groups, shards, timings, 2.0)
    second = census.projected_shard_seconds(groups, shards, timings, 2.0)
    assert first == second
    assert first == pytest.approx([
        1.3 * (2.0 + sum(timings[next(group["probe_id"] for group in groups
                                      if row_id in group["ids"])]
                         for row_id in shard)) + 60.0 for shard in shards])
    sizes = {group["probe_id"]: (index + 1) * 100
             for index, group in enumerate(groups)}
    disk_first = census.projected_shard_output_bytes(groups, shards, sizes)
    assert disk_first == census.projected_shard_output_bytes(groups, shards, sizes)
    assert all(type(value) is int and value > 0 for value in disk_first)


def test_capture_argument_contract_rejects_missing_or_extra_probe(tmp_path):
    with pytest.raises(ValueError, match="arguments"):
        census.run(tmp_path / "bad-probe", "probe", shard=0)
    with pytest.raises(ValueError, match="arguments"):
        census.run(tmp_path / "bad-capture", "capture", shard=0)


def test_hard_output_limit_rejects_row_before_write(tmp_path):
    path = tmp_path / "capture.jsonl"
    with path.open("x", encoding="utf-8") as stream:
        with pytest.raises(ValueError, match="would exceed"):
            census.append_jsonl(stream, {"large": "value"},
                                current_bytes=0, limit_bytes=1)
    assert path.read_bytes() == b""


def test_capture_source_selection_is_lazy_and_exact_order(monkeypatch):
    originals = [source_fixture(f"fit-{index}")[0] for index in range(3)]
    rows = [source_fixture(f"fit-{index}")[1] for index in range(3)]
    consumed = []

    def fake_rows(_path):
        for row in originals:
            consumed.append(row["id"])
            yield row

    monkeypatch.setattr(census, "iter_jsonl", fake_rows)
    shards = [[row["id"] for row in rows], [], [], [], [], []]
    ids, selected = census.selected_source_rows(
        "capture", 0, [], shards, {row["id"]: row for row in rows})

    assert ids == shards[0] and consumed == []
    assert next(selected)[0]["id"] == "fit-0"
    assert consumed == ["fit-0"]
    assert [value[0]["id"] for value in selected] == ["fit-1", "fit-2"]


def test_probe_authentication_requires_independent_verified_proof(tmp_path):
    directory = tmp_path / "probe"; directory.mkdir()
    (directory / "capture.jsonl").write_text("{}\n")
    capture_sha = census.sha256_file(directory / "capture.jsonl")
    census.write_json(directory / "capture.complete.json", {
        "rows": census.PROBES, "labels_opened": False,
        "capture_sha256": capture_sha})
    closure_sha = census.sha256_file(directory / "capture.complete.json")
    report = {
        "schema": census.SCHEMA, "completed": True, "mode": "probe",
        "shard": None, "rows": census.PROBES, "qualified": True,
        "producer_sha256": census.sha256_file(census.Path(census.__file__)),
        "generator_sha256": census.STATIC_HASHES[census.GENERATOR],
        "projected_shard_seconds": [100.0] * 6,
        "projected_shard_output_bytes": [1000] * 6,
        "projected_total_output_bytes": 6000,
        "capture_sha256": capture_sha, "closure_sha256": closure_sha,
    }
    census.write_json(directory / "report.json", report)
    report_sha = census.sha256_file(directory / "report.json")
    census.write_json(directory / "completion.json", {
        "completed": True, "report_sha256": report_sha})
    census.write_json(directory / "independent-verification.json", {
        "verified": True, "mode": "probe", "qualified": True,
        "report_sha256": report_sha, "capture_sha256": capture_sha})

    authenticated, pins = census.authenticate_probe(directory, report_sha)
    assert authenticated == report
    assert str((directory / "independent-verification.json").resolve()) in pins

    proof = json.loads((directory / "independent-verification.json").read_text())
    proof["qualified"] = False
    (directory / "independent-verification.json").write_text(json.dumps(proof))
    with pytest.raises(ValueError, match="probe artifact"):
        census.authenticate_probe(directory, report_sha)


def _synthetic_domain():
    rows, originals = [], []
    for index in range(16):
        identifier = f"fit-{index:02d}"
        original, expected = source_fixture(identifier)
        rows.append(expected); originals.append((original, expected, index % 6, index,
                                                  str(index) * 64))
    groups = [{"ids": [row["id"]], "probe_id": row["id"], "count": 1,
               "maximum_source_length": len(row["input"])} for row in rows]
    shards = [[row["id"] for row in rows[index::6]] for index in range(6)]
    return rows, originals, groups, shards


def _patch_resources(monkeypatch):
    monkeypatch.setattr(census, "system_swap_bytes", lambda: 0)
    monkeypatch.setattr(resource_helper, "system_swap_bytes", lambda: 0)
    monkeypatch.setattr(resource_helper, "rss_bytes", lambda: 1)


def test_synthetic_probe_run_closes_all_rows_and_projects_six_shards(
        monkeypatch, tmp_path):
    rows, originals, groups, shards = _synthetic_domain()
    monkeypatch.setattr(census, "authenticate_inputs",
                        lambda: (rows, {row["id"]: row for row in rows},
                                 groups, shards, {}))
    monkeypatch.setattr(census, "selected_source_rows",
                        lambda *args: ([row["id"] for row in rows], iter(originals)))
    _patch_resources(monkeypatch)

    report = census.run(tmp_path / "probe", "probe")

    assert report["completed"] is True and report["qualified"] is True
    assert report["rows"] == 16 and len(report["projected_shard_seconds"]) == 6
    assert len(report["projected_shard_output_bytes"]) == 6
    assert report["projected_total_output_bytes"] == sum(
        report["projected_shard_output_bytes"])
    assert report["counts"]["rows"] == 16
    assert report["counts"]["proposal_count"] > 0
    assert report["labels_opened"] is False and report["model_loaded"] is False
    assert sum(1 for _ in census.iter_jsonl(tmp_path / "probe/capture.jsonl")) == 16
    assert not (tmp_path / "probe/failure.json").exists()


def test_synthetic_capture_run_binds_probe_and_exact_shard_order(
        monkeypatch, tmp_path):
    rows, originals, groups, shards = _synthetic_domain()
    selected_ids = shards[0]
    selected = [row for row in originals if row[1]["id"] in selected_ids]
    selected.sort(key=lambda row: selected_ids.index(row[1]["id"]))
    monkeypatch.setattr(census, "authenticate_inputs",
                        lambda: (rows, {row["id"]: row for row in rows},
                                 groups, shards, {}))
    monkeypatch.setattr(census, "selected_source_rows",
                        lambda *args: (selected_ids, iter(selected)))
    probe_dir = tmp_path / "probe"; probe_dir.mkdir()
    (probe_dir / "completion.json").write_text("{}")
    (probe_dir / "independent-verification.json").write_text("{}")
    probe_report = {"projected_shard_seconds": [100.0] * 6,
                    "projected_shard_output_bytes": [1000] * 6,
                    "capture_sha256": "a" * 64, "closure_sha256": "b" * 64}
    monkeypatch.setattr(census, "authenticate_probe",
                        lambda *args: (probe_report, {}))
    _patch_resources(monkeypatch)

    report = census.run(tmp_path / "capture", "capture", shard=0,
                        probe=probe_dir, probe_report_sha256="c" * 64)

    assert report["ids"] == selected_ids
    assert report["projected_shard_seconds"] == [100.0] * 6
    assert report["projected_shard_output_bytes"] == [1000] * 6
    assert report["probe_binding"]["report_sha256"] == "c" * 64
    assert report["counts"]["rows"] == len(selected_ids)


def test_source_has_no_model_or_label_dependencies():
    text = (census.ROOT / "scripts/current_fit_recombination_census.py").read_text()
    assert "import torch" not in text
    assert "dev.jsonl" not in text.lower()
    assert "test.jsonl" not in text.lower()
    assert "train.jsonl" not in text.lower()
    assert "current-phonetic-train-features" not in text
