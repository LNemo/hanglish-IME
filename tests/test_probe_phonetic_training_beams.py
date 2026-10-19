import hashlib
import itertools
import json
from pathlib import Path

import pytest

import scripts.probe_phonetic_training_beams as module
from hanglish.core.phonology import romanize_word


def write_rows(path, rows):
    raw = b"".join((json.dumps(row, ensure_ascii=False) + "\n").encode() for row in rows)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def source_rows():
    return [
        {"id": "r1", "source_role": "natural_fit", "input": "gana"},
        {"id": "r2", "source_role": "natural_fit", "input": "daga gana"},
        {"id": "c1", "source_role": "challenge_fit", "input": "ignored"},
    ]


def test_runtime_chunk_extraction_matches_protection_and_ascii_bounds():
    assert module.runtime_chunks(
        "gana TEST Camel test@example.kr abc123 `literal`eun na a " + "z" * 41
    ) == ("gana", "test", "example", "kr", "abc", "literal", "eun", "na")


def test_selection_is_length_quantile_pilot_then_complete_deterministic_order():
    rows = [{"id": "r", "input": " ".join(["aa", "bbbb", "ccc", "ddddd", "ffffff"])}]
    selected = module.build_selection(rows, expected_inputs=5, pilot_rows=3)
    assert [row["input"] for row in selected] == ["aa", "bbbb", "ffffff", "ccc", "ddddd"]
    assert selected[0]["row_ids"] == ["r"]
    assert selected[0]["protected_row_ids"] == []
    assert selected[0]["unprotected_row_ids"] == ["r"]
    assert [row["index"] for row in selected] == list(range(5))


def test_source_loader_rejects_target_fields(tmp_path):
    path = tmp_path / "rows.jsonl"
    rows = [{"id": "r", "source_role": "natural_fit", "input": "gana", "expected": ["가나"]}]
    sha = write_rows(path, rows)
    with pytest.raises(ValueError, match="target field"):
        module.load_source_rows(path, sha, 1)


class FakeModel:
    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    def generate(self, source, beam_size):
        self.calls.append((source, beam_size))
        return self.mapping.get(source, [])


def fake_environment(tmp_path, monkeypatch, rows=None):
    rows_path = tmp_path / "rows.jsonl"
    rows_sha = write_rows(rows_path, rows or source_rows())
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    checkpoint = tmp_path / "phonetic.pt"
    checkpoint.write_bytes(b"weights")
    checkpoint.with_suffix(".json").write_text("{}")
    dependency = tmp_path / "dependency.py"
    dependency.write_text("# dependency\n")

    def pins(*_args):
        paths = [rows_path, manifest, checkpoint, checkpoint.with_suffix(".json"), dependency]
        return {str(path): module.file_sha256(path) for path in paths}

    monkeypatch.setattr(module, "source_pins", pins)
    monkeypatch.setattr(module.importlib.metadata, "version", lambda _: "test")
    return (rows_path, rows_sha, manifest, module.file_sha256(manifest), checkpoint,
            module.file_sha256(checkpoint), module.file_sha256(checkpoint.with_suffix(".json")))


def test_success_capture_is_closed_complete_target_free_and_forward_valid(tmp_path, monkeypatch):
    rows_path, rows_sha, manifest, manifest_sha, checkpoint, checkpoint_sha, metadata_sha = fake_environment(tmp_path, monkeypatch)
    model = FakeModel({
        "gana": [("가나", -0.1)],
        "daga": [("다가", -0.2)],
    })
    clock = iter(range(100))
    report = module.run(
        tmp_path / "out", rows_path=rows_path, manifest_path=manifest,
        checkpoint=checkpoint, rows_sha=rows_sha, manifest_sha=manifest_sha,
        checkpoint_sha=checkpoint_sha, metadata_sha=metadata_sha,
        expected_rows=2, expected_inputs=2, model_loader=lambda _: model,
        monotonic=lambda: next(clock), rss_reader=lambda: 100,
    )
    assert report["completed"] is True
    assert report["target_fields_read"] is False
    assert report["captured_inputs"] == 2
    selection = [json.loads(line) for line in (tmp_path / "out/selection.jsonl").read_text().splitlines()]
    capture = [json.loads(line) for line in (tmp_path / "out/capture.jsonl").read_text().splitlines()]
    assert [(row["index"], row["input"]) for row in capture] == [
        (row["index"], row["input"]) for row in selection
    ]
    assert all(set(row) == {"index", "input", "outputs"} for row in capture)
    assert model.calls == [(row["input"], 8) for row in selection]
    assert report["capture_sha256"] == module.file_sha256(tmp_path / "out/capture.jsonl")


def test_invalid_output_fails_closed_with_partial_evidence(tmp_path, monkeypatch):
    rows_path, rows_sha, manifest, manifest_sha, checkpoint, checkpoint_sha, metadata_sha = fake_environment(tmp_path, monkeypatch)
    model = FakeModel({"gana": [("다나", -0.1)]})
    with pytest.raises(ValueError, match="forward-valid"):
        module.run(
            tmp_path / "out", rows_path=rows_path, manifest_path=manifest,
            checkpoint=checkpoint, rows_sha=rows_sha, manifest_sha=manifest_sha,
            checkpoint_sha=checkpoint_sha, metadata_sha=metadata_sha,
            expected_rows=2, expected_inputs=2, model_loader=lambda _: model,
            monotonic=lambda: 0, rss_reader=lambda: 100,
        )
    failure = json.loads((tmp_path / "out/failure.json").read_text())
    assert failure["completed"] is False
    assert failure["completed_inputs"] == 1
    assert module.file_sha256(tmp_path / "out/capture.jsonl") == failure["capture_sha256"]


def test_pilot_projection_stops_after_closed_pilot_without_resume(tmp_path, monkeypatch):
    rows = [{"id": "r", "source_role": "natural_fit", "input": "aa bb cc dd"}]
    rows_path, rows_sha, manifest, manifest_sha, checkpoint, checkpoint_sha, metadata_sha = fake_environment(
        tmp_path, monkeypatch, rows
    )
    model = FakeModel({})
    monkeypatch.setitem(module.PROFILE, "pilot_rows", 2)
    monkeypatch.setitem(module.PROFILE, "time_limit_seconds", 65)
    times = itertools.count()
    with pytest.raises(module.ResourceGateError, match="pilot projects"):
        module.run(
            tmp_path / "out", rows_path=rows_path, manifest_path=manifest,
            checkpoint=checkpoint, rows_sha=rows_sha, manifest_sha=manifest_sha,
            checkpoint_sha=checkpoint_sha, metadata_sha=metadata_sha,
            expected_rows=1, expected_inputs=4, model_loader=lambda _: model,
            monotonic=lambda: next(times), rss_reader=lambda: 100,
        )
    capture = (tmp_path / "out/capture.jsonl").read_text().splitlines()
    failure = json.loads((tmp_path / "out/failure.json").read_text())
    assert len(capture) == failure["completed_inputs"] == 2
    assert failure["state"] == "resource_failed"
    assert failure["model_inference_run"] is True
    assert failure["elapsed_seconds"] == 7
    assert failure["peak_rss_bytes"] == 100
    assert not (tmp_path / "out/report.json").exists()


@pytest.mark.parametrize("outputs", [
    [("가나", float("nan"))],
    [("가나", True)],
    [("가나", -1.0)] * 2,
    [("가나", -1.0)] * 9,
    [("가나", -2.0), ("간아", -1.0)],
])
def test_output_validation_rejects_malformed_beams(outputs):
    with pytest.raises(ValueError):
        module.validate_outputs(romanize_word("가나"), outputs)
