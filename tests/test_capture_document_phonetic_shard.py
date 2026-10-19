from copy import deepcopy
import json

import pytest

from scripts import capture_document_phonetic_shard as module


@pytest.fixture
def probe():
    with (module.PREFLIGHT / "capture.jsonl").open() as stream:
        return json.loads(next(stream))


def test_actual_probe_replays_parent_without_claiming_parent_inference(probe):
    source = probe["request"]["input"]
    response = probe["proposed"]
    paths = probe["base_paths"]
    record = module.capture_record(probe["id"], source, response, paths,
                                   response["phonetic_choice"]["effective_weights"])
    assert record["baseline"]["candidates"] == probe["parent"]["candidates"]
    assert record["baseline"]["result"] == probe["parent"]["result"]
    assert record["baseline"]["kind"] == module.BASELINE_KIND
    assert "engine_ms" not in record["baseline"]
    assert set(record) == {"id", "request", "baseline", "proposed", "base_paths"}
    assert "expected" not in record and "parent" not in record


def test_ranked_missing_paths_is_rejected(probe):
    with pytest.raises(ValueError, match="no pre-calibration"):
        module.baseline_public(probe["request"]["input"], probe["proposed"], [])


def test_calibration_tampering_is_rejected(probe):
    response = deepcopy(probe["proposed"])
    response["phonetic_choice"]["rows"][0]["adjusted_score"] += 1
    with pytest.raises(ValueError, match="score differs"):
        module.capture_record(probe["id"], probe["request"]["input"], response,
                              probe["base_paths"], response["phonetic_choice"]["effective_weights"])


def test_unranked_baseline_preserves_candidates_without_aliasing(probe):
    response = deepcopy(probe["proposed"])
    response["sentence_ranking_applied"] = False
    response["phonetic_choice"]["rows"] = []
    baseline = module.baseline_public(probe["request"]["input"], response, [])
    assert baseline["candidates"] == response["candidates"]
    baseline["candidates"][0]["text"] = "changed"
    assert baseline["candidates"] != response["candidates"]


def test_unranked_calibration_paths_rejected(probe):
    response = dict(probe["proposed"], sentence_ranking_applied=False)
    with pytest.raises(ValueError, match="unranked"):
        module.baseline_public(probe["request"]["input"], response, probe["base_paths"])


def test_no_predecessor_allowed_for_first_shard(tmp_path):
    assert module.authenticate_previous(None, 0) == {}
    with pytest.raises(ValueError, match="must not have"):
        module.authenticate_previous(tmp_path, 0)


def test_second_shard_requires_predecessor():
    with pytest.raises(ValueError, match="requires verified terminal"):
        module.authenticate_previous(None, 1)


def test_second_shard_rejects_failure_before_reading_reports(tmp_path):
    (tmp_path / "failure.json").write_text("{}")
    with pytest.raises(ValueError, match="failure evidence"):
        module.authenticate_previous(tmp_path, 1)


def test_unfrozen_partition_stops_before_authorization(monkeypatch):
    monkeypatch.setattr(module, "SHARD_HASHES", {})
    monkeypatch.setattr(module, "authorize_inputs", lambda: pytest.fail("input read too early"))
    with pytest.raises(ValueError, match="independently frozen"):
        module.authorize(0, None)


@pytest.mark.parametrize("shard", [-1, 2, True, "0"])
def test_invalid_shard_rejected_before_file_reads(shard):
    with pytest.raises(ValueError, match="invalid shard"):
        module.authorize(shard, None)
