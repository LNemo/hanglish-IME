from copy import deepcopy
import json

import pytest

from scripts import capture_contextual_character_dev as capture


CALIBRATION = {
    "parameters": [0.8, 0.3, -0.1, 0.5, 0.2],
    "scale": [0.2, 0.4, 0.5, 0.8],
}


def candidate(index, *, status):
    text = chr(0xAC00 + index)
    return {
        "text": text,
        "segments": [{
            "input_start": 0,
            "input_end": 1,
            "source": "x",
            "text": text,
            "kind": "fake",
        }],
        "sentence_score": index / 10,
        "features": [0.4, 0.3, 0.2, 0.1],
        "status": status,
    }


def public(source, rows, scores):
    return capture.render_public(source, [
        {"text": row["text"], "segments": row["segments"], "score": score}
        for row, score in zip(rows, scores)
    ])


def ranked_parent():
    original = [candidate(index, status="preserved") for index in range(12)]
    novel = [candidate(index, status="scored") for index in range(12, 40)]
    rows = original + novel
    base = [capture.calibrated_score(row, CALIBRATION) for row in rows]
    return {
        "id": "dev-1",
        "request": {"input": "x", "context": ""},
        "dataset_split": "dev",
        "source_row_index": 7,
        "scoring": {
            "id": "dev-1",
            "request": {"input": "x", "context": ""},
            "ranked": True,
            "original": original,
            "novel": novel,
            "fallback_public": None,
        },
        "predictions": {
            "old_calibrated": public("x", original, base[:12]),
            "expanded_calibrated": public("x", rows, base),
        },
    }


class FakeModel:
    def __init__(self):
        self.sizes = []

    def prepare(self, source, candidates, scores):
        assert source == "x"
        self.sizes.append(len(candidates))
        return list(scores)

    def __call__(self, prepared):
        return {"scores": [score + 0.125 for score in prepared]}


def test_expanded_pool_is_scored_once_in_chunks_and_old_is_exact_prefix():
    parent = ranked_parent()
    snapshot = deepcopy(parent)
    model = FakeModel()
    result = capture.predict_row(parent, model, CALIBRATION, 7)
    assert model.sizes == [32, 8]
    scores = result["contextual_scores"]
    assert scores["old_contextual"] == scores["expanded_contextual"][:12]
    assert len(scores["expanded_contextual"]) == 40
    assert result["id"] == "dev-1"
    assert result["request"] == {"input": "x", "context": ""}
    assert result["dataset_split"] == "dev"
    assert result["source_row_index"] == 7
    assert tuple(result["predictions"]) == (
        "old_contextual", "expanded_contextual")
    assert parent == snapshot


def test_no_pool_preserves_parent_fallback_without_model_inference():
    fallback = [{
        "text": "x",
        "score": None,
        "segments": [{"input_start": 0, "input_end": 1, "source": "x",
                      "text": "x", "kind": "raw"}],
    }]
    parent = {
        "id": "dev-2",
        "request": {"input": "x", "context": ""},
        "dataset_split": "dev",
        "source_row_index": 3,
        "scoring": {
            "id": "dev-2", "request": {"input": "x", "context": ""},
            "ranked": False, "original": [], "novel": [],
            "fallback_public": fallback,
        },
        "predictions": {
            "old_calibrated": deepcopy(fallback),
            "expanded_calibrated": deepcopy(fallback),
        },
    }
    result = capture.predict_row(parent, None, CALIBRATION, 3)
    assert result["contextual_scores"] == {
        "old_contextual": [], "expanded_contextual": []}
    assert result["predictions"] == {
        "old_contextual": fallback, "expanded_contextual": fallback}
    assert result["predictions"]["old_contextual"] is not fallback


def test_parent_comparator_and_status_changes_are_rejected():
    parent = ranked_parent()
    parent["predictions"]["old_calibrated"][0]["score"] = -999.0
    with pytest.raises(ValueError, match="calibrated public"):
        capture.predict_row(parent, FakeModel(), CALIBRATION, 7)
    parent = ranked_parent()
    parent["scoring"]["novel"][0]["status"] = "unsupported"
    with pytest.raises(ValueError, match="candidate status"):
        capture.predict_row(parent, FakeModel(), CALIBRATION, 7)


def test_completed_full_training_and_independent_proof_are_required(tmp_path):
    training = tmp_path / "training"
    training.mkdir()
    for name in capture.OUTPUT_HASHES:
        (training / name).write_bytes((name + "\n").encode())
    pinned = tmp_path / "pinned.txt"
    pinned.write_text("fixed\n", encoding="utf-8")
    hashes = {name: capture.digest(training / name)
              for name in capture.OUTPUT_HASHES}
    report = {
        "completed": True,
        "scope": "full",
        "rows": 1642,
        "recipe": capture.RECIPE,
        "parameter_count": 31681,
        "updates": 6284,
        "counts": capture.TRAINING_COUNTS,
        "dev_opened": False,
        "test_opened": False,
        "production_changed": False,
        "model_source_sha256": capture.digest(capture.MODEL_SOURCE),
        "runner_sha256": capture.digest(capture.FULL_TRAINING),
        "hashes": hashes,
        "pins": {str(pinned): capture.digest(pinned)},
    }
    (training / "report.json").write_text(
        json.dumps(report), encoding="utf-8")
    proof = {
        "verified": True,
        "scope": "full",
        "rows": 1642,
        "report_sha256": capture.digest(training / "report.json"),
        "checkpoint_sha256": hashes["model.pt"],
        "before_sha256": hashes["before.jsonl"],
        "after_sha256": hashes["after.jsonl"],
        "resources_sha256": hashes["resources.jsonl"],
        "candidate_chunk": 32,
        "dev_opened": False,
        "test_opened": False,
    }
    (training / "independent-verification.json").write_text(
        json.dumps(proof), encoding="utf-8")
    authenticated_report, authenticated_proof, pins = (
        capture._authenticate_training(training))
    assert authenticated_report == report
    assert authenticated_proof == proof
    assert pins[str((training / "model.pt").resolve())] == hashes["model.pt"]
    proof["verified"] = False
    (training / "independent-verification.json").write_text(
        json.dumps(proof), encoding="utf-8")
    with pytest.raises(ValueError, match="independent full FIT"):
        capture._authenticate_training(training)


def test_additional_six_hundred_second_guard_is_enforced():
    class Guard:
        def observe(self, stage):
            return {"stage": stage, "seconds": 600.001}

    with pytest.raises(ValueError, match="time limit"):
        capture._observe(Guard(), "late")
