import contextlib
import json
from pathlib import Path

import pytest

from scripts import current_recombination_selection as selection
from scripts import evaluate_current_fit_recombination_selection as evaluator


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def segment(source, text):
    return [{"input_start": 0, "input_end": len(source), "source": source,
             "text": text, "kind": "fixture"}]


def candidate(source, text, score, feature, status="preserved"):
    return {"text": text, "sentence_score": score, "features": [feature, 0.0, 0.0, 0.0],
            "segments": segment(source, text), "status": status,
            "score_source": "captured_original" if status == "preserved" else None}


def scoring(identifier, source, *, ranked):
    if not ranked:
        return {"schema": "current-recombination-scoring-row-v1", "id": identifier,
                "request": {"input": source, "context": ""}, "ranked": False,
                "original": [], "novel": [],
                "fallback_public": [{"text": "fallback", "score": None,
                                     "segments": segment(source, "fallback")}]}
    return {"schema": "current-recombination-scoring-row-v1", "id": identifier,
            "request": {"input": source, "context": ""}, "ranked": True,
            "original": [candidate(source, "old", 2.0, 0.0)],
            "novel": [candidate(source, "new", 3.0, 0.0, "scored")],
            "fallback_public": None}


def feature_row(identifier, source, component, expected, candidates, *, ranked):
    positives = [index for index, text in enumerate(candidates) if text in set(expected)]
    return {"id": identifier, "input": source, "component_id": component,
            "expected": expected, "candidates": candidates,
            "base_scores": [1.0] * len(candidates), "features": [[0.0] * 4 for _ in candidates],
            "positive_indices": positives, "production_indices": list(range(len(candidates))),
            "public_candidates": candidates, "ranked": ranked,
            "training_eligible": bool(ranked and positives)}


class Watchdog:
    def __init__(self, started, swap, trace, **kwargs):
        self.trace = Path(trace)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    @contextlib.contextmanager
    def operation(self, name, update=False):
        yield

    def sample_and_append(self, stage):
        row = {"stage": stage}
        with self.trace.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        return row


def test_end_to_end_joins_reordered_labels_by_id_and_uses_real_helper(monkeypatch, tmp_path):
    features = tmp_path / "features.jsonl"
    model = tmp_path / "model.json"
    # Feature order follows preparation, while capture order follows frozen shards.
    feature_rows = [feature_row("a", "source-a", "ca", ["fallback"], [], ranked=False),
                    feature_row("b", "source-b", "cb", ["new"], ["old"], ranked=True)]
    feature_rows[1]["base_scores"] = [2.0]
    features.write_text("".join(json.dumps(row) + "\n" for row in feature_rows), encoding="utf-8")
    dump(model, {"schema": "phonetic-score-scale-model-v1", "arm": "learned_scaled",
                 "parameters": [1.0, 0.0, 0.0, 0.0, 0.0],
                 "scale": [1.0, 1.0, 1.0, 1.0]})
    shard_b, shard_a = tmp_path / "shard-b", tmp_path / "shard-a"
    shard_b.mkdir(); shard_a.mkdir()
    (shard_b / "capture.jsonl").write_text(json.dumps({"id": "b", "scoring": scoring(
        "b", "source-b", ranked=True)}) + "\n", encoding="utf-8")
    (shard_a / "capture.jsonl").write_text(json.dumps({"id": "a", "scoring": scoring(
        "a", "source-a", ranked=False)}) + "\n", encoding="utf-8")
    prepared = [{"id": "a", "input": "source-a", "component_id": "ca"},
                {"id": "b", "input": "source-b", "component_id": "cb"}]
    pins = {str(features.resolve()): evaluator.sha(features), str(model.resolve()): evaluator.sha(model)}
    auth = {"manifest": {"helper_sha256": "h" * 64}, "helper": selection,
            "prepared": prepared, "shard_ids": [["b"], ["a"]],
            "directories": [shard_b, shard_a], "pins": pins}
    monkeypatch.setattr(evaluator, "FEATURES", features)
    monkeypatch.setattr(evaluator, "MODEL", model)
    monkeypatch.setattr(evaluator, "FEATURES_SHA256", evaluator.sha(features))
    monkeypatch.setattr(evaluator, "MODEL_SHA256", evaluator.sha(model))
    monkeypatch.setattr(evaluator, "ROWS", 2)
    monkeypatch.setattr(evaluator, "authenticate", lambda *args: auth)
    monkeypatch.setattr(evaluator, "system_swap_bytes", lambda: 0)
    output = (tmp_path / "output").resolve()
    report = evaluator.run(output, (tmp_path / "intake.json").resolve(), "i" * 64,
                           "e" * 64, watchdog_cls=Watchdog)
    rows = [json.loads(line) for line in (output / "outcomes.jsonl").read_text().splitlines()]
    assert [row["id"] for row in rows] == ["b", "a"]
    assert report["summary"]["branches"]["expanded_sentence"]["first"] == 2
    assert report["summary"]["transitions"]["candidate_expansion_sentence"]["corrections"] == 1
    assert report["summary"]["candidate_counts"]["no_pool_rows"] == 1
    assert json.loads((output / "pins.json").read_text()) == report["pins"]


def test_rejects_helper_outcome_with_wrong_arm_schema():
    row = selection.compare_row(scoring("b", "source", ranked=True), ["new"],
                                {"parameters": [1.0, 0.0, 0.0, 0.0, 0.0],
                                 "scale": [1.0] * 4})
    del row["arms"]["expanded_calibrated"]
    with pytest.raises(ValueError, match="arms"):
        evaluator.validate_outcome(row, "b", True)


def test_rejects_incomplete_novel_status_count():
    row = selection.compare_row(scoring("b", "source", ranked=True), ["new"],
                                {"parameters": [1.0, 0.0, 0.0, 0.0, 0.0],
                                 "scale": [1.0] * 4})
    row["counts"]["scored_novel"] = 0
    with pytest.raises(ValueError, match="counts"):
        evaluator.validate_outcome(row, "b", True)


def test_authentication_failure_records_that_fit_labels_stayed_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluator, "authenticate",
                        lambda *args: (_ for _ in ()).throw(ValueError("bad proof")))
    monkeypatch.setattr(evaluator, "system_swap_bytes", lambda: 0)
    output = (tmp_path / "failed").resolve()
    with pytest.raises(ValueError, match="bad proof"):
        evaluator.run(output, (tmp_path / "intake.json").resolve(), "i" * 64,
                      "e" * 64, watchdog_cls=Watchdog)
    failure = json.loads((output / "failure.json").read_text())
    assert failure["fit_labels_opened"] is False


def test_validate_label_rejects_component_mismatch():
    source = {"id": "a", "input": "source", "component_id": "component-a"}
    row = feature_row("a", "source", "component-b", ["answer"], [], ranked=False)
    with pytest.raises(ValueError, match="identity"):
        evaluator.validate_label(row, source)
