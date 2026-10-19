import json

import pytest

from scripts import evaluate_current_phonetic_choice_dev as screen
from scripts import source_span_resources as resource_helper


DEPLOYED = {
    "schema": "phonetic-score-scale-model-v1", "arm": "learned_scaled",
    "parameters": [1.0, 1.0, 0.0, 0.0, 0.0],
    "scale": [2.0, 1.0, 1.0, 1.0],
}
PROPOSED = {
    "schema": "phonetic-score-scale-model-v1", "arm": "learned_scaled",
    "parameters": [0.5, 0.0, 2.0, 0.0, 0.0],
    "scale": [1.0, 4.0, 1.0, 1.0],
}


def segment(source, text):
    return [{"input_start": 0, "input_end": len(source), "source": source,
             "text": text, "kind": "synthetic"}]


def public(text, score=1.0):
    return [{"text": text, "score": score, "segments": segment("roman", text)}]


def ranked_rows():
    paths = [
        ["가", 2.0, segment("roman", "가")],
        ["나", 1.0, segment("roman", "나")],
    ]
    features = [
        {"text": "가", "base_score": 2.0, "features": [2.0, 0.0, 0.0, 0.0],
         "segments": paths[0][2]},
        {"text": "나", "base_score": 1.0, "features": [0.0, 4.0, 0.0, 0.0],
         "segments": paths[1][2]},
    ]
    base_public = screen.render_public("roman", paths)
    response = {
        "version": screen.FULL_VERSION, "input": "roman", "context": "",
        "external_context_used": False, "result": "가", "candidates": base_public,
        "sentence_ranking_applied": True,
        "phonetic_choice": {"parent_version": screen.FULL_PARENT_VERSION,
                            "rows": features},
    }
    source = {"id": "dev-1", "request": {"input": "roman", "context": ""},
              "proposed": response, "base_paths": paths,
              "union_reference": {"result": "가", "candidates": base_public,
                                  "sentence_ranking_applied": True}}
    current = {
        "version": screen.CURRENT_VERSION, "input": "roman", "context": "",
        "external_context_used": False, "result": "나", "candidates": public("나"),
        "sentence_ranking_applied": True,
    }
    comparator = {"id": "dev-1", "request": source["request"], "response": current}
    return source, comparator


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8")


def branch(result, candidates=None, ranked=True):
    texts = candidates or [result]
    return {"result": result,
            "candidates": [{"text": text, "score": float(index), "segments": []}
                           for index, text in enumerate(texts)],
            "sentence_ranking_applied": ranked, "latent_top5": texts[:5]}


def test_formula_and_actual_current_response_are_preserved():
    source, comparator = ranked_rows()
    result = screen.project_row(source, comparator, DEPLOYED, PROPOSED)

    assert result["branches"]["current_production"]["result"] == "나"
    assert result["score_rows"][0]["scores"] == {
        "base_full": 2.0, "deployed_full": 3.0, "proposed": 1.0}
    assert result["score_rows"][1]["scores"] == {
        "base_full": 1.0, "deployed_full": 1.0, "proposed": 2.5}
    assert result["branches"]["proposed"]["result"] == "나"


def test_ties_round_scores_and_reserve_raw_as_fifth_slot():
    paths = [(text, 1.23456, segment("raw", text)) for text in "fedcba"]
    rendered = screen.render_public("raw", paths)

    assert [row["text"] for row in rendered] == ["a", "b", "c", "d", "raw"]
    assert [row["score"] for row in rendered[:4]] == [1.2346] * 4
    assert rendered[-1] == screen.raw_candidate("raw")


@pytest.mark.parametrize("mutation", ["text", "score", "segments", "offset", "features"])
def test_bad_source_feature_alignment_fails_closed(mutation):
    source, _ = ranked_rows()
    row = source["proposed"]["phonetic_choice"]["rows"][0]
    if mutation == "text":
        row["text"] = "다"
    elif mutation == "score":
        row["base_score"] += 0.1
    elif mutation == "segments":
        row["segments"] = []
    elif mutation == "offset":
        row["segments"][0]["input_end"] -= 1
    else:
        row["features"] = [1.0, 2.0]

    with pytest.raises(ValueError, match="alignment|feature vector|offset"):
        screen.score_rows(source, DEPLOYED, PROPOSED)


def test_unranked_row_preserves_captured_fallback_for_all_full_branches():
    fallback = {
        "version": screen.FULL_VERSION, "input": "roman", "context": "",
        "external_context_used": False, "result": "roman",
        "candidates": [screen.raw_candidate("roman")],
        "sentence_ranking_applied": False,
        "phonetic_choice": {"parent_version": screen.FULL_PARENT_VERSION, "rows": []},
    }
    source = {"id": "dev-1", "request": {"input": "roman", "context": ""},
              "proposed": fallback, "base_paths": [],
              "union_reference": {"result": "roman", "candidates": fallback["candidates"],
                                  "sentence_ranking_applied": False}}

    rows, branches = screen.score_rows(source, DEPLOYED, PROPOSED)

    assert rows == []
    assert all({key: value[key] for key in ("result", "candidates",
                                             "sentence_ranking_applied")}
               == screen.compact_response(fallback) for value in branches.values())
    assert all(value["latent_top5"] == [] for value in branches.values())


def test_gate_rejects_even_when_net_top1_improves_if_one_first_regresses(
        monkeypatch, tmp_path):
    monkeypatch.setattr(screen, "ROWS", 3)
    predictions, labels = [], []
    cases = [
        ("가", "나", "가"),  # regression
        ("나", "가", "가"),  # correction
        ("나", "가", "가"),  # correction: proposed is net +1
    ]
    for index, (current, proposed, gold) in enumerate(cases):
        identifier = f"dev-{index}"
        branches = {
            "current_production": branch(current, [current, gold]),
            "base_full": branch(current, [current, gold]),
            "deployed_full": branch(current, [current, gold]),
            "proposed": branch(proposed, [proposed, gold]),
        }
        predictions.append({"id": identifier,
                            "request": {"input": f"roman{index}", "context": ""},
                            "branches": branches, "score_rows": []})
        labels.append({"id": identifier, "input": f"roman{index}",
                       "context": "", "split": "dev", "expected": [gold]})
    prediction_path, label_path = tmp_path / "predictions.jsonl", tmp_path / "labels.jsonl"
    write_jsonl(prediction_path, predictions); write_jsonl(label_path, labels)

    result = screen.evaluate(prediction_path, label_path)

    assert result["branches"]["current_production"]["top1"] == 1
    assert result["branches"]["proposed"]["top1"] == 2
    assert result["transitions"]["overall"]["regressions"] == 1
    assert result["quality_gate"] is False
    assert result["actual_runtime_authorized"] is False


def test_scored_coverage_and_latent_top5_are_separate_from_public_raw_slot(
        monkeypatch, tmp_path):
    monkeypatch.setattr(screen, "ROWS", 1)
    latent = ["일", "이", "삼", "사", "정답"]
    full = branch("일", ["일", "이", "삼", "사", "roman"])
    full["latent_top5"] = latent
    prediction = {
        "id": "dev-1", "request": {"input": "roman", "context": ""},
        "branches": {
            "current_production": {**branch("일"), "latent_top5": None},
            "base_full": full, "deployed_full": full, "proposed": full,
        },
        "score_rows": [{"text": text} for text in latent],
    }
    label = {"id": "dev-1", "input": "roman", "context": "", "split": "dev",
             "expected": ["정답"]}
    prediction_path, label_path = tmp_path / "predictions.jsonl", tmp_path / "labels.jsonl"
    write_jsonl(prediction_path, [prediction]); write_jsonl(label_path, [label])

    result = screen.evaluate(prediction_path, label_path)["branches"]["proposed"]

    assert result["top5"] == 0
    assert result["latent_top5"] == 1
    assert result["scored_pool_coverage"] == 1
    assert result["ranked_rows"] == 1


def test_synthetic_whole_run_closes_predictions_before_metrics_and_completes(
        monkeypatch, tmp_path):
    source, comparator = ranked_rows()
    source_dir, comparator_dir = tmp_path / "source", tmp_path / "comparator"
    source_dir.mkdir(); comparator_dir.mkdir()
    write_jsonl(source_dir / "capture.jsonl", [source])
    write_jsonl(comparator_dir / "capture.jsonl", [comparator])
    labels = tmp_path / "dev.jsonl"
    write_jsonl(labels, [{"id": "dev-1", "input": "roman", "context": "",
                          "split": "dev", "expected": ["나"]}])
    pins = {str(labels.resolve()): screen.sha256_file(labels)}
    monkeypatch.setattr(screen, "SOURCE", source_dir)
    monkeypatch.setattr(screen, "COMPARATOR", comparator_dir)
    monkeypatch.setattr(screen, "DEV", labels)
    monkeypatch.setattr(screen, "ROWS", 1)
    monkeypatch.setattr(screen, "CURRENT_REFERENCE", {
        "rows": 1, "top1": 1, "top5": 1,
        "ascii_reference_rows": 0, "ascii_exact_first": 0})
    monkeypatch.setattr(screen, "authorize", lambda: (PROPOSED, DEPLOYED, pins))
    monkeypatch.setattr(screen, "system_swap_bytes", lambda: 0)
    monkeypatch.setattr(resource_helper, "system_swap_bytes", lambda: 0)
    monkeypatch.setattr(resource_helper, "rss_bytes", lambda: 1)
    output = tmp_path / "output"

    report = screen.run(output)

    assert report["completed"] is True
    assert report["rows"] == 1
    assert report["model_inference_performed"] is False
    assert report["metrics"]["branches"]["proposed"]["scored_pool_coverage"] == 1
    assert json.loads((output / "closure.json").read_text())["labels_opened"] is False
    completion = json.loads((output / "completion.json").read_text())
    assert completion["completed"] is True
    assert not (output / "failure.json").exists()
