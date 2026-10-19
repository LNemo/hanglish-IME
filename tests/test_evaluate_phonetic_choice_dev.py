import copy

import pytest

from scripts import evaluate_phonetic_choice_dev as evaluator


def segment(source, text, kind="rule"):
    return [{"input_start": 0, "input_end": len(source), "source": source,
             "text": text, "kind": kind}]


def test_render_public_injects_canonical_raw_fifth_slot():
    source = "source"
    paths = [(f"후보{i}", 10.0 - i, segment(source, f"후보{i}")) for i in range(6)]

    rendered = evaluator.render_public(source, paths)

    assert [row["text"] for row in rendered] == ["후보0", "후보1", "후보2", "후보3", source]
    assert rendered[-1] == {
        "text": source,
        "score": None,
        "segments": [{"input_start": 0, "input_end": len(source), "source": source,
                      "text": source, "kind": "raw"}],
    }


def test_render_public_preserves_scored_structural_source():
    source = "source"
    source_segments = segment(source, source, "mixed")
    paths = [("후보", 2.0, segment(source, "후보")),
             (source, 1.25, source_segments)]

    rendered = evaluator.render_public(source, paths)

    assert rendered[-1] == {"text": source, "score": 1.25, "segments": source_segments}


def test_probe_groups_are_deterministic_balanced_and_cover_every_request():
    requests = [(f"id-{i:03d}", "x" * (1 + i % 19), {}) for i in range(37)]

    first = evaluator.probe_groups(requests)
    second = evaluator.probe_groups(list(reversed(requests)))

    assert first == second
    assert len(first) == 16
    assert max(row["count"] for row in first) - min(row["count"] for row in first) <= 1
    assert sorted(identifier for group in first for identifier in group["ids"]) == sorted(
        identifier for identifier, _, _ in requests)
    lengths = {identifier: len(source) for identifier, source, _ in requests}
    for group in first:
        assert group["probe_id"] == group["ids"][-1]
        assert group["maximum_source_length"] == max(lengths[i] for i in group["ids"])


def ranked_fixture():
    source = "abc"
    base_paths = [
        ("가", 2.0, segment(source, "가")),
        ("나", 1.0, segment(source, "나")),
    ]
    candidates = evaluator.render_public(source, base_paths)
    ctc_word = {
        "generation_requests": 0, "completed_generations": 0,
        "generated_proposals": 0, "accepted_proposals": 0,
        "prediction": {"reason": "unsupported_source"}, "generation_ms": 1.0,
    }
    trace = {
        "union": ["가", "나"], "public": [row["text"] for row in candidates],
        "generation_ms": 2.0, "selected_extras": [], "missing_extras": [],
    }
    frozen_response = {
        "result": "가", "candidates": candidates, "context": "",
        "external_context_used": False, "sentence_ranking_applied": True,
        "ctc_word": ctc_word, "ctc_union": {"generation_ms": 2.0},
        "phonetic_word_union": trace,
    }
    rows = [{"text": text, "base_score": score, "features": [0.0] * 4,
             "adjusted_score": score} for text, score, _ in base_paths]
    response = copy.deepcopy(frozen_response)
    response["phonetic_choice"] = {
        "effective_weights": [0.0] * 4, "rows": rows, "zero_weight_parity": True,
    }
    return source, {"proposed": frozen_response}, response, base_paths


def test_verify_capture_accepts_exact_zero_shadow():
    source, frozen, response, paths = ranked_fixture()
    evaluator.verify_capture(source, frozen, response, paths, [0.0] * 4)


@pytest.mark.parametrize("mutation", ["source_text", "base_score", "adjusted_score"])
def test_verify_capture_rejects_altered_source_or_scores(mutation):
    source, frozen, response, paths = ranked_fixture()
    if mutation == "source_text":
        paths = [("다", paths[0][1], paths[0][2]), *paths[1:]]
    elif mutation == "base_score":
        response["phonetic_choice"]["rows"][0]["base_score"] += 0.25
    else:
        response["phonetic_choice"]["rows"][0]["adjusted_score"] += 0.25

    with pytest.raises(ValueError):
        evaluator.verify_capture(source, frozen, response, paths, [0.0] * 4)


def metric_response(top1, top5):
    texts = ["gold" if top1 else "wrong"]
    if top5 and not top1:
        texts.append("gold")
    return {"result": texts[0], "candidates": [{"text": text} for text in texts]}


def comparison_rows(*, regress_union=False):
    rows, labels = [], []
    for index in range(316):
        production = metric_response(index < 262, index < 282)
        union = metric_response(index < 273, index < 291)
        learned_top1 = index < 274
        learned_top5 = index < 292
        if regress_union and index == 0:
            learned_top1 = learned_top5 = False
        proposed = metric_response(learned_top1, learned_top5)
        trace = {
            "generated_extras": [], "scoreable_extras": [], "union": [], "public": [],
            "generation_budget_exhausted": False, "skipped_known_english": [],
        }
        rows.append({
            "id": f"id-{index:03d}", "request": {"input": f"src{index}", "context": ""},
            "parent": production, "union_reference": union, "proposed": {**proposed,
                "sentence_ranking_applied": False, "phonetic_word_union": trace},
            "parent_pool": [],
        })
        labels.append({"id": f"id-{index:03d}", "input": f"src{index}", "context": "",
                       "split": "dev", "expected": ["gold"]})
    return rows, labels


def test_compare_all_requires_improvement_without_union_or_production_losses():
    rows, labels = comparison_rows()
    result = evaluator.compare_all(rows, labels)

    assert result["versus_production"]["proposed"]["counts"]["top1"] == 274
    assert result["versus_union"]["parent"]["counts"]["top1"] == 273
    assert result["versus_union"]["regressions"] == 0
    assert result["versus_union"]["top5_losses"] == 0
    assert result["quality_gate"] is True

    regressed_rows, labels = comparison_rows(regress_union=True)
    regressed = evaluator.compare_all(regressed_rows, labels)
    assert regressed["versus_union"]["regressions"] == 1
    assert regressed["versus_union"]["top5_losses"] == 1
    assert regressed["quality_gate"] is False
