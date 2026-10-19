import json

import pytest

from scripts.probe_sentence_context_side import (
    canonical_rows_sha256,
    evaluate_case,
    read_prefix,
    summarize,
    visible_candidates,
)


def row(**changes):
    value = {
        "id": "natural-1", "input": "raw", "context": "긴 앞 문맥과 중요한 끝 문맥",
        "expected": ["정답"], "candidates": ["오답", "정답", "다른 답"],
        "teacher_scores": [3., 2., 1.], "positive_indices": [1],
        "sentence_eligible": True,
        "actual_candidates": ["오답", "정답", "다른 답", "raw"],
    }
    value.update(changes)
    return value


def test_visible_candidates_reproduces_raw_append_policy():
    ranked = ["0", "1", "2", "3", "4", "5"]
    assert visible_candidates(ranked, "2") == ["0", "1", "2", "3", "4"]
    assert visible_candidates(ranked, "raw") == ["0", "1", "2", "3", "raw"]


def test_evaluate_case_changes_only_ranking_and_uses_labels_afterward():
    first = evaluate_case(row(), [3., 2., 1.], [1., 4., 2.])
    changed_labels = evaluate_case(
        row(expected=["다른 답"], positive_indices=[2]), [3., 2., 1.], [1., 4., 2.]
    )

    assert first["right_actual"] == changed_labels["right_actual"]
    assert first["left_actual"] == changed_labels["left_actual"]
    assert first["right_first"] == "오답"
    assert first["left_first"] == "정답"
    assert first["right_top1"] is False and first["left_top1"] is True


def test_fallback_keeps_recorded_candidates_and_is_not_scored():
    fallback = row(sentence_eligible=False, candidates=["기존", "raw"],
                   actual_candidates=["기존", "raw"], expected=["기존"],
                   teacher_scores=[], positive_indices=[])
    result = evaluate_case(fallback, [], [])

    assert result["right_actual"] == result["left_actual"] == ["기존", "raw"]
    assert result["pool_covered"] is None
    with pytest.raises(ValueError, match="must not be rescored"):
        evaluate_case(fallback, [1.], [])


def test_fresh_right_scores_must_reproduce_recorded_runtime_order():
    with pytest.raises(ValueError, match="recorded runtime"):
        evaluate_case(row(), [1., 4., 2.], [1., 4., 2.])


def test_summary_reports_paired_corrections_and_regressions():
    correction = evaluate_case(row(id="fix"), [3., 2., 1.], [1., 4., 2.])
    regression_row = row(id="regress", expected=["오답"], positive_indices=[0])
    regression = evaluate_case(regression_row, [3., 2., 1.], [1., 4., 2.])
    metrics = summarize([correction, regression])

    assert metrics["corrections"] == ["fix"]
    assert metrics["regressions"] == ["regress"]
    assert metrics["first_choice_changes"] == 2


def test_read_prefix_snapshots_exact_complete_lines(tmp_path):
    path = tmp_path / "append.jsonl"
    first = {"id": 1}
    second = {"id": 2}
    raw = ((json.dumps(first) + "\n" + json.dumps(second) + "\n").encode())
    path.write_bytes(raw + b'{"id":3')

    snapshot, rows = read_prefix(path, 2)

    assert snapshot == raw
    assert rows == [first, second]
    assert canonical_rows_sha256(rows) == canonical_rows_sha256([first, second])
    with pytest.raises(ValueError, match="incomplete"):
        read_prefix(path, 3)
