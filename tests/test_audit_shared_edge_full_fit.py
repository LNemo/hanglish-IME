from copy import deepcopy

import pytest

from hanglish.core.source_correspondence import align, compact_paths
from scripts.audit_shared_edge_full_fit import project, summarize_row


def _path(source, text):
    return align(
        source,
        text,
        [{"input_start": 0, "input_end": len(source), "source": source, "text": text}],
    )


def _project_inputs():
    source = "imnida"
    split_positive = align(source, "입니다", [
        {"input_start": 0, "input_end": 2, "source": "im", "text": "입"},
        {"input_start": 2, "input_end": 6, "source": "nida", "text": "니다"},
    ])
    paths = [_path(source, text) for text in ("입니다", "임니다", "읩니다")]
    paths.append(split_positive)
    record = compact_paths(paths)
    record.update(id="row-1", component_id="component-1")
    selected = [{
        "id": "row-1",
        "input": source,
        "component_id": "component-1",
        "candidates": ["임니다", "입니다"],
        "positive_indices": [1],
    }]
    capture = {
        "id": "row-1",
        "input": source,
        "component_id": "component-1",
        "split": "fit",
        # Deliberately differs from lexical candidate order.
        "score_texts": ["읩니다", "입니다", "임니다"],
        "scores": [0.3, 0.9, 0.4],
    }
    return selected, {"row-1": record}, {"row-1": capture}


def test_project_preserves_full_path_and_parent_score_domains():
    selected, records, captures = _project_inputs()

    prepared, groups, targets = project(selected, records, captures)

    assert prepared == [{
        "id": "row-1",
        "input": "imnida",
        "component_id": "component-1",
        "candidates": ["읩니다", "임니다", "입니다"],
        "positive_indices": [2],
        "selected_indices": [1, 2],
        "base_scores": [0.3, 0.4, 0.9],
        "candidate_path_counts": [1, 1, 2],
    }]
    assert [[path.text for path in candidate_paths] for candidate_paths in groups["row-1"]] == [
        ["읩니다"], ["임니다"], ["입니다", "입니다"]
    ]
    assert set(targets) == {"니다", "읩니다", "임니다", "입", "입니다"}


def test_project_requires_all_positives_to_remain_in_selected_domain():
    selected, records, captures = _project_inputs()
    selected[0]["candidates"] = ["임니다"]
    selected[0]["positive_indices"] = []

    with pytest.raises(ValueError, match="FIT positive domain"):
        project(selected, records, captures)


@pytest.mark.parametrize("mutation", [
    {"split": "inner_hold"},
    {"expected": ["입니다"]},
])
def test_project_rejects_non_fit_or_gold_bearing_source_capture(mutation):
    selected, records, captures = _project_inputs()
    captures = deepcopy(captures)
    captures["row-1"].update(mutation)

    with pytest.raises(ValueError):
        project(selected, records, captures)


def _groups(candidates):
    return [[_path("x", text)] for text in candidates]


@pytest.mark.parametrize("selected_count", [5, 7, 8])
def test_same_forward_restriction_has_no_apparent_expansion_gain(selected_count):
    candidates = [f"c{i}" for i in range(selected_count + 2)]
    row = {
        "candidates": candidates,
        "positive_indices": [2],
        "selected_indices": list(range(selected_count)),
    }
    scores = [float(-i) for i in range(len(candidates))]
    scores[2] = 5.0

    stats = summarize_row(row, scores, set(candidates), _groups(candidates))

    assert stats["restricted_top1"] == stats["full_top1"] == 1
    assert stats["restricted_top5"] == stats["full_top5"] == 1
    assert stats["top1_expansion_loss"] == stats["top5_expansion_loss"] == 0


def test_expansion_losses_are_monotone_for_top1_and_top4():
    candidates = [f"c{i}" for i in range(10)]
    row = {
        "candidates": candidates,
        "positive_indices": [4],
        "selected_indices": [0, 1, 2, 3, 4],
    }
    # The positive wins the restricted domain, while four excluded candidates
    # move it below the four ranked nonraw slots in the full domain.
    scores = [1.0, 2.0, 3.0, 4.0, 10.0, 14.0, 13.0, 12.0, 11.0, 0.0]

    stats = summarize_row(row, scores, set(), _groups(candidates))

    assert stats["restricted_top1"] == stats["restricted_top5"] == 1
    assert stats["full_top1"] == stats["full_top5"] == 0
    assert stats["top1_expansion_loss"] == stats["top5_expansion_loss"] == 1
    assert stats["winner_outside_selected"] == 1
    assert stats["excluded_above_best_positive"] == 4


def test_lexical_tie_uses_same_deterministic_order_in_both_domains():
    candidates = ["a", "b", "z"]
    row = {
        "candidates": candidates,
        "positive_indices": [1],
        "selected_indices": [1, 2],
    }

    stats = summarize_row(row, [3.0, 3.0, 0.0], set(), _groups(candidates))

    assert stats["restricted_top1"] == 1
    assert stats["full_top1"] == 0
    assert stats["top1_expansion_loss"] == 1
    assert stats["winner_outside_selected"] == 1
    # This count is intentionally strict-score-only; the separate winner/loss
    # fields retain expansion losses caused solely by lexical tie-breaking.
    assert stats["excluded_above_best_positive"] == 0


def test_summarize_rejects_positive_outside_restricted_domain():
    candidates = ["a", "b"]
    row = {"candidates": candidates, "positive_indices": [1], "selected_indices": [0]}

    with pytest.raises(ValueError, match="positive escaped restricted domain"):
        summarize_row(row, [0.0, 1.0], set(), _groups(candidates))
