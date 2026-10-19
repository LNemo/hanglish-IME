import pytest

from hanglish.core.lattice import WordOption
from scripts.audit_current_cached_lattice import (
    CachedOptions,
    ReplayRanker,
    align_piece,
    classify,
    option_from_dict,
    target_segments,
)


def audit_row(source="gak", learned=True, text="각"):
    return {"source": source, "learned": learned,
            "baseline": [{"text": text, "score": 1., "kind": "neural"}],
            "additions": [], "result_count": 1}


def test_cached_options_fail_closed_and_reject_duplicate_audit_keys():
    cache = CachedOptions([audit_row()])

    assert cache("gak", True) == (WordOption("각", 1., "neural"),)
    with pytest.raises(KeyError, match="uncaptured"):
        cache("missing", True)
    assert cache.cache_misses == 1
    with pytest.raises(ValueError, match="duplicate"):
        CachedOptions([audit_row(), audit_row()])


@pytest.mark.parametrize("value", [
    {}, {"text": "", "score": 1., "kind": "raw"},
    {"text": "x", "score": float("nan"), "kind": "raw"},
    {"text": "x", "score": 1., "kind": ""},
])
def test_malformed_cached_options_are_rejected(value):
    with pytest.raises(ValueError, match="malformed"):
        option_from_dict(value)


def test_ranker_replays_exact_features_and_rejects_mismatch():
    groups = [{"context": "ctx [MASK]", "candidates": ["가", "각"],
               "teacher_scores": [1.5, 2.5]}]
    replay = ReplayRanker(groups)

    assert replay.score_pairs(["ctx [MASK]"] * 2, ["가", "각"]) == [1.5, 2.5]
    with pytest.raises(ValueError, match="differs"):
        replay.score_pairs(["changed"] * 2, ["가", "각"])


def test_literal_characters_align_individually():
    segments = target_segments("gak-2", "각-2")

    assert segments == [(0, 3, "각", "letter_run"),
                        (3, 4, "-", "literal"),
                        (4, 5, "2", "literal")]
    assert align_piece(["각-2"], "gak-2", 3, 4)["targets"] == ["-"]


def test_piece_cut_inside_whole_letter_run_is_unaligned():
    # The mixed letter run is kept whole; splitting AI from 는 would infer liaison.
    aligned = align_piece(["AI는"], "AIneun", 0, 2)

    assert aligned["targets"] == []
    assert aligned["boundary_cut"] is True


@pytest.mark.parametrize('source,expected,start,end,target', [
    ('gak-2', '각-2', 3, 5, '-2'),
    ('gak  2', '각  2', 3, 5, '  '),
    ('12gak', '12각', 0, 5, '12각'),
    ('gak-2', '각-2', 0, 5, '각-2'),
])
def test_contiguous_literal_and_letter_segments_can_cover_one_piece(source, expected, start, end, target):
    aligned = align_piece([expected], source, start, end)
    assert aligned == {'targets': [target], 'boundary_cut': False, 'valid_expected': 1}


def test_extra_replayed_context_is_not_silently_truncated_by_zip():
    replay = ReplayRanker([{'context': 'ctx', 'candidates': ['각'], 'teacher_scores': [1.]}])
    with pytest.raises(ValueError, match='differs'):
        replay.score_pairs(['ctx', 'extra'], ['각'])


def piece(*texts):
    return ("source", 0, tuple(WordOption(text, 1., "raw") for text in texts))


def test_classification_distinguishes_pool_piece_cap_and_beam_loss():
    expected = ["정답"]
    present = [piece("정답")]
    missing = [piece("오답")]
    capped = [piece(*([f"오답{i}" for i in range(8)] + ["정답"]))]

    assert classify(expected, present, present, ["정답"], True) == "pool_covered"
    assert classify(expected, present, present, ["정답"], False) == "deterministic_path_present"
    assert classify(expected, missing, missing, ["오답"], True) == "missing_fullpiece_candidate"
    assert classify(expected, capped, missing, ["오답"], True) == "top8_option_cap"
    assert classify(expected, present, missing, ["오답"], True) == "beam_pruning"
