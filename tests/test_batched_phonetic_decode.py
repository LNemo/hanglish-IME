from types import SimpleNamespace

import pytest

from hanglish.models.batched_phonetic_decode import beam_search, generate_batched, rank_completed


def scalar_reference(initial, table, *, bos=0, eos=9, width=3, steps=4):
    beams = [([bos], 0.0, initial)]
    completed = []
    for _ in range(steps):
        expanded = []
        for tokens, score, state in beams:
            if tokens[-1] == eos:
                completed.append((tokens, score))
                continue
            for token, value, next_state in table[(state, tokens[-1])]:
                expanded.append((tokens + [token], score + value, next_state))
        if not expanded:
            break
        beams = sorted(expanded, key=lambda item: item[1], reverse=True)[:width]
    completed.extend((tokens, score) for tokens, score, _ in beams if tokens[-1] == eos)
    return completed


def test_batched_bookkeeping_matches_scalar_order_completion_and_pruning():
    table = {
        ("s0", 0): [(1, -0.1, "a"), (2, -0.1, "b"), (3, -2.0, "c")],
        ("a", 1): [(9, -0.2, "done-a"), (4, -0.5, "a2")],
        ("b", 2): [(5, -0.2, "b2"), (9, -0.8, "done-b")],
        ("c", 3): [(9, -0.1, "done-c")],
        ("a2", 4): [(9, -0.1, "done-a2")],
        ("b2", 5): [(9, -0.1, "done-b2")],
    }
    calls = []

    def expand(active):
        calls.append([(state, tokens[-1]) for tokens, _, state in active])
        return [table[(state, tokens[-1])] for tokens, _, state in active]

    actual = beam_search("s0", bos_id=0, eos_id=9, search_width=3,
                         max_steps=4, expand_active=expand)
    expected = scalar_reference("s0", table)
    assert actual == expected
    assert calls[0] == [("s0", 0)]
    assert all(token != 9 for call in calls for _, token in call)


def test_final_iteration_eos_is_retained_and_empty_expansion_stops():
    table = {("start", 0): [(1, -0.1, "middle")],
             ("middle", 1): [(9, -0.2, "done")]}
    actual = beam_search(
        "start", bos_id=0, eos_id=9, search_width=8, max_steps=2,
        expand_active=lambda active: [table[(state, tokens[-1])]
                                      for tokens, _, state in active],
    )
    assert actual == [([0, 1, 9], pytest.approx(-0.3))]
    stopped = beam_search(
        "start", bos_id=0, eos_id=9, search_width=8, max_steps=2,
        expand_active=lambda active: [[] for _ in active],
    )
    assert stopped == []


def test_rank_completed_preserves_stable_ties_dedup_and_forward_filter():
    output_tokens = ["<bos>", "A", "B", "C", "<eos>"]
    completed = [
        ([0, 1, 4], -1.0),
        ([0, 2, 4], -1.0),
        ([0, 1, 4], -0.5),
        ([0, 3, 4], -0.1),
    ]
    compose = lambda tokens: {("A",): "가", ("B",): "나", ("C",): "다"}.get(tuple(tokens))
    romanize = {"가": "ga", "나": "ga", "다": "da"}.__getitem__
    assert rank_completed(completed, "ga", 2, output_tokens=output_tokens,
                          compose=compose, romanize=romanize) == [
        ("가", -0.5), ("나", -1.0)]


def test_search_rejects_malformed_batched_expansion():
    with pytest.raises(ValueError, match="active beams"):
        beam_search("s", bos_id=0, eos_id=9, search_width=2, max_steps=1,
                    expand_active=lambda active: [])
    with pytest.raises(ValueError, match="log probability"):
        beam_search("s", bos_id=0, eos_id=9, search_width=2, max_steps=1,
                    expand_active=lambda active: [[(1, 0, "next")]])


@pytest.mark.parametrize("roman,beam,message", [
    ("Ga", 4, "lowercase ASCII"),
    ("a" * 49, 4, "at most 48"),
    ("ga", True, "integer from 1 to 16"),
    ("ga", 17, "integer from 1 to 16"),
])
def test_public_validation_matches_frozen_bounds_without_loading_a_model(roman, beam, message):
    transducer = SimpleNamespace(config=SimpleNamespace(max_roman_length=48))
    with pytest.raises(ValueError, match=message):
        generate_batched(transducer, roman, beam)
