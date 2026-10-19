from copy import deepcopy
import math

import pytest

from hanglish.core import source_edge_recombination as recombination


SOURCE = "abcdef"


def segment(start, end, text, kind="fixture"):
    return {"input_start": start, "input_end": end,
            "source": SOURCE[start:end], "text": text, "kind": kind}


def row(text, segments, index, *, origin="parent", score=0.0,
        candidate_id=None):
    identity = candidate_id or f"path-{index}"
    return {"candidate_id": identity, "path_id": identity,
            "pool_index": index, "text": text, "segments": segments,
            "origin": origin, "origins": [origin], "score": score}


def candidate(text, segments, score=0.0):
    return {"text": text, "segments": deepcopy(segments), "score": score}


def fixture():
    first = [segment(0, 2, "A", "first-a"),
             segment(2, 4, "B", "first-b"),
             segment(4, 6, "C", "first-c")]
    second = [segment(0, 2, "X", "second-x"),
              segment(2, 4, "Y", "second-y"),
              segment(4, 6, "Z", "second-z")]
    boundary = [segment(0, 3, "Q", "boundary-q"),
                segment(3, 4, "S", "boundary-s"),
                segment(4, 6, "C", "boundary-c")]
    pool = [row("ABC", first, 0), row("XYZ", second, 1),
            row("QSC", boundary, 2, origin="ctc_word")]
    base = [candidate("ABC", first, 99.0),
            candidate("XYZ", second, -99.0),
            {"text": SOURCE, "score": None,
             "segments": [{"input_start": 0, "input_end": len(SOURCE),
                            "source": SOURCE, "text": SOURCE, "kind": "raw"}]}]
    return pool, base


def assert_complete(source, proposal):
    cursor = 0
    output = []
    for item in proposal["segments"]:
        assert item["input_start"] == cursor
        assert item["input_end"] > cursor
        assert item["source"] == source[item["input_start"]:item["input_end"]]
        cursor = item["input_end"]
        output.append(item["text"])
    assert cursor == len(source)
    assert "".join(output) == proposal["text"]


def test_recombination_is_deterministic_novel_and_uses_only_saved_edges():
    pool, base = fixture()
    before_pool, before_base = deepcopy(pool), deepcopy(base)
    first = recombination.propose(SOURCE, pool, base)
    second = recombination.propose(SOURCE, pool, base)
    assert first == second
    assert pool == before_pool and base == before_base
    texts = [item["text"] for item in first["proposals"]]
    assert len(texts) == len(set(texts))
    assert not {"ABC", "XYZ", "QSC", SOURCE}.intersection(texts)
    assert "AYC" in texts
    allowed = {(part["input_start"], part["input_end"], part["text"])
               for value in pool for part in value["segments"]}
    inventory_ids = {edge["edge_id"] for edge in first["edge_inventory"]}
    for proposal in first["proposals"]:
        assert_complete(SOURCE, proposal)
        assert all((part["input_start"], part["input_end"], part["text"])
                   in allowed for part in proposal["segments"])
        assert set(proposal["edge_ids"]) <= inventory_ids
        assert "edge_evidence" not in proposal
        assert proposal["score"] is None and proposal["score_type"] is None
    assert all(first["proposals"][index]["proposal_id"] !=
               first["proposals"][other]["proposal_id"]
               for index in range(len(texts)) for other in range(index))


def test_boundary_changing_paths_and_cross_path_forms_are_reachable():
    pool, base = fixture()
    result = recombination.propose(SOURCE, pool, base)
    by_text = {item["text"]: item for item in result["proposals"]}
    assert "QSZ" in by_text
    assert [(part["input_start"], part["input_end"])
            for part in by_text["QSZ"]["segments"]] == [(0, 3), (3, 4), (4, 6)]
    assert by_text["QSZ"]["changed_edges"] >= 1


def test_edge_dedupe_keeps_first_segment_and_accumulates_all_provenance():
    pool, base = fixture()
    duplicate = [segment(0, 2, "A", "later-kind"),
                 segment(2, 4, "Y", "later-y"),
                 segment(4, 6, "Z", "later-z")]
    pool.append(row("AYZ", duplicate, 3, origin="growth"))
    result = recombination.propose(SOURCE, pool, base)
    edge = next(item for item in result["edge_inventory"]
                if item["structural_segment"]["input_start"] == 0
                and item["structural_segment"]["input_end"] == 2
                and item["structural_segment"]["text"] == "A")
    assert edge["structural_segment"]["kind"] == "first-a"
    assert [item["kind"] for item in edge["provenance"]] == [
        "first-a", "later-kind"]
    assert [item["origin"] for item in edge["provenance"]] == [
        "parent", "growth"]


def test_copy_action_is_exact_source_slice_and_output_is_never_sliced():
    copy_path = [segment(0, 2, "ab", "raw"), segment(2, 4, "K"),
                 segment(4, 6, "L")]
    alternate = [segment(0, 2, "M"), segment(2, 4, "cd", "raw"),
                 segment(4, 6, "N")]
    pool = [row("abKL", copy_path, 0), row("McdN", alternate, 1)]
    base = [candidate("abKL", copy_path), candidate("McdN", alternate)]
    result = recombination.propose(SOURCE, pool, base)
    copy_edges = [edge for edge in result["edge_inventory"]
                  if edge["action"] == "copy"]
    assert {(edge["structural_segment"]["source"],
             edge["structural_segment"]["text"]) for edge in copy_edges} == {
        ("ab", "ab"), ("cd", "cd")}
    assert all(edge["copy_exact"] is True for edge in copy_edges)


@pytest.mark.parametrize("bad_segments", [
    [segment(0, 2, "A"), segment(3, 6, "B")],
    [segment(0, 4, "A"), segment(3, 6, "B")],
    [{"input_start": 0, "input_end": 0, "source": "", "text": "A",
      "kind": "loop"}, segment(0, 6, "B")],
])
def test_gap_overlap_and_zero_length_loop_edges_are_rejected(bad_segments):
    bad = row("".join(item["text"] for item in bad_segments), bad_segments, 0)
    with pytest.raises(ValueError, match="coverage|forward"):
        recombination.propose(SOURCE, [bad], [candidate(bad["text"], bad_segments)])


def test_duplicate_path_ids_and_nonfinite_scores_are_irrelevant():
    pool, base = fixture()
    for value in pool:
        value["candidate_id"] = value["path_id"] = "duplicate"
        value["score"] = math.nan
    result = recombination.propose(SOURCE, pool, base)
    assert result["proposals"]


def test_complete_outputs_with_same_text_are_globally_deduplicated():
    granular = [segment(0, 2, "A"), segment(2, 4, ""), segment(4, 6, "B")]
    spanning = [segment(0, 4, "A"), segment(4, 6, "B")]
    variant = [segment(0, 2, "X"), segment(2, 4, ""), segment(4, 6, "B")]
    pool = [row("AB", granular, 0), row("AB", spanning, 1),
            row("XB", variant, 2)]
    base = [candidate("XB", variant)]
    result = recombination.propose(SOURCE, pool, base)
    texts = [item["text"] for item in result["proposals"]]
    assert len(texts) == len(set(texts))


def test_per_seed_pop_cap_is_reported_without_claiming_exhaustion(monkeypatch):
    pool, base = fixture()
    monkeypatch.setattr(recombination, "MAX_POPS_PER_SEED", 1)
    result = recombination.propose(SOURCE, pool, base)
    assert result["proposals"] == []
    assert {seed["termination_reason"] for seed in result["diagnostics"]["seeds"]} == {
        "max_pops"}
    assert all(seed["pops"] == 1 for seed in result["diagnostics"]["seeds"])


def test_per_seed_frontier_cap_is_reported(monkeypatch):
    pool, base = fixture()
    monkeypatch.setattr(recombination, "MAX_FRONTIER_PER_SEED", 1)
    result = recombination.propose(SOURCE, pool, base)
    assert {seed["termination_reason"] for seed in result["diagnostics"]["seeds"]} == {
        "max_frontier"}
    assert all(seed["max_frontier"] <= 1
               for seed in result["diagnostics"]["seeds"])


def test_global_limit_round_robins_seeds_and_reports_termination(monkeypatch):
    pool, base = fixture()
    monkeypatch.setattr(recombination, "MAX_PROPOSALS", 2)
    result = recombination.propose(SOURCE, pool, base)
    assert len(result["proposals"]) == 2
    assert result["diagnostics"]["global_limit_reached"] is True
    assert len({item["seed_index"] for item in result["proposals"]}) == 2
    assert "global_proposal_limit" in {
        seed["termination_reason"] for seed in result["diagnostics"]["seeds"]}


def test_seed_changed_edge_costs_are_monotonic_despite_round_robin():
    pool, base = fixture()
    result = recombination.propose(SOURCE, pool, base)
    for seed_index in range(result["diagnostics"]["seed_count"]):
        costs = [item["changed_edges"] for item in result["proposals"]
                 if item["seed_index"] == seed_index]
        assert costs == sorted(costs)
