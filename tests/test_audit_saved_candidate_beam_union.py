import pytest

from scripts import audit_saved_candidate_beam_union as module


def candidate(index, source, texts, bounds=None):
    bounds = bounds or [(0, len(source))]
    segments = []
    for number, ((start, end), text) in enumerate(zip(bounds, texts)):
        segments.append({"input_start": start, "input_end": end,
                         "source": source[start:end], "text": text,
                         "kind": "raw" if not text.isalpha() else "neural"})
    return {"source_index": index,
            "object": {"text": "".join(texts), "score": 0.0, "segments": segments}}


def test_saved_edges_recombine_competing_paths_without_losing_originals():
    row = {"index": 0, "id": "r", "input": "abcd",
           "finite_candidate_indices": [0, 1], "finite_candidates": [
        candidate(0, "abcd", ["가", "다"], [(0, 2), (2, 4)]),
        candidate(1, "abcd", ["나", "라"], [(0, 2), (2, 4)]),
    ]}
    graph = module.build_graph(row, {"abcd": {
        "outputs": [], "protected_row_ids": frozenset(),
        "unprotected_row_ids": frozenset({"r"})}})
    saved = module.arm_edges(graph, {"saved_candidate"})
    assert module.reachable("가다", saved, 4)
    assert module.reachable("나라", saved, 4)
    assert module.reachable("가라", saved, 4)
    assert saved == graph["edges"]


def test_beams_add_only_exact_unprotected_occurrence_edges():
    plain = {"index": 0, "id": "plain", "input": "gana",
             "finite_candidate_indices": [0], "finite_candidates": [
        candidate(0, "gana", ["가너"])]}
    beams = {"gana": {"outputs": [{"text": "가나", "score": -1.0}],
                      "protected_row_ids": frozenset({"protected"}),
                      "unprotected_row_ids": frozenset({"plain"})}}
    graph = module.build_graph(plain, beams)
    assert module.reachable("가나", module.arm_edges(
        graph, {"saved_candidate", "phonetic_beam"}), 4)
    assert not module.reachable("가나", module.arm_edges(
        graph, {"saved_candidate"}), 4)

    protected = {"index": 1, "id": "protected", "input": "`gana`",
                 "finite_candidate_indices": [0],
                 "finite_candidates": [candidate(0, "`gana`", ["`gana`"]) ]}
    protected_graph = module.build_graph(protected, beams)
    assert module.arm_edges(protected_graph, {"saved_candidate"}) == protected_graph["edges"]


def test_mixed_english_original_edge_is_preserved_without_fabricating_suffix_edge():
    row = {"index": 0, "id": "mixed", "input": "testreul",
           "finite_candidate_indices": [0],
           "finite_candidates": [candidate(0, "testreul", ["test를"])]}
    beams = {"testreul": {"outputs": [{"text": "테스트를", "score": -1.0}],
                           "protected_row_ids": frozenset(),
                           "unprotected_row_ids": frozenset({"mixed"})}}
    graph = module.build_graph(row, beams)
    assert module.reachable("test를", module.arm_edges(graph, {"saved_candidate"}), 8)
    assert module.reachable("test를", graph["edges"], 8)
    assert not any(edge["start"] == 4 and edge["text"] == "를"
                   for edge in graph["edges"])


def test_zero_length_or_nonclosing_original_edges_fail_closed():
    edges = {}
    with pytest.raises(ValueError, match="zero"):
        module.add_edge(edges, start=0, end=0, text="가", source="",
                        provenance={"type": "saved_candidate"})
    broken = candidate(0, "gana", ["가"])
    broken["object"]["segments"][0]["input_end"] = 3
    row = {"index": 0, "id": "r", "input": "gana",
           "finite_candidate_indices": [0], "finite_candidates": [broken]}
    with pytest.raises(ValueError, match="segment path"):
        module.build_graph(row, {})


def test_coverage_partition_distinguishes_exact_recombined_beam_and_missing():
    def graph(identity, saved, baseline_edges, union_edges):
        tagged = []
        for edge in union_edges:
            value = dict(edge)
            value["provenance"] = [{"type": "saved_candidate"}
                                   if edge in baseline_edges else {"type": "phonetic_beam"}]
            tagged.append(value)
        return {"id": identity, "input": "ab", "saved_candidate_texts": saved,
                "edges": tagged}
    edge_a = {"start": 0, "end": 1, "text": "가", "source": "a", "provenance": []}
    edge_b = {"start": 1, "end": 2, "text": "나", "source": "b", "provenance": []}
    edge_c = {"start": 1, "end": 2, "text": "다", "source": "b", "provenance": []}
    graphs = [
        graph("exact", ["가나"], [edge_a, edge_b], [edge_a, edge_b]),
        graph("recombined", ["기나"], [edge_a, edge_b], [edge_a, edge_b]),
        graph("beam", ["기나"], [edge_a, edge_c], [edge_a, edge_c, edge_b]),
        graph("missing", ["기나"], [edge_c], [edge_c]),
    ]
    targets = [{"id": value, "expected": ["가나"]}
               for value in ("exact", "recombined", "beam", "missing")]
    old_rows = module.ROWS
    module.ROWS = 4
    try:
        counts, _ = module.evaluate_graphs(graphs, targets)
    finally:
        module.ROWS = old_rows
    assert counts["saved_exact"] == counts["newly_recombined"] == 1
    assert counts["newly_beam"] == counts["still_missing"] == 1
    assert sum(counts["full_partition_" + key] for key in (
        "saved_exact", "newly_recombined", "newly_mechanical",
        "newly_beam_after_mechanical", "still_missing")) == 4
    assert counts["nonrecombined_beam_only"] == 1
    assert counts["nonrecombined_neither"] == 1


def test_mechanical_baseline_is_separate_arm_and_proposals_are_ignored():
    row = {"index": 0, "id": "r", "input": "gana",
           "finite_candidate_indices": [0], "finite_candidates": [
        candidate(0, "gana", ["가너"])]}
    beams = {"gana": {"outputs": [], "protected_row_ids": frozenset(),
                      "unprotected_row_ids": frozenset({"r"})}}
    mechanical = {"id": "r", "input": "gana", "pieces": [{
        "start": 0, "end": 4, "source": "gana",
        "baseline": [{"text": "가나", "score": 1.0, "kind": "syllable"}],
        "proposals": [{"text": "가마", "score": 9.0, "kind": "experimental"}],
    }]}
    graph = module.build_graph(row, beams, mechanical)
    saved = module.arm_edges(graph, {"saved_candidate"})
    mechanical_arm = module.arm_edges(graph, {"saved_candidate", "mechanical_baseline"})
    assert not module.reachable("가나", saved, 4)
    assert module.reachable("가나", mechanical_arm, 4)
    assert not module.reachable("가마", mechanical_arm, 4)
