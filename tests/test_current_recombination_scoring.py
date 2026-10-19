from copy import deepcopy
import ast
from pathlib import Path

import pytest

from scripts.current_recombination_scoring import canonical_sha256, prepare_row, score_row
import scripts.current_recombination_scoring as scoring


def segment(start, source, text, kind="fixture"):
    return {"input_start": start, "input_end": start + len(source),
            "source": source, "text": text, "kind": kind}


def score(pair_id, source, target, mean):
    tokens = 3 * len(target) + 1
    return {"pair_id": pair_id, "source": source, "target": target,
            "log_probability": mean * tokens, "token_count": tokens,
            "mean_log_probability": mean}


def fixture():
    source = "ganada"
    a_segments = [segment(0, "ga", "ga", "lexical"), segment(2, "na", "na"),
                  segment(4, "da", "다")]
    b_segments = [segment(0, "ga", "ga", "ctc"), segment(2, "na", "나"),
                  segment(4, "da", "da")]
    new_segments = [segment(0, "ga", "ga", "lexical"), segment(2, "na", "나"),
                    segment(4, "da", "다")]
    ga = {"input_start": 0, "input_end": 2, "source": "ga", "target": "가",
          "status": "supported", "pair_id": "pair-ga", "romanize_match": True}
    na_copy = {"input_start": 2, "input_end": 4, "source": "na", "target": "na",
               "status": "copy"}
    da = {"input_start": 4, "input_end": 6, "source": "da", "target": "다",
          "status": "supported", "pair_id": "pair-da", "romanize_match": True}
    ga_copy = {"input_start": 0, "input_end": 2, "source": "ga", "target": "ga",
               "status": "copy"}
    na = {"input_start": 2, "input_end": 4, "source": "na", "target": "나",
          "status": "supported", "pair_id": "pair-na", "romanize_match": True}
    da_copy = {"input_start": 4, "input_end": 6, "source": "da", "target": "da",
               "status": "copy"}
    records = [score("pair-na", "na", "나", -2.0),
               score("pair-da", "da", "다", -3.0)]
    rows = [
        {"text": "gana다", "base_score": 2.0,
         "features": [-1.0, 1 / 3, 0.0, 0.0], "segments": a_segments,
         "edges": [ga_copy, na_copy, da]},
        {"text": "ga나da", "base_score": 1.0,
         "features": [-2 / 3, 1 / 3, 0.0, 0.0], "segments": b_segments,
         "edges": [ga_copy, na, da_copy]},
    ]
    paths = [["gana다", 2.0, a_segments], ["ga나da", 1.0, b_segments]]
    public = [{"text": "gana다", "score": 2.0, "segments": a_segments},
              {"text": "ga나da", "score": 1.0, "segments": b_segments},
              {"text": source, "score": None,
               "segments": [segment(0, source, source, "raw")]}]
    original = {"id": "row-1", "component_id": "component-1",
                "request": {"input": source, "context": ""}, "base_paths": paths,
                "response": {"input": source, "context": "", "external_context_used": False,
                             "sentence_ranking_applied": True, "candidates": public,
                             "phonetic_choice": {"rows": rows, "score_records": records}}}

    def provenance(pool, part, kind="fixture"):
        identity = f"current-phonetic-fit:row-1:{pool}"
        return [{"candidate_id": identity, "path_id": identity, "pool_index": pool,
                 "segment_index": part, "origin": "captured_full_union",
                 "origins": ["captured_full_union"], "kind": kind}]

    inventory = [
        {"edge_id": "copy-ga", "structural_segment": new_segments[0],
         "provenance": provenance(0, 0, "lexical") + provenance(1, 0, "ctc")},
        {"edge_id": "convert-na", "structural_segment": new_segments[1],
         "provenance": provenance(1, 1)},
        {"edge_id": "convert-da", "structural_segment": new_segments[2],
         "provenance": provenance(0, 2)},
    ]
    proposal = {"proposal_id": "proposal-1", "text": "ga나다",
                "segments": new_segments,
                "edge_ids": ["copy-ga", "convert-na", "convert-da"]}
    census = {"schema": "current-fit-recombination-census-row-v1", "id": "row-1",
              "component_id": "component-1", "request": original["request"],
              "source_reference": {"source_shard": 0, "source_row_index": 3,
                                   "source_capture_sha256": "a" * 64,
                                   "base_paths_sha256": canonical_sha256(paths),
                                   "public_candidates_sha256": canonical_sha256(public),
                                   "pool_rows": 2, "ranked": True},
              "status": "generated", "proposals": [proposal],
              "used_edge_inventory": inventory, "fallback_public": None}
    return original, census


class Ranker:
    def __init__(self, unsupported=()):
        self.unsupported = set(unsupported)
        self.calls = []

    def can_score(self, text):
        return text not in self.unsupported

    def score_pairs(self, contexts, texts):
        self.calls.append((list(contexts), list(texts)))
        values = {"ga나다": 4.5, "gana다": 2.0, "ga나da": 1.0}
        return [values[text] for text in texts]


def test_scores_novel_candidate_and_preserves_original_scores_with_probe_parity():
    original, census = fixture(); ranker = Ranker()
    result = score_row(original, census, ranker, check_original_parity=True)
    assert [row["sentence_score"] for row in result["original"]] == [2.0, 1.0]
    assert result["novel"][0]["sentence_score"] == 4.5
    assert result["novel"][0]["status"] == "scored"
    assert result["original_score_parity"] == {
        "checked": True, "tolerance": 1e-5, "max_absolute_difference": 0.0, "rows": 2}
    assert ranker.calls == [([""], ["ga나다"]), (["", ""], ["gana다", "ga나da"])]


def test_original_probe_parity_uses_fixed_one_e_minus_five_tolerance():
    original, census = fixture()

    class Drifted(Ranker):
        def score_pairs(self, contexts, texts):
            values = super().score_pairs(contexts, texts)
            return ([values[0] + 2e-5, *values[1:]]
                    if texts == ["gana다", "ga나da"] else values)

    with pytest.raises(ValueError, match="score parity"):
        score_row(original, census, Drifted(), check_original_parity=True)


def test_multisegment_features_reuse_exact_captured_scores():
    original, census = fixture()
    prepared = prepare_row(original, census, Ranker())
    assert prepared["novel"][0]["features"] == pytest.approx([-5 / 3, 2 / 3, 0.0, 0.0])
    assert prepared["novel"][0]["edge_ids"] == ["copy-ga", "convert-na", "convert-da"]


def test_shared_structural_edge_accepts_heterogeneous_provenance_kinds():
    original, census = fixture()
    evidence = census["used_edge_inventory"][0]
    assert evidence["structural_segment"]["kind"] == "lexical"
    assert [row["kind"] for row in evidence["provenance"]] == ["lexical", "ctc"]
    assert prepare_row(original, census, Ranker())["novel"][0]["status"] == "ready"


@pytest.mark.parametrize("mutate,match", [
    (lambda o, c: c.update(id="wrong"), "identity"),
    (lambda o, c: c["used_edge_inventory"][0]["structural_segment"].update(text="foreign"),
     "provenance|identity"),
    (lambda o, c: o["response"]["phonetic_choice"]["score_records"][1].update(target="너"),
     "score identity"),
])
def test_rejects_source_edge_and_score_mismatches(mutate, match):
    original, census = fixture(); mutate(original, census)
    with pytest.raises(ValueError, match=match):
        prepare_row(original, census, Ranker())


def test_unsupported_candidate_is_retained_without_scoring_or_truncation():
    original, census = fixture(); ranker = Ranker(unsupported={"ga나다"})
    result = score_row(original, census, ranker)
    assert len(result["novel"]) == 1
    assert result["novel"][0]["status"] == "unsupported"
    assert result["novel"][0]["reason"] == "sentence_ranker_can_score_false"
    assert result["novel"][0]["sentence_score"] is None
    assert ranker.calls == []


def test_no_pool_preserves_nonraw_fallbacks_and_scores_nothing():
    source = "roman"
    fallback = [{"text": "로만", "score": None,
                 "segments": [segment(0, source, "로만")]},
                {"text": source, "score": None,
                 "segments": [segment(0, source, source, "raw")]}]
    original = {"id": "empty", "component_id": "component-empty",
                "request": {"input": source, "context": ""}, "base_paths": [],
                "response": {"input": source, "context": "", "external_context_used": False,
                             "sentence_ranking_applied": False, "candidates": fallback,
                             "phonetic_choice": {"rows": [], "score_records": []}}}
    census = {"schema": "current-fit-recombination-census-row-v1", "id": "empty",
              "component_id": "component-empty", "request": original["request"],
              "source_reference": {"source_shard": 2, "source_row_index": 9,
                                   "source_capture_sha256": "b" * 64,
                                   "base_paths_sha256": canonical_sha256([]),
                                   "public_candidates_sha256": canonical_sha256(fallback),
                                   "pool_rows": 0, "ranked": False},
              "status": "no_pool", "proposals": [], "used_edge_inventory": [],
              "fallback_public": deepcopy(fallback)}
    ranker = Ranker()
    result = score_row(original, census, ranker, check_original_parity=True)
    assert result["fallback_public"] == fallback
    assert result["original"] == result["novel"] == []
    assert result["original_score_parity"]["rows"] == 0
    assert ranker.calls == []


def test_core_has_no_file_model_or_label_loading_surface():
    tree = ast.parse(Path(scoring.__file__).read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not any(name.startswith(("torch", "transformers")) for name in imported)
    calls = {node.func.id for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert not {"open", "read_json", "iter_jsonl"}.intersection(calls)
