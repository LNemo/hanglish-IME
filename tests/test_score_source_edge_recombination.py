from copy import deepcopy

import pytest

from scripts import score_source_edge_recombination as scoring


SOURCE = "imnida test"


def path(text, *, score=0.0, origin="parent", identifier="p"):
    return {"candidate_id": identifier, "path_id": identifier, "text": text,
            "score": score, "origin": origin, "origins": [origin],
            "segments": [{"input_start": 0, "input_end": len(SOURCE),
                          "source": SOURCE, "text": text, "kind": origin}]}


def proposal(text="입니다 테스트", identifier="new"):
    return {"proposal_id": identifier, "text": text,
            "segments": path(text)["segments"], "edge_ids": [identifier + "-edge"],
            "seed_index": 0, "seed_text": "임니다 테스트",
            "changed_edges": 1, "score": None, "score_type": None,
            "origin": "source_edge_recombination", "selection_eligible": False}


def runtime_row(pool=None):
    pool = pool or [path("임니다 테스트", identifier="old"),
                    path(SOURCE, score=None, origin="raw", identifier="raw")]
    raw = scoring.canonical_raw(SOURCE)
    response = {"input": SOURCE, "context": "", "result": "임니다 테스트",
                "candidates": [{"text": "임니다 테스트", "score": 1.0,
                                "segments": pool[0]["segments"]}, raw],
                "sentence_ranking_applied": True, "ranking_applied": True,
                "pipeline": {"pool": pool}}
    return {"id": "r1", "request": {"input": SOURCE, "context": ""},
            "proposed": response}


def proposal_row(values=None):
    values = list(values or [])
    inventory = [{"edge_id": value["edge_ids"][0], "action": "convert",
                  "copy_exact": None, "structural_segment": value["segments"][0],
                  "provenance": []} for value in values]
    return {"id": "r1", "request": {"input": SOURCE, "context": ""},
            "edge_inventory": inventory, "proposals": values}


class Ranker:
    def __init__(self, unsupported=()):
        self.unsupported = set(unsupported); self.calls = []
    def can_score(self, text):
        return text not in self.unsupported
    def score_pairs(self, contexts, texts):
        self.calls.append((list(contexts), list(texts)))
        return [float(index) for index in range(len(texts))]


class CountingRanker(Ranker):
    def __init__(self):
        super().__init__(); self.fit_calls = []
    def can_score(self, text):
        self.fit_calls.append(text)
        return True


def test_first_original_structural_path_wins_and_raw_is_reserved():
    first = path("임니다 테스트", identifier="first")
    second = path("임니다 테스트", identifier="second")
    pools = scoring.policy_pools(runtime_row([first, second, path(SOURCE, identifier="raw")]),
                                  proposal_row([proposal()]))
    assert [row["candidate_id"] for row in pools["parent_only_256"]] == ["first"]
    assert [row["text"] for row in pools["recombined_pool_256"]] == [
        "임니다 테스트", "입니다 테스트"]


def test_proposal_text_already_in_original_is_not_novel():
    prepared = scoring.prepare_row(runtime_row(), proposal_row([
        proposal("임니다 테스트")]), lambda text: True)
    assert prepared["has_proposals"] is False and prepared["score_texts"] == []


def test_shared_union_texts_are_scored_once():
    ranker = CountingRanker()
    result = scoring.score_row(runtime_row(), proposal_row([proposal()]), ranker, "model")
    assert len(ranker.calls) == 1
    assert ranker.calls[0][0] == ["", ""]
    assert ranker.calls[0][1] == ["임니다 테스트", "입니다 테스트"]
    assert result["score_texts"] == ranker.calls[0][1]
    assert ranker.fit_calls == ["임니다 테스트", "입니다 테스트"]


def test_no_proposals_preserves_original_exactly_without_scoring():
    original = runtime_row(); ranker = Ranker()
    result = scoring.score_row(original, proposal_row(), ranker, "model")
    assert ranker.calls == []
    for policy in scoring.POLICIES:
        assert result["policies"][policy]["response"] == original["proposed"]


def test_compact_metrics_drop_unrelated_heavy_pipeline():
    ranker = Ranker()
    result = scoring.score_row(runtime_row(), proposal_row([proposal()]), ranker, "model")
    result["policies"]["parent_only_256"]["response"]["pipeline"] = {
        "large_unrelated_marker": ["x"] * 1000}
    compact = scoring.compact_metric_row(result)
    assert compact["policies"]["parent_only_256"]["response"] == {
        "result": result["policies"]["parent_only_256"]["response"]["result"],
        "candidates": result["policies"]["parent_only_256"]["response"]["candidates"]}
    assert "pipeline" not in compact["policies"]["parent_only_256"]["response"]


def test_policy_falls_back_whole_when_any_candidate_is_unsupported():
    original = runtime_row(); ranker = Ranker(unsupported={"입니다 테스트"})
    result = scoring.score_row(original, proposal_row([proposal()]), ranker, "model")
    assert result["policies"]["parent_only_256"]["applied"] is True
    recombined = result["policies"]["recombined_pool_256"]
    assert recombined["applied"] is False
    assert recombined["response"] == original["proposed"]
    assert [row["text"] for row in recombined["structural_pool"]] == [
        "임니다 테스트", "입니다 테스트"]


def test_applied_policy_uses_only_new_scores_and_canonical_raw_last():
    original = runtime_row()["proposed"]
    pool = [path("임니다 테스트"), path("입니다 테스트", identifier="new")]
    result = scoring.apply_policy(original, pool,
        {"임니다 테스트": 0.0, "입니다 테스트": 2.0}, supported=True,
        source=SOURCE, model_id="m")
    assert result["response"]["result"] == "입니다 테스트"
    assert result["response"]["candidates"][-1] == scoring.canonical_raw(SOURCE)
    assert {row["score_type"] for row in result["structural_pool"]} == {
        "sentence_ranker_256"}


def test_incomplete_score_map_cannot_mix_old_and_new_domains():
    with pytest.raises(ValueError):
        scoring.apply_policy(runtime_row()["proposed"],
            [path("임니다 테스트"), path("입니다 테스트", identifier="new")],
            {"입니다 테스트": 2.0}, supported=True, source=SOURCE, model_id="m")


def metric(top1, top5=None, literal=True):
    top5 = top1 if top5 is None else top5
    return {"counts": {"top1": int(top1), "top5": int(top5)}, "outcomes": [{
        "id": "r1", "top1": top1, "top5": top5,
        "literal_signature_exact": literal}]}


def test_decision_requires_incremental_gain_over_parent_control():
    original = metric(False)
    parent = metric(True)
    recombined = metric(True)
    assert scoring.decide(original, parent, recombined)["selected_for_runtime_evaluation"] is None
    parent = metric(False)
    decision = scoring.decide(original, parent, recombined)
    assert decision["selected_for_runtime_evaluation"] == "recombined_pool_256"


def test_decision_rejects_incremental_regression_or_literal_loss():
    original = {"counts": {"top1": 1, "top5": 2}, "outcomes": [
        {"id": "a", "top1": True, "top5": True, "literal_signature_exact": True},
        {"id": "b", "top1": False, "top5": True, "literal_signature_exact": True}]}
    parent = deepcopy(original)
    recombined = {"counts": {"top1": 1, "top5": 1}, "outcomes": [
        {"id": "a", "top1": False, "top5": False, "literal_signature_exact": False},
        {"id": "b", "top1": True, "top5": True, "literal_signature_exact": True}]}
    decision = scoring.decide(original, parent, recombined)
    assert decision["selected_for_runtime_evaluation"] is None
    assert decision["recombined_versus_parent_only"]["regression_ids"] == ["a"]


def test_missing_pure_identity_fails_before_model_import(tmp_path):
    with pytest.raises(ValueError, match="identity differs"):
        scoring.validate_inputs(tmp_path, tmp_path / "proof.json")


def test_proposal_edges_must_resolve_to_exact_inventory_segments():
    row = proposal_row([proposal()])
    row["edge_inventory"][0]["structural_segment"] = deepcopy(
        row["edge_inventory"][0]["structural_segment"])
    row["edge_inventory"][0]["structural_segment"]["text"] = "다른 경로"
    with pytest.raises(ValueError, match="edge references"):
        scoring.policy_pools(runtime_row(), row)


def test_self_test_is_model_free():
    assert scoring.self_test() == {"schema": scoring.SCHEMA, "self_test": True}
