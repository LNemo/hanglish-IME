import pytest

from scripts.evaluate_boundary_provenance_fit import TwoPassRecorder


class Ranker:
    def score_pairs(self, contexts, candidates):
        return [float(i) for i in range(len(candidates))]


def test_both_passes_retained_but_final_overlay_is_exposed():
    r = TwoPassRecorder(Ranker())
    assert r.score_pairs(["c"], ["parent"]) == [0.]
    assert r.score_pairs(["c", "c"], ["parent", "new"]) == [0., 1.]
    assert r.candidates == ["parent", "new"] and r.scores == [0., 1.]
    assert r.calls[0]["candidates"] == ["parent"]
    assert len(r.calls) == 2
    with pytest.raises(RuntimeError, match="more than"):
        r.score_pairs(["c", "c"], ["parent", "new"])
    r.reset()
    assert r.calls == [] and r.candidates is None and r.scores is None
    assert r.score_pairs(["next"], ["other"]) == [0.]


@pytest.mark.parametrize("contexts,candidates", [
    (["c", "c"], ["changed", "new"]), (["changed", "c"], ["parent", "new"]),
])
def test_second_pass_cannot_change_parent_scoring_inputs(contexts, candidates):
    r = TwoPassRecorder(Ranker())
    r.score_pairs(["c"], ["parent"])
    with pytest.raises(ValueError, match="exact baseline"):
        r.score_pairs(contexts, candidates)
