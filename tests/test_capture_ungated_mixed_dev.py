from copy import deepcopy
import threading

from hanglish.core.lattice import WordOption
from scripts.capture_ungated_mixed_dev import PARENT_VERSION, capture_row


def seg(source,start,end,text,kind):
    return {"input_start":start,"input_end":end,"source":source[start:end],"text":text,"kind":kind}


def records(ranked):
    source="api3beta imnida"; request={"input":source,"context":""}
    response={"input":source,"context":"","version":PARENT_VERSION,
      "sentence_ranking_applied":ranked,"ranking_applied":True,"warnings":[],
      "candidates":[{"text":"api3beta 임니다","score":0.,"segments":[
        seg(source,0,8,"api3beta","raw"),seg(source,8,9," ","raw"),seg(source,9,15,"임니다","lexicon")]},
        {"text":source,"score":None,"segments":[seg(source,0,len(source),source,"raw")]}]}
    return ({"id":"d1","request":request,"proposed":{"sentence_ranking_applied":ranked}},
            {"id":"d1","request":request,"response":response})


class Ranker:
    def can_score(self,text): return True
    def score_pairs(self,contexts,texts): return [4. if "베타" in x else 1. for x in texts]


class Engine:
    def __init__(self,response):
        self.response=deepcopy(response);self.lock=threading.RLock();self.version=PARENT_VERSION
        self.sentence_ranker=Ranker()
    def convert(self,text): return deepcopy(self.response)
    def load_phonetic_model(self):
        return type("M",(),{"generate":staticmethod(lambda word,beam_size=8:[])})()
    def word_options(self,word,learned=True):
        values={"api":WordOption("아피",2.,"lexicon"),"beta":WordOption("베타",1.,"lexicon")}
        return [values[word]] if word in values else [WordOption(word,0.,"raw")]


def test_ranked_row_is_exact_cached_passthrough_without_engine():
    source,comparator=records(True)
    row=capture_row(source,comparator,engine=None)
    assert row["selected_unranked"] is False and row["passthrough_exact"] is True
    assert "response" not in row


def test_selected_row_captures_exact_parent_trace_and_score_call_boundaries():
    source,comparator=records(False);engine=Engine(comparator["response"])
    row=capture_row(source,comparator,engine=engine)
    assert row["fresh_baseline"]["candidates"]==comparator["response"]["candidates"]
    assert row["trace"]["baseline"]==row["fresh_baseline"]
    assert row["trace"]["joint_trace"]["baseline"]==row["fresh_baseline"]
    joint=row["trace"]["joint_trace"]["pool"]
    union=row["trace"]["union_pool"]
    assert [(x["text"],x["segments"],x["supported"],x["score"]) for x in union[:len(joint)]] == [
        (x["text"],x["segments"],x["supported"],x["score"]) for x in joint]
    assert row["controls"]["parent_candidates_exact_cached"] is True
    assert [call["texts"] for call in row["sentence_score_calls"]] == [
        ["api3beta 임니다"],
        [item["text"] for item in row["trace"]["union_pool"]
        if item["origins"]==["mixed"] and item["supported"]]]
    assert row["response"]["candidates"][-1] == row["fresh_baseline"]["candidates"][-1]


def test_selected_row_retains_raw_copy_provenance_and_unsupported():
    source,comparator=records(False);engine=Engine(comparator["response"])
    engine.sentence_ranker.can_score=lambda text: "베타" not in text
    row=capture_row(source,comparator,engine=engine)
    unsupported=[item for item in row["trace"]["union_pool"] if not item["supported"]]
    assert unsupported and all(item["score"] is None for item in unsupported)
    assert all(item["origins"] for item in row["trace"]["union_pool"])
    for item in row["trace"]["union_pool"]:
        assert all(segment["text"]==segment["source"]
                   for segment in item["segments"] if segment["kind"]=="raw")


def test_fresh_parent_candidate_mismatch_is_rejected():
    import pytest
    source,comparator=records(False);engine=Engine(comparator["response"])
    engine.response["candidates"][0]["text"]="changed"
    with pytest.raises(ValueError,match="candidate identity"):
        capture_row(source,comparator,engine=engine)
