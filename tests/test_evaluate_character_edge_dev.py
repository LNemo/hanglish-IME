from copy import deepcopy

import pytest

from scripts.evaluate_character_edge_dev import ARMS, aggregate, evaluate_row


def fixture():
    capture = {"id":"dev-1", "dataset_split":"dev", "request":{"input":"testreul handa.","context":""},
               "scoring":{"ranked":True,"original":[{"text":"test를 한다."}],"novel":[]},
               "predictions":{arm:[{"text":"test를 한다."}] for arm in ARMS}}
    label = {"id":"dev-1","input":"testreul handa.","split":"dev","expected":["test를 한다."]}
    return capture,label


def test_correctness_and_ascii_regression_are_independent_of_coverage():
    capture,label=fixture()
    capture["predictions"]["expanded_character"]=[{"text":"테스트를 한다."}]
    row=evaluate_row(capture,label)
    assert row["arms"]["expanded_character"]["pool_covered"]
    assert row["transitions"]["overall"]["regressions"]==1
    assert row["transitions"]["overall"]["ascii_regressions"]==1
    assert not aggregate([row])["quality_gates"]["expanded_character"]


def test_multi_gold_and_unranked_rows_remain_in_denominator():
    capture,label=fixture(); label["expected"].append("test를 해요.")
    capture["scoring"]={"ranked":False,"original":[],"novel":[]}
    for arm in ARMS:capture["predictions"][arm]=[{"text":"test를 해요."}]
    result=aggregate([evaluate_row(capture,label)])
    assert result["branches"]["expanded_character"]["rows"]==1
    assert result["branches"]["expanded_character"]["first"]==1
    assert result["branches"]["expanded_character"]["pool_covered"]==0


def test_unsupported_gold_not_claimed_as_scored_pool_coverage():
    capture,label=fixture();capture["scoring"]["original"]=[]
    capture["scoring"]["novel"]=[{"text":label["expected"][0],"status":"unsupported"}]
    assert not evaluate_row(capture,label)["arms"]["expanded_character"]["pool_covered"]


def test_gain_does_not_hide_another_row_regression_or_top5_loss():
    capture,label=fixture(); capture["predictions"]["current_production"]=[{"text":"오답"}]
    gain=evaluate_row(capture,label)
    assert aggregate([gain])["quality_gates"]["expanded_character"]
    capture2,label2=fixture();capture2["id"]=label2["id"]="dev-2"
    capture2["predictions"]["expanded_character"]=[{"text":"오답"}]
    loss=evaluate_row(capture2,label2)
    assert not aggregate([gain,gain,loss])["quality_gates"]["expanded_character"]


def test_mismatched_source_rejected():
    capture,label=fixture(); label["input"]="another source"
    with pytest.raises(ValueError,match="contract"):evaluate_row(capture,label)
