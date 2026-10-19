import pytest
from scripts.evaluate_copy_child_fit import outcomes


def fixture():
    source={'id':'fit-1','request':{'input':'a','context':''},
            'scoring':{'ranked':True,'original':[{'text':'아'}],'novel':[], 'fallback_public':[]}}
    capture={'id':'fit-1','request':source['request'],'proposals':[{'text':'앗'}]}
    label={'id':'fit-1','input':'a','expected':['앗']}
    return source,capture,label


def test_new_gold_recovers_missing_candidate_without_ranking_claim():
    row=outcomes(*fixture())
    assert row['coverage_gain'] and row['expanded_available_covered']
    assert not row['old_scorable_covered'] and not row['coverage_loss']


def test_multiple_gold_and_unsupported_existing_gold_are_separate():
    source,capture,label=fixture()
    source['scoring']['novel']=[{'text':'아아','status':'unsupported'}]
    label['expected']=['앗','아아']
    row=outcomes(source,capture,label)
    assert row['old_available_covered'] and not row['old_scorable_covered']
    assert row['new_gold_proposals']==1 and not row['coverage_gain']


def test_no_pool_fallback_gold_stays_in_available_baseline():
    source,capture,label=fixture()
    source['scoring']={'ranked':False,'original':[],'novel':[],'fallback_public':[{'text':'아'}]}
    label['expected']=['아'];capture['proposals']=[]
    row=outcomes(source,capture,label)
    assert row['no_pool'] and row['old_available_covered'] and not row['old_scorable_covered']


def test_duplicate_old_candidate_and_mismatched_source_rejected():
    source,capture,label=fixture();capture['proposals']=[{'text':'아'}]
    with pytest.raises(ValueError,match='duplicates'):outcomes(source,capture,label)
    capture['proposals']=[];label['input']='different'
    with pytest.raises(ValueError,match='identity'):outcomes(source,capture,label)
