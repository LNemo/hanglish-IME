from copy import deepcopy
import pytest

from scripts.finite_final_choice_evaluation import evaluate_case,evaluation_gate,summarize

def row():
    def item(text,score,kind='neural'):
        return {'text':text,'score':score,'segments':[{'input_start':0,'input_end':6,'source':'imnida','text':text,'kind':kind}]}
    raw=item('imnida',None,'raw')
    return {'id':'one','group_id':'g','input':'imnida','context':'앞 문맥',
            'candidates':[item('임니다',2.),raw,item('입니다',1.)],
            'raw_bypass_response':{'result':'imnida','candidates':[item('imnida',0.,'raw')]},
            'expected':['입니다'],'candidate_can_score':[False,False,False],
            'selector_eligible':False}

class Ranker:
    def can_score(self,text):
        assert text != 'imnida'
        return True
    def score_pairs(self,contexts,candidates):
        assert candidates == ['임니다','입니다']
        return [0.,2.]

def test_new_finite_budget_and_index_mapping_preserve_all_objects():
    value=row();before=deepcopy(value)
    result=evaluate_case(Ranker(),value,'natural_evaluation_v3')
    assert result['chosen_index']==2 and result['scored_source_indices']==[0,2]
    assert result['top1'] and not result['before_top1']
    assert result['top5'] and result['before_top5']
    assert value==before

def test_labels_read_only_after_model_choice():
    state={'scored':False}
    class Guard(dict):
        def __getitem__(self,key):
            if key=='expected':assert state['scored']
            return super().__getitem__(key)
    class CheckedRanker(Ranker):
        def score_pairs(self,contexts,candidates):
            state['scored']=True
            return super().score_pairs(contexts,candidates)
    assert evaluate_case(CheckedRanker(),Guard(row()),'natural_evaluation_v3')['top1']

def suites(gain=2,regression=False,actions=2):
    return {'natural_evaluation_v3':{'top1':12+gain,'before_top1':12,'regressions':['bad'] if regression else [],'top5_losses':[],'accepted_actions':actions},
            'challenge_preservation':{'regressions':[],'top5_losses':[],'accepted_actions':0}}

@pytest.mark.parametrize('gain,regression,actions,passed',[(2,False,2,True),(1,False,1,False),(3,True,4,False),(2,False,0,False)])
def test_fixed_gate(gain,regression,actions,passed):
    assert evaluation_gate(suites(gain,regression,actions))['passed'] is passed

def test_preservation_losses_and_missing_suite_fail():
    value=suites();value['challenge_preservation']['top5_losses']=['loss']
    assert not evaluation_gate(value)['passed']
    with pytest.raises(ValueError):evaluation_gate({})

def test_summary_counts_actual_choices():
    result=evaluate_case(Ranker(),row(),'natural_evaluation_v3')
    report=summarize([result])
    assert report['corrections']==['one'] and report['regressions']==[]
    assert report['top1']==report['accepted_actions']==1
