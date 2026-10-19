import copy

import pytest
import torch

from scripts.verify_contextual_character_dev import reference_base, reference_public, replay_row


CALIBRATION = {'parameters':[.8,.3,-.1,.5,.2], 'scale':[.2,.4,.5,.8]}


def candidate(index):
    text = chr(0xac00+index)
    return {'text':text,'segments':[{'input_start':0,'input_end':1,'source':'x','text':text}],
            'sentence_score':index/10,'features':[.4,.3,.2,.1],'status':'scored'}


def test_reference_formula_and_public_raw_slot():
    c = candidate(1)
    assert reference_base(c,CALIBRATION) == pytest.approx(.8*.1+.3*2-.1*.75+.5*.4+.2*.125)
    rows = [candidate(i) for i in range(7)]
    result = reference_public('x',rows,list(range(7)))
    assert [c['text'] for c in result] == [rows[i]['text'] for i in (6,5,4,3)]+['x']
    assert result[-1]['score'] is None and result[-1]['segments'][0]['kind']=='raw'


def test_expanded_once_then_exact_old_prefix_and_independent_render():
    class FakeModel:
        def __init__(self): self.sizes = []
        def prepare(self,source,candidates,base):
            assert source == 'x'
            self.sizes.append(len(candidates))
            return torch.tensor(base,dtype=torch.float64)
        def __call__(self,prepared): return {'scores':prepared+.125}
    pool = [candidate(i) for i in range(40)]
    base = [reference_base(c,CALIBRATION) for c in pool]
    parent = {'request':{'input':'x','context':''},'scoring':{
        'ranked':True,'original':pool[:12],'novel':pool[12:],'fallback_public':None},
        'predictions':{'old_calibrated':reference_public('x',pool[:12],base[:12]),
                       'expanded_calibrated':reference_public('x',pool,base)}}
    snapshot = copy.deepcopy(parent)
    model = FakeModel()
    scores,predictions = replay_row(parent,model,CALIBRATION)
    assert model.sizes == [32,8]
    assert scores['old_contextual'] == scores['expanded_contextual'][:12]
    assert predictions['old_contextual'] == reference_public('x',pool[:12],scores['old_contextual'])
    assert parent == snapshot


def test_no_pool_keeps_fallback_without_inference():
    fallback = [{'text':'x','score':None}]
    parent = {'scoring':{'ranked':False,'original':[],'novel':[],'fallback_public':fallback}}
    scores,predictions = replay_row(parent,None,CALIBRATION)
    assert scores == {'old_contextual':[],'expanded_contextual':[]}
    assert predictions['old_contextual'] == fallback
    assert predictions['old_contextual'] is not fallback


def test_real_contextual_model_producer_matches_independent_replay():
    from hanglish.models.contextual_character_residual import ContextualCharacterResidualRanker
    from scripts.capture_contextual_character_dev import predict_row
    torch.manual_seed(41)
    model = ContextualCharacterResidualRanker(byte_dim=4,hidden=5,head_hidden=6).eval()
    with torch.no_grad():
        model.edge_head[-1].weight.normal_(0,.15)
    pool = [candidate(i) for i in range(40)]
    for c in pool[:12]: c['status'] = 'preserved'
    base = [reference_base(c,CALIBRATION) for c in pool]
    request = {'input':'x','context':''}
    parent = {'id':'synthetic','dataset_split':'dev','source_row_index':0,'request':request,
              'scoring':{'id':'synthetic','request':request,'ranked':True,'original':pool[:12],
                         'novel':pool[12:],'fallback_public':None},
              'predictions':{'old_calibrated':reference_public('x',pool[:12],base[:12]),
                             'expanded_calibrated':reference_public('x',pool,base)}}
    with torch.no_grad(): produced = predict_row(parent,model,CALIBRATION,0)
    scores,predictions = replay_row(parent,model,CALIBRATION)
    assert produced['contextual_scores'] == scores
    assert produced['predictions'] == predictions
