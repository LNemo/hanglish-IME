from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts.audit_copy_boundaries_fit import project_candidate
from scripts.verify_copy_boundaries_fit import expected_children


class Generator:
    english_suffixes=[('reul','를')]
    def options(self,source):
        return [SimpleNamespace(text='widget를',kind='mixed')]


def test_full_candidate_refinement_shifts_unicode_offsets_and_keeps_scores():
    source='😀 widgetreul'
    row={'text':'😀 widget를','sentence_score':1.25,'features':[0.1,0.2],
         'segments':[dict(input_start=0,input_end=2,source='😀 ',text='😀 ',kind='raw'),
                     dict(input_start=2,input_end=len(source),source='widgetreul',text='widget를',kind='mixed')]}
    before=deepcopy(row);inventory={}
    children,keys,counts=project_candidate(source,row,inventory,Generator())
    assert row==before
    assert children[1]==dict(input_start=2,input_end=8,source='widget',text='widget',kind='raw')
    assert children[2]==dict(input_start=8,input_end=12,source='reul',text='를',kind='mixed')
    assert counts['copy_after']==counts['copy_before']+1
    assert counts['mixed_after']==0 and counts['refined_candidates']==1
    assert keys==[('widgetreul','widget를','mixed')]
    assert inventory[keys[0]]['segments']==expected_children(keys[0],Generator())


def test_unknown_interior_ascii_is_not_invented_as_copy_boundary():
    row={'text':'w이dget를','sentence_score':3.0,
         'segments':[dict(input_start=0,input_end=10,source='widgetreul',text='w이dget를',kind='mixed')]}
    children,_,counts=project_candidate('widgetreul',row,{},Generator())
    assert children==row['segments'] and counts['refined_candidates']==0
    assert counts['mixed_after']==1


def test_invalid_candidate_coverage_rejected_before_derivation():
    row={'text':'widget를','segments':[dict(input_start=1,input_end=10,source='widgetreul',text='widget를',kind='mixed')]}
    with pytest.raises(ValueError,match='coverage'):
        project_candidate('widgetreul',row,{},Generator())


def test_independent_matcher_preserves_uppercase_suffix_source():
    actual=expected_children(('widgetREUL','widget를','mixed'),Generator())
    assert actual[1]['source']=='REUL' and actual[1]['text']=='를'


def test_refined_children_are_consumed_by_existing_character_model():
    from hanglish.models.character_edge_residual import CharacterEdgeResidualRanker
    row={'text':'widget를','segments':[dict(input_start=0,input_end=10,
        source='widgetreul',text='widget를',kind='mixed')]}
    segments,_,_=project_candidate('widgetreul',row,{},Generator())
    model=CharacterEdgeResidualRanker()
    prepared=model.prepare('widgetreul',[dict(text='widget를',segments=segments)],[1.25])
    assert prepared.edge_ids==((0,6,'widget'),(6,10,'를'))
    assert model(prepared)['scores'].tolist()==[1.25]
