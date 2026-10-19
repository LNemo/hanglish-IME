from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from scripts.current_recombination_selection import compare_row


def module(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


ROOT=Path(__file__).resolve().parents[1]
fixtures=module(Path(__file__).with_name('test_current_recombination_selection.py'),'selection_fixtures')
replay=module(ROOT/'.omx/scripts/replay_recombination_selection_values_v1.py','selection_replay')


def test_independent_numpy_replay_matches_scores_choices_and_transitions():
    scoring=fixtures.scored()
    for row in scoring['original']+scoring['novel'][:1]:
        row['features']=[row['features'][0],-.12,.31,-.05]
    model=dict(parameters=[.85,.69,.57,-.56,-.10],scale=[.13,.08,.04,.008])
    expected=['novel-gold']
    calculated=compare_row(scoring,expected,model)
    assert replay.verify_comparison(scoring,expected,model,calculated)==calculated


@pytest.mark.parametrize('field',['public','first','counts'])
def test_independent_replay_rejects_changed_candidate_correctness_or_denominator(field):
    scoring=fixtures.scored();model=fixtures.model();expected=['novel-gold']
    record=deepcopy(compare_row(scoring,expected,model))
    if field=='public':record['arms']['expanded_sentence']['public'][0]['text']='foreign'
    elif field=='first':record['arms']['expanded_sentence']['first']=False
    else:record['counts']['unsupported_novel']=0
    with pytest.raises(ValueError,match='independent selection differs'):
        replay.verify_comparison(scoring,expected,model,record)


def test_unranked_fallback_remains_correct_but_is_not_scored_pool_coverage():
    scoring=fixtures.scored();scoring.update(ranked=False,original=[],novel=[],
        fallback_public=[dict(text='fallback',score=None,segments=[])])
    model=fixtures.model();expected=['fallback']
    record=compare_row(scoring,expected,model)
    assert replay.verify_comparison(scoring,expected,model,record)==record
