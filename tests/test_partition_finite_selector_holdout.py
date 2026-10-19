from copy import deepcopy

import pytest

from scripts.partition_finite_selector_holdout import select_ids

def inputs():
    metadata=[{'id':str(i),'group_id':str(i//2)} for i in range(8)]
    natural={'reserve_ids':[str(i) for i in range(2,8)], 'reserve_group_ids':['1','2','3'],
             'fit_ids':['0','1'],'calibration_ids':[],'holdout_ids':[],
             'excluded_historical_group_sibling_ids':[], 'excluded_historical_group_ids':['0'],
             'calibration_group_ids':[],'holdout_group_ids':[]}
    previous={'eval_ids':['2','3'],'eval_group_ids':['1'],
              'remaining_ids':['4','5','6','7'],'remaining_group_ids':['2','3']}
    return metadata,natural,previous

def test_whole_groups_and_source_order_without_reading_targets():
    data,natural,previous=inputs()
    result=select_ids(data,natural,previous,target=3)
    assert len(result['eval_ids'])==4
    assert set(result['eval_ids'])=={'4','5','6','7'}
    assert result['eval_ids'][0:2] in [['4','5'],['6','7']]
    assert result['remaining_ids']==[]
    for row in data:row['expected']=object()
    assert select_ids(data,natural,previous,target=3)==result

def test_partition_is_deterministic_and_preserves_inputs():
    args=inputs(); before=deepcopy(args)
    result=select_ids(*args,target=1)
    assert len(result['eval_ids'])==2
    assert len(result['remaining_ids'])==2
    assert not set(result['eval_ids']) & set(result['remaining_ids'])
    assert result==select_ids(*args,target=1)
    assert args==before

@pytest.mark.parametrize('case',['duplicate','partial','exposed','missing','historical','insufficient'])
def test_invalid_reserve_fails(case):
    data,natural,previous=inputs()
    if case=='duplicate':data.append(data[-1])
    elif case=='partial':previous['remaining_ids'].remove('7');natural['reserve_ids'].remove('7')
    elif case=='exposed':previous['remaining_ids'].append('2')
    elif case=='missing':data.pop()
    elif case=='historical':natural['fit_ids'].append('4')
    with pytest.raises(ValueError):
        select_ids(data,natural,previous,target=5 if case=='insufficient' else 1)
