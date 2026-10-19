from copy import deepcopy

import pytest

from scripts.evaluate_phonetic_fallback_union import DEV_COUNTS, RUNTIME_VERSION, require_final_gate


def gate():
    suites={name:{'rows':count,'regressions':[],'top1':count,'before_top1':count}
            for name,count in DEV_COUNTS.items()}
    suites['natural_dev'].update(top1=72,before_top1=71)
    return {'scope':'fresh paired development','passed':True,'source_hashes':{'code':'hash'},
            'probe_report_sha256':'probe','parent_version':RUNTIME_VERSION,'suites':suites,
            'test_data_read':False,'deployed':False}


def test_complete_current_gain_allows_final():
    require_final_gate(gate(),{'code':'hash'},'probe')


@pytest.mark.parametrize('defect',['regression','partial','no_gain','missing_suite','old_code','old_probe','final_read'])
def test_final_requires_full_current_regression_free_evidence(defect):
    item=deepcopy(gate())
    if defect=='regression':item['suites']['selection_dev']['regressions']=['id']
    elif defect=='partial':item['suites']['original_dev']['rows']=139
    elif defect=='no_gain':item['suites']['natural_dev']['top1']=71
    elif defect=='missing_suite':del item['suites']['reasoning_dev']
    elif defect=='old_code':item['source_hashes']={'code':'old'}
    elif defect=='old_probe':item['probe_report_sha256']='old'
    else:item['test_data_read']=True
    with pytest.raises(ValueError,match='current complete'):
        require_final_gate(item,{'code':'hash'},'probe')
