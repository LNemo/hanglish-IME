import copy

import pytest

from scripts.evaluate_roberta_context_ranker import DEV_COUNTS, RUNTIME_VERSION, require_development_gate


def clean_gate():
    return {'passed': True, 'checkpoint_hashes': {'weights': 'w'},
            'training_report_sha256': 't', 'source_hashes': {'source': 's'},
            'parent_version': RUNTIME_VERSION, 'evaluation_code_hashes': {'main': 'm', 'aggregation': 'a'},
            'scope': 'actual hybrid development',
            'suites': {name: {'rows': count, 'regressions': [], 'top1': count, 'before_top1': count-1}
                       for name, count in DEV_COUNTS.items()}}


def check(gate):
    require_development_gate(gate, {'weights': 'w'}, 't', {'source': 's'}, {'main': 'm', 'aggregation': 'a'})


def test_only_complete_improved_development_authorizes_final():
    check(clean_gate())
    for field, value in (('scope', 'exposed final regression'), ('passed', False),
                         ('evaluation_code_hashes', {'main': 'm', 'aggregation': 'changed'}),
                         ('source_hashes', {'source': 'changed'})):
        gate = clean_gate()
        gate[field] = value
        with pytest.raises(ValueError):
            check(gate)


@pytest.mark.parametrize('mutation', ['missing_suite', 'wrong_count', 'regression', 'no_improvement'])
def test_summary_flag_cannot_hide_missing_or_failed_development(mutation):
    gate = copy.deepcopy(clean_gate())
    if mutation == 'missing_suite':
        del gate['suites']['reasoning_dev']
    elif mutation == 'wrong_count':
        gate['suites']['original_dev']['rows'] = 139
    elif mutation == 'regression':
        gate['suites']['selection_dev']['regressions'] = ['case']
    else:
        gate['suites']['natural_dev']['top1'] = gate['suites']['natural_dev']['before_top1']
    with pytest.raises(ValueError):
        check(gate)
