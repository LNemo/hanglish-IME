from copy import deepcopy
import pytest
from scripts.evaluate_constrained_runtime import verify_probe, development_passed
from scripts.probe_constrained_runtime import VERSION, POLICY, metrics


def evidence():
    rows = [{'id': str(i), 'input': 'gak', 'context': '', 'expected': ['각']} for i in range(512)]
    cases = []
    for i, row in enumerate(rows):
        case = dict(row)
        for side in ('before', 'new'):
            correct = side == 'new' or i > 0
            choices = ['각'] if correct else ['갂']
            case.update({f'{side}_candidates': choices, f'{side}_pool': choices,
                         f'{side}_top1': correct, f'{side}_top5': correct,
                         f'{side}_covered': correct})
        cases.append(case)
    report = {'source_hashes': {}, 'parent_version': VERSION, 'policy': POLICY,
              'test_data_read': False, 'development_inference_run': False,
              'deployed': False, 'metrics': metrics(cases)}
    return report, cases, rows


def test_gate_requires_fresh_complete_training_gain():
    report, cases, rows = evidence()
    verify_probe(report, cases, rows, {})
    with pytest.raises(ValueError):
        verify_probe(report, cases[:-1], rows, {})
    with pytest.raises(ValueError):
        verify_probe(report, cases, rows, {'new_source': 'changed'})


def test_gate_recomputes_outcomes_and_checks_source_alignment():
    report, cases, rows = evidence()
    cases[0]['new_top1'] = False
    with pytest.raises(ValueError, match='outcomes'):
        verify_probe(report, cases, rows, {})
    report, cases, rows = evidence()
    cases[0]['context'] = 'different'
    with pytest.raises(ValueError, match='alignment'):
        verify_probe(report, cases, rows, {})


@pytest.mark.parametrize('flag', ['test_data_read', 'development_inference_run', 'deployed'])
def test_gate_rejects_wrong_scope(flag):
    report, cases, rows = evidence()
    report[flag] = True
    with pytest.raises(ValueError):
        verify_probe(report, cases, rows, {})


def test_development_gate_requires_all_suites_and_no_regressions():
    from scripts.evaluate_roberta_context_ranker import DEV_COUNTS
    suites = {name: {'rows': count, 'regressions': [], 'before_top1': 0, 'new_top1': 1,
                     'top5_losses': [], 'coverage_losses': [],
                     'new_top5': 1, 'before_top5': 0, 'new_covered': 1, 'before_covered': 0}
              for name, count in DEV_COUNTS.items()}
    assert development_passed(suites)
    regression = deepcopy(suites)
    regression['selection_dev']['regressions'] = ['one']
    assert not development_passed(regression)
    for key in ('top5_losses', 'coverage_losses'):
        loss = deepcopy(suites)
        loss['natural_dev'][key] = ['one']
        assert not development_passed(loss)
    assert not development_passed({'natural_dev': suites['natural_dev']})
    suites['natural_dev']['new_top1'] = 0
    assert not development_passed(suites)
