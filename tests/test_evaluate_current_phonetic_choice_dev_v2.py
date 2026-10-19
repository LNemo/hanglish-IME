import pytest

from scripts.evaluate_current_phonetic_choice_dev_v2 import current_reference


def proof(proposed=55, verified=True):
    return {'verified': verified, 'comparisons': {
        branch: {'english_ascii_signature': {
            'rows_with_reference_ascii_words': 59,
            'parent_exact_signature': 54,
            'proposed_exact_signature': proposed}}
        for branch in ('parent', 'old_calibration')}}


def test_reference_uses_deployed_proposed_count_not_old_parent():
    assert current_reference(proof()) == dict(
        rows=316, top1=267, top5=284, ascii_reference_rows=59, ascii_exact_first=55)


@pytest.mark.parametrize('value', [proof(54), proof(55, False)])
def test_rejects_old_or_unverified_reference(value):
    with pytest.raises(ValueError):
        current_reference(value)
