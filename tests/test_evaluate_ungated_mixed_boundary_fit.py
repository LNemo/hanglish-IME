from scripts.evaluate_ungated_mixed_boundary_fit import coverage


def test_existing_full_pool_gold_is_not_counted_as_new():
    result = coverage(True, [{'text':'3위', 'token_support':{'supported':True}}], ['3위'])
    assert result['union_covered'] and result['proposal_gold_present']
    assert not result['new_gold_row'] and not result['new_supported_gold_row']


def test_unsupported_gold_is_recovered_but_not_claimed_scorable():
    result = coverage(False, [{'text':'3위', 'token_support':{'supported':False}}], ['3위'])
    assert result['new_gold_row'] and not result['new_supported_gold_row']


def test_no_proposal_keeps_full_pool_baseline():
    assert coverage(True, [], ['3위'])['union_covered']
    assert not coverage(False, [], ['3위'])['union_covered']
