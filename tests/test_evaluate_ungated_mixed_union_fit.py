from scripts.evaluate_ungated_mixed_union_fit import edit_distance, metrics, transitions


def test_unicode_edit_distance_counts_whitespace_and_literal_changes():
    assert edit_distance('입니다', '임니다') == 1
    assert edit_distance('test를 시행', 'test 시행') == 1
    assert edit_distance('', '한글') == 2


def test_multiple_answers_and_unsupported_gold():
    value = metrics('imnida', [{'text': '입니다'}], [{'text': '입니다'}], [], ['임니다', '입니다'])
    assert value['first'] and value['top5'] and value['all_pool_covered']
    assert not value['scorable_pool_covered'] and value['edit_distance'] == 0


def test_partial_regression_is_visible_when_both_sentences_wrong():
    before = metrics('testreul', [{'text': 'test릉'}], [], [], ['test를'])
    after = metrics('testreul', [{'text': '테스릉'}], [], [], ['test를'])
    delta = transitions(before, after)
    assert not delta['regressions'] and delta['partial_regressions'] == 1
    assert delta['ascii_regressions'] == 1 and delta['edit_distance_delta'] > 0
