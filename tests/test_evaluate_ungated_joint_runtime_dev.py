from scripts.evaluate_ungated_joint_runtime_dev import row_metrics, compare_metrics


def test_multiple_acceptable_outputs_and_top5_are_distinct():
    values = row_metrics('gachi', [{'text': '가치'}, {'text': '갗이'}], ['같이', '가치'])
    assert values['first'] and values['top5']
    values = row_metrics('gachi', [{'text': '갗이'}, {'text': '같이'}], ['같이', '가치'])
    assert not values['first'] and values['top5']


def test_english_word_corruption_is_counted_even_when_capitals_remain():
    before = row_metrics('WebRTCreul', [{'text': 'WebRTC를'}], ['WebRTC를'])
    after = row_metrics('WebRTCreul', [{'text': 'W에bRTC를'}], ['WebRTC를'])
    changes = compare_metrics(before, after)
    assert changes['ascii_regressions'] == 1
    assert changes['regressions'] == 1 and changes['top5_losses'] == 1
