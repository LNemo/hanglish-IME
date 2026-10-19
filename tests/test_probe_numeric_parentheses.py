from scripts.probe_numeric_parentheses import summarize


def test_full_sentence_counts_include_losses_and_absent_gold():
    def response(*texts):
        return {'candidates': [{'text': text} for text in texts]}
    captures = [
        {'id': 'a', 'suite': 'train', 'parent': response('raw'),
         'experiment': response('정답', 'raw')},
        {'id': 'b', 'suite': 'train', 'parent': response('정답', 'raw'),
         'experiment': response('오답', 'raw')},
        {'id': 'c', 'suite': 'dev', 'parent': response('raw'),
         'experiment': response('raw')},
    ]
    targets = {(row['suite'], row['id']): ['정답'] for row in captures}
    counts, changes = summarize(captures, targets)
    assert counts['train']['rows'] == 2
    assert counts['train']['fixes'] == counts['train']['losses'] == 1
    assert counts['train']['top5_losses'] == 1
    assert counts['dev']['rows'] == 1
    assert counts['dev']['experiment_top1'] == counts['dev']['experiment_top5'] == 0
    assert [row['id'] for row in changes] == ['a', 'b']
