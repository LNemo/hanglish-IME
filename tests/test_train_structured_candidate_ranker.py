from scripts.train_structured_candidate_ranker import evaluate, parent_prediction


def row():
    return dict(id='a', kind='source', source='mok', candidates=['목', '몫'],
                base_scores=[6., 6.], teacher_scores=[1., 2.], expected=['몫'])


def test_parent_prediction_matches_adjusted_score():
    result = parent_prediction(row())
    assert result['correct'] and result['result'] == '몫'


def test_unscorable_and_unaligned_rows_stay_in_denominator():
    class Model:
        def eval(self):
            pass

    class Ranker:
        model = Model()

        def can_score(self, row):
            return False

        def score_group(self, row):
            raise AssertionError('unscorable row must not enter model')

    values = [row(), dict(id='b', kind='fixed', fixed_result=None, expected=['몫'])]
    actual = evaluate(Ranker(), {'held': values}, lambda: None)['held']
    assert len(actual) == 2 and all(r['fallback'] for r in actual)
    assert actual[0]['correct'] and actual[1]['correct'] is False
