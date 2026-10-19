from scripts.audit_fit_generation_gate import LENGTH_WARNING, classify


def record(ranked, warning, requests, paths):
    return {'base_paths': paths, 'response': {
        'sentence_ranking_applied': ranked, 'warnings': warning,
        'phonetic_word_union': {'generation_requests': requests, 'generated_extras': []},
    }}


def test_distinguishes_length_rejection_from_other_unranked_cases():
    rejected = classify(record(False, [LENGTH_WARNING], 0, []))
    other = classify(record(False, [], 0, []))
    assert rejected['length_warning'] and not other['length_warning']
    assert rejected['empty_recorded_pool'] and other['empty_recorded_pool']


def test_zero_generation_requests_does_not_imply_length_rejection():
    result = classify(record(True, [], 0, [('입니다', 1, [])]))
    assert result['ranked'] and not result['length_warning']
    assert not result['empty_recorded_pool']
    assert result['generation_requests'] == 0
