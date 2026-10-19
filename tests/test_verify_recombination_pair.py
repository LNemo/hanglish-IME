import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'pair_verifier', ROOT / '.omx/artifacts/verify-recombination-pair-v1.py')
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def source_and_scores():
    row = {'id': 'fit-1', 'input': 'raw source', 'component_id': 'component',
           'split': 'fit', 'expected': ['가'],
           'original_candidates': ['가', '나', 'raw source'],
           'recombined_candidates': ['가', '나', 'raw source', '다'],
           'novel_candidates': ['다']}
    scored = {k: row[k] for k in ('id', 'input', 'component_id', 'split')}
    scored.update(score_texts=['가', '나', '다'], scores=[0., 1., 2.],
                  original={'applied': True}, recombined={'applied': True})
    return row, {row['id']: scored}


def test_selection_uses_actual_highest_negative_and_preserves_all_positives():
    row, scored = source_and_scores()
    result = verifier.expected_selection([row], scored)
    actual = result['rows'][0]
    assert actual['arms']['a_original']['candidates'] == ['가', '나']
    assert actual['arms']['b_union']['candidates'] == ['가', '다', '나']
    assert actual['selected_novel_negatives'] == ['다']
    assert result['selected_novel_negative_count'] == 1


def test_selection_excludes_both_arms_when_union_positive_cap_is_full():
    row, scored = source_and_scores()
    positives = [f'정답{i}' for i in range(8)]
    row.update(expected=positives, original_candidates=[positives[0], '오답'],
               recombined_candidates=[*positives, '오답'], novel_candidates=positives[1:])
    scored[row['id']].update(score_texts=[*positives, '오답'], scores=list(range(9)))
    result = verifier.expected_selection([row], scored)
    assert result['rows'] == []
    assert result['excluded'] == {'positive_cap_leaves_no_negative': 1}


def test_changed_parameter_scope_rejects_frozen_encoder_and_empty_changes():
    verifier.verify_changed_names(['electra.encoder.layer.10.attention.weight', 'classifier.weight'])
    for names in ([], ['electra.encoder.layer.9.attention.weight'], ['electra.embeddings.weight']):
        with pytest.raises(ValueError, match='top two'):
            verifier.verify_changed_names(names)


def test_literal_signature_retains_order_spaces_and_new_characters():
    assert verifier.literal('A1 B!') != verifier.literal('1A B!')
    assert verifier.literal('test') != verifier.literal('테스트!')
    assert verifier.literal('ABC 123!') == verifier.literal('ABC 테스트123!')


def test_metric_replay_keeps_uncovered_and_raw_rows_in_denominator():
    predictions = [
        {'id': 'one', 'input': 'raw', 'b_union': {'original': {
            'result': '오답', 'candidates': ['오답', 'raw']}}},
        {'id': 'two', 'input': 'URL!', 'b_union': {'original': {
            'result': 'URL!', 'candidates': ['URL!']}}},
    ]
    result = verifier.summarize(predictions, {'one': ['정답'], 'two': ['URL!']}, 'b_union', 'original')
    assert result['counts']['rows'] == 2
    assert result['counts']['top1'] == result['counts']['top5'] == 1


def test_changes_rejects_duplicate_predictions():
    outcome = {'id': 'one', 'top1': True, 'top5': True, 'literal_signature_preserved': True}
    with pytest.raises(ValueError, match='identities'):
        verifier.changes({'outcomes': [outcome, outcome]}, {'outcomes': [outcome]})
