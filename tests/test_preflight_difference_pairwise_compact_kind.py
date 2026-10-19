import pytest

from scripts.preflight_difference_pairwise_compact_kind import (
    CODEBOOK, canonical, decode_payload, encode_payload,
)


def semantic():
    return {'roman_input': 'imnida testreul', 'external_context': {'text': '문맥'},
            'raw_mode': False, 'mapping': {'1': 'KEEP', '2': 'SWITCH'},
            'CURRENT': {'text': '임니다 test를', 'score': 1.5, 'rank': 1,
                        'kinds': list(CODEBOOK.values())},
            'ALTERNATIVE': {'text': '입니다 test를', 'score': 0, 'rank': 2,
                            'kinds': ['syllable', *list(CODEBOOK.values())[1:6], 'lexicon']},
            'changed_spans': [{'roman': 'imnida', 'current_kind': 'syllable',
                               'alternative_kind': 'morphology'}]}


def test_all_codes_boundary_edits_and_exact_semantics():
    original = semantic()
    compact = encode_payload(original)
    assert compact['K'] == {'c': 'LXMNDRS', 'a': '0=S,6=L'}
    assert canonical(decode_payload(compact)) == canonical(original)
    assert original == semantic()


def test_empty_edits_and_long_alternating_sequence():
    original = semantic()
    original['CURRENT']['kinds'] = ['lexicon', 'raw'] * 80
    original['ALTERNATIVE']['kinds'] = original['CURRENT']['kinds'].copy()
    compact = encode_payload(original)
    assert compact['K'] == {'c': 'LR' * 80, 'a': ''}
    assert decode_payload(compact) == original


@pytest.mark.parametrize('edits', ['-1=N', '+1=N', '01=N', '1.0=N', '7=N', '0=N,0=S',
                                  '6=L,0=S', '=N', '0N', '0=N,', ',0=N', ' 0=N',
                                  '0 =N', '0= N', '0=n', '0=Z', '0=한', '0=L'])
def test_reject_malformed_edits(edits):
    compact = encode_payload(semantic())
    compact['K']['a'] = edits
    with pytest.raises(ValueError):
        decode_payload(compact)


@pytest.mark.parametrize('codes', ['', 'lX', 'LZ', 'L한', 'L R', 7, None])
def test_reject_malformed_codes(codes):
    compact = encode_payload(semantic())
    compact['K']['c'] = codes
    with pytest.raises(ValueError):
        decode_payload(compact)


@pytest.mark.parametrize('mutation', ['collision', 'empty', 'unequal', 'unknown'])
def test_reject_invalid_semantic_input(mutation):
    original = semantic()
    if mutation == 'collision':
        original['K'] = {}
    elif mutation == 'empty':
        original['CURRENT']['kinds'] = original['ALTERNATIVE']['kinds'] = []
    elif mutation == 'unequal':
        original['CURRENT']['kinds'].pop()
    else:
        original['CURRENT']['kinds'][0] = 'unknown'
    with pytest.raises(ValueError):
        encode_payload(original)


@pytest.mark.parametrize('mutation', ['omitted_edit', 'changed_code', 'text', 'context', 'score', 'span'])
def test_mutations_cannot_equal_original_semantics(mutation):
    original = semantic()
    compact = encode_payload(original)
    if mutation == 'omitted_edit':
        compact['K']['a'] = '6=L'
    elif mutation == 'changed_code':
        compact['K']['a'] = '0=N,6=L'
    elif mutation == 'text':
        compact['CURRENT']['text'] += 'x'
    elif mutation == 'context':
        compact['external_context']['text'] += 'x'
    elif mutation == 'score':
        compact['CURRENT']['score'] += 1
    else:
        compact['changed_spans'][0]['roman'] += 'x'
    assert canonical(decode_payload(compact)) != canonical(original)
