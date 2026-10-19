import pytest

from scripts.audit_opaque_fit_spans import classify_span, audit_row


def test_whole_words_and_nonlatin_literals_are_not_opaque():
    for value in ['imnida', 'gachi', ' ', '123', '한글']:
        assert classify_span(value) == []


def test_flags_distinguish_protection_from_generic_punctuation():
    assert set(classify_span('https://example.com/a2')) >= {
        'url_shape', 'digits', 'parent_protected_pattern', 'path_or_identifier_shape'}
    assert 'email_shape' in classify_span('a@example.com')
    assert 'backtick_shape' in classify_span('`api`reul')
    assert set(classify_span('brand(3wi)')) >= {'digits', 'punctuation', 'parent_protected_pattern'}
    assert set(classify_span('abcXdef')) == {'uppercase'}
    assert set(classify_span('abc『def』')) == {'punctuation', 'non_ascii'}


def make_row():
    source = '3wi imnida'
    segments = [dict(input_start=0,input_end=3,source='3wi',text='3위',kind='numeric'),
                dict(input_start=3,input_end=4,source=' ',text=' ',kind='raw'),
                dict(input_start=4,input_end=10,source='imnida',text='입니다',kind='lexical')]
    candidate = dict(text='3위 입니다',segments=segments)
    raw = dict(text=source, segments=[dict(input_start=0,input_end=len(source),
                                         source=source,text=source,kind='raw')])
    return dict(id='s3-train-test',request=dict(input=source,context=''),response=dict(
        input=source,sentence_ranking_applied=False,candidates=[candidate,candidate,raw],
        phonetic_word_union={'generation_requests':0}))


def test_deduplicates_offsets_but_retains_each_parent_and_excludes_raw_reserve():
    result = audit_row(make_row())
    assert result['parent_count'] == 2
    assert len(result['opaque_spans']) == 1
    span = result['opaque_spans'][0]
    assert span['source'] == '3wi' and len(span['parents']) == 2
    assert not any(p['source_copied_exactly'] for p in span['parents'])


def test_broken_source_offsets_are_rejected():
    row=make_row(); row['response']['candidates'][0]['segments'][0]['input_end']=2
    with pytest.raises(ValueError,match='source offsets'):audit_row(row)
