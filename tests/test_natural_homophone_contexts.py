from scripts.build_natural_homophone_contexts import (
    Contrasts, SPLIT_SALT, blocked_text_index, embeds_blocked, mine, select_training, stable_hash,
)
from hanglish.core.phonology import romanize_word


def contrast():
    return Contrasts({'bak':['박','밖'],'sut':['숯','숱'],'ot':['옷','옻'],'bit':['빛','빚','빗']},['은','는','이','으로'])


def train_group():
    return next('group-'+str(i) for i in range(100) if int(stable_hash(SPLIT_SALT,'group-'+str(i))[:16],16)%10!=0)


def test_compound_substitution_preserves_forward_input_and_exact_offsets():
    c=contrast()
    roman,candidates,changes=c.token('숯불')
    assert '숯불' in candidates and '숱불' in candidates
    assert all(romanize_word(word)==roman for word in candidates)
    for change in changes:
        assert '숯불'[change['start']:change['end']]==change['anchor']
        assert '숯불'[:change['start']]+change['replacement']+'숯불'[change['end']:]==change['candidate']
    assert c.token('하늘')[1]==('하늘',)


def test_substitution_with_different_pronunciation_is_excluded():
    roman,candidates,_=contrast().token('빛이')
    assert '빗이' not in candidates
    assert all(romanize_word(word)==roman for word in candidates)


def test_embedded_frozen_sentence_is_detected_inside_surrounding_text():
    index=blocked_text_index([{'text':'이 문장은 기존 평가 자료입니다.'}])
    assert embeds_blocked('앞 문장. 이 문장은 기존 평가 자료입니다. 뒤 문장.',index)
    assert not embeds_blocked('서로 다른 문장입니다.',index)


def test_mining_preserves_source_span_and_excludes_existing_document_and_source():
    base={'text':'화로 안에는 숯불이 남아 있었다.', 'left_context':'', 'group_id':train_group(),
          'source_ids':['new-source'], 'provenance':[]}
    rows,counts=mine([base],[],contrast())
    assert rows
    assert all(base['text'][r['word_start']:r['word_end']]==r['word'] for r in rows)
    assert all(r['sense_annotation'] is False for r in rows)
    assert not mine([base],[base],contrast())[0]
    frozen={**base,'group_id':'different-document'}
    assert not mine([base],[frozen],contrast())[0]


def test_selection_caps_all_anchors_and_never_includes_hold_or_duplicate_documents():
    rows=[{'id':str(i),'split':'train','group_id':str(i//2),'spelling_anchors':['박'] if i%3 else ['박','밖'],
           'match_kind':'simple_form' if i%2 else 'compound_substring'} for i in range(20)]
    rows.append({'id':'hold','split':'hold','group_id':'hold','spelling_anchors':['밖'],'match_kind':'simple_form'})
    chosen=select_training(rows,cap=3)
    for anchor in ('박','밖'):
        matching=[r for r in chosen if anchor in r['spelling_anchors']]
        assert len(matching)<=3 and len({r['group_id'] for r in matching})==len(matching)
    assert all(r['split']=='train' for r in chosen)
    assert {r['id'] for r in chosen}=={r['id'] for r in select_training(list(reversed(rows)),cap=3)}
