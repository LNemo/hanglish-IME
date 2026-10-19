from scripts.build_runtime_collision_contexts import (
    SPLIT_SALT, document_split, inventory, mine, select_training, selector_held, stable_hash,
)
import hashlib
import json
import pytest
from scripts import build_runtime_collision_contexts as builder


def group(split='train'):
    return next(str(i) for i in range(1000) if document_split(str(i))==split
                and not selector_held(str(i))
                and int(stable_hash(SPLIT_SALT,str(i))[:16],16)%10 != 0)


def test_inventory_keeps_unobserved_whole_forms_and_rejects_nonmatching_alias():
    families = inventory([{'inneun': {'있는': None, '잇는': None}, 'hello': {'있는': None}},
                          {'inneun': {'잊는': None, 'abc': None}, 'bit': {'빛': None}}])
    assert families == {'inneun': ['잇는','있는','잊는']}


def test_mining_requires_whole_word_and_preserves_source_offsets():
    row={'text':'있는 사람과 잇는 사람은 달랐다.', 'left_context':'', 'group_id':group(), 'source_ids':['new']}
    families={'inneun':['있는','잇는','잊는'], 'it':['잇','있']}
    found,_=mine([row],[],families)
    assert len(found)==2
    for item in found:
        assert item['text'][item['word_start']:item['word_end']]==item['word']
        assert item['word_input']=='inneun'
        assert item['split']=='train'
        assert not item['sense_annotation']
    assert not mine([row],[{**row,'group_id':'other'}],families)[0]


def test_old_embedded_text_and_original_dev_are_excluded():
    row={'text':'앞에 있는 사람은 안전하다.', 'left_context':'', 'group_id':group(), 'source_ids':['new']}
    frozen={'text':'있는 사람은 안전하다.', 'group_id':'old', 'source_ids':['old']}
    assert not mine([row],[frozen],{'inneun':['있는','잇는']})[0]
    dev=next(str(i) for i in range(1000) if int(stable_hash(SPLIT_SALT,str(i))[:16],16)%10==0)
    assert not mine([{**row,'group_id':dev}],[],{'inneun':['있는','잇는']})[0]


def test_reuses_hold_document_split():
    row={'text':'남아 있는 물건을 확인한다.', 'left_context':'', 'group_id':group('hold'), 'source_ids':['new']}
    found,_=mine([row],[],{'inneun':['있는','잇는']})
    assert found and all(r['split']=='hold' for r in found)
    assert not select_training(found)


def test_previous_selector_hold_partition_and_embedded_hold_context_are_excluded():
    held=next(str(i) for i in range(1000) if document_split(str(i))=='train' and selector_held(str(i))
              and int(stable_hash(SPLIT_SALT,str(i))[:16],16)%10 != 0)
    row={'text':'남아 있는 물건을 확인한다.', 'left_context':'', 'group_id':held, 'source_ids':['new']}
    assert not mine([row],[],{'inneun':['있는','잇는']})[0]
    old={'text':'다른 문장입니다.', 'left_context':'남아 있는 물건을 확인한다.'}
    assert not mine([{**row,'group_id':group()}],[],{'inneun':['있는','잇는']},[old])[0]


def test_balanced_selection_caps_deduplicates_and_excludes_hold_only_contrast():
    rows=[]
    for family in ('a','b'):
        for word,count in [('common',20),('rare',2)]:
            rows.extend({'id':f'{family}-{word}-{i}', 'word_input':family,'word':word,
                         'group_id':f'{family}-{word}-{i//2}', 'split':'train'} for i in range(count))
    rows += [{'id':'single','word_input':'c','word':'one','group_id':'x','split':'train'},
             {'id':'held','word_input':'c','word':'two','group_id':'y','split':'hold'}]
    selected=select_training(rows,limit=8,per_spelling=3,per_family=4)
    assert selected==select_training(list(reversed(rows)),limit=8,per_spelling=3,per_family=4)
    assert len(selected)==8
    for family in ('a','b'):
        subset=[r for r in selected if r['word_input']==family]
        assert len(subset)==4
        assert len({r['group_id'] for r in subset})==4
        assert {r['word'] for r in subset}=={'common','rare'}
    assert all(r['split']=='train' and r['word_input']!='c' for r in selected)


def test_previous_hold_source_cannot_reappear_as_train_under_new_group(tmp_path,monkeypatch):
    old={'group_id':group('hold'),'source_ids':['shared-source'],'split':'hold'}
    (tmp_path/'train.jsonl').write_text('')
    (tmp_path/'hold.jsonl').write_text(json.dumps(old)+'\n')
    manifest={'outputs':{role:{'file':role+'.jsonl','sha256':hashlib.sha256((tmp_path/(role+'.jsonl')).read_bytes()).hexdigest()}
                         for role in ('train','hold')}}
    path=tmp_path/'manifest.json';path.write_text(json.dumps(manifest))
    monkeypatch.setattr(builder,'PREVIOUS',tmp_path)
    monkeypatch.setattr(builder,'PREVIOUS_MANIFEST_SHA',hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError,match='crosses previous split'):
        builder.validate_previous_split([{'group_id':group(),'source_ids':['shared-source'],'split':'train'}])
