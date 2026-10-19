import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_sentence_challenge import OUTPUT, build, held_partition, natural_eligible, verify_splits
from scripts.sentence_data import SentenceExample, encoder_inputs, load_training_data, model_input, read_split


def test_complete_sentence_contract_and_disjoint_splits():
    splits = {s:read_split(OUTPUT, s, allow_test=s=='test') for s in ('train','dev','test')}
    assert {s:len(rows) for s,rows in splits.items()} == {'train':5656,'dev':316,'test':1000}
    assert verify_splits(splits)['external_context_rows'] == 0
    for split, rows in splits.items():
        natural = [r for r in rows if r['category']=='natural_sentence_context']
        assert len({r['group_id'] for r in natural}) == len(natural)
        for row in natural:
            assert len(row['input'].split()) >= 4
            assert 'left_context' not in row
            if split != 'train':
                assert held_partition(row['group_id']) == split


def test_target_and_metadata_do_not_enter_encoder_input():
    row = read_split(OUTPUT,'train')[0]
    changed = copy.deepcopy(row)
    changed['expected'] = ['전혀 다른 정답입니다']
    changed['provenance'] = '한국어 설명을 입력하면 안 됩니다'
    assert model_input(row) == model_input(changed)
    examples = [SentenceExample('x',row['input'],('다른 정답',))]
    assert encoder_inputs(examples) == [row['input']]
    for key,value in [('context','한국어 힌트'),('left_context','앞 한국어 문장'),('context_role','semantic_hint')]:
        invalid=dict(row,**{key:value})
        with pytest.raises(ValueError):model_input(invalid)


def test_training_loader_never_opens_test(monkeypatch):
    original=Path.read_bytes
    def guarded(path):
        if path.name=='test.jsonl':raise AssertionError('test leakage')
        return original(path)
    monkeypatch.setattr(Path,'read_bytes',guarded)
    loaded=load_training_data()
    assert set(loaded)=={'train','dev'}
    with pytest.raises(ValueError):read_split(OUTPUT,'test')


def test_training_lexicon_has_only_training_labels():
    from collections import Counter
    import re
    rows=read_split(OUTPUT,'train')
    expected=Counter(w for r in rows for t in r['expected'] for w in re.findall('[가-힣]+',t))
    lex=json.loads((OUTPUT/'train_lexicon.json').read_text())
    assert lex['entries']==[dict(text=w,count=n) for w,n in sorted(expected.items())]
    assert lex['train_sha256']==hashlib.sha256((OUTPUT/'train.jsonl').read_bytes()).hexdigest()


def test_legacy_sentences_preserved_and_old_hints_absent():
    parent=OUTPUT.parent/'v2.2'
    for split in ('train','dev','test'):
        before={r['id']:r for r in map(json.loads,(parent/(split+'.jsonl')).read_text().splitlines())}
        rows=read_split(OUTPUT,split,allow_test=split=='test')
        retained=[r for r in rows if 'legacy_id' in r]
        assert {r['legacy_id'] for r in retained}=={id for id,r in before.items() if not r['context'] and len(r['input'].split())>=2}
        for row in retained:
            original=before[row['legacy_id']]
            assert (row['input'],row['expected'])==(original['input'],original['expected'])


def test_deterministic_rebuild_and_no_overwrite(tmp_path):
    destination=tmp_path/'corpus'
    build(destination)
    for filename in ('train.jsonl','dev.jsonl','test.jsonl','train_lexicon.json','manifest.json'):
        assert (OUTPUT/filename).read_bytes()==(destination/filename).read_bytes()
    with pytest.raises(ValueError):build(destination)


def test_cross_document_leak_is_rejected():
    rows={s:read_split(OUTPUT,s,allow_test=s=='test') for s in ('train','dev','test')}
    rows['test'][-1]['source_ids']=list(rows['train'][-1]['source_ids'])
    with pytest.raises(ValueError,match='leakage'):verify_splits(rows)


def test_heading_fragments_and_known_source_typos_are_filtered():
    assert not natural_eligible({'text':'○성장성은 모바일, 규모는 인터넷온라인 시장 매출은 많이 늘었다.'}, [])
    assert not natural_eligible({'text':'직원은 뒤짚고 놓은 서류를 다시 확인하고 자리를 떠났다.'}, ['뒤짚고'])
    assert natural_eligible({'text':'직원은 회사에 있는 서류를 다시 확인하고 자리를 떠났다.'}, [])
