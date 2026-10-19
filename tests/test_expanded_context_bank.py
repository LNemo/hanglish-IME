from scripts.prepare_expanded_context_bank import document_components,excluded_closure,select


def record(id,group,sources,word='낫',text=None):
    text=text or '풀을 '+word+' 도구로 벤다'
    a=text.index(word)
    return dict(id=id,group_id=group,source_ids=sources,split='train',source_partition='upstream_train_and_existing_component_train',word=word,
                word_input='nat',word_start=a,word_end=a+len(word),text=text,left_context='')


def test_transitive_document_alias_exclusion():
    raw=[record('a','g1',['s1','s2']),record('b','g2',['s2','s3']),record('c','g3',['s3'])]
    groups,sources=excluded_closure(raw,[dict(group_id='g1',source_ids=[])])
    assert groups=={'g1','g2','g3'} and sources=={'s1','s2','s3'}
    assert len(set(document_components(raw).values()))==1


def test_bank_masks_observed_word_and_excludes_quarantine():
    raw=[record('a','g1',['s1']),record('b','g2',['s2'],'낯','얼굴과 낯 이야기')]
    bank,counts,forms,_,_=select(raw,[dict(candidates=['낫','낯'])],[],{'낯'})
    assert len(bank)==1 and bank[0]['expected']==['낫']
    assert bank[0]['context']=='\n풀을  [MASK]  도구로 벤다'
    assert counts['quarantined_form']==1


def test_vocabulary_not_derived_from_exclusion_labels():
    raw=[record('a','g1',['s1']),record('b','g2',['s2'],'낯','얼굴과 낯 이야기')]
    held=[dict(group_id='hold',source_ids=[],expected=['낯'])]
    bank,_,_,_,_=select(raw,[dict(candidates=['낫'])],held,set())
    assert [r['expected'] for r in bank]==[['낫']]


def test_duplicate_context_and_per_word_cap():
    raw=[record(str(i),'g'+str(i),['s'+str(i)]) for i in range(5)]
    bank,counts,forms,_,_=select(raw,[dict(candidates=['낫'])],[],set(),cap=24)
    assert len(bank)==1 and counts['duplicate_context_per_word']==4
