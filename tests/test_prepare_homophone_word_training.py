from copy import deepcopy
import pytest
from scripts.prepare_homophone_word_training import assemble, partition
from scripts.probe_natural_homophone_contexts import project


def row(identity, document, candidates=('박','밖')):
    target=dict(id=identity,group_id=document,split='train',text='밖',left_context='풍경',
                word_start=0,word_end=1,word='밖',word_input='bak',source_ids=[document],
                spelling_anchors=['밖'],match_kind='simple_form')
    capture={**project(target),'groups':[dict(source='bak',start=0,end=3,context='풍경 [MASK]',
        candidates=list(candidates),base_scores=[7.]*len(candidates),
        teacher_scores=[0.]*len(candidates),kinds=['lexicon']*len(candidates))]}
    return target,capture


def documents():
    # Obtain each role; the tested property is separation/denominator behavior.
    return {role:next(str(i) for i in range(100) if partition(str(i))==role) for role in ('fit','hold')}


def test_missing_gold_never_forced_into_training_but_remains_in_hold():
    docs=documents()
    a,ca=row('a',docs['fit'],('박','밬'))
    b,cb=row('b',docs['hold'],('박','밬'))
    pools,outcomes=assemble([ca,cb],{'a':a,'b':b},set())
    assert pools['fit']==[]
    assert pools['hold'][0]['expected']==['밖']
    assert pools['hold'][0]['candidates']==['박','밬']
    assert [r['status'] for r in outcomes]==['missing_gold','missing_gold']


def test_singletons_unscored_and_quarantine_keep_diagnostic_rows():
    docs=documents()
    a,ca=row('a',docs['fit'],('밖',))
    b,cb=row('b',docs['fit']);cb['groups']=[]
    c,cc=row('c',docs['hold'])
    pools,outcomes=assemble([ca,cb,cc],{'a':a,'b':b,'c':c},{'c'})
    assert pools=={'fit':[],'hold':[]}
    assert [r['status'] for r in outcomes]==['single','unscored','quarantined']


def test_same_document_stays_together_and_source_labels_are_not_candidates():
    document=documents()['fit']
    a,ca=row('a',document);b,cb=row('b',document)
    pools,outcomes=assemble([ca,cb],{'a':a,'b':b},set())
    assert len(pools['fit'])==2 and not pools['hold']
    assert {r['role'] for r in outcomes}=={'fit'}
    assert all(r['candidates']==['박','밖'] for r in pools['fit'])


def test_source_alias_crossing_documents_is_rejected():
    docs=documents()
    a,ca=row('a',docs['fit']);b,cb=row('b',docs['hold'])
    b['source_ids']=a['source_ids']
    with pytest.raises(ValueError,match='source identity leakage'):
        assemble([ca,cb],{'a':a,'b':b},set())


@pytest.mark.parametrize('change',['input','duplicate','missing','nontrain','scores'])
def test_corrupt_or_incomplete_capture_is_rejected(change):
    a,ca=row('a',documents()['fit'])
    captures=[deepcopy(ca)]
    if change=='input':captures[0]['input']='wrong'
    if change=='duplicate':captures.append(ca)
    if change=='missing':captures=[]
    if change=='nontrain':a['split']='hold'
    if change=='scores':captures[0]['groups'][0]['teacher_scores']=[]
    with pytest.raises(ValueError):assemble(captures,{'a':a},set())
