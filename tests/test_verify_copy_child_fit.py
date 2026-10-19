from types import SimpleNamespace

from hanglish.core.copy_child_proposals import propose_children
from scripts.verify_copy_child_fit import enumerate_proposals


class Generator:
    english_suffixes=[('reul','를')]
    def options(self,source):
        return [SimpleNamespace(text=source[:-4]+'를',kind='mixed')]


def parent(source='widgetreul',target='widget를'):
    return {'text':target,'segments':[dict(input_start=0,input_end=len(source),source=source,text=target,kind='mixed')]}


def test_independent_enumerator_matches_helper_with_ties_and_existing_texts():
    row=parent();beams={'reul':[['릀',-1.],['륿',-1.]]};generator=Generator()
    actual=enumerate_proposals('widgetreul',[row],generator,beams)
    produced=propose_children('widgetreul',row['text'],row['segments'],generator=generator,
                              generated=beams,existing_texts=[row['text']])['proposals']
    assert actual==[dict(p,parent_candidate_index=0) for p in produced]
    assert [p['text'] for p in actual]==['widget릀','widget륿']


def test_single_fallback_parent_can_generate_and_uppercase_copy_is_exact():
    row=parent('WIDGETREUL','WIDGET를');beams={'reul':[['륿',-1.]]}
    actual=enumerate_proposals('WIDGETREUL',[row],Generator(),beams)
    assert actual[0]['text']=='WIDGET륿'
    assert actual[0]['segments'][0]['source']==actual[0]['segments'][0]['text']=='WIDGET'
    assert actual[0]['segments'][1]['source']=='REUL'


def test_empty_beam_does_not_fabricate_a_candidate():
    assert enumerate_proposals('widgetreul',[parent()],Generator(),{'reul':[]})==[]
