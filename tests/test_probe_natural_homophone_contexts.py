from scripts.probe_natural_homophone_contexts import project,summarize
from hanglish.core.phonology import romanize_word


def test_projection_preserves_focus_alignment_but_omits_gold_and_alternatives():
    text='test 밖 풍경'
    row={'id':'1','text':text,'left_context':'앞 문맥','word_start':5,'word_end':6,
         'word_input':romanize_word('밖'),'word':'밖','contrast_candidates':['박','밖']}
    value=project(row)
    assert value['input'][value['focus_start']:value['focus_end']]=='bak'
    assert set(value)=={'id','input','context','focus_start','focus_end'}


def test_summary_keeps_unaligned_and_missing_gold_rows_in_denominator():
    captures=[{'id':'a','focus_start':0,'focus_end':3,'groups':[], 'ranked_pieces':[]},
              {'id':'b','focus_start':0,'focus_end':3,
               'groups':[{'start':0,'end':3,'candidates':['박','밖']}],
               'ranked_pieces':[{'start':0,'end':3,'options':[{'text':'박'}]}]},
              {'id':'c','focus_start':0,'focus_end':3,
               'groups':[{'start':0,'end':3,'candidates':['박']}],
               'ranked_pieces':[{'start':0,'end':3,'options':[{'text':'박'}]}]}]
    targets={key:{'word':'밖','contrast_candidates':['박','밖'],'spelling_anchors':['밖'],
                  'match_kind':'simple_form'} for key in ('a','b','c')}
    counts,_=summarize(captures,targets)
    assert counts['rows']==3 and counts['aligned_runtime_piece']==2
    assert counts['gold_in_scored_candidates']==1 and counts['informative_with_gold']==1
    assert counts['word_first_correct']==0
