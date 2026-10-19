import pytest
from scripts.probe_runtime_collision_contexts import measure, select_pilot


def test_pilot_samples_distinct_forms_and_never_accepts_held_rows():
    rows=[dict(id=f'{s}-{w}-{i}',word_input=s,word=w,split='train')
          for s in ('a','b','c') for w in ('one','two','three') for i in range(2)]
    chosen=select_pilot(rows,families=2)
    assert len(chosen)==4
    assert chosen==select_pilot(list(reversed(rows)),families=2)
    assert len({r['word_input'] for r in chosen})==2
    for source in {r['word_input'] for r in chosen}:
        assert len({r['word'] for r in chosen if r['word_input']==source})==2
    with pytest.raises(ValueError,match='TRAIN'):
        select_pilot([{**rows[0],'split':'hold'}])


def test_diagnostics_distinguish_missing_candidate_from_wrong_selection():
    target=dict(id='a',word_input='inneun',word='잇는',text='세상을 잇는 사람',contrast_candidates=['있는','잇는'])
    record=dict(focus_start=0,focus_end=6,groups=[dict(start=0,end=6,candidates=['있는','잇는'])],
                pieces=[dict(start=0,end=6,options=[dict(text='있는'),dict(text='잇는')])],
                sentence_calls=[dict(candidates=['세상을 있는 사람','세상을 잇는 사람'])],
                response=dict(candidates=[dict(text='세상을 있는 사람'),dict(text='세상을 잇는 사람')]))
    measured=measure(record,target)
    assert measured['word_gold_in_pool'] and measured['sentence_gold_in_pool']
    assert not measured['word_first_correct'] and not measured['sentence_first_correct']
    assert measured['sentence_top5_correct']
    absent=measure({**record,'groups':[],'pieces':[],'sentence_calls':[]},target)
    assert not absent['word_gold_in_pool'] and not absent['sentence_gold_in_pool']
