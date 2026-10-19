import json
import pytest

from scripts.evaluate_sentence_only import predict_sentence,read_inputs,run,sha,summarize


def corpus(tmp_path,context='',text='testreul sihaenghanda'):
    path=tmp_path/'dev.jsonl';manifest=tmp_path/'manifest.json'
    case=dict(id='x',split='dev',input=text,context=context,expected=['test를 시행한다'],category='mixed')
    path.write_text(json.dumps(case,ensure_ascii=False)+'\n')
    manifest.write_text(json.dumps(dict(corpus_version='test',input_contract='roman_sentence_only',sha256={path.name:sha(path)},counts={'dev':1})))
    return path,manifest


def test_nonempty_external_context_rejected(tmp_path):
    path,manifest=corpus(tmp_path,context='실험을 한다')
    with pytest.raises(ValueError,match='external Korean context'):read_inputs(path,manifest)


def test_supplied_hangul_in_input_rejected(tmp_path):
    path,manifest=corpus(tmp_path,text='test를 시행한다')
    with pytest.raises(ValueError,match='without supplied Hangul'):read_inputs(path,manifest)


def test_only_roman_input_and_fixed_options_reach_converter(tmp_path):
    path,manifest=corpus(tmp_path)
    class Spy:
        def convert(self,text,**kwargs):
            assert text=='testreul sihaenghanda'
            assert kwargs==dict(context='',engine='neural',mode='batch',raw=False)
            return dict(input=text,context='',external_context_used=False,candidates=[dict(text='test를 시행한다')])
    report=run(path,manifest,tmp_path/'out',Spy())
    capture=json.loads((tmp_path/'out/capture.jsonl').read_text())
    assert set(capture['request'])=={'input','context','engine','mode','raw'}
    assert 'expected' not in capture and report['passed']


def test_converter_cannot_claim_external_context_use():
    class Bad:
        def convert(self,text,**kwargs):return dict(input=text,context='',external_context_used=True)
    with pytest.raises(ValueError,match='violated'):predict_sentence(Bad(),'roman input')


def test_errors_and_partial_word_answers_do_not_disappear():
    cases=[dict(id='a',input='testreul sihaenghanda',expected=['test를 시행한다']),dict(id='b',input='jeoneun haksaengimnida',expected=['저는 학생입니다'])]
    result=summarize([dict(id='a',response=dict(candidates=[dict(text='시행한다')])),dict(id='b',error='failed')],cases)
    assert result['counts']==dict(rows=2,top1=0,top5=0,errors=1)
    assert result['passed'] is False
