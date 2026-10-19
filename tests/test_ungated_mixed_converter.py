from copy import deepcopy
import threading

import pytest

from hanglish.core.lattice import WordOption
from hanglish.core.ungated_mixed_converter import UngatedMixedConverter, recover_mixed


def seg(source, start, end, text, kind):
    return {'input_start': start, 'input_end': end, 'source': source[start:end],
            'text': text, 'kind': kind}


def baseline(source='api3beta imnida'):
    return {'input': source, 'context': '', 'version': 'parent-v1',
            'result': 'api3beta 임니다', 'sentence_ranking_applied': False,
            'ranking_applied': True, 'warnings': ['keep',
                '문장 모델의 길이 범위를 넘어 단어별 문맥 순위를 사용했습니다.'],
            'candidates': [
                {'text': 'api3beta 임니다', 'score': 0., 'segments': [
                    seg(source, 0, 8, 'api3beta', 'raw'), seg(source, 8, 9, ' ', 'raw'),
                    seg(source, 9, 15, '임니다', 'lexicon')]},
                {'text': source, 'score': None, 'segments': [
                    seg(source, 0, len(source), source, 'raw')]}],
            'phonetic_choice': {}, 'phonetic_score_scale': {}}


def joint_trace(base, *, supported=True):
    return {'baseline': deepcopy(base), 'applied': supported,
            'pool': [{'text': 'api3beta 임니다',
                      'segments': deepcopy(base['candidates'][0]['segments']),
                      'origins': ['parent'], 'supported': supported,
                      'score': 1.0 if supported else None}]}


class Ranker:
    def __init__(self, support=lambda text: True): self.support=support; self.calls=[]
    def can_score(self, text): return self.support(text)
    def score_pairs(self, contexts, texts):
        self.calls.append(list(texts)); return [4.0 for _ in texts]


def options(source, learned=True):
    values = {'api': WordOption('아피', 2., 'lexicon'),
              'beta': WordOption('베타', 1., 'lexicon')}
    return [values[source]] if source in values else [WordOption(source, 0., 'raw')]


def test_union_preserves_old_pool_scores_new_only_and_raw_reserve():
    base = baseline(); ranker = Ranker()
    result, trace = recover_mixed(base['input'], base, joint_trace(base), options, ranker)
    assert trace['union_pool'][0]['score'] == 1.0
    assert ranker.calls and all('api3beta 임니다' not in call for call in ranker.calls)
    assert trace['mixed_proposals'] and trace['applied']
    assert result['candidates'][-1] == base['candidates'][-1]
    assert '베타' in result['result']
    assert any(segment['source'] == '3' and segment['text'] == '3'
               for segment in result['candidates'][0]['segments'])
    assert 'phonetic_choice' not in result and 'phonetic_score_scale' not in result
    assert result['warnings'][0] == 'keep'


def test_duplicate_is_deduped_and_not_rescored(monkeypatch):
    base = baseline(); ranker = Ranker()
    monkeypatch.setattr('hanglish.core.ungated_mixed_converter.mixed_span_proposals.sentence_extras',
        lambda *a, **k: [('api3beta 임니다', 0., deepcopy(base['candidates'][0]['segments']))])
    result, trace = recover_mixed(base['input'], base, joint_trace(base), options, ranker)
    assert result is base and ranker.calls == []
    assert len(trace['union_pool']) == 1
    assert trace['union_pool'][0]['origins'] == ['parent', 'mixed']
    assert trace['supported_order'] == ['api3beta 임니다']


def test_unsupported_new_candidate_is_retained_without_score():
    base = baseline(); ranker = Ranker(lambda text: text == 'api3beta 임니다')
    result, trace = recover_mixed(base['input'], base, joint_trace(base), options, ranker)
    mixed = [row for row in trace['union_pool'] if 'mixed' in row['origins']]
    assert mixed and all(not row['supported'] and row['score'] is None for row in mixed)
    assert ranker.calls == [] and result is base


def test_mixed_supported_recovers_all_unsupported_joint_fallback():
    base = baseline(); ranker = Ranker(lambda text: text != 'api3beta 임니다')
    result, trace = recover_mixed(base['input'], base, joint_trace(base, supported=False),
                                  options, ranker)
    assert trace['applied'] and result['result'] != base['result']
    assert ranker.calls and trace['union_pool'][0]['score'] is None


class Parent:
    def __init__(self, ranked=False):
        self.lock=threading.RLock(); self.models_loaded=False
        self.version='parent-v1'
        self.response=baseline(); self.response['sentence_ranking_applied']=ranked
        self.sentence_ranker=Ranker(); self.word_options=options; self.generation_calls=0
    def convert(self, *args): return self.response
    def load_phonetic_model(self):
        self.generation_calls += 1
        return type('M',(),{'generate':staticmethod(lambda word,beam_size=8: [])})()
    def load_models(self): self.models_loaded=True; return 'loaded'


@pytest.mark.parametrize('kwargs,ranked', [({},True),({'engine':'baseline'},False),
    ({'raw':True},False),({'context':'hint'},False)])
def test_controls_are_exact_passthrough(kwargs, ranked):
    parent=Parent(ranked); adapter=UngatedMixedConverter(parent)
    result=adapter.convert(parent.response['input'], **kwargs)
    assert result is parent.response
    if ranked or kwargs: assert not parent.generation_calls


def test_no_mixed_extras_returns_joint_response_unchanged(monkeypatch):
    parent=Parent(); adapter=UngatedMixedConverter(parent)
    monkeypatch.setattr('hanglish.core.ungated_mixed_converter.mixed_span_proposals.sentence_extras',
                        lambda *a, **k: [])
    result=adapter.convert(parent.response['input'])
    assert adapter.last_trace['reason']=='no_mixed_proposals'
    assert adapter.last_trace['supported_order'] == ['api3beta 임니다']
    assert result.get('ungated_joint', {}).get('experimental') is True
    assert 'ungated_mixed' not in result


def test_english_and_literal_copy_are_preserved():
    source='test3beta imnida'; base=baseline(source)
    base['candidates'][0]['text']='test3beta 임니다'
    base['candidates'][0]['segments']=[seg(source,0,9,'test3beta','raw'),
        seg(source,9,10,' ','raw'),seg(source,10,16,'임니다','lexicon')]
    trace=joint_trace(base)
    def english(word, learned=True):
        return [WordOption(word,0.,'raw')] if word=='test' else options(word,learned)
    _, actual=recover_mixed(source,base,trace,english,Ranker())
    changed=next(row for row in actual['union_pool'] if 'mixed' in row['origins'])
    assert changed['text'].startswith('test3')
    assert any(s['source']=='3' and s['text']=='3' and s['kind']=='raw'
               for s in changed['segments'])


def test_model_loading_delegates():
    parent=Parent(); adapter=UngatedMixedConverter(parent)
    assert adapter.models_loaded is False and adapter.load_models()=='loaded'
    assert adapter.models_loaded is True


def test_version_includes_known_english_and_changed_metadata_nests_joint_parent():
    parent=Parent(); first=UngatedMixedConverter(parent,known_english={'test'})
    second=UngatedMixedConverter(parent,known_english={'world'})
    assert first.version != second.version
    result=first.convert(parent.response['input'])
    assert result['version'] == first.version
    assert all(result['model_artifacts'][name] == digest
               for name, digest in first.artifacts.items())
    assert 'ungated_joint' not in result
    assert result['ungated_mixed']['joint_parent']['experimental'] is True
    assert result['ungated_mixed']['new_supported_scored'] > 0
    assert result['ungated_mixed']['parent_warnings'] == parent.response['warnings']
    mixed = [row for row in first.last_trace['union_pool'] if 'mixed' in row['origins']]
    assert mixed and all(row['segments'] for row in mixed)
