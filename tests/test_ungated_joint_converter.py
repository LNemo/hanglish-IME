from copy import deepcopy
import threading

import pytest

from hanglish.core.ungated_joint_converter import UngatedJointConverter, recover_unranked


def baseline(source='gachi imnida', output='gachi 임니다'):
    parts = source.split(' ')
    rendered = output.split(' ')
    segments = []
    start = 0
    for index, (word, target) in enumerate(zip(parts, rendered)):
        if index:
            segments.append(dict(input_start=start, input_end=start + 1,
                                 source=' ', text=' ', kind='literal'))
            start += 1
        segments.append(dict(input_start=start, input_end=start + len(word),
                             source=word, text=target, kind='lexical'))
        start += len(word)
    return dict(input=source, context='', version='parent-v1', result=output,
                candidates=[dict(text=output, score=99., segments=segments),
                            dict(text=source, score=None, segments=[dict(
                                input_start=0, input_end=len(source), source=source,
                                text=source, kind='raw')])],
                sentence_ranking_applied=False, ranking_applied=True,
                warnings=['parent length warning'], engine_ms=1.,
                phonetic_choice={'public': [output]}, phonetic_score_scale={'parameters': []})


def generate(word, beam_size=8):
    assert beam_size == 8
    return {'gachi': [('같이', -1.), ('가치', -2.)],
            'imnida': [('입니다', -1.), ('임니다', -2.)]}[word]


class Ranker:
    def __init__(self, support=lambda text: True):
        self.support = support
        self.calls = []

    def can_score(self, text):
        return self.support(text)

    def score_pairs(self, contexts, texts):
        assert not any(contexts)
        assert all(self.can_score(text) for text in texts)
        self.calls.append(texts)
        return [10. if text == '같이 입니다' else 0. for text in texts]


def test_joint_repair_uses_multiple_spans_and_preserves_input():
    original = baseline()
    saved = deepcopy(original)
    result, trace = recover_unranked(original['input'], original, generate, Ranker())
    assert result['result'] == '같이 입니다'
    assert result['candidates'][-1] == original['candidates'][-1]
    assert original == saved
    assert trace['generation_requests'] == 2
    assert trace['applied']
    assert 'phonetic_choice' not in result and 'phonetic_score_scale' not in result
    assert result['ungated_joint']['selection'] == 'sentence_ranker_unscaled'


def test_oversized_parent_does_not_stop_generation_and_is_retained_unscored():
    original = baseline()
    ranker = Ranker(lambda text: text == '같이 입니다')
    result, trace = recover_unranked(original['input'], original, generate, ranker)
    assert result['result'] == '같이 입니다'
    parent = next(row for row in trace['pool'] if row['text'] == original['result'])
    assert not parent['supported'] and parent['score'] is None
    assert ranker.calls == [['같이 입니다']]
    assert result['ungated_joint']['unsupported'] > 0


def test_no_supported_candidates_keeps_exact_fallback_and_diagnostic_pool():
    original = baseline()
    ranker = Ranker(lambda text: False)
    result, trace = recover_unranked(original['input'], original, generate, ranker)
    assert result is original
    assert trace['generation_requests'] == 2 and trace['pool']
    assert trace['reason'] == 'no_supported_candidates'
    assert not ranker.calls


def test_known_english_copy_is_never_sent_to_generator():
    original = baseline('test imnida', 'test 임니다')
    calls = []

    def tracked(word, beam_size):
        calls.append(word)
        return generate(word, beam_size)

    result, trace = recover_unranked(original['input'], original, tracked, Ranker(),
                                     known_english={'test'})
    assert calls == ['imnida']
    assert all(row['text'].startswith('test ') for row in trace['pool'])
    assert result['candidates'][-1]['text'] == 'test imnida'


def test_recovery_preserves_unrelated_warnings_and_replaces_stale_length_warning():
    original = baseline()
    original['warnings'] = ['protected token warning',
        '문장 모델의 길이 범위를 넘어 단어별 문맥 순위를 사용했습니다.']
    result, _ = recover_unranked(original['input'], original, generate, Ranker())
    assert result['warnings'] == ['protected token warning']
    assert result['ungated_joint']['parent_warnings'] == original['warnings']


@pytest.mark.parametrize('bad_scores', [[float('nan')], [], [True]])
def test_malformed_ranker_scores_rejected(bad_scores):
    class BadRanker(Ranker):
        def score_pairs(self, contexts, texts):
            return bad_scores
    original = baseline()
    with pytest.raises(ValueError, match='invalid scores'):
        recover_unranked(original['input'], original, generate, BadRanker())


class Parent:
    def __init__(self, ranked=False):
        self.lock = threading.RLock()
        self.response = baseline()
        self.response['sentence_ranking_applied'] = ranked
        self.sentence_ranker = Ranker()
        self.generation_calls = 0

    def convert(self, *args):
        return self.response

    def load_phonetic_model(self):
        self.generation_calls += 1
        return type('Generator', (), {'generate': staticmethod(generate)})()


@pytest.mark.parametrize('kwargs,ranked', [({}, True), ({'engine': 'baseline'}, False),
                                         ({'raw': True}, False), ({'context': 'hint'}, False)])
def test_adapter_controls_are_exact_passthrough_without_extra_inference(kwargs, ranked):
    parent = Parent(ranked)
    wrapper = UngatedJointConverter(parent)
    result = wrapper.convert(parent.response['input'], **kwargs)
    assert result is parent.response
    assert parent.generation_calls == 0
    assert not parent.sentence_ranker.calls
    assert not wrapper.last_trace['applied']


def test_adapter_actual_recovery_has_distinct_version_and_clears_previous_trace():
    parent = Parent()
    wrapper = UngatedJointConverter(parent)
    result = wrapper.convert(parent.response['input'])
    assert result['result'] == '같이 입니다'
    assert result['version'].startswith('experimental-ungated-joint-1-')
    assert parent.response['version'] == 'parent-v1'
    assert wrapper.last_trace['applied']
    wrapper.convert(parent.response['input'], raw=True)
    assert not wrapper.last_trace['applied']


def test_invalid_source_alignment_rejected_before_inference():
    original = baseline()
    original['candidates'][0]['segments'][0]['input_end'] = 2
    with pytest.raises(ValueError, match='source offsets'):
        recover_unranked(original['input'], original, generate, Ranker())
