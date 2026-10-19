import threading
from types import SimpleNamespace

import pytest

from hanglish.core.mixed_span_converter import MixedSpanConverter
from hanglish.core import mixed_span_proposals


def converter():
    obj=object.__new__(MixedSpanConverter)
    obj.word_options=lambda *args,**kwargs: ()
    obj.sentence_ranker=SimpleNamespace(can_score=lambda text: text!='long',
                                      score_pairs=lambda contexts,texts:[5.]*len(texts))
    obj.lock=threading.RLock()
    return obj


def test_only_new_scoreable_paths_are_scored_and_parent_paths_preserved(monkeypatch):
    parent=[('old',3.,[]),('other',-1.,[])]
    monkeypatch.setattr(mixed_span_proposals,'sentence_extras',lambda *args,**kw:[('new',99.,[]),('long',100.,[])])
    obj=converter();result=obj._extend_sentence_paths('roman sentence','',parent)
    assert result==[('new',5.,[]),*parent]
    assert parent==[('old',3.,[]),('other',-1.,[])]
    assert obj.mixed_span_counts==dict(generated=2,accepted=1)


def test_no_addition_returns_identical_parent_object(monkeypatch):
    monkeypatch.setattr(mixed_span_proposals,'sentence_extras',lambda *args,**kw:[])
    parent=[('old',3.,[])]
    assert converter()._extend_sentence_paths('roman sentence','',parent) is parent


def test_invalid_scores_do_not_silently_change_ranking(monkeypatch):
    monkeypatch.setattr(mixed_span_proposals,'sentence_extras',lambda *args,**kw:[('new',1.,[])])
    obj=converter();obj.sentence_ranker.score_pairs=lambda *args:[float('nan')]
    with pytest.raises(RuntimeError):obj._extend_sentence_paths('roman sentence','',[('old',3.,[])])


def test_raw_and_non_neural_requests_never_generate_supplements(monkeypatch):
    from hanglish.core.neural_converter import NeuralConverter
    def forbidden(*args,**kwargs):
        raise AssertionError('raw or non-neural input reached supplemental generation')
    monkeypatch.setattr(mixed_span_proposals,'sentence_extras',forbidden)
    mixed=MixedSpanConverter(external_lexicon=False)
    source='je23hoe code_v2'
    raw=mixed.convert(source,raw=True)
    assert raw['result']==source and raw['mixed_span']==dict(generated=0,accepted=0)
    parent=NeuralConverter(external_lexicon=False)
    before=parent.convert(source,engine='morphology')
    after=mixed.convert(source,engine='morphology')
    assert before['candidates']==after['candidates']
    assert after['mixed_span']==dict(generated=0,accepted=0)


def test_implementation_identity_is_independent_of_install_directory(monkeypatch, tmp_path):
    from pathlib import Path
    from hanglish.core import mixed_span_converter as module
    from hanglish.core.neural_converter import NeuralConverter
    monkeypatch.setattr(NeuralConverter, '__init__', lambda self, **kwargs: setattr(self, 'version', 'parent'))
    versions=[]
    for directory in ('first', 'second'):
        root=tmp_path/directory
        root.mkdir()
        for name in ('mixed_span_converter.py', 'mixed_span_proposals.py'):
            (root/name).write_text('identical source')
        monkeypatch.setattr(module, '__file__', str(root/'mixed_span_converter.py'))
        monkeypatch.setattr(mixed_span_proposals, '__file__', str(root/'mixed_span_proposals.py'))
        obj=MixedSpanConverter()
        assert all(not Path(key).is_absolute() for key in obj.implementation_artifacts)
        assert set(obj.implementation_artifacts)=={
            'hanglish/core/mixed_span_converter.py', 'hanglish/core/mixed_span_proposals.py'}
        versions.append(obj.version)
    assert versions[0]==versions[1]
