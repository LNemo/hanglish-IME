import threading
from types import SimpleNamespace

from hanglish.core.lattice import WordOption
from hanglish.core.neural_converter import NeuralConverter


def fixture_converter(options, generated):
    converter=object.__new__(NeuralConverter)
    converter.generator=SimpleNamespace(options=lambda source:tuple(options))
    converter.transducer=SimpleNamespace(generate_constrained=lambda *args, **kwargs: [], generate=lambda source,beam_size:generated)
    converter.lock=threading.RLock()
    return converter


def test_unknown_neural_output_keeps_four_distinct_forward_valid_fallbacks():
    alternatives=[WordOption(text,1.,'syllable') for text in ('각','가','갂','갃','갉','갘','갔')]
    raw=WordOption('gak',0.,'raw')
    converter=fixture_converter(alternatives+[raw],[('각',-.2)])
    options=converter._word_options('gak',True)
    assert [o.text for o in options]==['각','gak','갂','갃','갉','갘']
    assert options[0].kind=='neural' and options[0].score>3
    assert options[1]==raw
    assert all(o.kind=='syllable' for o in options[2:])


def test_nonlearned_and_empty_generation_preserve_existing_options():
    options=[WordOption('각',1.,'syllable'),WordOption('gak',0.,'raw')]
    converter=fixture_converter(options,[])
    assert converter._word_options('gak',True)==tuple(options)
    assert converter._word_options('gak',False)==tuple(options)


def test_strong_english_raw_evidence_does_not_admit_free_korean_fallbacks():
    raw=WordOption('gak',7.5,'raw')
    converter=fixture_converter([raw,WordOption('갂',1.,'syllable')],[('각',-.2)])
    assert converter._word_options('gak',True)==(raw,)


def test_runtime_appends_constrained_candidates_after_preserved_options():
    raw = WordOption('gak', 0., 'raw')
    converter = fixture_converter([WordOption('각', 1., 'syllable'), raw], [('각', -.2)])
    converter.transducer.generate_constrained = lambda source, beam_size: [
        (text, -1.) for text in ('각', '갂', '갃', '갉', '갘', '갔')]
    result = converter._word_options('gak', True)
    assert [o.text for o in result] == ['각', 'gak', '갂', '갃', '갉', '갘']
    assert result[1] == raw
    assert all(o.kind == 'neural' for o in result[2:])
