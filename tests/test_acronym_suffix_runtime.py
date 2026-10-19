from types import MethodType, SimpleNamespace

import pytest

from hanglish.core.lattice import WordOption
from hanglish.core.neural_converter import NeuralConverter


def converter_for(source, suffix_options, baseline=None):
    baseline = baseline or (WordOption(source, 6., 'raw'),)
    calls = []

    def options(text):
        calls.append(text)
        return baseline if text == source else tuple(suffix_options)

    converter = SimpleNamespace(generator=SimpleNamespace(options=options), transducer=None)
    converter._word_options = MethodType(NeuralConverter._word_options, converter)
    return converter, calls, baseline


def test_runtime_preserves_existing_options_and_validates_learned_suffix():
    baseline = (WordOption('AIneun', 6., 'raw'), WordOption('기존', 7., 'mixed'))
    converter, calls, _ = converter_for('AIneun', [
        WordOption('는', 3.25, 'syllable'), WordOption('는', 2., 'syllable'),
        WordOption('닌', 1., 'syllable'), WordOption('neun', 0., 'raw'),
        WordOption('는!', 1., 'mixed'),
    ], baseline)
    result = converter._word_options('AIneun', True)
    assert result == baseline + (WordOption('AI는', 3.25, 'mixed'),)
    assert calls == ['AIneun', 'neun']


@pytest.mark.parametrize('source,learned', [
    ('Aneun', True), ('AiNeun', True), ('AI', True), ('AIneun', False),
    ('AINEUN', True), ('A1neun', True), ('A' * 39 + 'ne', True),
])
def test_runtime_exclusions_do_not_lookup_suffix(source, learned):
    converter, calls, baseline = converter_for(source, [WordOption('는', 3., 'neural')])
    assert converter._word_options(source, learned) == baseline
    assert calls == [source]


def test_runtime_caps_and_deduplicates_candidates_without_rescoring(monkeypatch):
    monkeypatch.setattr('hanglish.core.neural_converter.romanize_word', lambda _: 'AAneun')
    baseline = (WordOption('AAneun', 6., 'raw'), WordOption('AA가', 7., 'mixed'))
    suffixes = [WordOption(text, float(i) / 10, 'syllable') for i, text in enumerate(
        ['가', '각', '각', '간', '갇', '갈', '감'])]
    converter, _, _ = converter_for('AAneun', suffixes, baseline)
    result = converter._word_options('AAneun', True)
    assert result[:2] == baseline
    assert result[2:] == tuple(WordOption('AA' + suffixes[i].text, suffixes[i].score, 'mixed')
                              for i in [1, 3, 4, 5])


@pytest.mark.parametrize('score', [float('nan'), float('inf'), float('-inf')])
def test_runtime_rejects_nonfinite_suffix_scores(score):
    converter, _, _ = converter_for('AIneun', [
        WordOption('는', score, 'syllable'), WordOption('neun', 0., 'raw')])
    with pytest.raises(ValueError, match='non-finite'):
        converter._word_options('AIneun', True)
