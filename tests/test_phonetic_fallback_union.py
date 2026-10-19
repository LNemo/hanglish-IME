from types import SimpleNamespace

import pytest

from hanglish.core.lattice import WordOption
from scripts.phonetic_fallback_union import preserved_options


def converter(baseline, alternatives):
    return SimpleNamespace(_word_options=lambda source, learned: baseline,
                           generator=SimpleNamespace(options=lambda source: alternatives))


def test_baseline_preserved_and_forward_valid_fallbacks_appended():
    baseline = (WordOption('밥', 3., 'neural'), WordOption('bap', 0., 'raw'))
    alternatives = [WordOption('밥', 1., 'syllable'), WordOption('밮', .9, 'syllable'),
                    WordOption('밤', .8, 'syllable')]
    result = preserved_options(converter(baseline, alternatives), 'bap', True)
    assert result[:2] == baseline
    assert [o.text for o in result] == ['밥', 'bap', '밮']


@pytest.mark.parametrize('kind', ['lexicon', 'morphology', 'mixed', 'raw'])
def test_credible_paths_do_not_gain_free_spelling_alternatives(kind):
    baseline = (WordOption('강한 후보', 6., kind), WordOption('밥', 3., 'neural'))
    assert preserved_options(converter(baseline, []), 'bap', True) is baseline


def test_no_neural_or_nonlearned_path_is_unchanged():
    baseline = (WordOption('밥', 1., 'syllable'),)
    assert preserved_options(converter(baseline, []), 'bap', True) is baseline
    baseline = (WordOption('밥', 3., 'neural'),)
    assert preserved_options(converter(baseline, []), 'bap', False) is baseline


def test_bound_and_original_order():
    baseline = (WordOption('밥', 3., 'neural'),)
    alternatives = [WordOption('밮', 1., 'syllable'), WordOption('밦', .9, 'syllable')]
    result = preserved_options(converter(baseline, alternatives), 'bap', True, limit=1)
    assert [o.text for o in result] == ['밥', '밮']


@pytest.mark.parametrize('limit', [0, -1, 9, True, 1.5])
def test_invalid_limit_rejected(limit):
    with pytest.raises(ValueError, match='fallback limit'):
        preserved_options(converter((), []), 'bap', True, limit=limit)
