"""Independent spelling/pronunciation examples, not round trips."""
import pytest

from hanglish.core.phonology import romanize_word


@pytest.mark.parametrize(('text', 'expected'), [
    ('밟', 'bap'), ('밟다', 'bapda'), ('밟고', 'bapgo'), ('밟기', 'bapgi'),
    ('밟게', 'bapge'), ('밟는', 'bamneun'), ('밟습니다', 'bapseumnida'),
    ('밟지', 'bapji'), ('밟아', 'balba'), ('밟으니', 'balbeuni'),
    ('짓밟고', 'jitbapgo'), ('짓밟는', 'jitbamneun'),
])
def test_balp_stem_uses_labial_before_consonantal_endings(text, expected):
    assert romanize_word(text) == expected


@pytest.mark.parametrize(('text', 'expected'), [
    ('넓다', 'neolda'), ('넓고', 'neolgo'), ('넓어', 'neolbeo'),
    ('여덟', 'yeodeol'), ('여덟이', 'yeodeolbi'),
    ('여덟하고', 'yeodeolhago'), ('여덟-히', 'yeodeol-hi'),
    ('밝고', 'balgo'), ('읽고', 'ilgo'), ('흙과', 'heukgwa'),
])
def test_other_stems_and_vowel_liaison_are_preserved(text, expected):
    assert romanize_word(text) == expected


@pytest.mark.parametrize(('text', 'expected'), [
    ('밟히다', 'balpida'), ('밟혀', 'balpyeo'),
    ('넓히다', 'neolpida'), ('넓혀', 'neolpyeo'),
    ('짓밟히다', 'jitbalpida'), ('밟혔다', 'balpyeotda'),
    ('드넓히다', 'deuneolpida'),
])
def test_cluster_aspiration_retains_lateral(text, expected):
    assert romanize_word(text) == expected


def test_mixed_input_keeps_raw_boundaries():
    assert romanize_word('API밟고-test 2회') == 'APIbapgo-test 2hoe'
