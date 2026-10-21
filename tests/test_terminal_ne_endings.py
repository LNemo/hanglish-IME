"""Terminal -네/-네요 morphology precedes forward pronunciation."""
import pytest

from hanglish.core.productive_endings import ProductiveEndings


@pytest.mark.parametrize(('stem', 'roman', 'hangul'), [
    ('가', 'gane', '가네'), ('먹', 'meongne', '먹네'),
    ('살', 'sane', '사네'), ('알', 'ane', '아네'),
    ('울', 'uneyo', '우네요'), ('가늘', 'ganeuneyo', '가느네요'),
    ('핥', 'halle', '핥네'), ('넓', 'neolle', '넓네'),
    ('맑', 'mangne', '맑네'), ('듣', 'deunne', '듣네'),
    ('돕', 'domneyo', '돕네요'), ('짓', 'jinne', '짓네'),
    ('좋', 'jonneyo', '좋네요'),
])
def test_general_terminal_attachment_and_forward_verification(stem, roman, hangul):
    assert hangul in ProductiveEndings({stem}, {}).options(roman)


def test_plain_rieul_deletion_does_not_delete_compound_codas():
    endings = ProductiveEndings({'살', '핥', '넓', '맑'}, {})
    assert '살네' not in endings.options('salle')
    assert '하네' not in endings.options('hane')
    assert '너네' not in endings.options('neone')
    assert '마네' not in endings.options('mane')


def test_terminal_forms_require_an_attested_predicate_stem():
    endings = ProductiveEndings({'먹'}, {})
    assert '가네' not in endings.options('gane')
    assert '먹네' not in endings.options('meokne')
    assert '먹네까' not in endings.options('meongnekka')
