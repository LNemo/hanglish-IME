import pytest

from hanglish.core.productive_endings import ProductiveEndings


@pytest.fixture(scope='module')
def endings():
    return ProductiveEndings({'밟', '늙', '얽', '읽', '먹', '맑', '가', '살', '듣'},
                             {'듣': {'들으니'}})


@pytest.mark.parametrize(('source', 'expected'), [
    ('balbeuni', '밟으니'), ('neulgeona', '늙거나'), ('eolgeodeun', '얽거든'),
    ('ilgeuni', '읽으니'), ('meogeunikka', '먹으니까'),
    ('malgeona', '맑거나'), ('eolgeodeunyo', '얽거든요'),
    ('ilgoseo', '읽고서'), ('malgie', '맑기에'),
    ('gani', '가니'), ('sani', '사니'), ('sanikka', '사니까'),
    ('deureunikka', '들으니까'),
])
def test_unlisted_stem_ending_combinations(endings, source, expected):
    assert expected in endings.options(source)


@pytest.mark.parametrize('source', ['gukgeona', 'gareuni', 'saleuni', 'deudeuni'])
def test_no_unknown_stem_or_unlicensed_connective_allomorph(endings, source):
    assert not endings.options(source)


def test_ambiguous_irregular_classes_need_attested_forms():
    index = ProductiveEndings({'듣', '돕', '짓', '그렇'}, {})
    for source in ('deudeuni', 'dobeuni', 'jiseuni', 'geureoeuni'):
        assert not index.options(source)
    assert '듣거나' in index.options('deutgeona')


def test_index_scales_with_boundary_classes_not_all_endings(endings):
    assert endings.indexed_forms <= 4 * 9
