import pytest
from hanglish.core.phonology import romanize_word


@pytest.mark.parametrize(('text','expected'),[
    ('맑게','malge'),('묽고','mulgo'),('읽고','ilgo'),('굵고','gulgo'),('굵기','gulgi'),
    ('늙거나','neulgeona'),('얽거든','eolgeodeun'),('해맑게','haemalge'),('되읽고','doeilgo'),
    ('읽기는','ilgineun'),('읽기도','ilgido'),('읽기에서는','ilgieseoneun'),('읽고는','ilgoneun'),
])
def test_predicate_stem_cluster_before_velar_ending(text,expected):
    assert romanize_word(text)==expected


@pytest.mark.parametrize(('text','expected'),[
    ('흙과','heukgwa'),('닭고기','dakgogi'),('닭과','dakgwa'),('국고','gukgo'),('국기','gukgi'),
    ('읽다','ikda'),('읽는','ingneun'),('읽어요','ilgeoyo'),('맑다','makda'),
    ('닭게','dakge'),('맑가방','makgabang'),
])
def test_other_boundaries_and_unparsed_suffixes_keep_existing_rules(text,expected):
    assert romanize_word(text)==expected


def test_mixed_content_and_non_hangul_boundaries_are_unchanged():
    assert romanize_word('API읽고-test 맑게 2회')=='APIilgo-test malge 2hoe'
    assert romanize_word('읽-고')=='ik-go'


@pytest.mark.parametrize('contents',[None,'{"schema":"invalid"}'])
def test_missing_or_invalid_inventory_fails_before_engine_accepts_input(tmp_path,monkeypatch,contents):
    from hanglish.core import predicate_boundaries as boundaries
    from hanglish.core.neural_converter import NeuralConverter,neural_readiness
    path=tmp_path/'inventory.json'
    if contents is not None:
        path.write_text(contents)
    monkeypatch.setattr(boundaries,'INVENTORY',path)
    boundaries._inventory.cache_clear()
    try:
        assert neural_readiness()==(False,'용언 경계 사전 자료 확인 필요')
        with pytest.raises((FileNotFoundError,ValueError)):
            NeuralConverter(external_lexicon=False)
    finally:
        boundaries._inventory.cache_clear()
