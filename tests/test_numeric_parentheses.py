import pytest

from hanglish.core.neural_converter import NeuralConverter
from hanglish.core.numeric_parentheses import NumericParenthesesConverter
from hanglish.core.phonology import romanize_word


@pytest.fixture(scope='module')
def engines():
    baseline=NeuralConverter(context_ranking=False,external_lexicon=False)
    return baseline,NumericParenthesesConverter(baseline,experimental=True)


@pytest.mark.parametrize('source,expected',[
    ('6got(57.6%)eun','6곳(57.6%)은'),('474got(64.8%)i','474곳(64.8%)이'),
    ('89got(17%)eun','89곳(17%)은'),('3il(mok)kkaji','3일(목)까지'),
    ('7myeong(12%)ui','7명(12%)의'),('2ho(250%)neun','2호(250%)는'),
    ('8won(0.25％)eul!','8원(0.25％)을!'),('31il(geumyoil)ro','31일(금요일)로'),
    ('1il(hwa)','1일(화)'),('3.5won(10%)','3.5원(10%)'),
])
def test_recognizes_bounded_annotation_grammar_and_retains_raw(engines,source,expected):
    _,engine=engines
    response=engine.convert(source,engine='morphology')
    assert expected in [c['text'] for c in response['candidates']]
    assert source in [c['text'] for c in response['candidates']]
    for candidate in response['candidates']:
        assert ''.join(s['source'] for s in candidate['segments'])==source
        assert ''.join(s['text'] for s in candidate['segments'])==candidate['text']
        for segment in candidate['segments']:
            assert source[segment['input_start']:segment['input_end']]==segment['source']
            if segment['kind']=='raw':assert segment['source']==segment['text']
    assert romanize_word(expected)==source


@pytest.mark.parametrize('source',[
    '10kg(50%)eun','5test(50%)eun','6GOT(50%)eun','0il(mok)','32il(mok)',
    '3.5il(mok)','3il(foo)','3il(mok)euro','6got(50%)neun','2gae(50%)eun',
    '6goteun(50%)','6got((50%))eun','6got(50%eun','6got(50%)eun_more',
    'https://example.com/6got(50%)eun','user6got(50%)eun@example.com',
    '`6got(50%)eun`','v1.2.3(50%)','0xff(50%)','1e10(50%)',
    '6got(abc)eun','6got(50%)eun/bar','3il(mok)unknown','9'*100+'il(mok)',
])
def test_unknown_annotations_invalid_allomorphs_and_codes_stay_raw(engines,source):
    _,engine=engines
    response=engine.convert(source,engine='morphology')
    assert [c['text'] for c in response['candidates']]==[source]


@pytest.mark.parametrize('source',['12il','30yeojeomeul','3.5won','testreul sihaeng','imnida','192.168.0.1','10kg','gachi'])
def test_nonmatching_inputs_keep_exact_parent_candidates(engines,source):
    baseline,engine=engines
    assert engine.convert(source,engine='morphology')['candidates']==baseline.convert(source,engine='morphology')['candidates']


def test_raw_bypass_and_offsets_after_an_existing_prefix(engines):
    _,engine=engines
    source='약 6got(57.6%)eun 그대로'
    response=engine.convert(source,engine='morphology')
    assert any(c['text']=='약 6곳(57.6%)은 그대로' for c in response['candidates'])
    for candidate in response['candidates']:
        for segment in candidate['segments']:
            assert source[segment['input_start']:segment['input_end']]==segment['source']
    assert engine.convert(source,raw=True)['result']==source


def test_experimental_opt_in(engines):
    with pytest.raises(ValueError,match='experimental'):
        NumericParenthesesConverter(engines[0])


def test_does_not_override_parent_quantity_candidate_pruning(engines):
    baseline, engine = engines
    # In morphology mode the parent prunes 개 behind 가에 before numeric parsing.
    # This experiment changes annotation boundaries, not that separate policy.
    assert baseline._numeric_unit_pieces('2gae', 0, False) is None
    assert engine.convert('2gae(250%)neun', engine='morphology')['result'] == '2gae(250%)neun'
