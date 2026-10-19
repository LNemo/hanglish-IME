import copy

import pytest

from scripts.numeric_structure_candidates import NumericSpanGenerator, MAX_PATHS, validate_path


@pytest.fixture(scope='module')
def generator():
    return NumericSpanGenerator()


@pytest.mark.parametrize('source,target', [
    ('11wol29ilbuteo', '11월29일부터'), ('5~6gotgwa', '5~6곳과'),
    ('2015nyeon~2017nyeon', '2015년~2017년'), ('20%garyang', '20%가량'),
    ('31％e', '31％에'), ('3.3㎡dang', '3.3㎡당'),
    ('3gaeworiraneun', '3개월이라는'), ('600nyeoniran', '600년이란'),
    ('600~700mgaryang', '600~700m가량'), ('7000man~9000manwon', '7000만~9000만원'),
    ('(11wol29ilbuteo),', '(11월29일부터),'), ('“20%garyang”!', '“20%가량”!'),
])
def test_grammar_generates_full_paths_with_literal_and_offset_proof(generator, source, target):
    paths = generator.paths(source, 17)
    assert target in [p[0] for p in paths]
    assert len(paths) <= MAX_PATHS
    for path in paths:
        validate_path(source, 17, path)
        assert ''.join(s['source'] for s in path[2]) == source
        assert path[2][0]['input_start'] == 17
        assert path[2][-1]['input_end'] == 17 + len(source)


@pytest.mark.parametrize('source', [
    '10kg', '60fps', '3d', 'v1.2.3', '192.168.0.1', '0xff', '1e10',
    'https://example.com/v2', 'user2@example.com', '`12il`', 'main.py:42',
    '12abc', 'model2test', 'sha256', '12hours', '5test', '4API',
    '3gaeeul', '3gaeeun', '3myeongneun', '3myeongga', '3gaeeuro',
    '(3gaewol', '3gaewol)', '3gaewol~', 'x/3gaewol', 'x_3gaewol',
    '3nyeon' * 5, '1' * 97, '(((3nyeon)))', '1,2nyeon',
])
def test_protected_or_out_of_grammar_sources_have_no_supplement(generator, source):
    assert generator.paths(source) == ()


@pytest.mark.parametrize('mutation', ['offset', 'gap', 'literal', 'romanization', 'output', 'score'])
def test_path_validation_rejects_invalid_proofs(generator, mutation):
    source = '20%garyang'
    path = copy.deepcopy(generator.paths(source, 3)[0])
    text, score, spans = path
    if mutation == 'offset': spans[0]['input_start'] += 1
    elif mutation == 'gap': spans[1]['input_start'] += 1
    elif mutation == 'literal': spans[0]['text'] = '21'
    elif mutation == 'romanization': spans[-1]['text'] = '다른말'
    elif mutation == 'output': text = '다른말'
    elif mutation == 'score': score = float('nan')
    with pytest.raises(ValueError):
        validate_path(source, 3, (text, score, spans))


@pytest.mark.parametrize('source,target', [('20%neun','20%는'), ('20%raneun','20%라는'),
                                         ('5‰eun','5‰은'), ('10kg~20kggaryang','10kg~20kg가량')])
def test_symbol_particles_and_embedded_ascii_units(generator, source, target):
    assert target in [p[0] for p in generator.paths(source)]


@pytest.mark.parametrize('source', ['20%eun', '20%iraneun', '5‰neun'])
def test_symbol_particle_agreement(generator, source):
    assert generator.paths(source) == ()


@pytest.mark.parametrize('source', ['10kgneun', '10kgeun', '120kmreul', '60fpseul'])
def test_ascii_unit_reading_dependent_particles_wait_for_reading_model(generator, source):
    assert generator.paths(source) == ()


@pytest.mark.parametrize('score', [True, 5., 7., 0.])
def test_score_proof_requires_exact_grammar_cost(generator, score):
    text, _, spans = generator.paths('20%garyang')[0]
    with pytest.raises(ValueError):
        validate_path('20%garyang', 0, (text, score, spans))


def test_atom_cap_is_independent_of_conversion_cap(generator):
    # One Korean conversion but more than sixteen numeric/range atoms.
    assert generator.paths('1~' * 8 + '1il') == ()


def test_counter_inventory_is_bounded_before_index_construction():
    with pytest.raises(ValueError, match='vocabulary'):
        NumericSpanGenerator([chr(0xAC00 + i) for i in range(1025)])
