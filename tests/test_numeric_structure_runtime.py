import threading

from hanglish.core.lattice import WordOption
from hanglish.core.neural_converter import NeuralConverter, beam_paths
from hanglish.core.numeric_structure import NumericSpanGenerator, sentence_extras
from scripts.probe_numeric_structure_runtime import build_extras, stable_union


def pieces(source='11wol29ilbuteo'):
    number = (source, 0, (WordOption(source, 0., 'raw'),))
    space = (' ', len(source), (WordOption(' ', 0., 'raw'),))
    learned = [number, space, ('ga', len(source) + 1, (
        WordOption('가', 2., 'neural'), WordOption('ga', -10000., 'raw')))]
    static = [number, space, ('ga', len(source) + 1, (WordOption('ga', 0., 'raw'),))]
    return learned, static


def test_production_paths_match_frozen_first_assignment_policy():
    generator = NumericSpanGenerator(('월', '일', '년', '원', '명', '주'))
    for source in ('11wol29ilbuteo', '2015nyeon~2017nyeon', '20%garyang',
                   '7000man~9000manwon', '601,730myeongigo', '10kg', 'v1.2.3'):
        learned, static = pieces(source)
        baseline = stable_union(beam_paths(learned), beam_paths(static))
        frozen, _, _ = build_extras(learned, static, baseline, generator)
        actual = sentence_extras(learned, static, baseline, generator)
        assert actual == frozen
        assert len(actual) <= 48


class Ranker:
    def __init__(self, rejected=None):
        self.rejected = rejected
        self.pool = None

    def can_score(self, text):
        return self.rejected is None or self.rejected not in text

    def score_pairs(self, contexts, candidates):
        self.pool = list(candidates)
        return [10. if '월' in text else 0. for text in candidates]


def converter(rejected=None):
    value = NeuralConverter.__new__(NeuralConverter)
    value.numeric_units = ('월', '일')
    value.numeric_structure = None
    value.lock = threading.RLock()
    value.sentence_ranker = Ranker(rejected)
    value.ranker_kind = 'hybrid'
    value.context_ranking = True
    value.version = 'test'
    value.load_models = lambda: None
    learned, static = pieces()
    def prepare(text, context, *, learned, raw):
        if raw:
            original = [(text, 0, (WordOption(text, 0., 'raw'),))]
            return original, original, [], False, False
        left, right = pieces()
        return left, right, [], False, False
    value._prepare_candidates = prepare
    return value


def test_numeric_paths_are_added_before_real_sentence_scoring():
    value = converter()
    response = value.convert('11wol29ilbuteo ga')
    left, right = pieces()
    baseline = stable_union(beam_paths(left), beam_paths(right))
    assert value.sentence_ranker.pool[:len(baseline)] == [path[0] for path in baseline]
    assert '11월29일부터 가' in value.sentence_ranker.pool
    assert response['result'].startswith('11월29일부터')
    generator = value.numeric_structure
    value.convert('11wol29ilbuteo ga')
    assert value.numeric_structure is generator


def test_unscoreable_numeric_extras_keep_original_sentence_pool():
    value = converter(rejected='월')
    value.convert('11wol29ilbuteo ga')
    left, right = pieces()
    assert value.sentence_ranker.pool == [path[0] for path in stable_union(beam_paths(left), beam_paths(right))]


def test_unscoreable_original_pool_never_enters_numeric_augmentation():
    value = converter(rejected='11wol')
    response = value.convert('11wol29ilbuteo ga')
    assert value.numeric_structure is None
    assert value.sentence_ranker.pool is None
    assert response['sentence_ranking_applied'] is False


def test_raw_and_hybrid_single_token_do_not_enter_numeric_augmentation():
    value = converter()
    response = value.convert('11wol29ilbuteo ga', raw=True)
    assert response['result'] == '11wol29ilbuteo ga'
    assert value.numeric_structure is None
    value.convert('11wol29ilbuteo')
    assert value.numeric_structure is None
