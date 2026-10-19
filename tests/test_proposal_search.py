from hanglish.core.lattice import WordOption
from hanglish.core.neural_converter import beam_paths
from scripts.probe_proposal_search import can_compose, classify, proposal_variants, single_alternatives, wider_beam


def fixture():
    return [('a', 0, (WordOption('가', 10., 'lexicon'), WordOption('까', 1., 'lexicon'), WordOption('a', -10000., 'raw'))),
            (' ', 1, (WordOption(' ', 0., 'raw'),)),
            ('b', 2, (WordOption('나', 10., 'lexicon'), WordOption('다', 9., 'lexicon')))]


def test_wider_beam_matches_production_at_same_width():
    pieces = fixture()
    assert wider_beam(pieces, 24) == beam_paths(pieces)


def test_exact_reachability_handles_ambiguous_boundaries_and_missing_spelling():
    pieces = [('a', 0, (WordOption('가', 1., 'lexicon'), WordOption('가나', 0., 'lexicon'))),
              ('b', 1, (WordOption('나다', 1., 'lexicon'), WordOption('다', 0., 'lexicon')))]
    assert can_compose('가나다', pieces)
    assert not can_compose('가나다라', pieces)
    assert can_compose('가나나', [('a', 0, pieces[0][2]), ('b', 1, (WordOption('나', 1., 'lexicon'),))])
    assert not can_compose('가나나', [('a', 0, pieces[0][2]), ('b', 1, (WordOption('나', 1., 'lexicon'),))], 1)


def test_single_position_alternatives_preserve_offsets_and_skip_suppressed_raw():
    paths = single_alternatives(fixture())
    assert [p[0] for p in paths] == ['가 다', '까 나']
    for text, _, segments in paths:
        assert ''.join(s['text'] for s in segments) == text
        assert ''.join(s['source'] for s in segments) == 'a b'
        assert [(s['input_start'],s['input_end']) for s in segments] == [(0,1),(1,2),(2,3)]


def test_variant_budgets_uniqueness_and_baseline_retention():
    pieces = [(str(i), i, tuple(WordOption(chr(0xac00+i*8+j), float(-j), 'lexicon') for j in range(8))) for i in range(10)]
    variants = proposal_variants(pieces, pieces)
    for name, paths in variants.items():
        texts = [p[0] for p in paths]
        assert len(texts) == len(set(texts))
        assert len(texts) <= (96 if name.endswith('96') else 48)
    assert {p[0] for p in variants['baseline48']} <= {p[0] for p in variants['augment96']}


def test_failure_classification_distinguishes_generation_and_pruning():
    pieces = fixture()
    assert classify(['까 나'], pieces, pieces, []) == 'beam_pruning'
    assert classify(['없는 말'], pieces, pieces, []) == 'missing_piece_candidate'
    assert classify(['까 나'], pieces, pieces, beam_paths(pieces)) == 'present'
    many = [('a', 0, tuple(WordOption(str(i), float(-i), 'lexicon') for i in range(9)))]
    assert classify(['8'], many, many, []) == 'option_cap'
