import threading
from types import SimpleNamespace
import pytest
from hanglish.core.lattice import WordOption
from scripts.probe_constrained_runtime import extend_options


def test_preserves_parent_scores_order_and_bounds_new_candidates():
    baseline = (WordOption('각', 1., 'syllable'), WordOption('gak', 0., 'raw'))
    converter = SimpleNamespace(transducer=object(), lock=threading.RLock())
    result = extend_options(converter, 'gak', True, baseline,
                            lambda *a, **k: [(t, -1.) for t in ('각','갂','갃','갉','갘','갔')])
    assert result[:2] == baseline
    assert [o.text for o in result[2:]] == ['갂','갃','갉','갘']
    assert all(o.kind == 'neural' for o in result[2:])


@pytest.mark.parametrize('source,learned,score', [('gak',False,0), ('gak',True,5.5),
                                                   ('10kg',True,0), ('Test',True,0)])
def test_ineligible_inputs_do_not_generate(source, learned, score):
    baseline = (WordOption(source, score, 'raw'),)
    converter = SimpleNamespace(transducer=object())
    def forbidden(*args, **kwargs):
        pytest.fail('protected input reached generation')
    assert extend_options(converter, source, learned, baseline, forbidden) == baseline


def test_rejects_non_forward_valid_generation():
    converter = SimpleNamespace(transducer=object(), lock=threading.RLock())
    with pytest.raises(ValueError, match='invalid spelling'):
        extend_options(converter, 'gak', True, (), lambda *a, **k: [('가', -1.)])
