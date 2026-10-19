from types import SimpleNamespace
import subprocess
import sys

import pytest

from hanglish.models.phonetic_likelihood import ordered_batches, score_pairs, validate_pairs


def test_empty_and_duplicate_pair_identity():
    assert validate_pairs([]) == []
    pairs = [("imnida", "입니다"), ("imnida", "임니다"), ("imnida", "입니다")]
    assert validate_pairs(pairs) == pairs
    assert score_pairs(SimpleNamespace(config=SimpleNamespace(max_roman_length=48),
                                      network=SimpleNamespace(training=False)), []) == []


@pytest.mark.parametrize("pair", [("Ga", "가"), ("a" * 49, "가"), ("a", "가" * 25),
                                  ("a", "ㄱ"), ("a", "test를"), ("", "가"), ("a", "")])
def test_unsupported_pairs_fail_without_truncation(pair):
    with pytest.raises(ValueError):
        validate_pairs([pair])


def test_batches_preserve_every_request_including_duplicate_pairs():
    pairs = [("gana", "가나"), ("ga", "가"), ("ga", "가"), ("imnida", "입니다")]
    assert ordered_batches(pairs, 2) == [[1, 2], [0, 3]]
    assert sorted(sum(ordered_batches(pairs, 3), [])) == list(range(4))
    for invalid in (True, 0, 257):
        with pytest.raises(ValueError):
            ordered_batches(pairs, invalid)


def test_training_mode_is_not_silently_changed():
    model = SimpleNamespace(config=SimpleNamespace(max_roman_length=48),
                            network=SimpleNamespace(training=True))
    with pytest.raises(ValueError, match="evaluation-mode"):
        score_pairs(model, [("ga", "가")])
    assert model.network.training is True


def test_import_does_not_load_neural_runtime():
    subprocess.run([sys.executable, "-c", "import sys; import hanglish.models.phonetic_likelihood; "
                    "assert 'torch' not in sys.modules; "
                    "assert 'hanglish.models.phonetic_transducer' not in sys.modules"], check=True)
