import pytest

from hanglish.models.roman_sentence_ranker import encode_roman_pairs


class Tokenizer:
    def __init__(self, length=8):
        self.length = length
        self.seen = None

    def __call__(self, candidates, sources, **kwargs):
        self.seen = (candidates, sources, kwargs)
        return {'input_ids': [[1]*self.length for _ in sources]}

    def pad(self, encoded, **kwargs):
        return encoded


def test_both_whole_sequences_are_encoded_without_truncation():
    tokenizer = Tokenizer()
    encode_roman_pairs(tokenizer, ['jeoneun haksaengimnida'], ['저는 학생입니다'])
    assert tokenizer.seen == (['저는 학생입니다'], ['jeoneun haksaengimnida'],
                              {'padding': False, 'truncation': False})


@pytest.mark.parametrize('source', ['저는 haksaengimnida', 'ᄀ haksaengimnida', 'ㄱ haksaengimnida', 'imnida', ''])
def test_external_korean_and_incomplete_sources_rejected(source):
    with pytest.raises(ValueError):
        encode_roman_pairs(Tokenizer(), [source], ['저는 학생입니다'])


def test_long_pair_rejected_instead_of_truncated():
    with pytest.raises(ValueError, match='token budget'):
        encode_roman_pairs(Tokenizer(385), ['jeoneun haksaengimnida'], ['저는 학생입니다'])


def test_mismatched_pairs_rejected():
    with pytest.raises(ValueError):
        encode_roman_pairs(Tokenizer(), ['jeoneun haksaengimnida'], [])
