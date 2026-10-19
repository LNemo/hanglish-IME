import torch
import pytest

from hanglish.models.structured_candidate_ranker import candidate_ids, encode_candidates


class Tokenizer:
    sep_token_id, mask_token_id, cls_token_id = 1, 2, 3

    def encode(self, text, add_special_tokens=False):
        return [ord(c) + 10 for c in text]

    def num_special_tokens_to_add(self, pair):
        return 3

    def build_inputs_with_special_tokens(self, a, b):
        return [3] + a + [1] + b + [1]

    def create_token_type_ids_from_sequences(self, a, b):
        return [0] * (len(a) + 2) + [1] * (len(b) + 1)

    def pad(self, rows, padding, return_tensors):
        length = max(len(r['input_ids']) for r in rows)
        return {key: torch.tensor([r[key] + [0] * (length - len(r[key])) for r in rows]) for key in rows[0]}


def test_fields_and_active_mask_survive_long_context():
    tokenizer = Tokenizer()
    first = candidate_ids(tokenizer, '낯', 'nat', 'lexicon')
    result = encode_candidates(tokenizer, ['가' * 200 + '끝[MASK]뒤' + '나' * 200], ['낯'], ['nat'], ['lexicon'], 32)
    ids, types = result['input_ids'][0].tolist(), result['token_type_ids'][0].tolist()
    assert ids[1:1 + len(first)] == first
    second = [v for v, t in zip(ids, types) if t == 1][:-1]
    assert 2 in second and second[second.index(2) - 1] == ord('끝') + 10
    assert second[second.index(2) + 1] == ord('뒤') + 10
    assert len(ids) == 32


def test_no_mask_context_gets_explicit_slot_at_end():
    result = encode_candidates(Tokenizer(), ['얼굴 설명'], ['낯'], ['nat'], ['lexicon'])
    ids = result['input_ids'][0].tolist()
    assert ids[-2:] == [2, 1]


def test_candidate_fields_never_silently_truncated():
    with pytest.raises(ValueError, match='token budget'):
        encode_candidates(Tokenizer(), ['[MASK]'], ['가' * 100], ['ga' * 100], ['neural'], 32)


def test_padding_and_short_context_use_spare_right_budget():
    tokenizer = Tokenizer()
    result = encode_candidates(tokenizer, ['[MASK]끝', 'a[MASK]' + '나' * 100], ['낯', '목'], ['nat', 'mok'], ['lexicon', 'neural'], 32)
    assert result['input_ids'].shape == (2, 32)
    assert result['attention_mask'][0].sum() < 32
    assert result['attention_mask'][1].sum() == 32
    assert (result['input_ids'][0] == 2).sum() == 1


def test_metadata_alignment_rejected():
    with pytest.raises(ValueError, match='aligned'):
        encode_candidates(Tokenizer(), ['[MASK]'], ['낯'], [], ['lexicon'])
