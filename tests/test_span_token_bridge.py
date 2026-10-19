import pytest
import torch

from hanglish.models.sentence_ranker import encode_sentence_pairs
from hanglish.models.span_token_bridge import (
    encode_aligned_candidates,
    normalize_with_offsets,
    pooling_weights,
)


class _Batch(dict):
    def __init__(self, values, sequence_rows):
        super().__init__(values)
        self._sequence_rows = sequence_rows

    def sequence_ids(self, index):
        return list(self._sequence_rows[index])


class _FastTokenizer:
    is_fast = True

    def __init__(self, *, malformed_offset=None):
        self.malformed_offset = malformed_offset

    def num_special_tokens_to_add(self, pair=False):
        return 3 if pair else 2

    @staticmethod
    def _pieces(value):
        pieces = []
        start = 0
        while start < len(value):
            if value[start].isspace():
                start += 1
                continue
            end = start + 1
            while end < len(value) and not value[end].isspace():
                end += 1
            pieces.append((value[start:end], start, end))
            start = end
        return pieces

    def __call__(self, sentences, contexts=None, **kwargs):
        if isinstance(sentences, str):
            return {"input_ids": [100 + index for index, _ in enumerate(self._pieces(sentences))]}

        assert kwargs["truncation"] == "only_second"
        assert kwargs["max_length"] >= 4
        rows = []
        masks = []
        types = []
        offsets = []
        sequence_rows = []
        for sentence, context in zip(sentences, contexts):
            sentence_pieces = self._pieces(sentence)
            context_pieces = self._pieces(context)
            row = [1] + [100 + i for i in range(len(sentence_pieces))] + [2]
            row += [200 + i for i in range(len(context_pieces))] + [2]
            offset_row = [(0, 0)] + [(start, end) for _, start, end in sentence_pieces]
            offset_row += [(0, 0)] + [(start, end) for _, start, end in context_pieces] + [(0, 0)]
            sequence_row = [None] + [0] * len(sentence_pieces) + [None]
            sequence_row += [1] * len(context_pieces) + [None]
            rows.append(row)
            masks.append([1] * len(row))
            types.append([0] * (len(sentence_pieces) + 2) + [1] * (len(context_pieces) + 1))
            offsets.append(offset_row)
            sequence_rows.append(sequence_row)
        width = max((len(row) for row in rows), default=0)
        for row, mask, token_types, offset_row, sequence_row in zip(
                rows, masks, types, offsets, sequence_rows):
            padding = width - len(row)
            row.extend([0] * padding)
            mask.extend([0] * padding)
            token_types.extend([0] * padding)
            offset_row.extend([(0, 0)] * padding)
            sequence_row.extend([None] * padding)
        if self.malformed_offset is not None and offsets:
            offsets[0][1] = self.malformed_offset
        values = {
            "input_ids": torch.tensor(rows, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "token_type_ids": torch.tensor(types, dtype=torch.long),
        }
        if kwargs.get("return_offsets_mapping"):
            # Keep malformed scalar types intact so boundary tests exercise the
            # bridge rather than PyTorch's integer coercion.
            values["offset_mapping"] = offsets
        return _Batch(values, sequence_rows)


def test_normalization_maps_collapsed_whitespace_to_original_interval():
    normalized, offsets = normalize_with_offsets(" \talpha \n\u2003 beta  ")
    assert normalized == "alpha beta"
    assert offsets[:5] == [(2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]
    assert offsets[5] == (7, 11)
    assert offsets[6:] == [(11, 12), (12, 13), (13, 14), (14, 15)]


def test_unicode_offsets_are_python_code_points_and_keep_combining_marks():
    text = "  👩‍💻  e\u0301"
    normalized, offsets = normalize_with_offsets(text)
    assert normalized == "👩‍💻 e\u0301"
    assert len(normalized) == 6
    assert offsets[:3] == [(2, 3), (3, 4), (4, 5)]
    assert offsets[3] == (5, 7)
    assert offsets[4:] == [(7, 8), (8, 9)]


def test_aligned_encoding_matches_sentence_pair_ids_and_maps_tokens():
    tokenizer = _FastTokenizer()
    candidates = ["  👩‍💻  e\u0301 ", "한  글"]
    encoded, spans = encode_aligned_candidates(tokenizer, candidates, max_length=12)
    expected = encode_sentence_pairs(tokenizer, candidates, ["", ""], 12)
    assert torch.equal(encoded["input_ids"], expected["input_ids"])
    assert torch.equal(encoded["attention_mask"], expected["attention_mask"])
    assert torch.equal(encoded["token_type_ids"], expected["token_type_ids"])
    assert "offset_mapping" not in encoded
    assert spans == [
        [None, (2, 5), (7, 9), None, None],
        [None, (0, 1), (3, 4), None, None],
    ]


def test_empty_and_whitespace_candidates_only_have_unmapped_tokens():
    encoded, spans = encode_aligned_candidates(_FastTokenizer(), ["", " \t\n "], 8)
    assert encoded["input_ids"].shape == (2, 3)
    assert spans == [[None, None, None], [None, None, None]]
    assert pooling_weights(spans[0], 0, 0) == [0.0, 0.0, 0.0]


def test_pooling_weights_use_overlap_length_and_ignore_specials():
    spans = [None, (2, 5), (7, 9), None]
    assert pooling_weights(spans, 3, 8) == [0.0, 2.0, 1.0, 0.0]
    assert pooling_weights(spans, 5, 7) == [0.0, 0.0, 0.0, 0.0]
    assert pooling_weights(spans, 4, 4) == [0.0, 0.0, 0.0, 0.0]


def test_rejects_oversized_candidate_before_pair_encoding():
    with pytest.raises(ValueError, match="index 0.*token budget"):
        encode_aligned_candidates(_FastTokenizer(), ["a b c"], max_length=5)


@pytest.mark.parametrize("offset", [(-1, 1), (2, 1), (0, 99), (True, 1)])
def test_rejects_malformed_or_out_of_range_tokenizer_offsets(offset):
    with pytest.raises(ValueError, match="offset"):
        encode_aligned_candidates(
            _FastTokenizer(malformed_offset=offset), ["word"], max_length=8)


def test_validates_public_api_boundaries():
    with pytest.raises(TypeError, match="list"):
        encode_aligned_candidates(_FastTokenizer(), ("word",), 8)
    with pytest.raises(TypeError, match="string"):
        encode_aligned_candidates(_FastTokenizer(), [False], 8)
    with pytest.raises(ValueError, match="max_length"):
        encode_aligned_candidates(_FastTokenizer(), ["word"], True)
    with pytest.raises(ValueError, match="fast tokenizer"):
        encode_aligned_candidates(object(), ["word"], 8)
    with pytest.raises(TypeError, match="integers"):
        pooling_weights([], False, 1)
    with pytest.raises(ValueError, match="out of range"):
        pooling_weights([], 2, 1)
    with pytest.raises(ValueError, match="malformed span"):
        pooling_weights([[0, 1]], 0, 1)
    with pytest.raises(ValueError, match="out-of-range span"):
        pooling_weights([(False, 1)], 0, 1)
