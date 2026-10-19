from scripts.capture_ungated_phonetic_fit import nonraw_parents, support


class Tokenizer:
    def __call__(self, text, *, add_special_tokens, truncation):
        assert add_special_tokens is False and truncation is False
        return {'input_ids': text.split()}

    def num_special_tokens_to_add(self, *, pair):
        assert pair is True
        return 3


def test_support_boundary_keeps_unsupported_candidate_intact():
    candidate = {'text': '가 나 다', 'segments': [{'kind': 'lexicon'}]}
    assert support(candidate, Tokenizer(), 6)['sentence_support'] == 'supported'
    result = support(candidate, Tokenizer(), 5)
    assert result['sentence_support'] == 'unsupported'
    assert result['sentence_tokens'] == 6
    assert result['text'] == candidate['text'] and result['segments'] == candidate['segments']
    assert 'sentence_support' not in candidate


def test_mixed_parent_with_unknown_raw_span_is_kept():
    mixed = {'text': 'foo 입니다', 'segments': [{'kind': 'raw'}, {'kind': 'lexicon'}], 'score': 2}
    reserve = {'text': 'foo imnida', 'segments': [{'kind': 'raw'}], 'score': None}
    assert nonraw_parents([mixed, reserve]) == [{'text': mixed['text'], 'segments': mixed['segments']}]
