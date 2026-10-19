"""Experimental candidate/source/origin pair with an explicitly retained context slot."""
import torch

from hanglish.models.semantic_ranker import SemanticRanker

ENCODING = 'candidate-source-origin-masked-context-v1'
ORIGINS = {'lexicon': '사전', 'morphology': '활용', 'neural': '생성', 'raw': '원문'}


def candidate_ids(tokenizer, candidate, source, kind):
    if not isinstance(candidate, str) or not candidate or not isinstance(source, str) or not source:
        raise ValueError('nonempty candidate and original source required')
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    return encode(candidate) + [tokenizer.sep_token_id] + encode(source) + [tokenizer.sep_token_id] + encode(ORIGINS.get(kind, '기타'))


def encode_candidates(tokenizer, contexts, candidates, sources, kinds, max_length=128):
    if not len(contexts) == len(candidates) == len(sources) == len(kinds) or not candidates:
        raise ValueError('nonempty aligned structured inputs required')
    records = []
    for context, candidate, source, kind in zip(contexts, candidates, sources, kinds):
        if not isinstance(context, str):
            raise TypeError('context must be text')
        first = candidate_ids(tokenizer, candidate, source, kind)
        budget = max_length - len(first) - tokenizer.num_special_tokens_to_add(pair=True)
        if budget < 1:
            raise ValueError('candidate/source fields exceed token budget')
        left, separator, right = context.rpartition('[MASK]')
        if not separator:
            left, right = context, ''
        left_ids = tokenizer.encode(left, add_special_tokens=False)
        right_ids = tokenizer.encode(right, add_special_tokens=False)
        right_count = min(len(right_ids), 24, (budget - 1) // 3)
        left_count = min(len(left_ids), budget - 1 - right_count)
        right_count += min(len(right_ids) - right_count, budget - 1 - left_count - right_count)
        second = (left_ids[-left_count:] if left_count else []) + [tokenizer.mask_token_id] + right_ids[:right_count]
        ids = tokenizer.build_inputs_with_special_tokens(first, second)
        if len(ids) > max_length:
            raise ValueError('structured encoding exceeded token budget')
        records.append(dict(input_ids=ids,
                            token_type_ids=tokenizer.create_token_type_ids_from_sequences(first, second),
                            attention_mask=[1] * len(ids)))
    return tokenizer.pad(records, padding=True, return_tensors='pt')


class StructuredCandidateRanker:
    def __init__(self, ranker, max_length=128):
        self.model, self.tokenizer = ranker.model, ranker.tokenizer
        self.batch_size, self.max_length = ranker.batch_size, max_length

    @classmethod
    def from_parent(cls, path):
        return cls(SemanticRanker.load(path))

    @classmethod
    def load(cls, path):
        ranker = SemanticRanker.load(path)
        if ranker.metadata.get('encoding') != ENCODING:
            raise ValueError('structured candidate encoding metadata missing')
        return cls(ranker, ranker.max_length)

    def can_score(self, row):
        if not row['candidates'] or len(row['candidates']) != len(row['kinds']):
            raise ValueError('candidate origin alignment required')
        return all(len(candidate_ids(self.tokenizer, c, row['source'], k))
                   + self.tokenizer.num_special_tokens_to_add(pair=True) + 1 <= self.max_length
                   for c, k in zip(row['candidates'], row['kinds']))

    def encode_groups(self, rows):
        contexts, candidates, sources, kinds = [], [], [], []
        for row in rows:
            if len(row['candidates']) != len(row['kinds']):
                raise ValueError('candidate origins differ in length')
            contexts.extend([row['context']] * len(row['candidates']))
            sources.extend([row['source']] * len(row['candidates']))
            candidates.extend(row['candidates'])
            kinds.extend(row['kinds'])
        return encode_candidates(self.tokenizer, contexts, candidates, sources, kinds, self.max_length)

    @torch.inference_mode()
    def score_group(self, row):
        scores = []
        for start in range(0, len(row['candidates']), self.batch_size):
            chunk = dict(row, candidates=row['candidates'][start:start + self.batch_size],
                         kinds=row['kinds'][start:start + self.batch_size])
            encoded = self.encode_groups([chunk])
            scores.extend(float(v) for v in self.model(**encoded).logits.reshape(-1).cpu())
        return scores
