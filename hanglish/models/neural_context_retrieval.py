"""Frozen context encoding and TRAIN-centered candidate exemplar retrieval."""
from collections import defaultdict

import numpy as np
import torch

from hanglish.models.retrieved_context_evidence import ContextBank, context_text


def mean_content(hidden, attention_mask, special_tokens_mask):
    valid = attention_mask.bool() & ~special_tokens_mask.bool()
    weights = valid.unsqueeze(-1).to(hidden.dtype)
    return (hidden*weights).sum(1)/weights.sum(1).clamp_min(1)


class NeuralContextBank(ContextBank):
    def __init__(self, rows, embeddings):
        if not rows or len({r['id'] for r in rows}) != len(rows):
            raise ValueError('nonempty unique TRAIN rows required')
        self.rows, self.embeddings = rows, embeddings
        self.texts = [context_text(row['context']) for row in rows]
        matrix = np.array([embeddings[row['id']] for row in rows], dtype=np.float64)
        if matrix.ndim != 2 or not np.isfinite(matrix).all():
            raise ValueError('invalid embedding matrix')
        self.center = matrix.mean(axis=0)
        self.vectors = self.normalize(matrix)
        self.buckets = defaultdict(list)
        for i, row in enumerate(rows):
            for candidate in set(row['expected']) & set(row['candidates']):
                self.buckets[candidate].append(i)

    def normalize(self, matrix):
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape[-1] != self.center.shape[0] or not np.isfinite(matrix).all():
            raise ValueError('invalid query embedding')
        nonempty = np.linalg.norm(matrix, axis=-1, keepdims=True) > 0
        centered = (matrix-self.center)*nonempty
        norm = np.linalg.norm(centered, axis=-1, keepdims=True)
        return np.divide(centered, norm, out=np.zeros_like(centered), where=norm>0)

    def evidence(self, row):
        text = context_text(row['context'])
        query = self.normalize(self.embeddings[row['id']])
        found = []
        for candidate in row['candidates']:
            indices = self.buckets.get(candidate, [])
            scores = self.vectors[indices] @ query
            matches = []
            for i, score in zip(indices, scores):
                other = self.rows[i]
                same_source = (row['id'] == other['id']
                               or (row.get('group_id') is not None and row['group_id'] == other['group_id'])
                               or bool(set(row.get('source_ids') or ()) & set(other.get('source_ids') or ())))
                if score <= 0 or self.texts[i] == text or same_source:
                    continue
                matches.append(dict(id=self.rows[i]['id'], similarity=float(score)))
            found.append(sorted(matches, key=lambda m: (-m['similarity'], m['id']))[:3])
        return found
