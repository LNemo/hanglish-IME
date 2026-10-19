"""Experimental candidate evidence from other labeled TRAIN contexts."""
from collections import Counter, defaultdict
import math

import torch

FEATURES = ('score', 'score_margin', 'base', 'max_similarity', 'mean_top3',
            'other_similarity', 'retrieval_margin', 'support_count', 'length', 'supported')


def context_text(context):
    left, marker, right = context.rpartition('[MASK]')
    text = left[-64:] + ' ' + right[:24] if marker else context[-88:]
    return ' '.join(text.split())


def grams(text):
    return {text[i:i+n] for n in (2, 3) for i in range(len(text)-n+1)
            if text[i:i+n].strip()}


def same_document(a, b):
    return (a['id'] == b['id'] or a['group_id'] == b['group_id']
            or bool(set(a.get('source_ids') or ()) & set(b.get('source_ids') or ())))


class ContextBank:
    def __init__(self, rows):
        if not rows or len({r['id'] for r in rows}) != len(rows):
            raise ValueError('nonempty unique TRAIN rows required')
        self.rows = rows
        self.texts = [context_text(row['context']) for row in rows]
        sets = [grams(t) for t in self.texts]
        df = Counter(g for gs in sets for g in gs)
        self.idf = {g: math.log((1+len(rows))/(1+count))+1 for g, count in df.items()}
        self.vectors = [self.vector(t) for t in self.texts]
        self.buckets = defaultdict(list)
        for i, row in enumerate(rows):
            for candidate in set(row['expected']) & set(row['candidates']):
                self.buckets[candidate].append(i)

    def vector(self, text):
        values = {g: self.idf[g] for g in grams(text) if g in self.idf}
        norm = math.sqrt(sum(v*v for v in values.values()))
        return {g: v/norm for g, v in values.items()} if norm else {}

    def evidence(self, row):
        text = context_text(row['context'])
        query = self.vector(text)
        found = []
        for candidate in row['candidates']:
            matches = []
            for i in self.buckets.get(candidate, ()):
                other = self.rows[i]
                if self.texts[i] == text or same_document(row, other):
                    continue
                vector = self.vectors[i]
                small, large = (query, vector) if len(query) < len(vector) else (vector, query)
                similarity = sum(v*large.get(g, 0.) for g, v in small.items())
                if similarity > 0:
                    matches.append(dict(id=other['id'], similarity=similarity))
            found.append(sorted(matches, key=lambda m: (-m['similarity'], m['id']))[:3])
        return found

    def features(self, row):
        evidence = self.evidence(row)
        maxima = [matches[0]['similarity'] if matches else 0. for matches in evidence]
        top = max(row['teacher_scores'])
        values = []
        for i, (candidate, score, base, matches) in enumerate(zip(
                row['candidates'], row['teacher_scores'], row['base_scores'], evidence)):
            other = max(maxima[:i]+maxima[i+1:], default=0.)
            values.append([score/10, (score-top)/10, base/10, maxima[i],
                           sum(m['similarity'] for m in matches)/3, other,
                           maxima[i]-other, math.log1p(len(matches))/math.log(4),
                           len(candidate)/10, float(bool(matches))])
        return values, evidence


class RetrievedContextHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = torch.nn.Linear(len(FEATURES), 8, dtype=torch.float64)
        self.output = torch.nn.Linear(8, 1, dtype=torch.float64)
        torch.nn.init.zeros_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    def forward(self, values):
        residual = 2*torch.tanh(self.output(torch.tanh(self.hidden(values))).squeeze(-1))
        return residual*values[..., -1]

    def export(self):
        return {k: v.detach().tolist() for k, v in self.state_dict().items()}

    @classmethod
    def restore(cls, state):
        model = cls()
        tensors = {k: torch.tensor(v, dtype=torch.float64) for k, v in state.items()}
        if any(not torch.isfinite(v).all() for v in tensors.values()):
            raise ValueError('nonfinite retrieval weights')
        model.load_state_dict(tensors, strict=True)
        return model.eval()
