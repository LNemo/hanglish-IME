"""Experimental residual scorer over frozen neural evidence and candidate origin."""
import math

import torch

KINDS = ('lexicon', 'morphology', 'neural', 'raw', 'other')
FEATURES = ('context_score', 'neutral_score', 'context_gain', 'context_margin',
            'neutral_margin', 'base_score', 'candidate_length', 'context_length', *KINDS)


def features(row, neutral):
    candidates = row['candidates']
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError('unique nonempty candidate list required')
    if any(len(row[key]) != len(candidates) for key in ('teacher_scores', 'base_scores', 'kinds')):
        raise ValueError('candidate feature lengths differ')
    context = row['context'].replace('[MASK]', '')
    context_length = min(len(' '.join(context.split())), 88) / 88
    top_context = max(row['teacher_scores'])
    top_neutral = max(neutral[c] for c in candidates)
    output = []
    for c, score, base, kind in zip(candidates, row['teacher_scores'], row['base_scores'], row['kinds']):
        prior = neutral[c]
        kind = kind if kind in KINDS else 'other'
        values = [score / 10, prior / 10, (score - prior) / 10,
                  (score - top_context) / 10, (prior - top_neutral) / 10,
                  base / 10, len(c) / 10, context_length]
        values += [float(kind == name) for name in KINDS]
        if not all(math.isfinite(v) for v in values):
            raise ValueError('nonfinite candidate evidence')
        output.append(values)
    return output


class CandidateEvidenceHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = torch.nn.Linear(len(FEATURES), 8, dtype=torch.float64)
        self.output = torch.nn.Linear(8, 1, dtype=torch.float64)
        torch.nn.init.zeros_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    def forward(self, values):
        return 2 * torch.tanh(self.output(torch.tanh(self.hidden(values))).squeeze(-1))

    def export(self):
        return {name: value.detach().tolist() for name, value in self.state_dict().items()}

    @classmethod
    def restore(cls, state):
        model = cls()
        expected = model.state_dict()
        if set(state) != set(expected):
            raise ValueError('evidence head state keys differ')
        values = {name: torch.tensor(value, dtype=torch.float64) for name, value in state.items()}
        if any(values[name].shape != expected[name].shape or not torch.isfinite(values[name]).all() for name in values):
            raise ValueError('invalid evidence head tensor')
        model.load_state_dict(values, strict=True)
        return model.eval()


def pack(rows, neutral):
    if not rows:
        raise ValueError('empty training partition')
    width = max(len(r['candidates']) for r in rows)
    values = torch.zeros((len(rows), width, len(FEATURES)), dtype=torch.float64)
    baseline = torch.full((len(rows), width), -1e9, dtype=torch.float64)
    valid = torch.zeros_like(baseline, dtype=torch.bool)
    positive = torch.zeros_like(valid)
    retention = torch.zeros(len(rows), dtype=torch.float64)
    for i, row in enumerate(rows):
        count = len(row['candidates'])
        values[i, :count] = torch.tensor(features(row, neutral), dtype=torch.float64)
        baseline[i, :count] = torch.tensor([s + .15 * b for s, b in zip(row['teacher_scores'], row['base_scores'])], dtype=torch.float64)
        valid[i, :count] = True
        positives = [j for j, c in enumerate(row['candidates']) if c in row['expected']]
        if row['kind'] == 'source' and not 0 < len(positives) < count:
            raise ValueError('uninformative source training row')
        if row['kind'] not in ('source', 'retention'):
            raise ValueError('unknown training role')
        positive[i, positives or list(range(count))] = True
        retention[i] = row['kind'] == 'retention'
    return values, baseline, valid, positive, retention


def objective(model, packed):
    values, baseline, valid, positive, retention = packed
    delta = model(values).masked_fill(~valid, 0)
    scores = baseline + delta
    ce = torch.logsumexp(scores, 1) - torch.logsumexp(scores.masked_fill(~positive, -1e9), 1)
    teacher = torch.softmax(baseline / 2, 1)
    kl = (teacher * (torch.log_softmax(baseline / 2, 1) - torch.log_softmax(scores / 2, 1))).sum(1)
    regularization = .03 * delta.square().sum(1) / valid.sum(1)
    return (ce + retention * 4 * kl + regularization).mean()


@torch.inference_mode()
def predict(model, rows, neutral):
    results = []
    for row in rows:
        if row.get('kind') == 'fixed':
            result, delta = row['fixed_result'], []
        else:
            delta = model(torch.tensor(features(row, neutral), dtype=torch.float64)).tolist()
            scores = [s + .15 * b + d for s, b, d in zip(row['teacher_scores'], row['base_scores'], delta)]
            choice = min(range(len(scores)), key=lambda i: (-scores[i], row['candidates'][i]))
            result = row['candidates'][choice]
        results.append(dict(id=row['id'], result=result, correct=isinstance(result, str) and result in row['expected'], delta=delta))
    return results
