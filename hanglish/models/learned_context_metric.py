"""Train a spelling-sensitive metric over frozen context prototypes."""
import numpy as np
import torch

from hanglish.models.neural_context_retrieval import NeuralContextBank
from hanglish.models.retrieved_context_evidence import context_text


def prototype_inputs(bank_rows, query_rows, embeddings):
    bank = NeuralContextBank(bank_rows, embeddings)
    query, prototypes, query_indices, slots = [], [], [], []
    for j, row in enumerate(query_rows):
        raw_query = np.asarray(embeddings[row['id']], dtype=np.float64)
        query.append(raw_query-bank.center if np.any(raw_query) else np.zeros_like(raw_query))
        if not np.any(raw_query):
            continue
        text = context_text(row['context'])
        for slot, candidate in enumerate(row['candidates']):
            eligible = []
            for i in bank.buckets.get(candidate, ()):
                other = bank.rows[i]
                if (row['id'] == other['id'] or text == bank.texts[i]
                        or (row.get('group_id') is not None and row['group_id'] == other['group_id'])
                        or set(row.get('source_ids') or ()) & set(other.get('source_ids') or ())):
                    continue
                eligible.append(embeddings[other['id']])
            if eligible:
                prototypes.append(np.asarray(eligible, dtype=np.float64).mean(axis=0)-bank.center)
                query_indices.append(j)
                slots.append(slot)
    width = max(len(row['candidates']) for row in query_rows)
    dimension = len(bank.center)
    return dict(query=torch.tensor(np.asarray(query), dtype=torch.float32),
                prototypes=torch.tensor(np.asarray(prototypes).reshape(-1, dimension), dtype=torch.float32),
                query_indices=torch.tensor(query_indices, dtype=torch.long),
                slots=torch.tensor(slots, dtype=torch.long), width=width,
                center=torch.tensor(bank.center, dtype=torch.float64))


class LearnedContextMetric(torch.nn.Module):
    def __init__(self, dimension=768):
        super().__init__()
        self.projection = torch.nn.Linear(dimension, 64, bias=False)
        self.gain = torch.nn.Parameter(torch.zeros(()))

    def forward(self, values):
        project = lambda x: torch.nn.functional.normalize(torch.tanh(self.projection(x)), dim=-1)
        queries, prototypes = project(values['query']), project(values['prototypes'])
        cosine = (queries[values['query_indices']]*prototypes).sum(-1)
        residual = 4*torch.tanh(self.gain*cosine)
        output = residual.new_zeros((len(queries), values['width']))
        return output.index_put((values['query_indices'], values['slots']), residual)

    def export(self):
        return {k: v.detach().tolist() for k, v in self.state_dict().items()}

    @classmethod
    def restore(cls, state):
        model = cls(len(state['projection.weight'][0]))
        tensors = {k: torch.tensor(v, dtype=torch.float32) for k, v in state.items()}
        if any(not torch.isfinite(v).all() for v in tensors.values()):
            raise ValueError('nonfinite metric weights')
        model.load_state_dict(tensors, strict=True)
        return model.eval()
