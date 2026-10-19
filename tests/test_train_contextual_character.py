import copy

import pytest
import torch

from hanglish.models.contextual_character_residual import ContextualCharacterResidualRanker
from scripts.train_contextual_character import backward_listwise_chunks, score_chunks


def row():
    outputs = [('가', '나'), ('카', '나'), ('가', '다'), ('가', '타'), ('하', '가')]
    candidates = [{'text': a+b, 'segments': [
        {'input_start':0, 'input_end':1, 'source':'a', 'text':a},
        {'input_start':1, 'input_end':2, 'source':'b', 'text':b},
    ]} for a,b in outputs]
    return {'input':'ab', 'candidates':candidates, 'base_scores':[.2,-.4,.1,.8,-.1],
            'positive_indices':[0,2]}


def test_two_pass_multi_positive_gradients_match_unchunked_listwise_objective():
    torch.manual_seed(12)
    full = ContextualCharacterResidualRanker(byte_dim=4, hidden=5, head_hidden=6)
    with torch.no_grad():
        full.edge_head[-1].weight.normal_(0,.2)
    chunked = copy.deepcopy(full)
    data = row()
    scores = full(full.prepare(data['input'],data['candidates'],data['base_scores']))['scores']
    loss = torch.logsumexp(scores,0)-torch.logsumexp(scores[data['positive_indices']],0)
    loss.backward()
    replay_loss = backward_listwise_chunks(chunked,data,chunk_size=2)
    assert replay_loss == pytest.approx(float(loss.detach()),abs=1e-7)
    for (name,p),(other,q) in zip(full.named_parameters(),chunked.named_parameters()):
        assert name == other
        assert p.grad is not None and q.grad is not None
        assert torch.allclose(p.grad,q.grad,atol=1e-6,rtol=1e-5), name
    assert torch.allclose(score_chunks(chunked,data,2),scores,atol=1e-7,rtol=1e-7)


def test_missing_gold_rejected_and_chunk_size_must_be_positive():
    model = ContextualCharacterResidualRanker(byte_dim=3, hidden=4, head_hidden=5)
    data = row()
    data['positive_indices'] = []
    with pytest.raises(ValueError,match='positive row'):
        backward_listwise_chunks(model,data)
    with pytest.raises(ValueError,match='positive chunk'):
        score_chunks(model,data,0)
