from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from hanglish.core.source_correspondence import align
from hanglish.models.source_span_residual import FrozenCandidates, SourceSpanResidualRanker, source_byte_offsets


class FakeParent(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.linear = nn.Linear(4, 1)
        self.dropout = nn.Dropout(.8)


def fixture():
    torch.manual_seed(123)
    parent = SimpleNamespace(model=FakeParent(), tokenizer=None)
    model = SourceSpanResidualRanker(parent, byte_dim=8, source_hidden=4, segment_hidden=8)
    source = 'testreul hamnida'
    groups = []
    for ending in ('합니다', '함니다'):
        parts = [('test', 'test'), ('reul', '를'), (' ', ' '), ('hamnida', ending)]
        segments = []; cursor = 0
        for left, right in parts:
            segments.append(dict(input_start=cursor, input_end=cursor + len(left), source=left, text=right))
            cursor += len(left)
        groups.append([align(source, ''.join(b for _, b in parts), segments)])
    texts = tuple(g[0].text for g in groups)
    hidden = torch.randn(2, len(texts[0]), 4)
    features = FrozenCandidates(texts, torch.tensor([-.4, .7]), hidden,
        tuple(tuple((i, i + 1) for i in range(len(t))) for t in texts))
    return model, source, groups, features


def test_zero_residual_reproduces_parent_scores_exactly_and_parent_stays_frozen():
    model, source, groups, features = fixture()
    model.train()
    result = model(source, groups, features=features)
    assert torch.equal(result['scores'], features.logits)
    assert torch.count_nonzero(result['residuals']) == 0
    assert model.training and not model.parent.training
    assert all(not p.requires_grad for p in model.parent.parameters())


def test_real_backward_updates_source_and_head_but_not_frozen_parent():
    model, source, groups, features = fixture()
    before_parent = {k: v.clone() for k, v in model.parent.state_dict().items()}
    before_source = model.source_embedding.weight.detach().clone()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.05)
    for _ in range(3):
        optimizer.zero_grad()
        result = model(source, groups, features=features)
        loss = -torch.log_softmax(result['scores'], dim=0)[0]
        loss.backward()
        optimizer.step()
    assert not torch.equal(before_source, model.source_embedding.weight)
    assert all(torch.equal(v, before_parent[k]) for k, v in model.parent.state_dict().items())
    assert all(p.grad is None for p in model.parent.parameters())


def test_duplicate_paths_and_path_order_cannot_increase_candidate_score():
    model, source, groups, features = fixture()
    with torch.no_grad():
        model.segment_head[-1].weight.fill_(.2)
    coarse = align(source, groups[0][0].text, [dict(input_start=0, input_end=len(source),
                                                   source=source, text=groups[0][0].text)])
    base = [[groups[0][0], coarse], groups[1]]
    repeated = [[coarse, groups[0][0], coarse, groups[0][0]], groups[1]]
    assert torch.equal(model(source, base, features=features)['scores'],
                       model(source, repeated, features=features)['scores'])


def test_residual_roundtrip_preserves_scores_without_saving_parent_weights():
    model, source, groups, features = fixture()
    with torch.no_grad():
        model.segment_head[-1].weight.fill_(.3)
    expected = model(source, groups, features=features)['scores']
    state = model.residual_state_dict()
    assert not any(name.startswith('parent.') for name in state)
    fresh, _, _, _ = fixture()
    fresh.load_residual_state_dict(state)
    assert torch.equal(fresh(source, groups, features=features)['scores'], expected)
    with pytest.raises(ValueError, match='tensor names'):
        fresh.load_residual_state_dict({})


def test_cpu_residual_snapshot_is_not_a_live_parameter_alias():
    model, _, _, _ = fixture()
    state = model.residual_state_dict()
    expected = state['source_embedding.weight'].clone()
    with torch.no_grad():
        model.source_embedding.weight.add_(1.)
    assert torch.equal(state['source_embedding.weight'], expected)


def test_zero_initialization_delays_source_gradient_until_head_has_updated():
    model, source, groups, features = fixture()
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.1)
    for step in range(2):
        optimizer.zero_grad()
        loss = -torch.log_softmax(model(source, groups, features=features)['scores'], dim=0)[0]
        loss.backward()
        norm = model.source_embedding.weight.grad.abs().sum().item()
        assert (norm == 0) if step == 0 else (norm > 0)
        optimizer.step()


def test_scalar_path_max_distributes_tied_gradient():
    values = torch.tensor([0., 0.], requires_grad=True)
    values.max().backward()
    assert torch.equal(values.grad, torch.tensor([.5, .5]))


def test_cached_candidate_identity_and_gradient_contract_are_checked():
    model, source, groups, features = fixture()
    for wrong in [replace(features, candidates=tuple(reversed(features.candidates))),
                  replace(features, hidden=features.hidden.clone().requires_grad_())]:
        with pytest.raises(ValueError, match='cached frozen'):
            model(source, groups, features=wrong)


def test_source_bound_unicode_bytes_and_forged_output_offsets():
    assert source_byte_offsets('a😀한') == [0, 1, 5, 8]
    model, source, groups, features = fixture()
    model.max_source_bytes = 2
    with pytest.raises(ValueError, match='byte bound'):
        model(source, groups, features=features)
    model.max_source_bytes = 2048
    forged = replace(groups[0][0], spans=(replace(groups[0][0].spans[0], output_end=99),
                                         *groups[0][0].spans[1:]))
    with pytest.raises(ValueError, match='alignment'):
        model(source, [[forged], groups[1]], features=features)


def test_parent_feature_extraction_respects_batch_bound_and_output_order(monkeypatch):
    from hanglish.models import span_token_bridge

    calls = []
    class FeatureParent(FakeParent):
        def forward(self, input_ids, **unused):
            calls.append(len(input_ids))
            values = input_ids.float()
            return SimpleNamespace(logits=values[:, :1],
                                   hidden_states=(values[:, :, None].expand(-1, -1, 4),))

    encoded = {'input_ids': torch.arange(15).reshape(5, 3)}
    monkeypatch.setattr(span_token_bridge, 'encode_aligned_candidates',
                        lambda *args: (encoded, [[(0, 1), (1, 2), None] for _ in range(5)]))
    parent = SimpleNamespace(model=FeatureParent(), tokenizer=None, batch_size=2)
    model = SourceSpanResidualRanker(parent, byte_dim=8, source_hidden=4, segment_hidden=8)
    features = model.extract_features(['aa', 'bb', 'cc', 'dd', 'ee'])
    assert calls == [2, 2, 1]
    assert features.logits.tolist() == [0., 3., 6., 9., 12.]
    assert features.hidden.shape == (5, 3, 4)
