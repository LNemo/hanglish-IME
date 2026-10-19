from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from hanglish.core.source_correspondence import align
from hanglish.models.shared_edge_residual import (
    FrozenEdgeFeatures,
    SharedEdgeResidualRanker,
)
from hanglish.models.source_span_residual import FrozenCandidates, SourceSpanResidualRanker


class FakeParent(nn.Module):
    def __init__(self, hidden_size=4):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.linear = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(.8)


def make_model(batch_size=2):
    torch.manual_seed(7)
    ranker = SimpleNamespace(
        model=FakeParent(), tokenizer=None, batch_size=batch_size)
    return SharedEdgeResidualRanker(
        ranker, byte_dim=8, source_hidden=4, segment_hidden=8)


def make_path(source, pieces):
    segments = []
    cursor = 0
    for source_piece, output_piece in pieces:
        segments.append({
            'input_start': cursor,
            'input_end': cursor + len(source_piece),
            'source': source_piece,
            'text': output_piece,
        })
        cursor += len(source_piece)
    return align(source, ''.join(output for _, output in pieces), segments)


def fixture():
    model = make_model()
    source = 'abc'
    first = make_path(source, [('a', 'a'), ('b', 'X'), ('c', 'c')])
    second = make_path(source, [('a', 'a'), ('b', 'LONG'), ('c', 'c')])
    groups = [[first], [second]]
    candidates = tuple(group[0].text for group in groups)
    base = FrozenCandidates(
        candidates, torch.tensor([-.4, .7]),
        torch.randn(2, 99, 4), ((), ()))
    texts = ('a', 'X', 'c', 'LONG', 'unused')
    edge = FrozenEdgeFeatures(
        texts, torch.randn(len(texts), 4), torch.ones(len(texts), dtype=torch.bool))
    return model, source, groups, base, edge


def trainable_count(model):
    return sum(parameter.numel() for parameter in model.parameters()
               if parameter.requires_grad)


def test_inherits_exact_parent_weight_names_dimensions_and_zero_parity():
    model, source, groups, base, edge = fixture()
    original = SourceSpanResidualRanker(
        SimpleNamespace(model=FakeParent(), tokenizer=None, batch_size=2),
        byte_dim=8, source_hidden=4, segment_hidden=8)
    assert set(model.residual_state_dict()) == set(original.residual_state_dict())
    assert {key: value.shape for key, value in model.residual_state_dict().items()} == {
        key: value.shape for key, value in original.residual_state_dict().items()}
    assert trainable_count(model) == trainable_count(original)
    assert model.architecture['schema'] == 'shared-source-edge-residual-v1'
    result = model(source, groups, base_features=base, edge_features=edge)
    assert torch.equal(result['scores'], base.logits)
    assert torch.count_nonzero(result['residuals']) == 0
    assert all(not parameter.requires_grad for parameter in model.parent.parameters())

    production_shape = SharedEdgeResidualRanker(SimpleNamespace(
        model=FakeParent(hidden_size=256), tokenizer=None, batch_size=8))
    assert trainable_count(production_shape) == 71073


def test_common_edge_score_and_contribution_ignore_candidate_context_and_length():
    model, source, groups, base, edge = fixture()
    with torch.no_grad():
        model.segment_head[-1].weight.fill_(.2)
    source_calls = []
    handle = model.source_encoder.register_forward_hook(
        lambda _module, _arguments, _result: source_calls.append(True))
    try:
        result = model(source, groups, base_features=base, edge_features=edge)
    finally:
        handle.remove()
    assert source_calls == [True]
    a_id = result['edge_ids'].index((0, 1, 'a'))
    c_id = result['edge_ids'].index((2, 3, 'c'))
    left, right = result['path_audit'][0][0], result['path_audit'][1][0]
    for edge_id in (a_id, c_id):
        li = left['edge_ids'].index(edge_id)
        ri = right['edge_ids'].index(edge_id)
        assert torch.equal(left['contributions'][li], right['contributions'][ri])
    changed_hidden = replace(base, hidden=torch.randn(2, 1, 900))
    replay = model(source, groups, base_features=changed_hidden, edge_features=edge)
    assert torch.equal(result['edge_scores'], replay['edge_scores'])
    assert torch.equal(result['residuals'], replay['residuals'])


def test_candidate_order_is_invariant_with_same_superset_cache():
    model, source, groups, base, edge = fixture()
    with torch.no_grad():
        model.segment_head[-1].weight.fill_(.17)
        model.segment_head[-1].bias.fill_(.3)
    forward = model(source, groups, base_features=base, edge_features=edge)
    reverse_base = replace(
        base, candidates=tuple(reversed(base.candidates)), logits=base.logits.flip(0),
        hidden=base.hidden.flip(0), token_spans=tuple(reversed(base.token_spans)))
    reverse = model(source, list(reversed(groups)), base_features=reverse_base,
                    edge_features=edge)
    assert forward['edge_ids'] == reverse['edge_ids']
    assert torch.equal(forward['edge_scores'], reverse['edge_scores'])
    assert torch.equal(forward['scores'], reverse['scores'].flip(0))


def test_constant_edge_score_uses_length_weights_that_sum_to_one_across_splits():
    model = make_model()
    source = 'abc'
    coarse = make_path(source, [('abc', 'abc')])
    split = make_path(source, [('a', 'a'), ('b', 'b'), ('c', 'c')])
    groups = [[coarse, split]]
    base = FrozenCandidates(('abc',), torch.tensor([1.]), torch.empty(0), ())
    edge = FrozenEdgeFeatures(
        ('abc', 'a', 'b', 'c'), torch.zeros(4, 4), torch.ones(4, dtype=torch.bool))
    with torch.no_grad():
        for parameter in model.segment_head.parameters():
            parameter.zero_()
        model.segment_head[-1].bias.fill_(2.)
    result = model(source, groups, base_features=base, edge_features=edge)
    audits = result['path_audit'][0]
    assert all(item['weights'].sum().item() == pytest.approx(1.) for item in audits)
    assert all(item['residual'].item() == pytest.approx(2.) for item in audits)
    assert result['residuals'].item() == pytest.approx(2.)
    assert result['winning_path_ids'] == ((coarse.path_id, split.path_id)
                                          if coarse.path_id < split.path_id
                                          else (split.path_id, coarse.path_id),)


def test_edge_metadata_covers_copy_convert_ascii_unicode_and_empty_output():
    model = make_model()
    source = 'ab한z'
    path = make_path(source, [('ab', 'ab'), ('한', ''), ('z', 'A1')])
    groups = [[path]]
    base = FrozenCandidates((path.text,), torch.zeros(1), torch.empty(0), ())
    edge = FrozenEdgeFeatures(
        ('ab', '', 'A1'), torch.stack((torch.ones(4), torch.zeros(4), torch.ones(4))),
        torch.tensor([True, False, True]))
    captured = []
    handle = model.segment_head[0].register_forward_pre_hook(
        lambda _module, arguments: captured.append(arguments[0].detach().clone()))
    try:
        result = model(source, groups, base_features=base, edge_features=edge)
    finally:
        handle.remove()
    rows = {edge_id: captured[0][index, -8:]
            for index, edge_id in enumerate(result['edge_ids'])}
    assert rows[(0, 2, 'ab')][:4].tolist() == [1., 0., 0., 0.]
    assert rows[(2, 3, '')][:4].tolist() == [0., 1., 0., 1.]
    assert rows[(3, 4, 'A1')][:4].tolist() == [0., 1., 1., 0.]
    assert rows[(2, 3, '')][4:].tolist() == pytest.approx([1/4, 0., 2/4, 3/4])


def test_backward_reaches_shared_head_and_source_but_preserves_parent():
    model, source, groups, base, edge = fixture()
    with torch.no_grad():
        model.segment_head[-1].weight.fill_(.2)
    before = {name: value.clone() for name, value in model.parent.state_dict().items()}
    loss = -torch.log_softmax(
        model(source, groups, base_features=base, edge_features=edge)['scores'], 0)[0]
    loss.backward()
    assert model.segment_head[0].weight.grad.abs().sum().item() > 0
    assert model.source_embedding.weight.grad.abs().sum().item() > 0
    assert all(parameter.grad is None for parameter in model.parent.parameters())
    assert all(torch.equal(value, before[name])
               for name, value in model.parent.state_dict().items())


def test_extract_edge_features_is_isolated_bounded_and_overlap_weighted(monkeypatch):
    from hanglish.models import span_token_bridge

    calls = []
    model = make_model(batch_size=2)

    class EncodingParent(FakeParent):
        def forward(self, input_ids, **unused):
            calls.append(len(input_ids))
            hidden = input_ids.float()[:, :, None].expand(-1, -1, 4)
            return SimpleNamespace(hidden_states=(hidden,), logits=hidden[:, 0, :1])

    model.parent = EncodingParent()
    model.parent.eval()
    encoded = {'input_ids': torch.tensor([[99, 2, 4, 0], [99, 0, 0, 0],
                                          [99, 6, 8, 0]])}
    spans = [[None, (0, 1), (1, 2), None], [None] * 4,
             [None, (0, 1), (1, 2), None]]
    seen = []

    def fake_encode(tokenizer, texts, max_length):
        seen.append((texts, max_length))
        return encoded, spans

    monkeypatch.setattr(span_token_bridge, 'encode_aligned_candidates', fake_encode)
    features = model.extract_edge_features(('ab', '', '한글'))
    assert seen == [(['ab', '', '한글'], 256)]
    assert calls == [2, 1]
    assert features.has_tokens.tolist() == [True, False, True]
    assert features.vectors[:, 0].tolist() == [3., 0., 7.]
    assert not features.vectors.requires_grad


@pytest.mark.parametrize('mutate, message', [
    (lambda base, edge: (replace(base, candidates=('wrong', 'domain')), edge), 'base'),
    (lambda base, edge: (replace(base, logits=base.logits.double()), edge), 'base'),
    (lambda base, edge: (base, replace(edge, texts=('X', 'c', 'LONG', 'unused', 'z'))),
     'edge features'),
    (lambda base, edge: (base, replace(edge, vectors=edge.vectors.double())), 'edge features'),
    (lambda base, edge: (base, replace(edge, vectors=edge.vectors.clone().requires_grad_())),
     'edge features'),
    (lambda base, edge: (base, replace(
        edge, vectors=edge.vectors.clone().index_put((torch.tensor([0]),),
                                                     torch.tensor([[float('nan')]*4])))),
     'edge features'),
    (lambda base, edge: (base, FrozenEdgeFeatures(
        edge.texts, edge.vectors.clone().index_put((torch.tensor([0]),), torch.ones(1, 4)),
        torch.zeros(len(edge.texts), dtype=torch.bool))), 'token evidence'),
])
def test_malformed_feature_and_domain_contracts_are_rejected(mutate, message):
    model, source, groups, base, edge = fixture()
    bad_base, bad_edge = mutate(base, edge)
    with pytest.raises(ValueError, match=message):
        model(source, groups, base_features=bad_base, edge_features=bad_edge)


def test_duplicate_paths_do_not_add_count_weight_or_bonus():
    model, source, groups, base, edge = fixture()
    with torch.no_grad():
        model.segment_head[-1].bias.fill_(1.)
    original = model(source, groups, base_features=base, edge_features=edge)
    repeated = [[groups[0][0], groups[0][0], groups[0][0]], groups[1]]
    replay = model(source, repeated, base_features=base, edge_features=edge)
    assert torch.equal(original['scores'], replay['scores'])
    assert len(replay['path_audit'][0]) == 1


def test_edge_inventory_is_deterministic_and_validates_alignment_and_source_bound():
    model, source, groups, base, edge = fixture()
    expected = tuple(sorted(result for result in {
        (span.input_start, span.input_end, span.text)
        for group in groups for path in group for span in path.spans}))
    assert model.edge_inventory(source, list(reversed(groups))) == expected
    model.max_source_bytes = 2
    with pytest.raises(ValueError, match='byte bound'):
        model(source, groups, base_features=base, edge_features=edge)
