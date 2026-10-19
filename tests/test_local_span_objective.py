from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from hanglish.core.source_correspondence import align
from hanglish.models.local_span_objective import forward_with_local_preferences
from hanglish.models.source_span_residual import FrozenCandidates, SourceSpanResidualRanker
from hanglish.models.span_supervision import extract_preferences


class FakeParent(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.linear = nn.Linear(4, 1)


def path(parts):
    source = ''.join(left for left, _ in parts)
    cursor = 0; segments = []
    for left, right in parts:
        segments.append({'input_start': cursor, 'input_end': cursor + len(left),
                         'source': left, 'text': right})
        cursor += len(left)
    return align(source, ''.join(right for _, right in parts), segments)


def fixture():
    torch.manual_seed(831)
    model = SourceSpanResidualRanker(
        SimpleNamespace(model=FakeParent(), tokenizer=None),
        byte_dim=8, source_hidden=4, segment_hidden=8)
    preferred = path([('test', 'test'), ('reul', '를'), (' ', ' '),
                      ('hamnida', '합니다')])
    dispreferred = path([('test', 'test'), ('reul', '를'), (' ', ' '),
                         ('hamnida', '함니다')])
    groups = [[preferred], [dispreferred]]
    texts = tuple(group[0].text for group in groups)
    length = max(map(len, texts))
    hidden = torch.randn(2, length, 4)
    token_spans = tuple(tuple((index, index + 1) if index < len(text) else None
                              for index in range(length)) for text in texts)
    features = FrozenCandidates(texts, torch.tensor([-.4, .7]), hidden, token_spans)
    preferences, _ = extract_preferences(preferred.source, texts, [0], groups)
    return model, preferred.source, groups, features, preferences


def test_nonzero_global_scores_are_bitwise_identical_and_source_gru_runs_once():
    model, source, groups, features, preferences = fixture()
    with torch.no_grad():
        for parameter in model.segment_head.parameters():
            parameter.uniform_(-.3, .3)
    expected = model(source, groups, features=features)
    calls = []
    handle = model.source_encoder.register_forward_hook(
        lambda module, inputs, output: calls.append(1))
    actual = forward_with_local_preferences(
        model, source, groups, features, preferences)
    handle.remove()
    assert calls == [1]
    for key in ('scores', 'base_scores', 'residuals'):
        assert torch.equal(actual[key], expected[key])
    assert actual['local_margins'].shape == (1,)
    assert actual['local_loss'].ndim == 0


def test_local_loss_reaches_head_and_source_encoder_after_nonzero_head():
    model, source, groups, features, preferences = fixture()
    with torch.no_grad():
        model.segment_head[-1].weight.fill_(.2)
        model.segment_head[-1].bias.fill_(.1)
    result = forward_with_local_preferences(
        model, source, groups, features, preferences)
    result['local_loss'].backward()
    assert model.segment_head[-1].weight.grad.abs().sum().item() > 0
    assert model.source_embedding.weight.grad.abs().sum().item() > 0
    assert any(parameter.grad is not None and parameter.grad.abs().sum().item() > 0
               for parameter in model.source_encoder.parameters())
    assert all(parameter.grad is None for parameter in model.parent.parameters())


def test_empty_preferences_have_differentiable_zero_without_changing_global_scores():
    model, source, groups, features, _ = fixture()
    with torch.no_grad():
        model.segment_head[-1].weight.fill_(.2)
    expected = model(source, groups, features=features)
    result = forward_with_local_preferences(model, source, groups, features, [])
    assert torch.equal(result['scores'], expected['scores'])
    assert result['local_margins'].shape == (0,)
    assert result['local_loss'].item() == 0
    assert result['local_loss'].requires_grad
    result['local_loss'].backward()


def test_duplicate_witness_is_rejected():
    model, source, groups, features, preferences = fixture()
    forged = deepcopy(preferences)
    forged[0]['witnesses'].append(deepcopy(forged[0]['witnesses'][0]))
    with pytest.raises(ValueError, match='duplicate preference witness'):
        forward_with_local_preferences(model, source, groups, features, forged)


def test_duplicate_candidate_path_is_rejected_before_weighting():
    model, source, groups, features, preferences = fixture()
    duplicated = [[groups[0][0], groups[0][0]], groups[1]]
    with pytest.raises(ValueError, match='duplicate candidate path'):
        forward_with_local_preferences(
            model, source, duplicated, features, preferences)


def test_witness_losses_and_margins_are_averaged_within_one_preference():
    torch.manual_seed(912)
    model = SourceSpanResidualRanker(
        SimpleNamespace(model=FakeParent(), tokenizer=None),
        byte_dim=8, source_hidden=4, segment_hidden=8)
    coarse = path([('testreul', 'test를'), (' ', ' ')])
    fine = path([('test', 'test'), ('reul', '를'), (' ', ' ')])
    negative = path([('testreul', '테스트를'), (' ', ' ')])
    groups = [[coarse, fine], [negative]]
    texts = (coarse.text, negative.text)
    length = max(map(len, texts))
    features = FrozenCandidates(texts, torch.tensor([.2, -.1]),
        torch.randn(2, length, 4),
        tuple(tuple((i, i + 1) if i < len(text) else None
                    for i in range(length)) for text in texts))
    preferences, _ = extract_preferences(coarse.source, texts, [0], groups)
    assert len(preferences) == 1 and len(preferences[0]['witnesses']) == 2
    with torch.no_grad():
        for parameter in model.segment_head.parameters():
            parameter.uniform_(-.4, .4)
    combined = forward_with_local_preferences(
        model, coarse.source, groups, features, preferences)
    singles = []
    for witness in preferences[0]['witnesses']:
        one = deepcopy(preferences)
        one[0]['witnesses'] = [deepcopy(witness)]
        singles.append(forward_with_local_preferences(
            model, coarse.source, groups, features, one))
    assert torch.allclose(combined['local_loss'],
                          torch.stack([item['local_loss'] for item in singles]).mean())
    assert torch.allclose(combined['local_margins'][0],
                          torch.stack([item['local_margins'][0] for item in singles]).mean())


@pytest.mark.parametrize(('mutation', 'message'), [
    (lambda value: value[0]['witnesses'][0]['preferred'].update(path_id='missing'),
     'witness path'),
    (lambda value: value[0]['witnesses'][0]['preferred'].update(span_indices=[99]),
     'span indices'),
    (lambda value: value[0].update(preferred_index=1), 'candidate indices'),
    (lambda value: value[0]['witnesses'][0]['preferred'].update(actions=['copy']),
     'side evidence'),
    (lambda value: value[0]['witnesses'][0]['preferred'].update(output_start=0),
     'side evidence'),
    (lambda value: value[0]['witnesses'][0]['preferred'].update(
        unresolved_ascii_output=True), 'side evidence'),
])
def test_malformed_preference_evidence_fails_closed(mutation, message):
    model, source, groups, features, preferences = fixture()
    forged = deepcopy(preferences); mutation(forged)
    with pytest.raises(ValueError, match=message):
        forward_with_local_preferences(model, source, groups, features, forged)


def test_candidate_identity_and_outside_producer_spans_are_validated():
    model, source, groups, features, preferences = fixture()
    wrong_features = FrozenCandidates(tuple(reversed(features.candidates)),
        features.logits, features.hidden, features.token_spans)
    with pytest.raises(ValueError, match='cached frozen'):
        forward_with_local_preferences(
            model, source, groups, wrong_features, preferences)

    forged = deepcopy(preferences)
    forged[0]['prefix'] = 'different'
    with pytest.raises(ValueError, match='output context'):
        forward_with_local_preferences(model, source, groups, features, forged)


def test_equal_outside_text_with_different_producer_spans_is_rejected():
    model = SourceSpanResidualRanker(
        SimpleNamespace(model=FakeParent(), tokenizer=None),
        byte_dim=8, source_hidden=4, segment_hidden=8)
    preferred = path([('a', 'A'), ('b', 'B'), ('x', '가')])
    dispreferred = path([('ab', 'AB'), ('x', '나')])
    groups = [[preferred], [dispreferred]]
    texts = (preferred.text, dispreferred.text)
    features = FrozenCandidates(texts, torch.zeros(2), torch.randn(2, 3, 4),
        (((0, 1), (1, 2), (2, 3)), ((0, 1), (1, 2), (2, 3))))
    preference = {
        'preference_id': 'outside-segmentation-differs',
        'input_start': 2, 'input_end': 3, 'source': 'x',
        'preferred_index': 0, 'dispreferred_index': 1,
        'prefix': 'AB', 'suffix': '',
        'preferred_text': '가', 'dispreferred_text': '나',
        'witnesses': [{
            'preferred': {'path_id': preferred.path_id, 'span_indices': [2],
                          'output_start': 2, 'output_end': 3, 'text': '가',
                          'actions': ['convert'], 'unresolved_ascii_output': False},
            'dispreferred': {'path_id': dispreferred.path_id, 'span_indices': [1],
                             'output_start': 2, 'output_end': 3, 'text': '나',
                             'actions': ['convert'], 'unresolved_ascii_output': False},
        }],
    }
    with pytest.raises(ValueError, match='outside producer spans'):
        forward_with_local_preferences(
            model, preferred.source, groups, features, [preference])
