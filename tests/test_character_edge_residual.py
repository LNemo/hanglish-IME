import dataclasses

import pytest
import torch

from hanglish.models.character_edge_residual import CharacterEdgeResidualRanker


def candidate(source, parts):
    cursor = 0
    segments = []
    for source_part, target_part in parts:
        segments.append({
            'input_start': cursor,
            'input_end': cursor + len(source_part),
            'source': source_part,
            'text': target_part,
        })
        cursor += len(source_part)
    assert cursor == len(source)
    return {'text': ''.join(part[1] for part in parts), 'segments': segments}


def test_zero_initialized_residual_preserves_float64_baseline_exactly():
    model = CharacterEdgeResidualRanker()
    candidates = [
        candidate('gana', [('ga', '가'), ('na', '나')]),
        candidate('gana', [('ga', '카'), ('na', '나')]),
    ]
    baseline = [0.123456789012345, -9.87654321098765]
    prepared = model.prepare('gana', candidates, baseline)
    result = model(prepared)
    assert prepared.base_scores.dtype == torch.float64
    assert result['scores'].dtype == torch.float64
    assert torch.equal(result['scores'], prepared.base_scores)
    assert torch.count_nonzero(result['residuals']).item() == 0
    assert all(tensor.device.type == 'cpu'
               for tensor in model._prepared_tensors(prepared))
    with pytest.raises(dataclasses.FrozenInstanceError):
        prepared.source = 'changed'


@pytest.mark.parametrize('change, message', [
    (lambda rows, scores: (rows + [rows[0]], scores + [0.]), 'distinct'),
    (lambda rows, scores: ([{**rows[0], 'label': 1}, rows[1]], scores),
     'only text and segments'),
    (lambda rows, scores: (rows, [float('nan'), 0.]), 'finite'),
    (lambda rows, scores: ([
        {**rows[0], 'segments': [{**rows[0]['segments'][0], 'input_start': 1}]
                                + rows[0]['segments'][1:]}, rows[1]], scores),
     'coverage'),
])
def test_malformed_candidates_and_alignment_are_rejected(change, message):
    model = CharacterEdgeResidualRanker()
    rows = [
        candidate('ab', [('a', '아'), ('b', '비')]),
        candidate('ab', [('a', '아'), ('b', '브')]),
    ]
    bad_rows, bad_scores = change(rows, [0., 0.])
    with pytest.raises(ValueError, match=message):
        model.prepare('ab', bad_rows, bad_scores)


def test_optional_segment_kind_is_validated_then_ignored_as_provenance():
    model = CharacterEdgeResidualRanker()
    plain = candidate('ab', [('a', '아'), ('b', '비')])
    with_kind = {
        'text': plain['text'],
        'segments': [{**segment, 'kind': 'captured'}
                     for segment in plain['segments']],
    }
    prepared = model.prepare('ab', [with_kind], [1.])
    assert prepared.candidate_texts == ('아비',)
    bad = {'text': plain['text'], 'segments': [
        {**plain['segments'][0], 'kind': 3}, plain['segments'][1]]}
    with pytest.raises(ValueError, match='segment fields'):
        model.prepare('ab', [bad], [1.])


def test_complete_utf8_targets_preserve_rare_bytes_without_unknown_or_truncation():
    model = CharacterEdgeResidualRanker(byte_dim=5, hidden=7, max_target_bytes=12)
    rare = '힣'
    other = '가'
    prepared = model.prepare('ab', [
        candidate('ab', [('a', rare), ('b', '')]),
        candidate('ab', [('a', other), ('b', '')]),
    ], [0., 0.])
    rare_index = prepared.edge_ids.index((0, 1, rare))
    other_index = prepared.edge_ids.index((0, 1, other))
    rare_length = prepared.target_byte_lengths[rare_index].item()
    assert prepared.target_bytes[rare_index, :rare_length].tolist() == [
        byte + 1 for byte in rare.encode('utf-8')]
    assert prepared.target_bytes.max().item() <= 256
    result = model(prepared)
    assert not torch.equal(result['edge_target_vectors'][rare_index],
                           result['edge_target_vectors'][other_index])
    too_small = CharacterEdgeResidualRanker(max_target_bytes=2)
    with pytest.raises(ValueError, match='target edge'):
        too_small.prepare('a', [candidate('a', [('a', rare)])], [0.])


def test_shared_edge_is_indexed_once_and_has_the_same_weighted_contribution():
    model = CharacterEdgeResidualRanker(byte_dim=4, hidden=6, head_hidden=5)
    prepared = model.prepare('abc', [
        candidate('abc', [('a', 'a'), ('bc', '갑')]),
        candidate('abc', [('a', 'a'), ('bc', '값')]),
    ], [0., 0.])
    shared_id = prepared.edge_ids.index((0, 1, 'a'))
    assert sum(edge == (0, 1, 'a') for edge in prepared.edge_ids) == 1
    with torch.no_grad():
        model.edge_head[-1].bias.fill_(2.)
    result = model(prepared)
    occurrences = torch.nonzero(
        prepared.path_edge_indices == shared_id, as_tuple=False).reshape(-1)
    assert occurrences.numel() == 2
    shared = result['edge_contributions'].index_select(0, occurrences)
    assert torch.equal(shared[0], shared[1])
    assert shared[0].item() == pytest.approx(2 / 3)
    for candidate_index in range(2):
        weights = prepared.path_weights[
            prepared.path_candidate_indices == candidate_index]
        assert weights.sum().item() == pytest.approx(1.)


def test_packed_targets_ignore_padding_and_empty_target_is_exact_zero():
    torch.manual_seed(2)
    model = CharacterEdgeResidualRanker(byte_dim=4, hidden=5)
    prepared = model.prepare('abc', [
        candidate('abc', [('a', '긴문자'), ('b', ''), ('c', 'x')]),
        candidate('abc', [('a', '다른말'), ('b', ''), ('c', 'y')]),
    ], [0., 0.])
    result = model(prepared)
    empty_index = prepared.edge_ids.index((1, 2, ''))
    assert torch.count_nonzero(result['edge_target_vectors'][empty_index]).item() == 0

    x_index = prepared.edge_ids.index((2, 3, 'x'))
    one_bytes = prepared.target_bytes[x_index:x_index + 1, :1]
    one_length = prepared.target_byte_lengths[x_index:x_index + 1]
    isolated = model._encode_targets(one_bytes, one_length)[0]
    assert torch.allclose(result['edge_target_vectors'][x_index], isolated)


def test_synthetic_same_baseline_training_updates_character_model_and_reduces_loss():
    torch.manual_seed(4)
    model = CharacterEdgeResidualRanker(byte_dim=6, hidden=8, head_hidden=8)
    prepared = model.prepare('ga', [
        candidate('ga', [('ga', '가')]),
        candidate('ga', [('ga', '카')]),
    ], [0., 0.])
    optimizer = torch.optim.Adam(model.parameters(), lr=.03)
    losses = []
    for _ in range(30):
        optimizer.zero_grad()
        scores = model(prepared)['scores']
        loss = -torch.log_softmax(scores, dim=0)[0]
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    final = model(prepared)
    assert losses[-1] < losses[0] * .1
    assert final['scores'][0] > final['scores'][1]
    assert model.target_embedding.weight.grad is not None
    assert model.target_embedding.weight.grad.abs().sum().item() > 0
    assert model.target_embedding.num_embeddings == 257


def test_configuration_bounds_and_architecture_parameter_counts():
    with pytest.raises(ValueError, match='byte_dim'):
        CharacterEdgeResidualRanker(byte_dim=True)
    with pytest.raises(ValueError, match='source exceeds byte'):
        CharacterEdgeResidualRanker(max_source_bytes=2).prepare(
            '한', [candidate('한', [('한', 'han')])], [0.])
    model = CharacterEdgeResidualRanker(byte_dim=3, hidden=4, head_hidden=5)
    expected = sum(parameter.numel() for parameter in model.parameters())
    assert model.architecture['parameter_count'] == expected
    assert model.architecture['trainable_parameter_count'] == expected
    assert model.architecture['target_encoding'].startswith('isolated_complete_utf8')
