import dataclasses

import pytest
import torch

from hanglish.models.character_edge_residual import CharacterEdgeResidualRanker
from hanglish.models.contextual_character_residual import (
    ContextualCharacterResidualRanker,
    PreparedContextualCharacterEdges,
)


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


def test_initialization_and_parameter_contract_matches_parent_exactly():
    torch.manual_seed(17)
    parent = CharacterEdgeResidualRanker()
    torch.manual_seed(17)
    contextual = ContextualCharacterResidualRanker()

    assert parent.state_dict().keys() == contextual.state_dict().keys()
    for name, value in parent.state_dict().items():
        assert torch.equal(value, contextual.state_dict()[name]), name
    assert contextual.architecture['parameter_count'] == 31_681
    assert contextual.architecture['trainable_parameter_count'] == 31_681
    assert contextual.architecture['target_encoding'].startswith('full_candidate_')


def test_prepare_keeps_parent_validation_and_exact_utf8_occurrence_offsets():
    model = ContextualCharacterResidualRanker(byte_dim=5, hidden=7)
    rows = [candidate('abc', [('a', '힣'), ('b', ''), ('c', 'x')])]
    prepared = model.prepare('abc', rows, [0.25])

    assert isinstance(prepared, PreparedContextualCharacterEdges)
    assert prepared.base.candidate_texts == ('힣x',)
    assert prepared.candidate_target_byte_lengths.tolist() == [4]
    assert prepared.candidate_target_bytes[0].tolist() == [
        *(byte + 1 for byte in '힣'.encode('utf-8')),
        ord('x') + 1,
    ]
    assert prepared.occurrence_target_byte_starts.tolist() == [0, 3, 3]
    assert prepared.occurrence_target_byte_ends.tolist() == [3, 3, 4]
    assert prepared.path_edge_indices.numel() == 3
    assert all(tensor.device.type == 'cpu' and not tensor.requires_grad
               for tensor in model._prepared_tensors(prepared))
    with pytest.raises(dataclasses.FrozenInstanceError):
        prepared.base = prepared.base


def test_zero_head_preserves_float64_baselines_and_empty_output_is_exact_zero():
    model = ContextualCharacterResidualRanker(byte_dim=4, hidden=6)
    rows = [candidate('ab', [('a', ''), ('b', '')])]
    baseline = [0.123456789012345]
    prepared = model.prepare('ab', rows, baseline)
    result = model(prepared)

    assert prepared.candidate_target_bytes.shape == (1, 0)
    assert result['scores'].dtype == torch.float64
    assert torch.equal(result['scores'], prepared.base_scores)
    assert torch.count_nonzero(result['residuals']).item() == 0
    assert torch.count_nonzero(result['occurrence_target_vectors']).item() == 0


def test_shared_edge_target_vector_changes_with_neighbor_output_context():
    torch.manual_seed(3)
    model = ContextualCharacterResidualRanker(
        byte_dim=5, hidden=7, head_hidden=6)
    prepared = model.prepare('abc', [
        candidate('abc', [('a', '가'), ('bc', '나다')]),
        candidate('abc', [('a', '가'), ('bc', '마라')]),
    ], [0., 0.])
    result = model(prepared)
    shared_edge = prepared.edge_ids.index((0, 1, '가'))
    occurrences = torch.nonzero(
        prepared.path_edge_indices == shared_edge, as_tuple=False).reshape(-1)

    assert occurrences.tolist() == [0, 2]
    vectors = result['occurrence_target_vectors'].index_select(0, occurrences)
    assert not torch.allclose(vectors[0], vectors[1])
    assert torch.equal(
        result['occurrence_source_vectors'][occurrences[0]],
        result['occurrence_source_vectors'][occurrences[1]],
    )


def test_single_span_output_matches_parent_isolated_target_encoding():
    torch.manual_seed(11)
    parent = CharacterEdgeResidualRanker(
        byte_dim=6, hidden=8, head_hidden=7)
    contextual = ContextualCharacterResidualRanker(
        byte_dim=6, hidden=8, head_hidden=7)
    contextual.load_state_dict(parent.state_dict())
    with torch.no_grad():
        parent.edge_head[-1].weight.normal_(mean=0., std=.2)
        parent.edge_head[-1].bias.fill_(.3)
        contextual.load_state_dict(parent.state_dict())

    rows = [candidate('gana', [('gana', '가나')])]
    parent_result = parent(parent.prepare('gana', rows, [1.25]))
    contextual_result = contextual(contextual.prepare('gana', rows, [1.25]))

    assert torch.equal(
        contextual_result['occurrence_target_vectors'][0],
        parent_result['edge_target_vectors'][0],
    )
    assert torch.equal(
        contextual_result['occurrence_source_vectors'][0],
        parent_result['edge_source_vectors'][0],
    )
    assert torch.equal(contextual_result['occurrence_scores'],
                       parent_result['edge_scores'])
    assert torch.equal(contextual_result['scores'], parent_result['scores'])


def test_candidate_permutation_only_permutes_candidate_results():
    torch.manual_seed(23)
    model = ContextualCharacterResidualRanker(
        byte_dim=5, hidden=7, head_hidden=6)
    with torch.no_grad():
        model.edge_head[-1].weight.normal_()
        model.edge_head[-1].bias.normal_()
    rows = [
        candidate('abc', [('a', '가'), ('bc', '나다')]),
        candidate('abc', [('a', '카'), ('bc', '마라')]),
    ]
    original = model(model.prepare('abc', rows, [.5, -.25]))
    permuted = model(model.prepare('abc', list(reversed(rows)), [-.25, .5]))

    assert torch.allclose(original['scores'], permuted['scores'].flip(0))
    original_vectors = original['occurrence_target_vectors'].reshape(2, 2, -1)
    permuted_vectors = permuted['occurrence_target_vectors'].reshape(2, 2, -1)
    assert torch.allclose(original_vectors, permuted_vectors.flip(0))


def test_contextual_model_backpropagates_finite_gradients():
    torch.manual_seed(29)
    model = ContextualCharacterResidualRanker(
        byte_dim=6, hidden=8, head_hidden=8)
    with torch.no_grad():
        model.edge_head[-1].weight.normal_(std=.1)
    prepared = model.prepare('testreul', [
        candidate('testreul', [('test', 'test'), ('reul', '를')]),
        candidate('testreul', [('test', '테스트'), ('reul', '를')]),
    ], [0., 0.])
    loss = -torch.log_softmax(model(prepared)['scores'], dim=0)[0]
    loss.backward()

    assert torch.isfinite(loss)
    for name in ('target_embedding.weight', 'target_encoder.weight_ih_l0',
                 'source_embedding.weight', 'edge_head.0.weight'):
        gradient = dict(model.named_parameters())[name].grad
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name
    assert model.target_embedding.weight.grad.abs().sum().item() > 0


def test_complete_candidate_bounds_and_parent_malformed_inputs_are_rejected():
    model = ContextualCharacterResidualRanker(max_target_bytes=3)
    too_long_whole = candidate('ab', [('a', 'x'), ('b', '가')])
    with pytest.raises(ValueError, match='complete candidate target'):
        model.prepare('ab', [too_long_whole], [0.])

    with pytest.raises(ValueError, match='only text and segments'):
        model.prepare('a', [{
            **candidate('a', [('a', '가')]),
            'expected': '가',
        }], [0.])
    with pytest.raises(ValueError, match='source exceeds byte'):
        ContextualCharacterResidualRanker(max_source_bytes=2).prepare(
            '한', [candidate('한', [('한', 'han')])], [0.])
    parent_prepared = CharacterEdgeResidualRanker().prepare(
        'a', [candidate('a', [('a', '가')])], [0.])
    with pytest.raises(ValueError, match='prepared contextual'):
        model(parent_prepared)
