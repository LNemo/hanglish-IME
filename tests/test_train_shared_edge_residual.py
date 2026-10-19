import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from scripts.preflight_source_span_residual import sha, tensor_hashes
from scripts.train_shared_edge_residual import (
    AUTH_SCHEMA,
    CACHE_BYTES,
    INVENTORY,
    LIMITS,
    RECIPE,
    TARGET_CHUNK,
    LogitCache,
    audit_forward,
    authenticate_inputs,
    consume_authorization,
    subset_targets,
    token_census,
)


def test_authenticated_inputs_reconstruct_the_fixed_fit_inventory():
    data, groups, targets, inventory, before, control, pins = authenticate_inputs()
    assert len(data) == len(inventory) == len(before) == len(control) == 1483
    assert len(targets) == 16814
    assert sum(len(row['edges']) for row in inventory) == 47282
    assert sum(len(row['candidates']) for row in data) == 11793
    assert [row['id'] for row in data] == [row['id'] for row in inventory]
    assert set(groups) == {row['id'] for row in data}
    required = {
        INVENTORY / 'report.json', INVENTORY / 'targets.json',
        INVENTORY / 'rows.jsonl', Path(__file__).parents[1] /
        'hanglish/models/shared_edge_residual.py',
    }
    assert all(str(path) in pins for path in required)


def test_prelaunch_authorization_must_exactly_bind_recipe_domain_and_pins(tmp_path):
    data, _, targets, _, _, _, pins = authenticate_inputs()
    path = tmp_path / 'authorization.json'
    authorization = {'schema': AUTH_SCHEMA, 'rows': len(data), 'targets': len(targets),
                     'recipe': RECIPE, 'pins': pins}
    path.write_text(json.dumps(authorization), encoding='utf-8')
    consumed = consume_authorization(path, pins)
    assert consumed == {**pins, str(path.resolve()): sha(path)}

    for field, value in [('schema', 'wrong'), ('rows', 1482), ('targets', 16813),
                         ('recipe', {**RECIPE, 'epochs': 3}),
                         ('pins', {**pins, 'invented': '0' * 64})]:
        changed = {**authorization, field: value}
        path.write_text(json.dumps(changed), encoding='utf-8')
        with pytest.raises(ValueError, match='authorization differs'):
            consume_authorization(path, pins)


def test_recipe_cache_and_resource_bounds_are_frozen_to_the_plan():
    assert RECIPE == {
        'seed': 20260921, 'epochs': 2, 'learning_rate': .001,
        'weight_decay': .01, 'clip_norm': 1., 'threads': 2,
        'dtype': 'float32', 'optimizer': 'AdamW', 'accumulation': 1,
    }
    assert 1483 * RECIPE['epochs'] == 2966
    assert LIMITS == {
        'whole_seconds': 1800., 'update_seconds': 60.,
        'rss_bytes': 3 * 1024**3, 'swap_growth_bytes': 512 * 1024**2,
    }
    assert TARGET_CHUNK == 256
    assert CACHE_BYTES == 256 * 1024**2


class CensusTokenizer:
    def __init__(self, reserved=3):
        self.reserved = reserved
        self.calls = []

    def num_special_tokens_to_add(self, pair):
        assert pair is True
        return self.reserved

    def __call__(self, text, *, add_special_tokens, truncation):
        assert add_special_tokens is False and truncation is False
        self.calls.append(text)
        return {'input_ids': list(range(len(text)))}


def test_token_census_normalizes_and_visits_every_target_and_full_candidate():
    tokenizer = CensusTokenizer()
    targets = [' target ', '  ', '한']
    data = [{'candidates': [' first  candidate ', 'second']},
            {'candidates': ['third']}]
    census = token_census(tokenizer, data, targets)
    assert tokenizer.calls == ['target', '', '한', 'first candidate', 'second', 'third']
    assert census['target_counts'] == [9, 3, 4]
    assert census['candidate_counts'] == [[18, 9], [8]]
    assert census['max_target_tokens'] == 9
    assert census['max_candidate_tokens'] == 18


def test_token_census_rejects_complete_text_over_256_without_truncation():
    tokenizer = CensusTokenizer(reserved=3)
    with pytest.raises(ValueError, match='exceeds256tokens'):
        token_census(tokenizer, [{'candidates': ['x']}], ['y' * 254])
    assert tokenizer.calls == ['y' * 254]


def test_logit_cache_reads_only_logits_and_returns_empty_hidden_placeholders(
        tmp_path, monkeypatch):
    import safetensors
    import scripts.train_shared_edge_residual as trainer

    old = tmp_path / 'old'
    features = old / 'features'
    features.mkdir(parents=True)
    tensor_path = features / '0000.safetensors'
    save_file({'logits': torch.tensor([.25, -.5]),
               'hidden': torch.ones(2, 400, 256)}, tensor_path)
    metadata_path = tensor_path.with_suffix('.json')
    metadata = {'candidates': ['a', 'b'], 'tensors_sha256': sha(tensor_path)}
    metadata_path.write_text(json.dumps(metadata), encoding='utf-8')
    manifest = {'0': {'tensors_sha256': sha(tensor_path),
                      'metadata_sha256': sha(metadata_path)}}
    (old / 'feature-manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    monkeypatch.setattr(trainer, 'OLD', old)

    opened = []
    real_safe_open = safetensors.safe_open

    class TrackedOpen:
        def __init__(self, *args, **kwargs):
            self.inner = real_safe_open(*args, **kwargs)

        def __enter__(self):
            self.stream = self.inner.__enter__()
            return self

        def __exit__(self, *args):
            return self.inner.__exit__(*args)

        def get_tensor(self, name):
            opened.append(name)
            return self.stream.get_tensor(name)

    monkeypatch.setattr(safetensors, 'safe_open', TrackedOpen)
    value = LogitCache().get(0, ['a', 'b'])
    assert opened == ['logits']
    assert torch.equal(value.logits, torch.tensor([.25, -.5]))
    assert value.hidden.numel() == 0 and value.token_spans == ()
    assert value.candidates == ('a', 'b')


def audited_fixture():
    edge_scores = torch.tensor([2., 4., 6.])
    first_weights = torch.tensor([.5, .5])
    second_weights = torch.tensor([.5, .5])
    first_values = edge_scores[[0, 1]] * first_weights
    second_values = edge_scores[[0, 2]] * second_weights
    result = {
        'edge_ids': ((0, 1, 'a'), (1, 2, 'x'), (1, 2, 'y')),
        'edge_scores': edge_scores,
        'path_audit': ((
            {'path_id': 'p1', 'edge_ids': (0, 1), 'weights': first_weights,
             'contributions': first_values, 'residual': first_values.sum()},
            {'path_id': 'p2', 'edge_ids': (0, 2), 'weights': second_weights,
             'contributions': second_values, 'residual': second_values.sum()},
        ),),
        'residuals': torch.stack((second_values.sum(),)),
        'winning_path_ids': (('p2',),),
    }
    row = {
        'edges': [
            {'input_start': 0, 'input_end': 1, 'text': 'a'},
            {'input_start': 1, 'input_end': 2, 'text': 'x'},
            {'input_start': 1, 'input_end': 2, 'text': 'y'},
        ],
        'paths': [[
            {'path_id': 'p1', 'edge_indices': [0, 1],
             'weight_numerators': [1, 1], 'weight_denominator': 2},
            {'path_id': 'p2', 'edge_indices': [0, 2],
             'weight_numerators': [1, 1], 'weight_denominator': 2},
        ]],
    }
    return result, row


def test_forward_audit_accepts_complete_shared_edge_evidence():
    result, row = audited_fixture()
    audit_forward(result, row)


@pytest.mark.parametrize('tamper, message', [
    (lambda result: result['path_audit'][0][1]['contributions'].__setitem__(
        0, torch.tensor(99.)), 'weighted contributions'),
    (lambda result: result['path_audit'][0][0]['weights'].__setitem__(
        0, torch.tensor(.25)), 'source-length weights'),
    (lambda result: result.__setitem__(
        'edge_ids', tuple(reversed(result['edge_ids']))), 'edge domain/order'),
    (lambda result: result.__setitem__(
        'edge_scores', torch.cat((result['edge_scores'], torch.ones(1)))), 'edge domain/order'),
    (lambda result: result['residuals'].__setitem__(0, torch.tensor(99.)), 'max path reduction'),
    (lambda result: result.__setitem__('winning_path_ids', (('p1',),)), 'max path reduction'),
])
def test_forward_audit_rejects_contribution_weight_and_domain_tampering(tamper, message):
    result, row = audited_fixture()
    tamper(result)
    with pytest.raises(ValueError, match=message):
        audit_forward(result, row)


def test_target_cache_roundtrip_is_complete_and_tamper_evident(tmp_path):
    state = {'vectors': torch.arange(24, dtype=torch.float32).reshape(6, 4),
             'has_tokens': torch.tensor([True, False, True, True, False, True])}
    path = tmp_path / 'target-features.safetensors'
    save_file(state, path)
    expected_hashes = tensor_hashes(state)
    restored = load_file(path)
    assert set(restored) == {'vectors', 'has_tokens'}
    assert tensor_hashes(restored) == expected_hashes
    row = {'edges': [{'target_index': 5}, {'target_index': 1}, {'target_index': 5}]}
    subset = subset_targets(row, tuple('abcdef'), restored['vectors'], restored['has_tokens'])
    assert subset.texts == ('b', 'f')
    assert torch.equal(subset.vectors, state['vectors'][[1, 5]])
    assert torch.equal(subset.has_tokens, state['has_tokens'][[1, 5]])
    restored['vectors'][0, 0] += 1
    assert tensor_hashes(restored) != expected_hashes
