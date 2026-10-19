import pytest
import torch

from scripts.preflight_source_span_residual import train_loss
from scripts.train_local_span_objective import (
    LIMITS,
    OLD,
    PREFERENCES,
    RECIPE,
    VerifiedFeatureCache,
    authenticated_inputs,
    combined_loss,
)


def test_authenticated_inputs_are_the_verified_fixed_fit_domain():
    data, groups, labels, control, pins = authenticated_inputs()
    assert len(data) == len(labels) == len(control) == 1483
    assert set(groups) == {row['id'] for row in data}
    assert [row['id'] for row in data] == [row['id'] for row in labels]
    assert [row['id'] for row in data] == [row['id'] for row in control]
    assert sum(len(row['preferences']) for row in labels) == 6889
    assert all(row['schema'] == 'conditional-span-preference-v1' for row in labels)
    required = {
        PREFERENCES / 'report.json',
        PREFERENCES / 'independent-verification.json',
        PREFERENCES / 'preferences.jsonl',
        OLD / 'report.json',
        OLD / 'independent-verification.json',
        OLD / 'initial.safetensors',
        OLD / 'feature-manifest.json',
        OLD / 'source_local/residual.safetensors',
        OLD / 'source_local/after.jsonl',
    }
    assert all(str(path) in pins for path in required)


def test_combined_loss_is_exact_fixed_weight_sum_and_propagates_both_terms():
    scores = torch.tensor([.3, -.2, .7], requires_grad=True)
    local = torch.tensor(.8, requires_grad=True)
    total, candidate = combined_loss(scores, [0, 2], local)
    expected_candidate = train_loss(scores, [0, 2])
    assert torch.equal(candidate, expected_candidate)
    assert torch.equal(total, expected_candidate + .25 * local)
    total.backward()
    assert scores.grad is not None and scores.grad.abs().sum().item() > 0
    assert local.grad.item() == pytest.approx(.25)


def test_recipe_and_resource_limits_are_frozen_to_the_plan():
    assert RECIPE == {
        'seed': 20260921,
        'epochs': 2,
        'learning_rate': .001,
        'weight_decay': .01,
        'clip_norm': 1.,
        'threads': 2,
        'dtype': 'float32',
        'optimizer': 'AdamW',
        'accumulation': 1,
        'local_loss_weight': .25,
    }
    assert LIMITS == {
        'whole_seconds': 1800.,
        'update_seconds': 60.,
        'rss_bytes': 3 * 1024**3,
        'swap_growth_bytes': 512 * 1024**2,
    }


def test_historical_feature_cache_is_read_only_and_manifest_backed():
    cache = VerifiedFeatureCache()
    assert cache.directory == OLD / 'features'
    assert len(cache.entries) == 1483
    assert set(cache.entries) == set(range(1483))
    with pytest.raises(RuntimeError, match='read-only'):
        cache.put(0, object())
