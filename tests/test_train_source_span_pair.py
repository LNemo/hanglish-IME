from types import SimpleNamespace

import pytest
import torch
from torch import nn

from hanglish.core.source_correspondence import align
from hanglish.models.source_span_residual import FrozenCandidates, SourceSpanResidualRanker
from scripts.train_source_span_pair import FeatureCache, install_arm, whole_source_hook


def test_whole_source_control_uses_mean_and_propagates_gradients_to_all_positions():
    states = torch.arange(24., requires_grad=True).reshape(1, 6, 4)
    states.retain_grad()
    final = torch.ones(2, 1, 2)
    pooled, returned_final = whole_source_hook(None, None, (states, final))
    assert returned_final is final
    assert torch.equal(pooled[:, 0], states.mean(1))
    assert torch.equal(pooled[:, 0], pooled[:, -1])
    pooled[:, 2].sum().backward()
    assert torch.allclose(states.grad, torch.full_like(states, 1 / 6))


def make_model():
    parent = nn.Linear(4, 1)
    parent.config = SimpleNamespace(hidden_size=4)
    return SourceSpanResidualRanker(SimpleNamespace(model=parent, tokenizer=None),
                                   byte_dim=8, source_hidden=4, segment_hidden=8)


def test_control_hook_changes_only_source_content_and_preserves_zero_initialization():
    model = make_model()
    source = 'ab'; texts = ('가나', '나가')
    groups = [[align(source, text, [dict(input_start=i, input_end=i + 1,
               source=source[i], text=text[i]) for i in range(2)])] for text in texts]
    feature = FrozenCandidates(texts, torch.tensor([.2, -.4]), torch.randn(2, 2, 4),
                               (((0, 1), (1, 2)), ((0, 1), (1, 2))))
    state = model.residual_state_dict()
    handle = install_arm(model, 'whole_source')
    assert torch.equal(model(source, groups, features=feature)['scores'], feature.logits)
    assert all(torch.equal(model.residual_state_dict()[k], v) for k, v in state.items())
    handle.remove()
    assert install_arm(model, 'source_local') is None
    with pytest.raises(ValueError, match='unknown'):
        install_arm(model, 'typo')


def test_feature_cache_roundtrips_offsets_and_bounds_memory(tmp_path):
    cache = FeatureCache(tmp_path / 'cache', limit=1)
    feature = FrozenCandidates(('a', 'b'), torch.tensor([1., 2.]), torch.randn(2, 2, 4),
                               ((None, (0, 1)), ((0, 1), None)))
    cache.put(0, feature); cache.put(1, feature)
    for index in (0, 1, 0):
        restored = cache.get(index)
        assert restored.candidates == feature.candidates
        assert restored.token_spans == feature.token_spans
        assert torch.equal(restored.hidden, feature.hidden)
        assert torch.equal(restored.logits, feature.logits)
        assert len(cache.memory) == 1
    with pytest.raises(ValueError, match='duplicate'):
        cache.put(0, feature)
    cache.get(1)  # Evict0 before validating disk again.
    (cache.directory / '0000.json').write_text('{}')
    with pytest.raises(ValueError, match='changed'):
        cache.get(0)


def test_feature_cache_disk_limit_rejects_growth(tmp_path, monkeypatch):
    import scripts.train_source_span_pair as producer
    monkeypatch.setattr(producer, 'CACHE_BYTES', 1)
    cache = FeatureCache(tmp_path / 'cache')
    feature = FrozenCandidates(('a',), torch.zeros(1), torch.zeros(1, 1, 4), (((0, 1),),))
    with pytest.raises(ValueError, match='disk bound'):
        cache.put(0, feature)
