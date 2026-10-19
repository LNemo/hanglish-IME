import json

import pytest
import torch

from hanglish.models.contextual_character_residual import (
    ContextualCharacterResidualRanker,
)
from scripts import train_contextual_character as pilot
from scripts import train_contextual_character_full as full


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
    return {'text': ''.join(value for _, value in parts), 'segments': segments}


class Clock:
    def __init__(self, value=100.):
        self.value = value

    def __call__(self):
        return self.value


def test_full_resource_guard_uses_versioned_limits_and_records_evidence(tmp_path):
    clock = Clock()
    swap = [1000]
    rss = [1234]
    guard = full.FullResourceGuard(
        tmp_path,
        clock=clock,
        swap_reader=lambda: swap[0],
        rss_reader=lambda: rss[0],
    )
    clock.value += 12.5
    swap[0] += 256
    row = guard.observe('synthetic')
    assert row == {
        'stage': 'synthetic', 'seconds': 12.5,
        'rss_bytes': 1234, 'swap_growth_bytes': 256,
    }
    assert guard.peak_rss == 1234
    assert guard.peak_swap_growth == 256
    assert json.loads((tmp_path / 'resources.jsonl').read_text()) == row
    guard.require_row_time(clock.value - full.LIMITS['row_seconds'])
    with pytest.raises(ValueError, match='single row'):
        guard.require_row_time(clock.value - full.LIMITS['row_seconds'] - .01)


@pytest.mark.parametrize('kind', ('time', 'rss', 'swap'))
def test_full_resource_guard_rejects_each_limit_independently(tmp_path, kind):
    clock = Clock()
    swap = [0]
    rss = [0]
    guard = full.FullResourceGuard(
        tmp_path,
        clock=clock,
        swap_reader=lambda: swap[0],
        rss_reader=lambda: rss[0],
    )
    if kind == 'time':
        clock.value += full.LIMITS['whole_seconds'] + .01
    elif kind == 'rss':
        rss[0] = full.LIMITS['rss_bytes'] + 1
    else:
        swap[0] = full.LIMITS['swap_growth_bytes'] + 1
    with pytest.raises(ValueError, match='resource limit'):
        guard.observe(kind)


def test_frozen_recipe_and_full_denominators_are_exact():
    assert full.RECIPE is pilot.RECIPE
    assert full.RECIPE == {
        'epochs': 4,
        'learning_rate': .001,
        'weight_decay': .01,
        'clip_norm': 1.,
        'seed': 20260923,
        'threads': 2,
        'interop_threads': 1,
        'candidate_chunk': 32,
    }
    assert full.LIMITS == {
        'whole_seconds': 7200.,
        'row_seconds': 30.,
        'rss_bytes': 2 * 1024**3,
        'swap_growth_bytes': 512 * 1024**2,
    }
    assert full.backward_listwise_chunks is pilot.backward_listwise_chunks
    assert full.score_chunks is pilot.score_chunks
    assert full.EXPECTED_UPDATES == 6284

    records = []
    for index in range(1642):
        records.append({
            'first': index < 1453,
            'top5': index < 1529,
            'pool_covered': index < 1571,
            'no_pool': index >= 1629,
            'unsupported': 31 if index == 0 else 0,
        })
    assert full.summarize(records) == full.EXPECTED_BEFORE
    assert full.denominator_counts(records) == full.EXPECTED_COUNTS


def test_synthetic_epoch_updates_only_gold_present_rows(monkeypatch, tmp_path):
    rows = [
        {
            'input': 'ab',
            'candidates': [
                candidate('ab', [('a', '가'), ('b', '나')]),
                candidate('ab', [('a', '카'), ('b', '나')]),
            ],
            'base_scores': [0., 0.],
            'positive_indices': [0],
        },
        {
            'input': 'cd',
            'candidates': [candidate('cd', [('cd', '다')])],
            'base_scores': [0.],
            'positive_indices': [],
        },
    ]
    monkeypatch.setattr(full, 'rows_for', lambda authenticated: iter(rows))
    monkeypatch.setitem(full.EXPECTED_COUNTS, 'gold_present_rows', 1)
    model = ContextualCharacterResidualRanker(
        byte_dim=4, hidden=5, head_hidden=6)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=full.RECIPE['learning_rate'],
        weight_decay=full.RECIPE['weight_decay'],
    )
    clock = Clock()
    guard = full.FullResourceGuard(
        tmp_path,
        clock=clock,
        swap_reader=lambda: 0,
        rss_reader=lambda: 0,
    )
    before = {name: value.detach().clone()
              for name, value in model.state_dict().items()}
    mean_loss, updates = full.train_epoch(
        model, optimizer, authenticated={}, epoch=1, guard=guard)

    assert updates == 1
    assert mean_loss == pytest.approx(torch.log(torch.tensor(2.)).item())
    assert any(not torch.equal(before[name], value)
               for name, value in model.state_dict().items())
    stages = [json.loads(line)['stage'] for line in
              (tmp_path / 'resources.jsonl').read_text().splitlines()]
    assert stages == ['epoch-1:complete']
