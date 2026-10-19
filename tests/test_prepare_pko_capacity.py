import pytest
from scripts.prepare_pko_capacity import donors, select


def test_selection_uses_category_group_id_not_target_or_order():
    rows = [{'id': str(i), 'category': 'a' if i < 10 else 'b', 'expected': ['gold']}
            for i in range(20)]
    groups = {r['id']: str(int(r['id'])//2) for r in rows}
    allowed = {r['id'] for r in rows} - {'0', '10'}
    ids = select(rows, allowed, groups, {'a': 3, 'b': 3})
    assert len(ids) == 6 and len({groups[i] for i in ids}) == 6
    assert set(ids) <= allowed
    assert ids == select([{**r, 'expected': ['changed']} for r in reversed(rows)],
                         allowed, groups, {'a': 3, 'b': 3})


def test_quota_shortage_fails():
    with pytest.raises(ValueError, match='insufficient'):
        select([{'id': 'a', 'category': 'x'}], {'a'}, {'a': 'group'}, {'x': 2})


def test_donors_are_different_and_bijective_within_length_bins():
    rows = [{'id': str(i), 'source': 'a'*i+str(i)} for i in range(64)]
    mapping = donors(rows)
    assert mapping == donors(list(reversed(rows)))
    assert all(mapping[r['id']]['shuffled_source'] != r['source'] for r in rows)
    for b in range(4):
        ids = {i for i, r in mapping.items() if r['length_bin'] == b}
        assert len(ids) == 16
        assert {mapping[i]['donor_id'] for i in ids} == ids
