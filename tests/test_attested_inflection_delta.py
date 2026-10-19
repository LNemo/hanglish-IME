import pytest

from scripts.prepare_attested_inflection_delta import propose, quarantine


def base(surface, roman, role="train", roots=(), family=None):
    return {"surface": surface, "roman": roman, "split": role,
            "family_sha256": family or surface,
            "family_roots": [{"surface": s, "pos": p} for s, p in roots]}


def candidate(surface, roman):
    return {"surface": surface, "roman": roman, "split": "train"}


def test_only_exact_train_pos_anchor_and_attested_complete_ending():
    rows = [base("읽다", "ikda", roots=[("읽", "VV")]),
            base("먹다", "meokda", "lexical_holdout", [("먹", "VV")])]
    analyses = {
        "읽으니": {(('읽', 'VV'), ('으니', 'EC'))},
        "읽었": {(('읽', 'VV'), ('었', 'EP'))},
        "읽음": {(('읽', 'NNG'), ('음', 'ETN'))},
        "먹으니": {(('먹', 'VV'), ('으니', 'EC'))},
        "읽으면": {(('읽', 'VV'), ('으면', 'EC'))},
    }
    result, counts = propose(rows, analyses, set(analyses) - {"읽으면"},
                             lambda s: ('roman', s))
    assert [r['surface'] for r in result] == ["읽으니"]
    assert counts['proposed_new_pairs'] == 1


def test_authored_anchor_does_not_erase_pos():
    rows = [base("읽", "ik", roots=[("읽", "AUTHORED")])]
    result, _ = propose(rows, {"읽으니": {(('읽', 'VV'), ('으니', 'EC'))}},
                         {"읽으니"}, lambda s: ('ilgeuni', s))
    assert result == []


def test_existing_pairs_and_unsupported_lengths_are_not_added():
    rows = [base("읽으니", "ilgeuni", roots=[("읽", "VV")])]
    analyses = {s: {(('읽', 'VV'), ('으니', 'EC'))} for s in ['읽으니', '긴형태']}
    result, counts = propose(rows, analyses, set(analyses),
                             lambda s: ('ilgeuni', s) if s == '읽으니' else None)
    assert result == []
    assert counts['existing_pair'] == counts['outside_length_or_script_contract'] == 1


@pytest.mark.parametrize('role', ['lexical_holdout', 'morph_combination_holdout'])
def test_roman_collision_quarantines_both_hold_types(role):
    rows = [base('원형', 'shared', role)]
    accepted, rejected = quarantine(rows, [candidate('다른표기', 'shared')], {})
    assert accepted == []
    assert rejected[0]['held_splits'] == [role]


def test_second_lexical_root_and_alternative_analysis_bridge_are_closed():
    rows = [base('보류', 'held', 'lexical_holdout', [('먹', 'VV')])]
    additions = [candidate('복합활용', 'compound')]
    analyses = {'복합활용': {(('읽', 'VV'), ('으니', 'EC')),
                          (('읽', 'VV'), ('먹', 'VV'), ('다', 'EF'))}}
    accepted, rejected = quarantine(rows, additions, analyses)
    assert accepted == [] and len(rejected) == 1


def test_delta_bridge_quarantines_whole_component_not_only_direct_matches():
    rows = [base('보류', 'held', 'lexical_holdout', [('먹', 'VV')])]
    additions = [candidate('교량', 'same'), candidate('멀리', 'same')]
    analyses = {'교량': {(('먹', 'VV'), ('다', 'EF'))}}
    accepted, rejected = quarantine(rows, additions, analyses)
    assert accepted == [] and len(rejected) == 2
    assert len({r['component_sha256'] for r in rejected}) == 1


def test_original_family_and_authored_root_bridge_are_preserved():
    rows = [base('보류', 'held', 'morph_combination_holdout', family='group'),
            base('학습', 'train', roots=[('별칭', 'AUTHORED')], family='group')]
    additions = [candidate('학습', 'other')]
    accepted, rejected = quarantine(rows, additions, {})
    assert accepted == [] and len(rejected) == 1


def test_key_types_do_not_accidentally_connect_unrelated_families():
    rows = [base('보류', 'held', 'lexical_holdout', family='new')]
    additions = [candidate('새형태', 'new')]
    accepted, rejected = quarantine(rows, additions, {})
    assert len(accepted) == 1 and rejected == []


def test_order_does_not_change_component_ids_and_originals_are_immutable():
    rows = [base('학습', 'train', roots=[('읽', 'VV')])]
    additions = [candidate('읽으니', 'ilgeuni'), candidate('읽으면', 'ilgeumyeon')]
    analyses = {r['surface']: {(('읽', 'VV'), ('으니', 'EC'))} for r in additions}
    import copy
    before = copy.deepcopy((rows, additions, analyses))
    first, _ = quarantine(rows, additions, analyses)
    second, _ = quarantine(rows[::-1], additions[::-1], analyses)
    assert sorted(first, key=lambda r: r['surface']) == sorted(second, key=lambda r: r['surface'])
    assert (rows, additions, analyses) == before


def test_duplicates_and_unknown_roles_fail_closed():
    row = base('학습', 'train')
    with pytest.raises(ValueError, match='duplicate'):
        quarantine([row], [candidate('학습', 'train')], {})
    with pytest.raises(ValueError, match='unknown'):
        quarantine([{**row, 'split': 'typo'}], [], {})
