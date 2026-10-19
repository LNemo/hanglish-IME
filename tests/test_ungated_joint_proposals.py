from copy import deepcopy

import pytest

from hanglish.core.ungated_joint_proposals import propose_joint


def segments(source, values):
    rows = []
    cursor = 0
    for source_text, output in values:
        rows.append({
            "input_start": cursor,
            "input_end": cursor + len(source_text),
            "source": source_text,
            "text": output,
            "kind": "raw" if source_text == output else "fixture",
        })
        cursor += len(source_text)
    assert cursor == len(source)
    return rows


def candidate(source, values, **extra):
    rows = segments(source, values)
    return {"text": "".join(row["text"] for row in rows),
            "segments": rows, **extra}


def two_word_fixture():
    source = "imnida gat"
    parent = candidate(source, [("imnida", "imnida"), (" ", " "),
                                ("gat", "gat")], support={"ignored": True})
    imnida = candidate(source, [("imnida", "입니다"), (" ", " "),
                                 ("gat", "gat")], parent_index=0)
    gat = candidate(source, [("imnida", "imnida"), (" ", " "),
                             ("gat", "갓")], parent_index=0)
    beams = {"imnida": [("입니다", -0.1), ("임니다", -0.2)],
             "gat": [("갇", -0.1), ("갓", -0.2), ("같", -0.3)]}
    return source, [parent], [imnida, gat], beams


def test_repaired_seed_recovers_two_spans_only_from_admitted_edges():
    source, parents, singles, beams = two_word_fixture()

    result = propose_joint(
        source, parents, singles, beams, seed_mode="roman_repaired")

    assert result["seeds"][0]["text"] == "입니다 갓"
    assert [row["replacement"] for row in result["seeds"][0]["replacements"]] == [
        "입니다", "갓"]
    assert result["proposals"][0]["text"] == "입니다 갓"
    assert result["proposals"][0]["origin"] == "ungated_joint_repair_seed"
    # Earlier beam values 임니다/갇 are valid, but their position-specific edges
    # are absent from this pool and therefore cannot enter the seed.
    assert "임니다" not in result["seeds"][0]["text"]
    assert "갇" not in result["seeds"][0]["text"]


def test_mixed_output_extension_repairs_roman_stem_and_preserves_default():
    source = 'gachireul imnida'
    parent = candidate(source, [('gachireul', 'gachi를'), (' ', ' '), ('imnida', '임니다')])
    singles = [
        candidate(source, [('gachireul', '가치를'), (' ', ' '), ('imnida', '임니다')]),
        candidate(source, [('gachireul', 'gachi를'), (' ', ' '), ('imnida', '입니다')]),
    ]
    beams = {'gachireul': [('가치를', -1.)], 'imnida': [('입니다', -1.)]}
    before = deepcopy((parent, singles, beams))
    old = propose_joint(source, [parent], singles, beams, seed_mode='roman_repaired')
    new = propose_joint(source, [parent], singles, beams, seed_mode='roman_repaired', repair_mixed=True)
    assert old['seeds'][0]['text'] == 'gachi를 임니다'
    assert new['seeds'][0]['text'] == '가치를 임니다'
    assert '가치를 입니다' in {p['text'] for p in new['proposals']}
    assert old['diagnostics']['edge_inventory'] == new['diagnostics']['edge_inventory']
    assert (parent, singles, beams) == before


def test_priority_false_and_original_arm_true_preserve_exact_output():
    source, parents, singles, beams = two_word_fixture()
    default = propose_joint(
        source, parents, singles, beams, seed_mode='roman_repaired', repair_mixed=True)
    explicit_false = propose_joint(
        source, parents, singles, beams, seed_mode='roman_repaired', repair_mixed=True,
        prioritize_beams=False)
    original_false = propose_joint(
        source, parents, singles, beams, seed_mode='original', repair_mixed=True)
    original_true = propose_joint(
        source, parents, singles, beams, seed_mode='original', repair_mixed=True,
        prioritize_beams=True)
    assert explicit_false == default
    assert original_true == original_false
    assert 'beam_priority_examined_count' not in default['diagnostics']
    assert 'beam_priority_examined_count' not in original_true['diagnostics']


def test_priority_exposes_rank_zero_edits_in_source_position_order():
    source = 'imnida gim gat nat'
    parent = candidate(source, [('imnida', 'imnida'), (' ', ' '), ('gim', 'gim'),
                                (' ', ' '), ('gat', '갇'), (' ', ' '), ('nat', '낟')])
    singles = [
        candidate(source, [('imnida', '입니다'), (' ', ' '), ('gim', 'gim'),
                           (' ', ' '), ('gat', '갇'), (' ', ' '), ('nat', '낟')]),
        candidate(source, [('imnida', 'imnida'), (' ', ' '), ('gim', '김'),
                           (' ', ' '), ('gat', '갇'), (' ', ' '), ('nat', '낟')]),
        candidate(source, [('imnida', 'imnida'), (' ', ' '), ('gim', 'gim'),
                           (' ', ' '), ('gat', '갓'), (' ', ' '), ('nat', '낟')]),
        candidate(source, [('imnida', 'imnida'), (' ', ' '), ('gim', 'gim'),
                           (' ', ' '), ('gat', '갇'), (' ', ' '), ('nat', '낫')]),
    ]
    beams = {'imnida': [('입니다', -0.1)], 'gim': [('김', -0.1)],
             'gat': [('갓', -0.1)], 'nat': [('낫', -0.1)]}

    result = propose_joint(
        source, [parent], singles, beams, seed_mode='roman_repaired',
        prioritize_beams=True)

    assert [row['text'] for row in result['proposals'][:3]] == [
        '입니다 김 갇 낟', '입니다 김 갓 낟', '입니다 김 갇 낫']
    assert [row['source_start'] for row in result['proposals'][1:3]] == [
        source.index('gat'), source.index('nat')]
    assert all(row['beam_index'] == 0 for row in result['proposals'][1:3])
    assert result['diagnostics']['beam_priority_examined_count'] == 2
    assert result['diagnostics']['beam_priority_retained_count'] == 2


def test_priority_skips_unadmitted_beam_edges_and_uses_first_admitted_rank():
    source = 'imnida gat'
    parent = candidate(source, [('imnida', 'imnida'), (' ', ' '), ('gat', '갇')])
    singles = [
        candidate(source, [('imnida', '입니다'), (' ', ' '), ('gat', '갇')]),
        candidate(source, [('imnida', 'imnida'), (' ', ' '), ('gat', '갓')]),
    ]
    beams = {'imnida': [('입니다', -0.1)],
             'gat': [('같', -0.1), ('갓', -0.2)]}

    result = propose_joint(
        source, [parent], singles, beams, seed_mode='roman_repaired',
        prioritize_beams=True)

    priority = [row for row in result['proposals']
                if row['origin'] == 'ungated_joint_beam_priority']
    assert [(row['text'], row['beam_index']) for row in priority] == [
        ('입니다 갓', 1)]
    assert all('같' not in row['text'] for row in result['proposals'])


def test_mixed_output_without_admitted_replacement_stays_preserved():
    source = 'gachireul'
    parent = candidate(source, [(source, 'gachi를')])
    result = propose_joint(source, [parent], [], {source: [('가치를', -1.)]},
                           seed_mode='roman_repaired', repair_mixed=True)
    assert result['seeds'][0]['text'] == 'gachi를'
    assert not result['seeds'][0]['replacements']


def test_original_and_repaired_arms_share_identical_edge_inventory():
    source, parents, singles, beams = two_word_fixture()
    before = deepcopy((parents, singles, beams))

    original = propose_joint(source, parents, singles, beams, seed_mode="original")
    repaired = propose_joint(
        source, parents, singles, beams, seed_mode="roman_repaired")

    assert original["diagnostics"]["edge_inventory"] == repaired[
        "diagnostics"]["edge_inventory"]
    assert original["diagnostics"]["edge_count"] == repaired[
        "diagnostics"]["edge_count"]
    assert (parents, singles, beams) == before


def test_unadmitted_beam_output_never_enters_seed_or_proposals():
    source, parents, singles, beams = two_word_fixture()
    beams["imnida"] = [("임니다", -0.1), ("입니다", -0.2)]

    result = propose_joint(
        source, parents, singles, beams, seed_mode="roman_repaired")

    assert result["seeds"][0]["text"] == "입니다 갓"
    assert all("임니다" not in row["text"] for row in result["proposals"])


def test_english_punctuation_and_missing_beam_segments_are_unchanged():
    source = "test, imnida!"
    parent = candidate(source, [("test", "test"), (", ", ", "),
                                ("imnida", "imnida"), ("!", "!")])
    single = candidate(source, [("test", "test"), (", ", ", "),
                                ("imnida", "입니다"), ("!", "!")])

    result = propose_joint(
        source, [parent], [single], {"imnida": [("입니다", -0.1)]},
        seed_mode="roman_repaired")

    assert result["seeds"][0]["text"] == "test, 입니다!"
    assert result["seeds"][0]["segments"][0]["text"] == "test"
    assert result["seeds"][0]["segments"][1]["text"] == ", "
    assert result["seeds"][0]["segments"][-1]["text"] == "!"

    prioritized = propose_joint(
        source, [parent], [single], {"imnida": [("입니다", -0.1)]},
        seed_mode="roman_repaired", repair_mixed=True, prioritize_beams=True)
    assert all(row['text'].startswith('test, ') and row['text'].endswith('!')
               for row in prioritized['proposals'])


def test_original_mode_uses_first_four_parents_without_repairs():
    source = "gim"
    parents = [candidate(source, [("gim", value)])
               for value in ("김", "gim", "김", "gim", "김")]
    result = propose_joint(
        source, parents, [], {"gim": [("김", -0.1)]}, seed_mode="original")
    assert len(result["seeds"]) == 4
    assert [row["text"] for row in result["seeds"]] == [
        "김", "gim", "김", "gim"]
    assert all(row["replacements"] == [] for row in result["seeds"])


def test_both_arms_obey_shared_96_total_proposal_bound():
    source = "gim " + " ".join(["gat"] * 7)
    values = [("gim", "김")]
    for index in range(7):
        values.extend([(" ", " "), ("gat", "gat")])
    parent = candidate(source, values)
    singles = []
    for word_index in range(7):
        for output in ("갇", "갓", "같"):
            changed = deepcopy(values)
            changed[1 + word_index * 2 + 1] = ("gat", output)
            singles.append(candidate(source, changed))
    beams = {"gat": [("갇", -0.1), ("갓", -0.2), ("같", -0.3)]}

    original = propose_joint(source, [parent], singles, beams)
    repaired = propose_joint(
        source, [parent], singles, beams, seed_mode="roman_repaired")

    assert len(original["proposals"]) == 96
    assert len(repaired["proposals"]) == 96
    assert repaired["diagnostics"]["repair_seed_retained_count"] == 1
    assert repaired["diagnostics"]["recombination_retained_count"] == 95
    assert repaired["diagnostics"]["recombination_generated_count"] == 96
    assert repaired["diagnostics"]["recombination_truncated_count"] == 1

    prioritized = propose_joint(
        source, [parent], singles, beams, seed_mode="roman_repaired",
        prioritize_beams=True)
    assert len(prioritized["proposals"]) == 96
    assert prioritized["diagnostics"]["repair_seed_retained_count"] == 1
    assert prioritized["diagnostics"]["beam_priority_examined_count"] == 14
    assert prioritized["diagnostics"]["beam_priority_retained_count"] == 14
    assert prioritized["diagnostics"]["recombination_retained_count"] == 81
    assert prioritized["diagnostics"]["recombination_generated_count"] == 96
    assert prioritized["diagnostics"]["recombination_truncated_count"] == 15


@pytest.mark.parametrize("mode", ["", "repaired", None, 1])
def test_invalid_seed_mode_fails_closed(mode):
    with pytest.raises(ValueError, match="seed_mode"):
        propose_joint("", [], [], {}, seed_mode=mode)


def test_invalid_beam_and_more_than_96_singles_fail_closed():
    source, parents, singles, beams = two_word_fixture()
    beams["gat"] = [("갓", -0.2), ("갇", -0.1)]
    with pytest.raises(ValueError, match="sorted"):
        propose_joint(source, parents, singles, beams)
    with pytest.raises(ValueError, match="exceeds 96"):
        propose_joint(source, parents, singles * 49, {})
