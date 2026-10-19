from copy import deepcopy

from scripts.boundary_proposal_policy import boundary_proposal_allowed


def candidate(source, pieces, score=1.0):
    offset = 0
    segments = []
    output = []
    for roman, text, kind in pieces:
        start = source.index(roman, offset)
        if start > offset:
            literal = source[offset:start]
            segments.append({
                "input_start": offset, "input_end": start,
                "source": literal, "text": literal, "kind": "raw",
            })
            output.append(literal)
        end = start + len(roman)
        segments.append({
            "input_start": start, "input_end": end,
            "source": roman, "text": text, "kind": kind,
        })
        output.append(text)
        offset = end
    if offset < len(source):
        literal = source[offset:]
        segments.append({
            "input_start": offset, "input_end": len(source),
            "source": literal, "text": literal, "kind": "raw",
        })
        output.append(literal)
    return {"text": "".join(output), "score": score, "segments": segments}


def provenance(source="gungnip", text="국립"):
    return {
        "source": source,
        "text": text,
        "left_lexeme": "국",
        "right_lexeme": "립",
        "left_key": "guk",
        "right_key": "rip",
        "boundary": "changed",
        "surface": "ngn",
        "suffix": None,
    }


def test_direct_novel_weak_span_is_accepted():
    source = "gungnip"
    parent = candidate(source, [(source, "궁닙", "neural")])
    extra = candidate(source, [(source, "국립", "morphology")])
    assert boundary_proposal_allowed(source, parent, extra, [provenance()])


def test_old_option_recombination_without_direct_provenance_is_rejected():
    source = "gungnip imnida"
    parent = candidate(source, [
        ("gungnip", "궁닙", "neural"), ("imnida", "임니다", "neural"),
    ])
    recombined = candidate(source, [
        ("gungnip", "국립", "morphology"), ("imnida", "임니다", "neural"),
    ])
    assert not boundary_proposal_allowed(source, parent, recombined, [])


def test_mixed_direct_and_unproven_changes_are_rejected():
    source = "gungnip imnida"
    parent = candidate(source, [
        ("gungnip", "궁닙", "neural"), ("imnida", "임니다", "neural"),
    ])
    mixed = candidate(source, [
        ("gungnip", "국립", "morphology"), ("imnida", "입니다", "morphology"),
    ])
    assert not boundary_proposal_allowed(source, parent, mixed, [provenance()])


def test_splitting_recognized_whole_lexeme_is_rejected():
    source = "gungnip"
    parent = candidate(source, [(source, "궁닙", "lexicon")])
    extra = candidate(source, [(source, "국립", "morphology")])
    assert not boundary_proposal_allowed(source, parent, extra, [provenance()])


def test_changed_protected_english_span_is_rejected():
    source = "test"
    parent = candidate(source, [(source, "test", "raw")])
    extra = candidate(source, [(source, "테스트", "morphology")])
    fake = {
        "source": source, "text": "테스트", "left_lexeme": "테",
        "right_lexeme": "스트", "left_key": "te", "right_key": "seuteu",
        "boundary": "changed", "surface": "st", "suffix": None,
    }
    assert not boundary_proposal_allowed(source, parent, extra, [fake])


def test_unchanged_english_span_is_preserved_while_direct_span_changes():
    source = "test gungnip"
    parent = candidate(source, [
        ("test", "test", "raw"), ("gungnip", "궁닙", "neural"),
    ])
    extra = candidate(source, [
        ("test", "test", "raw"), ("gungnip", "국립", "morphology"),
    ])
    assert boundary_proposal_allowed(source, parent, extra, [provenance()])


def test_suffix_provenance_accepts_compound_with_internal_boundary_change():
    source = "banbuguisigi"
    parent = candidate(source, [(source, "반부귀시기", "neural")])
    extra = candidate(source, [(source, "반북의식이", "morphology")])
    record = {
        "source": source,
        "text": "반북의식이",
        "left_lexeme": "반북",
        "right_lexeme": "의식",
        "left_key": "banbuk",
        "right_key": "uisik",
        "boundary": "changed",
        "surface": "gui",
        "suffix": "이",
        "suffix_surface": "gi",
        "suffix_restored": "k",
    }
    assert boundary_proposal_allowed(source, parent, extra, [record])


def test_malformed_offsets_are_rejected():
    source = "gungnip"
    parent = candidate(source, [(source, "궁닙", "neural")])
    extra = candidate(source, [(source, "국립", "morphology")])
    extra["segments"][0]["input_end"] -= 1
    assert not boundary_proposal_allowed(source, parent, extra, [provenance()])


def test_incomplete_or_forged_provenance_is_rejected():
    source = "gungnip"
    parent = candidate(source, [(source, "궁닙", "neural")])
    extra = candidate(source, [(source, "국립", "morphology")])
    for mutation in (
        lambda record: record.pop("right_key"),
        lambda record: record.update(left_key="gup"),
        lambda record: record.update(boundary="unchanged"),
        lambda record: record.update(text="궁닙"),
    ):
        record = deepcopy(provenance())
        mutation(record)
        assert not boundary_proposal_allowed(source, parent, extra, [record])


def test_policy_signature_has_no_outcome_inputs():
    names = boundary_proposal_allowed.__code__.co_varnames
    for forbidden in ("expected", "label", "target", "row_id"):
        assert forbidden not in names
