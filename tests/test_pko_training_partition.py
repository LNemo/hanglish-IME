import pytest

from scripts.prepare_pko_training import partition


def row(ident, *, group=None, family=None, sources=(), source=None, target=None):
    return {"id": ident, "group_id": group or ident, "family": family or ident,
            "source_ids": list(sources), "input": source or ident + " source",
            "expected": [target or ident + " target"]}


def test_document_and_family_links_are_transitive_and_order_independent():
    rows = [row("a", sources=["doc1"]), row("b", sources=["doc1"], family="shared"),
            row("c", family="shared"), row("d")]
    groups = partition(rows)
    assert any(g["ids"] == ["a", "b", "c"] for g in groups)
    assert partition(list(reversed(rows))) == groups


def test_normalized_duplicate_input_or_any_target_cannot_cross_partition():
    rows = [row("a", source="Ａ  B"), row("b", source="a b", target="같은 문장"),
            row("c", target="다른 문장")]
    rows[2]["expected"].append("같은  문장")
    assert partition(rows)[0]["ids"] == ["a", "b", "c"]


def test_missing_group_and_duplicate_identity_rejected():
    with pytest.raises(ValueError, match="unique"):
        partition([row("a"), row("a")])
    example = row("a")
    example.pop("group_id")
    with pytest.raises(ValueError, match="grouping"):
        partition([example])
