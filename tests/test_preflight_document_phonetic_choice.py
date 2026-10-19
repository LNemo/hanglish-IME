import pytest

from scripts import preflight_document_phonetic_choice as probe


def test_byte_groups_cover_all_ids_balance_and_use_byte_length():
    rows = [{"id": f"id-{i:03}", "input": "x" * (i % 13 + 1)} for i in range(35)]
    rows += [{"id": "unicode", "input": "漢" * 10}]
    groups = probe.byte_probe_groups(rows)
    assert groups == probe.byte_probe_groups(list(reversed(rows)))
    assert len(groups) == 16
    assert max(g["count"] for g in groups) - min(g["count"] for g in groups) == 1
    assert sorted(i for g in groups for i in g["ids"]) == sorted(r["id"] for r in rows)
    assert groups[-1]["probe_id"] == "unicode"
    assert groups[-1]["max_utf8_bytes"] == 30


def test_groups_reject_too_few_sources():
    with pytest.raises(ValueError, match="few"):
        probe.byte_probe_groups([{"id": "one", "input": "one source"}])


def test_projection_distinguishes_paired_from_reused_generation():
    measured = [{"parent_seconds": 1., "learned_seconds": 1., "group_size": 1000}]
    result = probe.projection(100., measured)
    assert result["projected_paired_seconds"] == 2790.
    assert result["projected_shared_seconds"] == 1490.
    assert result["paired_qualified"] is False
    assert result["shared_qualified"] is True
    slower = [{**measured[0], "learned_seconds": 2.}]
    assert probe.projection(100., slower)["shared_qualified"] is False


def test_no_input_qualification_without_all_frozen_artifact_identities(monkeypatch):
    monkeypatch.setattr(probe, "INPUT_HASHES", {})
    monkeypatch.setattr(probe, "sha256_file", lambda _: pytest.fail("read before input hashes frozen"))
    with pytest.raises(ValueError, match="not yet frozen"):
        probe.authorize()
