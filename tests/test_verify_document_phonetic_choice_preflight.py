import copy

import pytest

from scripts import verify_document_phonetic_choice_preflight as verifier
from tests.test_evaluate_production_phonetic_choice_dev import production_fixture


def requests(count=1046):
    return [{"id": f"row-{index:04d}",
             "input": "a " + ("x" * (index % 400 + 1))}
            for index in range(count)]


def test_balanced_utf8_groups_cover_every_id_once_and_select_longest():
    rows = requests()
    groups = verifier.byte_probe_groups(rows)
    assert len(groups) == 16
    assert sum(group["count"] for group in groups) == 1046
    assert sorted(identifier for group in groups for identifier in group["ids"]) \
        == sorted(row["id"] for row in rows)
    by_id = {row["id"]: row for row in rows}
    assert all(group["max_utf8_bytes"] == len(
        by_id[group["probe_id"]]["input"].encode("utf-8")) for group in groups)
    assert [group["count"] for group in groups].count(66) == 6
    assert [group["count"] for group in groups].count(65) == 10


def test_byte_sort_distinguishes_multibyte_literals():
    rows = [{"id": f"ascii-{index:02d}", "input": "a b"}
            for index in range(15)]
    rows += [{"id": "short-chars", "input": "a ㄱ"}]
    groups = verifier.byte_probe_groups(rows)
    assert groups[-1]["probe_id"] == "short-chars"
    assert groups[-1]["max_utf8_bytes"] == len("a ㄱ".encode("utf-8"))


def test_projection_recomputes_paired_and_shared_branches():
    timings = [{"parent_seconds": 1.0, "learned_seconds": 2.0,
                "group_size": 3} for _ in range(16)]
    result = verifier.projections(10.0, timings)
    assert result["projected_paired_seconds"] == pytest.approx(
        1.3 * (10 + 16 * 3 * 3) + 60)
    assert result["projected_shared_seconds"] == pytest.approx(
        1.3 * (10 + 16 * 2 * 3) + 60)
    assert result["paired_qualified"] is result["shared_qualified"] is True
    timings[0]["learned_seconds"] = 1000.0
    result = verifier.projections(10.0, timings)
    assert result["paired_qualified"] is result["shared_qualified"] is False


def test_parent_paths_reconstruct_public_candidates_and_reject_drift():
    source, frozen, _, paths = production_fixture()
    parent = frozen["parent"]
    parent.update(input=source, version=verifier.PRODUCTION_VERSION)
    verifier.verify_parent(source, parent, paths)
    damaged = copy.deepcopy(paths)
    damaged = list(reversed(damaged))
    with pytest.raises(ValueError, match="path/public"):
        verifier.verify_parent(source, parent, damaged)


def test_unranked_parent_requires_empty_full_paths():
    source, frozen, _, _ = production_fixture()
    parent = frozen["parent"]
    parent.update(input=source, version=verifier.PRODUCTION_VERSION,
                  sentence_ranking_applied=False)
    verifier.verify_parent(source, parent, [])
    with pytest.raises(ValueError, match="unranked"):
        verifier.verify_parent(source, parent, [["x", 0.0, []]])


def test_operation_grammar_is_exactly_53_boundaries():
    groups = verifier.byte_probe_groups(requests())
    operations = verifier.operation_names(groups)
    assert len(operations) == 53
    assert operations[:2] == ["authenticate_source_inputs", "load_models"]
    assert operations[-3:] == ["close_capture", "write_report", "write_completion"]
    assert sum(name.startswith("parent:") for name in operations) == 16
    assert sum(name.startswith("zero:") for name in operations) == 16
    assert sum(name.startswith("learned:") for name in operations) == 16


def test_unresolved_identities_block_verification_before_artifact_reads():
    with pytest.raises(ValueError, match="not frozen"):
        verifier.resolved_identities(verifier.UNRESOLVED, {
            name: verifier.UNRESOLVED for name in verifier.INPUT_HASHES})
    producer, inputs = verifier.resolved_identities()
    assert producer == verifier.EXPECTED_PRODUCER_SHA256
    assert inputs == verifier.INPUT_HASHES
    resolved = {name: str(index) * 64 for index, name in enumerate(
        verifier.INPUT_HASHES, 1)}
    producer, inputs = verifier.resolved_identities("a" * 64, resolved)
    assert producer == "a" * 64 and inputs == resolved


def test_verifier_source_has_no_runtime_or_neural_imports():
    source = verifier.Path(verifier.__file__).read_text()
    forbidden = (
        "import torch", "production_phonetic_choice_converter",
        "phonetic_choice_features", "preflight_document_phonetic_choice import",
    )
    assert all(value not in source for value in forbidden)
