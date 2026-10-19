from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


SCRIPT = (Path(__file__).resolve().parents[1] / ".omx/artifacts"
          / "verify-source-correspondence-fit-v1.py")
SPEC = importlib.util.spec_from_file_location("verify_source_correspondence_fit", SCRIPT)
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def candidate(parts):
    source = "".join(left for left, _ in parts)
    cursor = 0
    segments = []
    for left, right in parts:
        segments.append({"input_start": cursor, "input_end": cursor + len(left),
                         "source": left, "text": right})
        cursor += len(left)
    return {"text": "".join(right for _, right in parts),
            "segments": segments}, source


def record_for(candidates, source, reference=0):
    paths = [verifier.path_from_candidate(source, value) for value in candidates]
    record = verifier.compact(source, paths)
    record.update(id="row", component_id="component",
                  reference_path_id=paths[reference]["path_id"])
    return record, paths


def test_raw_output_with_distinct_boundaries_is_a_real_segmentation_contrast():
    reference, source = candidate([("test", "test"), ("reul", "를")])
    raw_fine, _ = candidate([("test", "test"), ("reul", "reul")])
    raw_coarse, _ = candidate([("testreul", "testreul")])
    record, paths = record_for([reference, raw_fine, raw_coarse], source)
    stats = verifier.describe(record, paths)
    assert len(record["paths"]) == 3
    assert stats["counts"]["contrasts"] == 2
    assert stats["counts"]["raw_contrasts"] == 2
    assert stats["counts"]["nonraw_contrasts"] == 0
    assert stats["counts"]["outputs_with_multiple_paths"] == 1
    assert stats["counts"]["segmentation_only_blocks"] == 0


def test_equal_nonraw_output_with_different_boundaries_is_retained_and_compared():
    coarse, source = candidate([("testreul", "test를")])
    fine, _ = candidate([("test", "test"), ("reul", "를")])
    record, paths = record_for([coarse, fine], source)
    stats = verifier.describe(record, paths)
    assert len(record["paths"]) == 2
    assert stats["counts"]["outputs_with_multiple_paths"] == 1
    assert stats["counts"]["segmentation_only_blocks"] == 1
    assert stats["counts"]["nonraw_contrasts"] == 1


def test_coarse_mixed_convert_remains_unresolved_ascii_not_invented_copy():
    coarse, source = candidate([("testreul", "test를")])
    fine, _ = candidate([("test", "test"), ("reul", "를")])
    record, paths = record_for([coarse, fine], source)
    stats = verifier.describe(record, paths)
    assert stats["counts"]["unresolved_ascii_span_occurrences"] == 1
    assert stats["counts"]["paths_with_unresolved_ascii"] == 1
    assert stats["counts"]["paths_with_copy_and_convert"] == 1


@pytest.mark.parametrize("field,value", [
    ("input_start", 1), ("input_end", 99), ("source", "wrong"), ("text", None),
])
def test_corrupt_source_edges_fail_closed(field, value):
    value_candidate, source = candidate([("ab", "압")])
    value_candidate["segments"][0][field] = value
    with pytest.raises(ValueError):
        verifier.path_from_candidate(source, value_candidate)


def test_compact_edge_or_path_tampering_is_rejected():
    value_candidate, source = candidate([("ab", "압")])
    record, _ = record_for([value_candidate], source)
    changed = deepcopy(record)
    changed["edges"][0]["text"] = "조작"
    with pytest.raises(ValueError):
        verifier.expand(changed, changed["paths"][0])
    changed = deepcopy(record)
    changed["paths"][0]["edge_indices"] = [True]
    with pytest.raises(ValueError, match="edge index"):
        verifier.expand(changed, changed["paths"][0])


def test_replay_detects_dropped_distinct_structural_path():
    coarse, source = candidate([("testreul", "test를")])
    fine, _ = candidate([("test", "test"), ("reul", "를")])
    expected, _ = record_for([coarse, fine], source)
    actual = deepcopy(expected)
    actual["paths"].pop()
    assert actual != expected


def test_component_cannot_cross_fit_and_hold_partition(monkeypatch):
    monkeypatch.setattr(verifier, "ROWS", 2)
    monkeypatch.setattr(verifier, "FIT_ROWS", 1)
    monkeypatch.setattr(verifier, "HOLD_ROWS", 1)
    manifest = {"schema": "recombination-fresh-train-sources-v1",
        "completed": True, "rows": 2, "input_contract": "roman_sentence_only",
        "targets_in_selection_artifact": False,
        "component_metadata_used_for_exclusion_and_partition": True,
        "model_loaded": False, "model_inference": False, "dev_accessed": False,
        "test_accessed": False, "production_changed": False,
        "selected_ids": ["a", "b"], "partition": [
            {"id": "a", "component_id": "shared", "split": "fit"},
            {"id": "b", "component_id": "shared", "split": "inner_hold"}]}
    proof = {"schema": "recombination-fresh-train-sources-independent-verification-v1",
        "verified": True,
        "manifest_sha256": verifier.FIXED[verifier.SOURCES / "manifest.json"],
        "selection_sha256": verifier.FIXED[verifier.SOURCES / "selection.jsonl"],
        "rows": 2, "partition_counts": {"fit": 1, "inner_hold": 1},
        "contracts": {"old_components_excluded": True,
            "selection_uses_accuracy": False,
            "prediction_input_fields": ["id", "input"]}}
    selected = [{"id": "a", "input": "a"}, {"id": "b", "input": "b"}]
    with pytest.raises(ValueError, match="leaks"):
        verifier.validate_partition(manifest, proof, selected)


def test_row_stats_tamper_changes_independent_result():
    first, source = candidate([("a", "아"), ("b", "b")])
    second, _ = candidate([("a", "아"), ("b", "비")])
    record, paths = record_for([first, second], source)
    expected = verifier.describe(record, paths)
    corrupted = deepcopy(expected)
    corrupted["counts"]["changed_source_characters"] += 1
    assert corrupted != expected


def test_full_row_replay_puts_occurrences_in_aggregated_counts():
    item, source = candidate([('test', 'test'), ('reul', '를')])
    selected = {'id': 'fit-1', 'input': source}
    assignment = {'id': 'fit-1', 'split': 'fit', 'component_id': 'component'}
    request = {'input': source, 'context': ''}
    runtime = {'id': 'fit-1', 'request': request, 'proposed': {
        'input': source, 'context': '', 'pipeline': {'pool': [item]},
        'candidates': [item], 'result': item['text']}}
    generated = {'id': 'fit-1', 'request': request, 'proposals': []}
    _, stats = verifier.expected_row(selected, assignment, runtime, generated)
    assert set(stats) == {'counts', 'maxima'}
    assert stats['counts']['input_path_occurrences'] == 2


def test_alternate_source_directory_is_not_authorized(tmp_path):
    with pytest.raises(ValueError, match='fixed authorization'):
        verifier.verify(tmp_path, data=tmp_path)
