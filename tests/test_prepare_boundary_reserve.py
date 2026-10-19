from copy import deepcopy
import json

import pytest

from scripts import prepare_boundary_reserve as module


def fixtures():
    metadata = [
        {"id": "a1", "group_id": "a"},
        {"id": "a2", "group_id": "a"},
        {"id": "b1", "group_id": "b"},
        {"id": "c1", "group_id": "c"},
        {"id": "c2", "group_id": "c"},
        {"id": "d1", "group_id": "d"},
    ]
    v3 = {
        "remaining_ids": ["a1", "a2", "b1", "c1", "c2", "d1"],
        "remaining_group_ids": ["c", "a", "d", "b"],
    }
    exposed = [{"id": "a1", "group_id": "a"}, {"id": "a2", "group_id": "a"}]
    return metadata, v3, exposed


def test_selects_complete_groups_in_frozen_v3_order_and_is_disjoint():
    args = fixtures()
    before = deepcopy(args)
    selected = module.select_ids(*args, target=3)
    module.validate_selection(selected, args[2], target=3)
    assert selected["group_ids"] == ["c", "d"]
    assert selected["ids"] == ["c1", "c2", "d1"]
    assert selected["available_group_ids"] == ["c", "d", "b"]
    assert not set(selected["ids"]) & {"a1", "a2"}
    assert args == before


@pytest.mark.parametrize("mutation", ["partial_v3", "partial_exposed", "wrong_group", "duplicate"])
def test_invalid_or_partial_group_inventory_fails_closed(mutation):
    metadata, v3, exposed = fixtures()
    if mutation == "partial_v3":
        v3["remaining_ids"].remove("c2")
    elif mutation == "partial_exposed":
        exposed.pop()
    elif mutation == "wrong_group":
        exposed[0]["group_id"] = "b"
    else:
        metadata.append(dict(metadata[-1]))
    with pytest.raises(ValueError):
        module.select_ids(metadata, v3, exposed, target=1)


def test_metadata_projection_and_selection_reject_target_fields():
    raw = b'{"id":"one","group_id":"g","expected":["secret"],"text":"hidden"}\n'
    assert module.read_metadata(raw, expected_rows=1) == [{"id": "one", "group_id": "g"}]
    with pytest.raises(ValueError, match="non-identity"):
        module.read_selection(raw)


def test_manifest_provenance_flags_describe_raw_json_parsing_truthfully():
    source = open(module.__file__, encoding="utf-8").read()
    assert '"raw_target_bytes_read": True' in source
    assert '"raw_json_rows_parsed": True' in source
    assert '"raw_target_fields_parsed": True' in source
    assert '"selection_fields_used": ["id", "group_id"]' in source
    assert '"target_fields_used_for_selection": False' in source
    assert '"target_text_copied": False' in source
    assert '"target_fields_read"' not in source


def test_pinned_hash_mismatch_fails(tmp_path):
    path = tmp_path / "source.json"
    path.write_text(json.dumps({"value": 1}))
    expected = module.digest(path.read_bytes())
    assert module.read_pinned(path, expected)
    path.write_text(json.dumps({"value": 2}))
    with pytest.raises(ValueError, match="pinned source differs"):
        module.read_pinned(path, expected)
