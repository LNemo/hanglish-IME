import copy

import pytest

from scripts import verify_phonetic_choice_dev as verifier


def raw_segment(source, text, kind="rule"):
    return {"input_start": 0, "input_end": len(source), "source": source,
            "text": text, "kind": kind}


def test_render_public_replays_raw_slot_and_scored_source():
    source = "source"
    paths = [(f"후보{i}", 9.0 - i, [raw_segment(source, f"후보{i}")]) for i in range(6)]
    public = verifier.render_public(source, paths)
    assert [row["text"] for row in public] == ["후보0", "후보1", "후보2", "후보3", source]
    assert public[-1]["score"] is None

    scored = verifier.render_public(source, [
        ("후보", 2.0, [raw_segment(source, "후보")]),
        (source, 1.0, [raw_segment(source, source, "mixed")]),
    ])
    assert scored[-1]["score"] == 1.0
    assert scored[-1]["segments"][0]["kind"] == "mixed"


def test_pair_id_and_status_are_stable_and_shape_sensitive():
    assert verifier.pair_id("imnida", "입니다") == verifier.pair_id("imnida", "입니다")
    assert verifier.pair_id("imnida", "입니다") != verifier.pair_id("imnida", "임니다")
    assert verifier.expected_status("imnida", "입니다") == "supported"
    assert verifier.expected_status("API", "에이피아이") == "unsupported_ascii"
    assert verifier.expected_status("test", "test") == "copy"
    assert verifier.expected_status("test", "test를") == "unsupported_shape"


def test_verify_choice_row_recomputes_supported_feature_vector(monkeypatch):
    source, target = "ga", "가"
    identifier = verifier.pair_id(source, target)
    monkeypatch.setattr(verifier, "romanize_word", lambda text: "ga")
    edge = {"input_start": 0, "input_end": 2, "source": source, "target": target,
            "status": "supported", "pair_id": identifier, "romanize_match": True}
    segment = raw_segment(source, target)
    score = {"pair_id": identifier, "source": source, "target": target,
             "log_probability": -4.0, "token_count": 4, "mean_log_probability": -1.0}
    row = {"text": target, "base_score": 3.0, "segments": [segment], "edges": [edge],
           "features": [-1.0, 1.0, 0.0, 0.0]}
    values = verifier.verify_choice_row(source, (target, 3.0, [segment]), row, {identifier: score})
    assert values == [-1.0, 1.0, 0.0, 0.0]

    broken = copy.deepcopy(row)
    broken["features"][1] = 0.5
    with pytest.raises(ValueError, match="feature vector"):
        verifier.verify_choice_row(source, (target, 3.0, [segment]), broken, {identifier: score})


def test_probe_groups_cover_all_rows_with_balanced_counts():
    requests = [(f"id-{i:03d}", "x" * (i % 23 + 1), None) for i in range(316)]
    groups = verifier.probe_groups(requests)
    assert len(groups) == 16
    assert sum(group["count"] for group in groups) == 316
    assert max(group["count"] for group in groups) - min(group["count"] for group in groups) == 1
    assert sorted(identifier for group in groups for identifier in group["ids"]) == sorted(
        identifier for identifier, _, _ in requests)
