import pytest

from scripts.probe_source_edge_recombination import coverage, source_request


def test_source_request_extracts_only_original_input():
    row = {"id": "dev-1", "request": {"input": "gachi gaja", "context": ""},
           "proposed": {"input": "gachi gaja", "context": "", "pipeline": {"pool": []}}}
    assert source_request(row, set()) == ("dev-1", "gachi gaja")
    with pytest.raises(ValueError, match="duplicate"):
        source_request(row, {"dev-1"})
    row["request"]["context"] = "같이"
    with pytest.raises(ValueError, match="source-only"):
        source_request(row, set())


def test_coverage_counts_alternatives_and_preserves_parent_pool():
    inputs = [{"id": "a", "request": {"input": "ga na", "context": ""},
               "proposed": {"pipeline": {"pool": [{"text": "가 나"}]}}},
              {"id": "b", "request": {"input": "da ra", "context": ""},
               "proposed": {"pipeline": {"pool": [{"text": "다 라"}]}}}]
    captures = [{"id": "a", "request": inputs[0]["request"],
                 "proposals": [{"text": "가 다"}]},
                {"id": "b", "request": inputs[1]["request"],
                 "proposals": [{"text": "달아"}]}]
    labels = [{"id": "a", "input": "ga na", "expected": ["가 나", "가 다"]},
              {"id": "b", "input": "da ra", "expected": ["달아"]}]
    result = coverage(iter(inputs), iter(captures), iter(labels))
    assert result == {"rows": 2, "baseline": 1, "union": 2,
                      "gained_ids": ["b"], "lost_ids": []}


def test_coverage_rejects_truncation_and_source_mismatch():
    row = {"id": "a", "request": {"input": "ga na", "context": ""},
           "proposed": {"pipeline": {"pool": []}}}
    capture = {"id": "a", "request": row["request"], "proposals": []}
    label = {"id": "a", "input": "ga na", "expected": ["가 나"]}
    with pytest.raises(ValueError, match="denominator"):
        coverage(iter([row]), iter([]), iter([label]))
    label["input"] = "da ra"
    with pytest.raises(ValueError, match="identity"):
        coverage(iter([row]), iter([capture]), iter([label]))
