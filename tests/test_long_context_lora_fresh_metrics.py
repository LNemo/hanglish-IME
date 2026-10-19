import pytest

from scripts.long_context_lora_fresh_metrics import paired_document_bootstrap


def row(identity, group, before=False, after=True):
    return {"id": str(identity), "group_id": group,
            "before_correct": before, "after_correct": after}


@pytest.mark.parametrize("before,after,expected", [(False, True, 1), (True, True, 0), (True, False, -1)])
def test_interval_positive_zero_negative(before, after, expected):
    result = paired_document_bootstrap([row(i, str(i), before, after) for i in range(8)])
    assert result["lower_2_5_percentile"] == expected
    assert result["gain_and_interval_passed"] is (expected == 1)
    assert result["lower_index"] == 49


def test_multirow_document_correlation_and_row_weighted_denominator():
    rows = [row(i, "large") for i in range(10)] + [row(10, "small", True, False)]
    result = paired_document_bootstrap(rows)
    assert result["net_gain"] == 9
    assert result["accuracy_delta"] == 9 / 11
    assert result["lower_2_5_percentile"] == -1
    assert not result["gain_and_interval_passed"]
    assert result == paired_document_bootstrap(list(reversed(rows)))


def test_positive_interval_alone_cannot_replace_gain_requirement():
    result = paired_document_bootstrap([row("a", "document")])
    assert result["lower_2_5_percentile"] == 1
    assert not result["gain_and_interval_passed"]


@pytest.mark.parametrize("rows", [[], [row(1, "g"), row(1, "h")], [row(1, "")], [row(1, "g", 0, 1)]])
def test_malformed_outcomes_fail(rows):
    with pytest.raises(ValueError):
        paired_document_bootstrap(rows)
