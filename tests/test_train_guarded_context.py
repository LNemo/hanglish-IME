from types import SimpleNamespace
import json

import pytest

from scripts import train_guarded_context as training


def empty_guard(**changes):
    value = {
        "ids": set(),
        "group_ids": set(),
        "content": set(),
        "natural_expected": set(),
        "counts": {},
    }
    value.update(changes)
    return value


def source_row(**changes):
    value = {
        "id": "train-1",
        "origin": "natural",
        "split": "train",
        "group_id": "group-1",
        "input": "stale-input",
        "context": "주변 문맥",
        "expected": ["국밥"],
        "groups": [{"id": "stale-group-that-must-not-be-used"}],
    }
    value.update(changes)
    return value


def candidate_group(name):
    return {
        "id": name,
        "context": "주변 [MASK] 문맥",
        "candidates": ["정답", "오답"],
        "positive_indices": [0],
        "base_scores": [0.0, 0.0],
        "teacher_scores": [1.0, 0.0],
    }


def test_refresh_source_recomputes_input_and_drops_historical_groups():
    source, excluded = training.refresh_source(
        source_row(), "natural_train", empty_guard())

    assert excluded is None
    assert source["input"] == "gukbap"
    assert source["id"] == "train-1"
    assert source["split"] == "train"
    assert source["group_id"] == "group-1"
    assert "groups" not in source


def test_refresh_source_excludes_forward_incompatible_alternates(monkeypatch):
    monkeypatch.setattr(
        training, "romanize_word", lambda text: {"정답": "same", "별답": "other"}[text])

    source, excluded = training.refresh_source(
        source_row(origin="retention", expected=["정답", "별답"]),
        "retention_train", empty_guard())

    assert source is None
    assert excluded == {
        "id": "train-1",
        "reason": "alternate_forward_mismatch",
        "canonical_input": "same",
        "excluded_expected": ["별답"],
    }


@pytest.mark.parametrize(
    "changes",
    [{"split": "dev"}, {"origin": "retention"}],
)
def test_refresh_source_requires_fixed_training_provenance(changes):
    with pytest.raises(ValueError, match="provenance"):
        training.refresh_source(
            source_row(**changes), "natural_train", empty_guard())


@pytest.mark.parametrize(
    ("guard", "reason"),
    [
        (empty_guard(ids={"train-1"}), "development_identity_overlap"),
        (empty_guard(group_ids={"group-1"}), "development_identity_overlap"),
        (empty_guard(content={("gukbap", "주변 문맥", ("국밥",))}),
         "development_content_overlap"),
        (empty_guard(natural_expected={"국밥"}), "natural_dev_target_overlap"),
    ],
)
def test_refresh_source_excludes_each_development_leakage_form(guard, reason):
    source, excluded = training.refresh_source(
        source_row(), "natural_train", guard)

    assert source is None
    assert excluded["reason"] == reason


def test_single_word_dev_text_is_not_used_as_generic_substring_filter():
    guard = empty_guard(content={("nat", "다른 문맥", ("낫",))})

    source, excluded = training.refresh_source(
        source_row(context="낫을 사용하는 설명"), "natural_train", guard)

    assert source is not None
    assert excluded is None


def test_capture_case_freezes_features_before_target_alignment(monkeypatch):
    events = []
    group = {"id": "captured"}

    class Converter:
        def _prepare_candidates(self, text, context, *, learned, raw):
            events.append(("prepare", text, context, learned, raw))
            return None, ["pieces"], None, None, None

    def capture(pieces, records, context):
        events.append(("capture", pieces, records, context))
        return [group]

    def label(source, groups):
        events.append(("label", source["expected"], groups))
        return [{**group, "positive_indices": [0]}]

    monkeypatch.setattr(training, "capture_groups", capture)
    monkeypatch.setattr(training, "add_targets", label)
    monkeypatch.setattr(training, "validate_case",
                        lambda case, source: events.append(("validate", case["id"])))
    source, _ = training.refresh_source(source_row(), "natural_train", empty_guard())

    case = training.capture_case(Converter(), SimpleNamespace(records=[]), source)

    assert [event[0] for event in events] == ["prepare", "capture", "label", "validate"]
    assert events[0] == ("prepare", "gukbap", "주변 문맥", True, False)
    assert case["groups"][0]["positive_indices"] == [0]


def test_build_training_examples_applies_the_same_cap_to_both_roles(monkeypatch):
    recipe = {**training.RECIPE, "source_group_limit": 2,
              "selection_salt": "bounded-test"}
    monkeypatch.setattr(training, "RECIPE", recipe)
    data = {
        "natural_train": [{"origin": "natural", "groups": [
            candidate_group(f"n:{index}") for index in range(5)]}],
        "retention_train": [{"origin": "retention", "groups": [
            candidate_group(f"r:{index}") for index in range(5)]}],
    }

    examples, eligibility = training.build_training_examples(data)

    assert len(examples["natural_train"]) == 2
    assert len(examples["retention_train"]) == 2
    assert eligibility["natural_train"]["selected_before_labels"] == 2
    assert eligibility["retention_train"]["selected_before_labels"] == 2


def test_schedule_document_records_contiguous_group_and_candidate_offsets():
    schedule = [
        SimpleNamespace(case_id="a", origin="natural", candidates=("가", "나")),
        SimpleNamespace(case_id="b", origin="retention", candidates=("다",)),
        SimpleNamespace(case_id="a", origin="natural", candidates=("가", "나")),
    ]

    document = training.schedule_document(schedule)

    assert document["scheduled_groups"] == 3
    assert document["scheduled_candidates"] == 5
    assert [(row["index"], row["id"], row["candidate_start"],
             row["candidate_end"]) for row in document["groups"]] == [
        (0, "a", 0, 2), (1, "b", 2, 3), (2, "a", 3, 5)]


def test_preparation_uses_one_watchdog_operation_and_durable_event_per_case(
        tmp_path, monkeypatch):
    rows = [source_row(id="one"), source_row(id="two", group_id="group-2")]
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    monkeypatch.setattr(training, "TRAIN_FILES", {"natural_train": source})
    monkeypatch.setattr(training, "EXPECTED_ROWS", {"natural_train": 2})
    monkeypatch.setattr(
        training, "capture_case",
        lambda converter, recorder, row: {**row, "groups": []})
    events = []
    monkeypatch.setattr(
        training, "append_update",
        lambda path, event, **values: events.append((event, values)))

    class Operation:
        def __init__(self, operations, name, update):
            self.operations, self.name, self.update = operations, name, update

        def __enter__(self):
            self.operations.append((self.name, self.update))

        def __exit__(self, *args):
            return False

    class Guard:
        def __init__(self):
            self.operations = []

        def operation(self, name, *, update):
            return Operation(self.operations, name, update)

    guard = Guard()
    converter = SimpleNamespace(ranker=SimpleNamespace())

    prepared, exclusions = training.prepare_training_data(
        converter, tmp_path / "prepared", empty_guard(),
        tmp_path / "updates.jsonl", guard)

    assert [name for name, update in guard.operations if update] == [
        "prepare:natural_train:0", "prepare:natural_train:1"]
    assert [event for event, _ in events].count("preparation_case") == 2
    assert [event for event, _ in events][-1] == "preparation_role_complete"
    assert len(prepared["natural_train"]) == 2
    assert exclusions == {"natural_train": []}


def test_dev_guard_covers_every_declared_external_development_row():
    guard = training.build_dev_guard()

    assert guard["counts"] == {
        "natural_dev": 128,
        "original_dev": 140,
        "selection_dev": 60,
        "reasoning_dev": 40,
    }


def test_recipe_and_resource_limits_are_fixed_and_no_dev_is_a_train_role():
    assert training.RECIPE["epochs"] == 1
    assert training.RECIPE["source_group_limit"] == 2048
    assert training.RECIPE["group_batch"] == 4
    assert training.LIMITS == {
        "whole_seconds": 5400.0,
        "update_seconds": 60.0,
        "rss_bytes": 3 * 1024**3,
        "swap_growth_bytes": 1536 * 1024**2,
    }
    assert set(training.TRAIN_FILES) == {"natural_train", "retention_train"}
    assert set(training.DEV_FILES) == {
        "natural_dev", "original_dev", "selection_dev", "reasoning_dev"}
