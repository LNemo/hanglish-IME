from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import retrain_rr_phonetic as runner


class HeldInput(dict):
    """Fail if preparation accidentally reads a held-out target."""

    def get(self, key, default=None):
        if key == "surface":
            raise AssertionError("held-out surface was read")
        return super().get(key, default)


def test_refresh_uses_train_surfaces_only_and_records_dedup_and_diffs():
    base = [
        {"roman": "old", "surface": "가", "split": "train"},
        HeldInput(roman="held", split="lexical_holdout"),
    ]
    delta = [{"roman": "other", "surface": "가", "split": "train"}]

    result = runner.build_refreshed_records(base, delta, lambda _: "new")

    assert [(row["roman"], row["surface"]) for row in result["records"]] == [("new", "가")]
    assert result["counts"] == {
        "historical_train_sources": 2,
        "refreshed_train_pairs": 1,
        "roman_changed_sources": 2,
        "held_input_collision_sources_excluded": 0,
        "duplicate_sources_deduplicated": 1,
        "historical_held_inputs": 1,
    }
    assert {row["old_roman"] for row in result["differences"]} == {"old", "other"}
    assert len(result["records"][0]["origins"]) == 2


def test_refresh_excludes_new_input_collision_with_historical_holdout():
    base = [
        {"roman": "train", "surface": "가", "split": "train"},
        HeldInput(roman="held", split="morph_combination_holdout"),
    ]

    result = runner.build_refreshed_records(base, [], lambda _: "held")

    assert result["records"] == []
    assert result["excluded"] == [{
        "reason": "historical_held_input_collision", "roman": "held",
        "surface": "가", "source": "base", "index": 0, "old_roman": "train",
    }]


def test_refresh_rejects_non_train_delta():
    with pytest.raises(ValueError, match="non-TRAIN"):
        runner.build_refreshed_records(
            [{"roman": "ga", "surface": "가", "split": "train"}],
            [{"roman": "na", "surface": "나", "split": "lexical_holdout"}],
            lambda surface: {"가": "ga", "나": "na"}[surface])


def test_verify_hashes_detects_source_change(tmp_path: Path):
    source = tmp_path / "source.txt"
    source.write_text("frozen", encoding="utf-8")
    pins = {"source.txt": hashlib.sha256(source.read_bytes()).hexdigest()}
    runner.verify_hashes(tmp_path, pins)

    source.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="frozen source differs"):
        runner.verify_hashes(tmp_path, pins)


def test_all_current_romanizer_dependencies_are_individually_frozen(tmp_path: Path):
    pins = {}
    for name in runner.CURRENT_ROMANIZER_DEPENDENCIES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
        pins[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    runner.verify_hashes(tmp_path, pins)

    changed = tmp_path / runner.CURRENT_ROMANIZER_DEPENDENCIES[-1]
    changed.write_text("mutated inventory", encoding="utf-8")
    with pytest.raises(ValueError, match="predicate_boundary_inventory"):
        runner.verify_hashes(tmp_path, pins)


def test_load_prepared_rejects_mutated_dataset(tmp_path: Path, monkeypatch):
    source = tmp_path / "source.txt"; source.write_text("source", encoding="utf-8")
    records = [{"index": 0, "pair_id": "x", "roman": "ga", "surface": "가",
                "split": "train", "origins": []}]
    train = tmp_path / "train.jsonl"
    train.write_bytes(runner.canonical(records[0]) + b"\n")
    for name in ("roman-differences.jsonl", "excluded.jsonl", "held-inputs.jsonl"):
        (tmp_path / name).write_bytes(b"")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "UPDATES", 1)
    monkeypatch.setattr(runner, "BATCH_SIZE", 1)
    schedule = runner.cyclic_schedule(1)
    files = {name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
             for name in ("train.jsonl", "roman-differences.jsonl",
                          "excluded.jsonl", "held-inputs.jsonl")}
    manifest = {
        "schema": "rr-phonetic-retrain-prepared-v1", "completed": True,
        "recipe": runner.RECIPE, "model_config": runner.CONFIG,
        "operational_policy": runner.OPERATIONAL_POLICY,
        "contracts": {"fresh_evaluation_targets_read": False,
                      "evaluation_targets_used_for_training": False},
        "source_hashes": {"source.txt": hashlib.sha256(source.read_bytes()).hexdigest()},
        "files": files,
        "schedule": {"schedule_sha256": hashlib.sha256(
            runner.canonical(schedule)).hexdigest()},
    }
    (tmp_path / "prepared.json").write_text(json.dumps(manifest), encoding="utf-8")

    train.write_bytes(runner.canonical({**records[0], "roman": "na"}) + b"\n")
    with pytest.raises(ValueError, match="prepared file differs"):
        runner.load_prepared(tmp_path)


def test_atomic_write_refuses_existing_temporary_file(tmp_path: Path):
    target = tmp_path / "report.json"
    target.with_suffix(".json.tmp").write_text("occupied", encoding="utf-8")

    with pytest.raises(FileExistsError):
        runner.atomic_write(target, b"{}\n")
    assert not target.exists()


def test_probe_runtime_limit_must_match_training_limit(tmp_path: Path):
    report = {
        "schema": "rr-phonetic-retrain-probe-v1", "completed": True,
        "accepted": True, "prepared_sha256": "prepared",
        "recipe": runner.RECIPE, "limits": runner.runtime_limits(7200),
        "operational_policy": runner.OPERATIONAL_POLICY,
        "initial_tensor_sha256": runner.INITIAL_TENSOR_SHA256,
        "observations": [{} for _ in range(runner.PROBE_UPDATES)],
    }
    runner.write_json(tmp_path / "report.json", report)
    runner.write_json(tmp_path / "completion.json", {
        "schema": "rr-phonetic-retrain-completion-v1", "completed": True,
        "report_sha256": runner.sha256_file(tmp_path / "report.json"),
    })

    assert runner.validate_probe(
        tmp_path, "prepared", runner.runtime_limits(7200)) == report
    with pytest.raises(ValueError, match="accepted probe contract differs"):
        runner.validate_probe(tmp_path, "prepared", runner.runtime_limits(1800))
