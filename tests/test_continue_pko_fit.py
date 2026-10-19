import json
import math
from pathlib import Path

import pytest

from scripts import continue_pko_fit as stage


def test_prepare_replays_all_frozen_train_only_boundaries_without_model_load():
    authorization = stage.prepare()
    assert sum(map(len, authorization["replay_groups"])) == 153
    assert len(authorization["replay_groups"]) == 20
    assert len(authorization["replay_groups"][-1]) == 1
    assert len(authorization["unseen_rows"]) == 1963
    assert len(authorization["continuation_groups"]) == 246
    assert len(authorization["continuation_groups"][-1]) == 3
    assert len(authorization["pairs"]) == 64
    assert all("dev.jsonl" not in path and "test.jsonl" not in path
               for path in authorization["pins"])


def test_unseen_schedule_is_unique_and_disjoint_from_seen_and_inner():
    authorization = stage.prepare()
    unseen = [row["id"] for row in authorization["unseen_rows"]]
    seen = authorization["schedule_ids"][:stage.SEEN_MICROSTEPS]
    diagnostic_ids = set(authorization["diagnostic_targets"])
    assert len(unseen) == len(set(unseen)) == 1963
    assert set(unseen).isdisjoint(seen)
    assert set(unseen).isdisjoint(diagnostic_ids)


def test_validate_replay_contract_binds_ids_and_last_partial():
    rows = [{"id": f"id-{index}"} for index in range(6916)]
    parent_groups = stage.groups(rows[:4953])
    journals = [{"optimizer_update": index + 1,
                 "ids": [row["id"] for row in group],
                 "accumulation_size": len(group), "loss_divisor": len(group)}
                for index, group in enumerate(parent_groups)]
    result = stage.validate_replay_contract(rows, journals)
    assert len(result) == 20
    assert [len(group) for group in result] == [8] * 19 + [1]
    journals[619]["ids"] = ["tampered"]
    with pytest.raises(ValueError, match="IDs"):
        stage.validate_replay_contract(rows, journals)


def test_replayed_update_uses_explicit_tiny_fp32_tolerances():
    expected = {"ids": ["a"], "accumulation_size": 1, "loss_divisor": 1,
                "target_tokens": 3, "token_weighted_loss": 2.0,
                "gradient_norm": 1.0}
    stage.validate_replayed_update(dict(expected, token_weighted_loss=2.0 + 1e-6,
                                        gradient_norm=1.0 + 1e-6), expected)
    with pytest.raises(ValueError, match="loss"):
        stage.validate_replayed_update(
            dict(expected, token_weighted_loss=2.0 + 1e-3), expected)
    with pytest.raises(ValueError, match="gradient"):
        stage.validate_replayed_update(
            dict(expected, gradient_norm=1.0 + 1e-3), expected)


def test_stage_gate_requires_every_predeclared_condition():
    before_conditioning = {"normal_token_nll": 2.5, "median_row_delta": .5}
    after_conditioning = {"normal_token_nll": 2.0, "normal_lower_rows": 50,
                          "median_row_delta": .46}
    before_generation = {"correct_flags": [False] * 64,
                         "literal_flags": [True] * 64}
    after_generation = {"correct_flags": [True] * 4 + [False] * 60,
                        "literal_flags": [True] * 64}
    result = stage.stage_gate(before_conditioning, after_conditioning,
                              before_generation, after_generation)
    assert result["promising_for_next_train_only_stage"] is True
    assert result["new_exact"] == 4
    assert result["literal_signature_regressions"] == 0
    assert result["runtime_promotion"] is False

    for mutation in (
        lambda c, g: c.update(normal_token_nll=2.3),
        lambda c, g: c.update(normal_lower_rows=41),
        lambda c, g: c.update(median_row_delta=.44),
        lambda c, g: g["correct_flags"].__setitem__(3, False),
        lambda c, g: g["literal_flags"].__setitem__(0, False),
    ):
        condition = dict(after_conditioning)
        generated = {key: list(value) for key, value in after_generation.items()}
        mutation(condition, generated)
        assert stage.stage_gate(before_conditioning, condition,
                                before_generation, generated)[
                                    "promising_for_next_train_only_stage"] is False


def test_character_distance_known_examples():
    assert stage.character_distance("", "한글") == 2
    assert stage.character_distance("한글", "한글") == 0
    assert stage.character_distance("임니다", "입니다") == 1
    assert stage.character_distance("kitten", "sitting") == 3


def test_generation_scoring_includes_all_rows_and_labels_only_after_capture():
    captures, targets = [], {}
    for index in range(64):
        row_id = f"row-{index}"
        source = "testreul sihaeng" if index == 0 else "imnida"
        target = "test를 시행" if index == 0 else "입니다"
        prediction = target if index < 4 else "임니다"
        captures.append({"id": row_id, "source": source,
                         "generated_text": prediction, "reason": "complete",
                         "eos_completed": True, "token_ids": [0, 1]})
        targets[row_id] = target
    result = stage.score_generation(captures, targets)
    assert result["rows"] == 64
    assert result["whole_exact"] == 4
    assert result["eos_complete"] == 64
    assert result["labels_joined_after_closed_capture"] is True
    assert result["character_edits"] > 0
    assert result["target_characters"] == sum(map(len, targets.values()))
    assert 0 < result["character_error_rate"] < 1


def test_groups_divide_actual_last_accumulation():
    rows = [{"id": str(index)} for index in range(1963)]
    grouped = stage.groups(rows)
    assert len(grouped) == 246
    assert [len(group) for group in grouped[-2:]] == [8, 3]


def test_checkpoint_contains_model_optimizer_rng_identity_and_cursor(tmp_path):
    torch = pytest.importorskip("torch")

    class Saver:
        def save_pretrained(self, path, **kwargs):
            Path(path).mkdir(exist_ok=True)
            (Path(path) / "artifact.bin").write_bytes(b"safe")

    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4)
    state = {"continuation_updates": 100, "continuation_microsteps": 800,
             "next_group": 100}
    identity = {"pins": {"source": "a" * 64}}
    pointer = stage.save_checkpoint(
        tmp_path, Saver(), optimizer, Saver(), state, identity)
    assert pointer["state"] == state
    assert pointer["identity"] == identity
    assert pointer["model_files"]
    assert len(pointer["optimizer_sha256"]) == 64
    assert pointer["python_random_state"]
    payload = torch.load(tmp_path / pointer["optimizer_path"], weights_only=True)
    assert "optimizer" in payload and "torch_rng_state" in payload


def test_output_is_first_run_only(tmp_path):
    output = tmp_path / "stage-a"
    assert stage.prepare_output(output) == output
    with pytest.raises(FileExistsError):
        stage.prepare_output(output)


def test_resource_contract_is_bounded_and_distinguishes_prior_diagnostic():
    assert stage.LIMITS == {"rss_bytes": 6 * 1024**3,
                            "swap_growth_bytes": 512 * 1024**2,
                            "whole_seconds": 900.0,
                            "update_seconds": 120.0}
    frozen = json.loads((stage.DIAGNOSTIC / "report.json").read_text())
    assert frozen["resources"]["limits"]["update_seconds"] == 60.0
    assert frozen["metrics"]["normal_lower_rows"] == 64
