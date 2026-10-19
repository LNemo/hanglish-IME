import json

import pytest
import torch
from torch import nn

import scripts.train_supported_phonetic_transducer as module


def row(index, roman, surface, split="train", *, rooted=True, family_train=True):
    return {
        "index": index, "roman": roman, "surface": surface, "split": split,
        "known_family_root": rooted, "family_has_train": family_train,
    }


class TinyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(64, 8)
        self.output = nn.Linear(8, 64)

    def forward(self, source, target_input):
        return self.output(self.embedding(target_input))


class FakeModel:
    def __init__(self, outputs):
        self.outputs = outputs

    def generate(self, roman, beam_size):
        assert beam_size == 8
        return [(text, -index) for index, text in enumerate(self.outputs[roman])]


def test_quantile_probe_selections_include_cost_and_length_endpoints(monkeypatch):
    monkeypatch.setitem(module.RECIPE, "batch_size", 2)
    records = [row(i, "a" * (i + 2), "가" * (i + 1)) for i in range(40)]
    batches, selected, schedule = module.selected_update_batches(records, count=4)
    beams = module.selected_beam_inputs(records, count=5)
    assert len(batches) == 4 and len(selected) == 4
    assert len(schedule) == 120
    selected_indices = {index for batch in selected for index in batch["pair_indices"]}
    assert records[0]["index"] in selected_indices
    assert records[-1]["index"] in selected_indices
    assert beams[0] == records[0]["roman"] and beams[-1] == records[-1]["roman"]


def test_real_tiny_update_changes_weights_and_is_finite(monkeypatch):
    monkeypatch.setitem(module.RECIPE, "gradient_clip_norm", 1.0)
    dataset = module.PairDataset([("ga", "가"), ("na", "나")])
    batch = module.collate([dataset[0], dataset[1]])
    network = TinyNetwork()
    before = {name: value.detach().clone() for name, value in network.state_dict().items()}
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.01)
    losses = module.update_batches(network, [batch], optimizer, module.criterion())
    assert len(losses) == 1 and torch.isfinite(torch.tensor(losses)).all()
    assert any(not torch.equal(before[name], value)
               for name, value in network.state_dict().items())


def test_held_predictions_and_metrics_use_accepted_variant_sets_and_strata():
    records = [
        row(0, "nat", "낟", "lexical_holdout", rooted=True),
        row(1, "nat", "낫", "lexical_holdout", rooted=False),
        row(2, "ga", "가", "lexical_holdout", rooted=True, family_train=False),
    ]
    predictions = module.capture_held_predictions(
        FakeModel({"nat": ["낫", "낟"], "ga": ["가"]}), records
    )
    metrics = module.metrics_from_predictions(predictions, records)
    nat = next(value for value in predictions if value["roman"] == "nat")
    assert set(nat) == {"split", "roman", "pair_indices", "outputs"}
    assert nat["outputs"] == [
        {"text": "낫", "logprob": 0.0},
        {"text": "낟", "logprob": -1.0},
    ]
    assert nat["pair_indices"] == [0, 1]
    assert metrics["overall"]["inputs"] == 2
    assert metrics["overall"]["top1_input_accuracy"] == 1.0
    assert metrics["overall"]["top1_variant_recall"] == pytest.approx(2 / 3)
    assert metrics["root_known"]["top1_variant_recall"] == pytest.approx(1 / 2)
    assert metrics["family_without_train"]["top8_variant_recall"] == 1.0
    assert metrics["overall"]["empty_outputs"] == 0


def test_probe_contract_rejects_any_pin_or_recipe_change():
    config = module.TransducerConfig()
    pins = {"a": "hash"}
    report = {
        "schema": "supported-phonetic-transducer-probe-v1", "completed": True,
        "accepted": True, "contract": module.current_contract(pins, config),
        "model_updates_run": module.RECIPE["probe_update_batches"],
        "beam_inputs_run": module.RECIPE["probe_beam_inputs"],
        "prototype_discarded": True, "training_run": False, "held_evaluation_run": False,
        "morph_validation_forward_run": True, "lexical_validation_run": False,
        "source_hashes": pins, "source_hashes_after": pins,
        "source_hashes_unchanged": True,
        "update_selection_includes_min_and_max_cost": True,
        "update_selection_includes_shortest_and_longest_roman": True,
        "beam_selection_includes_shortest_and_longest": True,
        "mean_update_loss": 1.0, "morph_validation_token_ce": 1.0,
        "update_observations": [
            {"seconds": 0.1, "padded_cost": 10, "pair_indices": [index]}
            for index in range(16)
        ],
        "peak_rss_bytes": 100,
        "resource": {"time_limit_seconds": 1800, "rss_limit_bytes": 2 * 1024**3,
                     "elapsed_seconds": 3.0, "peak_rss_bytes": 100},
        "timing": {"updates_seconds": 1.6, "morph_validation_seconds": 1.0,
                   "beams_seconds": 1.0, "probe_elapsed_seconds": 3.0,
                   "projected_full_train_and_all_held_beam_seconds": 0.0},
        "projection": {"train_update_batches": 2412,
                       "full_training_padded_cost": 24120,
                       "training_schedule_sha256": "s" * 64,
                       "observed_update_padded_cost": 160,
                       "update_seconds_per_padded_unit": 0.01,
                       "morph_selection_full_passes": 6,
                       "held_unique_beam_inputs": 3292,
                       "held_beam_evaluation_passes": 1,
                       "multiplier": 1.3, "margin_seconds": 60},
    }
    report["timing"]["projected_full_train_and_all_held_beam_seconds"] = (
        3 + 60 + 1.3 * (1.6 / 160 * 24120 + 1 * 6 + 1 / 256 * 3292)
    )
    module.validate_probe_report(report, pins, config)
    changed = json.loads(json.dumps(report))
    changed["contract"]["source_hashes"]["a"] = "changed"
    with pytest.raises(ValueError, match="accepted probe"):
        module.validate_probe_report(changed, pins, config)


def test_resource_limits_fail_closed():
    assert any("deterministic romanize_word" in value
               for value in module.INTERPRETATION_LIMITS)
    with pytest.raises(module.ResourceGateError, match="1800"):
        module.check_resources(0.0, clock=lambda: 1800.1, rss_reader=lambda: 0)
    with pytest.raises(module.ResourceGateError, match="2 GiB"):
        module.check_resources(0.0, clock=lambda: 1.0,
                               rss_reader=lambda: module.RECIPE["rss_limit_bytes"] + 1)
