from types import SimpleNamespace

import pytest
import torch

from scripts import probe_listwise_semantic_resources as module


def row(index, candidates=1, length=10):
    return {
        "index": index, "eligible": True, "current_index": 0,
        "candidates": [{"text": str(value)} for value in range(candidates)],
        "source_indices": list(range(candidates)),
        "neural_views": [f"view-{value}" for value in range(candidates)],
        "token_lengths": [length] * candidates,
    }


def test_quantile_sample_includes_cost_extrema_and_is_deterministic():
    rows = [row(index, candidates=(index % 4) + 1, length=index + 1)
            for index in range(40)]
    selected = module.quantile_sample(rows)
    ordered = sorted(rows, key=lambda value: (module.workload(value), value["index"]))
    assert len(selected) == 16
    assert selected[0] == ordered[0] and selected[-1] == ordered[-1]
    assert selected == module.quantile_sample(list(reversed(rows)))


def test_validate_rows_requires_complete_bounded_groups():
    rows = [row(0, 4, 512), row(1, 1, 2)]
    assert module.validate_rows(rows, expected_rows=2)
    broken = row(0, 4, 512)
    broken["token_lengths"][2] = 513
    with pytest.raises(ValueError, match="bounded complete"):
        module.validate_rows([broken], expected_rows=1)


class Tokenizer:
    def __call__(self, views, **kwargs):
        assert kwargs == {"truncation": False, "padding": True,
                          "add_special_tokens": True, "return_tensors": "pt"}
        lengths = [2 + index for index in range(len(views))]
        width = max(lengths)
        ids = torch.zeros(len(views), width, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for index, length in enumerate(lengths):
            ids[index, :length] = 1
            mask[index, :length] = 1
        return {"input_ids": ids, "attention_mask": mask}


def test_runtime_encoding_must_match_prepared_lengths_without_truncation():
    value = row(0, candidates=3)
    value["token_lengths"] = [2, 3, 4]
    encoded = module.encode_row(Tokenizer(), value)
    assert encoded["input_ids"].shape == (3, 4)
    value["token_lengths"] = [2, 3, 5]
    with pytest.raises(ValueError, match="runtime tokens"):
        module.encode_row(Tokenizer(), value)


def test_projection_uses_three_fold_train_passes_and_one_inference_pass():
    projected = module.projected_seconds(0.1, 0.02, 100)
    assert projected == pytest.approx(1.5 * (0.1 * 300 + 0.02 * 100) + 120)


class FakeOutput(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1, 2))
        self.bias = torch.nn.Parameter(torch.ones(1))


class FakeModel(torch.nn.Module):
    base_model_prefix = "base_model"

    def __init__(self):
        super().__init__()
        self.base_model = torch.nn.Linear(2, 2)
        self.classifier = SimpleNamespace(out_proj=FakeOutput())
        self.add_module("out", self.classifier.out_proj)


def test_configure_model_zeros_output_and_keeps_dropout_disabled(monkeypatch):
    model = FakeModel()
    tokenizer, loading = object(), {"missing_keys": ["classifier"]}
    monkeypatch.setattr(module, "load_cross_encoder", lambda: (model, tokenizer, loading))

    def layers(value, count):
        assert value is model and count == 1
        for parameter in value.parameters():
            parameter.requires_grad = True
        return {"unfrozen_encoder_layers": 1}

    monkeypatch.setattr(module, "_set_trainable_layers", layers)
    monkeypatch.setattr(module, "_optimizer", lambda value, args, head_only:
                        torch.optim.AdamW(value.parameters(), lr=args.head_lr))
    configured, actual_tokenizer, optimizer, info, actual_loading = module.configure_model()
    assert int(configured.classifier.out_proj.weight.count_nonzero()) == 0
    assert int(configured.classifier.out_proj.bias.count_nonzero()) == 0
    assert configured.training is False and actual_tokenizer is tokenizer
    assert info == {"unfrozen_encoder_layers": 1} and actual_loading == loading
    assert optimizer.param_groups


def test_pin_verification_detects_mutation(tmp_path):
    source = tmp_path / "source"
    source.write_text("fixed")
    pins = {str(source): module.sha256(source)}
    module.verify_pins(pins)
    source.write_text("changed")
    with pytest.raises(ValueError, match="resource source changed"):
        module.verify_pins(pins)


@pytest.mark.parametrize("train,inference,rows", [
    (float("nan"), 1.0, 10), (-1.0, 1.0, 10), (1.0, 1.0, 0),
])
def test_projection_rejects_invalid_inputs(train, inference, rows):
    with pytest.raises(ValueError, match="projection inputs"):
        module.projected_seconds(train, inference, rows)


def test_run_persists_failure_report_without_loading_a_model(monkeypatch, tmp_path):
    output = tmp_path / "resource"
    monkeypatch.setattr(module, "load_inputs", lambda: (_ for _ in ()).throw(
        ValueError("synthetic input failure")))
    report = module.run(output)
    assert report["completed"] is False and report["passed"] is False
    assert "synthetic input failure" in report["failure"]
    assert (output / "report.json").is_file()


def test_projected_too_slow_failure_preserves_measurement_evidence():
    report = {}
    with pytest.raises(ValueError, match="exceeds fixed time"):
        module.record_probe_measurements(
            report, [1.0, 2.0], [0.5, 0.75], [0.7, 0.8],
            123456, 9.0, 3492)
    assert report["train_step_seconds"] == [1.0, 2.0]
    assert report["inference_step_seconds"] == [0.5, 0.75]
    assert report["losses"] == [0.7, 0.8]
    assert report["peak_rss_bytes"] == 123456
    assert report["projected_total_seconds"] > module.RECIPE["max_seconds"]
