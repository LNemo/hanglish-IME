import pytest
import torch

from scripts import probe_roberta_final_choice_resources as probe


def test_synthetic_batch_covers_max_candidate_workload_without_corpus():
    examples = probe.synthetic_examples()
    assert len(examples) == 4
    assert all(len(e.candidates) == 5 for e in examples)
    assert [e.parent_correct for e in examples] == [False, True, True, True]
    assert examples[0].positives == (1,)
    assert all(e.role == "synthetic" and e.loss_weight == 1 for e in examples)


def test_resource_encoding_requires_full_shape_attention_and_zero_types():
    encoded = {"input_ids": torch.ones((20, 96), dtype=torch.long),
               "attention_mask": torch.ones((20, 96), dtype=torch.long),
               "token_type_ids": torch.zeros((20, 96), dtype=torch.long)}
    probe.validate_encoded(encoded)
    encoded["attention_mask"][0, -1] = 0
    with pytest.raises(ValueError, match="padding"):
        probe.validate_encoded(encoded)
    encoded["attention_mask"][0, -1] = 1
    encoded["token_type_ids"][0, 1] = 1
    with pytest.raises(ValueError, match="token-type"):
        probe.validate_encoded(encoded)
    with pytest.raises(ValueError, match="twenty"):
        probe.validate_encoded({"input_ids": torch.ones(16, 96)})


def test_invalid_checkpoint_blocks_model_load(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "WARM", tmp_path / "checkpoint")
    def forbidden(*args, **kwargs):
        pytest.fail("model load must follow checkpoint pin verification")
    monkeypatch.setattr(probe.SentenceRanker, "load", forbidden)
    with pytest.raises(FileNotFoundError):
        probe.run(tmp_path / "out.json")


def test_resource_limits_fail_closed(monkeypatch):
    monkeypatch.setattr(probe, "rss_bytes", lambda: probe.PROFILE["rss_limit_bytes"] + 1)
    with pytest.raises(RuntimeError, match="resource limit"):
        probe.require_resources(probe.time.monotonic())
