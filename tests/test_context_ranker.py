import json
import hashlib

import pytest
import torch

from hanglish.models.context_ranker import ContextRanker, _RankerNet, _bounded_text, _features
from hanglish.core.lattice import CandidateGenerator
from scripts.train_context_ranker import _options_for_source, align_words


def test_alignment_handles_split_romanized_prefix():
    assert align_words("i beon-e gungnae-ro bonaenda", "이번에 국내로 보낸다") == [
        ("ibeon-e", "이번에"), ("gungnae-ro", "국내로"), ("bonaenda", "보낸다")
    ]


def test_options_preserve_hyphenated_english_and_allow_korean_separator():
    generator = CandidateGenerator()
    assert "x-request-id에서" in {x.text for x in _options_for_source("x-request-ideseo", generator)}
    assert "민수는" in {x.text for x in _options_for_source("minsu-neun", generator)}


def test_context_features_keep_mask_near_tail_and_bound_old_context():
    text = ("처음에만나오는표식 " + "오래된문맥" * 200 + " [MASK] "
            + "오른쪽문맥" * 200 + " 맨끝에만나오는표식")
    bounded = _bounded_text(text)
    features = _features(text, "context")
    assert len(bounded) == 500
    assert "[MASK]" in bounded
    assert any("[MASK]" in feature for feature in features)
    assert not any("처음에만나오는표식" in feature for feature in features)
    assert "맨끝에만나오는표식" not in bounded


def test_checkpoint_load_is_deterministic(tmp_path):
    torch.manual_seed(17)
    config = {"buckets": 128, "embedding_dim": 8, "hidden_dim": 12, "dropout": 0.5}
    model = _RankerNet(**config)
    path = tmp_path / "ranker.pt"
    torch.save({"format_version": 1, "model_config": config,
                "state_dict": model.state_dict()}, path)
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".json").write_text(json.dumps({
        "format_version": 1, "seed": 17, "checkpoint_sha256": checksum,
    }), encoding="utf-8")
    ranker = ContextRanker.load(path)
    first = ranker.score_pairs(["빛을 뜻하는 말은 [MASK]"] * 2, ["빗", "빛"])
    second = ranker.score_pairs(["빛을 뜻하는 말은 [MASK]"] * 2, ["빗", "빛"])
    assert first == second
    assert ranker.metadata["seed"] == 17


def test_score_pairs_validates_lengths():
    model = _RankerNet(buckets=32, embedding_dim=4, hidden_dim=4)
    ranker = ContextRanker(model)
    with pytest.raises(ValueError):
        ranker.score_pairs(["one"], ["하나", "원"])
    assert ranker.score_pairs([], []) == []


def test_default_model_stays_under_one_million_parameters():
    model = _RankerNet()
    assert sum(parameter.numel() for parameter in model.parameters()) < 1_000_000


def test_load_rejects_unknown_configuration(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"format_version": 1, "model_config": {"surprise": True}, "state_dict": {}}, path)
    path.with_suffix(".json").write_text(json.dumps({
        "format_version": 1, "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="configuration"):
        ContextRanker.load(path)


def test_load_rejects_checkpoint_checksum_mismatch(tmp_path):
    config = {"buckets": 32, "embedding_dim": 4, "hidden_dim": 4, "dropout": 0.1}
    model = _RankerNet(**config)
    path = tmp_path / "ranker.pt"
    torch.save({"format_version": 1, "model_config": config,
                "state_dict": model.state_dict()}, path)
    path.with_suffix(".json").write_text(
        json.dumps({"format_version": 1, "checkpoint_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        ContextRanker.load(path)


def test_load_requires_metadata_checksum(tmp_path):
    path = tmp_path / "ranker.pt"
    torch.save({"format_version": 1}, path)
    path.with_suffix(".json").write_text(json.dumps({"format_version": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="lacks a valid"):
        ContextRanker.load(path)


def test_load_rejects_unsupported_payload_format(tmp_path):
    path = tmp_path / "ranker.pt"
    torch.save({"format_version": 2}, path)
    path.with_suffix(".json").write_text(json.dumps({
        "format_version": 1, "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported context-ranker checkpoint"):
        ContextRanker.load(path)
