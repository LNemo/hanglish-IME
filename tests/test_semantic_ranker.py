import json

import pytest
import torch

from hanglish.models.semantic_ranker import SemanticRanker, context_neighborhood, render_candidate
from scripts.train_context_ranker import RankingExample
from scripts.train_semantic_ranker import (
    PINNED_MODEL, PINNED_REVISION, balanced_order, choose_options, parent_provenance,
    require_fresh_output, selection_key, validate_base_snapshot,
    validate_parent_training_provenance,
)
from hanglish.core.lattice import WordOption


class _Tokenizer:
    def __call__(self, texts, text_pair=None, **kwargs):
        if text_pair is not None:
            texts = [left + right for left, right in zip(texts, text_pair)]
        return {"input_ids": torch.tensor([[len(text)] for text in texts])}


class _Output:
    def __init__(self, logits):
        self.logits = logits


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids):
        return _Output(input_ids.float())


def test_render_candidate_replaces_active_last_mask():
    assert render_candidate("문맥 [MASK] 뒤 [MASK]", "가치") == "문맥 [MASK] 뒤 가치"
    assert render_candidate("문맥", "같이") == "문맥 같이"


def test_context_neighborhood_keeps_candidate_and_recent_left_context():
    context = "버릴앞문맥" * 100 + "최근 왼쪽 문맥 [MASK] 오른쪽 문맥" + "버릴뒤문맥" * 100
    neighborhood = context_neighborhood(context, "가치")
    assert "가치" in neighborhood
    assert "최근 왼쪽 문맥" in neighborhood
    assert "버릴앞문맥" * 20 not in neighborhood
    assert "버릴뒤문맥" * 20 not in neighborhood


def test_compatible_score_pairs_api_is_deterministic():
    ranker = SemanticRanker(_Model(), _Tokenizer(), batch_size=2)
    first = ranker.score_pairs(["[MASK]", "긴 [MASK]"], ["같이", "가치"])
    second = ranker.score_pairs(["[MASK]", "긴 [MASK]"], ["같이", "가치"])
    assert first == second
    assert len(first) == 2
    assert ranker.score_pairs([], []) == []


def test_score_pairs_validates_input():
    ranker = SemanticRanker(_Model(), _Tokenizer())
    with pytest.raises(ValueError):
        ranker.score_pairs(["문맥"], ["같이", "가치"])
    with pytest.raises(TypeError):
        ranker.score_pairs(("문맥",), ["같이"])


def test_choose_options_keeps_positive_and_top_hard_negatives():
    example = RankingExample(
        "[MASK]", tuple(WordOption(str(index), float(index), "lexicon") for index in range(8)),
        (0,), "case", "category")
    candidates, positives = choose_options(example, 4)
    assert candidates == ["0", "7", "6", "5"]
    assert positives == (0,)


def test_balanced_order_repeats_rare_train_categories_deterministically():
    options = (WordOption("정답", 1.0, "lexicon"), WordOption("오답", 0.0, "raw"))
    examples = [
        RankingExample("[MASK]", options, (0,), "a1", "common"),
        RankingExample("[MASK]", options, (0,), "a2", "common"),
        RankingExample("[MASK]", options, (0,), "b1", "rare"),
    ]
    first = balanced_order(examples, 3, 11)
    assert first == balanced_order(examples, 3, 11)
    assert len(first) == 6
    assert sum(examples[index].category == "rare" for index in first) == 3


def test_selection_key_uses_nll_as_accuracy_tiebreaker_and_accepts_zero():
    perfect = {"oracle_rank_accuracy": 1.0, "mean_listwise_nll": 0.0}
    less_confident = {"oracle_rank_accuracy": 1.0, "mean_listwise_nll": 0.1}
    assert selection_key(perfect) > selection_key(less_confident)


def test_fresh_output_guard_preserves_existing_directory(tmp_path):
    output = tmp_path / "nested" / "checkpoint"
    require_fresh_output(output)
    assert output.parent.is_dir() and not output.exists()
    output.mkdir(); marker = output / "keep"; marker.write_text("present")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        require_fresh_output(output)
    assert marker.read_text() == "present"


def test_load_rejects_metadata_before_importing_artifacts(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "metadata.json").write_text(json.dumps({"format_version": 2}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        SemanticRanker.load(checkpoint)


@pytest.mark.parametrize('hashes', [None, {}, {'model.safetensors': 'abc'}])
def test_load_requires_complete_hash_manifest(tmp_path, hashes):
    (tmp_path / 'metadata.json').write_text(json.dumps({'format_version': 1, 'artifact_hashes': hashes}))
    with pytest.raises(ValueError, match='complete artifact hash manifest'):
        SemanticRanker.load(tmp_path)


def test_load_rejects_unverified_tokenizer_sidecar(tmp_path):
    names = ['model.safetensors', 'config.json', 'tokenizer.json', 'vocab.txt',
             'tokenizer_config.json', 'special_tokens_map.json']
    (tmp_path / 'metadata.json').write_text(json.dumps({
        'format_version': 1, 'artifact_hashes': dict.fromkeys(names, 'abc'),
        'max_length': 96, 'inference_batch_size': 8,
    }))
    (tmp_path / 'added_tokens.json').write_text('{}')
    with pytest.raises(ValueError, match='unverified artifacts'):
        SemanticRanker.load(tmp_path)


def test_base_snapshot_requires_pinned_provenance(tmp_path):
    (tmp_path / "SOURCE.json").write_text(json.dumps({
        "repository": "different/model", "revision": "moving-main", "files": {}
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="pinned"):
        validate_base_snapshot(tmp_path)


def test_parent_provenance_records_verified_parent_hashes(tmp_path):
    (tmp_path / "metadata.json").write_text("{}", encoding="utf-8")
    provenance = parent_provenance(tmp_path, {
        "artifact_hashes": {"model.safetensors": "abc"},
        "source_hashes": {"train": "def"},
        "dev_metrics": {"oracle_rank_accuracy": 0.9},
    })
    assert len(provenance["metadata_sha256"]) == 64
    assert provenance["artifact_hashes"] == {"model.safetensors": "abc"}


@pytest.mark.parametrize("field,value", [
    ("max_length", True), ("max_length", 3), ("max_length", 513),
    ("inference_batch_size", False), ("inference_batch_size", 0),
    ("inference_batch_size", -1), ("inference_batch_size", 65),
])
def test_load_rejects_invalid_inference_bounds(tmp_path, field, value):
    names = ["model.safetensors", "config.json", "tokenizer.json", "vocab.txt",
             "tokenizer_config.json", "special_tokens_map.json"]
    metadata = {"format_version": 1, "artifact_hashes": dict.fromkeys(names, "abc"),
                "max_length": 96, "inference_batch_size": 8}
    metadata[field] = value
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid inference bounds"):
        SemanticRanker.load(tmp_path)


@pytest.mark.parametrize("batch_size", [True, 0, -1, 65])
def test_constructor_rejects_invalid_batch_size(batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        SemanticRanker(_Model(), _Tokenizer(), batch_size=batch_size)


def _resume_guard_and_metadata():
    sources = {"train": "a" * 64, "dev": "b" * 64}
    guard = {"source_hashes": sources, "held_out_test_sha256": "c" * 64,
             "corpus_version": "2.1", "candidate_lexicon": {"sha256": "e" * 64}}
    metadata = {
        "base_model": PINNED_MODEL, "base_revision": PINNED_REVISION,
        "base_source": {"repository": PINNED_MODEL, "revision": PINNED_REVISION},
        "test_data_read": False, "source_hashes": sources,
        "candidate_generator": {"data_hashes": {"data/challenge/train_lexicon.json": "e" * 64}},
    }
    return guard, metadata


def test_resume_provenance_allows_explicit_safe_legacy_checkpoint():
    guard, metadata = _resume_guard_and_metadata()
    result = validate_parent_training_provenance(metadata, guard)
    assert result["compatibility"] == "safe-legacy-v2.1"
    assert result["held_out_test_read"] is False


@pytest.mark.parametrize("mutation", ["test_read", "source_hash", "revision"])
def test_resume_provenance_rejects_unsafe_legacy_checkpoint(mutation):
    guard, metadata = _resume_guard_and_metadata()
    if mutation == "test_read":
        metadata["test_data_read"] = True
    elif mutation == "source_hash":
        metadata["source_hashes"] = {"train": "c" * 64, "dev": "b" * 64}
    else:
        metadata["base_revision"] = "moving-main"
    with pytest.raises(ValueError, match="resume checkpoint"):
        validate_parent_training_provenance(metadata, guard)


def test_resume_provenance_requires_new_training_guard_to_match():
    guard, metadata = _resume_guard_and_metadata()
    metadata["training_data_guard"] = {
        "held_out_test_read": False, "corpus_version": "wrong",
        "source_hashes": guard["source_hashes"],
    }
    with pytest.raises(ValueError, match="training guard"):
        validate_parent_training_provenance(metadata, guard)


def test_resume_provenance_rejects_unapproved_parent_augmentation():
    guard, metadata = _resume_guard_and_metadata()
    metadata["augmentation_data_guard"] = {"augmentation_sha256": "d" * 64}
    with pytest.raises(ValueError, match="unapproved augmentation lineage"):
        validate_parent_training_provenance(metadata, guard)


@pytest.mark.parametrize('value', [None, 'wrong'])
def test_resume_rejects_missing_or_wrong_candidate_lexicon(value):
    guard, metadata = _resume_guard_and_metadata()
    metadata['candidate_generator']['data_hashes']['data/challenge/train_lexicon.json'] = value
    with pytest.raises(ValueError, match='candidate lexicon'):
        validate_parent_training_provenance(metadata, guard)


def test_both_trainer_snapshots_reject_unverified_file_without_reading(tmp_path, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace
    import scripts.train_semantic_ranker as semantic
    import scripts.train_sentence_ranker as sentence
    folder = tmp_path / 'checkpoints' / 'semantic_ranker'
    folder.mkdir(parents=True)
    unexpected = folder / 'unverified-large.bin'
    unexpected.touch()
    original_open = Path.open
    def guarded_open(path, *args, **kwargs):
        assert path != unexpected, 'must reject unverified source before hashing'
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded_open)
    monkeypatch.setattr(sentence, 'ROOT', tmp_path)
    args = SimpleNamespace(train=tmp_path/'train', dev=tmp_path/'dev', manifest=tmp_path/'manifest',
                           augmentation=None, augmentation_manifest=None, augmentation_attestation=None,
                           resume=folder, base_model=folder)
    with pytest.raises(ValueError, match='unverified'):
        semantic.training_source_hashes(args)
    with pytest.raises(ValueError, match='unverified'):
        sentence.runtime_proposal_artifact_hashes()
