import hashlib
import json

import pytest
from transformers.tokenization_utils_base import BatchEncoding

import scripts.probe_roberta_final_choice_tokenizer as module


class Tokenizer:
    def __init__(self, lengths, reserved=4):
        self.lengths = lengths
        self.reserved = reserved
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return {"input_ids": list(range(self.lengths[text]))}

    def num_special_tokens_to_add(self, pair=False):
        assert pair is True
        return self.reserved


def row(scores=(1.0, .5, None), accepted=(0,), old=True):
    candidates = [{"text": text, "score": score} for text, score in
                  zip(("first", "second", "raw"), scores)]
    return {
        "id": "id", "group_id": "group", "candidates": candidates,
        "finite_indices": [i for i, score in enumerate(scores) if score is not None],
        "accepted_positive_indices": list(accepted),
        "selector_eligible": old, "all_candidates_can_score": old,
    }


def test_whole_candidate_budget_has_no_truncation_and_checks_every_candidate():
    tokenizer = Tokenizer({"first": 92, "second": 93, "raw": 1})
    lengths, reserved, all_fit = module.candidate_budget(row(), tokenizer)
    assert lengths == [92, 93, 1]
    assert reserved == 4
    assert all_fit is False
    assert [call[1] for call in tokenizer.calls] == [
        {"add_special_tokens": False, "truncation": False},
    ] * 3


def test_normalization_and_none_original_length_are_recorded():
    value = row()
    value["candidates"][2]["text"] = " raw   value "
    tokenizer = Tokenizer({"first": 1, "second": 2, "raw value": 3})
    diagnosed = module.diagnose_row(value, tokenizer, "natural_fit")
    assert diagnosed["candidate_token_lengths"] == [1, 2, 3]
    assert diagnosed["none_score_indices"] == [2]
    assert diagnosed["none_score_token_lengths"] == [3]


def test_normalized_length_accepts_huggingface_batch_encoding():
    class BatchTokenizer:
        def __call__(self, text, **kwargs):
            assert text == "two words"
            assert kwargs == {"add_special_tokens": False, "truncation": False}
            return BatchEncoding({"input_ids": [11, 12, 13]})

    assert module.normalized_token_length(
        BatchTokenizer(), "  two   words  ",
    ) == 3


def test_finite_original_index_is_required_for_new_eligibility():
    value = row(scores=(None, .5, .2), accepted=(1,))
    tokenizer = Tokenizer({"first": 1, "second": 1, "raw": 1})
    diagnosed = module.diagnose_row(value, tokenizer, "natural_fit")
    assert diagnosed["finite_indices"] == [1, 2]
    assert diagnosed["new_all_candidates_fit"] is True
    assert diagnosed["new_eligible"] is False


def test_complete_stored_indices_and_accepted_subset_are_enforced():
    value = row()
    value["finite_indices"] = [0]
    with pytest.raises(ValueError, match="finite or accepted"):
        module.diagnose_row(value, Tokenizer({"first": 1, "second": 1, "raw": 1}),
                            "natural_fit")
    value = row(accepted=(2,))
    with pytest.raises(ValueError, match="finite or accepted"):
        module.diagnose_row(value, Tokenizer({"first": 1, "second": 1, "raw": 1}),
                            "natural_fit")


class LabelOrderRow(dict):
    budget_finished = False

    def get(self, key, default=None):
        if key == "accepted_positive_indices" and not self.budget_finished:
            raise AssertionError("labels consulted before token budget")
        value = super().get(key, default)
        if key == "finite_indices":
            self.budget_finished = True
        return value


def test_alignment_is_consulted_only_after_complete_source_only_budget():
    value = LabelOrderRow(row())
    diagnosed = module.diagnose_row(
        value, Tokenizer({"first": 1, "second": 1, "raw": 1}), "natural_fit",
    )
    assert diagnosed["new_eligible"] is True
    assert diagnosed["parent_correct"] is True


def test_summary_distinguishes_positive_absence_and_eligibility_changes():
    rows = [
        {"role": "natural_fit", "group_id": "g", "old_eligible": False,
         "new_eligible": True, "parent_correct": False,
         "accepted_positive_count": 0, "none_score_indices": [2],
         "candidate_token_lengths": [1, 2, 3]},
        {"role": "natural_fit", "group_id": "g", "old_eligible": True,
         "new_eligible": False, "parent_correct": True,
         "accepted_positive_count": 1, "none_score_indices": [],
         "candidate_token_lengths": [5]},
    ]
    summary = module.summarize(rows)
    assert summary["groups"] == 1
    assert summary["old_to_new_eligible"] == 1
    assert summary["old_to_new_ineligible"] == 1
    assert summary["new_eligible_without_positive"] == 1
    assert summary["rows_with_none_original"] == 1


def test_checkpoint_preflight_verifies_all_artifacts_and_loads_tokenizer_only(
        tmp_path, monkeypatch):
    artifact_hashes = {}
    for name in module.REQUIRED_ARTIFACTS:
        raw = ("artifact:" + name).encode()
        (tmp_path / name).write_bytes(raw)
        artifact_hashes[name] = hashlib.sha256(raw).hexdigest()
    metadata = {
        "format_version": 1, "base_model": "klue/roberta-small",
        "max_length": 96, "artifact_hashes": artifact_hashes,
    }
    metadata_raw = json.dumps(metadata).encode()
    (tmp_path / "metadata.json").write_bytes(metadata_raw)
    monkeypatch.setattr(module, "METADATA_SHA256",
                        hashlib.sha256(metadata_raw).hexdigest())
    monkeypatch.setattr(module, "MODEL_SHA256",
                        artifact_hashes["model.safetensors"])
    calls = []

    def tokenizer_factory(path, **kwargs):
        calls.append((path, kwargs))
        return "tokenizer"

    tokenizer, _, _, observed = module.load_tokenizer_checkpoint(
        tmp_path, tokenizer_factory,
    )
    assert tokenizer == "tokenizer"
    assert observed == artifact_hashes
    assert calls == [(tmp_path, {
        "local_files_only": True, "trust_remote_code": False,
    })]
