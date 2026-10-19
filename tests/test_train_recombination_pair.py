from __future__ import annotations

import math
import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from scripts import train_recombination_pair as trainer


def source(row_id="fit-1", split="fit"):
    return {"id": row_id, "input": f"SRC {row_id} 1!", "split": split,
            "component_id": "component", "parent_result": "부모",
            "parent_candidates": ["부모", f"SRC {row_id} 1!"],
            "original_candidates": ["정답", "오답A", "오답B"],
            "recombined_candidates": ["정답", "오답A", "오답B", "신규"],
            "novel_candidates": ["신규"]}


def census(row_id="fit-1", split="fit", applied=True):
    return {"id": row_id, "input": f"SRC {row_id} 1!", "split": split,
            "component_id": "component",
            "score_texts": ["정답", "오답A", "오답B", "신규"],
            "scores": [0.0, 1.0, 3.0, 2.0],
            "original": {"applied": applied, "result": "오답B",
                         "candidates": ["오답B", "오답A", "정답",
                                        f"SRC {row_id} 1!"]},
            "recombined": {"applied": applied, "result": "오답B",
                           "candidates": ["오답B", "신규", "오답A", "정답",
                                          f"SRC {row_id} 1!"]}}


def response(result, candidates=None):
    return {"applied": True, "result": result,
            "candidates": candidates or [result, "SRC"]}


def prediction(row_id, frozen, arm_a, arm_b, source_text="SRC 1!"):
    return {"id": row_id, "input": source_text,
            "frozen": {"original": frozen, "recombined": frozen},
            "a_original": {"original": arm_a, "recombined": arm_a},
            "b_union": {"original": arm_b, "recombined": arm_b}}


def test_selected_candidates_keeps_all_positives_then_best_negatives():
    pool = ["p1", "low", "p2", "high", "mid"]
    selected = trainer.selected_candidates(
        pool, ["p1", "p2"], {"p1": -5, "p2": -4, "low": 0,
                              "high": 4, "mid": 2}, cap=4)
    assert selected == {"eligible": True,
                        "candidates": ["p1", "p2", "high", "mid"],
                        "positive_indices": [0, 1], "positive_count": 2,
                        "negative_count": 2}


@pytest.mark.parametrize("positive_count", [8, 9])
def test_selected_candidates_excludes_when_positives_leave_no_negative(positive_count):
    pool = [f"p{index}" for index in range(positive_count)] + ["negative"]
    selected = trainer.selected_candidates(
        pool, pool[:-1], {value: float(index) for index, value in enumerate(pool)})
    assert selected["eligible"] is False
    assert selected["reason"] == "positive_cap_leaves_no_negative"


def test_shared_selection_has_same_seeded_order_and_updates_for_both_arms():
    sources = [source(f"fit-{index}") for index in range(9)]
    censuses = [census(f"fit-{index}") for index in range(9)]
    labels = {row["id"]: ["정답"] for row in sources}
    first = trainer.build_shared_selection(sources, censuses, labels)
    second = trainer.build_shared_selection(sources, censuses, labels)
    assert first["row_ids"] == second["row_ids"]
    assert first["updates_per_arm"] == 3
    assert all(set(row["arms"]) == {"a_original", "b_union"}
               for row in first["rows"])


def test_shared_selection_excludes_both_arms_when_one_is_unsupported():
    selected = trainer.build_shared_selection(
        [source()], [census(applied=False)], {"fit-1": ["정답"]})
    assert selected["rows"] == []
    assert selected["excluded"] == {"unsupported_pool": 1}


def test_multi_positive_listwise_loss_matches_manual_formula_without_model():
    torch = pytest.importorskip("torch")
    scores = torch.tensor([0.0, 1.0, 2.0, -1.0], requires_grad=True)
    loss = trainer.multi_positive_listwise_loss(scores, [1, 2], torch)
    expected = math.log(sum(math.exp(value) for value in [0, 1, 2, -1]))
    expected -= math.log(math.exp(1) + math.exp(2))
    assert float(loss.detach()) == pytest.approx(expected, abs=2e-7)
    loss.backward()
    assert scores.grad is not None


def test_exact_parent_fallback_preserves_saved_display_order():
    row = source()
    row["parent_result"] = "saved"
    row["parent_candidates"] = ["saved", "different", row["input"]]
    assert trainer.base.apply_scores(
        row, row["original_candidates"], {}, False) == {
        "applied": False, "result": "saved",
        "candidates": ["saved", "different", row["input"]]}


def test_accumulation_groups_preserve_tail_for_actual_size_normalization():
    groups = trainer.accumulation_groups([{"id": str(index)} for index in range(11)])
    assert [len(group) for group in groups] == [4, 4, 3]
    assert [row["id"] for group in groups for row in group] == [str(i) for i in range(11)]


def test_metric_summary_and_gate_require_gains_without_any_frozen_losses():
    labels = {"one": ["gold"], "two": ["gold"], "three": ["gold"]}
    rows = [
        prediction("one", response("wrong"), response("wrong"), response("gold"), "SRC 1!"),
        prediction("two", response("wrong"), response("wrong"), response("gold"), "SRC 1!"),
        prediction("three", response("gold"), response("gold"), response("gold"), "SRC 1!"),
    ]
    metrics = {branch: {policy: trainer.metric_summary(rows, labels, branch, policy)
                        for policy in ("original", "recombined")}
               for branch in ("frozen", "a_original", "b_union")}
    gate = trainer.paired_gate(metrics)
    assert gate["passed"] is True
    assert gate["b_union_top1"] == gate["frozen_union_top1"] + 2


def test_gate_rejects_literal_signature_regression_even_with_top1_gain():
    labels = {"one": ["gold"], "two": ["gold"], "three": ["gold"]}
    rows = [
        prediction("one", response("wrong 1!"), response("wrong 1!"),
                   response("gold"), "src 1!"),
        prediction("two", response("wrong 1!"), response("wrong 1!"),
                   response("gold 1!"), "src 1!"),
        prediction("three", response("gold 1!"), response("gold 1!"),
                   response("gold 1!"), "src 1!"),
    ]
    metrics = {branch: {policy: trainer.metric_summary(rows, labels, branch, policy)
                        for policy in ("original", "recombined")}
               for branch in ("frozen", "a_original", "b_union")}
    gate = trainer.paired_gate(metrics)
    assert gate["passed"] is False
    assert gate["b_union_vs_frozen"]["literal_signature_preserved_loss_ids"] == ["one"]


def test_authorization_fails_closed_while_expected_hashes_are_pending(monkeypatch, tmp_path):
    monkeypatch.setattr(trainer, "EXPECTED_CENSUS_HASHES", {"report.json": "PENDING"})
    with pytest.raises(ValueError, match="authorization hashes are pending"):
        trainer.validate_authorization(tmp_path, tmp_path / "proof.json")


def test_full_synthetic_authorization_binds_report_proof_and_prepared(monkeypatch, tmp_path):
    census_dir = tmp_path / "census"
    census_dir.mkdir()
    prepared = tmp_path / "prepared.jsonl"
    prepared.write_text("{}\n", encoding="utf-8")
    capture = census_dir / "capture.jsonl"
    capture.write_text("{}\n", encoding="utf-8")
    capture_sha = trainer.sha256_file(capture)
    closure = census_dir / "capture.complete.json"
    closure.write_text(json.dumps({"rows": trainer.EXPECTED_ROWS,
                                   "capture_sha256": capture_sha,
                                   "source_only": True}), encoding="utf-8")
    report = census_dir / "report.json"
    report.write_text(json.dumps({
        "schema": trainer.CENSUS_SCHEMA, "completed": True,
        "counts": {"rows": trainer.EXPECTED_ROWS, "fit": trainer.FIT_ROWS,
                   "inner_hold": trainer.HOLD_ROWS},
        "adequacy_gate": {
            "passed": True,
            "inner_hold_novel_union_positive_fully_supported": 64,
            "inner_hold_parent_wrong_among_eligible": 16},
        "pins": {str(prepared): trainer.sha256_file(prepared)}}), encoding="utf-8")
    proof = census_dir / "independent-verification.json"
    verifier_sha = trainer.sha256_file(trainer.CENSUS_VERIFIER)
    gate = {"passed": True,
            "inner_hold_novel_union_positive_fully_supported": 64,
            "inner_hold_parent_wrong_among_eligible": 16}
    proof.write_text(json.dumps({
        "schema": trainer.CENSUS_PROOF_SCHEMA, "verified": True,
        "report_sha256": trainer.sha256_file(report),
        "capture_sha256": capture_sha,
        "closure_sha256": trainer.sha256_file(closure),
        "verifier_sha256": verifier_sha,
        "partition_counts": {"fit": trainer.FIT_ROWS, "inner_hold": trainer.HOLD_ROWS},
        "metrics": {"fit": {"rows": trainer.FIT_ROWS},
                    "inner_hold": {"rows": trainer.HOLD_ROWS}},
        "adequacy_gate": gate,
        "contracts": {"model_loaded": False, "tokenizer_loaded": True,
                      "source_scores_checked_before_labels": True,
                      "dev_accessed": False, "test_accessed": False,
                      "production_changed": False}}), encoding="utf-8")
    monkeypatch.setattr(trainer, "EXPECTED_CENSUS_VERIFIER_SHA256", verifier_sha)
    monkeypatch.setattr(trainer, "EXPECTED_CENSUS_HASHES", {
        "report.json": trainer.sha256_file(report),
        "capture.jsonl": capture_sha,
        "capture.complete.json": trainer.sha256_file(closure),
        "independent-verification.json": trainer.sha256_file(proof)})
    authorized = trainer.validate_authorization(census_dir, proof)
    assert authorized["prepared"] == prepared
    assert authorized["pins"][str(prepared.resolve())] == trainer.sha256_file(prepared)


def test_train_arm_uses_train_mode_and_same_tail_update_semantics(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        base_model_prefix = "base"

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0]))
            self.train_called = False
            self.ever_train_called = False

        def train(self, mode=True):
            self.train_called = bool(mode)
            self.ever_train_called |= bool(mode)
            return super().train(mode)

    model = Model()
    ranker = SimpleNamespace(model=model, tokenizer=object(), batch_size=2,
                             metadata={"inference_batch_size": 2})
    monkeypatch.setattr("hanglish.models.sentence_ranker.SentenceRanker.load",
                        lambda path: ranker)

    class Training:
        SentenceExample = lambda *args: args

        @staticmethod
        def _set_trainable_layers(model, top_layers):
            return {"trainable_parameters": 1, "unfrozen_encoder_layers": 2}

        @staticmethod
        def _optimizer(model, args, head_only):
            return torch.optim.SGD(model.parameters(), lr=0.1)

        @staticmethod
        def _loss(model, tokenizer, examples, args):
            return model.weight.square().sum()

    @contextmanager
    def operation(name, update=False):
        yield

    watchdog = SimpleNamespace(operation=operation)
    rows = [{"id": f"row-{index}", "arms": {"a_original": {
        "candidates": ["gold", "bad"], "positive_indices": [0]}}}
            for index in range(11)]
    selection = {"rows": rows, "updates_per_arm": 3}
    (tmp_path / "selection.json").write_text("{}", encoding="utf-8")
    saved = {}

    def fake_save(directory, wrapped, arm, selection_sha, probe,
                  expected_parameter_fingerprint, watchdog):
        saved.update(probe=probe, arm=arm, selection_sha=selection_sha)
        return {"ranker": wrapped,
                "verification": {"argmax_same": True, "max_score_delta": 0.0,
                                 "parameter_fingerprint": expected_parameter_fingerprint},
                "checkpoint_path": str(directory / "checkpoint")}

    monkeypatch.setattr(trainer, "save_and_verify_checkpoint", fake_save)
    result = trainer.train_arm("a_original", selection, tmp_path / "parent",
                               tmp_path, watchdog, torch, Training)
    assert model.ever_train_called is True
    assert result["updates"] == 3
    records = [json.loads(line) for line in
               (tmp_path / "a_original/updates.jsonl").read_text().splitlines()]
    assert [row["microsteps"] for row in records] == [4, 4, 3]
    assert [row["gradient_norm_before_clip"] for row in records] == pytest.approx(
        [2.0, 1.8, 1.6], abs=1e-5)
    assert result["changed_parameter_names"] == ["weight"]
    assert saved["probe"] == {"id": "row-0", "candidates": ["gold", "bad"]}


def test_checkpoint_reload_rejects_parameter_fingerprint_mismatch(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([value]))

        def save_pretrained(self, path, safe_serialization):
            (path / "model.safetensors").write_bytes(b"model")
            (path / "config.json").write_text("{}", encoding="utf-8")

    class Tokenizer:
        def save_pretrained(self, path):
            for name in ("tokenizer.json", "vocab.txt", "tokenizer_config.json",
                         "special_tokens_map.json"):
                (path / name).write_text("{}", encoding="utf-8")

    model = Model(1.0)
    ranker = SimpleNamespace(
        model=model, tokenizer=Tokenizer(),
        metadata={"inference_batch_size": 2},
        score_pairs=lambda contexts, candidates: [float(index)
                                                   for index in range(len(candidates))])
    reloaded = SimpleNamespace(
        model=Model(2.0),
        score_pairs=lambda contexts, candidates: [float(index)
                                                   for index in range(len(candidates))])
    monkeypatch.setattr("hanglish.models.sentence_ranker.SentenceRanker.load",
                        lambda path: reloaded)

    @contextmanager
    def operation(name, update=False):
        yield

    directory = tmp_path / "arm"
    directory.mkdir()
    with pytest.raises(ValueError, match="checkpoint reload score parity differs"):
        trainer.save_and_verify_checkpoint(
            directory, ranker, "a_original", "selection-sha",
            {"id": "probe", "candidates": ["one", "two"]},
            trainer.parameter_fingerprint(model),
            SimpleNamespace(operation=operation))


def test_self_test_does_not_open_model_or_data():
    assert trainer.self_test() == {
        "schema": trainer.SCHEMA + "-self-test", "passed": True,
        "model_loaded": False, "data_opened": False}
