"""Synthetic TRAIN replay checks; no real diagnostic labels or model weights."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from scripts import long_context_lora_evaluation as producer


PATH = Path(__file__).resolve().parents[1] / ".omx/artifacts/long-context-lora-work-v4/verify_training_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("diagnostics_v3", PATH)
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


class Tokenizer:
    def apply_chat_template(self, messages, *, tokenize, **kwargs):
        assert kwargs["enable_thinking"] is False
        return list(range(10)) if tokenize else "prompt"

    def encode(self, text, **kwargs):
        return list(range(10)) + ([] if text == "prompt" else [100 + int(text[-1])])


def item(index):
    source = "imnida"
    def candidate(text, score=1.):
        return {"text": text, "score": score, "segments": [{"source": source,
            "text": text, "kind": "lexicon", "input_start": 0, "input_end": len(source)}]}
    stratum = ("natural" if index < 512 else "challenge_lower_ascii_context_single"
               if index < 576 else "challenge_whitespace_sentence")
    return {"role": "natural_fit" if index < 512 else "challenge_fit", "stratum": stratum,
        "row": {"id": str(index), "input": source, "context": "문맥", "result": "임니다",
                "expected": ["입니다"],
                "candidates": [candidate("임니다"), candidate("입니다"), candidate(source, None)],
                "raw_bypass_response": {"result": source, "candidates": [candidate(source, None)]}}}


def declared(items, parent, base, trained):
    metrics = {name: producer.score_predictions(items, rows) for name, rows in (
        ("fresh_parent", parent), ("fresh_base", base), ("final_adapter", trained))}
    before = metrics["fresh_parent"]["metrics"]["all"]["top1"]
    after = metrics["final_adapter"]["metrics"]["all"]["top1"]
    gate = {"passed": after - before >= 5, "net_gain": after - before,
            "required_net_gain": 5, "parent_correct": before, "trained_correct": after,
            "corrections": metrics["final_adapter"]["gate"]["corrections"],
            "regressions": metrics["final_adapter"]["gate"]["regressions"]}
    return metrics, gate


def logits(record, index):
    return [3. if source_index == index else -2. for source_index, _ in record["choices"]]


@pytest.fixture
def evidence(monkeypatch):
    monkeypatch.setattr(producer, "system_message", lambda: {"role": "system", "content": "select"})
    items = [item(index) for index in range(640)]
    items[6]["row"]["expected"] = ["임니다"]  # Six corrections and one regression: net five.
    items[636]["row"]["expected"] = ["임니다", "입니다"]
    items[637]["row"]["candidates"][0]["segments"][0]["kind"] = "raw"
    items[638]["row"]["candidates"] = items[638]["row"]["candidates"][:1]
    records = producer.prepare_diagnostic(items, Tokenizer())
    records[639].update(tokens=list(range(1025)), digit_ids=[], eligible=False,
                        fallback_reason="prompt_over_1024")
    parent = [producer.apply_logits(record, None) for record in records]
    base = [producer.apply_logits(record, logits(record, 0) if record["eligible"] else None)
            for record in records]
    trained = [producer.apply_logits(record, logits(record, 1 if index < 7 or index == 637 else 0)
                                    if record["eligible"] else None)
               for index, record in enumerate(records)]
    metrics, gate = declared(items, parent, base, trained)
    return {"diagnostic_items": items, "parent": parent, "base": base, "trained": trained,
            "declared_metrics": metrics, "declared_gate": gate, "prepared_records": records}


def test_independent_counts_multigold_regression_fallback_and_identity(evidence, monkeypatch):
    # Calling producer scoring/selection from the verifier would now fail.
    def forbidden(*args, **kwargs):
        raise AssertionError("producer scoring or selection reused")
    monkeypatch.setattr(producer, "score_predictions", forbidden)
    monkeypatch.setattr(producer, "apply_logits", forbidden)
    result = verifier.verify_diagnostics(**evidence)
    assert result["train_gate"] == evidence["declared_gate"]
    assert result["train_gate"]["passed"] is True
    assert result["train_gate"]["corrections"] == 6
    assert result["train_gate"]["regressions"] == 1
    counts = result["metrics"]["final_adapter"]["metrics"]["all"]
    assert counts["rows"] == 640 and counts["fallbacks"] == 2
    identities = [{"position": index + 1, "role": item["role"], "id": item["row"]["id"]}
                  for index, item in enumerate(evidence["diagnostic_items"])]
    expected = hashlib.sha256(json.dumps(identities, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    assert result["row_identity_sha256"] == expected


@pytest.mark.parametrize("field,value", [
    ("source", "different"), ("context", "changed"), ("id", "other"),
    ("role", "different"), ("chosen_index", True), ("proposed_index", False),
    ("reason", "parent_baseline"), ("result", "임니다"),
    ("raw_sha256_before", "0" * 64), ("raw_sha256_after", "0" * 64),
    ("raw_canonical_json_before", "{}"), ("raw_canonical_json_after", "{}"),
])
def test_prediction_field_drift_rejected(evidence, field, value):
    evidence["trained"][0][field] = value
    with pytest.raises(verifier.DiagnosticVerificationError):
        verifier.verify_diagnostics(**evidence)


@pytest.mark.parametrize("target", ["candidate_score", "candidate_segment", "raw_object", "choice_order"])
def test_full_object_preservation_rejected(evidence, target):
    row = evidence["trained"][0] = deepcopy(evidence["trained"][0])
    if target == "candidate_score":
        row["candidates"][0]["score"] += .1
    elif target == "candidate_segment":
        row["candidates"][0]["segments"][0]["kind"] = "raw"
    elif target == "raw_object":
        row["raw_bypass_response"]["extra"] = "drift"
    else:
        row["choices"].reverse()
    with pytest.raises(verifier.DiagnosticVerificationError):
        verifier.verify_diagnostics(**evidence)


@pytest.mark.parametrize("values", [None, [], [0.], [1., float("nan")],
                                    [float("inf"), 1.], [True, 1.]])
def test_eligible_logits_cannot_be_skipped_or_malformed(evidence, values):
    evidence["trained"][0]["logits"] = values
    with pytest.raises(verifier.DiagnosticVerificationError):
        verifier.verify_diagnostics(**evidence)


@pytest.mark.parametrize("target", ["parent", "base", "trained", "prepared_records", "diagnostic_items"])
def test_row_order_is_bound(evidence, target):
    evidence[target][0], evidence[target][1] = evidence[target][1], evidence[target][0]
    with pytest.raises(verifier.DiagnosticVerificationError):
        verifier.verify_diagnostics(**evidence)


def test_global_duplicate_id_and_wrong_strata_rejected(evidence):
    evidence["diagnostic_items"][512]["row"]["id"] = "0"
    with pytest.raises(verifier.DiagnosticVerificationError):
        verifier.verify_diagnostics(**evidence)
    evidence["diagnostic_items"][512]["row"]["id"] = "512"
    evidence["diagnostic_items"][512]["stratum"] = "natural"
    with pytest.raises(verifier.DiagnosticVerificationError, match="stratum"):
        verifier.verify_diagnostics(**evidence)


def test_ineligible_and_parent_cannot_contain_logits(evidence):
    evidence["trained"][639]["logits"] = [1., 2.]
    with pytest.raises(verifier.DiagnosticVerificationError):
        verifier.verify_diagnostics(**evidence)
    evidence["trained"][639]["logits"] = None
    evidence["parent"][0]["logits"] = [1., 2.]
    with pytest.raises(verifier.DiagnosticVerificationError):
        verifier.verify_diagnostics(**evidence)


def test_protected_proposal_cannot_be_forced(evidence):
    assert evidence["trained"][637]["reason"] == "protected_or_unaligned"
    evidence["trained"][637]["chosen_index"] = 1
    with pytest.raises(verifier.DiagnosticVerificationError):
        verifier.verify_diagnostics(**evidence)


def test_prepared_fallback_cannot_drift(evidence):
    evidence["prepared_records"][639]["eligible"] = True
    with pytest.raises(verifier.DiagnosticVerificationError, match="eligibility"):
        verifier.verify_diagnostics(**evidence)


@pytest.mark.parametrize("target", ["metrics", "gate", "strata", "denominator"])
def test_declared_results_are_recomputed(evidence, target):
    if target == "gate":
        evidence["declared_gate"]["passed"] = False
    elif target == "strata":
        evidence["declared_metrics"]["final_adapter"]["strata"]["natural"]["top1"] += 1
    elif target == "denominator":
        evidence["trained"].pop()
    else:
        evidence["declared_metrics"]["final_adapter"]["metrics"]["all"]["top1"] += 1
    with pytest.raises(verifier.DiagnosticVerificationError):
        verifier.verify_diagnostics(**evidence)


def test_quality_rejection_is_evidence_but_not_success(evidence):
    evidence["trained"] = deepcopy(evidence["base"])
    evidence["declared_metrics"], evidence["declared_gate"] = declared(
        evidence["diagnostic_items"], evidence["parent"], evidence["base"], evidence["trained"])
    result = verifier.verify_diagnostics(**evidence)
    assert result["train_gate"]["passed"] is False
    assert result["train_gate"]["net_gain"] == 0


def test_ties_follow_prompt_order(evidence):
    record = evidence["prepared_records"][8]
    evidence["trained"][8] = producer.apply_logits(record, [1.] * len(record["choices"]))
    evidence["declared_metrics"], evidence["declared_gate"] = declared(
        evidence["diagnostic_items"], evidence["parent"], evidence["base"], evidence["trained"])
    verifier.verify_diagnostics(**evidence)
    assert evidence["trained"][8]["proposed_index"] == record["choices"][0][0]


@pytest.mark.parametrize("bad_identity", [False, True])
def test_path_adapter_reconstructs_locally_and_binds_identity(evidence, tmp_path, monkeypatch,
                                                            bad_identity):
    directory, attempt, data = tmp_path / "run", tmp_path / "attempt", tmp_path / "data"
    (directory / "base-diagnostic").mkdir(parents=True)
    attempt.mkdir(); data.mkdir()
    def write(path, value, *, jsonl=False):
        path.write_text("\n".join(json.dumps(row) for row in value) if jsonl else json.dumps(value))
    write(data / "diagnostic.jsonl", evidence["diagnostic_items"], jsonl=True)
    write(directory / "base-diagnostic/parent_predictions.jsonl", evidence["parent"], jsonl=True)
    write(directory / "base-diagnostic/base_predictions.jsonl", evidence["base"], jsonl=True)
    write(attempt / "trained_predictions.jsonl", evidence["trained"], jsonl=True)
    write(attempt / "manifest.json", {"diagnostic_metrics": evidence["declared_metrics"],
                                     "train_gate": evidence["declared_gate"]})
    expected = verifier.verify_diagnostics(**evidence)
    write(directory / "base-diagnostic/manifest.json", {"diagnostic_identity_sha256":
        "0" * 64 if bad_identity else expected["row_identity_sha256"]})
    calls = []
    tokenizer = Tokenizer()
    def load(path, **kwargs):
        calls.append((path, kwargs))
        return tokenizer
    def prepare(items, supplied_tokenizer):
        assert supplied_tokenizer is tokenizer
        assert items == evidence["diagnostic_items"]
        return evidence["prepared_records"]
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=load)))
    monkeypatch.setattr(producer, "prepare_diagnostic", prepare)
    if bad_identity:
        with pytest.raises(verifier.DiagnosticVerificationError, match="identity digest"):
            verifier.verify(directory, attempt, data)
    else:
        assert verifier.verify(directory, attempt, data) == {"passed": True, **expected}
    assert calls == [(verifier.MODEL, {"local_files_only": True, "trust_remote_code": False})]
