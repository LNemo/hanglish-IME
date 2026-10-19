from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts import evaluate_consensus_context_pilot as pilot


def candidate(source, text):
    return {"text": text, "score": 1.0, "segments": [{
        "input_start": 0, "input_end": len(source), "source": source,
        "text": text, "kind": "neural"}]}


def record(index, *, choices=2, expected=None):
    source = "gachi"
    candidates = [candidate(source, "가치"), candidate(source, "같이")][:choices]
    raw = {"result": source, "candidates": [{
        "text": source, "score": None, "segments": [{
            "input_start": 0, "input_end": len(source), "source": source,
            "text": source, "kind": "original"}]}]}
    return {"id": f"r{index}", "role": "natural_fit", "source": source,
            "context": "ctx", "candidates": candidates, "raw_bypass_response": raw,
            "choices": list(enumerate(c["text"] for c in candidates)),
            "messages": [{"role": "system", "content": "x"},
                         {"role": "user", "content": "prompt"}],
            "expected": expected or ["같이"]}


class FakeBackend:
    def __init__(self, long_index=None):
        self.long_index = long_index
        self.loaded = []
        self.cleared = 0
        self.inferences = 0

    def load_model(self, adapter=False):
        self.loaded.append(adapter)
        return object(), object()

    def digit_ids(self, _):
        return [1, 2, 3, 4, 5]

    def tokens(self, _, messages):
        content = messages[1]["content"]
        index = int(content.removeprefix("prompt")) if content.startswith("prompt") else 0
        return list(range(257)) if index == self.long_index else [1, index, 2]

    def infer(self, model, tokens, digit_ids, count):
        self.inferences += 1
        return [0.0, 1.0][:count]

    def peak(self):
        return 100

    def clear(self):
        self.cleared += 1


def stage_records():
    rows = [record(i) for i in range(128)]
    for i, row in enumerate(rows):
        row["messages"][1]["content"] = f"prompt{i}"
    rows[1] = record(1, choices=1)
    rows[1]["messages"][1]["content"] = "prompt1"
    return rows


def test_infer_all_accounts_for_eligible_single_choice_and_oversize(monkeypatch):
    monkeypatch.setattr(pilot, "require_resources", lambda *args: None)
    backend = FakeBackend(long_index=2)
    rows = stage_records()
    base = pilot.infer_all_stage(backend, rows, adapter=False, started=0)
    trained = pilot.infer_all_stage(backend, rows, adapter=True, expected=base, started=0)
    assert len(base) == len(trained) == 128
    assert base[1]["reason"] == "insufficient_choices"
    assert base[2]["reason"] == "prompt_too_long"
    assert sum(row["eligible"] for row in base) == 126
    assert backend.inferences == 252
    assert backend.loaded == [False, True]
    assert backend.cleared == 2


def test_stage_rejects_adapter_token_drift(monkeypatch):
    monkeypatch.setattr(pilot, "require_resources", lambda *args: None)
    backend = FakeBackend()
    rows = stage_records()
    base = pilot.infer_all_stage(backend, rows, adapter=False, started=0)
    base[0]["tokens"] = [99]
    with pytest.raises(RuntimeError, match="tokens or choices"):
        pilot.infer_all_stage(backend, rows, adapter=True, expected=base, started=0)


def test_stage_rejects_truncated_choice_logits(monkeypatch):
    monkeypatch.setattr(pilot, "require_resources", lambda *args: None)
    backend = FakeBackend()
    backend.infer = lambda *args: [0.0]
    with pytest.raises(RuntimeError, match="logit vector cardinality"):
        pilot.infer_all_stage(backend, stage_records(), adapter=False, started=0)
    assert backend.cleared == 1


def paired(rows, prediction=1):
    result = []
    for index, row in enumerate(rows):
        result.append({"record_index": index, "choices": row["choices"],
                       "tokens": [1, index], "eligible": len(row["choices"]) > 1,
                       "reason": "inferred" if len(row["choices"]) > 1 else "insufficient_choices",
                       "latency_seconds": 0.1 if len(row["choices"]) > 1 else 0.0,
                       "logits": [0.0, 1.0] if len(row["choices"]) > 1 else [],
                       "probabilities": [0.25, 0.75] if len(row["choices"]) > 1 else [],
                       "prediction_source_index": prediction if len(row["choices"]) > 1 else None})
    return result


def test_labels_are_consulted_only_by_post_inference_evaluation():
    rows = stage_records()
    base = paired(rows)
    trained = deepcopy(base)
    labels = [{"id": row["id"], "expected": row.pop("expected")} for row in rows]
    assert all("expected" not in row for row in rows)
    outcomes = pilot.evaluate_rows(rows, labels, base, trained)
    assert len(outcomes) == 128
    assert outcomes[0]["corrected"] is True
    assert outcomes[1]["chosen_index"] == 0
    assert outcomes[1]["ineligible_reason"] == "insufficient_choices"
    assert all(row["labels_used_after_both_inference_stages"] for row in outcomes)


def test_evaluate_rejects_pair_mismatch_and_malformed_labels():
    rows = stage_records()
    base = paired(rows)
    trained = deepcopy(base)
    labels = [{"id": row["id"], "expected": row["expected"]} for row in rows]
    trained[3]["choices"] = [(0, "changed")]
    with pytest.raises(RuntimeError, match="structurally"):
        pilot.evaluate_rows(rows, labels, base, trained)
    trained = deepcopy(base)
    labels[3]["expected"] = []
    with pytest.raises(ValueError, match="expected labels"):
        pilot.evaluate_rows(rows, labels, base, trained)


def test_gate_blocks_any_row_regression_or_top5_loss():
    clean = {"rows": 128, "before_top1": 50, "top1": 51,
             "before_top5": 100, "top5": 100, "corrections": ["a"],
             "regressions": [], "top5_losses": [], "actions": 1,
             "eligible": 100, "ineligible": 28,
             "base_inference_seconds": 1.0, "trained_inference_seconds": 1.0}
    metrics = {"all": deepcopy(clean), "natural": deepcopy(clean),
               "challenge_single": deepcopy(clean), "challenge_sentence": deepcopy(clean)}
    assert pilot.pilot_gate(metrics)["passed"] is True
    metrics["challenge_single"]["regressions"] = ["bad"]
    assert pilot.pilot_gate(metrics)["passed"] is False
    metrics["challenge_single"]["regressions"] = []
    metrics["natural"]["top5_losses"] = ["lost"]
    assert pilot.pilot_gate(metrics)["passed"] is False


def test_resource_gate_rejects_stale_or_failed_report_before_models(tmp_path):
    report = {"completed": False}
    raw = (pilot.json.dumps(report) + "\n").encode()
    path = tmp_path / "report.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="accepted resource"):
        pilot.validate_resource_smoke(path, pilot.digest(raw), tmp_path / "missing")


def test_full_run_reads_labels_only_after_both_model_stages(tmp_path, monkeypatch):
    backend = FakeBackend()

    class GuardRow(dict):
        def get(self, key, default=None):
            if key == "expected" and backend.cleared < 2:
                raise AssertionError("label read before both stages completed")
            return super().get(key, default)

    items = []
    for index in range(128):
        row = record(index, expected=["가치"])
        row["id"] = f"r{index}"
        row["group_id"] = f"g{index}"
        row["input"] = row.pop("source")
        row["raw_bypass_response"] = row["raw_bypass_response"]
        row["context"] = "ctx"
        items.append({"role": "natural_fit" if index < 64 else "challenge_fit",
                      "row": GuardRow(row)})
    monkeypatch.setattr(pilot, "validate_resource_smoke", lambda *args: {})
    monkeypatch.setattr(pilot, "load_inputs", lambda *args: {"rows": items})
    monkeypatch.setattr(pilot, "validate_frozen_sources", lambda *args, **kwargs: ({}, b"", b""))
    monkeypatch.setattr(pilot, "verify_hashes", lambda *args: None)
    monkeypatch.setattr(pilot, "require_resources", lambda *args: None)
    monkeypatch.setattr(pilot, "system_message",
                        lambda: {"role": "system", "content": "choose"})
    monkeypatch.setattr(pilot.importlib.metadata, "version", lambda _: "test")
    args = SimpleNamespace(
        output=tmp_path / "output",
        plan=pilot.ROOT / ".omx/artifacts/consensus-context-plan-v1.md",
        plan_sha256=pilot.PLAN_SHA256,
        resource_report=tmp_path / "resource.json",
        resource_report_sha256=pilot.RESOURCE_REPORT_SHA256,
        resource_verification=tmp_path / "verification.json",
        data=tmp_path / "data", manifest_sha256=pilot.PILOT_MANIFEST_SHA256,
        lineage_report=tmp_path / "lineage.json", lineage_sha256=pilot.LINEAGE_SHA256)
    report = pilot.run(args, backend_factory=lambda: backend)
    assert backend.loaded == [False, True]
    assert backend.cleared == 2
    assert report["metrics"]["all"]["rows"] == 128
    assert report["gate"]["passed"] is True
