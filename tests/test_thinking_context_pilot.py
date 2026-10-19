from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts import evaluate_thinking_context_pilot as pilot
from scripts import probe_thinking_context_resources as thinking


def candidate(source, text):
    return {"text": text, "score": 1.0, "segments": [{
        "input_start": 0, "input_end": len(source), "source": source,
        "text": text, "kind": "neural"}]}


def source_row(index):
    source = "gachi"
    candidates = [candidate(source, "가치"), candidate(source, "같이")]
    raw = {"result": source, "candidates": [{"text": source, "score": None,
                                               "segments": []}]}
    return {"id": f"r{index}", "group_id": f"g{index}", "input": source,
            "context": f"ctx{index}", "candidates": candidates,
            "raw_bypass_response": raw, "expected": ["같이"]}


def prompt(index):
    row = source_row(index)
    return thinking.snapshot_raw_record({"id": row["id"],
        "role": "natural_fit" if index < 64 else "challenge_fit",
        "source": row["input"], "context": row["context"],
        "candidates": row["candidates"], "raw_bypass_response": row["raw_bypass_response"],
        "choices": [(0, "가치"), (1, "같이")], "messages": [],
        "prompt_tokens": [8, index]})


class Generator:
    def __init__(self, tokens=(2, 3, 4)):
        self.tokens, self.closed = iter(tokens), False

    def __iter__(self): return self
    def __next__(self): return next(self.tokens), object()
    def close(self): self.closed = True


class Backend:
    def __init__(self):
        self.generator = None
        self.scored = []
        self.released = 0

    def load_model(self): return object(), object()
    def token_contract(self, tokenizer):
        return {"opener": 2, "closer": 4, "eos": 5,
                "digits": [10, 11, 12, 13, 14], "newline": [6]}
    def tokens(self, tokenizer, messages):
        context = messages[1]["content"].split("문맥: ctx", 1)[1].splitlines()[0]
        return [7, int(context)]
    def decode(self, tokenizer, tokens):
        if tokens and all(token in {1, 2, 3, 4, 5} for token in tokens):
            return "".join({1: " ", 2: "<think>", 3: "reason",
                            4: "</think>", 5: "<eos>"}[token] for token in tokens)
        return "prompt"
    def start_generation(self, model, tokens, seed):
        self.generator = Generator()
        return self.generator
    def score(self, model, tokens, digits, count):
        assert self.generator.closed
        self.scored.append(tokens)
        return [0.0, 1.0]
    def close_generation(self, generator): generator.close()
    def peak(self): return 100
    def release(self): self.released += 1


def test_row_loop_uses_new_whole_clock_and_closes_before_score(monkeypatch):
    monkeypatch.setattr(pilot, "require_resources", lambda *args, **kwargs: True)
    backend = Backend()
    result = pilot.run_reasoning_row(
        backend, object(), object(), backend.token_contract(None), prompt(0),
        pilot.time.monotonic())
    assert result["reasoning_valid"] is True
    assert result["decision"]["chosen_index"] == 1
    assert backend.generator.closed and len(backend.scored) == 1
    assert pilot.PROFILE["time_limit_seconds"] == 7200
    assert thinking.PROFILE["time_limit_seconds"] == 600


def test_late_close_and_late_score_fallback(monkeypatch):
    row, contract = prompt(0), Backend().token_contract(None)
    backend = Backend()
    checks = 0
    def late_close(*args, row_started=None, **kwargs):
        nonlocal checks
        active = row_started if row_started is not None else (args[2] if len(args) > 2 else None)
        if active is not None:
            checks += 1
            return checks < 6
        return True
    monkeypatch.setattr(pilot, "require_resources", late_close)
    result = pilot.run_reasoning_row(
        backend, object(), object(), contract, row, pilot.time.monotonic())
    assert result["completion_reason"] == "row_timeout"
    assert result["prediction_source_index"] is None and backend.scored == []
    backend = Backend()
    def score_late(*args, row_started=None, **kwargs):
        active = row_started if row_started is not None else (args[2] if len(args) > 2 else None)
        return not (active is not None and bool(backend.scored))
    monkeypatch.setattr(pilot, "require_resources", score_late)
    result = pilot.run_reasoning_row(
        backend, object(), object(), contract, row, pilot.time.monotonic())
    assert result["completion_reason"] == "row_timeout"
    assert result["digit_logits"] == [] and result["prediction_source_index"] is None


def inference(index, chosen=1):
    row = prompt(index)
    decision = thinking.single_model_decision(row, chosen)
    return {"seed": 1, "prompt_tokens": row["prompt_tokens"],
            "emitted_tokens": [2, 3, 4], "prompt_token_count": 2,
            "generated_token_count": 3, "reasoning_valid": True,
            "completion_reason": "completed_reasoning", "digit_logits": [0.0, 1.0],
            "digit_probabilities": [0.25, 0.75], "prediction_source_index": chosen,
            "decision": decision, "latency_seconds": 1.0,
            "upstream_may_prefetch_one_unused_token": True}


def test_labels_attach_only_after_complete_inference_and_gate_losses():
    records = [prompt(index) for index in range(128)]
    labels = [source_row(index) for index in range(128)]
    inferred = [inference(index) for index in range(128)]
    outcomes = pilot.attach_labels(records, labels, inferred)
    metrics = {"all": pilot.summarize(outcomes),
               "natural": pilot.summarize(outcomes[:64]),
               "challenge_single": pilot.summarize(outcomes[64:96]),
               "challenge_sentence": pilot.summarize(outcomes[96:])}
    assert pilot.pilot_gate(metrics)["passed"] is True
    outcomes[70]["regressed"] = True
    metrics["challenge_single"] = pilot.summarize(outcomes[64:96])
    assert pilot.pilot_gate(metrics)["passed"] is False
    outcomes[70]["regressed"] = False
    outcomes[70]["top5_lost"] = True
    metrics["challenge_single"] = pilot.summarize(outcomes[64:96])
    assert pilot.pilot_gate(metrics)["passed"] is False


def test_attach_rejects_partial_rows_before_label_access():
    with pytest.raises(ValueError, match="cardinality"):
        pilot.attach_labels([prompt(0)], [source_row(0)], [inference(0)])


def test_resource_gate_rejects_unaccepted_hash_before_read(tmp_path):
    with pytest.raises(ValueError, match="accepted thinking"):
        pilot.validate_resource(tmp_path / "missing", "0" * 64, tmp_path / "proof")


def test_full_mock_run_guards_labels_until_all_128_inferred(tmp_path, monkeypatch):
    backend = Backend()
    class GuardRow(dict):
        def get(self, key, default=None):
            if (key == "expected"
                    and (len(backend.scored) < 128 or backend.released < 1)):
                raise AssertionError("label read before inference and model release completed")
            return super().get(key, default)
    items = []
    for index in range(128):
        row = GuardRow(source_row(index))
        items.append({"role": "natural_fit" if index < 64 else "challenge_fit", "row": row})
    monkeypatch.setattr(pilot, "validate_resource", lambda *args: {})
    monkeypatch.setattr(pilot, "load_inputs", lambda *args: {"rows": items})
    monkeypatch.setattr(pilot, "validate_frozen_sources",
                        lambda *args, **kwargs: ({}, b"", b""))
    monkeypatch.setattr(pilot, "verify_hashes", lambda *args: None)
    monkeypatch.setattr(pilot.thinking, "package_sources", lambda: {})
    monkeypatch.setattr(pilot.thinking, "require_versions", lambda: {"mlx": "x"})
    monkeypatch.setattr(pilot, "require_resources", lambda *args, **kwargs: True)
    monkeypatch.setattr(pilot, "system_message",
                        lambda: {"role": "system", "content": "choose"})
    args = SimpleNamespace(output=tmp_path / "output", plan=thinking.PLAN,
        plan_sha256=pilot.PLAN_SHA256, resource_report=tmp_path / "resource",
        resource_report_sha256=pilot.RESOURCE_REPORT_SHA256,
        resource_verification=tmp_path / "proof", data=tmp_path / "data",
        manifest_sha256=pilot.PILOT_MANIFEST_SHA256,
        lineage_report=tmp_path / "lineage", lineage_sha256=pilot.LINEAGE_SHA256)
    report = pilot.run(args, backend_factory=lambda: backend)
    assert len(backend.scored) == 128 and backend.released == 1
    assert report["metrics"]["all"]["rows"] == 128
    assert report["gate"]["passed"] is True
    assert report["labels_used_only_after_all_128_inference_rows"] is True


def test_resource_guard_has_independent_whole_and_row_limits(monkeypatch):
    monkeypatch.setattr(pilot, "rss_bytes", lambda: 0)
    with pytest.raises(RuntimeError, match="whole time"):
        pilot.require_resources(pilot.time.monotonic() - 7201)
    assert pilot.require_resources(
        pilot.time.monotonic(), row_started=pilot.time.monotonic() - 121) is False
