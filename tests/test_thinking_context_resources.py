from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts import probe_thinking_context_resources as probe


def candidate(source, text):
    return {"text": text, "score": 1.0, "segments": [{
        "input_start": 0, "input_end": len(source), "source": source,
        "text": text, "kind": "neural"}]}


def record(index, role="natural_fit", prompt_length=10):
    source = "gachi"
    candidates = [candidate(source, "가치"), candidate(source, "같이")]
    return probe.snapshot_raw_record({"id": f"r{index}", "role": role, "source": source,
            "context": f"ctx{index}", "candidates": candidates,
            "choices": [(0, "가치"), (1, "같이")],
            "messages": [], "prompt_tokens": list(range(prompt_length)),
            "raw_bypass_response": {"result": source, "candidates": [{
                "text": source, "score": None, "segments": []}]}})


def test_seed_uses_exact_frozen_source_bytes():
    expected = int(probe.hashlib.sha256(
        (probe.SALT + "\0").encode()
        + probe.json.dumps(["gachi", "ctx"], ensure_ascii=False).encode()
    ).hexdigest()[:8], 16)
    assert probe.seed_for("gachi", "ctx") == expected


def test_reasoning_format_accepts_only_natural_nonempty_close():
    decode = lambda values: "".join({1: " ", 2: "<think>", 3: "reason",
                                      4: "</think>", 5: "<eos>"}[v] for v in values)
    complete = probe.inspect_reasoning([1, 2, 3, 4], 2, 4, 5, decode)
    assert complete["valid"] is True and complete["body"] == "reason"
    cases = [
        ([3], "missing_opener"), ([2, 2], "nested_opener"),
        ([2, 1, 4], "empty_reasoning"), ([2, 3, 5], "eos_before_close"),
    ]
    for tokens, reason in cases:
        assert probe.inspect_reasoning(tokens, 2, 4, 5, decode)["reason"] == reason
    assert probe.inspect_reasoning([2, 3], 2, 4, 5, decode)["terminal"] is False


def test_selection_is_longest_then_fixed_source_order():
    rows = [record(i, prompt_length=10 + i) for i in range(64)]
    rows += [record(i, "challenge_fit", 50) for i in range(64, 128)]
    selected = probe.select_rows(rows)
    assert [index for index, _ in selected] == [63, 0, 64, 96]
    rows[63]["prompt_tokens"] = list(range(1025))
    assert probe.select_rows(rows)[0][0] == 62


def test_single_model_move_preserves_objects_and_raw():
    row = record(0)
    before = deepcopy(row)
    decision = probe.single_model_decision(row, 1)
    assert decision["chosen_index"] == 1
    assert decision["candidates"][0] is row["candidates"][1]
    assert decision["candidates"][1] is row["candidates"][0]
    assert row == before
    fallback = probe.single_model_decision(row, None)
    assert fallback["chosen_index"] == 0
    assert fallback["raw_bypass_response"] == row["raw_bypass_response"]
    assert fallback["raw_canonical_json_before"] == fallback["raw_canonical_json_after"]
    assert fallback["raw_sha256_before"] == fallback["raw_sha256_after"]


class FakeGenerator:
    def __init__(self, values):
        self.values = iter(values)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        value = next(self.values)
        return value, object()

    def close(self):
        self.closed = True


class FakeBackend:
    def __init__(self, generated=(2, 3, 4), logits=(0.0, 1.0)):
        self.generated, self.logits = generated, logits
        self.generator = None
        self.closed = 0
        self.scored = []
        self.released = 0

    def start_generation(self, model, prompt, seed):
        self.generator = FakeGenerator(self.generated)
        return self.generator

    def decode(self, tokenizer, values):
        return "".join({1: " ", 2: "<think>", 3: "reason",
                        4: "</think>", 5: "<eos>"}[v] for v in values)

    def score(self, model, full_tokens, digit_ids, count):
        assert self.generator is not None and self.generator.closed
        self.scored.append(full_tokens)
        return list(self.logits)

    def close_generation(self, generator):
        generator.close()
        self.closed += 1

    def peak(self):
        return 100

    def release(self):
        self.released += 1


def test_reasoning_row_closes_before_exact_recompute(monkeypatch):
    monkeypatch.setattr(probe, "require_resources", lambda *args, **kwargs: True)
    backend = FakeBackend()
    row = record(0)
    contract = {"opener": 2, "closer": 4, "eos": 5,
                "digits": [10, 11, 12, 13, 14], "newline": [6, 6]}
    outcome = probe.run_reasoning_row(
        backend, object(), object(), contract, row, probe.time.monotonic())
    assert backend.generator.closed and backend.closed == 1
    assert backend.scored == [row["prompt_tokens"] + [2, 3, 4, 6, 6]]
    assert outcome["reasoning_valid"] is True
    assert outcome["prediction_source_index"] == 1
    assert outcome["decision"]["chosen_index"] == 1


def test_incomplete_eos_timeout_never_scores_or_forces_close(monkeypatch):
    contract = {"opener": 2, "closer": 4, "eos": 5,
                "digits": [10, 11], "newline": [6]}
    row = record(0)
    for generated, resource_values, reason in (
        ((2, 3, 5), [True] * 10, "eos_before_close"),
        ((2, 3), [True] * 10, "token_limit_incomplete"),
        ((2, 3, 3), [True, True, True, False], "row_timeout"),
    ):
        backend = FakeBackend(generated=generated)
        values = iter(resource_values)
        monkeypatch.setattr(probe, "require_resources",
                            lambda *args, **kwargs: next(values, True))
        outcome = probe.run_reasoning_row(
            backend, object(), object(), contract, row, probe.time.monotonic())
        assert outcome["completion_reason"] == reason
        assert outcome["decision"]["chosen_index"] == 0
        assert backend.scored == [] and backend.generator.closed


def test_late_close_and_final_score_overrun_both_fallback(monkeypatch):
    contract = {"opener": 2, "closer": 4, "eos": 5,
                "digits": [10, 11], "newline": [6]}
    row = record(0)
    backend = FakeBackend()
    row_checks = 0

    def late_close(*args, row_started=None, **kwargs):
        nonlocal row_checks
        active_row = row_started if row_started is not None else (
            args[2] if len(args) > 2 else None)
        if active_row is not None:
            row_checks += 1
            return row_checks < 6
        return True

    monkeypatch.setattr(probe, "require_resources", late_close)
    outcome = probe.run_reasoning_row(
        backend, object(), object(), contract, row, probe.time.monotonic())
    assert outcome["completion_reason"] == "row_timeout"
    assert outcome["prediction_source_index"] is None and backend.scored == []

    backend = FakeBackend()
    def score_deadline(*args, row_started=None, **kwargs):
        active_row = row_started if row_started is not None else (
            args[2] if len(args) > 2 else None)
        return not (active_row is not None and bool(backend.scored))

    monkeypatch.setattr(probe, "require_resources", score_deadline)
    outcome = probe.run_reasoning_row(
        backend, object(), object(), contract, row, probe.time.monotonic())
    assert outcome["completion_reason"] == "row_timeout"
    assert outcome["prediction_source_index"] is None
    assert outcome["digit_logits"] == [] and len(backend.scored) == 1


def test_malformed_final_logits_fail_after_generator_close(monkeypatch):
    monkeypatch.setattr(probe, "require_resources", lambda *args, **kwargs: True)
    backend = FakeBackend(logits=(float("nan"), 1.0))
    with pytest.raises(RuntimeError, match="candidate logits"):
        probe.run_reasoning_row(
            backend, object(), object(),
            {"opener": 2, "closer": 4, "eos": 5,
             "digits": [10, 11], "newline": [6]},
            record(0), probe.time.monotonic())
    assert backend.generator.closed


def test_raw_snapshot_precedes_inference_and_detects_mutation():
    row = record(0)
    row["raw_bypass_response"]["result"] = "changed"
    with pytest.raises(RuntimeError, match="raw bypass"):
        probe.single_model_decision(row, None)
    row = record(0)
    row["candidates"][0]["score"] = 99
    with pytest.raises(RuntimeError, match="candidate objects"):
        probe.single_model_decision(row, None)


def test_tokenizer_wrapper_must_return_plain_integer_list():
    backend = probe.ThinkingBackend.__new__(probe.ThinkingBackend)

    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            assert kwargs["return_dict"] is False
            return {"input_ids": [1, 2]}

    with pytest.raises(RuntimeError, match="integer token list"):
        backend.tokens(Tokenizer(), [])


def test_resource_guard_checks_whole_row_and_memory(monkeypatch):
    monkeypatch.setattr(probe, "rss_bytes", lambda: 0)
    with pytest.raises(RuntimeError, match="whole time"):
        probe.require_resources(probe.time.monotonic() - 601)
    monkeypatch.setattr(probe, "rss_bytes", lambda: probe.PROFILE["rss_limit_bytes"] + 1)
    with pytest.raises(RuntimeError, match="RSS"):
        probe.require_resources(probe.time.monotonic())
    monkeypatch.setattr(probe, "rss_bytes", lambda: 0)
    backend = FakeBackend()
    backend.peak = lambda: probe.PROFILE["mlx_peak_limit_bytes"] + 1
    with pytest.raises(RuntimeError, match="MLX"):
        probe.require_resources(probe.time.monotonic(), backend)
    assert probe.require_resources(probe.time.monotonic(), row_started=probe.time.monotonic() - 121) is False


def test_module_import_and_pure_tests_do_not_load_mlx():
    assert "mlx.core" not in probe.sys.modules


def test_full_mock_run_selects_fixed_rows_without_reading_labels(tmp_path, monkeypatch):
    class GuardRow(dict):
        def get(self, key, default=None):
            if key == "expected":
                raise AssertionError("resource probe read a label")
            return super().get(key, default)

    items = []
    for index in range(128):
        base = record(index, "natural_fit" if index < 64 else "challenge_fit")
        row = GuardRow({"id": base["id"], "group_id": f"g{index}",
                        "input": base["source"], "context": f"ctx{index}",
                        "candidates": base["candidates"],
                        "raw_bypass_response": base["raw_bypass_response"],
                        "expected": object()})
        items.append({"role": base["role"], "row": row})

    class FullBackend(FakeBackend):
        def load_model(self):
            return object(), object()

        def token_contract(self, tokenizer):
            return {"opener": 2, "closer": 4, "eos": 5,
                    "digits": [10, 11, 12, 13, 14], "newline": [6]}

        def tokens(self, tokenizer, messages):
            context = messages[1]["content"].split("문맥: ctx", 1)[1].splitlines()[0]
            index = int(context)
            return list(range(100 + index if index < 64 else 20))

        def decode(self, tokenizer, values):
            if values and all(value in {1, 2, 3, 4, 5} for value in values):
                return super().decode(tokenizer, values)
            return "prompt"

    backend = FullBackend()
    monkeypatch.setattr(probe, "load_inputs", lambda *args: {"rows": items})
    monkeypatch.setattr(probe, "validate_frozen_sources",
                        lambda *args, **kwargs: ({}, b"", b""))
    monkeypatch.setattr(probe, "verify_hashes", lambda *args: None)
    monkeypatch.setattr(probe, "package_sources", lambda: {})
    monkeypatch.setattr(probe, "require_versions",
                        lambda: {"mlx": "x", "mlx-lm": "x", "transformers": "x"})
    monkeypatch.setattr(probe, "require_resources", lambda *args, **kwargs: True)
    monkeypatch.setattr(probe, "system_message",
                        lambda: {"role": "system", "content": "choose"})
    args = SimpleNamespace(
        output=tmp_path / "output", plan=probe.PLAN,
        plan_sha256=probe.PLAN_SHA256, data=tmp_path / "data",
        manifest_sha256=probe.PILOT_MANIFEST_SHA256,
        lineage_report=tmp_path / "lineage", lineage_sha256=probe.LINEAGE_SHA256)
    report = probe.run(args, backend_factory=lambda: backend)
    assert [row["source_row_index"] for row in report["outcomes"]] == [63, 0, 64, 96]
    assert report["resource_feasible"] is True
    assert report["expected_or_labels_read"] is False
    assert len(backend.scored) == 4 and backend.released == 1
