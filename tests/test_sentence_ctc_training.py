from dataclasses import asdict
import json
import math

import pytest
import torch

from hanglish.models.sentence_ctc import SentenceCTC, SentenceCTCConfig
from scripts import train_sentence_ctc as trainer


def response(config=None):
    config = config or SentenceCTCConfig()
    return {
        "text": "a", "reason": "complete", "generated_tokens": 1,
        "labels": [98], "log_probability": -1.,
        "frames": 3 * (config.max_source_tokens - 2),
        "decode_mode": "ctc_prefix_beam",
        "beam": {
            "frames": 3 * (config.max_source_tokens - 2),
            "beam_size": config.beam_size,
            "token_shortlist": config.token_shortlist,
            "max_label_tokens": 511,
            "shortlist_pruned": True,
            "beam_pruned": True,
            "repeat_tokens_restored": 0,
            "approximate_pruning": True,
        },
        "greedy": {"text": "a", "reason": "complete",
                   "generated_tokens": 1, "labels": [98]},
    }


def qualification_evidence(config=None, rows=2):
    config = config or SentenceCTCConfig()
    updates = [
        {"kind": "synthetic_length_bin", "elapsed_s": .1 * (index + 1),
         "loss": 2., "gradient_norm": 3., "rows": 16,
         "source_tokens": length, "max_target_tokens": length - 1,
         "max_input_frames": 3 * (length - 2)}
        for index, length in enumerate((128, 256, 384, 512))
    ]
    return {
        "completed": True,
        "phase": "ctc_preflight",
        "config": asdict(config),
        "training_rows": rows,
        "pins": {"fixture": "hash"},
        "frame_audit": {"rows": rows, "invalid": 0,
                        "max_required_per_source_byte": 1.5,
                        "max_source_tokens": 20, "max_target_tokens": 18},
        "parameters": trainer.EXPECTED_PARAMETERS,
        "real_update": {"elapsed_s": .1, "loss": 2., "gradient_norm": 3.,
                        "rows": 16, "source_tokens": 20,
                        "max_target_tokens": 18, "max_input_frames": 54},
        "updates": updates,
        "reload_max_logit_delta": 0.,
        "generation_probe": {"batch_size": 8, "source_tokens": 512,
                             "elapsed_s": 1.,
                             "responses": [response(config) for _ in range(8)]},
        "estimated_8epoch_training_s": 10.,
        "estimated_dev_generation_s": 50.,
        "qualifies": True,
        "elapsed_s": 10.,
        "peak_rss_bytes": 1_000_000,
    }


def qualify_fixture(tmp_path, monkeypatch, mutation=None):
    rows = [dict(id=str(index), input="a a", expected=["가 가"])
            for index in range(2)]
    report = qualification_evidence(rows=len(rows))
    if mutation == "phase":
        report["phase"] = "other"
    elif mutation == "pins":
        report["pins"] = {"changed": "hash"}
    elif mutation == "frame":
        report["frame_audit"]["invalid"] = 1
    elif mutation == "bin":
        report["updates"][-1]["max_input_frames"] -= 1
    elif mutation == "nonfinite":
        report["updates"][0]["loss"] = float("nan")
    elif mutation == "generation":
        report["generation_probe"]["responses"][0]["frames"] -= 3
    elif mutation == "reload":
        report["reload_max_logit_delta"] = 1e-9
    elif mutation == "parameters":
        report["parameters"] -= 1
    elif mutation == "memory":
        report["peak_rss_bytes"] = trainer.MAX_RSS + 1
    elif mutation == "training_estimate":
        report["estimated_8epoch_training_s"] += 1
    elif mutation == "dev_estimate":
        report["estimated_dev_generation_s"] += 1
    elif mutation == "qualifies":
        report["qualifies"] = False
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps(report))
    monkeypatch.setattr(trainer, "preflight_pins", lambda extra: {"fixture": "hash"})
    monkeypatch.setattr(trainer, "frame_audit", lambda values: {
        "rows": len(values), "invalid": 0, "max_required_per_source_byte": 1.5,
        "max_source_tokens": 20, "max_target_tokens": 18})
    monkeypatch.setattr(trainer, "preflight_estimate", lambda *args, **kwargs: 10.)
    monkeypatch.setattr(trainer, "check_pins", lambda pins: None)
    return path, rows


def test_qualification_recomputes_every_authorizing_bound(tmp_path, monkeypatch):
    path, rows = qualify_fixture(tmp_path, monkeypatch)
    assert trainer.qualify(path, rows)["qualifies"] is True


@pytest.mark.parametrize("mutation", [
    "phase", "pins", "frame", "bin", "nonfinite", "generation", "reload",
    "parameters", "memory", "training_estimate", "dev_estimate", "qualifies",
])
def test_qualification_rejects_tampered_evidence(tmp_path, monkeypatch, mutation):
    path, rows = qualify_fixture(tmp_path, monkeypatch, mutation)
    with pytest.raises(ValueError):
        trainer.qualify(path, rows)


def tiny_config(dropout=.0):
    return SentenceCTCConfig(
        width=12, heads=2, encoder_layers=1, feedforward=24,
        dropout=dropout, max_source_tokens=64, beam_size=2, token_shortlist=4)


def tiny_rows(count=2):
    return [
        {"id": str(index), "input": f"jeoneun haksaengimnida a{index}",
         "expected": ["저는 학생입니다"]}
        for index in range(count)
    ]


def test_optimizer_step_logs_complete_target_denominator_and_updates_weights():
    torch.manual_seed(7)
    model = SentenceCTC(tiny_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    step = trainer.optimizer_step(model, optimizer, tiny_rows())
    assert step["rows"] == 2
    assert step["nonpadding_target_tokens"] > 0
    assert step["max_input_frames"] == 3 * (step["source_tokens"] - 2)
    assert math.isfinite(step["loss"]) and math.isfinite(step["gradient_norm"])
    assert any(not torch.equal(before[name], value)
               for name, value in model.state_dict().items())


def empty_evaluation(model, output):
    (output / "dev-predictions.jsonl").write_text("")
    return {
        "rows": 0, "whole_correct": 0, "valid_complete": 0,
        "forward_equal": 0, "greedy_whole_correct": 0,
        "greedy_valid_complete": 0, "reasons": {}, "greedy_reasons": {},
        "literal_exact": {"rows": 0, "correct": 0},
        "numeric_exact": {"rows": 0, "correct": 0},
        "elapsed_s": 0., "labels_joined_after_complete_capture": True,
        "outcomes": [],
    }


def test_training_resume_matches_uninterrupted_ctc_weights_and_logs(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    rows = tiny_rows(33)
    config = tiny_config(dropout=.2)
    qualification = tmp_path / "qualification.json"
    qualification.write_text("{}")
    monkeypatch.setattr(trainer, "qualify", lambda *args, **kwargs: {
        "updates": qualification_evidence(config)["updates"]})
    monkeypatch.setattr(trainer, "preflight_estimate", lambda *args, **kwargs: 1.)
    monkeypatch.setattr(trainer, "evaluate", empty_evaluation)
    monkeypatch.setattr(trainer, "check_pins", lambda pins: None)
    monkeypatch.setattr(trainer, "rss_bytes", lambda: 1_000_000)
    monkeypatch.setattr(trainer, "CHECKPOINT_INTERVAL", 2)
    baseline = tmp_path / "baseline"
    interrupted = tmp_path / "interrupted"
    baseline.mkdir()
    interrupted.mkdir()

    torch.manual_seed(91)
    trainer.train(baseline, rows, {}, config, qualification, epochs=2)
    original = trainer.optimizer_step
    calls = 0
    def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise InterruptedError("simulated process boundary")
        return original(*args, **kwargs)
    monkeypatch.setattr(trainer, "optimizer_step", interrupt)
    torch.manual_seed(91)
    with pytest.raises(InterruptedError):
        trainer.train(interrupted, rows, {}, config, qualification, epochs=2)
    monkeypatch.setattr(trainer, "optimizer_step", original)
    torch.manual_seed(888)
    trainer.train(interrupted, rows, {}, config, qualification, epochs=2, resume=True)

    full = torch.load(baseline / "model.pt", weights_only=True)["state_dict"]
    resumed = torch.load(interrupted / "model.pt", weights_only=True)["state_dict"]
    assert full.keys() == resumed.keys()
    assert all(torch.equal(full[name], resumed[name]) for name in full)
    logs = [[json.loads(line) for line in (path / "updates.jsonl").read_text().splitlines()]
            for path in (baseline, interrupted)]
    fields = ("step", "epoch", "ids", "loss", "gradient_norm",
              "nonpadding_target_tokens")
    assert [[{key: row[key] for key in fields} for row in values]
            for values in logs][0] == [
                {key: row[key] for key in fields} for row in logs[1]]
    reports = [json.loads((path / "report.json").read_text())
               for path in (baseline, interrupted)]
    assert reports[0]["epochs"] == reports[1]["epochs"]
    assert reports[1]["resumed"] is True


def test_dev_capture_passes_only_sources_then_joins_labels(tmp_path, monkeypatch):
    rows = [
        {"id": str(index), "input": f"roman sentence a{index}",
         "expected": [f"정답 문장 {index}"]}
        for index in range(9)
    ]
    monkeypatch.setattr(trainer, "read_split",
                        lambda corpus, split: rows if split == "dev"
                        else pytest.fail("non-DEV split was read"))
    monkeypatch.setattr(trainer, "rss_bytes", lambda: 1_000_000)
    calls = []
    class Model:
        def generate(self, sources):
            calls.append(list(sources))
            results = []
            for source in sources:
                index = int(source.rsplit("a", 1)[1])
                text = f"정답 문장 {index}"
                results.append({"text": text, "reason": "complete",
                    "greedy": {"text": text, "reason": "complete"}})
            return results
    metrics = trainer.evaluate(Model(), tmp_path)
    assert [len(call) for call in calls] == [8, 1]
    assert calls == [[row["input"] for row in rows[:8]], [rows[-1]["input"]]]
    captures = [json.loads(line) for line in
                (tmp_path / "dev-predictions.jsonl").read_text().splitlines()]
    assert all(set(row) == {"id", "input", "response"} for row in captures)
    assert metrics["whole_correct"] == metrics["greedy_whole_correct"] == 9
    assert metrics["labels_joined_after_complete_capture"] is True
    assert metrics["labels_opened_after_complete_capture"] is False
    assert metrics["gold_not_passed_to_predictor"] is True
