import json

from scripts import probe_electra_plausibility as probe


def test_feature_metrics_keeps_unscorable_and_missing_gold_rows():
    rows = [
        {"id": "a", "source_role": "natural_fit", "candidate_texts": ["같이", "가치"],
         "current_result": "가치", "expected": ["같이"],
         "plausibility": [{"mean_original_logprob": -.1}, {"mean_original_logprob": -1.}]},
        {"id": "b", "source_role": "natural_fit", "candidate_texts": ["밖", "박"],
         "current_result": "밖", "expected": ["밖"], "plausibility": None},
        {"id": "c", "source_role": "challenge_fit", "candidate_texts": ["없는 후보"],
         "current_result": "없는 후보", "expected": ["정답"],
         "plausibility": [{"mean_original_logprob": -.01}]},
    ]
    result = probe.feature_metrics(rows, "mean_original_logprob")
    assert result["natural_fit"]["rows"] == 2
    assert result["natural_fit"]["baseline_correct"] == 1
    assert result["natural_fit"]["correct"] == 2
    assert result["natural_fit"]["correction_ids"] == ["a"]
    assert result["challenge_fit"]["rows"] == 1
    assert result["challenge_fit"]["correct"] == 0


def test_target_join_rejects_source_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    hashes = {}
    captures = []
    for role in ("natural_fit", "challenge_fit"):
        relative = f"data/natural_context/final-choice-v1/{role}/cases.jsonl"
        path = tmp_path / relative
        path.parent.mkdir(parents=True)
        row = {"id": role, "input": "gachi", "context": "친구", "group_id": role,
               "expected": ["같이"]}
        path.write_text(json.dumps(row) + "\n")
        hashes[relative] = probe.digest(path)
        captures.append({key: value for key, value in row.items() if key != "expected"}
                        | {"source_role": role})
    result = probe.join_targets(captures, {"source_hashes": hashes})
    assert all(row["expected"] == ["같이"] for row in result)
    assert all("expected" not in row for row in captures)
    captures[0]["context"] = "금액"
    import pytest
    with pytest.raises(ValueError, match="identity"):
        probe.join_targets(captures, {"source_hashes": hashes})


def test_capture_closes_before_targets_and_accepts_scorer_diagnostics(tmp_path, monkeypatch):
    import torch
    from pathlib import Path
    from hanglish.models import electra_plausibility

    sources = [{"id": f"r{fold}", "component": f"c{fold}", "fold": fold,
                "group_id": f"g{fold}", "source_role": "natural_fit",
                "input": "gachi", "context": "친구와", "current_result": "가치",
                "finite_candidates": [{"object": {"text": text, "score": score}}
                                      for text, score in [("가치", 1.), ("같이", 0.)]]}
               for fold in range(4)]

    class FakeScorer:
        source_hashes = {}

        @classmethod
        def load(cls):
            return cls()

        def score_candidates(self, context, candidates):
            assert context == "친구와" and candidates == ["가치", "같이"]
            return [{"mean_original_logprob": score, "changed_original_logprob": score,
                     "candidate_tokens": 1, "changed_tokens": 1, "context_tokens": 2,
                     "changed_span_fallback": False} for score in (-2., -.1)]

    output = tmp_path / "probe"
    opened = []
    original_open = Path.open

    def tracked_open(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        if path == output / "capture.jsonl" and args and args[0] == "x":
            opened.append(stream)
        return stream

    def join_after_close(captures, manifest):
        assert opened and opened[0].closed
        assert len(captures) == 4
        assert all("expected" not in row for row in captures)
        assert "expected" not in (output / "capture.jsonl").read_text()
        return [{**row, "expected": ["같이"]} for row in captures]

    monkeypatch.setattr(Path, "open", tracked_open)
    monkeypatch.setattr(electra_plausibility, "ElectraPlausibility", FakeScorer)
    monkeypatch.setattr(probe, "load_sources", lambda: ({"source_hashes": {}}, sources))
    monkeypatch.setattr(probe, "join_targets", join_after_close)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _: None)
    report = probe.run(output)
    assert report["completed"] and report["rows"] == 4
    assert report["scored_rows"] == 4
    assert len(report["calibration"]["oof_outcomes"]) == 4
