from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts import evaluate_numeric_composition_downstream as module


MODEL = {"model": "a" * 64}


def candidate(text):
    return {"text": text, "score": 1.0, "segments": [{"input_start": 0,
        "input_end": 1, "source": "a", "text": text, "kind": "neural"}]}


def fake_rows():
    sources, targets, baselines = [], [], []
    for index in range(module.TOTAL_ROWS):
        fresh = index < module.FRESH_ROWS
        row_id = f"r{index}"
        sources.append({"id": row_id, "group_id": f"g{index % module.FRESH_GROUPS}",
                        "origin": "fresh_natural" if fresh else "challenge",
                        "input": "a", "context": ""})
        targets.append({"id": row_id, "group_id": sources[-1]["group_id"],
                        "role": "fresh_natural" if fresh else "challenge_preservation",
                        "expected": ["가"]})
        if not fresh:
            baselines.append({"id": row_id})
    return sources, targets, baselines


def fake_capture(bundle, source, baseline, artifacts):
    index = int(source["id"][1:])
    fresh = index < module.FRESH_ROWS
    corrected = fresh and index < 12
    parent_text = "나" if corrected else "가"
    parent = {"result": parent_text, "candidates": [candidate(parent_text)]}
    child = {"result": "가", "candidates": [candidate("가")]}
    raw = {"result": "a", "candidates": [{"text": "a"}]}
    path = candidate(parent_text)
    return {**source, "parent": parent, "child": child, "parent_raw": raw,
            "child_raw": deepcopy(raw), "parent_pool": [path], "parent_base": [path],
            "trace": {"base": [path], "accepted_old_extras": [],
                      "accepted_new_extras": [], "scoring_pool": [candidate("가")]},
            "parent_sentence_scored": False, "child_sentence_scored": False,
            "child_score_inputs": None, "parent_ms": 1.0, "child_ms": 2.0}


def response(text="가", *, raw=False, score=1.0, sentence_scored=False):
    chosen = ({"text": "a", "score": 0.0, "segments": [{"input_start": 0,
        "input_end": 1, "source": "a", "text": "a", "kind": "raw"}]}
        if raw else candidate(text))
    return {"input": "a", "context": "", "engine": "neural", "mode": "batch",
            "raw": raw, "version": module.RUNTIME_VERSION, "candidates": [chosen],
            "result": chosen["text"], "ranker": None if raw else "hybrid",
            "model_artifacts": {} if raw else MODEL, "context_used": False,
            "external_context_used": False, "ranking_applied": sentence_scored,
            "sentence_ranking_applied": sentence_scored, "warnings": [],
            "engine_ms": score}


class Recorder:
    def __init__(self):
        self.candidates = None

    def reset(self):
        self.candidates = None


class Converter:
    def __init__(self, recorder, *, child=False, raw_text="a", score=1.0,
                 trace_base=None):
        self.recorder = recorder
        self.child = child
        self.raw_text = raw_text
        self.score = score
        self.audit_pieces = [("가", 1.0, candidate("가")["segments"])]
        self.audit_static_pieces = []
        self.numeric_composition_trace = None
        self.trace_base = trace_base

    def convert(self, text, context, raw=False):
        if raw:
            value = response(raw=True)
            if self.raw_text != "a":
                value["result"] = self.raw_text
                value["candidates"][0]["text"] = self.raw_text
                value["candidates"][0]["segments"][0]["text"] = self.raw_text
            return value
        value = response(score=self.score)
        if self.child:
            base = [candidate("가")] if self.trace_base is None else self.trace_base
            self.numeric_composition_trace = {"base": base, "accepted_old_extras": [],
                "accepted_new_extras": [], "scoring_pool": base,
                "eligible_piece_count": 0}
        return value


def entry():
    plan = module.ROOT / ".omx/plans/numeric-composition-downstream-v1.md"
    return {"runtime": {"accepted_weight_hashes": MODEL},
            "source_hashes": {module.path_key(plan): module.file_digest(plan)}}


def test_artifact_schema_and_callable_capture_api_are_fixed():
    assert module.SCHEMA == "numeric-composition-downstream-v1"
    assert callable(module.validate_static_entry)
    assert callable(module.load_and_split_targets)
    assert callable(module.capture_source)
    assert callable(module.score_closed_captures)
    assert callable(module.execute_pipeline)


def test_pipeline_closes_captures_releases_models_then_reopens_targets(tmp_path, monkeypatch):
    events = []
    sources, targets, baselines = fake_rows()
    def target_loader(_entry, access):
        access.update(fresh_target_bytes_accessed=True, preservation_target_bytes_accessed=True)
        return sources, targets, baselines, entry()["source_hashes"]
    def model_loader(_artifacts):
        events.append("load"); return {"live": True}
    def capture(bundle, source, baseline, artifacts):
        assert bundle["live"]
        events.append("capture")
        return fake_capture(bundle, source, baseline, artifacts)
    def release(bundle):
        events.append("release"); bundle.clear()
    original_read = module.read_jsonl
    def reopened(path):
        events.append("target_reopen")
        assert events[-2] == "release"
        return original_read(path)
    monkeypatch.setattr(module, "read_jsonl", reopened)
    report = module.execute_pipeline(tmp_path / "out", entry(), target_loader, model_loader,
                                     capture, release, lambda *_: {"ok": True})
    assert report["completed"] is True
    assert report["captures_closed"] is True
    assert report["models_released_before_target_join"] is True
    assert report["targets_joined"] is True
    assert events.count("capture") == module.TOTAL_ROWS
    assert events[-2:] == ["release", "target_reopen"]
    assert len(report["resource_observations"]) == module.TOTAL_ROWS + 1
    assert set(report["files"]) == {"sources.jsonl", "targets.jsonl",
                                    "preservation_baselines.jsonl", "captures.jsonl"}


def test_only_source_records_are_passed_to_capture(tmp_path):
    sources, targets, baselines = fake_rows()
    seen = []
    def capture(bundle, source, baseline, artifacts):
        assert set(source) == {"id", "group_id", "origin", "input", "context"}
        assert "expected" not in source
        seen.append(source["id"])
        return fake_capture(bundle, source, baseline, artifacts)
    report = module.execute_pipeline(tmp_path / "out", entry(),
        lambda _e, access: (sources, targets, baselines, entry()["source_hashes"]),
        lambda _: {}, capture,
        lambda bundle: None, lambda *_: {})
    assert report["completed"] and len(seen) == module.TOTAL_ROWS


def test_metric_gate_uses_net_bootstrap_and_exact_preservation(monkeypatch):
    sources, targets, baselines = fake_rows()
    captures = [fake_capture({}, source, None, MODEL) for source in sources]
    result = module.score_closed_captures(sources, targets, captures)
    assert result["bootstrap"]["net_gain"] == 12
    assert result["gate"]["passed"] is True
    assert result["gate"]["promotion_allowed"] is False
    assert result["roles"]["fresh_natural"]["regressions"] == []
    assert result["roles"]["challenge_preservation"]["top1"] == module.PRESERVATION_ROWS


def test_fresh_regressions_are_reported_without_universal_zero_regression_gate():
    sources, targets, _ = fake_rows()
    captures = [fake_capture({}, source, None, MODEL) for source in sources]
    captures[20]["parent"] = {"result": "가", "candidates": [candidate("가")]}
    captures[20]["child"] = {"result": "나", "candidates": [candidate("나"), candidate("가")]}
    result = module.score_closed_captures(sources, targets, captures)
    assert result["roles"]["fresh_natural"]["regressions"] == ["r20"]
    assert result["bootstrap"]["net_gain"] == 11


def test_top5_loss_or_missing_preservation_row_rejects():
    sources, targets, _ = fake_rows()
    captures = [fake_capture({}, source, None, MODEL) for source in sources]
    captures[20]["parent"] = {"result": "가", "candidates": [candidate("가")]}
    captures[20]["child"] = {"result": "나", "candidates": [candidate("나")]}
    assert module.score_closed_captures(sources, targets, captures)["gate"]["passed"] is False
    with pytest.raises(ValueError, match="incomplete"):
        module.score_closed_captures(sources[:-1], targets[:-1], captures[:-1])


def test_metric_replay_rejects_raw_pool_trace_and_nonfinite_mutations():
    sources, targets, _ = fake_rows()
    captures = [fake_capture({}, source, None, MODEL) for source in sources]
    captures[0]["child_raw"]["result"] = "changed"
    with pytest.raises(ValueError, match="raw response"):
        module.score_closed_captures(sources, targets, captures)
    captures = [fake_capture({}, source, None, MODEL) for source in sources]
    captures[0]["parent_base"] = []
    with pytest.raises(ValueError, match="base pool"):
        module.score_closed_captures(sources, targets, captures)
    captures = [fake_capture({}, source, None, MODEL) for source in sources]
    captures[0]["parent_sentence_scored"] = True
    captures[0]["parent_pool"] = []
    with pytest.raises(ValueError, match="full pool"):
        module.score_closed_captures(sources, targets, captures)
    captures = [fake_capture({}, source, None, MODEL) for source in sources]
    captures[0]["child_sentence_scored"] = True
    captures[0]["child_score_inputs"] = ["wrong"]
    with pytest.raises(ValueError, match="scorer inputs"):
        module.score_closed_captures(sources, targets, captures)
    captures = [fake_capture({}, source, None, MODEL) for source in sources]
    captures[0]["parent_ms"] = float("inf")
    with pytest.raises(ValueError, match="Out of range float"):
        module.score_closed_captures(sources, targets, captures)


def test_capture_rejects_changed_raw_base_and_nonfinite_response(monkeypatch):
    from scripts import audit_final_choice_candidate_stages as prior
    monkeypatch.setattr(prior, "beam_paths", lambda pieces: deepcopy(pieces))
    source = {"id": "one", "group_id": "group", "origin": "fresh_natural",
              "input": "a", "context": ""}
    recorder = Recorder()
    with pytest.raises(ValueError, match="raw"):
        module.capture_source({"parent": Converter(recorder),
            "child": Converter(recorder, child=True, raw_text="b"), "recorder": recorder},
            source, None, MODEL)
    with pytest.raises(ValueError, match="original base pool"):
        module.capture_source({"parent": Converter(recorder),
            "child": Converter(recorder, child=True, trace_base=[]), "recorder": recorder},
            source, None, MODEL)
    with pytest.raises(ValueError, match="metadata"):
        module.capture_source({"parent": Converter(recorder, score=float("nan")),
            "child": Converter(recorder, child=True), "recorder": recorder},
            source, None, MODEL)
    with pytest.raises(ValueError, match="artifacts"):
        module.capture_source({"parent": Converter(recorder),
            "child": Converter(recorder, child=True), "recorder": recorder},
            source, None, {"model": "b" * 64})


def test_source_hash_and_handoff_tampering_are_rejected(tmp_path, monkeypatch):
    changed = tmp_path / "changed.txt"
    changed.write_text("changed")
    with pytest.raises(ValueError, match="source changed"):
        module.verify_hashes({str(changed): "0" * 64})
    original = module.file_digest
    monkeypatch.setattr(module, "file_digest",
        lambda path: "0" * 64 if Path(path) == module.HANDOFF else original(path))
    with pytest.raises(ValueError, match="handoff changed"):
        module.validate_static_entry()


def test_target_split_replays_frozen_collision_selection_and_source_context(
        tmp_path, monkeypatch):
    from scripts import evaluate_final_choice_holdout as holdout
    from scripts import prepare_long_context_lora_data as preparation
    selection = [{"id": f"f{i}", "group_id": f"g{i % module.FRESH_GROUPS}"}
                 for i in range(module.FRESH_ROWS)]
    raw_rows = [{**row, "split": "train", "text": "가", "left_context": f"문맥{i}"}
                for i, row in enumerate(selection)]
    fresh_path = tmp_path / "fresh.jsonl"
    fresh_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                  for row in raw_rows))
    preservation = tmp_path / "preservation"
    preservation.mkdir()
    for name in ("manifest.json", "cases.jsonl", "selection.jsonl"):
        (preservation / name).write_text("{}\n")
    preservation_rows = [{"id": f"p{i}", "group_id": f"pg{i}",
        "origin": "challenge", "input": "a", "context": "", "expected": ["가"],
        "result": "가", "candidates": [candidate("가")], "response_flags": {},
        "runtime_version": module.RUNTIME_VERSION, "raw_bypass_response": {}}
        for i in range(module.PRESERVATION_ROWS)]
    groups = list(dict.fromkeys(row["group_id"] for row in selection))
    selected_sources = [{"id": row["id"], "group_id": row["group_id"],
                         "input": "ga", "context": raw_rows[i]["left_context"]}
                        for i, row in enumerate(selection)]
    monkeypatch.setattr(module, "FRESH_RAW", fresh_path)
    monkeypatch.setattr(module, "FRESH_RAW_SHA256", module.file_digest(fresh_path))
    monkeypatch.setattr(module, "PRESERVATION", preservation)
    original_path_key = module.path_key
    monkeypatch.setattr(module, "path_key", lambda path: (
        original_path_key(path) if module.ROOT in Path(path).resolve().parents
        else str(Path(path).resolve())))
    monkeypatch.setattr(preparation, "read_pinned_json", lambda *_: (b"{}", {}))
    monkeypatch.setattr(preparation, "collision_inventory",
        lambda: (set(), set(), set(), {module.path_key(
            module.ROOT / ".omx/plans/numeric-composition-downstream-v1.md"):
            module.file_digest(module.ROOT / ".omx/plans/numeric-composition-downstream-v1.md")}))
    monkeypatch.setattr(preparation, "select_fresh",
        lambda *_args: (groups, selected_sources, {}, module.FRESH_GROUPS,
                        module.FRESH_ROWS))
    monkeypatch.setattr(holdout, "load_evaluation_capture", lambda *_: {
        "rows": preservation_rows, "manifest": {"runtime_version": module.RUNTIME_VERSION,
            "model_artifact_hashes": MODEL, "source_hashes": {}}})
    access = {}
    loaded_sources, targets, baselines, hashes = module.load_and_split_targets({
        "selection": selection, "runtime": {"accepted_weight_hashes": MODEL},
        "manifest": {"fresh": {"group_ids": groups, "excluded_groups": {},
            "admissible_groups": module.FRESH_GROUPS,
            "admissible_rows": module.FRESH_ROWS}}}, access)
    assert access == {"fresh_target_bytes_accessed": True,
                      "preservation_target_bytes_accessed": True}
    assert set(loaded_sources[0]) == {"id", "group_id", "origin", "input", "context"}
    assert loaded_sources[0]["input"] == "ga" and loaded_sources[0]["context"] == "문맥0"
    assert targets[0]["expected"] == ["가"]
    assert len(baselines) == module.PRESERVATION_ROWS and hashes


def test_bootstrap_net_gain_boundary_fails_below_eight():
    sources, targets, _ = fake_rows()
    captures = [fake_capture({}, source, None, MODEL) for source in sources]
    for index in range(7, 12):
        captures[index]["parent"] = {"result": "가", "candidates": [candidate("가")]}
    result = module.score_closed_captures(sources, targets, captures)
    assert result["bootstrap"]["net_gain"] == 7
    assert result["gate"]["passed"] is False


def test_wrong_fresh_group_count_fails_the_fixed_gate():
    sources, targets, _ = fake_rows()
    captures = [fake_capture({}, source, None, MODEL) for source in sources]
    for rows in (sources, targets, captures):
        rows[386]["group_id"] = "g0"
    result = module.score_closed_captures(sources, targets, captures)
    assert result["bootstrap"]["groups"] == module.FRESH_GROUPS - 1
    assert result["gate"]["passed"] is False


def test_failure_report_never_claims_join_or_promotion(tmp_path):
    def fail_loader(_entry, access):
        raise RuntimeError("target failure")
    report = module.execute_pipeline(tmp_path / "out", entry(), fail_loader,
                                     lambda _: {}, fake_capture)
    assert report["completed"] is False
    assert report["terminal_status"] == "failed"
    assert report["targets_joined"] is False
    assert report["promotion_allowed"] is False
    saved = json.loads((tmp_path / "out/report.json").read_bytes())
    assert saved == report


def test_capture_failure_releases_loaded_models(tmp_path):
    sources, targets, baselines = fake_rows()
    released = []
    def fail_capture(*_args):
        raise RuntimeError("capture failure")
    def release(bundle):
        released.append(bundle)
    report = module.execute_pipeline(tmp_path / "out", entry(),
        lambda _e, access: (sources, targets, baselines, entry()["source_hashes"]),
        lambda _: {"live": True}, fail_capture, release, lambda *_: {})
    assert report["completed"] is False
    assert report["models_released_on_failure"] is True
    assert report["captured_rows"] == 0
    assert len(report["partial_captures_sha256"]) == 64
    assert released == [{"live": True}]
