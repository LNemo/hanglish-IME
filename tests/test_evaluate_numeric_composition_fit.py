import copy
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

import scripts.evaluate_numeric_composition_fit as module


def path(text, score=0.0):
    return {"text": text, "score": score, "segments": [{
        "input_start": 0,
        "input_end": len(text),
        "source": text,
        "text": text,
        "kind": "raw",
    }]}


def row_and_capture(index, *, expected="base", child_candidates=None,
                    parent_result="base", child_result="base",
                    parent_scored=True, child_scored=True):
    identity = {"id": f"row-{index:03d}", "input": "source", "context": "context"}
    parent_candidates = [path("base")]
    child_candidates = child_candidates or [path("base")]
    parent = {"result": parent_result, "candidates": parent_candidates}
    child = {"result": child_result, "candidates": child_candidates}
    base = [path("base")]
    accepted_new = [candidate for candidate in child_candidates
                    if candidate["text"] != "base"]
    scoring_pool = [*base, *accepted_new]
    capture = {
        **identity,
        "parent": copy.deepcopy(parent),
        "child": copy.deepcopy(child),
        "parent_raw": {"result": "source", "raw": True},
        "child_raw": {"result": "source", "raw": True},
        "parent_pool": copy.deepcopy(base),
        "parent_base": copy.deepcopy(base),
        "trace": {
            "base": copy.deepcopy(base),
            "accepted_old_extras": [],
            "accepted_new_extras": copy.deepcopy(accepted_new),
            "scoring_pool": copy.deepcopy(scoring_pool),
            "eligible_piece_count": int(bool(accepted_new)),
        },
        "parent_sentence_scored": parent_scored,
        "child_sentence_scored": child_scored,
        "child_score_inputs": ([item["text"] for item in scoring_pool]
                               if child_scored else None),
    }
    row = {**identity, "expected": [expected], **copy.deepcopy(parent)}
    return row, capture


def passing_cohort():
    pairs = [row_and_capture(index) for index in range(511)]
    pairs.append(row_and_capture(
        511,
        expected="new",
        child_candidates=[path("base"), path("new", 1.0)],
    ))
    return [pair[0] for pair in pairs], [pair[1] for pair in pairs]


def test_summary_gate_requires_512_rows_and_real_top5_gain():
    rows, captures = passing_cohort()
    result = module.summarize(rows, captures)
    assert result["gate"] == {
        "passed": True,
        "original_pool_preserved": True,
        "raw_preserved": True,
        "zero_first_choice_losses": True,
        "zero_top5_losses": True,
        "at_least_one_new_top5": True,
    }
    assert result["counts"]["rows"] == 512
    assert result["counts"]["before_top5"] == 511
    assert result["counts"]["top5"] == 512
    assert result["top5_gain_ids"] == ["row-511"]


def test_first_choice_and_top5_regressions_fail_gate_with_exact_ids():
    rows, captures = passing_cohort()
    rows[8]["expected"] = ["base"]
    captures[8]["child"]["result"] = "wrong"
    captures[8]["child"]["candidates"] = [path("wrong")]
    result = module.summarize(rows, captures)
    assert result["gate"]["passed"] is False
    assert result["regression_ids"] == ["row-008"]
    assert result["top5_loss_ids"] == ["row-008"]


@pytest.mark.parametrize("mutation, message", [
    (lambda capture: capture.update(child_raw={"raw": True, "result": "changed"}),
     "raw response changed"),
    (lambda capture: capture["trace"]["base"][0].update(text="changed"),
     "original base pool objects changed"),
    (lambda capture: capture.update(child_score_inputs=["different"]),
     "trace differs from actual scorer input"),
    (lambda capture: (capture.update(child_sentence_scored=False),
                      capture.update(child_score_inputs=["unexpected"])),
     "unexpected sentence scorer call"),
])
def test_preservation_claims_are_proven_by_rejecting_tampering(mutation, message):
    rows, captures = passing_cohort()
    mutation(captures[0])
    with pytest.raises(ValueError, match=message):
        module.summarize(rows, captures)


def test_unscored_singleton_still_rejects_changed_preserved_base():
    rows, captures = passing_cohort()
    captures[0]["parent_sentence_scored"] = False
    captures[0]["child_sentence_scored"] = False
    captures[0]["child_score_inputs"] = None
    captures[0]["trace"]["base"][0]["score"] = 99.0
    with pytest.raises(ValueError, match="original base pool objects changed"):
        module.summarize(rows, captures)


def test_summary_rejects_partial_or_reordered_capture_before_scoring():
    rows, captures = passing_cohort()
    with pytest.raises(ValueError, match="all FIT rows"):
        module.summarize(rows, captures[:-1])
    captures[1]["id"] = captures[0]["id"]
    with pytest.raises(ValueError, match="identities differ or repeat"):
        module.summarize(rows, captures)


def test_latency_report_is_explicitly_not_browser_evidence():
    source = open(module.__file__, encoding="utf-8").read()
    assert "not browser or cold model loading" in source
    assert "browser" not in module.summarize.__code__.co_varnames
    assert "latency" not in module.summarize.__code__.co_varnames


class LabelGuard(dict):
    capture_complete = False

    def __getitem__(self, key):
        if key == "expected" and not type(self).capture_complete:
            raise AssertionError("label read before all source captures")
        return super().__getitem__(key)


def test_run_captures_every_source_before_first_expected_read(tmp_path, monkeypatch):
    rows = []
    for index in range(512):
        row, _ = row_and_capture(index)
        row["candidates"][0]["segments"][0].update(
            input_end=len(row["input"]), source=row["input"],
            text="base", kind="neural",
        )
        row.update(response_flags={})
        rows.append(LabelGuard(row))
    LabelGuard.capture_complete = False

    audit = tmp_path / "audit"
    audit.mkdir()
    (audit / "report.json").write_text('{"source_hashes": {}}')
    (audit / "cases.jsonl").write_text("".join(
        json.dumps({"id": row["id"]}) + "\n" for row in rows
    ))
    monkeypatch.setattr(module, "AUDIT", audit)
    original_sha = module.sha
    monkeypatch.setattr(
        module,
        "sha",
        lambda value: (module.AUDIT_REPORT_SHA if value == audit / "report.json"
                       else module.AUDIT_CASES_SHA if value == audit / "cases.jsonl"
                       else original_sha(value)),
    )

    class Recorder:
        def __init__(self, delegate):
            self.delegate = delegate
            self.candidates = None

        def reset(self):
            self.candidates = None

        def can_score(self, text):
            return True

    class Parent:
        version = "runtime"
        model_artifacts = {}

        def __init__(self, **kwargs):
            self.version = "runtime"
            self.model_artifacts = {}
            self.sentence_ranker = SimpleNamespace(can_score=lambda text: True)

        def load_models(self):
            return None

        def convert(self, text, context, raw=False):
            assert not LabelGuard.capture_complete
            segment = {
                "input_start": 0, "input_end": len(text), "source": text,
                "text": "base", "kind": "neural",
            }
            self.audit_pieces = [("source", 0, ())]
            self.audit_static_pieces = [("source", 0, ())]
            self.audit_path = ("base", 0.0, [segment])
            candidate = path(text if raw else "base", 0.0)
            candidate["segments"][0].update(
                input_end=len(text), source=text,
                text=text if raw else "base",
                kind="raw" if raw else "neural",
            )
            return {
                "input": text, "context": context, "engine": "neural",
                "mode": "batch", "raw": raw, "version": self.version,
                "candidates": [candidate], "result": candidate["text"],
                "ranker": None if raw else "hybrid",
                "model_artifacts": {} if raw else self.model_artifacts,
                "sentence_ranking_applied": False, "engine_ms": 1.0,
            }

    class Child:
        def convert(self, text, context, raw=False):
            response = Parent.convert(self, text, context, raw)
            if not raw:
                base = [{"text": "base", "score": 0.0, "segments": [{
                    "input_start": 0, "input_end": len(text),
                    "source": text, "text": "base", "kind": "neural",
                }]}]
                self.numeric_composition_trace = {
                    "base": base, "accepted_old_extras": [],
                    "accepted_new_extras": [], "scoring_pool": base,
                    "eligible_piece_count": 0,
                }
            return response

    prior = ModuleType("scripts.audit_final_choice_candidate_stages")
    prior.FIT = tmp_path / "fit"
    prior.FIT_MANIFEST_SHA256 = "manifest"
    prior.RUNTIME_VERSION = "runtime"
    prior.StageConverter = Parent
    prior.SentenceRecorder = Recorder
    prior.verify_hashes = lambda hashes: None
    prior.require_resources = lambda started: None
    prior.beam_paths = lambda pieces: [("base", 0.0, [{
        "input_start": 0, "input_end": len("source"),
        "source": "source", "text": "base", "kind": "neural",
    }])]
    prior.reconstruct_pool = lambda *args, **kwargs: ([], [], [], 0, True)
    capture_module = ModuleType("scripts.final_choice_capture")
    capture_module._response_flags = lambda response: {}

    def validate(response, source, context, version, artifacts, *, raw):
        assert not LabelGuard.capture_complete
        assert raw is response["raw"]
    capture_module.validate_response = validate
    runtime_module = ModuleType("scripts.numeric_composition_runtime")
    runtime_module.NumericCompositionRuntime = Child
    train_module = ModuleType("scripts.train_final_choice_selector")
    train_module.load_fit_role = lambda *args: {
        "rows": rows,
        "manifest": {"model_artifact_hashes": {}},
    }
    torch_module = ModuleType("torch")
    torch_module.set_num_threads = lambda count: None
    for name, replacement in {
        "torch": torch_module,
        "scripts.audit_final_choice_candidate_stages": prior,
        "scripts.final_choice_capture": capture_module,
        "scripts.numeric_composition_runtime": runtime_module,
        "scripts.train_final_choice_selector": train_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, replacement)
        if name.startswith("scripts."):
            import scripts
            monkeypatch.setattr(scripts, name.rsplit(".", 1)[1], replacement, raising=False)

    original_summarize = module.summarize

    def summarize_after_capture(actual_rows, captures):
        assert len(captures) == 512
        LabelGuard.capture_complete = True
        return original_summarize(actual_rows, captures)

    monkeypatch.setattr(module, "summarize", summarize_after_capture)
    output = tmp_path / "output"
    report = module.run(output)
    assert report["completed"] is True
    assert report["rows_captured_before_accuracy"] == 512
    assert len((output / "captures.jsonl").read_text().splitlines()) == 512
    assert report["latency_scope"] == (
        "engine calls after model load; not browser or cold model loading"
    )
