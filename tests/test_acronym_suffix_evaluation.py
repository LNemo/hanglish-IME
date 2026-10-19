import copy

import pytest

import scripts.evaluate_acronym_suffix_candidates as module


def clean_report():
    return {
        "scope": "Actual child conversion on fixed natural TRAIN first512 only",
        "policy": module.TRAIN_POLICY, "parent_version": module.RUNTIME_VERSION,
        "preparation_manifest_sha256": module.PREPARATION_MANIFEST,
        "source_hashes": {"fixed": "hash"},
        "natural_dev_inference_run": False, "development_inference_run": False,
        "test_data_read": False, "deployed": False,
        "selection_sha256": "selection", "cases_sha256": "cases",
        "option_audit_sha256": "audit",
        "metrics": {"rows": 512, "before_top1": 10, "new_top1": 11,
                    "regressions": [], "before_top5": 12, "new_top5": 13,
                    "top5_regressions": [], "before_pool_covered": 12,
                    "new_pool_covered": 13, "pool_coverage_losses": []},
        "candidate_audit": {"baseline_prefix_preserved": True,
                            "scores_unmodified_from_suffix_options": True,
                            "all_additions_forward_valid": True, "max_added": 4},
    }


def install_clean_gate_mocks(monkeypatch, report):
    monkeypatch.setattr(module, "train_source_hashes", lambda: {"fixed": "hash"})
    monkeypatch.setattr(module, "file_sha256", lambda path: {
        "selection.json": "selection", "cases.jsonl": "cases",
        "option-audit.jsonl": "audit"}.get(path.name, module.TRAIN_PROBE_SHA256))
    monkeypatch.setattr(module, "train_metrics", lambda cases: report["metrics"])
    monkeypatch.setattr(module, "validate_audits", lambda rows: None)


def test_complete_train_gate_is_accepted(monkeypatch):
    report = clean_report()
    install_clean_gate_mocks(monkeypatch, report)
    source = {"id": "id", "input": "in", "context": "ctx", "expected": ["ok"]}

    module.validate_train_gate(
        report, module.TRAIN_PROBE_SHA256, module.TRAIN_PROBE_SHA256,
        [source] * 512, [source] * 512, [], [source] * 512,
    )


@pytest.mark.parametrize("mutation", [
    "explicit", "scope", "policy", "runtime", "manifest", "source", "incomplete",
    "gain", "regression", "top5", "pool", "audit",
])
def test_train_gate_rejects_stale_incomplete_or_tampered_inputs(monkeypatch, mutation):
    report = clean_report()
    explicit = module.TRAIN_PROBE_SHA256
    if mutation == "explicit": explicit = "changed"
    elif mutation == "scope": report["scope"] = "changed"
    elif mutation == "policy": report["policy"] = {"changed": True}
    elif mutation == "runtime": report["parent_version"] = "changed"
    elif mutation == "manifest": report["preparation_manifest_sha256"] = "changed"
    elif mutation == "source": report["source_hashes"] = {"changed": "hash"}
    elif mutation == "incomplete": report["development_inference_run"] = True
    elif mutation == "gain": report["metrics"]["new_top1"] = 10
    elif mutation == "regression": report["metrics"]["regressions"] = ["case"]
    elif mutation == "top5": report["metrics"]["top5_regressions"] = ["case"]
    elif mutation == "pool": report["metrics"]["pool_coverage_losses"] = ["case"]
    else: report["candidate_audit"]["baseline_prefix_preserved"] = False
    install_clean_gate_mocks(monkeypatch, report)
    source = {"id": "id", "input": "in", "context": "ctx", "expected": ["ok"]}
    with pytest.raises(ValueError):
        module.validate_train_gate(report, explicit, module.TRAIN_PROBE_SHA256,
                                   [source] * 512, [source] * 512, [], [source] * 512)


def test_missing_corpus_identity_hash_is_rejected():
    with pytest.raises(ValueError, match="complete development"):
        module.validate_suite_hashes({"data/challenge/dev.jsonl": "a" * 64})


class TraceCase(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs); self.target_reads = 0

    def __getitem__(self, key):
        if key == "expected": self.target_reads += 1
        return super().__getitem__(key)


class Recorder:
    candidates = None
    scores = None


class Converter:
    def __init__(self, case, parent, child, recorder, fail=False):
        self.case, self.parent, self.child = case, parent, child
        self.word_options, self.recorder, self.fail = parent, recorder, fail

    def convert(self, text, context):
        assert self.case.target_reads == 0
        if self.word_options is self.child and self.fail:
            raise RuntimeError("child failed")
        value = "parent" if self.word_options is self.parent else "child"
        self.recorder.candidates = [value, text]
        self.recorder.scores = [2., 1.]
        return {"candidates": [{"text": value, "score": 2.},
                                {"text": text, "score": 1.}],
                "sentence_ranking_applied": True, "engine_ms": 1.}


def test_actual_pair_reads_labels_after_inference_and_restores_parent():
    case = TraceCase(id="id", input="raw", context="ctx", expected=["child"])
    parent, child, recorder = object(), object(), Recorder()
    converter = Converter(case, parent, child, recorder)

    result = module.evaluate_case(converter, parent, child, recorder, case)

    assert case.target_reads > 0 and converter.word_options is parent
    assert result["before_candidates"][0]["text"] == "parent"
    assert result["candidates"][0]["text"] == "child"
    assert result["before_sentence_scores"] == result["sentence_scores"] == [2., 1.]
    assert result["before_top1"] is False and result["top1"] is True


def test_parent_options_restore_even_when_child_conversion_fails():
    case = TraceCase(id="id", input="raw", context="ctx", expected=["child"])
    parent, child, recorder = object(), object(), Recorder()
    converter = Converter(case, parent, child, recorder, fail=True)

    with pytest.raises(RuntimeError, match="child failed"):
        module.evaluate_case(converter, parent, child, recorder, case)
    assert converter.word_options is parent and case.target_reads == 0


def test_reused_row_gate_rejects_first_top5_and_pool_losses():
    clean = {name: {"rows": count, "before_top1": 1, "top1": 2,
                    "regressions": [], "top5_regressions": [],
                    "sentence_pool_coverage_losses": []}
             for name, count in module.DEV_COUNTS.items()}
    assert module.development_passed(clean)
    for key in ("regressions", "top5_regressions", "sentence_pool_coverage_losses"):
        changed = copy.deepcopy(clean); changed["selection_dev"][key] = ["case"]
        assert not module.development_passed(changed)


def test_post_validation_resource_overflow_cannot_reach_eligibility(monkeypatch):
    eligibility_calls = []
    monkeypatch.setattr(module, "rss_bytes",
                        lambda: module.PROFILE["rss_limit_bytes"] + 1)
    monkeypatch.setattr(module, "development_passed",
                        lambda results: eligibility_calls.append(results) or True)

    with pytest.raises(RuntimeError, match="post-validation"):
        module.finalize_development_gate(
            {"natural_dev": {"top1": 1}}, {"source": "same"},
            {"source": "same"}, 0.,
        )

    assert eligibility_calls == []
