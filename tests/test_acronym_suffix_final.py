import copy
import hashlib
import json

import pytest

import scripts.evaluate_acronym_suffix_final as module


def clean_suites():
    return {name: {"rows": count, "before_top1": count - 1, "top1": count,
                   "before_top5": count, "top5": count,
                   "before_sentence_scored_rows": count,
                   "sentence_scored_rows": count,
                   "before_sentence_pool_covered": count,
                   "sentence_pool_covered": count,
                   "corrections": [name], "regressions": [], "top5_gains": [],
                   "top5_regressions": [], "sentence_pool_coverage_gains": [],
                   "sentence_pool_coverage_losses": []}
            for name, count in module.DEVELOPMENT_PROFILE["suite_counts"].items()}


def clean_development():
    suites = clean_suites()
    outcomes = []
    for name, count in module.DEVELOPMENT_PROFILE["suite_counts"].items():
        for index in range(count):
            outcomes.append({"id": f"{name}-{index}", "suite": name,
                             "before_top1": index != 0, "top1": True,
                             "before_top5": True, "top5": True,
                             "before_sentence_pool_covered": True,
                             "sentence_pool_covered": True})
    # Derive exact summaries rather than relying on hand-written fixture metrics.
    suites = {name: module.summarize([row for row in outcomes if row["suite"] == name])
              for name in module.DEVELOPMENT_PROFILE["suite_counts"]}
    return {"passed": True, "eligible_for_further_evaluation": True,
            "scope": "fresh actual hybrid development only",
            "parent_version": module.RUNTIME_VERSION,
            "training_probe_sha256": module.TRAIN_PROBE_SHA256,
            "preparation_manifest_sha256": module.PREPARATION_MANIFEST,
            "experimental_policy_sha256": module.policy_fingerprint(),
            "source_state": {"fixed": True}, "cases_sha256": "cases",
            "child_option_audit": {"sha256": "audit"},
            "test_data_read": False, "deployed": False,
            "outcomes": outcomes, "suites": suites}


@pytest.mark.parametrize("mutation", [
    "sha", "passed", "eligible", "scope", "train", "prep", "policy", "state",
    "cases", "audit", "count", "metrics", "regression",
])
def test_development_gate_rejects_stale_tampered_or_incomplete_report(mutation):
    report = clean_development()
    explicit = actual = module.DEVELOPMENT_REPORT_SHA256
    if mutation == "sha": explicit = "stale"
    elif mutation == "passed": report["passed"] = False
    elif mutation == "eligible": report["eligible_for_further_evaluation"] = False
    elif mutation == "scope": report["scope"] = "diagnostic"
    elif mutation == "train": report["training_probe_sha256"] = "changed"
    elif mutation == "prep": report["preparation_manifest_sha256"] = "changed"
    elif mutation == "policy": report["experimental_policy_sha256"] = "changed"
    elif mutation == "state": report["source_state"] = {"changed": True}
    elif mutation == "cases": report["cases_sha256"] = "changed"
    elif mutation == "audit": report["child_option_audit"]["sha256"] = "changed"
    elif mutation == "count": report["outcomes"].pop()
    elif mutation == "metrics": report["suites"]["natural_dev"]["top1"] -= 1
    else:
        row = next(row for row in report["outcomes"]
                   if row["suite"] == "selection_dev" and row["before_top1"])
        row["top1"] = False
    with pytest.raises(ValueError):
        module.validate_development_report(
            report, explicit, actual, {"fixed": True}, "cases", "audit"
        )


def test_failed_development_authorization_prevents_final_file_read():
    class Bomb:
        def read_bytes(self):
            raise AssertionError("final content was opened")

    with pytest.raises(PermissionError, match="development gate"):
        module.read_final_source(False, Bomb(), Bomb())


def test_development_option_audit_summary_is_recomputed(monkeypatch):
    rows = [{"source": "AIneun", "learned": True, "baseline": [],
             "additions": [{"text": "AI는"}], "result_count": 1}]
    report = {"child_option_audit": {
        "sha256": "audit", "unique_calls": 1, "changed_calls": 1,
        "added_candidates": 1, "max_added": 4,
        "baseline_prefix_preserved": True,
        "scores_unmodified_from_suffix_options": True,
        "all_additions_forward_valid": True}}
    monkeypatch.setattr(module, "validate_audits", lambda keyed: None)
    monkeypatch.setattr(module, "file_sha256", lambda path: "audit")

    with pytest.raises(ValueError, match="summary"):
        module.validate_development_option_audit(report, rows)


def test_final_identity_and_source_hashes_are_both_required(monkeypatch, tmp_path):
    source = tmp_path / "final.jsonl"; source.write_text("{}\n")
    identity = tmp_path / "identity.json"; identity.write_text("{}")
    monkeypatch.setattr(module, "file_sha256", lambda path: "changed")

    with pytest.raises(ValueError, match="identity"):
        module.read_final_source(True, source, identity)


def test_exact_read_bytes_reject_swap_even_if_path_hash_claims_expected(monkeypatch):
    good_source = b'{"id":"fixed"}\n'
    bad_source = b'{"id":"swapped"}\n'
    good_sha = hashlib.sha256(good_source).hexdigest()
    identity_bytes = json.dumps({"sha256": good_sha}).encode()

    class TracePath:
        def __init__(self, payload):
            self.payload = payload
            self.reads = 0

        def read_bytes(self):
            self.reads += 1
            return self.payload

    source = TracePath(bad_source)
    identity = TracePath(identity_bytes)
    monkeypatch.setattr(module, "FINAL_SOURCE_SHA256", good_sha)
    monkeypatch.setattr(module, "FINAL_IDENTITY_SHA256",
                        hashlib.sha256(identity_bytes).hexdigest())
    monkeypatch.setattr(module, "file_sha256", lambda path: good_sha)

    with pytest.raises(ValueError, match="final source"):
        module.read_final_source(True, source, identity)

    assert source.reads == 1
    assert identity.reads == 1


@pytest.mark.parametrize("loss", ["first", "top5", "pool", "incomplete_top5"])
def test_final_row_level_gate_requires_all_top5_and_zero_losses(loss):
    metrics = {"rows": 1000, "before_top5": 1000, "top5": 1000,
               "regressions": [], "top5_regressions": [],
               "sentence_pool_coverage_losses": []}
    if loss == "first": metrics["regressions"] = ["case"]
    elif loss == "top5": metrics["top5_regressions"] = ["case"]
    elif loss == "pool": metrics["sentence_pool_coverage_losses"] = ["case"]
    else: metrics["top5"] = 999
    assert module.final_passed(metrics) is False
    clean = {**metrics, "top5": 1000, "regressions": [],
             "top5_regressions": [], "sentence_pool_coverage_losses": []}
    assert module.final_passed(clean) is True


def test_post_validation_resource_overflow_blocks_final_gate(monkeypatch):
    calls = []
    monkeypatch.setattr(module, "rss_bytes", lambda: module.PROFILE["rss_limit_bytes"] + 1)
    monkeypatch.setattr(module, "final_passed", lambda metrics: calls.append(metrics) or True)

    with pytest.raises(RuntimeError, match="post-validation"):
        module.finalize_gate({"rows": 1000}, {"same": 1}, {"same": 1}, 0.)
    assert calls == []
