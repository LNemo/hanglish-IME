import copy
import hashlib
import json

import pytest

import scripts.evaluate_numeric_structure_final as module


def development_cases():
    rows = []
    for name, count in module.DEV_COUNTS.items():
        for index in range(count):
            rows.append({
                "id": f"{name}-{index}", "suite": name,
                "before_top1": index != 0, "top1": True,
                "before_top5": True, "top5": True,
                "before_sentence_pool_covered": True,
                "sentence_pool_covered": True,
            })
    return rows


def development_report(rows):
    suites = {name: module.summarize([row for row in rows if row["suite"] == name])
              for name in module.DEV_COUNTS}
    return {
        "passed": True, "eligible_for_further_evaluation": True,
        "scope": "fresh actual hybrid development only for numeric sentence-pool overlay",
        "profile": module.DEVELOPMENT_PROFILE,
        "runtime_version": module.RUNTIME_VERSION,
        "training_report_sha256": module.TRAIN_REPORT_SHA256,
        "training_cases_sha256": module.TRAIN_CASES_SHA256,
        "training_adapter_source_sha256": module.TRAIN_SOURCE_SHA256,
        "accepted_acronym_development_report_sha256": module.ACRONYM_REPORT_SHA256,
        "accepted_acronym_development_cases_sha256": module.ACRONYM_CASES_SHA256,
        "source_hashes": {"scripts/evaluate_numeric_structure_development.py":
                          module.DEVELOPMENT_SOURCE_SHA256},
        "source_hashes_after": {"scripts/evaluate_numeric_structure_development.py":
                                module.DEVELOPMENT_SOURCE_SHA256},
        "model_artifact_hash_count": 16, "cases_sha256": "cases",
        "labels_used_for_candidate_production": False, "final_data_read": False,
        "new_weights_created": False, "deployed": False,
        "overlay": {"baseline_pool_and_score_prefix_preserved": True,
                    "word_options_identity_preserved": True,
                    "beam_hook_restored": True},
        "suites": suites,
    }


def test_development_gate_rebuilds_metrics_and_rejects_row_losses():
    rows = development_cases()
    report = development_report(rows)
    report["cases_sha256"] = module.DEVELOPMENT_CASES_SHA256
    module.validate_development_report(
        report, rows, module.DEVELOPMENT_REPORT_SHA256,
        module.DEVELOPMENT_REPORT_SHA256, module.DEVELOPMENT_CASES_SHA256,
    )
    lost = copy.deepcopy(rows)
    row = next(item for item in lost if item["suite"] == "original_dev")
    row["top1"] = False
    with pytest.raises(ValueError, match="metrics|gate"):
        module.validate_development_report(
            report, lost, module.DEVELOPMENT_REPORT_SHA256,
            module.DEVELOPMENT_REPORT_SHA256, module.DEVELOPMENT_CASES_SHA256,
        )
    with pytest.raises(ValueError, match="approved numeric development"):
        module.validate_development_report(
            report, rows, "0" * 64, module.DEVELOPMENT_REPORT_SHA256,
            module.DEVELOPMENT_CASES_SHA256,
        )


def test_failed_development_gate_prevents_every_final_read():
    class Bomb:
        def read_bytes(self):
            raise AssertionError("final content was opened")

    with pytest.raises(PermissionError, match="development gate"):
        module.read_final_inputs(
            False, final_source=Bomb(), final_identity=Bomb(),
            acronym_report_path=Bomb(), acronym_cases_path=Bomb(),
        )


def test_final_gate_requires_988_parent_full_top5_and_no_row_losses():
    metrics = {"rows": 1000, "before_top1": 988, "top1": 989,
               "before_top5": 1000, "top5": 1000, "regressions": [],
               "top5_regressions": [], "sentence_pool_coverage_losses": []}
    assert module.final_gate(metrics)
    for key in ("regressions", "top5_regressions", "sentence_pool_coverage_losses"):
        changed = copy.deepcopy(metrics)
        changed[key] = ["lost"]
        assert not module.final_gate(changed)
    changed = copy.deepcopy(metrics)
    changed["before_top1"] = 987
    assert not module.final_gate(changed)


def test_final_source_and_identity_use_the_same_pinned_byte_snapshots(
        tmp_path, monkeypatch):
    source_raw = b'{"id":"one","input":"a","context":"","expected":["a"]}\n'
    identity_raw = json.dumps({"sha256": hashlib.sha256(source_raw).hexdigest()}).encode()
    cases_raw = b'{"id":"one"}\n'
    report_raw = json.dumps({"cases_sha256": hashlib.sha256(cases_raw).hexdigest()}).encode()
    paths = []
    for name, raw in (("source", source_raw), ("identity", identity_raw),
                      ("report", report_raw), ("cases", cases_raw)):
        path = tmp_path / name
        path.write_bytes(raw)
        paths.append(path)
    monkeypatch.setattr(module, "FINAL_SOURCE_SHA256", hashlib.sha256(source_raw).hexdigest())
    monkeypatch.setattr(module, "FINAL_IDENTITY_SHA256", hashlib.sha256(identity_raw).hexdigest())
    monkeypatch.setattr(module, "ACRONYM_FINAL_REPORT_SHA256",
                        hashlib.sha256(report_raw).hexdigest())
    monkeypatch.setitem(module.PROFILE, "rows", 1)
    original = type(paths[0]).read_bytes
    calls = []

    def traced(self):
        calls.append(self)
        return original(self)

    monkeypatch.setattr(type(paths[0]), "read_bytes", traced)
    loaded = module.read_final_inputs(
        True, final_source=paths[0], final_identity=paths[1],
        acronym_report_path=paths[2], acronym_cases_path=paths[3],
    )
    assert loaded["source_bytes"] == source_raw
    assert calls == paths


def test_acronym_reference_metrics_are_recomputed():
    rows = []
    for index in range(module.PROFILE["rows"]):
        rows.append({"id": str(index), "before_top1": index < 988,
                     "top1": index < 988, "before_top5": True, "top5": True,
                     "before_sentence_pool_covered": None,
                     "sentence_pool_covered": None})
    metrics = module.summarize(rows)
    report = {"passed": True, "goal_met": False,
              "scope": "gated exposed final regression",
              "final_source_sha256": module.FINAL_SOURCE_SHA256,
              "final_identity_sha256": module.FINAL_IDENTITY_SHA256,
              "cases_sha256": "cases", "deployed": False,
              "outcomes": rows, "metrics": metrics}
    module.validate_acronym_final(report, rows)
    report["metrics"]["top1"] -= 1
    with pytest.raises(ValueError, match="metrics"):
        module.validate_acronym_final(report, rows)
