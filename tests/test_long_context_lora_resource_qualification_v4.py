"""CPU tests for the policy-v4 historical resource qualification."""
import copy
import importlib.util
from pathlib import Path

import pytest


PATH = (Path(__file__).resolve().parents[1] /
        ".omx/artifacts/long-context-lora-work-v4/qualify_resources.py")
SPEC = importlib.util.spec_from_file_location("resource_qualification_v4", PATH)
q = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(q)


def test_actual_failed_v3_artifacts_qualify_without_relabeling():
    result = q.qualify_actual()
    assert result["qualification_passed"] is True
    assert result["qualified_for_one_fresh_fit"] is True
    assert result["historical_probe_passed"] is False
    assert result["historical_failure_stage"] == "runtime_projection"
    assert result["historical_process_returncode"] == 1
    assert result["historical_budget_seconds"] == 21600.0
    assert result["qualified_budget_seconds"] == 28800.0
    assert result["source_files_verified"] == 113
    assert result["resource_observations_verified"] == 497
    assert result["durable_snapshots_verified"] == 6
    assert result["stores_verified"] == 5
    assert result["retained_payload_pairs_byte_verified"] == 10
    assert result["checkpoint_helper_exact_two_literal_transform"] is True
    assert not any(result[key] for key in
                   ("real_fit_rows_read", "labels_read", "mlx_imported",
                    "qwen_model_loaded", "actual_training_run"))


@pytest.mark.parametrize("field,value", [
    ("completed", True), ("resource_feasible", True),
    ("failure_stage", "anything_else"), ("active_gate", "anything_else"),
    ("failure", "different failure"), ("real_fit_rows_read", True),
    ("labels_read", True), ("real_training_run", True),
])
def test_historical_failure_and_source_free_mutations_are_rejected(field, value):
    report = q._read_json(q.REPORT)
    report[field] = value
    with pytest.raises(q.QualificationError):
        q._validate_report(q._load_parent_verifier(), q.ARTIFACT, report)


@pytest.mark.parametrize("mutation", ["missing", "list", "metric", "global_metric"])
def test_real_stage_keyed_checkpoint_resource_mapping_mutations_fail(mutation):
    v = q._load_parent_verifier()
    report = q._read_json(q.REPORT)
    records = copy.deepcopy(report["main"]["updates"])
    observations = copy.deepcopy(report["resource_observations"])
    mapping = records[0]["checkpoint"]["resource_observations"]
    if mutation == "missing":
        mapping.pop("precommit")
    elif mutation == "list":
        records[0]["checkpoint"]["resource_observations"] = list(mapping.values())
    elif mutation == "metric":
        mapping["precommit"]["rss_peak_bytes"] += 1
    else:
        row = next(item for item in observations
                   if item["stage"] == "update_1_checkpoint_precommit")
        (row["post"] or row["pre"])["rss_peak_bytes"] += 1
    with pytest.raises(q.QualificationError):
        q._bind_checkpoint_resource_mapping(v, records, observations, "")


def test_actual_stage_keyed_checkpoint_resources_bind_all_runs():
    v = q._load_parent_verifier()
    report = q._read_json(q.REPORT)
    q._bind_checkpoint_resource_mapping(v, report["reference"]["updates"],
                                        report["resource_observations"], "reference_")
    q._bind_checkpoint_resource_mapping(v, report["main"]["updates"],
                                        report["resource_observations"], "")
    for occurrence, branch in enumerate(report["branches"], 1):
        q._bind_checkpoint_resource_mapping(v, branch["updates"],
                                            report["resource_observations"], "", occurrence)


def test_policy_and_evidence_pins_match_disk():
    for key, path in {"report": q.REPORT, "process_audit": q.PROCESS_AUDIT,
                      "failure_audit": q.FAILURE_AUDIT,
                      "parent_verifier": q.PARENT_VERIFIER,
                      "checkpoint_helper_v2": q.FROZEN_HELPER,
                      "checkpoint_helper_v3": q.SUCCESSOR_HELPER}.items():
        assert q.sha256_file(path) == q.EVIDENCE_SHA256[key]
    assert all(q.sha256_file(q.ROOT / path) == digest
               for path, digest in q.POLICY_FILES.items())


def test_qualification_schema_has_no_caller_selected_policy():
    result = q.qualify_actual()
    assert set(result) == {
        "schema", "qualification_passed", "qualified_for_one_fresh_fit",
        "historical_probe_passed", "historical_failure_stage",
        "historical_process_returncode", "historical_budget_seconds",
        "qualified_budget_seconds", "observed_estimated_training_seconds",
        "runtime_estimate", "reason", "physical_limits_unchanged_and_passed",
        "parity_and_restart_evidence_passed",
        "checkpoint_helper_exact_two_literal_transform", "source_files_verified",
        "resource_observations_verified", "durable_snapshots_verified",
        "stores_verified", "retained_payload_pairs_byte_verified",
        "real_fit_rows_read", "labels_read", "mlx_imported", "qwen_model_loaded",
        "actual_training_run", "artifact_sha256", "qualifier_sha256",
    }
    assert result["runtime_estimate"]["within_21600_seconds"] is False
    assert result["observed_estimated_training_seconds"] <= 28800.0


@pytest.mark.parametrize("mutation", ("extra", "one", "old", "third"))
def test_checkpoint_helper_allows_exactly_two_budget_literal_changes(mutation):
    frozen = q.FROZEN_HELPER.read_bytes()
    successor = frozen.replace(b"21600.0", b"28800.0")
    if mutation == "extra":
        successor += b"\n# drift\n"
    elif mutation == "one":
        successor = successor.replace(b"28800.0", b"21600.0", 1)
    elif mutation == "old":
        frozen = frozen.replace(b"21600.0", b"21599.0", 1)
    else:
        successor += b"\n28800.0\n"
    with pytest.raises(q.QualificationError):
        q._validate_helper_successor(frozen, successor)


@pytest.mark.parametrize("mutation", ("math", "logit", "timing", "projection", "physical"))
def test_actual_report_math_logit_timing_projection_and_physical_mutations_fail(mutation):
    report = q._read_json(q.REPORT)
    if mutation == "math":
        report["reference"]["updates"][0]["mean_loss"] += 0.25
    elif mutation == "logit":
        report["branches"][0]["final_state"]["source_only_logits"][0]["logits"][0] += 1.0
    elif mutation == "timing":
        report["main"]["updates"][0]["seconds"] = 121.0
    elif mutation == "projection":
        report["runtime_estimate"]["estimated_training_seconds"] -= 1.0
    else:
        report["resource_observations"][0]["pre"]["rss_peak_bytes"] = 7 * 1024**3
    with pytest.raises(Exception):
        q._validate_report(q._load_parent_verifier(), q.ARTIFACT, report)


def test_retained_actual_tensor_mutation_is_rejected(tmp_path):
    v = q._load_parent_verifier()
    source = (q.ARTIFACT / "reference/checkpoints/segment-0001/update-0004/"
              "optimizer.safetensors")
    changed = bytearray(source.read_bytes())
    changed.pop()
    target = tmp_path / "optimizer.safetensors"
    target.write_bytes(changed)
    with pytest.raises(v.VerificationError):
        v.inspect_tensors(target, kind="optimizer", update=4)


def test_process_and_policy_pin_mutations_fail(monkeypatch):
    monkeypatch.setitem(q.EVIDENCE_SHA256, "process_audit", "0" * 64)
    with pytest.raises(q.QualificationError):
        q.qualify_actual()
    monkeypatch.undo()
    monkeypatch.setitem(q.POLICY_FILES, ".omx/plans/long-context-lora-budget-v4.md", "0" * 64)
    with pytest.raises(q.QualificationError):
        q.qualify_actual()
