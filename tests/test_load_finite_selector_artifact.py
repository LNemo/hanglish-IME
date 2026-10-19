import hashlib
import json

import pytest

import scripts.load_finite_selector_artifact as module


def test_explicit_hashes_are_required_before_any_file_access(tmp_path):
    with pytest.raises(ValueError, match="explicit selector report"):
        module.load_selector(tmp_path / "missing", "bad", "a" * 64, {})
    with pytest.raises(ValueError, match="explicit selector checkpoint"):
        module.load_selector(tmp_path / "missing", "a" * 64, "bad", {})


def test_fit_result_rejects_internal_arithmetic_and_identity_errors():
    value = {"rows": 2, "correct": 1, "outcomes": [
        {"id": "a", "role": "natural_fit", "parent_correct": True,
         "chosen_finite_index": 0, "chosen_source_index": 0, "correct": True},
        {"id": "b", "role": "challenge_fit", "parent_correct": False,
         "chosen_finite_index": 1, "chosen_source_index": 2, "correct": False},
    ]}
    assert module._validate_fit_result(value, expected_rows=2) == [
        ("natural_fit", "a", True), ("challenge_fit", "b", False)]
    value["correct"] = 2
    with pytest.raises(ValueError, match="arithmetic"):
        module._validate_fit_result(value, expected_rows=2)
    value["correct"] = 1
    value["outcomes"][1]["id"] = "a"
    value["outcomes"][1]["role"] = "natural_fit"
    with pytest.raises(ValueError, match="arithmetic"):
        module._validate_fit_result(value, expected_rows=2)


def schedule_fixture(tmp_path):
    preserves = [("natural_fit", f"p{i}") for i in range(2712)]
    repairs = [("natural_fit", f"r{i}") for i in range(79)]
    repair_draws = [item for item in repairs[:35] for _ in range(12)]
    repair_draws += [item for item in repairs[35:] for _ in range(11)]
    rows = []
    for step in range(904):
        repair = repair_draws[step]
        keep = preserves[step * 3:(step + 1) * 3]
        rows.append({"step": step + 1, "examples": [
            {"role": repair[0], "id": repair[1], "parent_correct": False},
            *({"role": role, "id": identity, "parent_correct": True}
              for role, identity in keep),
        ]})
    raw = b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows)
    (tmp_path / "schedule.jsonl").write_bytes(raw)
    return rows, raw


def test_schedule_validation_rejects_report_or_file_tamper(tmp_path, monkeypatch):
    rows, raw = schedule_fixture(tmp_path)
    digest = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(module, "SCHEDULE_SHA256", digest)
    repeats = {f"natural_fit:r{i}": 12 if i < 35 else 11 for i in range(79)}
    report = {"schedule_ids": rows, "repair_repeats": repeats}
    assert module._validate_schedule(tmp_path / "report.json", report) == raw
    report["schedule_ids"][0]["step"] = 2
    with pytest.raises(ValueError, match="reported schedule"):
        module._validate_schedule(tmp_path / "report.json", report)


def test_nonfinite_or_oversized_resource_values_fail_report_contract(tmp_path):
    # A short malformed report reaches the completed-report fail-closed gate;
    # it cannot be accepted merely because its explicit hash matches.
    report = {"schema": "finite-candidate-roberta-final-choice-selector-v3",
              "peak_rss_bytes": float("nan")}
    raw = json.dumps(report).encode()
    report_path = tmp_path / "report.json"
    report_path.write_bytes(raw)
    (tmp_path / "checkpoint").mkdir()
    (tmp_path / "checkpoint/metadata.json").write_text("{}")
    (tmp_path / "schedule.jsonl").write_text("")
    with pytest.raises(ValueError):
        module.load_selector(report_path, hashlib.sha256(raw).hexdigest(),
                             "a" * 64, {})


def test_trained_metadata_requires_exact_integer_inference_bounds():
    metadata = {"max_length": 96, "inference_batch_size": 8}
    assert module._valid_inference_bounds(metadata) is True
    for key, value in (("max_length", 95), ("max_length", True),
                       ("inference_batch_size", 9),
                       ("inference_batch_size", True)):
        changed = dict(metadata)
        changed[key] = value
        assert module._valid_inference_bounds(changed) is False


def test_metadata_bound_change_fails_with_same_report_and_model_hashes(
        tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    schedule = [{"examples": [
        {"role": "natural_fit", "id": "repair", "parent_correct": False},
        {"role": "natural_fit", "id": "preserve", "parent_correct": True},
    ]}]
    fit_ids = [("natural_fit", "preserve", True),
               ("natural_fit", "repair", False)]
    warm_artifacts = {name: hashlib.sha256(f"warm:{name}".encode()).hexdigest()
                      for name in module.REQUIRED_ARTIFACTS}
    warm_artifacts["model.safetensors"] = module.WARM_WEIGHT_SHA256
    checkpoint_artifacts = {
        name: hashlib.sha256(f"trained:{name}".encode()).hexdigest()
        for name in module.REQUIRED_ARTIFACTS
    }
    checkpoint_sha = checkpoint_artifacts["model.safetensors"]
    runtime_sources = {
        "scripts/train_finite_final_choice_selector.py": module.TRAINER_SHA256}
    weights = {f"weight-{index}": hashlib.sha256(str(index).encode()).hexdigest()
               for index in range(16)}
    sources = {**runtime_sources, **weights,
               **{f".omx/artifacts/roberta-context-ranker-v1/checkpoint/{name}": digest
                  for name, digest in warm_artifacts.items()}}
    partition = {"path": "fresh/partition.json",
                 "sha256": module.EVALUATION_PARTITION_SHA256,
                 "rows": 1024, "groups": 785}
    report = {
        "schema": "finite-candidate-roberta-final-choice-selector-v3",
        "scope": "FIT-only finite-candidate RoBERTa selector; no deployment decision",
        "recipe": module.RECIPE, "runtime_version": module.RUNTIME_VERSION,
        "plan_sha256": module.PLAN_SHA256,
        "fresh_evaluation_partition": partition,
        "fit_manifest_sha256": module.FIT_MANIFESTS,
        "preflight_sha256": module.PREFLIGHT_SHA256,
        "warm_lineage_sha256": module.LINEAGE_SHA256,
        "warm_checkpoint_metadata_sha256": module.WARM_METADATA_SHA256,
        "warm_checkpoint_artifact_hashes": warm_artifacts,
        "eligibility": module.EXPECTED_ELIGIBILITY,
        "schedule_sha256": module.SCHEDULE_SHA256, "optimizer_steps": 904,
        "peak_rss_bytes": 1, "elapsed_seconds": 1., "training_seconds": 1.,
        "mean_loss": 0., "save_load_score_max_delta": 0.,
        "checkpoint_sha256": checkpoint_sha, "finite_candidate_view": True,
        "model_quality_gate_passed": False, "evaluation_cases_read": False,
        "development_data_read": False, "final_data_read": False,
        "eligible_for_deployment": False, "deployed": False,
        "captured_production_fit": {
            "all_rows": 3492, "all_top1": 1, "eligible_rows": 2791,
            "eligible_top1": 2712,
            "source": "captured production parent responses; not warm RoBERTa scores"},
        "warm_roberta_fit": {}, "trained_roberta_fit": {},
        "schedule_ids": schedule, "source_hashes": sources,
        "source_hashes_after": sources,
    }
    metadata = {
        "format_version": 1, "experiment_schema": report["schema"],
        "training": module.RECIPE, "max_length": 96,
        "inference_batch_size": 8, "frozen_source_hashes": sources,
        "parent_checkpoint_sha256": module.WARM_WEIGHT_SHA256,
        "parent_checkpoint_metadata_sha256": module.WARM_METADATA_SHA256,
        "parent_runtime_version": module.RUNTIME_VERSION,
        "parent_metadata": {"artifact_hashes": warm_artifacts, "max_length": 96},
        "fit_captures": {
            role: {"path": f"capture/{role}", "manifest_sha256": digest}
            for role, digest in module.FIT_MANIFESTS.items()},
        "finite_plan": {"sha256": module.PLAN_SHA256},
        "fresh_evaluation_partition": partition,
        "warm_lineage_sha256": module.LINEAGE_SHA256,
        "preflight_sha256": module.PREFLIGHT_SHA256,
        "schedule_sha256": module.SCHEDULE_SHA256,
        "training_eligibility": module.EXPECTED_ELIGIBILITY,
        "finite_candidate_view": True, "all_case_weight": 1.,
        "evaluation_cases_read": False, "development_data_read": False,
        "final_data_read": False, "eligible_for_deployment": False,
        "deployed": False, "artifact_hashes": checkpoint_artifacts,
    }
    report_raw = json.dumps(report, sort_keys=True).encode()
    report_path = tmp_path / "report.json"
    report_path.write_bytes(report_raw)
    (tmp_path / "schedule.jsonl").write_text("")

    monkeypatch.setattr(module, "_validate_schedule", lambda *_: b"schedule")
    monkeypatch.setattr(module, "_validate_fit_result", lambda *_a, **_k: fit_ids)
    monkeypatch.setattr(
        module, "file_sha256",
        lambda path: (checkpoint_artifacts[path.name]
                      if path.parent == checkpoint else sources[str(path).replace(
                          str(module.ROOT) + "/", "")]))

    def write_metadata(value):
        (checkpoint / "metadata.json").write_text(json.dumps(value))

    write_metadata(metadata)
    explicit_report_sha = hashlib.sha256(report_raw).hexdigest()
    loaded = module.load_selector(report_path, explicit_report_sha,
                                  checkpoint_sha, {
                                      "passed": True,
                                      "integrated_runtime_version": module.RUNTIME_VERSION,
                                      "integrated_runtime_source_hashes": runtime_sources,
                                      "runtime_source_hashes_after": runtime_sources,
                                      "accepted_weight_hashes": weights,
                                  })
    assert loaded["metadata"]["max_length"] == 96

    invalid_report = dict(report)
    invalid_report["fresh_evaluation_partition"] = dict(partition, path="")
    invalid_raw = json.dumps(invalid_report, sort_keys=True).encode()
    report_path.write_bytes(invalid_raw)
    with pytest.raises(ValueError, match="completed finite selector"):
        module.load_selector(report_path, hashlib.sha256(invalid_raw).hexdigest(),
                             checkpoint_sha, {
                                 "passed": True,
                                 "integrated_runtime_version": module.RUNTIME_VERSION,
                                 "integrated_runtime_source_hashes": runtime_sources,
                                 "runtime_source_hashes_after": runtime_sources,
                                 "accepted_weight_hashes": weights,
                             })
    report_path.write_bytes(report_raw)

    metadata["max_length"] = 95
    write_metadata(metadata)
    with pytest.raises(ValueError, match="checkpoint metadata"):
        module.load_selector(report_path, explicit_report_sha, checkpoint_sha, {
            "passed": True,
            "integrated_runtime_version": module.RUNTIME_VERSION,
            "integrated_runtime_source_hashes": runtime_sources,
            "runtime_source_hashes_after": runtime_sources,
            "accepted_weight_hashes": weights,
        })
