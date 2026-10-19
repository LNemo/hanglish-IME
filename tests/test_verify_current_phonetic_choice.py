from copy import deepcopy
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pytest

from scripts import verify_current_phonetic_choice as verifier


def component_for(wanted):
    index = 0
    while True:
        component = f"component-{wanted}-{index}"
        if verifier.fold_for(component) == wanted:
            return component
        index += 1


def row(identifier="row-1", component=None, candidates=None, scores=None,
        expected=None, production=None, ranked=True, public=None, source="raw"):
    candidates = ["정답", "오답"] if candidates is None else candidates
    scores = [float(len(candidates) - index)
              for index in range(len(candidates))] if scores is None else scores
    expected = ["정답"] if expected is None else expected
    positives = [index for index, text in enumerate(candidates)
                 if text in set(expected)]
    return {
        "id": identifier, "input": source,
        "component_id": component or component_for(0), "source_shard": 0,
        "ranked": ranked, "training_eligible": bool(ranked and positives),
        "candidates": candidates, "base_scores": scores,
        "features": [[float(index), .1, .2, .3]
                     for index in range(len(candidates))],
        "positive_indices": positives,
        "production_indices": (list(range(len(candidates)))
                               if production is None else production),
        "public_candidates": (list(candidates[:5]) if public is None else public),
        "expected": expected,
    }


def source_rows(rows):
    return [{"id": value["id"], "input": value["input"],
             "component_id": value["component_id"]} for value in rows]


def prepare(rows, monkeypatch):
    monkeypatch.setattr(verifier, "ROWS", len(rows))
    return verifier.validate_feature_rows(rows, source_rows(rows))


def test_fold_assignment_is_component_stable_and_covers_four_folds():
    components = [component_for(fold) for fold in range(4)]
    assert [verifier.fold_for(value) for value in components] == [0, 1, 2, 3]
    assert verifier.fold_for(components[0]) == verifier.fold_for(components[0])


def test_validator_derives_gradient_eligibility_without_dropping_rows(monkeypatch):
    rows = [
        row("gradient"),
        row("all-positive", candidates=["정답", "허용"],
            expected=["정답", "허용"]),
        row("no-positive", candidates=["오답"], expected=["정답"]),
        row("unranked", candidates=[], scores=[], expected=["정답"],
            production=[], ranked=False, public=["원문"]),
    ]
    eligible = prepare(rows, monkeypatch)
    assert [value["id"] for value in eligible] == ["gradient"]
    assert [value["gradient_eligible"] for value in rows] == \
        [True, False, False, False]


def test_validator_rejects_tampered_positive_indices(monkeypatch):
    value = row()
    value["positive_indices"] = [1]
    monkeypatch.setattr(verifier, "ROWS", 1)
    with pytest.raises(ValueError, match="eligibility"):
        verifier.validate_feature_rows([value], source_rows([value]))


def test_public_top5_replaces_latent_fifth_with_raw_source():
    value = row(candidates=["일", "이", "삼", "사", "정답", "raw"],
                scores=[6., 5., 4., 3., 2., 1.], expected=["정답"],
                source="raw")
    outcome = verifier.score_outcome(value, verifier.ANCHOR, np.ones(4))
    assert outcome["latent_top5"] is True
    assert outcome["top5"] is False
    assert outcome["public_top5"] == ["일", "이", "삼", "사", "raw"]


def test_unranked_public_fallback_is_accurate_but_not_scored_coverage(monkeypatch):
    value = row(candidates=[], scores=[], production=[], ranked=False,
                public=["정답", "raw"], expected=["정답"])
    prepare([value], monkeypatch)
    outcome = verifier.score_outcome(
        value, np.array([9., 8., 7., 6., 5.]), np.ones(4))
    assert outcome["first"] is outcome["top5"] is True
    assert outcome["covered"] is False
    assert outcome["cross_entropy"] is None
    assert outcome["fallback_public"] is True


def test_all_positive_has_zero_ce_but_no_gradient(monkeypatch):
    value = row(candidates=["정답", "허용"], expected=["정답", "허용"])
    assert prepare([value], monkeypatch) == []
    outcome = verifier.score_outcome(value, verifier.ANCHOR, np.ones(4))
    assert outcome["covered"] is True
    assert outcome["cross_entropy"] == pytest.approx(0.0)


def test_production_subset_uses_its_own_coverage_domain():
    value = row(candidates=["정답", "오답"], scores=[10., 1.],
                expected=["정답"], production=[1])
    full = verifier.score_outcome(value, verifier.ANCHOR, np.ones(4))
    production = verifier.score_outcome(
        value, verifier.ANCHOR, np.ones(4), [1])
    assert full["first"] is full["covered"] is True
    assert production["first"] is production["covered"] is False
    assert production["cross_entropy"] is None


def test_summary_preserves_all_rows_and_separate_fallback_denominator(monkeypatch):
    ranked = row("ranked", component=component_for(0))
    fallback = row("fallback", component=component_for(1), candidates=[],
                   scores=[], production=[], ranked=False, public=["정답"])
    prepare([ranked, fallback], monkeypatch)
    predictions = [
        verifier.evaluate_row(ranked, verifier.ANCHOR, np.ones(4),
                              verifier.ANCHOR, np.ones(4)),
        verifier.evaluate_row(fallback, verifier.ANCHOR, np.ones(4),
                              verifier.ANCHOR, np.ones(4)),
    ]
    summary = verifier.summarize(predictions)["base_full"]
    assert summary["rows"] == 2
    assert summary["ranked_rows"] == summary["scored_pool_rows"] == 1
    assert summary["fallback_public_rows"] == 1
    assert summary["first"] == 2
    assert summary["coverage"] == summary["coverage_denominator"] == 1
    assert summary["covered_cross_entropy_rows"] == 1


def test_qualification_rejects_denominator_identity_tampering(monkeypatch):
    monkeypatch.setattr(verifier, "ROWS", 2)
    summary = {
        "base_full": {"rows": 2, "evaluated_id_sha256": "same",
                      "covered_cross_entropy_id_sha256": "same",
                      "first": 1, "covered_cross_entropy": 1.0},
        "proposed": {"rows": 2, "evaluated_id_sha256": "same",
                     "covered_cross_entropy_id_sha256": "same",
                     "first": 2, "covered_cross_entropy": .9},
    }
    assert verifier.qualified(summary) is True
    changed = deepcopy(summary)
    changed["proposed"]["covered_cross_entropy_id_sha256"] = "other"
    with pytest.raises(ValueError, match="domains differ"):
        verifier.qualified(changed)


def test_recursive_comparison_rejects_prediction_tampering():
    expected = {"branches": {"proposed": {"first": True,
                                            "cross_entropy": .5}}}
    actual = deepcopy(expected)
    actual["branches"]["proposed"]["first"] = False
    with pytest.raises(ValueError, match="first differs"):
        verifier.recursive_close(actual, expected)


def test_launch_command_binds_feature_and_trainer_arguments(tmp_path):
    features, output = tmp_path / "features", tmp_path / "output"
    digest = "a" * 64
    assert verifier.expected_launch_command(features, digest, output) == [
        str(verifier.ROOT / ".venv/bin/python"), "-m",
        "scripts.train_current_phonetic_choice", "--features",
        str(features.resolve()), "--features-report-sha256", digest,
        "--output", str(output.resolve()),
    ]


def test_process_absent_rejects_live_pid_and_accepts_missing_pid():
    def missing(pid, signal):
        assert (pid, signal) == (42, 0)
        raise ProcessLookupError

    assert verifier.process_absent(42, missing) is True
    with pytest.raises(ValueError, match="still present"):
        verifier.process_absent(42, lambda pid, signal: None)


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False,
                                     sort_keys=True) + "\n")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_full_verifier_accepts_synthetic_completed_training_artifact(
        tmp_path, monkeypatch):
    # The production verifier remains independent. The trainer is imported only
    # here to manufacture a schema-accurate completed artifact.
    from scripts import train_current_phonetic_choice as trainer

    rows = []
    for fold in range(4):
        value = row(f"gradient-{fold}", component=component_for(fold),
                    candidates=["오답", "정답"], scores=[1.0, .9])
        value["features"] = [[0., 0., 0., 0.], [2., .1, .2, .3]]
        rows.append(value)
    rows.extend([
        row("all-positive", component=component_for(0), candidates=["정답"],
            scores=[1.0], expected=["정답"]),
        row("no-positive", component=component_for(1),
            candidates=["오답", "다른 오답"], scores=[1.0, .5],
            expected=["정답"]),
        row("no-candidate", component=component_for(2), candidates=[], scores=[],
            expected=["정답"], production=[], ranked=False, public=["raw"]),
        row("all-positive-two", component=component_for(3),
            candidates=["정답", "허용"], scores=[1.0, .5],
            expected=["정답", "허용"]),
    ])
    trainer.validate_rows(rows, {value["id"]: {
        "id": value["id"], "input": value["input"],
        "component_id": value["component_id"]} for value in rows})

    prepared = tmp_path / "prepared"
    prepared.mkdir()
    shard_ids = [[rows[0]["id"], rows[1]["id"]],
                 [rows[2]["id"], rows[3]["id"]],
                 [rows[4]["id"]], [rows[5]["id"]],
                 [rows[6]["id"]], [rows[7]["id"]]]
    requests = {"rows": [{"id": value["id"], "input": value["input"],
                           "component_id": value["component_id"]}
                          for value in rows], "shards": shard_ids}
    _write_json(prepared / "requests.json", requests)
    _write_json(prepared / "manifest.json", {"rows": len(rows)})
    _write_json(prepared / "independent-verification.json", {"verified": True})
    prepared_hashes = {name: _sha(prepared / name) for name in
                       ("requests.json", "manifest.json",
                        "independent-verification.json")}

    shard_directories = []
    evidence = []
    for fold, identifiers in enumerate(shard_ids):
        directory = tmp_path / f"shard-{fold}"
        directory.mkdir()
        report = {"completed": True, "mode": "capture", "shard": fold,
                  "rows": len(identifiers), "ids": identifiers,
                  "capture_sha256": f"capture-{fold}", "labels_opened": False}
        _write_json(directory / "report.json", report)
        _write_json(directory / "completion.json", {"completed": True})
        _write_json(directory.with_suffix(".launch.json"), {"shard": fold})
        proof = {"verified": True, "mode": "capture", "shard": fold,
                 "capture_sha256": report["capture_sha256"],
                 "report_sha256": _sha(directory / "report.json"),
                 "completion_sha256": _sha(directory / "completion.json"),
                 "launch_sha256": _sha(directory.with_suffix(".launch.json"))}
        _write_json(directory / "independent-verification.json", proof)
        shard_directories.append(directory)
        evidence.append({"shard": fold, "rows": len(identifiers),
                         "capture_sha256": report["capture_sha256"],
                         "report_sha256": proof["report_sha256"],
                         "proof_sha256": _sha(
                             directory / "independent-verification.json"),
                         "launch_sha256": proof["launch_sha256"],
                         "completion_sha256": proof["completion_sha256"]})

    features = tmp_path / "features"
    features.mkdir()
    with (features / "rows.jsonl").open("w") as stream:
        for value in rows:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    closure = {"schema": "current-phonetic-train-source-closure-v1",
               "closed": True, "rows": len(rows), "labels_opened": False,
               "shards": evidence}
    _write_json(features / "source-capture.complete.json", closure)
    feature_report = {
        "schema": "current-phonetic-train-features-v1", "completed": True,
        "counts": {"rows": len(rows)}, "inputs": {"shards": evidence},
        "outputs": {"rows.jsonl": _sha(features / "rows.jsonl"),
                    "source-capture.complete.json": _sha(
                        features / "source-capture.complete.json")},
        "all_rows_preserved": True, "positive_candidates_injected": False,
        "non_fit_json_decoded": False, "non_fit_target_values_used": False,
        "dev_test_hold_reserve_labels_opened": False,
    }
    _write_json(features / "report.json", feature_report)
    feature_report_sha = _sha(features / "report.json")
    _write_json(features / "completion.json",
                {"completed": True, "report_sha256": feature_report_sha})

    deployed = tmp_path / "deployed.json"
    _write_json(deployed, {"schema": "phonetic-score-scale-model-v1",
                           "arm": "learned_scaled", "steps": 200,
                           "parameters": [1., 0., 0., 0., 0.],
                           "scale": [1., 1., 1., 1.]})
    dummy_paths = [tmp_path / name for name in
                   ("feature-preparer.py", "plan.md", "resource.py", "preflight.py")]
    for path in dummy_paths:
        path.write_text(path.name + "\n")
    feature_preparer, plan, resource_helper, preflight = dummy_paths

    monkeypatch.setattr(verifier, "ROWS", len(rows))
    monkeypatch.setattr(verifier, "STEPS", 2)
    monkeypatch.setattr(verifier, "PREPARED", prepared)
    monkeypatch.setattr(verifier, "PREPARED_HASHES", prepared_hashes)
    monkeypatch.setattr(verifier, "SHARDS", tuple(shard_directories))
    monkeypatch.setattr(verifier, "DEPLOYED", deployed)
    monkeypatch.setattr(verifier, "DEPLOYED_SHA256", _sha(deployed))
    monkeypatch.setattr(verifier, "FEATURE_PREPARER", feature_preparer)
    monkeypatch.setattr(verifier, "FEATURE_PREPARER_SHA256", _sha(feature_preparer))
    monkeypatch.setattr(verifier, "PLAN", plan)
    monkeypatch.setattr(verifier, "PLAN_SHA256", _sha(plan))
    monkeypatch.setattr(verifier, "RESOURCE_HELPER", resource_helper)
    monkeypatch.setattr(verifier, "PREFLIGHT", preflight)

    expected_pins = {
        str((features / name).resolve()): _sha(features / name)
        for name in ("report.json", "completion.json", "rows.jsonl",
                     "source-capture.complete.json")}
    expected_pins.update({str((prepared / name).resolve()): digest
                          for name, digest in prepared_hashes.items()})
    trainer_path = Path(trainer.__file__).resolve()
    for path in (deployed, feature_preparer, trainer_path,
                 verifier.NUMERICAL_CORE, plan, resource_helper, preflight):
        expected_pins[str(path.resolve())] = _sha(path)
    expected_pins = dict(sorted(expected_pins.items()))

    def fake_authenticate(path, digest):
        assert Path(path) == features and digest == feature_report_sha
        return rows, json.loads(deployed.read_text()), expected_pins, feature_report

    class Guard:
        def __init__(self, started, unused_swap, trace, **unused):
            self.started, self.trace = started, Path(trace)
            self.operation_name = None
            self.operation_started = None

        def _record(self, stage):
            elapsed = time.monotonic() - self.started
            record = {"stage": stage, "elapsed_s": elapsed, "rss_bytes": 1,
                      "swap_growth_bytes": 0, "operation": self.operation_name,
                      "operation_elapsed_s": (None if self.operation_started is None
                          else time.monotonic() - self.operation_started)}
            with self.trace.open("a") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            return record

        def __enter__(self):
            return self

        def __exit__(self, *unused):
            return False

        @contextmanager
        def operation(self, name, update=False):
            self.operation_name = name
            self.operation_started = time.monotonic() if update else None
            self._record("operation_start")
            yield
            self._record("operation_end")
            self.operation_name = self.operation_started = None

        def sample_and_append(self, stage):
            return self._record(stage)

    monkeypatch.setattr(trainer, "ROWS", len(rows))
    monkeypatch.setattr(trainer, "STEPS", 2)
    monkeypatch.setattr(trainer, "authenticate", fake_authenticate)
    monkeypatch.setattr(trainer, "AtomicResourceWatchdog", Guard)
    monkeypatch.setattr(trainer, "system_swap_bytes", lambda: 0)
    output = tmp_path / "output"
    report = trainer.run(features, feature_report_sha, output)
    assert report["qualification_passed"] is True

    trainer_sha = _sha(trainer_path)
    log = output.with_suffix(".log")
    log.write_text("completed\n")
    launch = {"pid": 42,
              "command": verifier.expected_launch_command(
                  features, feature_report_sha, output),
              "output": str(output.resolve()), "log": str(log),
              "runner_sha256": trainer_sha,
              "launched_at": "2026-09-23T00:00:00+00:00"}
    _write_json(output.with_suffix(".launch.json"), launch)
    proof = verifier.verify(features, feature_report_sha, output, trainer_sha,
                            pid_probe=lambda unused_pid, unused_signal: (
                                (_ for _ in ()).throw(ProcessLookupError())))
    assert proof["verified"] is True
    assert proof["qualification_passed"] is True
    assert proof["counts"]["rows"] == len(rows)
    assert (output / "independent-verification.json").is_file()
