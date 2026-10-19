import ast
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts import verify_production_phonetic_choice_dev as verifier


def segment(source, text, kind="fixture"):
    return {"input_start": 0, "input_end": len(source), "source": source,
            "text": text, "kind": kind}


def fixture():
    source, target = "ga", "가"
    path = (target, 2.0, [segment(source, target)])
    identifier = verifier.pair_id(source, target)
    score = {"pair_id": identifier, "source": source, "target": target,
             "log_probability": -4.0, "token_count": 4,
             "mean_log_probability": -1.0}
    edge = {"input_start": 0, "input_end": 2, "source": source,
            "target": target, "status": "supported", "pair_id": identifier,
            "romanize_match": True}
    row = {"text": target, "base_score": 2.0, "features": [-1.0, 1.0, 0.0, 0.0],
           "residual": 0.0, "adjusted_score": 2.0,
           "segments": path[2], "edges": [edge]}
    candidates = verifier.render_public(source, [path])
    ctc_word = {"generation_ms": 1.0, "stable": "word"}
    ctc_union = {"generation_ms": 2.0, "stable": "union"}
    artifacts = {"adapter.py": "hash"}
    version_payload = json.dumps({"parent": verifier.PRODUCTION_VERSION,
                                  "artifacts": artifacts, "weights": [0.0] * 4,
                                  "source": "override"},
                                 sort_keys=True, separators=(",", ":"))
    version = "experimental-production-phonetic-choice-1-" + hashlib.sha256(
        version_payload.encode()).hexdigest()[:12]
    parent = {"input": source, "context": "", "external_context_used": False,
              "version": verifier.PRODUCTION_VERSION, "result": target,
              "candidates": candidates, "sentence_ranking_applied": True,
              "ctc_word": ctc_word, "ctc_union": ctc_union, "engine_ms": 3.0}
    proposed = {**copy.deepcopy(parent), "version": version,
                "phonetic_choice": {
                    "experimental": True, "choice_ms": 1.0, "likelihood_ms": .5,
                    "calibration_ms": .25, "parent_version": verifier.PRODUCTION_VERSION,
                    "version": version, "artifacts": artifacts,
                    "effective_weights": [0.0] * 4, "calibration_source": "override",
                    "rows": [row], "score_records": [score],
                    "supported_pair_count": 1,
                    "edge_status_counts": {"supported": 1},
                    "zero_weight_parity": True,
                    "zero_weight_score_parity": True,
                    "zero_weight_order_parity": True,
                    "public": [item["text"] for item in candidates],
                }}
    expected = {"id": "case", "request": {"input": source, "context": ""},
                "parent": copy.deepcopy(parent), "parent_pool": [target]}
    probe = {"id": "case", "request": {"input": source, "context": ""},
             "parent": copy.deepcopy(parent), "proposed": proposed,
             "parent_paths": [path], "base_paths": [path],
             "exact_full_path_parity": True}
    group = {"probe_id": "case"}
    return source, path, expected, probe, group


def test_verifier_has_no_producer_adapter_projector_or_tensor_imports():
    tree = ast.parse(Path(verifier.__file__).read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    forbidden = ("evaluate_production_phonetic_choice_dev",
                 "production_phonetic_choice_converter",
                 "phonetic_choice_features", "torch")
    assert not any(name.endswith(item) or name == item
                   for name in imports for item in forbidden)


def test_zero_probe_directly_checks_both_recorded_full_paths(monkeypatch):
    monkeypatch.setattr(verifier, "romanize_word", lambda text: "ga")
    _, _, expected, probe, group = fixture()
    values = verifier.verify_probe(group, probe, expected)
    assert values == {"paths": 1, "pairs": 1, "supported_edges": 1}

    broken = copy.deepcopy(probe)
    broken["parent_paths"][0] = ("나", 2.0, broken["parent_paths"][0][2])
    with pytest.raises(ValueError, match="full path parity"):
        verifier.verify_probe(group, broken, expected)


def test_response_replays_features_and_rejects_union_proposal_trace(monkeypatch):
    monkeypatch.setattr(verifier, "romanize_word", lambda text: "ga")
    source, path, expected, probe, _ = fixture()
    verifier.verify_response(source, expected, probe["proposed"], [path], [0.0] * 4)

    bad_feature = copy.deepcopy(probe["proposed"])
    bad_feature["phonetic_choice"]["rows"][0]["features"][1] = .5
    with pytest.raises(ValueError, match="feature vector"):
        verifier.verify_response(source, expected, bad_feature, [path], [0.0] * 4)

    bad_generator = copy.deepcopy(probe["proposed"])
    bad_generator["phonetic_word_union"] = {"generated_extras": []}
    with pytest.raises(ValueError, match="proposal path"):
        verifier.verify_response(source, expected, bad_generator, [path], [0.0] * 4)


def test_response_rejects_changed_production_pool_order(monkeypatch):
    monkeypatch.setattr(verifier, "romanize_word", lambda text: "ga")
    source, path, expected, probe, _ = fixture()
    other = ("나", 1.0, [segment(source, "나")])
    expected["parent_pool"] = ["가", "나"]
    with pytest.raises(ValueError, match="base pool"):
        verifier.verify_response(source, expected, probe["proposed"],
                                 [other, path], [0.0] * 4)


def test_probe_groups_are_balanced_and_cover_all_316_sources():
    requests = [(f"id-{index:03d}", "x" * (index % 23 + 1), None)
                for index in range(316)]
    groups = verifier.probe_groups(requests)
    assert len(groups) == 16
    assert sum(group["count"] for group in groups) == 316
    assert max(group["count"] for group in groups) - min(
        group["count"] for group in groups) == 1
    assert sorted(identifier for group in groups for identifier in group["ids"]) == sorted(
        identifier for identifier, _, _ in requests)


def test_exact_jsonl_reader_rejects_trailing_records(tmp_path):
    path = tmp_path / "capture.jsonl"
    path.write_text('{"id":"one"}\n{"id":"injected"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="row count differs"):
        verifier.read_exact_jsonl(path, 1)


def test_resources_reject_overlong_intermediate_watchdog_sample():
    stamp = datetime.now(timezone.utc).isoformat()

    def row(stage, elapsed, operation, operation_elapsed):
        return {"at": stamp, "elapsed_s": elapsed, "operation": operation,
                "operation_elapsed_s": operation_elapsed, "rss_bytes": 1,
                "swap_used_bytes": 100, "swap_growth_bytes": 0, "stage": stage}

    rows = [
        row("operation_start", 0.0, "work", 0.0),
        row("watchdog", 1.0, "work", 61.0),
        row("operation_end", 2.0, "work", 2.0),
        row("after_report", 2.1, None, None),
        row("operation_start", 2.2, "write_completion", 0.0),
        row("operation_end", 2.3, "write_completion", .1),
        row("after_completion", 2.4, None, None),
    ]
    with pytest.raises(ValueError, match="sampled operation duration"):
        verifier.verify_resources(rows, ["work", "write_completion"])


def test_aggregate_requires_correction_without_regression_or_ascii_loss():
    parent = {"result": "ganeun", "candidates": [{"text": "ganeun"}],
              "sentence_ranking_applied": True}
    proposed = {"result": "가는", "candidates": [{"text": "가는"}],
                "sentence_ranking_applied": True}
    rows = [{"id": "one", "request": {"input": "ganeun test", "context": ""},
             "parent": parent, "parent_pool": ["가는"], "proposed": proposed}]
    labels = [{"id": "one", "input": "ganeun test", "context": "", "split": "dev",
               "expected": ["가는"], "category": "fixture"}]
    metrics = verifier.aggregate(rows, labels)
    assert metrics["corrections"] == 1
    assert metrics["regressions"] == 0
    assert metrics["top5_losses"] == 0
    assert metrics["quality_gate"] is True
