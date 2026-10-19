import ast
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = ROOT / ".omx/artifacts/verify-source-span-pair-dev-v1.py"
SPEC = importlib.util.spec_from_file_location("source_span_pair_dev_verifier", VERIFIER_PATH)
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def dump_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def segment(source, text, kind="morphology"):
    return {"input_start": 0, "input_end": len(source), "source": source,
            "text": text, "kind": kind}


def candidate(source, text):
    return {"text": text, "score": 0.0, "segments": [segment(source, text)]}


def fixtures(row_id="row-1", source="abc"):
    first = candidate(source, "가")
    second = candidate(source, "나")
    proposal_segment = segment(source, "다", "neural")
    proposal = {"proposal_id": "proposal-one", "text": "다",
                "segments": [proposal_segment], "edge_ids": ["edge-one"],
                "seed_index": 0, "seed_text": "가", "changed_edges": 1,
                "score": None, "score_type": None,
                "origin": "source_edge_recombination", "selection_eligible": False}
    proposed = {"result": "가", "candidates": [first, second],
                "pipeline": {"pool": [first, second]}, "other": "preserved"}
    runtime = {"id": row_id, "request": {"input": source, "context": ""},
               "elapsed_ms": 12.5, "proposed": proposed}
    proposal_row = {"id": row_id, "request": runtime["request"],
                    "edge_inventory": [{"edge_id": "edge-one",
                        "structural_segment": proposal_segment, "provenance": ["test"]}],
                    "proposals": [proposal]}
    return runtime, proposal_row


def prediction(runtime, proposal, supported=True):
    texts, representatives, groups, path_ids = verifier.reconstruct_domain(runtime, proposal)
    source = runtime["request"]["input"]
    scores = {"parent": [1.0, 3.0, 2.0],
              "whole_source": [2.0, 1.0, 3.0],
              "source_local": [3.0, 2.0, 1.0]}
    responses = {"captured_runtime": verifier.compact_response(runtime["proposed"])}
    hashes = {"captured_runtime": verifier.canonical_hash(runtime["proposed"])}
    if supported:
        for model in verifier.MODELS:
            full = verifier.apply_policy(runtime["proposed"], representatives, texts,
                                         scores[model], source, model)
            responses[model] = verifier.compact_response(full)
            hashes[model] = verifier.canonical_hash(full)
        support = {"supported": True, "reason": None}
        score_rows = scores
    else:
        for model in verifier.MODELS:
            responses[model] = verifier.compact_response(runtime["proposed"])
            hashes[model] = verifier.canonical_hash(runtime["proposed"])
        support = {"supported": False, "reason": "candidate_exceeds_256_paired_tokens"}
        score_rows = {}
    protected = verifier.protected_paths(source, groups)
    response_contracts = {arm: verifier.protected_response(source, response)
                          for arm, response in responses.items()}
    return {"id": runtime["id"], "request": runtime["request"],
            "candidate_texts": texts, "path_ids": path_ids, "support": support,
            "scores": score_rows,
            "latency_ms": {"captured_runtime": 12.5, "parent_features": 1.0,
                           "whole_source": 2.0, "source_local": 3.0},
            "responses": responses, "full_response_sha256": hashes,
            "protected_paths": protected, "protected_responses": response_contracts}


def test_verifier_imports_no_torch_tokenizer_models_or_evaluator():
    tree = ast.parse(VERIFIER_PATH.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not any(name == "torch" or name.startswith("transformers")
                   or name.startswith("hanglish") or name.startswith("scripts")
                   for name in imported)


def test_domain_reconstruction_keeps_policy_order_and_every_distinct_path():
    runtime, proposal = fixtures()
    alternate = candidate("abc", "가")
    alternate["segments"] = [
        {"input_start": 0, "input_end": 1, "source": "a", "text": ""},
        {"input_start": 1, "input_end": 3, "source": "bc", "text": "가"},
    ]
    runtime["proposed"]["candidates"].append(alternate)
    texts, _, _, path_ids = verifier.reconstruct_domain(runtime, proposal)
    assert texts == ["가", "나", "다"]
    assert len(path_ids[0]) == 2 and all(value.startswith("alignment-v1-") for value in path_ids[0])
    proposal["proposals"][0]["edge_ids"] = ["missing"]
    with pytest.raises(ValueError, match="edge/path"):
        verifier.reconstruct_domain(runtime, proposal)


def test_supported_prediction_replays_score_sort_text_tie_break_and_raw_fifth():
    runtime, proposal = fixtures()
    saved = prediction(runtime, proposal)
    assert verifier.verify_prediction(saved, runtime, proposal) is True
    assert saved["responses"]["parent"]["candidates"][-1] == verifier.raw_candidate("abc")
    assert [row["text"] for row in saved["responses"]["parent"]["candidates"][:3]] == ["나", "다", "가"]
    saved["responses"]["parent"]["candidates"][0]["text"] = "변조"
    with pytest.raises(ValueError, match="ranked response"):
        verifier.verify_prediction(saved, runtime, proposal)


def test_fallback_requires_exact_original_compact_response_and_common_empty_scores():
    runtime, proposal = fixtures()
    saved = prediction(runtime, proposal, supported=False)
    assert verifier.verify_prediction(saved, runtime, proposal) is False
    saved["responses"]["source_local"]["result"] = "변조"
    with pytest.raises(ValueError, match="fallback changed"):
        verifier.verify_prediction(saved, runtime, proposal)


def test_protected_url_email_and_explicit_copy_require_copy_spans():
    source = "https://x.test a@b.kr `keep` reul"
    protected_end = source.index(" reul")
    good = {"text": source[:protected_end] + " 를", "segments": [
        {"input_start": 0, "input_end": protected_end,
         "source": source[:protected_end], "text": source[:protected_end]},
        {"input_start": protected_end, "input_end": len(source),
         "source": source[protected_end:], "text": " 를"},
    ]}
    values = verifier.fixtures(source)
    assert {value["kind"] for value in values} == {"url", "email", "explicit_copy"}
    assert all(verifier.path_preserves(source, good, value) for value in values)
    bad = json.loads(json.dumps(good))
    bad["segments"][0]["text"] = bad["segments"][0]["text"].replace("keep", "keap")
    bad["text"] = "".join(segment["text"] for segment in bad["segments"])
    assert not verifier.path_preserves(source, bad, next(v for v in values
                                                          if v["kind"] == "explicit_copy"))


def test_metric_and_decision_replay_rejects_report_mutation(monkeypatch):
    monkeypatch.setattr(verifier, "ROWS", 2)
    runtime1, proposal1 = fixtures("one")
    runtime2, proposal2 = fixtures("two")
    first, second = prediction(runtime1, proposal1), prediction(runtime2, proposal2)
    labels = [{"id": "one", "input": "abc", "expected": ["나"], "category": "hard"},
              {"id": "two", "input": "abc", "expected": ["가"], "category": "hard"}]
    predictions = [first, second]
    metrics = {arm: verifier.summarize(predictions, labels, arm) for arm in verifier.ARMS}
    assert metrics["captured_runtime"]["counts"]["top1"] == 1
    assert metrics["parent"]["counts"]["top1"] == 1
    decision = verifier.decide(metrics)
    mutated = json.loads(json.dumps(decision))
    mutated["source_localization_claim_supported"] = not mutated[
        "source_localization_claim_supported"]
    assert mutated != verifier.decide(metrics)


def resource_row(operation, stage, elapsed, baseline=2000):
    return {"at": "2026-09-21T00:00:00+00:00", "elapsed_s": elapsed,
            "rss_bytes": 1000, "swap_used_bytes": baseline, "swap_growth_bytes": 0,
            "operation_elapsed_s": elapsed if operation is not None else None,
            "stage": stage, "operation": operation}


def test_resource_sequence_matches_only_supported_rows_and_rejects_transition_race(tmp_path):
    runtime1, proposal1 = fixtures("one")
    runtime2, proposal2 = fixtures("two")
    predictions = [prediction(runtime1, proposal1), prediction(runtime2, proposal2, False)]
    names = verifier.expected_operations(predictions)
    trace, elapsed = [], 0.0
    for name in names:
        trace.append(resource_row(name, "operation_start", elapsed)); elapsed += .001
        trace.append(resource_row(name, "watchdog", elapsed)); elapsed += .001
        trace.append(resource_row(name, "operation_end", elapsed)); elapsed += .001
        trace.append(resource_row(None, "watchdog", elapsed)); elapsed += .001
    dump_jsonl(tmp_path / "resources.jsonl", trace)
    report = {"baseline_swap_bytes": 2000, "elapsed_s": elapsed,
              "final_resources": resource_row(None, "final", elapsed)}
    result = verifier.verify_resources(tmp_path, report, predictions)
    assert result["operation_pairs"] == 4 and result["supported_rows"] == 1
    trace[1]["operation"] = names[1]
    dump_jsonl(tmp_path / "resources.jsonl", trace)
    with pytest.raises(ValueError, match="transition"):
        verifier.verify_resources(tmp_path, report, predictions)
