import hashlib
import json

import pytest

from scripts import prepare_roman_sentence_dev as prepare


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def fixture(tmp_path):
    corpus = tmp_path / "corpus"
    audit = tmp_path / "audit"
    baseline = tmp_path / "baseline"
    corpus.mkdir(); audit.mkdir(); baseline.mkdir()
    labels = [
        {"id": "dev-a", "split": "dev", "input": "roman sentence",
         "context": "", "context_role": "none", "input_contract": "roman_sentence_only",
         "expected": ["정답 문장"], "category": "natural_sentence_context"},
        {"id": "dev-b", "split": "dev", "input": "second sentence",
         "context": "", "context_role": "none", "input_contract": "roman_sentence_only",
         "expected": ["다른 정답"], "category": "liaison"},
    ]
    dev = corpus / "dev.jsonl"
    write_jsonl(dev, labels)
    manifest = corpus / "manifest.json"
    manifest.write_text(json.dumps({
        "input_contract": "roman_sentence_only", "model_input_fields": ["input"],
        "counts": {"dev": 2}, "sha256": {"dev.jsonl": sha(dev)},
    }))
    baseline_rows = []
    audit_rows = []
    for index, row in enumerate(labels):
        displayed = [{"text": f"shown-{index}", "score": 1.0, "segments": []}]
        baseline_rows.append({
            "id": row["id"],
            "request": {"input": row["input"], "context": "", "engine": "neural",
                        "mode": "batch", "raw": False},
            "response": {"version": "v1", "candidates": displayed,
                         "result": f"shown-{index}"},
        })
        audit_rows.append({
            "id": row["id"], "source": row["input"], "source_only": True,
            "target_fields_read": False,
            "final_pre_display_pool": [
                {"text": f"pool-{index}-a", "score": 3.25 + index},
                {"text": f"pool-{index}-b", "score": -7.5 - index},
            ],
            "ranker_called": {"word": True, "sentence": index == 0},
            "displayed_candidates": displayed, "result": f"shown-{index}",
            "version": "v1",
        })
    baseline_capture = baseline / "capture.jsonl"
    write_jsonl(baseline_capture, baseline_rows)
    baseline_report = baseline / "report.json"
    baseline_report.write_text(json.dumps({
        "source_sha256": sha(dev), "manifest_sha256": sha(manifest),
        "capture_sha256": sha(baseline_capture), "model_artifacts": {},
    }))
    audit_capture = audit / "capture.jsonl"
    write_jsonl(audit_capture, audit_rows)
    audit_report = audit / "report.json"
    audit_report.write_text(json.dumps({
        "completed": True, "schema": "sentence-only-stage-audit-v1",
        "input_contract": "roman_sentence_only", "external_context_allowed": False,
        "labels_joined_after_all_captures": True,
        "capture_sha256": sha(audit_capture), "source_sha256": sha(dev),
        "manifest_sha256": sha(manifest),
        "baseline_capture_sha256": sha(baseline_capture),
        "model_artifacts": {}, "counts": {"rows": 2},
    }))
    return corpus, audit, baseline, audit_rows


def refresh_audit_hash(audit):
    report_path = audit / "report.json"
    report = json.loads(report_path.read_text())
    report["capture_sha256"] = sha(audit / "capture.jsonl")
    report_path.write_text(json.dumps(report))


def test_preserves_fixed_sources_candidates_scores_and_displayed_texts(tmp_path):
    corpus, audit, baseline, _ = fixture(tmp_path)
    output = tmp_path / "output"
    report = prepare.run(output, audit, baseline, corpus)
    rows = prepare.read_jsonl(output / "dev.jsonl")
    assert rows[0] == {
        "id": "dev-a", "input": "roman sentence",
        "candidates": ["pool-0-a", "pool-0-b"],
        "teacher_scores": [3.25, -7.5], "result": "shown-0",
        "sentence_ranked": True, "displayed": ["shown-0"],
        "expected": ["정답 문장"], "category": "natural_sentence_context",
    }
    assert rows[1]["sentence_ranked"] is False
    assert report["rows"] == 2
    assert report["dev_sha256"] == sha(output / "dev.jsonl")
    assert report["inference_run"] is False


def test_rejects_mismatched_capture_hash(tmp_path):
    corpus, audit, baseline, rows = fixture(tmp_path)
    rows[0]["final_pre_display_pool"][0]["score"] = 99.0
    write_jsonl(audit / "capture.jsonl", rows)
    with pytest.raises(ValueError, match="capture hash"):
        prepare.run(tmp_path / "output", audit, baseline, corpus)


def test_rejects_source_mismatch_even_with_updated_capture_hash(tmp_path):
    corpus, audit, baseline, rows = fixture(tmp_path)
    rows[0]["source"] = "altered sentence"
    write_jsonl(audit / "capture.jsonl", rows)
    refresh_audit_hash(audit)
    with pytest.raises(ValueError, match="source differs"):
        prepare.run(tmp_path / "output", audit, baseline, corpus)


def test_rejects_id_or_order_mismatch(tmp_path):
    corpus, audit, baseline, rows = fixture(tmp_path)
    write_jsonl(audit / "capture.jsonl", list(reversed(rows)))
    refresh_audit_hash(audit)
    with pytest.raises(ValueError, match="IDs/order"):
        prepare.run(tmp_path / "output", audit, baseline, corpus)


@pytest.mark.parametrize("score", [None, float("inf"), float("nan")])
def test_rejects_missing_or_nonfinite_actual_score(tmp_path, score):
    corpus, audit, baseline, rows = fixture(tmp_path)
    rows[0]["final_pre_display_pool"][0]["score"] = score
    write_jsonl(audit / "capture.jsonl", rows)
    refresh_audit_hash(audit)
    with pytest.raises(ValueError, match="nonfinite candidate score"):
        prepare.run(tmp_path / "output", audit, baseline, corpus)
