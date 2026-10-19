import hashlib
import json

import pytest

from scripts import audit_sentence_only_stages as audit


def option(text):
    return {"text": text, "score": 0.0, "kind": "fixture"}


def capture(*, source="ab", result="wrong", pool=(), before=None, after=None,
            displayed=(), warnings=(), sentence=True):
    pieces = before if before is not None else [
        {"source": source, "offset": 0, "options": [option("wrong")]}
    ]
    return {
        "id": "x", "source": source, "result": result,
        "pieces_before": pieces,
        "pieces_after": after if after is not None else pieces,
        "final_pre_display_pool": [{"text": text} for text in pool],
        "displayed_candidates": [{"text": text} for text in displayed],
        "ranker_called": {"word": True, "sentence": sentence},
        "runtime_flags": {"warnings": list(warnings)},
    }


def test_classifier_accepts_any_alternate_and_distinguishes_selection_stages():
    assert audit.classify(["first", "alternate"], capture(result="alternate")) == "correct"
    assert audit.classify(["gold"], capture(pool=("wrong", "gold"))) == "final-selection"
    displaced = capture(
        source="roman", pool=("a", "b", "c", "d", "gold", "e"),
        displayed=("a", "b", "c", "d", "roman"),
    )
    assert audit.classify(["other", "gold"], displaced) == "raw-slot-displacement"


def test_classifier_distinguishes_missing_option_cap_and_beam_pruning():
    missing = [{"source": "ab", "offset": 0, "options": [option("wrong")]}]
    assert audit.classify(["gold"], capture(before=missing)) == "missing_piece"

    many = [{"source": "ab", "offset": 0,
             "options": [option(str(index)) for index in range(8)] + [option("gold")]}]
    assert audit.classify(["gold"], capture(before=many)) == "option_cap"

    composable = [
        {"source": "a", "offset": 0, "options": [option("go")]},
        {"source": "b", "offset": 1, "options": [option("ld")]},
    ]
    assert audit.classify(["gold"], capture(before=composable)) == "beam_pruning"


def test_length_bypass_is_independent_of_primary_category():
    value = capture(result="gold", sentence=False,
                    warnings=("문장 모델의 길이 범위를 넘어 단어별 순위를 사용했습니다.",))
    assert audit.classify(["gold"], value) == "correct"
    assert audit.length_bypass(value) is True


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_run_captures_all_sources_before_joining_labels(tmp_path):
    data = tmp_path / "dev.jsonl"
    rows = [
        {"id": "a", "split": "dev", "input": "roman sentence", "context": "",
         "context_role": "none", "expected": ["정답 문장"]},
        {"id": "b", "split": "dev", "input": "second sentence", "context": "",
         "context_role": "none", "expected": ["다른 정답"]},
    ]
    data.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "input_contract": "roman_sentence_only", "model_input_fields": ["input"],
        "corpus_version": "fixture", "sha256": {"dev.jsonl": file_sha(data)},
        "counts": {"dev": 2},
    }))
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    baseline_rows = []
    for row in rows:
        candidate = {"text": "wrong", "score": 0.0, "segments": []}
        baseline_rows.append({
            "id": row["id"],
            "request": {"input": row["input"], "context": "", "engine": "neural",
                        "mode": "batch", "raw": False},
            "response": {"version": "v1", "candidates": [candidate], "result": "wrong"},
        })
    baseline_capture = baseline / "capture.jsonl"
    baseline_capture.write_text("".join(json.dumps(row) + "\n" for row in baseline_rows))
    (baseline / "report.json").write_text(json.dumps({
        "source_sha256": file_sha(data), "manifest_sha256": file_sha(manifest),
        "corpus_version": "fixture", "capture_sha256": file_sha(baseline_capture),
        "counts": {"rows": 2}, "model_artifacts": {},
    }))

    class Converter:
        model_artifacts = {}
        def load_models(self):
            pass

    calls = []
    def trace_fn(_converter, source, context, **kwargs):
        calls.append(source)
        assert context == "" and kwargs == {"engine": "neural", "raw": False}
        candidate = {"text": "wrong", "score": 0.0, "segments": []}
        piece = {"source": source, "offset": 0, "options": [option("wrong")]}
        return {
            "source_only": True, "target_fields_read": False,
            "source": source, "context": "", "pieces_before_word_ranking": [piece],
            "pieces_after_word_ranking": [piece], "final_pre_display_paths": [candidate],
            "word_ranker_score_pairs": [{}], "sentence_score_pairs": [{}],
            "displayed_candidates": [candidate], "result": "wrong",
            "runtime": {"version": "v1", "ranking_applied": True,
                        "sentence_ranking_applied": True,
                        "external_context_used": False, "warnings": []},
        }

    output = tmp_path / "out"
    report = audit.run(data, manifest, baseline, output, Converter(), trace_fn)
    captures = audit.read_jsonl(output / "capture.jsonl")
    assert calls == ["roman sentence", "second sentence"]
    assert all("expected" not in row for row in captures)
    assert report["labels_joined_after_all_captures"] is True
    assert report["counts"] == {"rows": 2, "denominator": 2}
    assert sum(report["categories"].values()) == 2


def test_parity_requires_exact_candidates_and_version():
    value = capture(displayed=("wrong",))
    value.update(version="v1")
    value["displayed_candidates"] = [{"text": "wrong"}]
    value["runtime_flags"]["external_context_used"] = False
    with pytest.raises(ValueError, match="version"):
        audit.parity_check(value, {"version": "v2", "candidates": [{"text": "wrong"}],
                                   "result": "wrong"})
    with pytest.raises(ValueError, match="displayed candidates"):
        audit.parity_check(value, {"version": "v1", "candidates": [{"text": "other"}],
                                   "result": "wrong"})
