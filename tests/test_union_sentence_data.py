import copy
import json

import pytest

import scripts.prepare_union_sentence_data as module


class TraceSource(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_reads = 0

    def __getitem__(self, key):
        if key == "expected":
            self.target_reads += 1
        return super().__getitem__(key)


class SentenceRanker:
    def __init__(self, source, scores=(2., 1., 0.)):
        self.source = source
        self.scores = scores
        self.calls = 0

    def score_pairs(self, contexts, candidates):
        assert self.source.target_reads == 0
        self.calls += 1
        return list(self.scores[:len(candidates)])


def path(text):
    return (text, 0., [])


def source(expected=("같이 간다", "가치 간다")):
    return TraceSource({
        "id": "case", "origin": "natural", "split": "dev", "group_id": "doc",
        "input": "gachi ganda", "context": "앞 문맥", "expected": list(expected),
        "candidates": ["가치 간다", "까치 간다"],
    })


def test_fresh_capture_proposes_and_scores_before_target_alignment(monkeypatch):
    row = source()
    ranker = SentenceRanker(row)
    converter = type("Converter", (), {"sentence_ranker": ranker})()

    def fake_propose(actual_converter, student, text, context):
        assert row.target_reads == 0
        assert text == row["input"] and context == row["context"]
        return {"parent_paths": [path("가치 간다"), path("까치 간다")],
                "union_paths": [path("가치 간다"), path("까치 간다"), path("같이 간다")],
                "sentence_eligible": True}

    monkeypatch.setattr(module, "propose", fake_propose)
    prepared = module.prepare_fresh_row(row, converter, object())

    assert ranker.calls == 1 and row.target_reads > 0
    assert prepared["parent_candidates"] == row["candidates"]
    assert prepared["candidates"] == ["가치 간다", "까치 간다", "같이 간다"]
    assert prepared["positive_indices"] == [0, 2]
    assert prepared["teacher_scores"] == [2., 1., 0.]
    assert prepared["training_eligible"] is True


def test_uncovered_and_fallback_rows_are_retained_without_teacher_scores():
    row = source(expected=("같이 간다",))
    prepared = module.align_row(
        row, row["candidates"], row["candidates"], False, []
    )

    assert prepared["positive_indices"] == []
    assert prepared["sentence_eligible"] is False
    assert prepared["training_eligible"] is False
    assert prepared["teacher_scores"] == []


def test_fallback_never_calls_sentence_teacher():
    row = source()
    ranker = SentenceRanker(row)

    assert module.score_teacher(ranker, row["candidates"], row["context"], False) == []
    assert ranker.calls == 0


def test_reused_probe_requires_exact_source_order_labels_and_parent_prefix():
    row = source()
    case = {"id": row["id"], "input": row["input"], "context": row["context"],
            "expected": row["expected"], "parent_candidates": row["candidates"],
            "union_candidates": row["candidates"] + ["같이 간다"],
            "sentence_eligible": True}
    module.validate_reused_case(case, row)
    for key, value in (("id", "other"), ("expected", ["다른 정답"]),
                       ("parent_candidates", ["다른 후보"])):
        changed = copy.deepcopy(case)
        changed[key] = value
        with pytest.raises(ValueError):
            module.validate_reused_case(changed, row)


@pytest.mark.parametrize("mutation", ["nan", "short", "bad_positive", "bad_prefix"])
def test_row_validation_rejects_corrupt_teacher_and_alignment(mutation):
    raw = source()
    row = module.align_row(raw, raw["candidates"], raw["candidates"], True, [1., 0.])
    changed = copy.deepcopy(row)
    if mutation == "nan":
        changed["teacher_scores"][0] = float("nan")
    elif mutation == "short":
        changed["teacher_scores"] = [1.]
    elif mutation == "bad_positive":
        changed["positive_indices"] = []
    else:
        changed["candidates"].reverse()
    with pytest.raises(ValueError):
        module.validate_row(changed, raw)


def test_probe_gate_requires_explicit_hash_complete_alignment_and_new_coverage(monkeypatch, tmp_path):
    monkeypatch.setattr(module, "PROBE", tmp_path)
    monkeypatch.setattr(module, "PROFILE", {**module.PROFILE, "natural_train": 1})
    monkeypatch.setattr(module, "probe_source_hashes", lambda: {"source": "pinned"})
    raw = source()
    case = {"id": raw["id"], "input": raw["input"], "context": raw["context"],
            "expected": raw["expected"], "parent_candidates": raw["candidates"],
            "union_candidates": raw["candidates"] + ["같이 간다"],
            "sentence_eligible": True}
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(json.dumps(case, ensure_ascii=False) + "\n")
    report = {"profile": module.PROBE_PROFILE, "parent_version": module.RUNTIME_VERSION,
              "student_weight_sha256": module.STUDENT_WEIGHT,
              "preparation_manifest_sha256": module.CONTEXT_MANIFEST,
              "source_hashes": {"source": "pinned"}, "eligible_for_joint_training": True,
              "all_parent_candidates_preserved": True, "test_data_read": False,
              "development_inference_run": False, "regressions": ["reported-not-hidden"],
              "new_coverage": [raw["id"]],
              "cases_sha256": module.file_sha256(cases_path)}
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))
    pin = module.file_sha256(report_path)

    assert module.load_probe(pin, {"natural_train": [raw]})[1] == [case]
    with pytest.raises(ValueError, match="trusted hash"):
        module.load_probe("0" * 64, {"natural_train": [raw]})
    report["new_coverage"] = []
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="training gate"):
        module.load_probe(module.file_sha256(report_path), {"natural_train": [raw]})


def test_manifest_validation_rejects_tampered_output(monkeypatch, tmp_path):
    profile = {**module.PROFILE, "natural_train": 1, "natural_dev": 1, "retention_train": 1}
    monkeypatch.setattr(module, "PROFILE", profile)
    raw = source()
    joint = {role: [raw] for role in module.ROLES}
    probe_report = {"verified": True}
    monkeypatch.setattr(module, "validate_context_data", lambda *_: ({}, joint))
    monkeypatch.setattr(module, "load_probe", lambda *_: (probe_report, [{}]))
    monkeypatch.setattr(module, "source_hashes", lambda: {"source": "pinned"})
    monkeypatch.setattr(module, "source_digest", lambda data: "digest")
    outputs = {}
    prepared = module.align_row(raw, raw["candidates"], raw["candidates"], False, [])
    for role in module.ROLES:
        path_out = tmp_path / f"{role}.jsonl"
        path_out.write_text(json.dumps(prepared, ensure_ascii=False) + "\n")
        outputs[role] = {"rows": 1, "bytes": path_out.stat().st_size,
                         "sha256": module.file_sha256(path_out)}
    manifest = {"schema_version": 1, "profile": profile,
                "parent_runtime_version": module.RUNTIME_VERSION,
                "student_weight_sha256": module.STUDENT_WEIGHT,
                "context_manifest_sha256": module.CONTEXT_MANIFEST,
                "probe_report_sha256": "probe-pin", "probe_report": probe_report,
                "source_hashes": {"source": "pinned"}, "source_digest": "digest",
                "outputs": outputs, "test_data_read": False,
                "uncovered_rows_retained": True, "fallback_rows_excluded_from_training": True}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    pin = module.file_sha256(manifest_path)

    module.validate_union_data(tmp_path, pin)
    target = tmp_path / "natural_dev.jsonl"
    target.write_text(target.read_text().replace("까치", "변조"))
    with pytest.raises(ValueError, match="integrity"):
        module.validate_union_data(tmp_path, pin)


def test_prepare_refuses_existing_output_before_any_capture(tmp_path):
    with pytest.raises(FileExistsError):
        module.prepare(tmp_path, "probe")
