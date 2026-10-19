import copy
from contextlib import contextmanager

import pytest

import scripts.evaluate_numeric_structure_development as module


def train_case():
    return {
        "id": "train", "parent_top1": False, "child_top1": True,
        "parent_top5": False, "child_top5": True,
        "parent_pool_covered": False, "child_pool_covered": True,
    }


def train_report(cases):
    metrics = module.train_metrics(cases)
    return {
        "passed": True, "eligible_for_development": True,
        "scope": "fresh actual hybrid natural TRAIN first512 numeric overlay probe",
        "runtime_version": module.RUNTIME_VERSION,
        "probe_source_sha256": module.TRAIN_SOURCE_SHA256,
        "cases_sha256": module.TRAIN_CASES_SHA256,
        "profile": module.TRAIN_PROFILE,
        "source_hashes": {"same": "hash"}, "source_hashes_after": {"same": "hash"},
        "labels_used_for_candidate_production": False,
        "development_data_read": False, "test_data_read": False,
        "new_weights_created": False, "deployed": False,
        "metrics": metrics,
        "adapter": {"baseline_prefix_and_scores_preserved": True,
                    "only_first_all_converted_assignment": True,
                    "word_options_identity_preserved": True,
                    "hook_restored_after_every_child": True},
    }


def test_train_gate_requires_exact_report_and_row_level_no_regressions(monkeypatch):
    cases = [dict(train_case(), id=str(index)) for index in range(module.TRAIN_PROFILE["rows"])]
    report = train_report(cases)
    module.validate_train_gate(report, cases, module.TRAIN_REPORT_SHA256,
                               module.TRAIN_REPORT_SHA256)
    bad = copy.deepcopy(report)
    bad["metrics"]["regressions"] = ["lost"]
    monkeypatch.setattr(module, "train_metrics", lambda rows: bad["metrics"])
    with pytest.raises(ValueError, match="TRAIN gate"):
        module.validate_train_gate(bad, cases, module.TRAIN_REPORT_SHA256,
                                   module.TRAIN_REPORT_SHA256)


def candidate(text="한글", score=1.0, kind="lexicon"):
    return {"text": text, "score": score, "segments": [
        {"input_start": 0, "input_end": 3, "source": "han", "text": text, "kind": kind}
    ]}


def test_candidate_parity_allows_shared_none_and_rejects_score_drift():
    assert module.candidate_objects_equal([candidate(score=None)], [candidate(score=None)])
    assert module.candidate_objects_equal([candidate(score=1.00001)], [candidate(score=1.0)])
    assert not module.candidate_objects_equal([candidate(score=None)], [candidate(score=1.0)])
    assert not module.candidate_objects_equal([candidate(score=1.1)], [candidate(score=1.0)])


def test_parent_parity_requires_full_pool_and_score_match():
    parent = {"candidates": [candidate()], "sentence_ranking_applied": True}
    reference = {"candidates": [candidate()], "sentence_pool": ["한글"],
                 "sentence_scores": [1.0]}
    module.validate_parent_parity(parent, ["한글"], [1.00001], reference)
    with pytest.raises(ValueError, match="accepted acronym"):
        module.validate_parent_parity(parent, ["한글"], [], reference)


def test_candidate_validation_rejects_changed_raw_and_invalid_numeric_segments():
    base = {"result": "han", "version": module.RUNTIME_VERSION,
            "candidates": [candidate("han", score=None, kind="raw")]}
    module.validate_segments(base, "han")
    changed = copy.deepcopy(base)
    changed["candidates"][0]["segments"][0]["text"] = "한"
    changed["candidates"][0]["text"] = changed["result"] = "한"
    with pytest.raises(ValueError, match="raw"):
        module.validate_segments(changed, "han")
    numeric = copy.deepcopy(changed)
    numeric["candidates"][0]["segments"][0]["kind"] = "numeric"
    numeric["candidates"][0]["segments"][0]["text"] = "가"
    numeric["candidates"][0]["text"] = numeric["result"] = "가"
    with pytest.raises(ValueError, match="forward valid"):
        module.validate_segments(numeric, "han")


class TraceSource(dict):
    def __init__(self, **values):
        super().__init__(**values)
        self.label_reads = 0

    def __getitem__(self, key):
        if key == "expected":
            self.label_reads += 1
        return super().__getitem__(key)


class FakeRecorder:
    def __init__(self):
        self.candidates = None
        self.scores = None

    def reset(self):
        self.candidates = self.scores = None

    def can_score(self, text):
        return True


class FakeConverter:
    def __init__(self, source, recorder):
        self.source = source
        self.sentence_ranker = recorder
        self.word_options = object()
        self.child = False

    def convert(self, text, context):
        assert self.source.label_reads == 0
        output = "한글" if not self.child else "한글"
        self.sentence_ranker.candidates = None
        self.sentence_ranker.scores = None
        return {"result": output, "version": module.RUNTIME_VERSION,
                "candidates": [candidate(output, score=None)],
                "sentence_ranking_applied": False, "engine_ms": 1.0,
                "ranker": "hybrid", "context_used": True,
                "ranking_applied": True, "warnings": []}


@contextmanager
def inactive_overlay(converter, generator):
    converter.child = True
    try:
        yield {"applied": False, "calls": 2, "extras": []}
    finally:
        converter.child = False


def test_paired_fallback_is_exact_and_labels_are_read_after_inference():
    source = TraceSource(id="id", input="han", context="ctx", expected=["한글"])
    recorder = FakeRecorder()
    converter = FakeConverter(source, recorder)
    reference = {"candidates": [candidate("한글", score=None)],
                 "sentence_pool": None, "sentence_scores": None, "expected": ["한글"]}
    row = module.evaluate_case(converter, recorder, object(), source, reference,
                               overlay=inactive_overlay)
    assert source.label_reads == 1
    assert row["before_candidates"] == row["candidates"]
    assert row["overlay_applied"] is False


def test_development_gate_requires_natural_gain_and_no_row_loss():
    clean = {name: {"rows": count, "before_top1": 1, "top1": 1,
                    "regressions": [], "top5_regressions": [],
                    "sentence_pool_coverage_losses": []}
             for name, count in module.DEV_COUNTS.items()}
    clean["natural_dev"]["top1"] = 2
    assert module.development_gate(clean)
    for key in ("regressions", "top5_regressions", "sentence_pool_coverage_losses"):
        changed = copy.deepcopy(clean)
        changed["original_dev"][key] = ["lost"]
        assert not module.development_gate(changed)


def test_read_pinned_hashes_and_parses_the_same_single_read(tmp_path):
    path = tmp_path / "cases.jsonl"
    raw = b'{"id":"one"}\n'
    path.write_bytes(raw)
    digest = module.sha256_bytes(raw)
    original = type(path).read_bytes
    calls = []

    def traced(self):
        calls.append(self)
        return original(self)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(type(path), "read_bytes", traced)
        snapshot, rows = module.read_pinned(path, digest, jsonl=True)
    assert snapshot == raw and rows == [{"id": "one"}] and calls == [path]


def test_dev_loader_uses_both_accepted_hash_maps_and_rejects_conflicts(tmp_path, monkeypatch):
    import hashlib
    import json
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'NATURAL_DEV', tmp_path / 'natural.jsonl')
    monkeypatch.setattr(module, 'NATURAL_IDENTITY', tmp_path / 'natural-identity.json')
    monkeypatch.setattr(module, 'DEV_SUITES', [('dev', 'challenge.jsonl', 'identity.json')])
    monkeypatch.setattr(module, 'DEV_COUNTS', {'natural_dev': 1, 'dev': 1})
    (tmp_path / '.omx/artifacts').mkdir(parents=True)
    row = {'id': 'one', 'input': 'imnida', 'context': ''}
    content = (json.dumps(row) + '\n').encode()
    payloads = {'natural.jsonl': content, 'natural-identity.json': b'{}',
                'challenge.jsonl': content, '.omx/artifacts/identity.json': b'{}'}
    hashes = {}
    for name, raw in payloads.items():
        (tmp_path / name).write_bytes(raw)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    report = {'source_state': {
        'source_hashes': {name: digest for name, digest in hashes.items() if name.startswith('natural')},
        'development_suite_hashes': {name: digest for name, digest in hashes.items() if not name.startswith('natural')},
    }}
    groups = {'natural_dev': [row], 'dev': [row]}
    suites, snapshots = module.read_development_sources(report, groups)
    assert suites == groups
    assert snapshots == hashes
    bad = copy.deepcopy(report)
    bad['source_state']['development_suite_hashes']['natural.jsonl'] = '0' * 64
    with pytest.raises(ValueError, match='hash maps conflict'):
        module.read_development_sources(bad, groups)
