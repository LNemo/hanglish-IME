import json
from types import SimpleNamespace

import pytest

import scripts.prepare_context_training as module
from scripts.prepare_natural_word_ranker import Recorder


def option(text, score=7.0, kind="lexicon"):
    return SimpleNamespace(text=text, score=score, kind=kind)


class TraceSource(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.events = []

    def __getitem__(self, key):
        if key == "expected":
            self.events.append("target")
        return super().__getitem__(key)


class FakeConverter:
    def __init__(self, source):
        self.source = source
        self.ranker = None
        self.calls = 0

    def _prepare_candidates(self, text, context, *, learned, raw):
        assert self.source.events == []
        assert learned is True and raw is False
        self.calls += 1
        pieces = [
            ("gachi", 0, (option("가치"), option("같이"))),
            (" ", 5, (option(" ", 0.0, "raw"),)),
            ("gachi", 6, (option("가치"), option("까치"))),
        ]
        preliminary = "가치 가치"
        contexts = ["  [MASK]  가치", "  [MASK]  가치",
                    " 가치  [MASK] ", " 가치  [MASK] "]
        candidates = ["가치", "같이", "가치", "까치"]
        self.ranker.score_pairs(contexts, candidates)
        assert preliminary == "".join(group[2][0].text for group in pieces)
        return pieces, pieces, [], False, True


def test_joint_capture_is_source_only_once_and_retains_multi_positive_repeated_spans():
    source = TraceSource({
        "id": "repeat", "family": "f", "input": "gachi gachi", "context": "",
        "expected": ["같이 가치", "가치 가치"],
    })
    converter = FakeConverter(source)
    recorder = Recorder(SimpleNamespace(score_pairs=lambda contexts, candidates: [1.0, 0.0, 1.0, 0.0]))
    converter.ranker = recorder

    case = module.prepare_case(source, "retention_train", converter, recorder)

    assert converter.calls == 1
    assert source.events and source.events[0] == "target"
    assert len(case["positive_indices"]) == 2
    assert case["groups"][0]["expected"] == ["가치", "같이"]
    assert case["groups"][0]["positive_indices"] == [0, 1]
    assert case["groups"][1]["expected"] == ["가치"]
    assert case["groups"][1]["positive_indices"] == [0]


def test_load_sources_keeps_all_1820_retention_rows_in_original_order(monkeypatch):
    natural_train = [{"id": f"nt-{i}"} for i in range(512)]
    natural_dev = [{"id": f"nd-{i}"} for i in range(128)]
    natural_train = module.select_rows(natural_train, "natural_train", 512)
    natural_dev = module.select_rows(natural_dev, "natural_dev", 128)
    retention = [{"id": f"r-{i}"} for i in range(1820)]
    monkeypatch.setattr(module, "selected_sources", lambda: ({
        "natural_train": natural_train,
        "natural_dev": natural_dev,
        "retention_train": [{"id": "ignored-selection"}],
    }, {"verified": True}))
    monkeypatch.setattr(module, "read_rows", lambda path: retention)

    sources, guard = module.load_sources()

    assert guard == {"verified": True}
    assert sources["retention_train"] == retention
    assert [row["id"] for row in sources["retention_train"]] == [f"r-{i}" for i in range(1820)]


def test_mixed_split_guard_rejects_group_and_target_leakage():
    clean = {
        "natural_train": [{"id": "nt", "group_id": "train-doc", "context": "", "expected": ["학습"]}],
        "retention_train": [{"id": "r", "group_id": "old", "context": "", "expected": ["보존"]}],
        "natural_dev": [{"id": "nd", "group_id": "dev-doc", "context": "", "expected": ["검증 문장"]}],
    }
    assert module.mixed_split_guard(clean)["natural_group_overlap_count"] == 0
    leaked = json.loads(json.dumps(clean, ensure_ascii=False))
    leaked["retention_train"][0]["context"] = "앞 문맥에 검증 문장이 있음"
    with pytest.raises(ValueError, match="mixed training"):
        module.mixed_split_guard(leaked)
    leaked = json.loads(json.dumps(clean, ensure_ascii=False))
    leaked["natural_train"][0]["group_id"] = "dev-doc"
    with pytest.raises(ValueError, match="mixed training"):
        module.mixed_split_guard(leaked)


def test_joint_validation_retains_uncovered_sentence_row():
    raw = {"id": "missing", "family": "f", "input": "gachi", "context": "",
           "expected": ["같이"]}
    case = {**module.source_fields(raw, "retention_train"), "groups": [],
            "candidates": ["까치"], "positive_indices": []}

    module.validate_joint_case(case, raw, "retention_train")


def test_validation_requires_explicit_manifest_pin_and_rejects_tampering(monkeypatch, tmp_path):
    profile = {**module.PROFILE, "natural_train": 1, "natural_dev": 1, "retention_train": 1}
    monkeypatch.setattr(module, "PROFILE", profile)
    raw_sources = {role: [{"id": role, "expected": ["정답"], "context": "", "group_id": role}]
                   for role in module.ROLES}
    prepared = raw_sources
    monkeypatch.setattr(module, "load_sources", lambda: (raw_sources, {"verified": True}))
    monkeypatch.setattr(module, "source_fields", lambda row, role: row)
    monkeypatch.setattr(module, "source_digest", lambda rows: "source-digest")
    monkeypatch.setattr(module, "source_hashes", lambda: {"runtime": "pinned"})
    monkeypatch.setattr(module, "mixed_split_guard", lambda rows: {"clean": True})
    monkeypatch.setattr(module, "validate_joint_case", lambda *args: None)
    outputs = {}
    for role in module.ROLES:
        path = tmp_path / f"{role}.jsonl"
        path.write_text(json.dumps({"id": role}) + "\n")
        outputs[role] = {"rows": 1, "bytes": path.stat().st_size,
                         "sha256": module.file_sha256(path)}
    manifest = {
        "schema_version": 1, "profile": profile, "runtime_version": module.RUNTIME_VERSION,
        "no_gold_injection": True, "test_data_read": False,
        "uncovered_rows_retained": True, "feature_policy": module.FEATURE_POLICY,
        "challenge_guard": {"verified": True}, "mixed_split_guard": {"clean": True},
        "source_digest": "source-digest", "source_hashes": {"runtime": "pinned"},
        "outputs": outputs,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    pin = module.file_sha256(manifest_path)

    assert module.validate_context_data(tmp_path, pin)[1].keys() == prepared.keys()
    with pytest.raises(ValueError, match="explicitly pinned"):
        module.validate_context_data(tmp_path, "0" * 64)
    target = tmp_path / "natural_train.jsonl"
    target.write_text('{"id":"tampered"}\n')
    with pytest.raises(ValueError, match="integrity"):
        module.validate_context_data(tmp_path, pin)


def test_prepare_refuses_existing_output_before_loading_models(tmp_path):
    with pytest.raises(FileExistsError):
        module.prepare(tmp_path)


def test_source_hashes_include_counter_dictionary_source_and_sentence_bundle():
    hashes = module.source_hashes()

    assert "data/external/mecab-ko-dic/NNBC.csv" in hashes
    assert "data/external/mecab-ko-dic/SOURCE.json" in hashes
    assert all(f"data/external/mecab-ko-dic/{name}.csv" in hashes for name in module.FILES)
    assert "checkpoints/sentence_ranker/model.safetensors" in hashes
    assert "checkpoints/sentence_ranker/metadata.json" in hashes
