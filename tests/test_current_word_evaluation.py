import copy

import pytest

import scripts.evaluate_current_word_ranker as module


def training_gate():
    artifacts = {
        "model.safetensors": "child", "config.json": "config",
        "tokenizer.json": "tokenizer", "vocab.txt": "vocab",
        "tokenizer_config.json": "tokenizer-config",
        "special_tokens_map.json": "special",
    }
    hashes = {**artifacts, "metadata.json": "metadata"}
    frozen = {"source": "fixed"}
    recipe = {"name": "fixed"}
    training = {
        "eligible_for_actual_hybrid_dev": True, "regressions": [],
        "test_data_read": False, "deployed": False,
        "development_rows_read": 128, "development_sampled": False,
        "runtime_version": module.RUNTIME_VERSION,
        "preparation_manifest_sha256": module.PREPARATION_MANIFEST,
        "parent_checkpoint_sha256": module.PARENT_WEIGHT,
        "checkpoint_sha256": "child", "source_hashes": frozen,
        "parent": {"cases": 128, "correct": 80},
        "final": {"cases": 128, "correct": 81}, "recipe": recipe,
    }
    metadata = {
        "artifact_hashes": artifacts,
        "parent_runtime_version": module.RUNTIME_VERSION,
        "parent_checkpoint_sha256": module.PARENT_WEIGHT,
        "frozen_source_hashes": frozen, "test_data_read": False,
        "training": recipe,
        "current_preparation": {"manifest_sha256": module.PREPARATION_MANIFEST},
    }
    return training, metadata, hashes, frozen


def test_training_gate_requires_explicit_report_and_complete_provenance():
    training, metadata, hashes, frozen = training_gate()
    module.validate_training_gate(
        training, metadata, hashes, "report", "report", frozen
    )
    mutations = ("report", "eligible", "regression", "gain", "runtime",
                 "manifest", "parent", "checkpoint", "source", "metadata")
    for mutation in mutations:
        current = copy.deepcopy(training)
        meta = copy.deepcopy(metadata)
        explicit = "report"
        if mutation == "report":
            explicit = "other"
        elif mutation == "eligible":
            current["eligible_for_actual_hybrid_dev"] = False
        elif mutation == "regression":
            current["regressions"] = ["case"]
        elif mutation == "gain":
            current["final"]["correct"] = current["parent"]["correct"]
        elif mutation == "runtime":
            current["runtime_version"] = "changed"
        elif mutation == "manifest":
            meta["current_preparation"]["manifest_sha256"] = "changed"
        elif mutation == "parent":
            meta["parent_checkpoint_sha256"] = "changed"
        elif mutation == "checkpoint":
            current["checkpoint_sha256"] = "changed"
        elif mutation == "source":
            current["source_hashes"] = {"source": "changed"}
        else:
            meta["artifact_hashes"]["config.json"] = "changed"
        with pytest.raises(ValueError):
            module.validate_training_gate(
                current, meta, hashes, explicit, "report", frozen
            )


def test_diagnostic_keeps_failed_group_gate_and_still_requires_provenance():
    training, metadata, hashes, frozen = training_gate()
    training.update(eligible_for_actual_hybrid_dev=False, regressions=['group'])
    module.validate_training_gate(training, metadata, hashes, 'report', 'report', frozen,
                                  diagnostic_only=True)
    with pytest.raises(ValueError):
        module.validate_training_gate(training, metadata, hashes, 'report', 'report', frozen)
    with pytest.raises(ValueError):
        module.validate_training_gate(training, metadata, hashes, 'changed', 'report', frozen,
                                      diagnostic_only=True)
    training['final']['correct'] = training['parent']['correct']
    with pytest.raises(ValueError):
        module.validate_training_gate(training, metadata, hashes, 'report', 'report', frozen,
                                      diagnostic_only=True)


def test_diagnostic_never_grants_further_evaluation_even_with_hybrid_gain():
    assert module.result_gate(clean_results(), diagnostic_only=False) is True
    assert module.result_gate(clean_results(), diagnostic_only=True) is False


def test_suite_hashes_include_corpus_and_identity_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'DEV_SUITES', [('dev', 'corpus.jsonl', 'identity.json')])
    (tmp_path / '.omx/artifacts').mkdir(parents=True)
    corpus = tmp_path / 'corpus.jsonl'
    identity = tmp_path / '.omx/artifacts/identity.json'
    corpus.write_text('original')
    identity.write_text('identity')
    before = module.suite_hashes()
    assert set(before) == {'corpus.jsonl', '.omx/artifacts/identity.json'}
    identity.write_text('changed')
    assert before != module.suite_hashes()
    identity.write_text('identity')
    corpus.write_text('changed')
    assert before != module.suite_hashes()


def test_sentence_recorder_keeps_actual_scores():
    class Delegate:
        def score_pairs(self, contexts, candidates):
            return [.3, -.7]
    recorder = module.SentencePoolRecorder(Delegate())
    assert recorder.score_pairs(['ctx', 'ctx'], ['a', 'b']) == [.3, -.7]
    assert recorder.candidates == ['a', 'b']
    assert recorder.scores == [.3, -.7]


@pytest.mark.parametrize('scores', [[float('nan')], [float('inf')], [True], []])
def test_sentence_recorder_rejects_invalid_scores(scores):
    class Delegate:
        def score_pairs(self, contexts, candidates):
            return scores
    with pytest.raises(ValueError):
        module.SentencePoolRecorder(Delegate()).score_pairs(['ctx'], ['a'])


def clean_results():
    return {name: {
        "rows": count, "before_top1": count - 1,
        "top1": count, "regressions": [], "top5_regressions": [],
        "sentence_pool_coverage_losses": [],
    } for name, count in module.DEV_COUNTS.items()}


@pytest.mark.parametrize("mutation", [
    "missing", "count", "no_gain", "regression", "top5", "coverage",
])
def test_development_gate_requires_natural_gain_and_zero_row_losses(mutation):
    results = clean_results()
    if mutation == "missing":
        del results["reasoning_dev"]
    elif mutation == "count":
        results["selection_dev"]["rows"] -= 1
    elif mutation == "no_gain":
        results["natural_dev"]["top1"] = results["natural_dev"]["before_top1"]
    elif mutation == "regression":
        results["original_dev"]["regressions"] = ["case"]
    elif mutation == "top5":
        results["selection_dev"]["top5_regressions"] = ["case"]
    else:
        results["reasoning_dev"]["sentence_pool_coverage_losses"] = ["case"]
    assert module.development_passed(results) is False
    assert module.development_passed(clean_results()) is True


class TraceCase(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_reads = 0

    def __getitem__(self, key):
        if key == "expected":
            self.target_reads += 1
        return super().__getitem__(key)


class Recorder:
    candidates = None
    scores = None


class Converter:
    def __init__(self, case, parent, child, recorder):
        self.case = case
        self.parent = parent
        self.child = child
        self.ranker = parent
        self.sentence_ranker = recorder

    def convert(self, text, context):
        assert self.case.target_reads == 0
        if self.ranker is self.parent:
            choices = ["parent", text]
            self.sentence_ranker.candidates = ["parent", text]
        else:
            choices = ["child", text]
            self.sentence_ranker.candidates = ["child", text]
        self.sentence_ranker.scores = [2., 1.]
        return {"result": choices[0], "candidates": [{"text": value, "score": 2. - i}
                                                   for i, value in enumerate(choices)],
                "sentence_ranking_applied": True, "engine_ms": 1.}


def test_actual_pair_is_source_only_preserves_full_top5_and_restores_parent():
    case = TraceCase({"id": "case", "input": "raw", "context": "ctx",
                      "expected": ["child"]})
    parent, child, recorder = object(), object(), Recorder()
    converter = Converter(case, parent, child, recorder)

    result = module.evaluate_case(converter, parent, child, recorder, case)

    assert case.target_reads > 0
    assert converter.ranker is parent
    assert result["before_actual"] == ["parent", "raw"]
    assert result["actual"] == ["child", "raw"]
    assert result["before_top1"] is False and result["top1"] is True
    assert result["before_sentence_pool_covered"] is False
    assert result["sentence_pool_covered"] is True
    assert result['before_sentence_pool'] == ['parent', 'raw']
    assert result['sentence_pool'] == ['child', 'raw']
    assert result['before_sentence_scores'] == result['sentence_scores'] == [2., 1.]
    assert result['candidates'] == [{'text': 'child', 'score': 2.}, {'text': 'raw', 'score': 1.}]


def test_summary_records_top1_top5_and_pool_coverage_losses_per_row():
    rows = [
        {"id": "regress", "before_top1": True, "top1": False,
         "before_top5": True, "top5": False,
         "before_sentence_pool_covered": True, "sentence_pool_covered": False},
        {"id": "gain", "before_top1": False, "top1": True,
         "before_top5": False, "top5": True,
         "before_sentence_pool_covered": False, "sentence_pool_covered": True},
    ]

    result = module.summarize(rows)

    assert result["regressions"] == ["regress"]
    assert result["top5_regressions"] == ["regress"]
    assert result["sentence_pool_coverage_losses"] == ["regress"]
    assert result["corrections"] == ["gain"]
    assert result["top5_gains"] == ["gain"]
