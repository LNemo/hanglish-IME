from dataclasses import dataclass

import scripts.audit_final_choice_candidate_stages as module


@dataclass(frozen=True)
class Option:
    text: str
    score: float
    kind: str


class Ranker:
    def can_score(self, text):
        return True


class Recorder(Ranker):
    def __init__(self):
        self.candidates = None
        self.scores = None

    def reset(self):
        self.candidates = None
        self.scores = None


class LabelGuard(dict):
    inference_finished = False

    def __getitem__(self, key):
        if key == "expected" and not self.inference_finished:
            raise AssertionError("label accessed before inference")
        return super().__getitem__(key)


def segment(source, start, text):
    return {"input_start": start, "input_end": start + len(source),
            "source": source, "text": text, "kind": "syllable"}


class Converter:
    version = module.RUNTIME_VERSION
    model_artifacts = {"weight": "a" * 64}

    def __init__(self, source_row):
        self.source_row = source_row
        self.sentence_ranker = Recorder()
        self.audit_pieces = [
            ("ga", 0, (Option("가", 2., "syllable"),)),
            (" ", 2, (Option(" ", 0., "raw"),)),
            ("na", 3, (Option("나", 2., "syllable"), Option("너", 1., "syllable"))),
        ]
        self.audit_static_pieces = self.audit_pieces

    def convert(self, text, context):
        assert (text, context) == ("ga na", "ctx")
        paths = module.unique_paths([
            *module.beam_paths(self.audit_pieces),
            *module.beam_paths(self.audit_static_pieces),
        ])
        self.sentence_ranker.candidates = [path[0] for path in paths]
        self.sentence_ranker.scores = [1.0, .5]
        response = self.source_row["response"]
        self.source_row.inference_finished = True
        return response

    def _numeric_sentence_extras(self, pieces, static, complete):
        return []


def guarded_row(expected):
    candidates = [{"text": "가 나", "score": 1.0,
                   "segments": [segment("ga", 0, "가"), segment(" ", 2, " "),
                                segment("na", 3, "나")]},
                  {"text": "가 너", "score": .5,
                   "segments": [segment("ga", 0, "가"), segment(" ", 2, " "),
                                segment("na", 3, "너")]}]
    response = {
        "input": "ga na", "context": "ctx", "engine": "neural", "mode": "batch",
        "raw": False, "version": module.RUNTIME_VERSION, "candidates": candidates,
        "result": "가 나", "ranker": "hybrid", "model_artifacts": {"weight": "a" * 64},
        "context_used": True, "external_context_used": True,
        "ranking_applied": True, "sentence_ranking_applied": True,
        "warnings": [], "engine_ms": 1.0,
    }
    row = LabelGuard({
        "id": "id", "group_id": "group", "input": "ga na", "context": "ctx",
        "expected": expected, "candidates": candidates, "result": "가 나",
        "runtime_version": module.RUNTIME_VERSION,
        "response_flags": module._response_flags(response), "response": response,
    })
    return row


def test_labels_do_not_affect_actual_conversion_or_reconstructed_pool():
    left = guarded_row(["가 나"])
    right = guarded_row(["가 너"])
    left_converter = Converter(left)
    left_result = module.audit_case(left_converter, left_converter.sentence_ranker, left)
    right_converter = Converter(right)
    right_result = module.audit_case(right_converter, right_converter.sentence_ranker, right)
    assert left_result["observed_sentence_pool"] == right_result["observed_sentence_pool"]
    assert left_result["learned_pieces"] == right_result["learned_pieces"]
    assert left_result["top1"] is True
    assert right_result["top1"] is False


def test_raw_slot_displacement_and_length_bypass_are_distinct():
    row = {"expected": ["wanted"], "input": "a b"}
    response = {"candidates": [
        {"text": "one", "score": 1.}, {"text": "two", "score": .9},
        {"text": "three", "score": .8}, {"text": "four", "score": .7},
        {"text": "a b", "score": None},
    ]}
    pieces = [("a b", 0, (Option("wanted", 1., "syllable"),
                            Option("other", 0., "syllable")))]
    recorder = Recorder()
    recorder.candidates = ["one", "two", "three", "four", "wanted"]
    recorder.scores = [1., .9, .8, .7, .6]
    learned = [("one", 0., []), ("two", 0., []), ("three", 0., []),
               ("four", 0., []), ("wanted", 0., [])]
    reason, _, _, displaced, bypass, _ = module.stage_reason(
        row, response, recorder, pieces, pieces,
        learned, learned, 5, True,
    )
    assert (reason, displaced, bypass) == ("raw_slot_displacement", True, False)

    recorder.candidates = None
    recorder.scores = None
    multi_pieces = [
        ("a", 0, (Option("one", 1., "syllable"),)),
        (" ", 1, (Option(" ", 0., "raw"),)),
        ("b", 2, (Option("two", 1., "syllable"),
                  Option("wanted", .5, "syllable"))),
    ]
    reason, _, _, displaced, bypass, _ = module.stage_reason(
        row, {"candidates": [{"text": "one two", "score": 1.}]}, recorder,
        multi_pieces, multi_pieces, [("one two", 1., [])],
        [("one two", 1., [])], 1, False,
    )
    assert (reason, displaced, bypass) == ("length_bypass", False, True)


def test_sentence_recorder_rejects_truncated_scores():
    class Delegate(Ranker):
        def score_pairs(self, contexts, candidates):
            return [1.0]

    recorder = module.SentenceRecorder(Delegate())
    try:
        recorder.score_pairs(["", ""], ["a", "b"])
    except RuntimeError as error:
        assert "malformed" in str(error)
    else:
        raise AssertionError("truncated scores accepted")


def test_raw_displacement_uses_scored_rank_five_not_input_order_or_rank_six():
    row = {"expected": ["wanted"], "input": "a b"}
    response = {"candidates": [
        {"text": str(index), "score": float(10 - index)} for index in range(4)
    ] + [{"text": "a b", "score": None}]}
    pieces = [("a", 0, (Option("x", 1., "syllable"),
                         Option("y", 0., "syllable")))]
    recorder = Recorder()
    # Input order puts wanted fifth, but its score ranks it sixth.
    recorder.candidates = ["0", "1", "2", "3", "wanted", "4"]
    recorder.scores = [6., 5., 4., 3., 1., 2.]
    learned = [(text, score, []) for text, score in
               zip(recorder.candidates, recorder.scores)]
    reason, _, ranked, displaced, _, _ = module.stage_reason(
        row, response, recorder, pieces, pieces, learned, learned, 6, True,
    )
    assert ranked[5] == "wanted"
    assert displaced is False
    assert reason == "sentence_ranking_loss"


def test_numeric_extra_only_coverage_is_sentence_ranking_loss():
    row = {"expected": ["numeric-extra"], "input": "1 nyeon"}
    response = {"candidates": [
        {"text": str(index), "score": float(10 - index)} for index in range(5)
    ]}
    pieces = [("base", 0, (Option("base", 1., "syllable"),
                            Option("other", 0., "syllable")))]
    recorder = Recorder()
    recorder.candidates = ["0", "1", "2", "3", "4", "numeric-extra"]
    recorder.scores = [6., 5., 4., 3., 2., 1.]
    learned = [("base", 1., [])]
    reconstructed = [*learned, ("numeric-extra", 0., [])]
    reason, pool, _, displaced, _, lattice = module.stage_reason(
        row, response, recorder, pieces, pieces, learned, reconstructed, 1, True,
    )
    assert "numeric-extra" in pool
    assert displaced is False
    assert lattice == "missing_fullpiece_candidate"
    assert reason == "sentence_ranking_loss"


def test_no_observed_sentence_pool_uses_learned_paths_not_static_union():
    row = {"expected": ["static-only"], "input": "a b"}
    response = {"candidates": [{"text": "learned", "score": 1.}]}
    learned_pieces = [("a b", 0, (Option("learned", 1., "syllable"),))]
    static_pieces = [("a b", 0, (Option("static-only", 1., "syllable"),))]
    recorder = Recorder()
    learned = [("learned", 1., [])]
    reconstructed = [*learned, ("static-only", 1., [])]
    reason, pool, ranked, _, _, _ = module.stage_reason(
        row, response, recorder, learned_pieces, static_pieces,
        learned, reconstructed, 2, True,
    )
    assert pool == ranked == ["learned"]
    assert reason == "deterministic_path_present"


def test_sentence_recorder_rejects_repeated_and_nonfinite_calls():
    class Delegate(Ranker):
        def __init__(self, scores):
            self.scores = scores

        def score_pairs(self, contexts, candidates):
            return self.scores

    recorder = module.SentenceRecorder(Delegate([float("nan")]))
    try:
        recorder.score_pairs([""], ["a"])
    except RuntimeError as error:
        assert "malformed" in str(error)
    else:
        raise AssertionError("non-finite score accepted")
    recorder = module.SentenceRecorder(Delegate([1.]))
    assert recorder.score_pairs([""], ["a"]) == [1.]
    try:
        recorder.score_pairs([""], ["a"])
    except RuntimeError as error:
        assert "more than once" in str(error)
    else:
        raise AssertionError("repeated sentence scoring accepted")
