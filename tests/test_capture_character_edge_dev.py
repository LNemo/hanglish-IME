from copy import deepcopy
import json

import pytest

from scripts import capture_character_edge_dev as capture


def segment(source, text=None):
    return [{"input_start": 0, "input_end": len(source), "source": source,
             "text": source if text is None else text, "kind": "fake"}]


def public(source, text, score):
    return [{"text": text, "score": score, "segments": segment(source, text)},
            {"text": source, "score": None, "segments": segment(source)}]


def source_row(source="ab", identifier="dev-1", ranked=True):
    paths = [["갑", 9.25, segment(source, "갑")]] if ranked else []
    union = capture.raw_paths_public(source, paths) if ranked else public(source, "fallback", 1.0)
    return {"id": identifier, "request": {"input": source, "context": ""},
            "parent": {}, "union_reference": {"candidates": union},
            "proposed": {"sentence_ranking_applied": ranked,
                         "candidates": deepcopy(union),
                         "phonetic_choice": {"rows": [], "score_records": []}},
            "base_paths": paths}


class FakeSentence:
    def can_score(self, text):
        return text != "unsupported"

    def score_pairs(self, contexts, candidates):
        return [40.0 + index for index, _ in enumerate(candidates)]


class FakeCharacter:
    def __init__(self, adjustment=100.0):
        self.adjustment = adjustment
        self.calls = []

    def prepare(self, source, candidates, base_scores):
        self.calls.append((source, deepcopy(candidates), list(base_scores)))
        return list(base_scores)

    def __call__(self, prepared):
        return {"scores": [value + self.adjustment for value in prepared]}


CALIBRATION = {"parameters": [1.0, 0.0, 0.0, 0.0, 0.0],
               "scale": [1.0, 1.0, 1.0, 1.0]}


def scored_row(source="ab", *, ranked=True, unsupported=False):
    original = [{"text": "갑", "sentence_score": 9.25,
                 "features": [1.0, 2.0, 3.0, 4.0],
                 "segments": segment(source, "갑"), "status": "preserved",
                 "score_source": "captured_original"}] if ranked else []
    novel = []
    if ranked:
        novel.append({"text": "unsupported" if unsupported else "을",
                      "sentence_score": None if unsupported else 3.5,
                      "features": [4.0, 3.0, 2.0, 1.0],
                      "segments": segment(source, "unsupported" if unsupported else "을"),
                      "status": "unsupported" if unsupported else "scored",
                      "reason": "sentence_ranker_can_score_false" if unsupported else None})
    fallback = public(source, "fallback", 1.0) if not ranked else None
    return {"schema": "current-recombination-scoring-row-v1", "id": "dev-1",
            "component_id": None, "request": {"input": source, "context": ""},
            "ranked": ranked, "source_reference": {}, "original": original,
            "novel": novel, "fallback_public": fallback,
            "score_texts": [], "original_score_parity": None}


def patch_pipeline(monkeypatch, scored):
    observed = {}

    def fake_census(original, expected, source_shard, source_row_index, capture_sha,
                    **kwargs):
        observed.update(original=deepcopy(original), expected=deepcopy(expected),
                        source_shard=source_shard, source_row_index=source_row_index,
                        capture_sha=capture_sha)
        return {"schema": "current-fit-recombination-census-row-v1",
                "id": original["id"], "status": "generated" if scored["ranked"] else "no_pool"}

    monkeypatch.setattr(capture, "prepare_record", fake_census)
    monkeypatch.setattr(capture, "prepare_row", lambda original, census, ranker: {"prepared": True})
    monkeypatch.setattr(capture, "score_prepared",
                        lambda prepared, ranker, verify_original_parity: deepcopy(scored))
    return observed


def test_adapter_seeds_raw_union_public_and_preserves_actual_identity():
    row = source_row(identifier="actual-dev-id")
    adapted = capture.adapt_source_row(row)
    assert adapted["id"] == "actual-dev-id"
    assert adapted["component_id"] is None
    assert adapted["response"]["candidates"] == row["union_reference"]["candidates"]
    assert adapted["base_paths"][0][1] == 9.25
    altered = deepcopy(row)
    altered["union_reference"]["candidates"][0]["score"] = -99.0
    with pytest.raises(ValueError, match="raw full-union public seed"):
        capture.adapt_source_row(altered)


def test_capture_preserves_original_scores_and_marks_dev_legacy_namespace(monkeypatch):
    scored = scored_row()
    observed = patch_pipeline(monkeypatch, scored)
    model = FakeCharacter()
    comparator = {"id": "dev-1", "request": {"input": "ab", "context": ""},
                  "response": {"candidates": public("ab", "production", 7.0)}}
    row = capture.capture_row(source_row(), comparator, 11, FakeSentence(), model, CALIBRATION)
    assert row["dataset_split"] == "dev"
    assert row["source_row_index"] == 11
    assert observed["expected"] == {"id": "dev-1", "input": "ab", "component_id": None}
    assert observed["capture_sha"] == capture.SOURCE_CAPTURE_SHA
    assert observed["original"]["response"]["candidates"] == source_row()["union_reference"]["candidates"]
    assert row["scoring"]["original"][0]["sentence_score"] == 9.25
    assert row["predictions"]["current_production"] == comparator["response"]["candidates"]


def test_no_pool_uses_source_fallback_for_four_derived_arms(monkeypatch):
    scored = scored_row(ranked=False)
    patch_pipeline(monkeypatch, scored)
    comparator = {"id": "dev-1", "request": {"input": "ab", "context": ""},
                  "response": {"candidates": public("ab", "production", 7.0)}}
    row = capture.capture_row(source_row(ranked=False), comparator, 2,
                              FakeSentence(), FakeCharacter(), CALIBRATION)
    fallback = scored["fallback_public"]
    assert row["predictions"]["current_production"] != fallback
    assert all(row["predictions"][arm] == fallback for arm in capture.ARMS[1:])
    assert row["character_scores"] == {"old_character": [], "expanded_character": []}


def test_unsupported_novel_is_retained_but_excluded_from_expanded_pools(monkeypatch):
    scored = scored_row(unsupported=True)
    patch_pipeline(monkeypatch, scored)
    model = FakeCharacter(0.0)
    comparator = {"id": "dev-1", "request": {"input": "ab", "context": ""},
                  "response": {"candidates": public("ab", "production", 7.0)}}
    row = capture.capture_row(source_row(), comparator, 0, FakeSentence(), model, CALIBRATION)
    assert row["scoring"]["novel"][0]["status"] == "unsupported"
    assert len(model.calls) == 2
    assert len(model.calls[0][1]) == len(model.calls[1][1]) == 1
    assert row["character_scores"]["old_character"] == row["character_scores"]["expanded_character"]


def test_all_five_public_arms_and_character_score_alignment(monkeypatch):
    scored = scored_row()
    patch_pipeline(monkeypatch, scored)
    model = FakeCharacter(100.0)
    comparator = {"id": "dev-1", "request": {"input": "ab", "context": ""},
                  "response": {"candidates": public("ab", "production", 7.0)}}
    row = capture.capture_row(source_row(), comparator, 0, FakeSentence(), model, CALIBRATION)
    assert tuple(row["predictions"]) == capture.ARMS
    assert row["character_scores"]["old_character"] == [109.25]
    assert row["character_scores"]["expanded_character"] == [109.25, 103.5]
    assert [item["text"] for item in row["predictions"]["expanded_character"]] == ["갑", "을", "ab"]
    assert row["predictions"]["old_calibrated"][0]["score"] == 9.25
    assert row["predictions"]["expanded_calibrated"][1]["score"] == 3.5


def test_offset_index_and_independent_probe_authorization(tmp_path):
    indexed_path = tmp_path / "indexed.jsonl"
    indexed_path.write_text(
        json.dumps({"id": "one", "request": {"input": "abc", "context": ""}}) + "\n"
        + json.dumps({"id": "two", "request": {"input": "def", "context": ""}}) + "\n",
        encoding="utf-8")
    index = capture.index_jsonl(indexed_path)
    assert [(row["id"], row["input"]) for row in index] == [("one", "abc"), ("two", "def")]
    assert index[0]["offset"] == 0 < index[1]["offset"]
    assert capture.read_indexed(indexed_path, index[1])["id"] == "two"

    pins = {"/frozen/source": "a" * 64}
    capture_path = tmp_path / "capture.jsonl"
    capture_path.write_text("{}\n", encoding="utf-8")
    capture_sha = capture.digest(capture_path)
    closure = {"capture_sha256": capture_sha}
    capture.write_json(tmp_path / "capture.complete.json", closure)
    report = {"completed": True, "mode": "probe", "rows": 16,
              "capture_sha256": capture_sha, "labels_opened": False,
              "pins": pins, "qualification": {"qualified": True},
              "hashes": {"capture.complete.json": capture.digest(
                  tmp_path / "capture.complete.json")}}
    capture.write_json(tmp_path / "report.json", report)
    proof = {"verified": True, "mode": "probe", "rows": 16,
             "report_sha256": capture.digest(tmp_path / "report.json"),
             "capture_sha256": capture_sha, "pins": pins,
             "labels_opened": False, "source_only": True}
    capture.write_json(tmp_path / "independent-verification.json", proof)
    authorized = capture._probe_authorized(tmp_path, pins)
    assert set(authorized) == {str((tmp_path / name).resolve()) for name in (
        "report.json", "capture.complete.json", "independent-verification.json")}
