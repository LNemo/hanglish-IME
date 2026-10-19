from copy import deepcopy
import math

import pytest

from scripts import probe_consensus_context_resources as probe


def candidate(text):
    return {"text": text, "score": None, "segments": [{"source": "x", "target": text}]}


def aligned_candidate(source, text):
    return {"text": text, "score": 1.0, "segments": [{
        "input_start": 0, "input_end": len(source), "source": source,
        "text": text, "kind": "neural"}]}


def raw_response(source):
    return {"result": source, "version": "v", "candidates": [
        {"text": source, "score": None, "segments": [{
            "input_start": 0, "input_end": len(source), "source": source,
            "text": source, "kind": "original"}]}]}


def test_prompt_record_uses_historical_mapping_and_no_labels():
    row = {"id": "n1", "input": "gachi", "context": "uri",
           "candidates": [candidate("같이"), candidate("가치"), candidate("gachi")],
           "raw_bypass_response": raw_response("gachi"),
           "expected": object()}
    record = probe.prompt_record({"role": "natural_fit", "row": row},
                                 {"role": "system", "content": "choose"})
    assert sorted(record["choices"]) == [(0, "같이"), (1, "가치")]
    assert "expected" not in record
    assert record["messages"][1]["content"].startswith("로마자 입력: gachi\n문맥: uri")


def test_finite_distribution_rejects_bad_logits_and_has_stable_tie():
    probabilities, selected = probe.finite_distribution([2.0, 2.0, -1.0])
    assert selected == 0
    assert sum(probabilities) == pytest.approx(1.0)
    for values in ([], [math.nan], [math.inf], [1, 2, 3, 4, 5, 6]):
        with pytest.raises(ValueError):
            probe.finite_distribution(values)


def test_preserve_decision_keeps_objects_and_raw_bypasses_predictions():
    candidates = [aligned_candidate("gachi", "가치"), aligned_candidate("gachi", "같이"),
                  aligned_candidate("gachi", "gachi")]
    actual_raw = raw_response("gachi")
    record = {"source": "gachi", "candidates": candidates,
              "raw_bypass_response": actual_raw}
    before = deepcopy(candidates)
    selected, raw = probe.preserve_decision(record, 1, 1)
    assert selected["chosen_index"] == 1
    assert selected["candidates"][0] is candidates[1]
    assert selected["candidates"][1] is candidates[0]
    assert selected["candidates"][2] is candidates[2]
    assert candidates == before
    assert raw["decision"] == {"chosen_index": 0, "result": "gachi",
                               "candidates": actual_raw["candidates"],
                               "reason": "raw_mode"}
    assert raw["response"] == actual_raw
    assert raw["decision"]["candidates"] is actual_raw["candidates"]


def test_preserve_decision_rejects_altered_captured_raw_response():
    normal = [aligned_candidate("gachi", "가치"), aligned_candidate("gachi", "같이")]
    for bad_raw in (
        {**raw_response("gachi"), "result": "같이"},
        {**raw_response("gachi"), "candidates": [aligned_candidate("gachi", "같이")]},
    ):
        with pytest.raises(RuntimeError, match="raw bypass"):
            probe.preserve_decision(
                {"source": "gachi", "candidates": normal,
                 "raw_bypass_response": bad_raw}, 0, 0)


class FakeBackend:
    def __init__(self, *, long_first=False):
        self.loads = []
        self.clears = 0
        self.long_first = long_first

    def load_model(self, adapter=False):
        self.loads.append(adapter)
        return {"adapter": adapter}, object()

    def digit_ids(self, tokenizer):
        return [10, 11, 12, 13, 14]

    def tokens(self, tokenizer, messages):
        source = messages[1]["content"].splitlines()[0]
        if self.long_first and source.endswith("s0"):
            return list(range(257))
        return [1, len(source), 2]

    def infer(self, model, tokens, digit_ids, count):
        return [float(index) for index in range(count)]

    def peak(self):
        return 1024

    def clear(self):
        self.clears += 1


def records(count=5):
    return [{"id": f"r{i}", "role": "natural_fit", "source": f"s{i}",
             "context": "ctx", "candidates": [candidate("가"), candidate("나")],
             "choices": [(0, "가"), (1, "나")],
             "messages": [{"role": "system", "content": "x"},
                          {"role": "user", "content": f"로마자 입력: s{i}"}]}
            for i in range(count)]


def test_infer_stage_skips_oversize_without_truncation_and_releases():
    backend = FakeBackend(long_first=True)
    outcome = probe.infer_stage(backend, records(), adapter=False,
                                started=probe.time.monotonic())
    assert len(outcome) == 4
    assert [row["record_index"] for row in outcome] == [1, 2, 3, 4]
    assert all(len(row["tokens"]) == 3 for row in outcome)
    assert backend.loads == [False]
    assert backend.clears == 1


def test_infer_stage_requires_four_and_releases_on_failure():
    backend = FakeBackend()
    with pytest.raises(RuntimeError, match="fewer than four"):
        probe.infer_stage(backend, records(3), adapter=True,
                          started=probe.time.monotonic())
    assert backend.clears == 1


def test_infer_stage_checks_resources_immediately_after_load():
    backend = FakeBackend()
    backend.peak = lambda: probe.PROFILE["mlx_peak_limit_bytes"] + 1
    backend.infer = lambda *_: pytest.fail("inference ran after resource overflow")
    with pytest.raises(RuntimeError, match="MLX"):
        probe.infer_stage(backend, records(), adapter=False,
                          started=probe.time.monotonic())
    assert backend.clears == 1


def test_resource_guard_checks_time_rss_and_mlx(monkeypatch):
    monkeypatch.setattr(probe, "rss_bytes", lambda: 0)
    with pytest.raises(RuntimeError, match="time"):
        probe.require_resources(probe.time.monotonic() - 181)
    monkeypatch.setattr(probe, "rss_bytes", lambda: probe.PROFILE["rss_limit_bytes"] + 1)
    with pytest.raises(RuntimeError, match="RSS"):
        probe.require_resources(probe.time.monotonic())
    monkeypatch.setattr(probe, "rss_bytes", lambda: 0)
    with pytest.raises(RuntimeError, match="MLX"):
        probe.require_resources(probe.time.monotonic(),
                                probe.PROFILE["mlx_peak_limit_bytes"] + 1)


def test_lineage_overlap_shape_is_coherent_but_overlap_is_not_a_gate(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    records = []
    for index in range(128):
        role = "natural_fit" if index < 64 else "challenge_fit"
        records.append({"role": role, "row": {"id": f"r{index}",
                                                "group_id": f"g{index}"}})
    cases = b"".join((probe.json.dumps(record) + "\n").encode() for record in records)
    (data / "cases.jsonl").write_bytes(cases)
    manifest = {
        "schema": "consensus-context-train-pilot-data-v1", "completed": True,
        "rows": 128, "cases_sha256": probe.digest(cases),
        "runtime_version": probe.RUNTIME_VERSION, "plan_sha256": probe.PLAN_SHA256,
        "source_hashes": {"x": "a" * 64}, "source_hashes_after": {"x": "a" * 64},
        "shape_counts": {"natural": 64, "challenge_single": 32,
                         "challenge_sentence": 32},
        "selection": {
            "natural": {"rows": 64, "ids": [f"r{i}" for i in range(64)],
                        "group_ids": [f"g{i}" for i in range(64)]},
            "challenge_single": {"rows": 32, "ids": [f"r{i}" for i in range(64, 96)],
                                 "group_ids": [f"g{i}" for i in range(64, 96)]},
            "challenge_sentence": {"rows": 32, "ids": [f"r{i}" for i in range(96, 128)],
                                   "group_ids": [f"g{i}" for i in range(96, 128)]},
        },
        "mlx_model_inference_run": False, "development_data_read": False,
        "final_data_read": False, "deployed": False,
        "selection_uses_labels_categories_eligibility_or_outcomes": False,
    }
    manifest_raw = (probe.json.dumps(manifest) + "\n").encode()
    (data / "manifest.json").write_bytes(manifest_raw)
    lineage = {
        "completed": True, "pilot_manifest_sha256": probe.digest(manifest_raw),
        "source_hashes": {"x": "a" * 64}, "source_hashes_after": {"x": "a" * 64},
        "family_group_independence_established": False,
        "standalone_development_corpus_read": False,
        "historical_validation_records_read": 8,
        "final_data_read": False,
        "optimizer_overlap": {"historical_rows": 58, "id_overlap_count": 1,
                              "id_overlap": [{"role": "natural_fit", "id": "r0"}],
                              "exact_input_context_overlap_count": 1,
                              "exact_input_context_overlap": [
                                  {"role": "natural_fit", "id": "r1"}],
                              "historical_id_set_sha256": "b" * 64},
        "validation_overlap": {"historical_rows": 8, "id_overlap_count": 0,
                               "id_overlap": [], "exact_input_context_overlap_count": 0,
                               "exact_input_context_overlap": [],
                               "historical_id_set_sha256": "c" * 64},
    }
    lineage_path = tmp_path / "lineage.json"
    lineage_raw = (probe.json.dumps(lineage) + "\n").encode()
    lineage_path.write_bytes(lineage_raw)
    loaded = probe.load_inputs(data, probe.digest(manifest_raw),
                               lineage_path, probe.digest(lineage_raw))
    assert loaded["lineage"]["optimizer_overlap"]["id_overlap_count"] == 1
    lineage["optimizer_overlap"]["id_overlap_count"] = 2
    bad = (probe.json.dumps(lineage) + "\n").encode()
    lineage_path.write_bytes(bad)
    with pytest.raises(ValueError, match="overlap record"):
        probe.load_inputs(data, probe.digest(manifest_raw), lineage_path, probe.digest(bad))


def test_lineage_overlap_rejects_nonmember_and_duplicate_dict_identities():
    pilot = {("natural_fit", "r0")}
    assert (probe.normalize_overlap_identities(
        [{"role": "natural_fit", "id": "r0"}], pilot
    ) == [("natural_fit", "r0")])
    for entries, message in (
        (["r0"], "malformed"),
        ([{"role": "natural_fit", "id": "missing"}], "outside pilot"),
        ([{"role": "natural_fit", "id": "r0"}] * 2, "duplicated"),
    ):
        with pytest.raises(ValueError, match=message):
            probe.normalize_overlap_identities(entries, pilot)


def test_verify_hashes_streams_without_path_read_bytes(tmp_path, monkeypatch):
    path = tmp_path / "large.bin"
    path.write_bytes(b"abc" * 1000)
    expected = probe.digest(path.read_bytes())
    original = probe.Path.read_bytes

    def guarded(self):
        if self == path:
            raise AssertionError("large artifact must be hashed as a stream")
        return original(self)

    monkeypatch.setattr(probe.Path, "read_bytes", guarded)
    probe.verify_hashes({str(path): expected})
