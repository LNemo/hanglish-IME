"""Pure mutation tests for the independent pairwise resource verifier."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import struct

import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts/verify_difference_pairwise_lora_resources.py"
SPEC = importlib.util.spec_from_file_location("pairwise_resource_verifier", PATH)
v = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v)


def token_rows():
    rows = []
    for index in range(1928):
        tokens = [index % 31] * (1 + index % 17)
        rows.append({"pair_id": f"p-{index:04d}", "tokens": tokens,
                     "prompt_tokens": len(tokens), "tokens_sha256": v.digest_value(tokens),
                     "digit_continuation_ids": {"1": 16, "2": 17}})
    return rows


def test_representatives_and_schedule_are_reconstructed_from_real_lengths():
    rows = token_rows()
    representatives, schedule = v.expected_representatives_and_schedule(rows)
    ordered = sorted(rows, key=lambda row: (row["prompt_tokens"], row["pair_id"]))
    assert [row["pair_id"] for row in representatives] == [
        ordered[0]["pair_id"], ordered[(len(rows)-1)//2]["pair_id"],
        ordered[v.math.ceil(.95*len(rows))-1]["pair_id"], ordered[-1]["pair_id"]]
    assert len(schedule) == 20
    assert [len(row["components"]) for row in schedule] == [4, 4, 4, 4, 3] * 4
    assert [row["update"] for row in schedule] == [1]*4+[2]*4+[3]*4+[4]*4+[5]*4
    assert all(row["components"][0]["target"] == "SWITCH" for row in schedule)
    assert all(abs(sum(item["weight"] for item in row["components"])-1) < 1e-15
               for row in schedule)


def test_token_digest_and_digit_boundary_mutations_are_rejected():
    rows = token_rows()
    rows[0]["tokens_sha256"] = "0" * 64
    with pytest.raises(v.VerificationError, match="token row"):
        v.expected_representatives_and_schedule(rows)
    rows = token_rows()
    rows[0]["digit_continuation_ids"]["1"] = 99
    with pytest.raises(v.VerificationError, match="token row"):
        v.expected_representatives_and_schedule(rows)


def observation(stage, elapsed=1.0):
    metrics = {"rss_peak_bytes": 100, "mlx_active_bytes": 50, "mlx_cache_bytes": 10,
               "mlx_peak_bytes": 100, "free_percent": 50, "swap_used_bytes": 20,
               "swap_growth_bytes": 0, "elapsed_seconds": elapsed}
    return {"stage": stage, "action": "none", "pre": metrics, "post": None,
            "errors": [], "cleanup_seconds": 0.0, "checkpoint_seconds": .1}


def test_resource_observations_require_exact_stage_order_and_recompute_limits():
    records = [observation("a"), observation("b")]
    summary, feasible = v.verify_observations(
        records, ["a", "b"], {"free_percent": 50, "swap_used_bytes": 20})
    assert feasible and summary["rss_peak_bytes"] == 100
    records[1]["stage"] = "a"
    with pytest.raises(v.VerificationError, match="stage sequence"):
        v.verify_observations(records, ["a", "b"],
                              {"free_percent": 50, "swap_used_bytes": 20})


@pytest.mark.parametrize("field,value", [
    ("rss_peak_bytes", v.LIMITS["rss_bytes"] + 1),
    ("mlx_peak_bytes", v.LIMITS["mlx_peak_bytes"] + 1),
    ("mlx_cache_bytes", v.LIMITS["cache_bytes"] + 1),
    ("free_percent", v.LIMITS["minimum_system_free_percent"] - 1),
    ("swap_growth_bytes", v.LIMITS["maximum_swap_growth_bytes"] + 1),
])
def test_each_resource_bound_is_recomputed(field, value):
    row = observation("only")
    row["pre"][field] = value
    if field == "swap_growth_bytes":
        row["pre"]["swap_used_bytes"] = 20 + value
    _, feasible = v.verify_observations(
        [row], ["only"], {"free_percent": 50, "swap_used_bytes": 20})
    assert feasible is False


def test_logit_checks_enforce_argmax_and_tolerances():
    representatives = [{"pair_id": f"p-{i}"} for i in range(4)]
    rows = [{"pair_id": row["pair_id"], "selected_logits": [1.0, 0.0],
             "ordinary_logits": [.9, 0.0]} for row in representatives]
    v.verify_logits(rows, representatives, reload=False)
    rows[0]["ordinary_logits"] = [0.0, 1.0]
    with pytest.raises(v.VerificationError, match="argmax"):
        v.verify_logits(rows, representatives, reload=False)


def test_reload_check_uses_absolute_plus_relative_tolerance():
    representatives = [{"pair_id": f"p-{i}"} for i in range(4)]
    rows = [{"pair_id": row["pair_id"], "actual_logits": [10.0001, 0.0],
             "expected_logits": [10.0, 0.0]} for row in representatives]
    v.verify_logits(rows, representatives, reload=True)
    rows[0]["actual_logits"][0] = 10.001
    with pytest.raises(v.VerificationError, match="tolerance"):
        v.verify_logits(rows, representatives, reload=True)


def measured_report(schedule, representatives):
    units = []
    for unit in schedule:
        components = [{"component_index": index, "target": item["target"],
                       "weight": item["weight"], "loss": 1.0,
                       "gradient_seconds": .1 + index/100}
                      for index, item in enumerate(unit["components"])]
        units.append({"unit": unit["unit"], "update": unit["update"], "loss": 1.0,
                      "components": components, "unit_gradient_seconds": .5})
    updates = [{"update": index, "unit_ids": list(range(4*index-3, 4*index+1)),
                "gradient_norm_before_clip": 1.0, "optimizer_seconds": .2,
                "complete_update_seconds": 3.0} for index in range(1, 6)]
    inference = [{"pair_id": shape["pair_id"], "repetition": repetition,
                  "seconds": .05, "logits": [1.0, 0.0]}
                 for shape in representatives for repetition in range(1, 11)]
    timing = {"component_gradient_p95": .13, "unit_gradient_p95": .5,
              "optimizer_update_p95": .2, "pair_inference_p95": .05,
              "save_reload_finalization_seconds": 2.0}
    return {"units": units, "updates": updates, "inference": inference,
            "finalization_interval": {"start_elapsed": 5.0, "end_elapsed": 7.0},
            "timing_summary": timing}


def test_measurements_recompute_counts_weights_losses_and_p95():
    representatives, schedule = v.expected_representatives_and_schedule(token_rows())
    report = measured_report(schedule, representatives)
    timing, complete = v.verify_measurements(report, schedule, representatives)
    assert timing == report["timing_summary"] and complete == [3.0] * 5
    report["units"][0]["components"][0]["weight"] = .4
    with pytest.raises(v.VerificationError, match="component evidence"):
        v.verify_measurements(report, schedule, representatives)


def test_complete_update_limit_is_not_replaced_by_optimizer_time():
    representatives, schedule = v.expected_representatives_and_schedule(token_rows())
    report = measured_report(schedule, representatives)
    report["updates"][0]["complete_update_seconds"] = 121.0
    report["updates"][0]["optimizer_seconds"] = .001
    with pytest.raises(v.VerificationError, match="optimizer update evidence"):
        v.verify_measurements(report, schedule, representatives)


def test_hierarchical_durations_cannot_be_shorter_than_recorded_children():
    representatives, schedule = v.expected_representatives_and_schedule(token_rows())
    report = measured_report(schedule, representatives)
    report["units"][0]["unit_gradient_seconds"] = .01
    with pytest.raises(v.VerificationError, match="shorter than its component"):
        v.verify_measurements(report, schedule, representatives)
    report = measured_report(schedule, representatives)
    report["updates"][0]["complete_update_seconds"] = 1.0
    with pytest.raises(v.VerificationError, match="shorter than unit"):
        v.verify_measurements(report, schedule, representatives)


def test_numerics_counter_shape_is_exact():
    exact = {"component_losses_finite": 76, "component_gradients_finite": 76,
             "unit_losses_finite": 20, "unit_gradients_finite": 20,
             "optimizer_states_finite": 5}
    v.verify_numerics(exact)
    changed = deepcopy(exact)
    changed["component_gradients_finite"] = 75
    with pytest.raises(v.VerificationError, match="numerical-check accounting"):
        v.verify_numerics(changed)


def write_safetensors(path):
    header, chunks, offset = {}, [], 0
    for index, name in enumerate(v.EXPECTED_NAMES):
        shape = v.EXPECTED_SHAPES[name]
        chunk = bytes([index % 251]) * (v.math.prod(shape) * 4)
        header[name] = {"dtype": "F32", "shape": shape,
                        "data_offsets": [offset, offset + len(chunk)]}
        chunks.append(chunk); offset += len(chunk)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(chunks))
    return v.read_safetensors(path)


def test_safetensors_parser_reconstructs_names_shapes_scalars_and_hashes(tmp_path):
    path = tmp_path / "adapter.safetensors"
    hashes = write_safetensors(path)
    assert sorted(hashes) == v.EXPECTED_NAMES and len(hashes) == 16


def test_safetensors_parser_rejects_malformed_header_and_shape(tmp_path):
    malformed = tmp_path / "malformed.safetensors"
    malformed.write_bytes(struct.pack("<Q", 99) + b"{}")
    with pytest.raises(v.VerificationError, match="header"):
        v.read_safetensors(malformed)
    valid = tmp_path / "shape.safetensors"
    write_safetensors(valid)
    raw = valid.read_bytes(); size = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8+size]); header[v.EXPECTED_NAMES[0]]["shape"] = [1]
    encoded = json.dumps(header, separators=(",", ":")).encode()
    valid.write_bytes(struct.pack("<Q", len(encoded)) + encoded + raw[8+size:])
    with pytest.raises(v.VerificationError, match="metadata"):
        v.read_safetensors(valid)


def test_safetensors_tensor_byte_mutation_changes_reconstructed_hash(tmp_path):
    path = tmp_path / "adapter.safetensors"
    expected = write_safetensors(path)
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(raw)
    actual = v.read_safetensors(path)
    assert actual != expected


def test_missing_resource_report_exits_invalid_without_proof(tmp_path):
    output = tmp_path / "proof.json"
    assert v.main(["--resource", str(tmp_path / "missing"), "--output", str(output)]) == 1
    assert not output.exists()


def test_proof_envelope_is_canonical_and_immutable(tmp_path):
    output = tmp_path / "proof.json"
    payload = {"stage": "resource", "passed": True}
    envelope = v.write_proof(output, payload)
    assert envelope["digest"] == v.digest_value(payload)
    with pytest.raises(v.VerificationError, match="already exists"):
        v.write_proof(output, payload)
