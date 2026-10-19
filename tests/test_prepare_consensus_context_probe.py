from copy import deepcopy
import hashlib
from types import SimpleNamespace

import pytest

import scripts.prepare_consensus_context_probe as module


def row(identity, source, context, group="g", **labels):
    return {"id": identity, "group_id": group, "input": source,
            "context": context, **labels}


def rows_for_selection():
    natural = [row(f"n{i:03}", "sentence input", "context", f"ng{i}",
                   expected=[f"natural-{i}"], selector_eligible=bool(i % 2))
               for i in range(70)]
    challenge = [
        *[row(f"w{i:03}", "word", "meaning", f"wg{i}", expected=["word"])
          for i in range(40)],
        *[row(f"s{i:03}", "two words", "", f"sg{i}", expected=["sentence"])
          for i in range(40)],
        row("excluded-upper", "Word", "meaning"),
        row("excluded-empty", "word", ""),
    ]
    return natural, challenge


def test_selection_is_deterministic_bounded_and_label_independent():
    natural, challenge = rows_for_selection()
    first = module.select_pilot(natural, challenge)
    changed_natural, changed_challenge = deepcopy(natural), deepcopy(challenge)
    for item in [*changed_natural, *changed_challenge]:
        item["expected"] = ["changed label"]
        item["category"] = "changed"
        item["selector_eligible"] = not item.get("selector_eligible", False)
    second = module.select_pilot(changed_natural, changed_challenge)
    assert [(role, item["id"]) for role, item in first] == [
        (role, item["id"]) for role, item in second]
    assert [role for role, _ in first] == [
        *(["natural_fit"] * 64), *(["challenge_fit"] * 64)]
    assert all(item["input"] == "word" and item["context"]
               for _, item in first[64:96])
    assert all(" " in item["input"] for _, item in first[96:])
    expected_natural = sorted(
        natural, key=lambda item: module.selection_key("natural_fit", item["id"])
    )[:64]
    assert [item["id"] for _, item in first[:64]] == [
        item["id"] for item in expected_natural]


def test_selection_preserves_original_rows_and_rejects_short_shapes():
    natural, challenge = rows_for_selection()
    before = deepcopy((natural, challenge))
    selected = module.select_pilot(natural, challenge)
    assert (natural, challenge) == before
    assert selected[0][1] is natural[
        next(i for i, item in enumerate(natural)
             if item["id"] == selected[0][1]["id"])]
    with pytest.raises(ValueError, match="fixed pilot shapes"):
        module.select_pilot(natural[:63], challenge)
    with pytest.raises(ValueError, match="fixed pilot shapes"):
        module.select_pilot(natural, challenge[:31])


def fit_manifest(role, runtime):
    return {"role": role, "partition_manifest_sha256": "p",
            "runtime_version": runtime["integrated_runtime_version"],
            "runtime_source_hashes": runtime["integrated_runtime_source_hashes"],
            "model_artifact_hashes": runtime["accepted_weight_hashes"],
            "source_hashes": {"same": "a" * 64}, "train_inputs": {"x": 1},
            "partition_ordered_ids": [role]}


def test_runtime_validation_rejects_version_or_weight_drift(monkeypatch):
    runtime = {"passed": True, "integrated_runtime_version": "v",
               "integrated_runtime_source_hashes": {"source": "a" * 64},
               "runtime_source_hashes_after": {"source": "a" * 64},
               "accepted_weight_hashes": {f"w{i}": "b" * 64 for i in range(16)}}
    natural = {"manifest": fit_manifest("natural_fit", runtime)}
    challenge = {"manifest": fit_manifest("challenge_fit", runtime)}
    monkeypatch.setattr(module, "validate_fit_pair", lambda *_: None)
    module.validate_runtime(natural, challenge, runtime)
    challenge["manifest"]["runtime_version"] = "old"
    with pytest.raises(ValueError, match="current runtime"):
        module.validate_runtime(natural, challenge, runtime)
    challenge["manifest"]["runtime_version"] = "v"
    challenge["manifest"]["model_artifact_hashes"] = {"changed": "c" * 64}
    with pytest.raises(ValueError, match="current runtime"):
        module.validate_runtime(natural, challenge, runtime)


def test_run_rejects_plan_before_loading_fit_or_runtime(tmp_path, monkeypatch):
    plan = tmp_path / "plan.md"
    plan.write_text("plan")
    touched = []
    monkeypatch.setattr(module, "read_runtime_evidence",
                        lambda *_: touched.append("runtime"))
    monkeypatch.setattr(module, "load_fit_role",
                        lambda *_: touched.append("fit"))
    args = SimpleNamespace(plan=plan, plan_sha256="0" * 64,
                           runtime_evidence=tmp_path / "runtime.json",
                           output=tmp_path / "output")
    with pytest.raises(ValueError, match="plan hash"):
        module.run(args)
    assert touched == []


def test_run_rejects_rehashed_unfrozen_plan_before_inputs(tmp_path, monkeypatch):
    plan = tmp_path / "plan.md"
    plan.write_text("different but internally matching plan")
    supplied = hashlib.sha256(plan.read_bytes()).hexdigest()
    touched = []
    monkeypatch.setattr(module, "read_runtime_evidence",
                        lambda *_: touched.append("runtime"))
    monkeypatch.setattr(module, "load_fit_role", lambda *_: touched.append("fit"))
    args = SimpleNamespace(plan=plan, plan_sha256=supplied,
                           runtime_evidence=tmp_path / "runtime.json",
                           output=tmp_path / "output")
    with pytest.raises(ValueError, match="frozen consensus"):
        module.run(args)
    assert touched == []


def test_fit_case_mutation_fails_frozen_provenance(tmp_path, monkeypatch):
    natural_dir, challenge_dir = tmp_path / "natural", tmp_path / "challenge"
    natural_dir.mkdir()
    challenge_dir.mkdir()
    for directory in (natural_dir, challenge_dir):
        (directory / "cases.jsonl").write_text("case\n")
        (directory / "selection.jsonl").write_text("selection\n")
    monkeypatch.setattr(module, "NATURAL_FIT", natural_dir)
    monkeypatch.setattr(module, "CHALLENGE_FIT", challenge_dir)
    monkeypatch.setattr(module, "NATURAL_MANIFEST_SHA256", "1" * 64)
    monkeypatch.setattr(module, "CHALLENGE_MANIFEST_SHA256", "2" * 64)
    manifest = lambda role: {
        "source_hashes": {},
        "cases_sha256": hashlib.sha256(b"case\n").hexdigest(),
        "selection_sha256": hashlib.sha256(b"selection\n").hexdigest(),
    }
    natural = {"manifest": manifest("natural")}
    challenge = {"manifest": manifest("challenge")}
    plan, runtime = tmp_path / "plan", tmp_path / "runtime"
    plan.write_text("p")
    runtime.write_text("r")
    hashes = module.provenance(
        natural, challenge,
        {"integrated_runtime_source_hashes": {}, "accepted_weight_hashes": {}},
        plan_path=plan, plan_sha=hashlib.sha256(b"p").hexdigest(),
        runtime_path=runtime, runtime_sha=hashlib.sha256(b"r").hexdigest(),
        runner_sha=hashlib.sha256(
            (module.ROOT / "scripts/prepare_consensus_context_probe.py").read_bytes()
        ).hexdigest())
    (natural_dir / "cases.jsonl").write_text("mutated\n")
    with pytest.raises(ValueError, match="input changed"):
        key = module._path_key(natural_dir / "cases.jsonl")
        module.verify_hashes({key: hashes[key]})


def test_selection_key_uses_exact_predeclared_bytes():
    expected = hashlib.sha256(
        (module.SALT + "\0challenge_fit\0id").encode()).hexdigest()
    assert module.selection_key("challenge_fit", "id") == (expected, "id")
