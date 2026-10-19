import ast
from pathlib import Path

import pytest

from scripts import verify_document_phonetic_shard_v2 as verifier


def test_v2_has_no_capture_producer_neural_or_tensor_imports():
    tree = ast.parse(Path(verifier.__file__).read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    forbidden = (
        "capture_document_phonetic_shard", "torch", "phonetic_likelihood",
        "production_phonetic_choice_converter", "preflight_document_phonetic_choice",
    )
    assert not any(item in name for name in imports for item in forbidden)


def test_exact_shard_zero_command_has_no_predecessor():
    directory = verifier.RUNS[0].resolve()
    assert verifier.expected_launch_command(directory, 0) == [
        str(verifier.ROOT / ".venv/bin/python"),
        str(verifier.PRODUCER.resolve()),
        "--output", str(directory), "--shard", "0",
    ]


def test_exact_shard_one_command_binds_verified_predecessor():
    directory = verifier.RUNS[1].resolve()
    assert verifier.expected_launch_command(directory, 1) == [
        str(verifier.ROOT / ".venv/bin/python"),
        str(verifier.PRODUCER.resolve()),
        "--output", str(directory), "--shard", "1",
        "--previous", str(verifier.RUNS[0].resolve()),
    ]


def test_runtime_pin_and_v2_audit_identity_are_distinct():
    pins = verifier.critical_runtime_pins(verifier.PRODUCER_SHA256)
    runtime = verifier.path_key(verifier.RUNTIME_VERIFIER)
    audit = verifier.path_key(Path(verifier.__file__))
    assert runtime != audit
    assert pins[runtime] == verifier.RUNTIME_VERIFIER_SHA256
    assert audit not in pins
    assert verifier.sha(verifier.RUNTIME_VERIFIER) == verifier.RUNTIME_VERIFIER_SHA256
    assert verifier.sha(Path(verifier.__file__)) != verifier.RUNTIME_VERIFIER_SHA256


@pytest.mark.parametrize("shard", [-1, 2, True, "1"])
def test_expected_command_rejects_invalid_shard(shard):
    with pytest.raises(ValueError, match="invalid shard"):
        verifier.expected_launch_command(verifier.RUNS[1], shard)


def test_v2_top_level_verifier_rejects_shard_zero_before_reads():
    with pytest.raises(ValueError, match="restricted to shard 1"):
        verifier.verify(verifier.RUNS[0], 0)
