from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "data" / "challenge" / "reasoning-dev-v1.1"
CASES_PATH = CORPUS_DIR / "cases.jsonl"
MANIFEST_PATH = CORPUS_DIR / "manifest.json"
EXPECTED_FIELDS = {
    "id",
    "input",
    "context",
    "expected",
    "category",
    "family",
    "split",
    "provenance",
    "context_role",
    "boundary",
}
SENSES = {
    ("nat", "낫"),
    ("nat", "낮"),
    ("nat", "낯"),
    ("bit", "빗"),
    ("bit", "빛"),
    ("bit", "빚"),
    ("got", "곧"),
    ("got", "곳"),
    ("mat", "맛"),
    ("mat", "맞"),
    ("gachi", "같이"),
    ("gachi", "가치"),
    ("sut", "숯"),
    ("sut", "숱"),
    ("bak", "밖"),
    ("bak", "박"),
    ("mok", "몫"),
    ("mok", "목"),
    ("ot", "옷"),
    ("ot", "옻"),
}


def load_cases() -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in CASES_PATH.read_text(encoding="utf-8").splitlines()
    ]


def test_reasoning_dev_counts_and_contract() -> None:
    rows = load_cases()
    assert len(rows) == 40
    assert all(set(row) == EXPECTED_FIELDS for row in rows)
    assert all(row["split"] == "selection_dev" for row in rows)
    assert all(row["category"] == "homophone_context" for row in rows)
    assert all(row["context_role"] == "semantic_hint" for row in rows)
    assert all(
        row["provenance"]
        == "authored-reasoning-development-manual-v1.1-context-correction"
        for row in rows
    )
    assert Counter((row["input"], row["expected"][0]) for row in rows) == Counter(
        {sense: 2 for sense in SENSES}
    )
    assert {row["input"] for row in rows} == {
        "nat",
        "bit",
        "got",
        "mat",
        "gachi",
        "sut",
        "bak",
        "mok",
        "ot",
    }
    assert all(row["expected"][0] not in row["context"] for row in rows)
    assert all(
        row["boundary"]
        == ("stem_fragment" if row["expected"] == ["맞"] else "isolated_pronunciation")
        for row in rows
    )
    assert all("어간" in row["context"] for row in rows if row["expected"] == ["맞"])


def test_reasoning_dev_uniqueness_and_manifest() -> None:
    rows = load_cases()
    assert len({row["id"] for row in rows}) == 40
    assert len({row["context"] for row in rows}) == 40
    assert len({row["family"] for row in rows}) == 40
    assert len(
        {(row["input"], row["context"], tuple(row["expected"])) for row in rows}
    ) == 40

    payload = CASES_PATH.read_bytes()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["name"] == "reasoning-dev-v1.1"
    assert manifest["case_count"] == 40
    assert manifest["sense_count"] == 20
    assert manifest["cases_per_sense"] == 2
    assert manifest["cases_sha256"] == hashlib.sha256(payload).hexdigest()
    assert "not a blind or naturally sampled test set" in manifest["claim"]
    assert manifest["correction"] == {
        "kind": "context-only",
        "parent": "reasoning-dev-v1",
        "parent_cases_sha256": (
            "8e1c5d87c45b045d44bb36e2cffb30971ca66a388ad03b84ceeff998d1e16ba2"
        ),
        "corrected_case_ids": ["reasoning-dev-0015", "reasoning-dev-0029"],
        "input_changes": 0,
        "expected_korean_changes": 0,
    }


def test_reasoning_builder_is_reproducible_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "reasoning-dev-v1.1"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "build_reasoning_dev.py"),
        "--output-dir",
        str(output_dir),
    ]
    first = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    second = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    assert first.returncode == 0
    assert (output_dir / "cases.jsonl").read_bytes() == CASES_PATH.read_bytes()
    assert (output_dir / "manifest.json").read_bytes() == MANIFEST_PATH.read_bytes()
    assert second.returncode == 2
    assert "refusing to overwrite immutable output" in second.stderr
