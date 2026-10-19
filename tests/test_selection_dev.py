from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "data" / "challenge" / "selection-dev-v1.1"
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
    ("nat", "낫"), ("nat", "낮"), ("nat", "낯"),
    ("bit", "빗"), ("bit", "빛"), ("bit", "빚"),
    ("got", "곧"), ("got", "곳"),
    ("mat", "맛"), ("mat", "맞"),
    ("gachi", "같이"), ("gachi", "가치"),
    ("sut", "숯"), ("sut", "숱"),
    ("bak", "밖"), ("bak", "박"),
    ("mok", "몫"), ("mok", "목"),
    ("ot", "옷"), ("ot", "옻"),
}


def load_cases() -> list[dict[str, object]]:
    return [json.loads(line) for line in CASES_PATH.read_text(encoding="utf-8").splitlines()]


def test_selection_dev_counts_and_schema() -> None:
    rows = load_cases()
    assert len(rows) == 60
    assert all(set(row) == EXPECTED_FIELDS for row in rows)
    assert all(row["split"] == "selection_dev" for row in rows)
    assert all(
        row["provenance"] == "authored-selection-development-manual-v1.1-rr-correction"
        for row in rows
    )
    assert all(isinstance(row["expected"], list) and len(row["expected"]) == 1 for row in rows)

    hints = [row for row in rows if row["context_role"] == "semantic_hint"]
    sentences = [row for row in rows if row["context_role"] == "none"]
    assert len(hints) == 40
    assert len(sentences) == 20
    assert all(len(str(row["expected"][0]).split()) <= 12 for row in sentences)
    assert all(len(str(row["input"]).split()) == len(str(row["expected"][0]).split()) for row in sentences)
    assert Counter(row["boundary"] for row in sentences) == Counter(
        {"word_spaced": 12, "latin_suffix": 4, "protected_backtick": 4}
    )
    assert Counter((row["input"], row["expected"][0]) for row in hints) == Counter(
        {sense: 2 for sense in SENSES}
    )
    assert all(row["expected"][0] not in row["context"] for row in hints)
    assert all(
        row["boundary"] == ("stem_fragment" if row["expected"] == ["맞"] else "isolated_pronunciation")
        for row in hints
    )


def test_selection_dev_is_unique_and_manifest_matches() -> None:
    rows = load_cases()
    assert len({row["id"] for row in rows}) == 60
    assert len({(row["input"], row["context"], tuple(row["expected"])) for row in rows}) == 60
    assert len({row["context"] for row in rows if row["context"]}) == 40
    assert len({row["family"] for row in rows}) == 60

    payload = CASES_PATH.read_bytes()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["case_count"] == 60
    assert manifest["semantic_hint_count"] == 40
    assert manifest["sentence_count"] == 20
    assert manifest["cases_sha256"] == hashlib.sha256(payload).hexdigest()
    assert manifest["claim"] == "authored-development; not a blind or naturally sampled evaluation set"
    assert manifest["name"] == "selection-dev-v1.1"
    assert manifest["correction"]["parent_cases_sha256"] == (
        "697e22a6c029495b7c4670fe43b44f7b0ffb3e351d5e341f0a8aa3fb0f4c28fb"
    )
    assert manifest["correction"]["expected_korean_changes"] == 0


def test_builder_refuses_to_overwrite(tmp_path: Path) -> None:
    output_dir = tmp_path / "selection-dev-v1.1"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "build_selection_dev.py"),
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
