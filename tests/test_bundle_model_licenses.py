"""The packaged model-license record covers the exact distributed weights."""
from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import shutil
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGING = ROOT / "native/macos/Packaging"
sys.path.insert(0, str(PACKAGING))
SPEC = importlib.util.spec_from_file_location("bundle_engine", PACKAGING / "bundle_engine.py")
bundle_engine = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(bundle_engine)


def synthetic_project(tmp_path, monkeypatch):
    project = tmp_path / "project"
    license_dir = project / "licenses/hanglish-models"
    license_dir.mkdir(parents=True)
    shutil.copy2(ROOT / "licenses/hanglish-models/LICENSE.txt", license_dir / "LICENSE.txt")
    shutil.copy2(ROOT / "licenses/hanglish-models/NOTICE.txt", license_dir / "NOTICE.txt")
    weights = []
    for index, path in enumerate((
        "checkpoints/sentence_ranker/model.safetensors",
        ".omx/artifacts/ctc/training-word/model.pt",
        ".omx/artifacts/ctc/training-growth/model.pt",
    )):
        payload = f"synthetic licensed weight {index}".encode()
        weights.append({"path": path, "sha256": hashlib.sha256(payload).hexdigest()})
    license_record = {
        "schema": "hanglish-model-license-v1",
        "license": "CC-BY-SA-4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "license_file": "LICENSE.txt",
        "notice_file": "NOTICE.txt",
        "weights": weights,
    }
    (license_dir / "models.json").write_text(
        json.dumps(license_record, indent=2) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(bundle_engine, "ROOT", project)
    return project, license_record


def stage_synthetic_weights(engine, license_record):
    for index, item in enumerate(license_record["weights"]):
        packaged = engine / item["path"]
        packaged.parent.mkdir(parents=True, exist_ok=True)
        packaged.write_bytes(f"synthetic licensed weight {index}".encode())


def test_model_license_files_are_bundled_and_match_exact_weights(tmp_path, monkeypatch):
    _project, license_record = synthetic_project(tmp_path, monkeypatch)
    engine = tmp_path / "Engine"
    notices = tmp_path / "Licenses"
    stage_synthetic_weights(engine, license_record)

    bundled = bundle_engine.bundle_model_licenses(engine, notices)

    assert bundled == license_record
    assert bundled["license"] == "CC-BY-SA-4.0"
    assert bundled["license_url"] == "https://creativecommons.org/licenses/by-sa/4.0/"
    target = notices / "Hanglish-models"
    assert {path.name for path in target.iterdir()} == {
        "LICENSE.txt", "NOTICE.txt", "models.json"
    }
    assert json.loads((target / "models.json").read_text(encoding="utf-8")) == bundled
    assert "Attribution-ShareAlike 4.0 International" in (
        target / "LICENSE.txt"
    ).read_text(encoding="utf-8")
    notice = (target / "NOTICE.txt").read_text(encoding="utf-8")
    assert "permits commercial reuse" in notice
    assert "does not apply to these three weight files" in notice


def test_model_license_bundle_rejects_weight_mismatch(tmp_path, monkeypatch):
    _project, license_record = synthetic_project(tmp_path, monkeypatch)
    engine = tmp_path / "Engine"
    stage_synthetic_weights(engine, license_record)
    (engine / license_record["weights"][1]["path"]).write_bytes(b"changed weight")

    with pytest.raises(ValueError, match="Licensed model identity differs"):
        bundle_engine.bundle_model_licenses(engine, tmp_path / "Licenses")
