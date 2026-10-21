"""The packaged model-license record covers the exact distributed weights."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PACKAGING = ROOT / "native/macos/Packaging"
sys.path.insert(0, str(PACKAGING))
SPEC = importlib.util.spec_from_file_location("bundle_engine", PACKAGING / "bundle_engine.py")
bundle_engine = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(bundle_engine)


def test_model_license_files_are_bundled_and_match_exact_weights(tmp_path):
    license_record = json.loads(
        (ROOT / "licenses/hanglish-models/models.json").read_text(encoding="utf-8")
    )
    engine = tmp_path / "Engine"
    notices = tmp_path / "Licenses"
    for item in license_record["weights"]:
        packaged = engine / item["path"]
        packaged.parent.mkdir(parents=True, exist_ok=True)
        packaged.symlink_to(ROOT / item["path"])

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
