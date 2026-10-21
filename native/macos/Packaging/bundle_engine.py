"""Copy the installed, pinned runtime into a relocatable macOS app payload."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from release_metadata import PRIVATE_PATH, sanitize_metadata

ROOT = Path(__file__).resolve().parents[3]
SEEDS = ("torch", "transformers", "numpy", "wordfreq", "safetensors", "tokenizers")


def distributions():
    result = {}
    pending = list(SEEDS)
    while pending:
        name = canonicalize_name(pending.pop())
        if name in result:
            continue
        dist = metadata.distribution(name)
        result[name] = dist
        for raw in dist.requires or []:
            requirement = Requirement(raw)
            if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
                pending.append(requirement.name)
    return result


def copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resources", type=Path, required=True)
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11) or platform.machine() != "arm64":
        raise SystemExit("This release requires the pinned Python 3.11 arm64 build environment.")
    subprocess.run([sys.executable, str(ROOT / "scripts/prepare_runtime_assets.py"),
                    "check", "--profile", "package"], check=True)
    resources = args.resources.resolve()
    resources.mkdir(parents=True, exist_ok=True)
    runtime, engine = resources / "Python", resources / "Engine"
    # Never mutate an existing installed payload in place.
    if runtime.exists() or engine.exists():
        raise SystemExit("Build into a clean staging app; Python/Engine already exists.")
    base = Path(sys.base_prefix).resolve()
    shutil.copytree(base, runtime, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site = runtime / "lib" / version / "site-packages"
    # uv standalone may contain bootstrap packaging tools; distribute only the closure below.
    if site.exists():
        shutil.rmtree(site)
    site.mkdir(parents=True)
    packages = distributions()
    package_manifest = []
    for name, dist in sorted(packages.items()):
        source_base = Path(dist.locate_file("")).resolve()
        copied = 0
        for relative in dist.files or []:
            relative = Path(relative)
            if relative.is_absolute() or ".." in relative.parts or "__pycache__" in relative.parts or relative.suffix == ".pyc":
                continue
            source = source_base / relative
            if source.is_file():
                copy_file(source, site / relative)
                copied += 1
        if not copied:
            raise ValueError(f"No distribution files copied for {name}")
        package_manifest.append({"name": name, "version": dist.version,
                                 "license": dist.metadata.get("License-Expression") or dist.metadata.get("License", "See distribution metadata"),
                                 "files": copied})
    specification = json.loads((Path(__file__).parent / "engine-files.json").read_text())
    patterns = [pattern for group in specification["required"].values() for pattern in group]
    patterns += specification["optional_compatibility"]["files"]
    patterns += specification["source_files_hashed_by_runtime_version"]
    selected = set()
    for pattern in patterns:
        matches = list(ROOT.glob(pattern))
        if not matches:
            raise FileNotFoundError(pattern)
        for source in matches:
            if source.is_file():
                selected.add(source)
    metadata_changes = {}
    for source in sorted(selected):
        relative = source.relative_to(ROOT)
        destination = engine / relative
        copy_file(source, destination)
        if (source.suffix == ".json" and source.name != "tokenizer.json"
                and relative.parts[0] in ("checkpoints", ".omx")):
            original = json.loads(source.read_text(encoding="utf-8"))
            cleaned = sanitize_metadata(original)
            if cleaned != original:
                destination.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                metadata_changes[str(relative)] = {
                    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "packaged_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                    "transformation": "repository-relative-provenance-v1",
                }
    copy_file(Path(__file__).with_name("engine_service.py"), engine / "engine_service.py")
    # Retain upstream license/attribution documents independently of importability.
    notice = resources / "Licenses"
    notice.mkdir()
    for name in ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md"):
        copy_file(ROOT / name, notice / "Hanglish" / name)
    for name in ("PROJECT-LICENSE.txt", "SOURCE.json"):
        copy_file(ROOT / "licenses/koelectra" / name, notice / "KoELECTRA-project" / name)
    for source in ROOT.glob("data/external/koelectra-small-v3/*"):
        if source.name.lower().startswith(("license", "readme", "source")) and source.is_file():
            copy_file(source, notice / "KoELECTRA" / source.name)
    source_notice = Path(__file__).with_name("THIRD_PARTY_NOTICES.md")
    copy_file(source_notice, notice / source_notice.name)
    copy_file(Path(__file__).with_name("wordfreq-NOTICE.md"), notice / "wordfreq-NOTICE.md")
    copy_file(Path(__file__).with_name("wordfreq-LICENSE.txt"), notice / "wordfreq-LICENSE.txt")
    (notice / "python-distributions.json").write_text(json.dumps(package_manifest, ensure_ascii=False, indent=2) + "\n")
    for link in runtime.rglob("*"):
        if link.is_symlink() and not link.resolve().is_relative_to(runtime):
            raise ValueError(f"External runtime symlink: {link.relative_to(runtime)}")
    for path in engine.rglob("*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        # Tokenizer vocabulary keys such as '/' are tokens, not provenance paths.
        private_path = (bool(PRIVATE_PATH.search(path.read_text(encoding="utf-8")))
                        if path.name == "tokenizer.json" else sanitize_metadata(value) != value)
        if private_path:
            raise ValueError(f"Private build path remains in packaged metadata: {path.relative_to(engine)}")
    # Metadata participates in the engine identity. Measure the packaged identity
    # instead of falsely advertising the original source version after redaction.
    probe = subprocess.run(
        [str(runtime / "bin/python3.11"), "-I", "-B", str(engine / "engine_service.py")],
        input=json.dumps({"request_id": "package-version", "input": "imnida"}) + "\n",
        text=True, capture_output=True, timeout=120, check=True,
        env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": os.environ.get("HOME", "/tmp")},
    )
    reply = json.loads(probe.stdout)
    packaged_version = reply.get("version")
    if (reply.get("request_id") != "package-version" or reply.get("result") != "입니다"
            or packaged_version != specification["selected_version"]):
        raise ValueError("Packaged engine identity/conversion check failed")
    manifest = {"schema": "hanglish-bundled-engine-v1", "python": platform.python_version(),
                "architecture": "arm64", "minimum_macos": "14.0",
                "engine_version": packaged_version,
                "source_engine_version": specification["selected_version"],
                "model_selection": specification["release_model_selection"],
                "metadata_transformations": metadata_changes,
                "packages": [{"name": p["name"], "version": p["version"]} for p in package_manifest],
                "engine_files": {str(p.relative_to(engine)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(engine.rglob("*")) if p.is_file()}}
    (resources / "engine-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"Bundled {len(package_manifest)} Python distributions and {len(manifest['engine_files'])} engine files.")
    print(f"Payload: {resources}")


if __name__ == "__main__":
    main()
