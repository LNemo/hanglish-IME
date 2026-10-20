"""Packaged installation guides retain only usable standalone links."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "native/macos/Packaging/package_guides.py"
    spec = importlib.util.spec_from_file_location("package_guides", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


guides = load_module()
COMMIT = "a" * 40


def assert_packaged_links_resolve(guide: Path) -> None:
    for target in re.findall(r"\]\(([^)]+)\)", guide.read_text(encoding="utf-8")):
        parts = urlsplit(target)
        if parts.scheme or target.startswith("#"):
            continue
        assert (guide.parent / unquote(parts.path)).is_file(), target


def test_packaged_guides_rewrite_repository_links_and_preserve_external_links(tmp_path):
    repo = tmp_path / "repo"
    source_dir = repo / "native" / "macos"
    source_dir.mkdir(parents=True)
    (repo / "LICENSE").write_text("license", encoding="utf-8")
    (repo / "THIRD_PARTY_NOTICES.md").write_text("notices", encoding="utf-8")
    (source_dir / "README.md").write_text(
        "[English](README.md) · [한국어](README.ko.md)\n"
        "[section](#download) [external](https://example.com/help)\n"
        "[script](install.sh) [license](../../LICENSE)\n",
        encoding="utf-8",
    )
    (source_dir / "README.ko.md").write_text(
        "[English build steps](README.md#build-and-verify)\n"
        "[notices](../../THIRD_PARTY_NOTICES.md)\n",
        encoding="utf-8",
    )
    output = tmp_path / "stage"
    output.mkdir()
    (output / "LICENSE").write_text("license", encoding="utf-8")
    (output / "THIRD_PARTY_NOTICES.md").write_text("notices", encoding="utf-8")

    guides.render_guides(repo, output, COMMIT)

    english = (output / "Installation Guide.md").read_text(encoding="utf-8")
    korean = (output / "설치 안내.md").read_text(encoding="utf-8")
    assert "[English](Installation%20Guide.md)" in english
    assert "[한국어](%EC%84%A4%EC%B9%98%20%EC%95%88%EB%82%B4.md)" in english
    assert "[section](#download)" in english
    assert "[external](https://example.com/help)" in english
    assert f"[script](https://github.com/LNemo/hanglish-IME/blob/{COMMIT}/native/macos/install.sh)" in english
    assert "[license](LICENSE)" in english
    assert (
        f"[English build steps](https://github.com/LNemo/hanglish-IME/blob/{COMMIT}/native/macos/README.md#build-and-verify)"
        in korean
    )
    assert "[notices](THIRD_PARTY_NOTICES.md)" in korean

    for guide in (output / "Installation Guide.md", output / "설치 안내.md"):
        assert_packaged_links_resolve(guide)


def test_current_guides_have_no_broken_links_after_packaging(tmp_path):
    (tmp_path / "LICENSE").write_bytes((ROOT / "LICENSE").read_bytes())
    (tmp_path / "THIRD_PARTY_NOTICES.md").write_bytes(
        (ROOT / "native/macos/Packaging/THIRD_PARTY_NOTICES.md").read_bytes()
    )
    guides.render_guides(ROOT, tmp_path, COMMIT)

    english = tmp_path / "Installation Guide.md"
    korean = tmp_path / "설치 안내.md"
    assert_packaged_links_resolve(english)
    assert_packaged_links_resolve(korean)
    assert (
        f"https://github.com/LNemo/hanglish-IME/blob/{COMMIT}/native/macos/README.md#build-and-verify"
        in korean.read_text(encoding="utf-8")
    )


def test_packaged_guides_require_an_exact_source_commit(tmp_path):
    try:
        guides.render_guides(tmp_path, tmp_path / "stage", "main")
    except ValueError as error:
        assert "40-character" in str(error)
    else:
        raise AssertionError("an unpinned guide source was accepted")
