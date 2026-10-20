#!/usr/bin/env python3
"""Create standalone installation guides for the packaged DMG."""

from __future__ import annotations

import argparse
import posixpath
import re
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit


SOURCE_URL = "https://github.com/LNemo/hanglish-IME/blob"
GUIDE_OUTPUTS = {
    "README.md": "Installation Guide.md",
    "README.ko.md": "설치 안내.md",
}
LOCAL_DOCUMENTS = {
    "LICENSE": "LICENSE",
    "THIRD_PARTY_NOTICES.md": "THIRD_PARTY_NOTICES.md",
}
INLINE_LINK = re.compile(
    r"(?P<prefix>!?\[[^\]\n]*\]\()"
    r"(?P<destination><[^>\n]+>|[^\s)\n]+)"
    r"(?P<suffix>(?:\s+(?:\"[^\"\n]*\"|'[^'\n]*'|\([^\n)]*\)))?\))"
)
COMMIT = re.compile(r"[0-9a-fA-F]{40}")


def _rewrite_destination(destination: str, source_relative: Path, commit: str) -> str:
    wrapped = destination.startswith("<") and destination.endswith(">")
    raw = destination[1:-1] if wrapped else destination
    parts = urlsplit(raw)
    if parts.scheme or parts.netloc or raw.startswith(("/", "#")):
        return destination

    source_parent = source_relative.parent.as_posix()
    normalized = posixpath.normpath(posixpath.join(source_parent, unquote(parts.path)))
    if normalized == ".." or normalized.startswith("../"):
        raise ValueError(f"guide link escapes the repository: {raw}")

    local_target = None
    if not parts.query and not parts.fragment:
        for source_name, output_name in GUIDE_OUTPUTS.items():
            sibling = posixpath.normpath(posixpath.join(source_parent, source_name))
            if normalized == sibling:
                local_target = quote(output_name)
                break
    if local_target is None and normalized in LOCAL_DOCUMENTS:
        local_target = LOCAL_DOCUMENTS[normalized]

    if local_target is not None:
        rewritten = local_target
        if parts.query:
            rewritten += f"?{parts.query}"
        if parts.fragment:
            rewritten += f"#{parts.fragment}"
    else:
        rewritten = f"{SOURCE_URL}/{commit}/{quote(normalized, safe='/')}"
        if parts.query:
            rewritten += f"?{parts.query}"
        if parts.fragment:
            rewritten += f"#{parts.fragment}"
    return f"<{rewritten}>" if wrapped else rewritten


def render_guide(source: Path, output: Path, repo: Path, commit: str) -> None:
    source_relative = source.resolve().relative_to(repo.resolve())
    text = source.read_text(encoding="utf-8")

    def replace(match: re.Match[str]) -> str:
        destination = _rewrite_destination(
            match.group("destination"), source_relative, commit
        )
        return f'{match.group("prefix")}{destination}{match.group("suffix")}'

    output.write_text(INLINE_LINK.sub(replace, text), encoding="utf-8")


def render_guides(repo: Path, output_dir: Path, commit: str) -> None:
    if not COMMIT.fullmatch(commit):
        raise ValueError("commit must be a full 40-character hexadecimal Git commit")
    source_dir = repo / "native" / "macos"
    output_dir.mkdir(parents=True, exist_ok=True)
    for source_name, output_name in GUIDE_OUTPUTS.items():
        render_guide(source_dir / source_name, output_dir / output_name, repo, commit)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    render_guides(args.repo, args.output_dir, args.commit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
