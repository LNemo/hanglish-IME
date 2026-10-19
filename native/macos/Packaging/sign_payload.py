"""Sign embedded Mach-O files before sealing a macOS application bundle."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


MACHO_MAGIC = {
    b"\xcf\xfa\xed\xfe",
    b"\xce\xfa\xed\xfe",
    b"\xfe\xed\xfa\xcf",
    b"\xfe\xed\xfa\xce",
    b"\xca\xfe\xba\xbe",
    b"\xbe\xba\xfe\xca",
    b"\xca\xfe\xba\xbf",
    b"\xbf\xba\xfe\xca",
}


def embedded_macho_files(app: Path) -> list[Path]:
    paths: list[Path] = []
    for path in app.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        with path.open("rb") as stream:
            if stream.read(4) in MACHO_MAGIC:
                paths.append(path)
    return sorted(paths, key=lambda item: (-len(item.parts), str(item)))


def sign_command(path: Path, identity: str, release: bool) -> list[str]:
    command = ["/usr/bin/codesign", "--force", "--sign", identity]
    if release:
        command.extend(["--options", "runtime", "--timestamp"])
    command.append(str(path))
    return command


def sign_app(app: Path, identity: str, release: bool) -> int:
    count = 0
    for path in embedded_macho_files(app):
        subprocess.run(
            sign_command(path, identity, release),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        count += 1
    subprocess.run(sign_command(app, identity, release), check=True)
    subprocess.run(
        ["/usr/bin/codesign", "--verify", "--deep", "--strict", "--verbose=2", str(app)],
        check=True,
    )
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("app", type=Path)
    parser.add_argument("--identity", default="-")
    parser.add_argument("--release", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = args.app.resolve()
    if not app.is_dir() or app.suffix != ".app":
        raise SystemExit(f"Application bundle not found: {app}")
    if args.release and args.identity == "-":
        raise SystemExit("Release signing requires a Developer ID Application identity.")
    count = sign_app(app, args.identity, args.release)
    mode = "Developer ID" if args.release else "ad-hoc development"
    print(f"Signed {count} embedded Mach-O files and verified {app.name} ({mode}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
