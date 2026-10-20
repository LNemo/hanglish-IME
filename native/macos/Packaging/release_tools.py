#!/usr/bin/env python3
"""Testable helpers for producing and validating macOS distribution artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import plistlib
import subprocess
import sys
import tempfile
from typing import Any, Callable, Sequence
from urllib.parse import urlsplit


Runner = Callable[..., subprocess.CompletedProcess[str]]


def run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, capture_output=True)


def parse_json_output(output: str, description: str) -> dict[str, Any]:
    try:
        value = json.loads(output)
    except json.JSONDecodeError as error:
        raise ValueError(f"{description} did not return valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} returned a non-object JSON value")
    return value


def release_preflight(
    repo: Path,
    identity: str,
    team_id: str,
    profile: str,
    runner: Runner = run,
) -> str:
    if not identity.startswith("Developer ID Application:"):
        raise ValueError("release identity must be a Developer ID Application identity")
    commit = runner(["/usr/bin/git", "-C", str(repo), "rev-parse", "HEAD"]).stdout.strip()
    dirty = runner(["/usr/bin/git", "-C", str(repo), "status", "--porcelain"]).stdout
    if dirty:
        raise ValueError("release packaging requires a clean Git worktree")
    identities = runner(["/usr/bin/security", "find-identity", "-v", "-p", "codesigning"]).stdout
    matching = [line for line in identities.splitlines() if f'"{identity}"' in line]
    if not matching:
        raise ValueError("configured Developer ID Application identity was not found")
    if not any(f"({team_id})" in line for line in matching):
        raise ValueError("configured team ID does not match the signing identity")
    history = runner(
        [
            "/usr/bin/xcrun",
            "notarytool",
            "history",
            "--keychain-profile",
            profile,
            "--output-format",
            "json",
        ]
    )
    parse_json_output(history.stdout, "notarytool history")
    return commit


def verify_source_state(repo: Path, expected_commit: str, runner: Runner = run) -> None:
    commit = runner(["/usr/bin/git", "-C", str(repo), "rev-parse", "HEAD"]).stdout.strip()
    dirty = runner(["/usr/bin/git", "-C", str(repo), "status", "--porcelain"]).stdout
    if commit != expected_commit:
        raise ValueError("source commit changed during release packaging")
    if dirty:
        raise ValueError("source worktree changed during release packaging")


def notarize(dmg: Path, profile: str, log_path: Path, runner: Runner = run) -> str:
    submission = runner(
        [
            "/usr/bin/xcrun",
            "notarytool",
            "submit",
            str(dmg),
            "--keychain-profile",
            profile,
            "--wait",
            "--output-format",
            "json",
        ]
    )
    result = parse_json_output(submission.stdout, "notarytool submit")
    submission_id = result.get("id")
    if not isinstance(submission_id, str) or not submission_id:
        raise ValueError("notarytool submit returned no submission ID")
    log = runner(
        [
            "/usr/bin/xcrun",
            "notarytool",
            "log",
            submission_id,
            "--keychain-profile",
            profile,
            "--output-format",
            "json",
        ]
    )
    log_path.write_text(log.stdout.rstrip() + "\n", encoding="utf-8")
    if result.get("status") != "Accepted":
        raise ValueError(
            f"notarization status is {result.get('status', 'missing')}; see {log_path}"
        )
    return submission_id


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_digest(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    for item in sorted(path.rglob("*")):
        relative = str(item.relative_to(path))
        if item.is_symlink():
            kind, value = "symlink", item.readlink().as_posix()
        elif item.is_file():
            kind, value = "file", sha256(item)
        else:
            continue
        digest.update(f"{relative}\0{kind}\0{value}\n".encode())
        count += 1
    return {"file_count": count, "tree_sha256": digest.hexdigest()}


def write_manifest(
    repo: Path,
    plist_path: Path,
    artifact: Path,
    output: Path,
    mode: str,
    arch: str,
    inputs: list[Path],
    tree_inputs: list[tuple[str, Path]],
    required_commit: str | None = None,
    require_clean: bool = False,
) -> None:
    with plist_path.open("rb") as stream:
        plist = plistlib.load(stream)
    commit = run(["/usr/bin/git", "-C", str(repo), "rev-parse", "HEAD"]).stdout.strip()
    dirty = bool(run(["/usr/bin/git", "-C", str(repo), "status", "--porcelain"]).stdout)
    if required_commit is not None and commit != required_commit:
        raise ValueError("source commit changed during release packaging")
    if require_clean and dirty:
        raise ValueError("source worktree changed during release packaging")
    inventory = []
    for path in inputs:
        resolved = path.resolve()
        try:
            name = str(resolved.relative_to(repo.resolve()))
        except ValueError:
            name = resolved.name
        inventory.append({"path": name, "sha256": sha256(resolved)})
    trees = []
    for label, path in tree_inputs:
        trees.append({"name": label, **tree_digest(path.resolve())})
    manifest = {
        "schema": 1,
        "artifact": {"basename": artifact.name, "sha256": sha256(artifact)},
        "source": {
            "commit": commit,
            "clean": not dirty,
            "inputs": inventory,
            "bundled_trees": trees,
        },
        "application": {
            "version": plist["CFBundleShortVersionString"],
            "build": plist["CFBundleVersion"],
            "architecture": arch,
            "minimum_macos": plist["LSMinimumSystemVersion"],
        },
        "packaging": {
            "mode": mode,
            "release_signing": mode == "release",
            "publication_ready": False,
            "nonpublic_development_build": mode == "development",
            "manual_validation_required": ["clean Mac", "minimum supported macOS"],
        },
    }
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def find_stage_issues(stage: Path, forbidden_root: str) -> list[str]:
    issues: list[str] = []
    internal_words = ("audit", "readiness", "followup", "worklog")
    for path in sorted(stage.rglob("*")):
        if path.is_dir() or path.is_symlink():
            continue
        relative = str(path.relative_to(stage))
        if any(word in path.name.lower() for word in internal_words):
            issues.append(f"internal document name: {relative}")
        if path.suffix.lower() != ".json":
            continue
        project_metadata = ".omx" in path.parts or "checkpoints" in path.parts
        tokenizer_vocabulary = path.name == "tokenizer.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            if project_metadata:
                issues.append(f"invalid project metadata JSON: {relative}")
            continue
        pending = [value]
        while pending:
            current = pending.pop()
            if isinstance(current, dict):
                pending.extend(current.keys())
                pending.extend(current.values())
            elif isinstance(current, list):
                pending.extend(current)
            elif isinstance(current, str):
                if urlsplit(current).scheme in ("http", "https"):
                    continue
                windows_absolute = (
                    len(current) >= 3
                    and current[0].isalpha()
                    and current[1] == ":"
                    and current[2] in ("/", "\\")
                )
                known_private_path = current.startswith(
                    ("/Users/", "/home/", "/private/", "/var/folders/", "/tmp/", "/Volumes/", "file://")
                ) or windows_absolute
                if (
                    project_metadata
                    and (
                        known_private_path
                        or (not tokenizer_vocabulary and current.startswith("/"))
                    )
                    or (forbidden_root and forbidden_root in current)
                ):
                    issues.append(f"absolute build path in {relative}")
    return issues


def verify_release_artifact(dmg: Path, runner: Runner = run) -> None:
    artifact_commands = [
        ["/usr/bin/hdiutil", "verify", str(dmg)],
        ["/usr/bin/codesign", "--verify", "--verbose=2", str(dmg)],
        ["/usr/bin/xcrun", "stapler", "validate", str(dmg)],
        [
            "/usr/sbin/spctl",
            "--assess",
            "--type",
            "open",
            "--context",
            "context:primary-signature",
            "--verbose=4",
            str(dmg),
        ],
    ]
    for command in artifact_commands:
        runner(command)

    with tempfile.TemporaryDirectory(prefix="hanglish-release-verify-") as temporary:
        mount_point = Path(temporary) / "volume"
        mount_point.mkdir()
        runner(
            [
                "/usr/bin/hdiutil",
                "attach",
                "-readonly",
                "-nobrowse",
                "-noautoopen",
                "-mountpoint",
                str(mount_point),
                str(dmg),
            ]
        )
        try:
            installer = mount_point / "Hanglish Installer.app"
            input_method = installer / "Contents/Resources/Hanglish.app"
            for description, app in (
                ("installer application", installer),
                ("input method application", input_method),
            ):
                if not app.is_dir():
                    raise ValueError(f"mounted DMG is missing the {description}: {app}")
                runner(
                    [
                        "/usr/bin/codesign",
                        "--verify",
                        "--deep",
                        "--strict",
                        "--verbose=2",
                        str(app),
                    ]
                )
                runner(
                    [
                        "/usr/sbin/spctl",
                        "--assess",
                        "--type",
                        "execute",
                        "--verbose=4",
                        str(app),
                    ]
                )
        finally:
            runner(["/usr/bin/hdiutil", "detach", str(mount_point)])


def app_version(plist_path: Path) -> str:
    with plist_path.open("rb") as stream:
        return str(plistlib.load(stream)["CFBundleShortVersionString"])


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--repo", type=Path, required=True)
    preflight.add_argument("--identity", required=True)
    preflight.add_argument("--team-id", required=True)
    preflight.add_argument("--profile", required=True)
    preflight.add_argument("--commit-output", type=Path)
    source_state = subparsers.add_parser("verify-source")
    source_state.add_argument("--repo", type=Path, required=True)
    source_state.add_argument("--commit", required=True)
    notarization = subparsers.add_parser("notarize")
    notarization.add_argument("--dmg", type=Path, required=True)
    notarization.add_argument("--profile", required=True)
    notarization.add_argument("--log", type=Path, required=True)
    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--repo", type=Path, required=True)
    manifest.add_argument("--plist", type=Path, required=True)
    manifest.add_argument("--artifact", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--mode", choices=("development", "release"), required=True)
    manifest.add_argument("--arch", required=True)
    manifest.add_argument("--input", type=Path, action="append", default=[])
    manifest.add_argument("--tree-input", action="append", default=[])
    manifest.add_argument("--require-commit")
    manifest.add_argument("--require-clean", action="store_true")
    audit = subparsers.add_parser("audit-stage")
    audit.add_argument("--stage", type=Path, required=True)
    audit.add_argument("--forbid-root", default="")
    verification = subparsers.add_parser("verify-release")
    verification.add_argument("--dmg", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "preflight":
            commit = release_preflight(
                args.repo, args.identity, args.team_id, args.profile
            )
            if args.commit_output is not None:
                args.commit_output.write_text(commit + "\n", encoding="utf-8")
            print(f"Release credentials verified for {commit}.")
        elif args.command == "verify-source":
            verify_source_state(args.repo, args.commit)
            print(f"Release source remains clean at {args.commit}.")
        elif args.command == "notarize":
            submission_id = notarize(args.dmg, args.profile, args.log)
            print(f"Notarization accepted: {submission_id}")
        elif args.command == "manifest":
            tree_inputs = []
            for item in args.tree_input:
                if "=" not in item:
                    raise ValueError("tree input must use LABEL=PATH")
                label, path = item.split("=", 1)
                if not label or not path:
                    raise ValueError("tree input must use LABEL=PATH")
                tree_inputs.append((label, Path(path)))
            write_manifest(
                args.repo,
                args.plist,
                args.artifact,
                args.output,
                args.mode,
                args.arch,
                args.input,
                tree_inputs,
                args.require_commit,
                args.require_clean,
            )
            print(f"Artifact manifest: {args.output}")
        elif args.command == "audit-stage":
            issues = find_stage_issues(args.stage, args.forbid_root)
            if issues:
                raise ValueError("stage audit failed:\n- " + "\n- ".join(issues))
            print("Stage audit passed.")
        else:
            verify_release_artifact(args.dmg)
            print("Signed, stapled, and Gatekeeper-assessed release artifact passed.")
            print("Manual validation remains required: clean Mac and minimum supported macOS.")
    except (OSError, subprocess.CalledProcessError, ValueError, KeyError) as error:
        print(f"release packaging error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
