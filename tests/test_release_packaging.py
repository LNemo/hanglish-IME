"""Release packaging fails closed without invoking real signing or notarization."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import plistlib
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


release = load_module("release_tools", "native/macos/Packaging/release_tools.py")
signing = load_module("sign_payload", "native/macos/Packaging/sign_payload.py")


class FakeRunner:
    def __init__(self, outputs: list[str]):
        self.outputs = iter(outputs)
        self.commands: list[list[str]] = []

    def __call__(self, command):
        self.commands.append(list(command))
        return subprocess.CompletedProcess(command, 0, next(self.outputs), "")


def test_release_preflight_validates_clean_source_identity_team_and_notary_profile(tmp_path):
    identity = "Developer ID Application: Example Developer (TEAM123456)"
    runner = FakeRunner([
        "abc123\n",
        "",
        f'  1) HASH "{identity}"\n',
        '{"history": []}',
    ])
    assert release.release_preflight(
        tmp_path, identity, "TEAM123456", "hanglish-notary", runner
    ) == "abc123"
    assert runner.commands[-1][1:3] == ["notarytool", "history"]
    assert "hanglish-notary" in runner.commands[-1]


def test_preflight_cli_persists_the_exact_verified_commit(tmp_path, monkeypatch):
    output = tmp_path / "commit.txt"
    monkeypatch.setattr(release, "release_preflight", lambda *_args: "verified-commit")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "release_tools.py",
            "preflight",
            "--repo",
            str(tmp_path),
            "--identity",
            "Developer ID Application: Example (TEAM123456)",
            "--team-id",
            "TEAM123456",
            "--profile",
            "profile",
            "--commit-output",
            str(output),
        ],
    )
    assert release.main() == 0
    assert output.read_text() == "verified-commit\n"


@pytest.mark.parametrize(
    ("outputs", "message"),
    [
        (["abc\n", " M file\n"], "clean Git worktree"),
        (["abc\n", "", "0 valid identities\n"], "identity was not found"),
        (
            [
                "abc\n",
                "",
                '1) HASH "Developer ID Application: Example (OTHERTEAM)"\n',
            ],
            "team ID",
        ),
    ],
)
def test_release_preflight_fails_closed(outputs, message, tmp_path):
    runner = FakeRunner(outputs)
    with pytest.raises(ValueError, match=message):
        release.release_preflight(
            tmp_path,
            "Developer ID Application: Example (OTHERTEAM)",
            "TEAM123456",
            "profile",
            runner,
        )


def test_notarization_rejection_saves_log_and_fails(tmp_path):
    runner = FakeRunner([
        '{"id": "submission-1", "status": "Invalid"}',
        '{"issues": [{"message": "bad signature"}]}',
    ])
    log = tmp_path / "notary.json"
    with pytest.raises(ValueError, match="Invalid"):
        release.notarize(tmp_path / "Hanglish.dmg", "profile", log, runner)
    assert json.loads(log.read_text()) == {"issues": [{"message": "bad signature"}]}
    assert runner.commands[0][1:3] == ["notarytool", "submit"]
    assert "--wait" in runner.commands[0]


def test_release_preflight_rejects_unusable_notary_profile(tmp_path):
    identity = "Developer ID Application: Example (TEAM123456)"
    runner = FakeRunner(["abc\n", "", f'1) HASH "{identity}"\n', "not-json"])
    with pytest.raises(ValueError, match="notarytool history"):
        release.release_preflight(
            tmp_path, identity, "TEAM123456", "missing-profile", runner
        )


def test_source_state_rejects_changes_before_artifact_creation(tmp_path):
    with pytest.raises(ValueError, match="worktree changed"):
        release.verify_source_state(
            tmp_path, "abc", FakeRunner(["abc\n", " M source\n"])
        )


def test_stage_audit_rejects_private_json_paths_and_internal_documents(tmp_path):
    (tmp_path / "metadata.json").write_text(
        json.dumps({"source": "/Users/developer/repo/model.bin"}), encoding="utf-8"
    )
    (tmp_path / "release-readiness.md").write_text("private", encoding="utf-8")
    issues = release.find_stage_issues(tmp_path, "/Users/developer/repo")
    assert any("absolute build path" in issue for issue in issues)
    assert any("internal document name" in issue for issue in issues)
    assert all("/Users/developer" not in issue for issue in issues)


def test_stage_audit_preserves_urls_and_rejects_invalid_project_json(tmp_path):
    public = tmp_path / "public.json"
    public.write_text(
        json.dumps({"source": "https://example.org/home/project/license"}), encoding="utf-8"
    )
    project = tmp_path / "Payload/Engine/.omx/model.json"
    project.parent.mkdir(parents=True)
    project.write_text("{invalid", encoding="utf-8")
    issues = release.find_stage_issues(tmp_path, "")
    assert issues == ["invalid project metadata JSON: Payload/Engine/.omx/model.json"]


def test_stage_audit_rejects_other_project_absolute_path_forms(tmp_path):
    project = tmp_path / "Payload/Engine/checkpoints/model.json"
    project.parent.mkdir(parents=True)
    project.write_text(
        json.dumps({"unix": "/tmp/build/model", "windows": "C:\\Users\\dev\\model"}),
        encoding="utf-8",
    )
    issues = release.find_stage_issues(tmp_path, "")
    assert issues == [
        "absolute build path in Payload/Engine/checkpoints/model.json",
        "absolute build path in Payload/Engine/checkpoints/model.json",
    ]


def test_stage_audit_treats_tokenizer_slash_tokens_as_vocabulary(tmp_path):
    tokenizer = tmp_path / "Payload/Engine/.omx/checkpoint/tokenizer.json"
    tokenizer.parent.mkdir(parents=True)
    tokenizer.write_text(
        json.dumps({"model": {"vocab": {"/": 1, "//": 2, "word": 3}}}),
        encoding="utf-8",
    )
    assert release.find_stage_issues(tmp_path, "") == []


def test_stage_audit_checks_project_metadata_keys(tmp_path):
    metadata = tmp_path / "Payload/Engine/checkpoints/model.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(
        json.dumps({"source_hashes": {"/Users/developer/train.json": "abc"}}),
        encoding="utf-8",
    )
    issues = release.find_stage_issues(tmp_path, "")
    assert issues == ["absolute build path in Payload/Engine/checkpoints/model.json"]


def test_artifact_manifest_uses_portable_name_and_records_inputs(tmp_path, monkeypatch):
    artifact = tmp_path / "Hanglish-0.2.6-arm64.dmg"
    artifact.write_bytes(b"dmg")
    source = tmp_path / "input.json"
    source.write_text("{}", encoding="utf-8")
    plist = tmp_path / "Info.plist"
    with plist.open("wb") as stream:
        plistlib.dump(
            {
                "CFBundleShortVersionString": "0.2.6",
                "CFBundleVersion": "8",
                "LSMinimumSystemVersion": "14.0",
            },
            stream,
        )
    runner = FakeRunner(["deadbeef\n", ""])
    monkeypatch.setattr(release, "run", runner)
    output = tmp_path / "artifact.json"
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "python").write_bytes(b"python")
    release.write_manifest(
        tmp_path,
        plist,
        artifact,
        output,
        "development",
        "arm64",
        [source],
        [("bundled-python", runtime)],
    )
    value = json.loads(output.read_text())
    assert value["artifact"]["basename"] == artifact.name
    assert value["source"]["commit"] == "deadbeef"
    assert value["source"]["clean"] is True
    assert value["source"]["inputs"][0]["path"] == "input.json"
    assert value["source"]["bundled_trees"][0]["file_count"] == 1
    assert value["packaging"]["publication_ready"] is False
    assert value["packaging"]["nonpublic_development_build"] is True


def test_release_manifest_rejects_source_drift(tmp_path, monkeypatch):
    plist = tmp_path / "Info.plist"
    with plist.open("wb") as stream:
        plistlib.dump(
            {
                "CFBundleShortVersionString": "0.2.6",
                "CFBundleVersion": "8",
                "LSMinimumSystemVersion": "14.0",
            },
            stream,
        )
    artifact = tmp_path / "candidate.dmg"
    artifact.write_bytes(b"candidate")
    monkeypatch.setattr(release, "run", FakeRunner(["changed-commit\n", " M source\n"]))
    with pytest.raises(ValueError, match="source commit changed"):
        release.write_manifest(
            tmp_path,
            plist,
            artifact,
            tmp_path / "manifest.json",
            "release",
            "arm64",
            [],
            [],
            required_commit="original-commit",
            require_clean=True,
        )


def test_release_artifact_verification_includes_gatekeeper_and_staple():
    runner = FakeRunner(["", "", "", ""])
    release.verify_release_artifact(Path("Hanglish.dmg"), runner)
    flattened = [" ".join(command) for command in runner.commands]
    assert any("stapler validate" in command for command in flattened)
    assert any("spctl --assess --type open" in command for command in flattened)


def test_signing_commands_distinguish_development_and_release():
    path = Path("binary")
    assert signing.sign_command(path, "-", False) == [
        "/usr/bin/codesign", "--force", "--sign", "-", "binary"
    ]
    assert signing.sign_command(path, "Developer ID Application: Example", True) == [
        "/usr/bin/codesign",
        "--force",
        "--sign",
        "Developer ID Application: Example",
        "--options",
        "runtime",
        "--timestamp",
        "binary",
    ]


def test_release_signing_failure_propagates_without_adhoc_fallback(tmp_path, monkeypatch):
    app = tmp_path / "Hanglish.app"
    binary = app / "Contents/MacOS/Hanglish"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"\xcf\xfa\xed\xfe" + b"payload")
    commands = []

    def fail_release(command, **_kwargs):
        commands.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(signing.subprocess, "run", fail_release)
    with pytest.raises(subprocess.CalledProcessError):
        signing.sign_app(app, "Developer ID Application: Example", True)
    assert len(commands) == 1
    assert "Developer ID Application: Example" in commands[0]
    assert "-" not in commands[0]
