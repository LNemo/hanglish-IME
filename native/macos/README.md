# Hanglish for macOS

[English](README.md) · [한국어](README.ko.md) · [Project overview](../../README.md)

Hanglish is for people who can speak Korean but find it difficult to write in Hangul. Type Korean pronunciation with Latin letters, request Korean candidates, and choose the spelling you want.

The current prototype is **0.2.9 (build 11), for Apple Silicon (arm64), with a deployment target of macOS 14 or later**. Local testing has been performed on macOS 27. Other macOS versions, including macOS 14, are an **unverified preview**. The minimum target is a build setting, not a compatibility guarantee. This is an experimental preview. Check the release notes for the tested systems and known limitations of the package you receive.

A complete installer includes the input method, Python 3.11 runtime, conversion engine, and trained models. After installation, conversion runs locally; typed text is not sent to a server. A development server, virtual environment, and Tailscale connection are not required.

## Download and install

When a public installer is available, download it from the [Hanglish Releases page](https://github.com/LNemo/hanglish-IME/releases). If no installer assets are listed, there is no public download for that version. These steps also apply to a complete installer supplied by the maintainer. An unavailable Releases page does not mean your Mac needs additional setup.

Choose the Apple Silicon `.dmg` asset, such as `Hanglish-0.2.9-arm64.dmg`. The **Source code** ZIP and tar.gz archives are for developers and cannot be installed as an input method. Intel Macs are not supported. You can check your Mac's chip and macOS version in **Apple menu → About This Mac**.

Packages with `development-NONPUBLIC` in the filename are separate development builds with ad-hoc signatures. They are not the public installer. Release-mode packages require Developer ID signing and Apple notarization; their accompanying manifest and notarization log identify the exact artifact. Do not disable macOS security checks to install a package that is blocked; check that you obtained the intended package and report the exact message.

1. Open the DMG and run **Hanglish Installer**.
2. Choose **Install / Update**. On a Mac using Korean, this is shown as **설치 / 업데이트**. Administrator credentials are not required for this user-level installation.
3. If Hanglish is not enabled automatically, open **System Settings → Keyboard → Text Input → Edit → +**, find **Korean → Hanglish**, and add it. Labels may vary with your macOS language and version.
4. Select Hanglish from the input menu in the menu bar. Its icon reads `han`.
5. Open a new TextEdit document and try the first conversion below. Once installation succeeds, close the installer and eject the DMG; Hanglish runs from the installed copy.

The installer uses English by default and includes Korean localization. It installs for the current macOS user. The engine and models are included; you do not need Terminal commands, a Python installation, an Apple Developer account, or a separately started server for normal use.

If Hanglish does not appear, save your work before logging out and back in. On the development Mac, a persistent missing-input-source problem was resolved after restarting and removing/re-adding **Hanglish** in Keyboard settings. That is an observed recovery, not a guarantee for every Mac. Updates from 0.2.0 also changed the input-mode registration format and may require removing/re-adding Hanglish.

The app installs to `~/Library/Input Methods/Hanglish.app`. Updates preserve the previous installation under `~/Library/Application Support/Hanglish/Backups`. The installer attempts to restore it if copying or registration fails; if that recovery also fails, it reports the backup location.

## Select Hanglish and set switching shortcuts

If the input menu is hidden, open **System Settings → Keyboard → Text Input → Edit** and enable **Show Input menu in menu bar**. Add Hanglish there if it is missing, then select it from that menu.

To review or change the switching keys, open **System Settings → Keyboard → Keyboard Shortcuts → Input Sources**. The usual shortcuts are **Control–Space** for the previous input source and **Control–Option–Space** for the next one. Your settings may differ. If a shortcut only toggles ABC and another input source, first select Hanglish directly from the menu and check that it was added successfully. See Apple's [input-source settings](https://support.apple.com/guide/mac-help/mchl84525d76/mac) and [keyboard shortcuts](https://support.apple.com/102650).

Hanglish has no separate settings app yet. Input-source selection and switching shortcuts are configured in macOS. The conversion engine starts automatically when needed.

## Type Korean

For a first test in TextEdit, select Hanglish, type `jeoneun haksaengimnida`, press Enter to request **저는 학생입니다** (“I am a student”), then press Enter again to insert it. The first conversion may take longer while the engine starts. `imnida` can become **입니다**, and `testreul sihaenghanda` can become **test를 시행한다** while preserving the English word.

| Action | Result |
|---|---|
| Type Latin letters | Compose the original text |
| Space | Add a space within the sentence |
| Enter or Tab | Request conversion of the current sentence |
| Enter after candidates appear | Insert the selected candidate |
| Tab / Shift+Tab, Down / Up | Preview the next / previous candidate |
| Click a candidate | Insert that candidate |
| Esc | Keep and insert the original Latin text |
| Backspace | Delete from the original text and continue editing |
| Shift+Enter | Insert the current text and pass Enter to the app |
| Move with arrow keys, use a shortcut, or switch input sources | Finish the current composition and return control to the app |

When candidates are visible, Up/Down navigate candidates as shown above. Other navigation follows the composition behavior of the current app.

Input follows Revised Romanization as a starting point. Pronunciation can correspond to several Korean spellings, so results can be wrong. This tool does not translate English sentences or transcribe speech. It does not yet provide pronunciation guides or translations for candidates; choosing correctly may require help if you cannot read Hangul.

The engine starts automatically on the first conversion, which may take longer. Later requests reuse the same process. After 120 seconds without requests it shuts down and starts again when needed. Duplicate conversion requests are suppressed while work is in progress. Errors preserve the original text; retry with Enter or keep it with Esc.

Compositions are limited to 2,000 characters. Number keys remain part of the text rather than selecting numbered candidates. Editing inside a composition and reconverting individual words are not implemented. Switch to ABC when you want to type a longer passage in English.

## Update or remove

To update, run **Hanglish Installer** from the new DMG and select **Install / Update** (**설치 / 업데이트** in Korean). If Hanglish is selected, the installer switches to ABC before stopping the old input method and replacing it. Save your work before logging out/in if an old process or input-menu entry persists.

To remove it, use **Remove and Save Backup** (**제거 후 보관** in Korean) in the installer. The app is moved into the backup directory, and its input sources are disabled. If an entry remains in System Settings, remove Hanglish from the input-source list there.

The older [install.sh](install.sh) refuses to install a lightweight app without its engine. Use a complete DMG for installation and updates.

## Build and verify

A fresh clone runs the model-free baseline. The full learned engine still requires external/generated data and model weights. Prepare the locked dependencies and import verified assets as shown below; the package command checks [runtime-assets.lock.json](Packaging/runtime-assets.lock.json) before building. Xcode Command Line Tools and Python 3.11 arm64 are required. Public model downloads are not available yet.

Use a separate checkout/environment if you already run an engine from `.venv`:
`uv sync` can remove packages outside the selected dependency set.
From the repository root:

```sh
# Model-free Python checks, using only locked test dependencies.
uv sync --frozen --only-group dev --python 3.11
.venv/bin/python scripts/prepare_runtime_assets.py check --profile baseline
.venv/bin/python -m pytest tests/test_dictionary.py tests/test_romanizer.py tests/test_baseline_assets.py tests/test_manual_diagnostics.py tests/test_prepare_runtime_assets.py tests/test_prepare_english_dictionary.py -q

# Full engine: install dependencies and import from a trusted complete installation.
uv sync --frozen --extra learning --python 3.11
.venv/bin/python scripts/prepare_runtime_assets.py import --from "$HOME/Library/Input Methods/Hanglish.app/Contents/Resources/Engine" --profile package
.venv/bin/python scripts/prepare_runtime_assets.py check --profile package
```

The importer checks file sizes and SHA-256 before copying and refuses conflicting
local files. Model weights are not downloaded automatically. Hashes verify file
identity; they do not grant redistribution rights. The source bundle must match the
current lock exactly. Installations from 0.2.6 or earlier contain different models or metadata and do not
match the current lock, so they cannot be used as the import source for this version.

```sh
# Composition, process-protocol, and type checks; no models needed.
bash native/macos/test.sh

# Build a nonpublic development installer, once all required local assets are available.
bash native/macos/package.sh

# Check representative conversions using an already installed complete bundle.
bash native/macos/test.sh --live "$HOME/Library/Input Methods/Hanglish.app/Contents/Resources"
```

Development mode is the default. `package.sh` builds the input method and installer,
runs checks, and produces these files without overwriting existing artifacts:

```text
dist/Hanglish-0.2.9-arm64-development-NONPUBLIC.dmg
dist/Hanglish-0.2.9-arm64-development-NONPUBLIC.dmg.sha256
dist/Hanglish-0.2.9-arm64-development-NONPUBLIC.dmg.json
```

The JSON manifest records the DMG SHA-256, source commit and clean-worktree state,
the hashed release inputs, version `0.2.9`, build `11`, architecture, minimum deployment
target, and packaging mode. A development manifest marks the artifact as nonpublic and
not publication-ready. The complete package includes Python, so the destination Mac
does not need a separate Python installation.

Release mode is only for a clean worktree on a Mac where the Developer ID Application
certificate and an authenticated `notarytool` keychain profile are already available:

```sh
HANGLISH_PACKAGE_MODE=release \
HANGLISH_SIGNING_IDENTITY='Developer ID Application: Your Name (TEAMID)' \
HANGLISH_TEAM_ID='TEAMID' \
HANGLISH_NOTARY_PROFILE='hanglish-notary' \
bash native/macos/package.sh
```

Release preflight rejects a missing or mismatched identity, team ID, profile, or dirty
worktree. When those checks pass, the script signs with hardened runtime and timestamps,
submits the DMG for notarization, staples the accepted ticket, verifies it with
`hdiutil`, `codesign`, `stapler`, and Gatekeeper, then writes
`Hanglish-0.2.9-arm64.dmg`, its checksum, manifest, and notarization log. These technical checks do not publish a GitHub Release. The manifest identifies the exact source commit and validation gaps.

`test.sh` checks composition state, engine-process handling, and Swift types. `--live` additionally starts the supplied bundled engine, checks representative conversions, and verifies its selected model version. Actual input-source selection, typing, and icon appearance still need UI testing.

## Distribution and compatibility

- The native build targets Apple Silicon with macOS 14 as its minimum deployment target. Local testing was performed on macOS 27. Untested versions, including macOS 14, are offered as an unverified preview; real installation and input-source use on those versions remain unverified.
- macOS or individual apps may restrict input methods in password fields and special text controls.
- Developer ID signing and notarization are required before a general installer release. The current development artifact is ad-hoc signed, marked nonpublic, and must not be published as the public release.
- Preserve bundled third-party licenses and notices. For the KoELECTRA base model, Hanglish uses the official project’s Apache-2.0 license as its licensing basis; see the third-party notices for the pinned source and training-data terms.

Original Hanglish code uses [CC BY-NC 4.0](../../LICENSE). The bundled sentence-ranker weight and two KLUE-trained CTC weights are distributed separately under [CC BY-SA 4.0](../../licenses/hanglish-models/LICENSE.txt); their exact paths, hashes and attribution are in the [model notice](../../licenses/hanglish-models/NOTICE.txt). Those three weights permit commercial reuse subject to attribution and ShareAlike, without changing the noncommercial license on Hanglish code. See the [third-party notices](../../THIRD_PARTY_NOTICES.md) for the remaining components.
