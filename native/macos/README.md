# Hanglish for macOS

[English](README.md) · [한국어](README.ko.md) · [Project overview](../../README.md)

Hanglish is for people who can speak Korean but find it difficult to write in Hangul. Type Korean pronunciation with Latin letters, request Korean candidates, and choose the spelling you want.

The current prototype is **0.2.5, for Apple Silicon (arm64) and macOS 14 or later**. There is no published GitHub Release yet. The instructions below apply if you have received a development installer.

A complete installer includes the input method, Python 3.11 runtime, conversion engine, and trained models. After installation, conversion runs locally; typed text is not sent to a server. A development server, virtual environment, and Tailscale connection are not required.

## Install a development build

The current local build produces:

```text
dist/Hanglish-0.2.5-arm64.dmg
```

1. Open the DMG and run **Hanglish Installer**.
2. Choose **설치 / 업데이트** (Install / Update). Administrator credentials are not required for this user-level installation.
3. If Hanglish is not enabled automatically, open **System Settings → Keyboard → Text Input → Edit → +**, find **Korean → Hanglish**, and add it. Labels may vary with your macOS language and version.
4. Select Hanglish from the input menu in the menu bar. Its icon reads `han`.

The installer UI is currently in Korean. The package has a development ad-hoc signature; Developer ID signing and Apple notarization have not been completed.

If Hanglish does not appear, save your work before logging out and back in. On the development Mac, a persistent missing-input-source problem was resolved after restarting and removing/re-adding **Hanglish** in Keyboard settings. That is an observed recovery, not a guarantee for every Mac. Updates from 0.2.0 also changed the input-mode registration format and may require removing/re-adding Hanglish.

The app installs to `~/Library/Input Methods/Hanglish.app`. Updates preserve the previous installation under `~/Library/Application Support/Hanglish/Backups`. The installer attempts to restore it if copying or registration fails; if that recovery also fails, it reports the backup location.

## Type Korean

For example, type `jeoneun haksaengimnida`, press Enter to request **저는 학생입니다** (“I am a student”), then press Enter again to insert it. `imnida` can become **입니다**, and `testreul sihaenghanda` can become **test를 시행한다** while preserving the English word.

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

To update, run **Hanglish Installer** from the new DMG and select **설치 / 업데이트** (Install / Update). If Hanglish is selected, the installer switches to ABC before stopping the old input method and replacing it. Save your work before logging out/in if an old process or input-menu entry persists.

To remove it, use **제거 후 보관** (Remove and Archive) in the installer. The app is moved into the backup directory, and its input sources are disabled. If an entry remains in System Settings, remove Hanglish from the input-source list there.

The older [install.sh](install.sh) refuses to install a lightweight app without its engine. Use a complete DMG for installation and updates.

## Build and verify

A fresh clone lacks model weights and several required data files. Building a complete installer is **not yet reproducible from the repository alone**. Required inputs are listed in [engine-files.json](Packaging/engine-files.json); some currently live in local experiment directories. Xcode Command Line Tools, the project's Python 3.11 arm64 build environment, and those assets are required.

From the repository root:

```sh
# Composition, process-protocol, and type checks; no models needed.
bash native/macos/test.sh

# Build a complete installer, once all required local assets are available.
bash native/macos/package.sh

# Check representative conversions using an already installed complete bundle.
bash native/macos/test.sh --live "$HOME/Library/Input Methods/Hanglish.app/Contents/Resources"
```

`package.sh` builds the input method and installer, runs checks, and produces `dist/Hanglish-0.2.5-arm64.dmg` with a SHA-256 file. The complete package includes Python, so the destination Mac does not need a separate Python installation.

`test.sh` checks composition state, engine-process handling, and Swift types. `--live` additionally starts the supplied bundled engine, checks representative conversions, and verifies its selected model version. Actual input-source selection, typing, and icon appearance still need UI testing.

## Distribution and compatibility

- The native build targets Apple Silicon and macOS 14 or later.
- macOS or individual apps may restrict input methods in password fields and special text controls.
- Developer ID signing and notarization are planned before a general installer release; they are not prerequisites for publishing source code.
- Preserve bundled third-party licenses and notices. Public redistribution of trained models still needs a separate review.
- See the [package record](../../docs/macos-package.md), [app compatibility checks](../../docs/macos-ime-validation.md), and [platform comparison](../../docs/ime-platform-comparison.md). These detailed records are currently in Korean.
