# Hanglish IME

**Speak Korean. Type it with Latin letters.**

[English](README.md) · [한국어](README.ko.md) · [日本語](README.ja.md) · [简体中文](README.zh-CN.md) · [Español](README.es.md)

Hanglish is an experimental input method for people who **can speak Korean but find it difficult to write in Hangul**. Type the Korean words you know using Latin letters, and Hanglish suggests how to write them in Hangul. You do not need to learn a Korean keyboard layout to get started.

For example, if you can say *jeoneun haksaengimnida* (“I am a student”), you can type that pronunciation instead of entering each Hangul letter yourself.

| What you type | Suggested Korean | Meaning |
|---|---|---|
| `jeoneun haksaengimnida` | 저는 학생입니다 | I am a student. |
| `imnida` | 입니다 | A formal polite form of “to be.” |
| `testreul sihaenghanda` | test를 시행한다 | Carry out a test; keep the English word `test`. |

These examples are covered by the bundled-engine smoke tests. They are examples of supported inputs, not a promise that every sentence will be correct.

## How it works

1. Select **Hanglish** from the macOS input menu.
2. Type Korean pronunciation in Latin letters, including spaces between words.
3. Press **Enter** or **Tab** to request Korean candidates.
4. Use **Tab / Shift+Tab** or **Down / Up** to move between candidates. Press **Enter** to insert your choice.
5. Press **Esc** if you want to keep the original Latin text.

Input is based on Revised Romanization of Korean. Familiar informal spellings may not always work. This is a typing tool: enter Korean words, rather than an English sentence to translate. It does not listen to your voice.

Korean pronunciation does not uniquely determine spelling. Hanglish tries to recover the intended spelling, but it can suggest the wrong word. Choosing among candidates still requires some recognition of written Korean; someone who cannot read Hangul at all may need help checking the result. Reading aids and translations for candidates are not implemented yet.

## Availability

**Current native prototype: macOS 0.2.9 (build 11), Apple Silicon, with a deployment target of macOS 14 or later.** Local testing has been performed on macOS 27. Other macOS versions, including macOS 14, are offered as an **unverified preview**; the deployment target does not guarantee compatibility. Windows, Android, and iOS input methods are not implemented yet.

Download the signed and notarized macOS preview from [GitHub Releases](https://github.com/LNemo/hanglish-IME/releases/tag/v0.2.9). Original Hanglish code uses noncommercial terms; the three KLUE-trained weights bundled with the installer have a separate CC BY-SA 4.0 notice.

For installation and input-source setup, follow the [macOS installation guide](native/macos/README.md) ([한국어](native/macos/README.ko.md)). A complete installer bundles the conversion engine, models, and Python runtime. Once installed, conversion runs locally and does not send your typed text to a server. You do not need a development server, Tailscale, or a separate Python installation to use that package.

The installer uses English by default and includes Korean localization. Development packages use an ad-hoc signature. Release-mode packages require Developer ID signing and Apple notarization; check the artifact manifest and notarization log for the specific package. App compatibility is still being tested.

## Limitations

- This is an experimental input method, not a finished spelling or grammar checker.
- Sound changes, ambiguous words, unfamiliar vocabulary, and mixed Korean–English text can produce incorrect candidates.
- The selected engine matched the first answer on 988 of 1,000 development-exposed regression sentences; an accepted answer appeared in its top five on 1,000. These are **not independent real-world accuracy measurements**. There were 12 remaining first-choice errors.
- Editing inside a composition and reconverting individual words are not supported yet.
- The experimental web interface is local development tooling and is excluded from this repository's distribution scope.

## For developers

The repository contains the conversion source, native macOS input method, build tools, and tests. It does **not** include model weights, downloaded or generated datasets, local experiment artifacts, or installer images.

**A fresh clone can run the model-free baseline, but needs additional assets for the complete learned engine and installer.** Follow [development setup](native/macos/README.md#build-and-verify) for locked dependencies, asset checks and importing from a trusted local installation. [The asset lock](native/macos/Packaging/runtime-assets.lock.json) pins required files by SHA-256. Import requires a bundle that exactly matches the current lock; installations from 0.2.6 or earlier do not match. Models are included in the macOS installer; standalone model downloads are not provided.

On a Mac with Xcode Command Line Tools, run the model-free native checks from the repository root:

```sh
bash native/macos/test.sh
```

These check composition state, engine-process protocol handling, and Swift types. They do not prove that input-source selection, actual typing, or model conversion works. Manual keyboard diagnostics have moved to `scripts/manual/` and require `--run`; importing them sends no keystrokes. Use the fresh-clone test commands in [development setup](native/macos/README.md#build-and-verify); research tests may require additional assets.

## License

Original Hanglish code and authored seed data use [CC BY-NC 4.0](LICENSE). Sharing and adaptation require attribution and a noncommercial purpose; indicate changes when sharing modified material. This is source-available software with noncommercial terms, not OSI-defined open source.

The bundled sentence-ranker weight and the two KLUE-trained CTC weights are distributed separately under [CC BY-SA 4.0](licenses/hanglish-models/LICENSE.txt), with the exact files, hashes and KLUE attribution listed in the [model notice](licenses/hanglish-models/NOTICE.txt). Those weights may be reused commercially subject to attribution and ShareAlike; this does not change the noncommercial license on Hanglish code. Other external components retain their respective terms. See the [third-party notices](THIRD_PARTY_NOTICES.md).
