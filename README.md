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

**Current native prototype: macOS 0.2.5, Apple Silicon, macOS 14 or later.** Windows, Android, and iOS input methods are not implemented yet.

There is **no published GitHub Release yet**. The repository is currently private, and the project license and public model distribution arrangements are still being prepared.

If you have received the development installer, follow the [macOS installation guide](native/macos/README.md) ([한국어](native/macos/README.ko.md)). A complete installer bundles the conversion engine, models, and Python runtime. Once installed, conversion runs locally and does not send your typed text to a server. You do not need a development server, Tailscale, or a separate Python installation to use that package.

The development installer uses an ad-hoc signature. Developer ID signing and Apple notarization have not been completed, and app compatibility is still being tested.

## Limitations

- This is an experimental input method, not a finished spelling or grammar checker.
- Sound changes, ambiguous words, unfamiliar vocabulary, and mixed Korean–English text can produce incorrect candidates.
- The selected engine matched the first answer on 954 of 1,000 development-exposed regression sentences; an accepted answer appeared in its top five on 975. These are **not independent real-world accuracy measurements**. There were 46 remaining first-choice errors.
- Editing inside a composition and reconverting individual words are not supported yet.
- The experimental web interface is local development tooling and is excluded from this repository's distribution scope.

## For developers

The repository contains the conversion source, native macOS input method, build tools, tests, and development notes. It does **not** include model weights, downloaded or generated datasets, local experiment artifacts, or installer images.

**A fresh clone cannot yet run the complete learned engine or build the standalone installer.** Required assets are listed in [the engine file manifest](native/macos/Packaging/engine-files.json); some still use local experiment paths. A reproducible asset download and verification process is needed before public distribution.

On a Mac with Xcode Command Line Tools, run the model-free native checks from the repository root:

```sh
bash native/macos/test.sh
```

These check composition state, engine-process protocol handling, and Swift types. They do not prove that input-source selection, actual typing, or model conversion works. Some files under `tests/` are manual keyboard diagnostics that run when imported, so **do not run the entire pytest suite indiscriminately**. Those diagnostics need to be separated before a general test command is provided.

## More information

Most detailed development records are currently in Korean:

- [macOS installation and development (English)](native/macos/README.md)
- [Package structure (한국어)](docs/macos-package.md)
- [GitHub and Release preparation (한국어)](docs/github-release.md)
- [What works and what needs improvement (한국어)](docs/development-retrospective.md)
- [Training and data provenance (한국어)](docs/neural-training.md)
- [Platform comparison (한국어)](docs/ime-platform-comparison.md)

English is the default README. Translations should keep the same supported platforms, limitations, and release status as this file.
