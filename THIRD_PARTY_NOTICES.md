# Third-party software, dictionaries and models

Project licensing does not replace the terms of external software, data or model
weights. This inventory records known sources; it is not a declaration that every
asset is cleared for public redistribution.

## English dictionary — SCOWL / English Speller Database

`data/english_dict.txt` now contains **89,140 words** from the official standard
US English SCOWL/ESDB 2026.02.25 expanded plain list. The selected native macOS
engine does not bundle this optional legacy-engine file.

The [upstream permission notice](licenses/scowl/Copyright) allows use, copying,
modification, distribution and sale, subject to preserving copyright and permission
notices. The full upstream notice is retained unchanged. This list retains those
terms; Hanglish's CC BY-NC license does not replace them.

[SOURCE.json](licenses/scowl/SOURCE.json) pins the official release revision,
source/output SHA-256 hashes and transformations: lowercase, retain ASCII-letter
words, deduplicate, sort and normalize line endings. `scripts/download_data.sh`
invokes the standard-library generator, which verifies the source and output
before atomically replacing the dictionary. No Hunspell dependency is needed.

Official sources: [release/download guide](https://wordlist.aspell.net/dicts/),
[pinned expanded list](https://github.com/en-wl/wordlist-diff/blob/71d7dd07676edb60ade43552e10b41314b7e9287/en_US.txt),
[pinned copyright](https://github.com/en-wl/wordlist-diff/blob/71d7dd07676edb60ade43552e10b41314b7e9287/Copyright).

## Other data and model components

- Small Hanglish seed dictionaries: see [data/README.md](data/README.md).
- MeCab Korean dictionary: Apache-2.0; preserve `COPYING` and `SOURCE.json` with
  imported files. The tracked derived predicate inventory has adjacent
  `hanglish/core/data/mecab-ko-dic.COPYING` and embedded source/version information.
- wordfreq 3.1.1: software Apache-2.0, frequency data CC BY-SA 4.0 with source
  attributions. Preserve the [upstream notice](native/macos/Packaging/wordfreq-NOTICE.md)
  and [license texts](native/macos/Packaging/wordfreq-LICENSE.txt).
- Trained rankers and transducers: weights and training datasets stay outside
  Git. KoELECTRA source licensing does not by itself establish permission for
  all base weights, training sources or derived checkpoints. Their public
  distribution remains pending review.
- KLUE v1.1: packaged sentence-ranker and CTC training metadata trace to
  CC BY-SA 4.0 data. Credit KLUE Team, *KLUE: Korean Language Understanding
  Evaluation* (2021), https://arxiv.org/abs/2105.09680. Extracted/normalized
  training corpora retain applicable upstream terms; the original project's
  CC BY-NC license does not replace them. This lineage alone does not determine
  whether trained weights are legally adaptations.
- Packaged Python and native dependencies retain individual licenses. See the
  [package notices](native/macos/Packaging/THIRD_PARTY_NOTICES.md) and the generated
  `Resources/Licenses/python-distributions.json` in a development package.

The [runtime asset lock](native/macos/Packaging/runtime-assets.lock.json) verifies
file identity only. Its distribution labels identify review status, not new grants.
