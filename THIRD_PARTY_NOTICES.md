# Third-party software, dictionaries and models

Project licensing does not replace the terms of external software, data or model
weights. The installation package contains separately licensed components;
Hanglish’s CC BY-NC 4.0 applies to original project code, not to every bundled file.

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
  Git. The official KoELECTRA v3 project release links to the exact base model
  used here. Its [Apache-2.0 project license](licenses/koelectra/PROJECT-LICENSE.txt)
  and [pinned provenance](licenses/koelectra/SOURCE.json) are preserved. The pinned
  Hugging Face model revision has no separate license file or model-card license
  metadata. Hanglish uses the official project’s Apache-2.0 license as the
  licensing basis for this base model, based on that direct release link. This
  records the project’s distribution policy; it does not assert that a separate
  license grant was obtained from the Hugging Face repository.
- KLUE v1.1: packaged sentence-ranker and CTC training metadata trace to
  CC BY-SA 4.0 data. Credit KLUE Team, *KLUE: Korean Language Understanding
  Evaluation* (2021), https://arxiv.org/abs/2105.09680. Extracted/normalized
  training corpora retain applicable upstream terms; the original project's
  CC BY-NC license does not replace them. As a conservative distribution policy,
  Hanglish separately offers its rights in the sentence-ranker and two retrained
  CTC weight files under **CC BY-SA 4.0**. The exact three files and their hashes
  are listed in [the model license manifest](licenses/hanglish-models/models.json);
  see [the model notice](licenses/hanglish-models/NOTICE.txt) and
  [full license](licenses/hanglish-models/LICENSE.txt). Those weights permit
  commercial reuse under their BY-SA terms; Hanglish’s NC condition does not
  apply to them. Applicable upstream base-model licenses remain separate.
  This policy does not assert that every trained model is legally an adaptation.
- Packaged Python and native dependencies retain individual licenses. See the
  [package notices](native/macos/Packaging/THIRD_PARTY_NOTICES.md) and the generated
  `Resources/Licenses/python-distributions.json` in a development package.

The [runtime asset lock](native/macos/Packaging/runtime-assets.lock.json) verifies
file identity only. Its distribution labels identify review status, not new grants.
