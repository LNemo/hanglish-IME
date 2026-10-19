# Third-party software, dictionaries and models

Project licensing does not replace the terms of external software, data or model
weights. This inventory records known sources; it is not a declaration that every
asset is cleared for public redistribution.

See the [2026-10-19 redistribution review](docs/redistribution-audit-20261019.md)
for the exact selected engine, training lineage, additional upstream evidence,
and remaining decisions.

## Legacy English dictionary

`data/english_dict.txt` is the legacy optional English word list. The selected
native macOS engine does not bundle this file. Its unique word set matches
`words_alpha.txt` in [dwyl/english-words at revision
20f5cc9](https://github.com/dwyl/english-words/tree/20f5cc9b3f0ccc8ce45d814c532b7c2031bba31c).
The local file has different ordering/formatting; the audit does not establish
which historical revision originally produced it.

Upstream declares **Unlicense**, not MIT. The unmodified license text is in
[licenses/dwyl-english-words/UNLICENSE](licenses/dwyl-english-words/UNLICENSE),
and [SOURCE.json](licenses/dwyl-english-words/SOURCE.json) records the pinned
URLs, upstream hashes and existing local file hash.

However, the [upstream README](https://github.com/dwyl/english-words/blob/20f5cc9b3f0ccc8ce45d814c532b7c2031bba31c/README.md)
attributes the underlying list to Infochimps and says “Copyright still belongs to
them.” Therefore the original dataset's redistribution basis remains unresolved.
Before making this repository public, resolve that provenance or replace/remove
the list, including historical copies as appropriate. It is retained in this
private repository for existing behavior while that decision is outstanding.

The pinned repository also contains Moby provenance metadata recording a
public-domain grant. This supports a possible underlying source, but the audit
did not establish the complete transformation/addition chain to this word list.
See the review for the source links and jurisdiction qualification.

`scripts/download_data.sh` downloads a fixed revision, verifies its SHA-256,
normalizes line endings/order, and replaces the local file only after successful
verification. It does not resolve the upstream rights question.

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
  whether trained weights are legally adaptations; see the review.
- Packaged Python and native dependencies retain individual licenses. See the
  [package notices](native/macos/Packaging/THIRD_PARTY_NOTICES.md) and the generated
  `Resources/Licenses/python-distributions.json` in a development package.

The [runtime asset lock](native/macos/Packaging/runtime-assets.lock.json) verifies
file identity only. Its distribution labels identify review status, not new grants.
