# Hanglish private development package — third-party notices

This package is a local development build, not a license-cleared public release.
Do not infer permission to redistribute trained model weights from the software
licenses alone. Training and base-model provenance remain in Engine/checkpoints/*/metadata.json.
Some metadata records historical build paths; these are provenance, not runtime dependencies.

## Included software and data

- CPython 3.11 from uv's python-build-standalone distribution. Preserve the Python
  license files under Resources/Python, including bundled library licenses.
  https://github.com/astral-sh/python-build-standalone
- Installed Python distributions and exact versions are listed in
  `python-distributions.json`. Their full METADATA and license directories remain
  in `Resources/Python/lib/python3.11/site-packages/*.dist-info/`.
- PyTorch: BSD-style license, preserved in its distribution metadata.
  https://github.com/pytorch/pytorch/blob/main/LICENSE
- Transformers: Apache-2.0; NumPy and other components retain their own license notices.
- wordfreq: code Apache-2.0; frequency data CC BY-SA 4.0 with additional attribution
  requirements described in the upstream NOTICE. Preserve all bundled data notices.
  https://github.com/rspeer/wordfreq/blob/master/NOTICE.md
- MeCab Korean counter data: `Engine/data/external/mecab-ko-dic/NNBC.csv`, with the
  adjacent `COPYING` and `SOURCE.json` preserved. This package does not include the
  complete dictionary or external evaluation/training corpora.
- KoELECTRA-derived rankers: the base repository is
  `monologg/koelectra-small-v3-discriminator`. Checkpoint metadata explicitly notes
  that repository license and checkpoint license metadata are not interchangeable.
  Existing model cards/provenance are preserved; public model distribution needs review.

No Qwen/EXAONE/Roberta weights, datasets, developer environments, or web server are
included. The current trained Hanglish engine is retained without retraining.

## Distribution boundary

The app is ad-hoc signed for local development. It is not Developer ID signed or
notarized. Public distribution requires appropriate signing/notarization and an
audit of software, data, and model redistribution terms. These notes do not replace
the upstream licenses supplied in the package.
