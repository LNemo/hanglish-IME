# Hanglish macOS package — third-party notices

These notices apply to both development and signed release-candidate packages.
Original Hanglish code uses CC BY-NC 4.0. Preserve the LICENSE
and NOTICE copied into Resources/Licenses/Hanglish; third-party terms remain separate.
Do not infer permission to redistribute trained model weights from the software
licenses alone. Training and base-model provenance remain in Engine/checkpoints/*/metadata.json.
Machine-local build paths in project metadata are normalized to repository-relative
provenance. Model weights and upstream attribution remain unchanged.

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
- MeCab Korean dictionary subsets: the selected CSV files under
  `Engine/data/external/mecab-ko-dic/`, with adjacent Apache-2.0 `COPYING` and
  `SOURCE.json` preserved. See engine-files.json for the complete selected list.
  External evaluation/training corpora are not included.
- KoELECTRA-derived rankers: the base repository is
  `monologg/koelectra-small-v3-discriminator`. The official v3 project release links
  to that model; its Apache-2.0 project license and pinned source record are included
  under `Resources/Licenses/KoELECTRA-project`. The pinned model revision has no
  separate license file or model-card license metadata. This leaves uncertainty
  about weight-license scope; it is not a stated redistribution prohibition.
- KLUE v1.1 training provenance: the packaged sentence ranker and both CTC models
  were trained using KLUE-derived data; the original corpus is not bundled.
  Credit KLUE Team, *KLUE: Korean Language Understanding Evaluation* (2021),
  https://arxiv.org/abs/2105.09680. Upstream data terms are CC BY-SA 4.0:
  https://github.com/KLUE-benchmark/KLUE/blob/3efd98708a40ff49251fddde35453f8fbb11f536/License.md
  Preparation included text-window extraction/normalization and romanization.
  This attribution does not settle the distribution terms of trained weights.

No Qwen/EXAONE/Roberta weights, datasets, developer environments, or web server are
included. The current trained Hanglish engine is retained without retraining.

## Distribution boundary

The adjacent artifact manifest records whether this is a nonpublic ad-hoc
development build or a Developer ID signed and notarized release candidate.
Signing and notarization do not grant rights to third-party models or data.
These notes do not replace the upstream licenses supplied in the package.
