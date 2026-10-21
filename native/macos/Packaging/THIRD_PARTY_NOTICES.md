# Hanglish macOS package — third-party notices

These notices apply to both development and signed release-candidate packages.
Original Hanglish code uses CC BY-NC 4.0. Preserve the LICENSE
and NOTICE copied into Resources/Licenses/Hanglish; third-party terms remain separate.
External model and data terms remain separate from Hanglish’s own code license.
The licensing basis for the KoELECTRA base model is recorded below. Ranker training and base-model provenance remain in Engine/checkpoints/*/metadata.json.
Retrained transducer provenance accompanies the selected weights under Engine/.omx/artifacts/.
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
  separate license file or model-card license metadata. Hanglish uses the official
  project’s Apache-2.0 license as the licensing basis for this base model, based
  on the direct release link. This is the project’s distribution policy, not a
  claim that a separate license grant was obtained from the model repository.
- KLUE v1.1 training provenance: the packaged sentence ranker and both CTC models
  were trained using KLUE-derived data; the original corpus is not bundled.
  Credit KLUE Team, *KLUE: Korean Language Understanding Evaluation* (2021),
  https://arxiv.org/abs/2105.09680. Upstream data terms are CC BY-SA 4.0:
  https://github.com/KLUE-benchmark/KLUE/blob/3efd98708a40ff49251fddde35453f8fbb11f536/License.md
  Preparation included text-window extraction/normalization and romanization.
  This attribution does not settle the distribution terms of trained weights.

No Qwen/EXAONE/Roberta weights, datasets, developer environments, or web server are
included. Version 0.2.7 selects retrained phonetic and CTC transducers based on the
corrected romanization rules. The existing semantic and sentence rankers are retained.
The engine manifest records the selected model paths and hashes; candidate-only flags
in frozen training metadata describe the training run before release selection.

## Distribution boundary

The adjacent artifact manifest records whether this is a nonpublic ad-hoc
development build or a Developer ID signed and notarized release candidate.
Signing and notarization do not grant rights to third-party models or data.
These notes do not replace the upstream licenses supplied in the package.
