# Dictionary and training data

The source repository includes these small Hanglish seed dictionaries:

| File | Purpose / provenance |
| --- | --- |
| `workbench_lexicon.json` | Hand-authored romanization/spelling examples for the shared baseline engine. It is also used by the native engine service. |
| `morphology_lexicon.json` | Independently authored seed vocabulary and productive suffixes; provenance is embedded in the file. |
| `common_english.txt`, `it_terms.txt` | Small project-maintained English/technical word lists for mixed text. |

These are development inputs, not an independent evaluation corpus. Do not
interpret accuracy on their examples as held-out generalization.

The optional legacy `english_dict.txt` is a separate external word list; see
[third-party notices](../THIRD_PARTY_NOTICES.md). It is not included in the native
package manifest. MeCab downloads, generated training lexicons, corpora and model
weights remain local. The selected native engine needs them; follow the
[development setup](../docs/development-setup.md) to verify and import exact assets.
