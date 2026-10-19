#!/bin/bash
# Rebuild the optional English dictionary from a pinned SCOWL release.
set -euo pipefail
task_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$task_dir/prepare_english_dictionary.py" "$@"
