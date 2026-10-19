#!/bin/bash
# Optional legacy English dictionary; see THIRD_PARTY_NOTICES.md.
# Upstream declares Unlicense but also retains an Infochimps copyright note.
set -euo pipefail
task_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
data_dir="$task_dir/../data"
mkdir -p "$data_dir"
staging="$(mktemp -d "$data_dir/.english-download.XXXXXX")"
trap 'rm -rf "$staging"' EXIT
revision=20f5cc9b3f0ccc8ce45d814c532b7c2031bba31c
curl --fail --location --silent --show-error --proto '=https' --proto-redir '=https' \
  "https://raw.githubusercontent.com/dwyl/english-words/$revision/words_alpha.txt" \
  -o "$staging/words_alpha.txt"
python3 - "$staging/words_alpha.txt" "$staging/english_dict.txt" <<'VERIFY'
import hashlib
from pathlib import Path
import sys
raw = Path(sys.argv[1]).read_bytes()
expected = '3ed0c94610d8bcf7c11bbb49c56aa49c7234d32b66824df91f554169e572da48'
if hashlib.sha256(raw).hexdigest() != expected:
    raise SystemExit('Checksum mismatch; existing dictionary was not changed.')
words = sorted(set(raw.decode('ascii').lower().splitlines()))
if not words or any(not word.isalpha() for word in words):
    raise SystemExit('Unexpected word list format; existing dictionary was not changed.')
Path(sys.argv[2]).write_text('\n'.join(words) + '\n', encoding='ascii')
print(f'Verified {len(words)} words. See THIRD_PARTY_NOTICES.md for provenance limitations.')
VERIFY
mv "$staging/english_dict.txt" "$data_dir/english_dict.txt"
