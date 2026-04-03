#!/bin/bash
# Download public datasets for Hanglish IME
# English word list + Korean corpus source

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../data"

mkdir -p "$DATA_DIR"

echo "=== Downloading English word list ==="
# Source: dwyl/english-words (MIT license)
curl -sL "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt" \
  | tr '[:upper:]' '[:lower:]' \
  | sort -u \
  > "$DATA_DIR/english_dict.txt"

WORD_COUNT=$(wc -l < "$DATA_DIR/english_dict.txt")
echo "Downloaded $WORD_COUNT English words to data/english_dict.txt"

echo ""
echo "=== Korean corpus ==="
echo "For Korean n-gram data, run: python scripts/build_ngram.py"
echo "This will generate data/unigram.json and data/bigram.json"
echo ""
echo "Done!"
