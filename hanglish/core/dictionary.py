"""English dictionary loader for language discrimination."""

from pathlib import Path


class EnglishDictionary:
    def __init__(self, dict_path: str | Path | None = None):
        data_dir = Path(__file__).parent.parent.parent / "data"
        if dict_path is None:
            dict_path = data_dir / "english_dict.txt"
        else:
            dict_path = Path(dict_path)

        self._words: set[str] = set()
        self._common_words: set[str] = set()

        # Load main dictionary
        self._load_file(dict_path)

        # Load IT terms and user dictionary
        for extra in ("it_terms.txt", "user_dict.txt"):
            self._load_file(data_dir / extra)

        # Load common/high-frequency English words
        common_path = data_dir / "common_english.txt"
        if common_path.exists():
            with open(common_path, "r", encoding="utf-8") as f:
                for line in f:
                    word = line.strip().lower()
                    if word:
                        self._common_words.add(word)
                        self._words.add(word)

    def _load_file(self, path: Path) -> None:
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip().lower()
                if word:
                    self._words.add(word)

    def is_english(self, word: str) -> bool:
        return word.lower() in self._words

    def is_common_english(self, word: str) -> bool:
        """Check if word is a high-frequency English word."""
        return word.lower() in self._common_words

    def prefix_match(self, text: str) -> list[str]:
        """Find English words that are prefixes of text, longest first."""
        text_lower = text.lower()
        matches: list[str] = []
        for length in range(len(text_lower), 0, -1):
            candidate = text_lower[:length]
            if candidate in self._words and len(candidate) >= 2:
                matches.append(candidate)
        return matches
