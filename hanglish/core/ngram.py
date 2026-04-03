"""N-gram probability model for Korean syllable sequences."""

import json
import math
from pathlib import Path


class NgramModel:
    def __init__(
        self,
        unigram_path: str | Path | None = None,
        bigram_path: str | Path | None = None,
        alpha: float = 1.0,
    ):
        data_dir = Path(__file__).parent.parent.parent / "data"
        if unigram_path is None:
            unigram_path = data_dir / "unigram.json"
        if bigram_path is None:
            bigram_path = data_dir / "bigram.json"

        with open(unigram_path, "r", encoding="utf-8") as f:
            self._unigrams: dict[str, int] = json.load(f)

        with open(bigram_path, "r", encoding="utf-8") as f:
            self._bigrams: dict[str, int] = json.load(f)

        self._alpha = alpha
        self._total_unigrams = sum(self._unigrams.values())
        self._vocab_size = len(self._unigrams)

    def prob_unigram(self, syllable: str) -> float:
        count = self._unigrams.get(syllable, 0)
        prob = (count + self._alpha) / (self._total_unigrams + self._alpha * self._vocab_size)
        return math.log(prob)

    def prob_bigram(self, prev: str, curr: str) -> float:
        bigram_key = f"{prev}|{curr}"
        bigram_count = self._bigrams.get(bigram_key, 0)
        prev_count = self._unigrams.get(prev, 0)
        prob = (bigram_count + self._alpha) / (prev_count + self._alpha * self._vocab_size)
        return math.log(prob)

    def score_sequence(self, syllables: list[str]) -> float:
        if not syllables:
            return 0.0
        score = self.prob_unigram(syllables[0])
        for i in range(1, len(syllables)):
            score += self.prob_bigram(syllables[i - 1], syllables[i])
        return score
