"""Small character-to-jamo transducer for phonetic Hangul candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from hanglish.core.phonology import _INSERTION_EXCEPTIONS, romanize_word


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "phonetic_transducer.pt"
ROMAN_RE = re.compile(r"[a-z]+\Z")

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
NO_CODA = "<none>"
INPUT_TOKENS = [PAD, BOS, EOS, "<unk>"] + list("abcdefghijklmnopqrstuvwxyz")

ONSETS = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
VOWELS = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
CODAS = [NO_CODA, "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ",
         "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ",
         "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
OUTPUT_TOKENS = [PAD, BOS, EOS] + list(dict.fromkeys(ONSETS + VOWELS + CODAS))

INPUT_TO_ID = {token: index for index, token in enumerate(INPUT_TOKENS)}
OUTPUT_TO_ID = {token: index for index, token in enumerate(OUTPUT_TOKENS)}

_ONSET_IDS = tuple(OUTPUT_TO_ID[token] for token in ONSETS)
_VOWEL_IDS = tuple(OUTPUT_TO_ID[token] for token in VOWELS)
_CODA_IDS = tuple(OUTPUT_TO_ID[token] for token in CODAS)


@dataclass(frozen=True)
class TransducerConfig:
    embedding_dim: int = 80
    encoder_hidden: int = 160
    decoder_hidden: int = 320
    dropout: float = 0.1
    max_roman_length: int = 48


def hangul_to_jamo(text: str) -> list[str]:
    """Decompose precomposed Hangul into fixed onset/vowel/coda triples."""
    tokens: list[str] = []
    for char in text:
        if not "가" <= char <= "힣":
            raise ValueError("target must contain only precomposed Hangul syllables")
        value = ord(char) - 0xAC00
        onset, rest = divmod(value, 21 * 28)
        vowel, coda = divmod(rest, 28)
        tokens.extend((ONSETS[onset], VOWELS[vowel], CODAS[coda]))
    return tokens


def jamo_to_hangul(tokens: list[str]) -> str | None:
    """Compose strict onset/vowel/coda triples, or return ``None``."""
    if not tokens or len(tokens) % 3:
        return None
    output = []
    for offset in range(0, len(tokens), 3):
        onset, vowel, coda = tokens[offset:offset + 3]
        try:
            value = ONSETS.index(onset) * 21 * 28
            value += VOWELS.index(vowel) * 28 + CODAS.index(coda)
        except ValueError:
            return None
        output.append(chr(0xAC00 + value))
    return "".join(output)


def _allowed_output_ids(generated: int) -> tuple[int, ...]:
    """Return structurally valid next tokens after ``generated`` jamo."""
    phase = generated % 3
    if phase == 0:
        ids = _ONSET_IDS
        if generated:
            ids += (OUTPUT_TO_ID[EOS],)
        return ids
    if phase == 1:
        return _VOWEL_IDS
    return _CODA_IDS


class _Network(nn.Module):
    def __init__(self, config: TransducerConfig):
        super().__init__()
        self.config = config
        self.input_embedding = nn.Embedding(len(INPUT_TOKENS), config.embedding_dim,
                                            padding_idx=INPUT_TO_ID[PAD])
        self.output_embedding = nn.Embedding(len(OUTPUT_TOKENS), config.embedding_dim,
                                             padding_idx=OUTPUT_TO_ID[PAD])
        self.encoder = nn.GRU(config.embedding_dim, config.encoder_hidden,
                              batch_first=True, bidirectional=True)
        self.decoder = nn.GRU(config.embedding_dim + config.decoder_hidden,
                              config.decoder_hidden, batch_first=True)
        self.dropout = nn.Dropout(config.dropout)
        self.output = nn.Linear(config.decoder_hidden * 2, len(OUTPUT_TOKENS))

    def encode(self, source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = source.ne(INPUT_TO_ID[PAD])
        embedded = self.dropout(self.input_embedding(source))
        lengths = mask.sum(dim=1).cpu()
        packed = pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        packed_encoded, hidden = self.encoder(packed)
        encoded, _ = pad_packed_sequence(
            packed_encoded, batch_first=True, total_length=source.shape[1]
        )
        initial = torch.cat((hidden[-2], hidden[-1]), dim=-1).unsqueeze(0)
        return encoded, initial, mask

    def decode_step(
        self,
        token: torch.Tensor,
        hidden: torch.Tensor,
        encoded: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = hidden[-1].unsqueeze(2)
        scores = torch.bmm(encoded, query).squeeze(2)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), encoded)
        embedded = self.output_embedding(token).unsqueeze(1)
        decoded, hidden = self.decoder(torch.cat((embedded, context), dim=-1), hidden)
        logits = self.output(torch.cat((decoded, context), dim=-1)).squeeze(1)
        return logits, hidden

    def forward(self, source: torch.Tensor, target_input: torch.Tensor) -> torch.Tensor:
        encoded, hidden, mask = self.encode(source)
        outputs = []
        for position in range(target_input.shape[1]):
            logits, hidden = self.decode_step(
                target_input[:, position], hidden, encoded, mask
            )
            outputs.append(logits)
        return torch.stack(outputs, dim=1)


class PhoneticTransducer:
    """Inference wrapper which returns only complete forward-valid candidates."""

    def __init__(self, network: _Network, config: TransducerConfig, device: str = "cpu"):
        self.network = network.to(device).eval()
        self.config = config
        self.device = torch.device(device)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CHECKPOINT, device: str = "cpu") -> "PhoneticTransducer":
        path = Path(path)
        metadata_path = path.with_suffix(".json")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"checkpoint metadata is missing: {metadata_path}")
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("format_version") != 1:
            raise ValueError("unsupported checkpoint metadata format_version")
        expected_hash = metadata.get("checkpoint_sha256")
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected_hash != actual_hash:
            raise ValueError("checkpoint SHA-256 does not match its metadata")
        payload = torch.load(path, map_location=device, weights_only=True)
        if payload.get("format_version") != 1:
            raise ValueError("unsupported checkpoint payload format_version")
        config = TransducerConfig(**payload["config"])
        if payload["input_tokens"] != INPUT_TOKENS or payload["output_tokens"] != OUTPUT_TOKENS:
            raise ValueError("checkpoint vocabulary does not match this runtime")
        network = _Network(config)
        network.load_state_dict(payload["state_dict"], strict=True)
        return cls(network, config, device)

    def generate_constrained(self, roman: str, beam_size: int = 8):
        """Return extra forward-valid proposals using stable-prefix pruning."""
        return generate_constrained(self, roman, beam_size)

    def generate(self, roman: str, beam_size: int = 4) -> list[tuple[str, float]]:
        """Return up to ``beam_size`` ``(Hangul, log_probability)`` pairs.

        Invalid, incomplete, and forward-inconsistent hypotheses are discarded;
        an uncertain model therefore returns an empty list rather than a prefix.
        """
        if not isinstance(roman, str) or not ROMAN_RE.fullmatch(roman):
            raise ValueError("roman must contain lowercase ASCII letters only")
        if len(roman) > self.config.max_roman_length:
            raise ValueError(f"roman must be at most {self.config.max_roman_length} letters")
        if not isinstance(beam_size, int) or isinstance(beam_size, bool) or not 1 <= beam_size <= 16:
            raise ValueError("beam_size must be an integer from 1 to 16")

        source = torch.tensor(
            [[INPUT_TO_ID[BOS], *(INPUT_TO_ID.get(char, INPUT_TO_ID["<unk>"]) for char in roman),
              INPUT_TO_ID[EOS]]],
            dtype=torch.long,
            device=self.device,
        )
        search_width = min(len(OUTPUT_TOKENS), max(8, beam_size * 4))
        max_steps = min(3 * len(roman) + 4, 3 * 24 + 1)
        completed: list[tuple[list[int], float]] = []
        with torch.inference_mode():
            encoded, initial, mask = self.network.encode(source)
            beams = [([OUTPUT_TO_ID[BOS]], 0.0, initial)]
            for _ in range(max_steps):
                expanded = []
                for tokens, score, hidden in beams:
                    if tokens[-1] == OUTPUT_TO_ID[EOS]:
                        completed.append((tokens, score))
                        continue
                    token = torch.tensor([tokens[-1]], device=self.device)
                    logits, next_hidden = self.network.decode_step(token, hidden, encoded, mask)
                    log_probs = torch.log_softmax(logits[0], dim=-1)
                    allowed = torch.tensor(
                        _allowed_output_ids(len(tokens) - 1), device=self.device
                    )
                    allowed_probs = log_probs.index_select(0, allowed)
                    count = min(search_width, len(allowed))
                    values, positions = torch.topk(allowed_probs, count)
                    indices = allowed.index_select(0, positions)
                    for value, index in zip(values.tolist(), indices.tolist()):
                        expanded.append((tokens + [index], score + value, next_hidden.clone()))
                if not expanded:
                    break
                beams = sorted(expanded, key=lambda item: item[1], reverse=True)[:search_width]
            completed.extend((tokens, score) for tokens, score, _ in beams
                             if tokens[-1] == OUTPUT_TO_ID[EOS])

        valid: dict[str, float] = {}
        for token_ids, score in sorted(completed, key=lambda item: item[1], reverse=True):
            output_tokens = [OUTPUT_TOKENS[index] for index in token_ids[1:-1]]
            hangul = jamo_to_hangul(output_tokens)
            if hangul is not None and romanize_word(hangul) == roman:
                valid[hangul] = max(score, valid.get(hangul, float("-inf")))
            if len(valid) >= beam_size:
                break
        return sorted(valid.items(), key=lambda item: item[1], reverse=True)


def stable_roman_prefix(hangul):
    """RR symbols which appending complete syllables cannot change.

    The local pair rules can still alter the final written coda. Exact-word
    insertion exceptions override those local rules, so defer their prefixes.
    """
    if not hangul or any(key.startswith(hangul) for key in _INSERTION_EXCEPTIONS):
        return None
    last = ord(hangul[-1])
    no_coda = chr(last - (last - 0xAC00) % 28)
    return romanize_word(hangul[:-1] + no_coda)


def compatible_prefix(hangul, roman, *, complete=False):
    if complete:
        return romanize_word(hangul) == roman
    stable = stable_roman_prefix(hangul)
    return stable is None or roman.startswith(stable)


def generate_constrained(transducer, roman, beam_size=8, *, stats=None):
    if not isinstance(roman, str) or not ROMAN_RE.fullmatch(roman):
        raise ValueError('roman must contain lowercase ASCII letters only')
    if len(roman) > transducer.config.max_roman_length:
        raise ValueError('roman exceeds transducer input bound')
    if type(beam_size) is not int or not 1 <= beam_size <= 16:
        raise ValueError('beam size must be an integer from 1 to 16')
    width = min(len(OUTPUT_TOKENS), max(8, beam_size * 4))
    max_steps = min(3 * len(roman) + 4, 3 * 24 + 1)
    source = torch.tensor([[INPUT_TO_ID[BOS], *(INPUT_TO_ID[c] for c in roman), INPUT_TO_ID[EOS]]],
                          dtype=torch.long, device=transducer.device)
    completed, counts = {}, {'steps':0, 'expanded':0, 'prefix_pruned':0, 'eos_rejected':0}
    with torch.inference_mode():
        encoded, initial, mask = transducer.network.encode(source)
        beams = [([OUTPUT_TO_ID[BOS]], 0., initial)]
        for _ in range(max_steps):
            expanded = []
            counts['steps'] += 1
            for tokens, score, hidden in beams:
                token = torch.tensor([tokens[-1]], device=transducer.device)
                logits, next_hidden = transducer.network.decode_step(token, hidden, encoded, mask)
                probabilities = torch.log_softmax(logits[0], dim=-1)
                allowed = torch.tensor(_allowed_output_ids(len(tokens)-1), device=transducer.device)
                values, positions = torch.topk(probabilities.index_select(0, allowed), min(width, len(allowed)))
                indices = allowed.index_select(0, positions)
                for value, index in zip(values.tolist(), indices.tolist()):
                    counts['expanded'] += 1
                    new_score = score + value
                    if index == OUTPUT_TO_ID[EOS]:
                        hangul = jamo_to_hangul([OUTPUT_TOKENS[i] for i in tokens[1:]])
                        if hangul is not None and compatible_prefix(hangul, roman, complete=True):
                            completed[hangul] = max(new_score, completed.get(hangul, float('-inf')))
                        else:
                            counts['eos_rejected'] += 1
                        continue
                    output = tokens + [index]
                    if (len(output)-1) % 3 == 0:
                        hangul = jamo_to_hangul([OUTPUT_TOKENS[i] for i in output[1:]])
                        if hangul is None or not compatible_prefix(hangul, roman):
                            counts['prefix_pruned'] += 1
                            continue
                    expanded.append((output, new_score, next_hidden.clone()))
            if not expanded:
                break
            beams = sorted(expanded, key=lambda item:item[1], reverse=True)[:width]
    if stats is not None:
        stats.update(counts)
    return sorted(completed.items(), key=lambda item:item[1], reverse=True)[:beam_size]


def checkpoint_payload(network: _Network, config: TransducerConfig) -> dict:
    """Build a checkpoint compatible with ``torch.load(weights_only=True)``."""
    return {
        "format_version": 1,
        "config": asdict(config),
        "input_tokens": INPUT_TOKENS,
        "output_tokens": OUTPUT_TOKENS,
        "state_dict": network.state_dict(),
    }


__all__ = [
    "DEFAULT_CHECKPOINT",
    "PhoneticTransducer",
    "TransducerConfig",
    "checkpoint_payload",
    "hangul_to_jamo",
    "jamo_to_hangul",
]
