"""Experimental complete-Roman-sentence to mixed Korean/Latin transducer.

Input bytes avoid unknown Roman tokens. Typed onset/vowel/coda tokens can
construct every Hangul syllable without a learned whole-syllable vocabulary.
Other characters are losslessly represented as UTF-8 bytes.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re

import torch
from torch import nn

PAD, BOS, EOS = 0, 1, 2
BYTE_OFFSET = 3
ONSET_OFFSET = BYTE_OFFSET + 256
VOWEL_OFFSET = ONSET_OFFSET + 19
CODA_OFFSET = VOWEL_OFFSET + 21
SOURCE_VOCAB = ONSET_OFFSET
TARGET_VOCAB = CODA_OFFSET + 28
FORMAT = 'whole-sentence-byte-jamo-v1'
HANGUL = re.compile('[가-힣ᄀ-ᇿ㄰-㆏ꥠ-꥿ힰ-퟿]')


@dataclass(frozen=True)
class SentenceTransducerConfig:
    width: int = 192
    heads: int = 4
    encoder_layers: int = 2
    decoder_layers: int = 2
    feedforward: int = 768
    dropout: float = .1
    max_tokens: int = 512
    # None preserves checkpoints trained with sqrt(width) embedding scaling.
    embedding_scale: float | None = None


def encode_source(text):
    if not isinstance(text, str) or len(text.split()) < 2 or HANGUL.search(text):
        raise ValueError('model source must be a whole Roman sentence without Korean hints')
    return [BOS, *(BYTE_OFFSET + byte for byte in text.encode('utf-8')), EOS]


def encode_target(text):
    tokens = []
    for char in text:
        if '가' <= char <= '힣':
            onset, rest = divmod(ord(char) - 0xAC00, 21 * 28)
            vowel, coda = divmod(rest, 28)
            tokens.extend((ONSET_OFFSET + onset, VOWEL_OFFSET + vowel, CODA_OFFSET + coda))
        else:
            tokens.extend(BYTE_OFFSET + byte for byte in char.encode('utf-8'))
    return tokens


def decode_target(tokens):
    """Reject partial syllables, special tokens, and malformed UTF-8."""
    result = bytearray()
    cursor = 0
    while cursor < len(tokens):
        token = tokens[cursor]
        if BYTE_OFFSET <= token < ONSET_OFFSET:
            result.append(token - BYTE_OFFSET)
            cursor += 1
        elif ONSET_OFFSET <= token < VOWEL_OFFSET:
            if cursor + 2 >= len(tokens):
                raise ValueError('incomplete Hangul syllable')
            vowel, coda = tokens[cursor + 1:cursor + 3]
            if not (VOWEL_OFFSET <= vowel < CODA_OFFSET and CODA_OFFSET <= coda < TARGET_VOCAB):
                raise ValueError('invalid typed Hangul sequence')
            value = ((token - ONSET_OFFSET) * 21 + vowel - VOWEL_OFFSET) * 28 + coda - CODA_OFFSET
            result.extend(chr(0xAC00 + value).encode('utf-8'))
            cursor += 3
        else:
            raise ValueError('invalid target token')
    return result.decode('utf-8', errors='strict')


def pad_sequences(rows):
    if not rows:
        raise ValueError('cannot pad an empty batch')
    result = torch.full((len(rows), max(map(len, rows))), PAD, dtype=torch.long)
    for index, row in enumerate(rows):
        result[index, :len(row)] = torch.tensor(row, dtype=torch.long)
    return result


def training_batch(pairs, max_tokens=512):
    encoded = [(encode_source(source), encode_target(target)) for source, target in pairs]
    if any(len(source) > max_tokens or len(target) + 1 > max_tokens for source, target in encoded):
        raise ValueError('complete sentence exceeds token budget; truncation is forbidden')
    return (pad_sequences([source for source, _ in encoded]),
            pad_sequences([[BOS, *target] for _, target in encoded]),
            pad_sequences([[*target, EOS] for _, target in encoded]))


class SentenceTransducer(nn.Module):
    def __init__(self, config=SentenceTransducerConfig()):
        super().__init__()
        self.config = config
        if config.embedding_scale is not None and (
                isinstance(config.embedding_scale, bool)
                or not math.isfinite(config.embedding_scale) or config.embedding_scale <= 0):
            raise ValueError('embedding scale must be finite and positive')
        self.source_embedding = nn.Embedding(SOURCE_VOCAB, config.width, padding_idx=PAD)
        self.target_embedding = nn.Embedding(TARGET_VOCAB, config.width, padding_idx=PAD)
        self.transformer = nn.Transformer(
            d_model=config.width, nhead=config.heads, num_encoder_layers=config.encoder_layers,
            num_decoder_layers=config.decoder_layers, dim_feedforward=config.feedforward,
            dropout=config.dropout, batch_first=True)
        self.output = nn.Linear(config.width, TARGET_VOCAB)
        positions = torch.arange(config.max_tokens).float().unsqueeze(1)
        scales = torch.exp(torch.arange(0, config.width, 2).float() * (-math.log(10000.) / config.width))
        positional = torch.zeros(config.max_tokens, config.width)
        positional[:, 0::2] = torch.sin(positions * scales)
        positional[:, 1::2] = torch.cos(positions * scales)
        self.register_buffer('positions', positional.unsqueeze(0))

    def embedded(self, tokens, embedding):
        if tokens.shape[1] > self.config.max_tokens:
            raise ValueError('complete sequence exceeds token limit')
        scale = math.sqrt(self.config.width) if self.config.embedding_scale is None else self.config.embedding_scale
        return embedding(tokens) * scale + self.positions[:, :tokens.shape[1]]

    def encode(self, source):
        mask = source.eq(PAD)
        memory = self.transformer.encoder(self.embedded(source, self.source_embedding), src_key_padding_mask=mask)
        return memory, mask

    def decode(self, target, memory, source_mask):
        causal = torch.ones(target.shape[1], target.shape[1], dtype=torch.bool, device=target.device).triu(1)
        decoded = self.transformer.decoder(
            self.embedded(target, self.target_embedding), memory, tgt_mask=causal,
            tgt_key_padding_mask=target.eq(PAD), memory_key_padding_mask=source_mask)
        return self.output(decoded)

    def forward(self, source, target):
        memory, mask = self.encode(source)
        return self.decode(target, memory, mask)

    @torch.inference_mode()
    def generate(self, sources):
        """Greedy generation receives only complete source strings, never targets."""
        self.eval()
        rows = [encode_source(source) for source in sources]
        if any(len(row) > self.config.max_tokens for row in rows):
            raise ValueError('source exceeds token limit; truncation is forbidden')
        device = next(self.parameters()).device
        source = pad_sequences(rows).to(device)
        memory, mask = self.encode(source)
        target = torch.full((len(rows), 1), BOS, dtype=torch.long, device=device)
        finished = [False] * len(rows)
        generated = [[] for _ in rows]
        # Literal bytes can only come from the Roman source. This is a vocabulary
        # constraint, not a guarantee of exact copying or semantic faithfulness.
        literal_ids = [sorted(set(row) - {BOS, EOS}) for row in rows]
        for _ in range(self.config.max_tokens):
            logits = self.decode(target, memory, mask)[:, -1]
            filtered = torch.full_like(logits, -torch.inf)
            for index, sequence in enumerate(generated):
                if finished[index]:
                    allowed = [EOS]
                elif sequence and ONSET_OFFSET <= sequence[-1] < VOWEL_OFFSET:
                    allowed = range(VOWEL_OFFSET, CODA_OFFSET)
                elif sequence and VOWEL_OFFSET <= sequence[-1] < CODA_OFFSET:
                    allowed = range(CODA_OFFSET, TARGET_VOCAB)
                else:
                    allowed = [EOS, *literal_ids[index], *range(ONSET_OFFSET, VOWEL_OFFSET)]
                filtered[index, list(allowed)] = logits[index, list(allowed)]
            if not torch.isfinite(filtered.max(dim=1).values).all():
                raise ValueError('nonfinite generation logits')
            chosen = filtered.argmax(dim=1)
            for index, token in enumerate(chosen.tolist()):
                if not finished[index]:
                    if token == EOS:
                        finished[index] = True
                    else:
                        generated[index].append(token)
            target = torch.cat((target, chosen[:, None]), dim=1)
            if all(finished):
                break
        outputs = []
        for sequence, complete in zip(generated, finished):
            text = None
            reason = 'token_limit'
            if complete:
                try:
                    text = decode_target(sequence)
                    reason = 'complete'
                except (ValueError, UnicodeError):
                    reason = 'invalid_encoding'
            outputs.append(dict(text=text, reason=reason, generated_tokens=len(sequence)))
        return outputs

    def save(self, path, provenance):
        path = Path(path)
        if path.exists() or path.with_suffix('.json').exists():
            raise FileExistsError('preserve existing model artifacts')
        torch.save(dict(format=FORMAT, config=asdict(self.config), state_dict=self.state_dict()), path)
        metadata = dict(format=FORMAT, config=asdict(self.config), provenance=provenance,
                        checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        path.with_suffix('.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')

    @classmethod
    def load(cls, path):
        path = Path(path)
        metadata = json.loads(path.with_suffix('.json').read_text())
        if metadata.get('format') != FORMAT or hashlib.sha256(path.read_bytes()).hexdigest() != metadata['checkpoint_sha256']:
            raise ValueError('checkpoint identity mismatch')
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload['format'] != FORMAT or payload['config'] != metadata['config']:
            raise ValueError('checkpoint configuration mismatch')
        model = cls(SentenceTransducerConfig(**payload['config']))
        model.load_state_dict(payload['state_dict'], strict=True)
        return model.eval()
