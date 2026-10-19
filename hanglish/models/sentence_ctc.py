"""Experimental monotone CTC model for complete Roman sentences.

This module reuses the sentence transducer's byte source and typed
onset/vowel/coda target alphabets.  Every raw UTF-8 source byte owns exactly
three output frames; BOS, EOS and batch padding never produce CTC frames.
"""
from __future__ import annotations

import codecs
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from hanglish.models.sentence_transducer import (
    BOS,
    BYTE_OFFSET,
    CODA_OFFSET,
    EOS,
    FORMAT as AR_FORMAT,
    ONSET_OFFSET,
    PAD,
    SOURCE_VOCAB,
    TARGET_VOCAB,
    VOWEL_OFFSET,
    decode_target,
    encode_source,
    encode_target,
    pad_sequences,
)


FORMAT = "whole-sentence-byte-jamo-ctc-v1"
BLANK = 0
LABEL_OFFSET = 2
CTC_VOCAB = TARGET_VOCAB - LABEL_OFFSET
SUBFRAMES = 3
MAX_TARGET_TOKENS = 511


@dataclass(frozen=True)
class SentenceCTCConfig:
    width: int = 192
    heads: int = 4
    encoder_layers: int = 2
    feedforward: int = 768
    dropout: float = .1
    max_source_tokens: int = 512
    embedding_scale: float = 1.0
    beam_size: int = 8
    token_shortlist: int = 32


def _validate_config(config):
    integer_bounds = {
        "width": (4, 1024), "heads": (1, 32), "encoder_layers": (1, 12),
        "feedforward": (4, 8192), "max_source_tokens": (4, 4096),
        "beam_size": (1, 128), "token_shortlist": (1, CTC_VOCAB - 1),
    }
    for name, (minimum, maximum) in integer_bounds.items():
        value = getattr(config, name)
        if (isinstance(value, bool) or not isinstance(value, int)
                or not minimum <= value <= maximum):
            raise ValueError(f"CTC {name} is outside its supported bound")
    if config.width % config.heads:
        raise ValueError("CTC width must be divisible by heads")
    if config.width % 2:
        raise ValueError("CTC width must be even for sinusoidal positions")
    for name in ("dropout", "embedding_scale"):
        value = getattr(config, name)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value)):
            raise ValueError(f"CTC {name} must be finite")
    if not 0 <= config.dropout < 1 or config.embedding_scale <= 0:
        raise ValueError("CTC dropout/embedding scale is outside its supported bound")


def ctc_labels(text):
    """Map original target tokens 3..326 to nonblank CTC labels 1..324."""
    labels = [token - LABEL_OFFSET for token in encode_target(text)]
    if any(not 1 <= label < CTC_VOCAB for label in labels):
        raise ValueError("target contains an invalid CTC label")
    return labels


def required_ctc_frames(labels):
    """Minimum frames including blanks needed between adjacent repeats."""
    values = list(labels)
    if any(isinstance(value, bool) or not isinstance(value, int)
           or not 1 <= value < CTC_VOCAB for value in values):
        raise ValueError("CTC labels must be nonblank vocabulary IDs")
    return len(values) + sum(left == right for left, right in zip(values, values[1:]))


def training_batch(pairs, config=SentenceCTCConfig()):
    """Encode complete pairs without truncation and audit CTC feasibility."""
    _validate_config(config)
    if not isinstance(pairs, (list, tuple)) or not pairs:
        raise ValueError("CTC training requires at least one source/target pair")
    sources = []
    rows = []
    input_lengths = []
    target_lengths = []
    for pair in pairs:
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or not all(isinstance(value, str) for value in pair)):
            raise ValueError("CTC pair must contain source and target text")
        source, target = pair
        encoded_source = encode_source(source)
        labels = ctc_labels(target)
        frames = SUBFRAMES * (len(encoded_source) - 2)
        if len(encoded_source) > config.max_source_tokens:
            raise ValueError("complete CTC source exceeds token limit")
        if len(labels) > MAX_TARGET_TOKENS:
            raise ValueError("complete CTC target exceeds 511-token limit")
        if required_ctc_frames(labels) > frames:
            raise ValueError("CTC target and adjacent repeats exceed source frame budget")
        sources.append(encoded_source)
        rows.extend(labels)
        input_lengths.append(frames)
        target_lengths.append(len(labels))
    return (
        pad_sequences(sources),
        torch.tensor(rows, dtype=torch.long),
        torch.tensor(input_lengths, dtype=torch.long),
        torch.tensor(target_lengths, dtype=torch.long),
    )


def collapse_ctc(path):
    """Collapse repeats before removing blanks, per the CTC definition."""
    result = []
    previous = None
    for label in path:
        if (isinstance(label, bool) or not isinstance(label, int)
                or not 0 <= label < CTC_VOCAB):
            raise ValueError("CTC path contains an invalid vocabulary ID")
        if label != previous and label != BLANK:
            result.append(label)
        previous = label
    return result


def decode_ctc_labels(labels):
    """Strictly decode one collapsed label sequence with explicit failures."""
    values = list(labels)
    if not values:
        return {"text": None, "reason": "empty_output", "generated_tokens": 0}
    if any(isinstance(label, bool) or not isinstance(label, int)
           or not 1 <= label < CTC_VOCAB for label in values):
        raise ValueError("collapsed CTC labels must be nonblank vocabulary IDs")
    tokens = [label + LABEL_OFFSET for label in values]
    try:
        text = decode_target(tokens)
    except UnicodeDecodeError:
        return {"text": None, "reason": "invalid_utf8", "generated_tokens": len(values)}
    except ValueError:
        return {"text": None, "reason": "invalid_jamo", "generated_tokens": len(values)}
    return {"text": text, "reason": "complete", "generated_tokens": len(values)}


def _valid_utf8(data, final):
    try:
        decoder = codecs.getincrementaldecoder("utf-8")("strict")
        decoder.decode(bytes(data), final=final)
        return True
    except UnicodeDecodeError:
        return False


def _advance_grammar(state, label):
    phase, pending = state
    if not 1 <= label < CTC_VOCAB:
        return None
    token = label + LABEL_OFFSET
    if BYTE_OFFSET <= token < ONSET_OFFSET:
        if phase != "free":
            return None
        decoder = codecs.getincrementaldecoder("utf-8")("strict")
        try:
            decoder.decode(pending + bytes([token - BYTE_OFFSET]), final=False)
        except UnicodeDecodeError:
            return None
        return "free", decoder.getstate()[0]
    if ONSET_OFFSET <= token < VOWEL_OFFSET:
        return ("vowel", b"") if phase == "free" and not pending else None
    if VOWEL_OFFSET <= token < CODA_OFFSET:
        return ("coda", b"") if phase == "vowel" else None
    if CODA_OFFSET <= token < TARGET_VOCAB:
        return ("free", b"") if phase == "coda" else None
    return None


def valid_label_prefix(labels, *, final=False):
    """Enforce literal UTF-8 and mandatory onset→vowel→coda grammar."""
    state = ("free", b"")
    for label in labels:
        state = _advance_grammar(state, label)
        if state is None:
            return False
    phase, pending = state
    if final:
        return phase == "free" and not pending
    return True


def _logadd(*values):
    finite = [value for value in values if value != -math.inf]
    if not finite:
        return -math.inf
    maximum = max(finite)
    return maximum + math.log(sum(math.exp(value - maximum) for value in finite))


def ctc_prefix_beam(log_probs, *, beam_size=8, token_shortlist=32):
    """Grammar-constrained CTC prefix beam with standard repeat merging.

    Token shortlisting makes this an approximate decoder when the nonblank
    vocabulary is larger than ``token_shortlist``.  The returned audit records
    both shortlist and beam pruning.
    """
    if (not isinstance(log_probs, torch.Tensor) or log_probs.ndim != 2
            or log_probs.shape[1] != CTC_VOCAB):
        raise ValueError("CTC prefix beam expects [frames, vocabulary] log probabilities")
    if (isinstance(beam_size, bool) or not isinstance(beam_size, int)
            or not 1 <= beam_size <= 128
            or isinstance(token_shortlist, bool) or not isinstance(token_shortlist, int)
            or not 1 <= token_shortlist < CTC_VOCAB):
        raise ValueError("CTC prefix beam bounds are invalid")
    values = log_probs.detach().to(dtype=torch.float64, device="cpu")
    if (torch.isnan(values).any() or torch.isposinf(values).any()
            or not torch.isfinite(values).any(dim=1).all()):
        raise ValueError("CTC prefix beam received invalid log-probability support")
    beams = {(): (0.0, -math.inf)}
    grammar_states = {(): ("free", b"")}
    shortlist_pruned = token_shortlist < CTC_VOCAB - 1
    beam_pruned = False
    repeat_tokens_restored = 0
    for frame in values:
        nonblank = torch.topk(frame[1:], k=min(token_shortlist, CTC_VOCAB - 1)).indices
        shortlisted = [int(index) + 1 for index in nonblank]
        shortlisted_set = set(shortlisted)
        next_beams = {}

        def update(prefix, blank=None, nonblank_value=None):
            blank = None if blank == -math.inf else blank
            nonblank_value = None if nonblank_value == -math.inf else nonblank_value
            if blank is None and nonblank_value is None:
                return
            old_blank, old_nonblank = next_beams.get(prefix, (-math.inf, -math.inf))
            next_beams[prefix] = (
                _logadd(old_blank, blank) if blank is not None else old_blank,
                _logadd(old_nonblank, nonblank_value)
                if nonblank_value is not None else old_nonblank,
            )

        for prefix, (prefix_blank, prefix_nonblank) in beams.items():
            grammar_state = grammar_states[prefix]
            total = _logadd(prefix_blank, prefix_nonblank)
            update(prefix, blank=total + float(frame[BLANK]))
            tokens = shortlisted
            if prefix and prefix[-1] not in shortlisted_set:
                # A repeated nonblank stays on the existing collapsed prefix.
                # Keep that exact transition mass even when approximate global
                # shortlisting omitted the prefix's last label.
                tokens = [*shortlisted, prefix[-1]]
                repeat_tokens_restored += 1
            for token in tokens:
                probability = float(frame[token])
                if probability == -math.inf:
                    continue
                if prefix and token == prefix[-1]:
                    # The nonblank→same-nonblank transition stays on the same
                    # collapsed prefix.  Only a preceding blank can extend it.
                    update(prefix, nonblank_value=prefix_nonblank + probability)
                    next_state = (_advance_grammar(grammar_state, token)
                                  if len(prefix) < MAX_TARGET_TOKENS else None)
                    if prefix_blank != -math.inf and next_state is not None:
                        extended = (*prefix, token)
                        grammar_states[extended] = next_state
                        update(extended, nonblank_value=prefix_blank + probability)
                else:
                    next_state = (_advance_grammar(grammar_state, token)
                                  if len(prefix) < MAX_TARGET_TOKENS else None)
                    if next_state is not None:
                        extended = (*prefix, token)
                        grammar_states[extended] = next_state
                        update(extended, nonblank_value=total + probability)
        ordered = sorted(next_beams.items(),
                         key=lambda item: (-_logadd(*item[1]), item[0]))
        if len(ordered) > beam_size:
            beam_pruned = True
        beams = dict(ordered[:beam_size])
        grammar_states = {prefix: grammar_states[prefix] for prefix in beams}
    completed = [
        {"labels": list(prefix), "log_probability": _logadd(*probabilities)}
        for prefix, probabilities in beams.items()
        if grammar_states[prefix] == ("free", b"")
    ]
    completed.sort(key=lambda item: (-item["log_probability"], item["labels"]))
    return completed, {
        "frames": int(values.shape[0]),
        "beam_size": beam_size,
        "token_shortlist": token_shortlist,
        "max_label_tokens": MAX_TARGET_TOKENS,
        "shortlist_pruned": shortlist_pruned,
        "beam_pruned": beam_pruned,
        "repeat_tokens_restored": repeat_tokens_restored,
        "approximate_pruning": shortlist_pruned or beam_pruned,
    }


def subframe_memory(memory, source_lengths, offsets):
    """Expand raw-byte encoder states while excluding BOS/EOS and padding."""
    if (not isinstance(memory, torch.Tensor) or memory.ndim != 3
            or not isinstance(source_lengths, torch.Tensor) or source_lengths.ndim != 1
            or source_lengths.shape[0] != memory.shape[0]
            or not isinstance(offsets, torch.Tensor)
            or offsets.shape != (SUBFRAMES, memory.shape[2])):
        raise ValueError("CTC subframe tensors have incompatible shapes")
    lengths = source_lengths.to(dtype=torch.long, device="cpu")
    if (lengths < 3).any() or (lengths > memory.shape[1]).any():
        raise ValueError("CTC source lengths are invalid")
    raw_max = int(lengths.max()) - 2
    frames = memory[:, 1:1 + raw_max, None, :] + offsets[None, None, :, :]
    frames = frames.reshape(memory.shape[0], raw_max * SUBFRAMES, memory.shape[2])
    return frames, (lengths - 2) * SUBFRAMES


class SentenceCTC(nn.Module):
    def __init__(self, config=SentenceCTCConfig()):
        super().__init__()
        _validate_config(config)
        self.config = config
        self.source_embedding = nn.Embedding(SOURCE_VOCAB, config.width, padding_idx=PAD)
        layer = nn.TransformerEncoderLayer(
            d_model=config.width, nhead=config.heads,
            dim_feedforward=config.feedforward, dropout=config.dropout,
            batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.encoder_layers)
        positions = torch.arange(config.max_source_tokens).float().unsqueeze(1)
        scales = torch.exp(torch.arange(0, config.width, 2).float()
                           * (-math.log(10000.) / config.width))
        positional = torch.zeros(config.max_source_tokens, config.width)
        positional[:, 0::2] = torch.sin(positions * scales)
        positional[:, 1::2] = torch.cos(positions * scales)
        self.register_buffer("positions", positional.unsqueeze(0))
        self.subframe_offsets = nn.Parameter(torch.empty(SUBFRAMES, config.width))
        nn.init.normal_(self.subframe_offsets, std=.02)
        self.output = nn.Linear(config.width, CTC_VOCAB)
        self.ctc_loss = nn.CTCLoss(blank=BLANK, reduction="mean", zero_infinity=False)

    def _source_lengths(self, source):
        if (not isinstance(source, torch.Tensor) or source.ndim != 2
                or source.dtype != torch.long or not source.shape[0]):
            raise ValueError("CTC source must be a nonempty rank-two long tensor")
        if source.shape[1] > self.config.max_source_tokens:
            raise ValueError("padded CTC source exceeds token limit")
        lengths = source.ne(PAD).sum(dim=1)
        if int(lengths.max()) > self.config.max_source_tokens:
            raise ValueError("complete CTC source exceeds token limit")
        for row, length_tensor in zip(source, lengths):
            length = int(length_tensor)
            values = row.tolist()
            if (length < 3 or values[0] != BOS or values[length - 1] != EOS
                    or any(value == PAD for value in values[:length])
                    or any(value != PAD for value in values[length:])
                    or any(not BYTE_OFFSET <= value < SOURCE_VOCAB
                           for value in values[1:length - 1])):
                raise ValueError("CTC source tensor violates byte/BOS/EOS framing")
        return lengths

    def forward(self, source):
        lengths = self._source_lengths(source)
        mask = source.eq(PAD)
        embedded = (self.source_embedding(source) * self.config.embedding_scale
                    + self.positions[:, :source.shape[1]])
        memory = self.encoder(embedded, src_key_padding_mask=mask)
        frames, input_lengths = subframe_memory(memory, lengths, self.subframe_offsets)
        return self.output(frames).transpose(0, 1), input_lengths

    def loss(self, source, targets, input_lengths, target_lengths):
        logits, actual_lengths = self(source)
        if (not torch.equal(actual_lengths.cpu(), input_lengths.cpu())
                or targets.ndim != 1 or target_lengths.ndim != 1
                or target_lengths.shape[0] != source.shape[0]
                or int(target_lengths.sum()) != targets.numel()):
            raise ValueError("CTC loss lengths differ from the encoded batch")
        cursor = 0
        for frames, count in zip(input_lengths.tolist(), target_lengths.tolist()):
            labels = targets[cursor:cursor + count].tolist()
            if count > MAX_TARGET_TOKENS or required_ctc_frames(labels) > frames:
                raise ValueError("CTC loss target exceeds its audited frame budget")
            cursor += count
        return self.ctc_loss(
            F.log_softmax(logits, dim=2), targets,
            actual_lengths.to(dtype=torch.long), target_lengths.to(dtype=torch.long))

    @torch.inference_mode()
    def generate(self, sources, *, beam_size=None, token_shortlist=None):
        """Decode complete source strings only, retaining greedy diagnostics."""
        self.eval()
        if not isinstance(sources, list) or not sources:
            raise ValueError("CTC generation requires a nonempty source list")
        encoded = [encode_source(source) for source in sources]
        if any(len(row) > self.config.max_source_tokens for row in encoded):
            raise ValueError("complete CTC source exceeds token limit")
        device = next(self.parameters()).device
        logits, lengths = self(pad_sequences(encoded).to(device))
        log_probs = F.log_softmax(logits, dim=2)
        beam_size = self.config.beam_size if beam_size is None else beam_size
        token_shortlist = (self.config.token_shortlist
                           if token_shortlist is None else token_shortlist)
        outputs = []
        for index, length in enumerate(lengths.tolist()):
            row_logits = logits[:length, index]
            if not torch.isfinite(row_logits).all():
                raise ValueError("CTC generation logits are nonfinite")
            row = log_probs[:length, index]
            greedy_labels = collapse_ctc(row.argmax(dim=1).tolist())
            greedy = decode_ctc_labels(greedy_labels)
            greedy["labels"] = greedy_labels
            beams, audit = ctc_prefix_beam(
                row, beam_size=beam_size, token_shortlist=token_shortlist)
            if beams:
                labels = beams[0]["labels"]
                result = decode_ctc_labels(labels)
                result["labels"] = labels
                result["log_probability"] = beams[0]["log_probability"]
            else:
                result = {"text": None, "reason": "no_valid_path",
                          "generated_tokens": 0, "labels": [],
                          "log_probability": None}
            result.update({"frames": length, "decode_mode": "ctc_prefix_beam",
                           "beam": audit, "greedy": greedy})
            outputs.append(result)
        return outputs

    def save(self, path, provenance):
        path = Path(path)
        metadata_path = path.with_suffix(".json")
        if path.exists() or metadata_path.exists():
            raise FileExistsError("preserve existing CTC model artifacts")
        payload = {"format": FORMAT, "source_format": AR_FORMAT,
                   "config": asdict(self.config), "state_dict": self.state_dict()}
        torch.save(payload, path)
        metadata = {
            "format": FORMAT,
            "source_format": AR_FORMAT,
            "config": asdict(self.config),
            "provenance": provenance,
            "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")

    @classmethod
    def load(cls, path):
        path = Path(path)
        metadata_path = path.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text())
        if (set(metadata) != {"format", "source_format", "config", "provenance",
                             "checkpoint_sha256"}
                or metadata["format"] != FORMAT or metadata["source_format"] != AR_FORMAT
                or hashlib.sha256(path.read_bytes()).hexdigest()
                != metadata["checkpoint_sha256"]):
            raise ValueError("CTC checkpoint identity mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (not isinstance(payload, dict)
                or set(payload) != {"format", "source_format", "config", "state_dict"}
                or payload["format"] != FORMAT or payload["source_format"] != AR_FORMAT
                or payload["config"] != metadata["config"]
                or not isinstance(payload["state_dict"], dict)):
            raise ValueError("CTC checkpoint payload differs from metadata")
        model = cls(SentenceCTCConfig(**metadata["config"]))
        model.load_state_dict(payload["state_dict"], strict=True)
        return model.eval()


__all__ = [
    "BLANK", "CTC_VOCAB", "FORMAT", "MAX_TARGET_TOKENS", "SUBFRAMES",
    "SentenceCTC", "SentenceCTCConfig", "collapse_ctc", "ctc_labels",
    "ctc_prefix_beam", "decode_ctc_labels", "required_ctc_frames",
    "subframe_memory", "training_batch", "valid_label_prefix",
]
