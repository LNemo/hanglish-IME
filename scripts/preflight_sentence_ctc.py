"""Resource and serialization gate for the architecture-only CTC control."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from scripts.train_sentence_transducer import (
    SEED, BATCH, LENGTH_BINS, MAX_RSS, MAX_PREFLIGHT_SECONDS,
    read_training, batches, row_length, sha, write_json, check_pins,
)
from scripts.evaluate_ensemble_regression import rss_bytes
from hanglish.models.sentence_transducer import encode_source, encode_target


def frame_audit(rows):
    result = dict(rows=len(rows), invalid=0, max_required_per_source_byte=0.,
                  max_source_tokens=0, max_target_tokens=0)
    for row in rows:
        byte_count = len(row['input'].encode('utf-8'))
        target = encode_target(row['expected'][0])
        required = len(target) + sum(a == b for a, b in zip(target, target[1:]))
        result['invalid'] += int(required > 3 * byte_count or byte_count + 2 > 512 or len(target) > 511)
        result['max_required_per_source_byte'] = max(result['max_required_per_source_byte'], required / byte_count)
        result['max_source_tokens'] = max(result['max_source_tokens'], byte_count + 2)
        result['max_target_tokens'] = max(result['max_target_tokens'], len(target))
    return result


def estimate(updates, rows, epochs=8):
    times = {}
    largest = 0.
    for step in updates:
        largest = max(largest, step['elapsed_s'])
        times[step['source_tokens']] = largest
    total = 0.
    for epoch in range(epochs):
        for group in batches(rows, epoch):
            upper = next(size for size in LENGTH_BINS if max(map(row_length, group)) <= size)
            total += times[upper]
    return total * 1.25


def run(output, extra):
    from hanglish.models.sentence_ctc import SentenceCTC, SentenceCTCConfig, training_batch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    random.seed(SEED)
    rows = read_training(extra)
    audit = frame_audit(rows)
    if audit['invalid']:
        raise ValueError('CTC frame or complete-sequence length audit failed')
    output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), ROOT / 'hanglish/models/sentence_ctc.py',
             ROOT / 'hanglish/models/sentence_transducer.py',
             ROOT / 'scripts/train_sentence_transducer.py', ROOT / 'scripts/sentence_data.py',
             ROOT / 'scripts/evaluate_ensemble_regression.py',
             ROOT / 'data/challenge/sentence-v3/train.jsonl',
             ROOT / 'data/challenge/sentence-v3/dev.jsonl',
             ROOT / 'data/challenge/sentence-v3/manifest.json',
             extra / 'train.jsonl', extra / 'manifest.json']
    pins = {str(path): sha(path) for path in paths}
    config = SentenceCTCConfig()
    started = time.monotonic()

    def guard():
        if rss_bytes() > MAX_RSS or time.monotonic() - started > MAX_PREFLIGHT_SECONDS:
            raise RuntimeError('CTC preflight exceeded 3GiB/180seconds')

    model = SentenceCTC(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)

    def step(pairs):
        model.train()
        source, labels, lengths, target_lengths = training_batch(pairs, config)
        optimizer.zero_grad(set_to_none=True)
        begin = time.monotonic()
        loss = model.loss(source, labels, lengths, target_lengths)
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite CTC loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(norm):
            raise RuntimeError('nonfinite CTC gradient')
        optimizer.step()
        guard()
        return dict(elapsed_s=time.monotonic() - begin, loss=float(loss.detach()),
                    gradient_norm=float(norm), rows=len(pairs),
                    source_tokens=source.shape[1], max_target_tokens=int(target_lengths.max()),
                    max_input_frames=int(lengths.max()))

    real = step([(row['input'], row['expected'][0]) for row in rows[:BATCH]])
    steps = []
    for length in LENGTH_BINS:
        # Deliberately synthetic shapes, not linguistic training examples.
        # Repeated literal labels also exercise mandatory blank separation.
        source = ('a ' * length)[:length - 2]
        target = 'a' * (length - 1)
        measured = step([(source, target)] * BATCH)
        if measured['source_tokens'] != length or measured['max_target_tokens'] != length - 1:
            raise ValueError('maximum declared CTC shape was not exercised')
        steps.append(dict(kind='synthetic_length_bin', **measured))
    model.eval()
    source = training_batch([(rows[0]['input'], rows[0]['expected'][0])], config)[0]
    with torch.inference_mode():
        expected, expected_lengths = model(source)
    model.save(output / 'preflight.pt', dict(pins=pins, updates=5, resource_probe_only=True))
    restored = SentenceCTC.load(output / 'preflight.pt')
    with torch.inference_mode():
        actual, actual_lengths = restored(source)
    delta = float((actual - expected).abs().max())
    if delta != 0 or not torch.equal(actual_lengths, expected_lengths):
        raise ValueError('CTC serialized reload differs')
    probe_source = ('a ' * 256)[:510]
    begin = time.monotonic()
    generated = restored.generate([probe_source] * 8)
    generation_s = time.monotonic() - begin
    guard()
    if len(generated) != 8:
        raise ValueError('generation resource probe denominator differs')
    check_pins(pins)
    training_estimate = estimate(steps, rows)
    evaluation_estimate = generation_s * math.ceil(316 / 8) * 1.25
    report = dict(completed=True, phase='ctc_preflight', config=asdict(config),
                  training_rows=len(rows), pins=pins, frame_audit=audit,
                  parameters=sum(p.numel() for p in model.parameters()),
                  real_update=real, updates=steps, reload_max_logit_delta=delta,
                  generation_probe=dict(batch_size=8, source_tokens=512, elapsed_s=generation_s,
                                        responses=generated, synthetic_input=True),
                  estimated_8epoch_training_s=training_estimate,
                  estimated_dev_generation_s=evaluation_estimate,
                  qualifies=training_estimate <= 1800 and evaluation_estimate <= 900,
                  elapsed_s=time.monotonic() - started, peak_rss_bytes=rss_bytes(),
                  test_accessed=False, production_changed=False)
    write_json(output / 'report.json', report)
    print(json.dumps({key: report[key] for key in ('completed', 'parameters', 'qualifies',
          'estimated_8epoch_training_s', 'estimated_dev_generation_s', 'elapsed_s', 'peak_rss_bytes')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--extra', type=Path, default=ROOT / 'data/challenge/sentence-train-extra-v2')
    args = parser.parse_args()
    try:
        run(args.output, args.extra)
    except Exception as error:
        if args.output.is_dir() and not (args.output / 'report.json').exists():
            write_json(args.output / 'failure.json', dict(completed=False, error=repr(error)))
        raise
