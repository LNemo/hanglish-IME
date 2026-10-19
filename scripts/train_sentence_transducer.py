"""Bounded whole-sentence restoration learning, isolated from production weights."""
import argparse
from dataclasses import asdict
import hashlib
import fcntl
import json
import math
from pathlib import Path
import random
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from hanglish.models.sentence_transducer import (
    PAD, EOS, SentenceTransducer, SentenceTransducerConfig,
    encode_source, encode_target, training_batch,
)
from hanglish.core.phonology import romanize_word
from scripts.sentence_data import read_split, DEFAULT_CORPUS
from scripts.evaluate_ensemble_regression import rss_bytes
from scripts.sentence_transducer_recovery import save_checkpoint, restore_checkpoint

SEED = 20260921
BATCH = 16
MAX_RSS = 3 * 1024**3
MAX_TRAIN_SECONDS = 1800
MAX_EVAL_SECONDS = 900
MAX_PREFLIGHT_SECONDS = 180
LENGTH_BINS = (128, 256, 384, 512)
CHECKPOINT_INTERVAL = 50


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def read_training(extra=None):
    rows = read_split(DEFAULT_CORPUS, 'train')
    if extra is not None:
        rows += read_split(extra, 'train')
    if len({row['id'] for row in rows}) != len(rows) or len({row['input'] for row in rows}) != len(rows):
        raise ValueError('duplicate training identity or source')
    for row in rows:
        # Validate complete source and target lengths before allocating a model.
        training_batch([(row['input'], row['expected'][0])])
    return rows


def batches(rows, epoch):
    rng = random.Random(SEED + epoch)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    groups = []
    for offset in range(0, len(shuffled), 256):
        bucket = sorted(shuffled[offset:offset + 256], key=lambda row: max(
            len(encode_source(row['input'])), len(encode_target(row['expected'][0])) + 1))
        groups.extend(bucket[start:start + BATCH] for start in range(0, len(bucket), BATCH))
    rng.shuffle(groups)
    return groups


def optimizer_step(model, optimizer, rows, *, padded_tokens=None):
    model.train()
    source, shifted, expected = training_batch([(row['input'], row['expected'][0]) for row in rows], model.config.max_tokens)
    if padded_tokens is not None:
        def full(tensor):
            if tensor.shape[1] > padded_tokens:
                raise ValueError('resource probe must not truncate real sentences')
            return torch.nn.functional.pad(tensor, (0, padded_tokens - tensor.shape[1]), value=PAD)
        source, shifted, expected = map(full, (source, shifted, expected))
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    logits = model(source, shifted)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), expected.flatten(), ignore_index=PAD)
    if not torch.isfinite(loss):
        raise RuntimeError('nonfinite restoration loss')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    if not torch.isfinite(norm):
        raise RuntimeError('nonfinite restoration gradient')
    optimizer.step()
    return dict(loss=float(loss.detach()), gradient_norm=float(norm),
                elapsed_s=time.monotonic() - started, rows=len(rows),
                source_tokens=source.shape[1], target_tokens=expected.shape[1],
                nonpadding_target_tokens=int(expected.ne(PAD).sum()))


def input_pins(extra=None):
    files = [Path(__file__), ROOT/'hanglish/models/sentence_transducer.py',
             ROOT/'scripts/sentence_data.py', ROOT/'hanglish/core/phonology.py',
             ROOT/'scripts/sentence_transducer_recovery.py',
             DEFAULT_CORPUS/'train.jsonl', DEFAULT_CORPUS/'dev.jsonl', DEFAULT_CORPUS/'manifest.json']
    if extra is not None:
        files.extend((extra/'train.jsonl', extra/'manifest.json'))
    return {str(path): sha(path) for path in files}


def check_pins(pins):
    if any(sha(path) != digest for path, digest in pins.items()):
        raise RuntimeError('training/evaluation input or implementation changed')


def guard(started, seconds):
    if rss_bytes() > MAX_RSS or time.monotonic() - started > seconds:
        raise RuntimeError('sentence restoration resource limit exceeded')


def preflight(output, rows, pins, config):
    model = SentenceTransducer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    started = time.monotonic()
    steps = []
    by_length = sorted(rows, key=row_length, reverse=True)
    for index, length in enumerate(LENGTH_BINS):
        group = [row for row in by_length if row_length(row) <= length][:BATCH]
        if len(group) != BATCH:
            raise ValueError('not enough complete training sentences for length probe')
        step = optimizer_step(model, optimizer, group, padded_tokens=length)
        steps.append(dict(step=index + 1, kind='length_bin', measured_length=length, **step))
        guard(started, MAX_PREFLIGHT_SECONDS)
    longest = by_length[:BATCH]
    guard(started, MAX_PREFLIGHT_SECONDS)
    probe = training_batch([(longest[0]['input'], longest[0]['expected'][0])])
    model.eval()
    with torch.inference_mode():
        expected = model(probe[0], probe[1])
    model.save(output/'preflight.pt', dict(input_contract='roman_sentence_only', pins=pins, updates=4))
    reloaded = SentenceTransducer.load(output/'preflight.pt')
    with torch.inference_mode():
        delta = float((expected - reloaded(probe[0], probe[1])).abs().max())
    if delta != 0:
        raise RuntimeError('checkpoint reload logits differ')
    # Exercise the actual autoregressive path at its maximum decoder length.
    # Suppress EOS only in this explicitly synthetic resource probe; do not
    # change weights or the generation policy used for accuracy evaluation.
    probe_sources = [row['input'] + ' ' * (config.max_tokens - 2 - len(row['input'].encode()))
                     for row in longest[:8]]
    original_decode = reloaded.decode
    def without_eos(*args):
        logits = original_decode(*args)
        logits[..., EOS] = -torch.inf
        return logits
    decode_started = time.monotonic()
    with patch.object(reloaded, 'decode', side_effect=without_eos):
        generated = reloaded.generate(probe_sources)
    generation = dict(kind='forced_no_eos_maximum_shape', batch_size=8,
                      source_tokens=config.max_tokens, max_generated_tokens=config.max_tokens,
                      elapsed_s=time.monotonic()-decode_started,
                      reasons=[row['reason'] for row in generated],
                      generated_tokens=[row['generated_tokens'] for row in generated],
                      source_ids=[row['id'] for row in longest[:8]], synthetic_right_padding=True)
    if (len(generated) != 8 or any(row['reason'] != 'token_limit'
                                  or row['generated_tokens'] != config.max_tokens for row in generated)):
        raise RuntimeError('maximum generation probe did not exercise its declared bound')
    guard(started, MAX_PREFLIGHT_SECONDS)
    check_pins(pins)
    result = dict(completed=True, phase='preflight', pins=pins, config=asdict(config),
                  rows=len(rows), parameters=sum(p.numel() for p in model.parameters()),
                  updates=steps, peak_rss_bytes=rss_bytes(), elapsed_s=time.monotonic()-started,
                  generation_probe=generation,
                  reload_max_logit_delta=delta, test_accessed=False, production_changed=False)
    write_json(output/'report.json', result)
    print(json.dumps(result), flush=True)


def qualify(path, rows, pins, config):
    report = json.loads(path.read_text())
    if (not report.get('completed') or report.get('phase') != 'preflight' or report.get('pins') != pins
            or report.get('config') != asdict(config) or report.get('rows') != len(rows)
            or report.get('reload_max_logit_delta') != 0):
        raise ValueError('matching measured preflight required')
    peak=report.get('peak_rss_bytes')
    if (len(report.get('updates', [])) != 4 or not isinstance(peak,(int,float))
            or not math.isfinite(peak) or not 0 < peak <= MAX_RSS):
        raise ValueError('resource preflight did not qualify')
    if not 0 < report.get('elapsed_s', 0) <= MAX_PREFLIGHT_SECONDS:
        raise ValueError('preflight elapsed bound exceeded')
    expected_kinds=['length_bin']*4
    if [step.get('kind') for step in report['updates']] != expected_kinds:
        raise ValueError('preflight update kinds differ')
    for step, length in zip(report['updates'], LENGTH_BINS):
        if any(not isinstance(step[key], (int, float)) or not math.isfinite(step[key]) or step[key] < 0
               for key in ('loss', 'gradient_norm', 'elapsed_s')):
            raise ValueError('invalid preflight optimizer evidence')
        for key in ('source_tokens','target_tokens'):
            if not isinstance(step.get(key),int) or not 1 <= step[key] <= config.max_tokens:
                raise ValueError('invalid measured training shape')
        if step.get('measured_length') != length or step['source_tokens'] != length or step['target_tokens'] != length:
            raise ValueError('length-bin shape was not measured')
        if step.get('rows') != BATCH or not isinstance(step.get('nonpadding_target_tokens'),int) or step['nonpadding_target_tokens'] <= 0:
            raise ValueError('invalid measured training batch')
    worst=report['updates'][-1]
    if worst['source_tokens'] != config.max_tokens or worst['target_tokens'] != config.max_tokens:
        raise ValueError('maximum training shape was not measured')
    probe=report.get('generation_probe',{})
    if (probe.get('kind') != 'forced_no_eos_maximum_shape' or probe.get('batch_size') != 8
            or probe.get('source_tokens') != config.max_tokens
            or probe.get('max_generated_tokens') != config.max_tokens
            or probe.get('reasons') != ['token_limit']*8
            or probe.get('generated_tokens') != [config.max_tokens]*8
            or not isinstance(probe.get('elapsed_s'),(int,float))
            or not math.isfinite(probe['elapsed_s']) or probe['elapsed_s'] <= 0):
        raise ValueError('maximum autoregressive generation was not measured')
    dev_count=json.loads((DEFAULT_CORPUS/'manifest.json').read_text())['counts']['dev']
    if probe['elapsed_s'] * math.ceil(dev_count/8) * 1.25 > MAX_EVAL_SECONDS:
        raise ValueError('conservative dev generation estimate exceeds time budget')
    return report


def row_length(row):
    return max(len(encode_source(row['input'])), len(encode_target(row['expected'][0]))+1)


def estimate_training_seconds(report, rows, epochs):
    # Each batch uses the measured upper bound for BOTH source and target length.
    # Monotone timings avoid crediting noise where a larger bin appears faster.
    timing={}; maximum=0.
    for step in report['updates']:
        maximum=max(maximum,step['elapsed_s']); timing[step['measured_length']]=maximum
    total=0.
    for epoch in range(epochs):
        for group in batches(rows, epoch):
            length=max(row_length(row) for row in group)
            upper=next(bound for bound in LENGTH_BINS if length<=bound)
            total+=timing[upper]
    return total*1.25


def evaluate(model, output):
    rows = read_split(DEFAULT_CORPUS, 'dev')
    inputs = [dict(id=row['id'], input=row['input']) for row in rows]
    predictions = []
    started = time.monotonic()
    with (output/'dev-predictions.jsonl').open('x') as stream:
        for offset in range(0, len(inputs), 8):
            items = inputs[offset:offset+8]
            responses = model.generate([item['input'] for item in items])
            for item, response in zip(items, responses):
                prediction = dict(**item, response=response)
                predictions.append(prediction)
                stream.write(json.dumps(prediction, ensure_ascii=False)+'\n')
            stream.flush()
            guard(started, MAX_EVAL_SECONDS)
            if len(predictions) % 40 == 0:
                print(json.dumps(dict(phase='dev_generation', captured=len(predictions), total=len(inputs))), flush=True)
    # Expected strings never cross the generation boundary. Join only after all
    # input-only predictions have been written; incomplete outputs count as wrong.
    outcomes = [dict(id=prediction['id'], input=row['input'], expected=row['expected'],
                     **prediction['response'], correct=prediction['response']['text'] in row['expected'])
                for prediction, row in zip(predictions, rows)]
    return dict(rows=len(rows), correct=sum(row['correct'] for row in outcomes),
                complete=sum(row['reason']=='complete' for row in outcomes),
                forward_equal=sum(row['text'] is not None and romanize_word(row['text']) == row['input'] for row in outcomes),
                elapsed_s=time.monotonic()-started, outcomes=outcomes)


def recovery_identity(rows, pins, config, preflight_path, epochs):
    schedules=[[[row['id'] for row in group] for group in batches(rows, epoch)] for epoch in range(epochs)]
    return dict(pins=pins, config=asdict(config), seed=SEED, batch_size=BATCH, epochs=epochs,
                preflight_sha256=sha(preflight_path), training_rows=len(rows),
                row_ids_sha256=hashlib.sha256(json.dumps([row['id'] for row in rows]).encode()).hexdigest(),
                schedule_sha256=hashlib.sha256(json.dumps(schedules).encode()).hexdigest())


def validate_cursor(state, rows, epochs):
    epoch=state['next_epoch']; batch_index=state['next_batch']
    if not isinstance(epoch,int) or not 0 <= epoch <= epochs or not isinstance(batch_index,int):
        raise ValueError('invalid recovery cursor')
    if epoch==epochs:
        if batch_index!=0:raise ValueError('completed cursor has a pending batch')
    elif not 0 <= batch_index < len(batches(rows,epoch)):
        raise ValueError('recovery batch cursor exceeds schedule')
    expected=sum(len(batches(rows,index)) for index in range(epoch))+batch_index
    if expected!=state['updates'] or len(state['epochs_report'])!=epoch:
        raise ValueError('recovery cursor/update accounting mismatch')
    if batch_index==0 and (state['epoch_loss_sum']!=0 or state['epoch_token_sum']!=0):
        raise ValueError('new epoch has uncommitted loss totals')


def train(output, rows, pins, config, preflight_path, epochs, resume=False):
    if not 1 <= epochs <= 20:
        raise ValueError('epochs must be between1and20')
    qualified = qualify(preflight_path, rows, pins, config)
    estimated = estimate_training_seconds(qualified, rows, epochs)
    if estimated > MAX_TRAIN_SECONDS:
        raise ValueError(f'planned training estimate {estimated:.1f}s exceeds {MAX_TRAIN_SECONDS}s')
    model = SentenceTransducer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    identity=recovery_identity(rows,pins,config,preflight_path,epochs)
    state=dict(updates=0,next_epoch=0,next_batch=0,epochs_report=[],epoch_loss_sum=0.,epoch_token_sum=0,
               training_elapsed_s=0.,peak_rss_bytes=0,discarded_tail_updates=0)
    if resume:
        state=restore_checkpoint(output,model,optimizer,identity)
        validate_cursor(state,rows,epochs)
    if state['training_elapsed_s']>=MAX_TRAIN_SECONDS or state['peak_rss_bytes']>MAX_RSS:
        raise RuntimeError('restored run already exceeds resource budget')
    prior_seconds=state['training_elapsed_s']
    started = time.monotonic()
    with (output/'updates.jsonl').open('a' if resume else 'x') as stream:
        for epoch in range(state['next_epoch'],epochs):
            schedule=batches(rows,epoch)
            first=state['next_batch'] if epoch==state['next_epoch'] else 0
            for batch_index in range(first,len(schedule)):
                group=schedule[batch_index]
                step = optimizer_step(model, optimizer, group)
                state['updates']+=1
                state['training_elapsed_s']=prior_seconds+time.monotonic()-started
                state['peak_rss_bytes']=max(state['peak_rss_bytes'],rss_bytes())
                step.update(step=state['updates'], epoch=epoch+1, ids=[row['id'] for row in group],
                            training_elapsed_s=state['training_elapsed_s'],peak_rss_bytes=state['peak_rss_bytes'])
                stream.write(json.dumps(step)+'\n'); stream.flush()
                state['epoch_loss_sum']+=step['loss']*step['nonpadding_target_tokens']
                state['epoch_token_sum']+=step['nonpadding_target_tokens']
                if batch_index+1==len(schedule):
                    state['epochs_report'].append(dict(epoch=epoch+1,target_loss=state['epoch_loss_sum']/state['epoch_token_sum'],tokens=state['epoch_token_sum']))
                    state.update(next_epoch=epoch+1,next_batch=0,epoch_loss_sum=0.,epoch_token_sum=0)
                else:
                    state.update(next_epoch=epoch,next_batch=batch_index+1)
                if state['updates']%CHECKPOINT_INTERVAL==0 or state['next_batch']==0:
                    save_checkpoint(output,model,optimizer,identity,state,stream)
                guard(started-prior_seconds, MAX_TRAIN_SECONDS)
                if state['updates']%50==0:
                    print(json.dumps(dict(phase='training', step=state['updates'], epoch=epoch+1, loss=step['loss'], elapsed_s=state['training_elapsed_s'])), flush=True)
    updates=state['updates']
    epochs_report=state['epochs_report']
    check_pins(pins)
    model.save(output/'model.pt', dict(input_contract='roman_sentence_only', pins=pins,
                                     seed=SEED, epochs=epochs, updates=updates, training_rows=len(rows)))
    # Actual held-out prediction uses the serialized model, not an in-memory-only state.
    model = SentenceTransducer.load(output/'model.pt')
    train_seconds = prior_seconds+time.monotonic()-started
    metrics = evaluate(model, output)
    check_pins(pins)
    report = dict(completed=True, phase='trained_and_evaluated', input_contract='roman_sentence_only',
                  config=asdict(config), pins=pins, preflight_sha256=sha(preflight_path),
                  epochs=epochs_report, updates=updates, training_rows=len(rows),
                  training_elapsed_s=train_seconds, estimated_training_s=estimated,
                  dev=metrics, peak_rss_bytes=max(state['peak_rss_bytes'],rss_bytes()), test_accessed=False,
                  discarded_tail_updates=state.get('discarded_tail_updates',0), resumed=resume,
                  production_changed=False, promoted=False,
                  model_sha256=sha(output/'model.pt'), predictions_sha256=sha(output/'dev-predictions.jsonl'))
    write_json(output/'report.json', report)
    print(json.dumps(dict(completed=True, updates=updates, correct=metrics['correct'], rows=metrics['rows'],
                         complete=metrics['complete'], forward_equal=metrics['forward_equal'],
                         training_seconds=train_seconds, generation_seconds=metrics['elapsed_s'])), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--extra', type=Path)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--qualification', type=Path)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--embedding-scale', type=float)
    args = parser.parse_args()
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.manual_seed(SEED); random.seed(SEED)
    rows = read_training(args.extra)
    pins = input_pins(args.extra)
    config = SentenceTransducerConfig(embedding_scale=args.embedding_scale)
    if args.resume:
        if args.preflight or not (args.output/'recovery.pt').exists() or (args.output/'report.json').exists() or (args.output/'failure.json').exists():
            raise ValueError('resume requires an interrupted checkpointed run without terminal report/failure')
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    lock=(args.output/'.run.lock').open('a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError('training output is already owned by a live process')
    try:
        if args.preflight:
            preflight(args.output, rows, pins, config)
        else:
            if args.qualification is None:
                raise ValueError('a measured preflight report is required')
            train(args.output, rows, pins, config, args.qualification, args.epochs,args.resume)
    except Exception as error:
        write_json(args.output/'failure.json', dict(error_type=type(error).__name__, error=str(error),
                                                   pins=pins, production_changed=False))
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    main()
