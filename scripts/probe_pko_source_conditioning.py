"""TRAIN-inner paired source intervention for the frozen PKO pilot.

Teacher-forced NLL is a diagnostic, never a source-only accuracy measurement.
No training, DEV/TEST reads, or changes to production occur here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import train_pko_seq2seq as trainer
from scripts import preflight_pko_seq2seq as preflight

RUN = ROOT / '.omx/artifacts/pko-t5-pilot-v1'
SEED = 'pko-source-conditioning-v1-20260921'
PINS = {
    RUN / 'report.json': '093456fcd7e695abf5ed84ecd71b5240a696b166c2668b39cb7a0edbce6615e9',
    RUN / 'independent-verification.json': '963c6bd132c1979c752e43bdf3de4de3addadc11faf1eee6c89f22e848e91848',
    Path(trainer.__file__): '7cb34f7cde2ce8a493e2f0bf75793329e63096f8baea9ac42fc7f02a55663113',
    Path(preflight.__file__): '89bc4bd2a9875ace9eb010ca443168df4db84ed34772342dd55923e2ab7518f2',
}
LIMITS = {'rss_bytes': 3 * 1024**3, 'swap_growth_bytes': 512 * 1024**2,
          'whole_seconds': 300., 'update_seconds': 60.}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sample_pairs(rows, groups, *, per_bin=16):
    """Four source-character-length quantiles; hash sample unique groups.

    Selection sees source/ID/group only. Targets and measured losses cannot affect it.
    """
    require(len({r['id'] for r in rows}) == len(rows), 'duplicate row id')
    ordered = sorted(rows, key=lambda r: (len(r['source']), r['id']))
    seen_groups, pairs = set(), []
    for bin_index in range(4):
        bucket = ordered[len(ordered)*bin_index//4:len(ordered)*(bin_index+1)//4]
        bucket.sort(key=lambda r: hashlib.sha256((SEED+'\n'+r['id']).encode()).hexdigest())
        selected = []
        for row in bucket:
            if groups[row['id']] in seen_groups:
                continue
            seen_groups.add(groups[row['id']]); selected.append(row)
            if len(selected) == per_bin:
                break
        require(len(selected) == per_bin and per_bin >= 2, 'insufficient independent groups')
        offset = next((k for k in range(1, per_bin)
                       if all(r['source'] != selected[(j+k)%per_bin]['source']
                              for j, r in enumerate(selected))), None)
        require(offset is not None, 'no source derangement available')
        for j, row in enumerate(selected):
            donor = selected[(j+offset)%per_bin]
            pairs.append({'id': row['id'], 'donor_id': donor['id'], 'bin': bin_index,
                          'source': row['source'], 'shuffled_source': donor['source'],
                          'source_chars': len(row['source']),
                          'shuffled_source_chars': len(donor['source'])})
    return pairs


def summarize(captures):
    require(bool(captures), 'empty capture')
    for row in captures:
        require(type(row['target_tokens']) is int and row['target_tokens'] > 0,
                'invalid target token count')
        require(all(type(row[k]) in (int, float) and math.isfinite(row[k]) and row[k] >= 0
                    for k in ('normal_nll', 'shuffled_nll')), 'invalid NLL')
    deltas = [r['shuffled_nll']-r['normal_nll'] for r in captures]
    tokens = sum(r['target_tokens'] for r in captures)
    normal = sum(r['normal_nll']*r['target_tokens'] for r in captures)/tokens
    shuffled = sum(r['shuffled_nll']*r['target_tokens'] for r in captures)/tokens
    rng = random.Random(SEED)
    means = sorted(sum(rng.choices(deltas, k=len(deltas)))/len(deltas) for _ in range(2000))
    interval = [means[49], means[1949]]
    return {'rows': len(captures), 'target_tokens': tokens,
            'normal_token_nll': normal, 'shuffled_token_nll': shuffled,
            'token_weighted_delta': shuffled-normal,
            'mean_row_delta': sum(deltas)/len(deltas),
            'mean_row_delta_bootstrap95': interval,
            'bootstrap_scope': 'descriptive row resampling; reused donor sources make an independence-based confidence interpretation inappropriate',
            'normal_lower_rows': sum(d > 0 for d in deltas),
            'source_signal_gate': interval[0] > 0 and shuffled-normal >= .1,
            'gate_scope': 'diagnostic source sensitivity; not spelling accuracy or a promotion gate'}


class Watchdog(preflight.ResourceWatchdog):
    def _monitor(self):
        while not self.stop_event.wait(self.interval_s):
            try:
                preflight.append_jsonl(self.trace, self._sample('watchdog'), self.lock)
            except BaseException as error:
                self.failure = f'{type(error).__name__}: {error}'
                preflight.write_json(self.trace.parent/'report.json', {
                    'schema': SEED, 'completed': False, 'error': self.failure,
                    'dev_opened': False, 'test_opened': False, 'production_changed': False})
                os.kill(os.getpid(), signal.SIGUSR1)
                if not self.stop_event.wait(5.):
                    os._exit(124)
                return


def prepare():
    for path, digest in PINS.items():
        require(preflight.sha256_file(path) == digest, 'frozen evidence changed: '+str(path))
    report = json.loads((RUN/'report.json').read_text())
    proof = json.loads((RUN/'independent-verification.json').read_text())
    require(report['completed'] and proof['verified'], 'unverified model')
    pins = dict(PINS)
    for name, digest in report['model']['files'].items():
        path = RUN/'model'/name
        require(preflight.sha256_file(path) == digest, 'model file changed')
        pins[path] = digest
    _, inner, partition_proof = trainer.load_train_partition()
    partition = json.loads(trainer.PARTITION.read_text())
    groups = {ident: g['group_sha256'] for g in partition['groups'] for ident in g['ids']}
    pairs = sample_pairs(inner, groups)
    for path, digest in preflight.EXPECTED_FILES.items():
        if path != preflight.BASE/'dev.jsonl':
            pins[path] = digest
    pins[Path(__file__)] = preflight.sha256_file(Path(__file__))
    return pairs, {r['id']: r['target'] for r in inner}, report, pins, partition_proof


def run(output):
    import torch
    from transformers import T5ForConditionalGeneration, T5TokenizerFast
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        pairs, targets, training, pins, partition = prepare()
        preflight.write_json(output/'selection.json', {'seed': SEED, 'pairs': pairs,
            'selected_without_targets': True, 'partition': partition,
            'gate': 'bootstrap95 lower mean row delta >0 and token-weighted delta >=0.1 nats'})
        shutil.copyfile(__file__, output/'diagnostic-source.py')
        baseline_swap = preflight.system_swap_bytes()
        torch.set_num_threads(2); torch.set_num_interop_threads(1)
        captured = []
        with Watchdog(started, baseline_swap, output/'resources.jsonl',
                      limits=LIMITS, interval_s=1.) as watchdog:
            with watchdog.operation('load_model', update=True):
                tokenizer = T5TokenizerFast.from_pretrained(str(RUN/'model'), local_files_only=True)
                preflight.adapt_lossless_tokenizer(tokenizer)
                model = T5ForConditionalGeneration.from_pretrained(
                    str(RUN/'model'), local_files_only=True, weights_only=True, torch_dtype='float32')
                model.eval()
                require(preflight.parameter_fingerprint(model) == training['model']['parameter_sha256'],
                        'runtime parameters differ')
            with torch.inference_mode():
                for pair in pairs:
                    labels = trainer.tokenizer_encode(tokenizer, targets[pair['id']])['input_ids']
                    row = dict(pair, target_tokens=int(labels.numel()))
                    for field, key in [('source', 'normal_nll'), ('shuffled_source', 'shuffled_nll')]:
                        encoded = trainer.tokenizer_encode(tokenizer, pair[field])
                        row[field+'_tokens'] = int(encoded['input_ids'].numel())
                        with watchdog.operation(pair['id']+':'+key, update=True):
                            row[key] = float(model(**encoded, labels=labels).loss)
                    captured.append(row)
                    preflight.append_jsonl(output/'capture.jsonl', row)
            final_parameters = preflight.parameter_fingerprint(model)
            require(final_parameters == training['model']['parameter_sha256'], 'diagnostic altered weights')
            observation = watchdog._sample('final')
        for path, digest in pins.items():
            require(preflight.sha256_file(path) == digest, 'input changed during diagnostic')
        result = {'schema': SEED, 'completed': True, 'metrics': summarize(captured),
            'pins': {str(p.relative_to(ROOT)): v for p, v in pins.items()},
            'capture_sha256': preflight.sha256_file(output/'capture.jsonl'),
            'selection_sha256': preflight.sha256_file(output/'selection.json'),
            'resources': {'limits': LIMITS, 'baseline_swap_bytes': baseline_swap, 'final': observation,
                          'trace_sha256': preflight.sha256_file(output/'resources.jsonl')},
            'elapsed_s': time.monotonic()-started, 'parameter_sha256': final_parameters,
            'training_performed': False, 'generation_performed': False,
            'teacher_forced_labels_from_TRAIN_inner': True,
            'dev_opened': False, 'test_opened': False, 'production_changed': False}
        preflight.write_json(output/'report.json', result)
        print(json.dumps({'completed': True, 'metrics': result['metrics'], 'elapsed_s': result['elapsed_s']}))
        return result
    except BaseException as error:
        preflight.write_json(output/'report.json', {'schema': SEED, 'completed': False,
            'error': f'{type(error).__name__}: {error}', 'dev_opened': False,
            'test_opened': False, 'production_changed': False})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
