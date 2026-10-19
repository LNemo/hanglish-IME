"""Measure a pinned local cross-encoder's training resources, not its accuracy."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from hanglish.models.semantic_ranker import encode_pairs
from scripts.train_natural_sentence_ranker import rss_bytes
from scripts.train_sentence_ranker import _optimizer, _set_trainable_layers

SOURCE = ROOT / 'data/external/klue-roberta-small'
REVISION = 'b6b4c36d827e0293ae2fcf04d527072f10a23064'
WEIGHT_SHA = '737e7dbcb9f159d4719a06095268836a2634a673ac925de96c13c9d0bf4d895b'
SOURCE_SHA = '5166197923917d229b81d58f0e3561de3b141222d8597b7d0313bab4bb0b7d9c'
LIMIT_BYTES = 3 * 1024**3


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_source():
    if sha256(SOURCE / 'SOURCE.json') != SOURCE_SHA:
        raise ValueError('unexpected RoBERTa source manifest')
    source = json.loads((SOURCE / 'SOURCE.json').read_text())
    if (source['repository'] != 'klue/roberta-small' or source['revision'] != REVISION
            or source['files']['model.safetensors'] != WEIGHT_SHA):
        raise ValueError('unexpected RoBERTa source')
    expected = {**source['files'], source['license']['file']: source['license']['sha256']}
    for filename, digest in expected.items():
        if sha256(SOURCE / filename) != digest:
            raise ValueError(f'RoBERTa source checksum mismatch: {filename}')
    return {**expected, 'SOURCE.json': sha256(SOURCE / 'SOURCE.json')}


def load_cross_encoder():
    validate_source()
    config = AutoConfig.from_pretrained(SOURCE, local_files_only=True, trust_remote_code=False)
    config.torchscript = False
    config.return_dict = True
    config.num_labels = 1
    model, loading = AutoModelForSequenceClassification.from_pretrained(
        SOURCE, config=config, local_files_only=True, trust_remote_code=False,
        use_safetensors=True, output_loading_info=True, attn_implementation='eager')
    missing = set(loading['missing_keys'])
    expected = {name for name, _ in model.named_parameters() if name.startswith('classifier.')}
    if (missing != expected or loading['mismatched_keys'] or loading['error_msgs']
            or any(not name.startswith('lm_head.') for name in loading['unexpected_keys'])):
        raise ValueError(f'unexpected encoder loading: {loading}')
    tokenizer = AutoTokenizer.from_pretrained(SOURCE, local_files_only=True, trust_remote_code=False)
    pair = tokenizer('후보', '주변 문맥')
    if tokenizer.pad_token_id != config.pad_token_id or config.pad_token_id != 1:
        raise ValueError('padding mismatch')
    if any(value != 0 for value in pair.get('token_type_ids', [])):
        raise ValueError('RoBERTa pair segment IDs exceed its single token type')
    return model, tokenizer, loading


def memory(device):
    driver = torch.mps.driver_allocated_memory() if device == 'mps' else 0
    # Conservative bound: unified allocations may also be counted in RSS.
    return {'peak_rss_bytes': rss_bytes(), 'mps_driver_bytes': driver,
            'conservative_combined_bytes': rss_bytes() + driver}


def synchronize(device):
    if device == 'mps':
        torch.mps.synchronize()


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable in this process')
    torch.set_num_threads(2)
    torch.manual_seed(20260914)
    before = validate_source()
    model, tokenizer, loading = load_cross_encoder()
    layers = _set_trainable_layers(model, 2)
    model.to(args.device)
    optimizer = _optimizer(model, SimpleNamespace(head_lr=5e-5, encoder_lr=1e-5, weight_decay=.01), False)
    # Resource-only synthetic groups. No benchmark rows are read.
    candidates = ['문맥', '문맹', '문명', '문건'] * (args.batch_size // 4)
    contexts = ['이 문장은 한국어 입력 변환의 [MASK] 정보를 비교하는 실험이다. ' * 8] * args.batch_size
    encoded = encode_pairs(tokenizer, contexts, candidates, 96)
    # Force the maximum workload; ordinary short candidate contexts are smaller.
    encoded = tokenizer.pad(encoded, padding='max_length', max_length=96, return_tensors='pt')
    encoded = {key: value.to(args.device) for key, value in encoded.items()}
    times, losses, memories = [], [], []
    started = time.monotonic()
    model.train()
    for step in range(11):
        synchronize(args.device)
        tick = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        scores = model(**encoded).logits.reshape(-1, 4)
        loss = (torch.logsumexp(scores, dim=1) - scores[:, 0]).mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('nonfinite loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not bool(torch.isfinite(norm)):
            raise RuntimeError('nonfinite gradients')
        optimizer.step()
        synchronize(args.device)
        elapsed = (time.perf_counter() - tick) * 1000
        current = memory(args.device)
        memories.append(current)
        if current['conservative_combined_bytes'] > LIMIT_BYTES or time.monotonic() - started > 180:
            raise RuntimeError(f'resource cap exceeded: {current}')
        if step >= 3:
            times.append(elapsed)
            losses.append(float(loss.detach().cpu()))
    del optimizer
    model.eval()
    inference = []
    small = {key: value[:8] for key, value in encoded.items()}
    with torch.inference_mode():
        for step in range(11):
            synchronize(args.device)
            tick = time.perf_counter()
            logits = model(**small).logits
            synchronize(args.device)
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError('nonfinite inference')
            if step >= 3:
                inference.append((time.perf_counter() - tick) * 1000)
    # Check the saved tokenizer preserves the source's all-zero pair type IDs.
    with tempfile.TemporaryDirectory() as temporary:
        tokenizer.save_pretrained(temporary)
        reloaded = AutoTokenizer.from_pretrained(temporary, local_files_only=True)
        original = encode_pairs(tokenizer, contexts[:2], candidates[:2], 96)
        restored = encode_pairs(reloaded, contexts[:2], candidates[:2], 96)
        if set(original) != set(restored) or any(not torch.equal(original[k], restored[k]) for k in original):
            raise RuntimeError('tokenizer save/load changed pair encoding')
    if before != validate_source():
        raise RuntimeError('source changed during pilot')
    report = {'device': args.device, 'threads': 2, 'batch_size': args.batch_size,
              'sequence_length': 96, 'inference_batch_size': 8, 'warmup_steps': 3,
              'measured_steps': len(times), 'train_step_ms': times,
              'train_median_ms': statistics.median(times),
              'inference_ms': inference, 'inference_median_ms': statistics.median(inference),
              'memory_peak': {key: max(row[key] for row in memories) for key in memories[0]},
              'trainable_layers': layers, 'finite_loss_and_gradients': True,
              'resource_cap_bytes': LIMIT_BYTES, 'passed': True,
              'source_hashes': before, 'loading_info': loading,
              'torch_version': torch.__version__, 'attention_implementation': 'eager',
              'tokenizer_roundtrip_verified': True,
              'accuracy_measured': False, 'checkpoint_saved': False, 'test_data_read': False,
              'script_sha256': sha256(Path(__file__))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=('cpu', 'mps'), required=True)
    parser.add_argument('--batch-size', type=int, choices=(8, 16, 32), default=16)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
