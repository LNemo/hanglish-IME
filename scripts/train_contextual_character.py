"""Matched FIT training with complete-output contextual character states."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_character_edge_full_fit import (
    ResourceGuard, authenticate_inputs, append_jsonl, build_row, digest,
    iter_captures, listwise_loss, read_json, require, row_metrics, verify_pins,
)

PLAN = ROOT / '.omx/plans/contextual-character-fit-v1.md'
MODEL = ROOT / 'hanglish/models/contextual_character_residual.py'
PARENT_MODEL = ROOT / 'hanglish/models/character_edge_residual.py'
PILOT_DATA = ROOT / '.omx/artifacts/character-edge-residual-pilot-v1/data.jsonl'
PILOT_REPORT = PILOT_DATA.parent / 'report.json'
RECIPE = dict(epochs=4, learning_rate=.001, weight_decay=.01, clip_norm=1.,
              seed=20260923, threads=2, interop_threads=1, candidate_chunk=32)


def score_chunks(model, row, chunk_size=32):
    """No-grad caller determines whether scores retain an autograd graph."""
    import torch
    require(type(chunk_size) is int and chunk_size > 0, 'positive chunk required')
    pieces = []
    for start in range(0, len(row['candidates']), chunk_size):
        prepared = model.prepare(row['input'], row['candidates'][start:start+chunk_size],
                                 row['base_scores'][start:start+chunk_size])
        pieces.append(model(prepared)['scores'])
    return torch.cat(pieces)


def backward_listwise_chunks(model, row, chunk_size=32):
    """Exact listwise objective with recomputed, bounded activation batches.

    All chunk gradients accumulate before the caller performs one optimizer step.
    The model has no stochastic/dropout layers. Floating reduction order differs
    from an unchunked autograd graph, so tests compare with numerical tolerance.
    """
    import torch
    positive = row['positive_indices']
    require(bool(positive), 'positive row required')
    with torch.no_grad():
        scores = score_chunks(model, row, chunk_size)
        loss = listwise_loss(scores, positive)
        require(bool(torch.isfinite(loss)), 'nonfinite loss')
        derivative = torch.softmax(scores, dim=0)
        derivative[positive] -= torch.softmax(scores[positive], dim=0)
    for start in range(0, len(row['candidates']), chunk_size):
        prepared = model.prepare(row['input'], row['candidates'][start:start+chunk_size],
                                 row['base_scores'][start:start+chunk_size])
        recomputed = model(prepared)['scores']
        require(torch.allclose(recomputed.detach(), scores[start:start+chunk_size],
                               atol=1e-7, rtol=1e-7), 'score changed during gradient replay')
        recomputed.backward(derivative[start:start+chunk_size])
    return float(loss)


def authenticate(scope):
    authenticated = authenticate_inputs()
    pins = authenticated['pins']
    for path in (PLAN, MODEL, PARENT_MODEL, Path(__file__).resolve(),
                 ROOT/'scripts/train_character_edge_full_fit.py',
                 ROOT/'scripts/current_recombination_selection.py',
                 ROOT/'hanglish/core/source_correspondence.py'):
        pins[str(path)] = digest(path)
    if scope == 'pilot':
        old_report = read_json(PILOT_REPORT)
        require(digest(PILOT_DATA) == old_report['hashes']['data.jsonl']
                == 'e015df25ce53553dea73cf239c6a48a166633ae7b41aecb67ddf9e4c28377f80',
                'source-selected pilot data changed')
        rows = [json.loads(line) for line in PILOT_DATA.open()]
        require(len(rows) == 32 and len({row['id'] for row in rows}) == 32,
                'pilot denominator differs')
        for row in rows:
            row['no_pool'] = not bool(row['candidates'])
        authenticated['pilot_rows'] = rows
        for path in (PILOT_DATA, PILOT_REPORT):
            pins[str(path)] = digest(path)
    else:
        pilot = ROOT / '.omx/artifacts/contextual-character-pilot-v1'
        proof = read_json(pilot/'independent-verification.json')
        report = read_json(pilot/'report.json')
        require(proof['verified'] and report['completed']
                and proof['report_sha256'] == digest(pilot/'report.json')
                and report['model_source_sha256'] == digest(MODEL)
                and report['runner_sha256'] == digest(Path(__file__))
                and report['recipe'] == RECIPE,
                'verified matched pilot required')
        for name in ('report.json', 'independent-verification.json'):
            pins[str(pilot/name)] = digest(pilot/name)
    return authenticated


def rows_for(authenticated, scope):
    if scope == 'pilot':
        yield from authenticated['pilot_rows']
    else:
        for identity, scored in iter_captures(authenticated['shards']):
            yield build_row(identity, scored, authenticated['labels'][identity],
                            authenticated['calibration'])


def observe(guard, stage, scope):
    value = guard.observe(stage)
    require(value['seconds'] <= (600 if scope == 'pilot' else 1800),
            'contextual training time limit exceeded')


def evaluate(model, authenticated, scope, output, phase, guard, initial=False):
    import torch
    records, losses = [], []
    model.eval()
    with torch.no_grad():
        for index, row in enumerate(rows_for(authenticated, scope)):
            started = time.monotonic()
            values = score_chunks(model, row) if row['candidates'] else []
            if initial and row['candidates']:
                require(torch.equal(values, torch.tensor(row['base_scores'], dtype=torch.float64)),
                        'initial exact baseline changed')
            if row['positive_indices']:
                losses.append(float(listwise_loss(values, row['positive_indices'])))
            result = row_metrics(row, values)
            result.update(no_pool=row['no_pool'], unsupported=row['unsupported'])
            records.append(result)
            append_jsonl(output/f'{phase}.jsonl', result)
            guard.require_row_time(started)
            if (index+1) % 32 == 0:
                observe(guard, f'{phase}:row-{index+1}', scope)
    return records, sum(losses)/len(losses)


def summarize(records):
    return {'rows': len(records), **{
        key: sum(int(row[key]) for row in records)
        for key in ('first', 'top5', 'pool_covered', 'no_pool', 'unsupported')}}


def execute(output, scope):
    output.mkdir(parents=True, exist_ok=False)
    guard = None
    try:
        auth = authenticate(scope)
        guard = ResourceGuard(output)
        observe(guard, 'authenticated', scope)
        import torch
        from hanglish.models.contextual_character_residual import ContextualCharacterResidualRanker
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.manual_seed(RECIPE['seed'])
        torch.use_deterministic_algorithms(True)
        model = ContextualCharacterResidualRanker()
        require(sum(p.numel() for p in model.parameters()) == 31681, 'parameter budget changed')
        before, initial_loss = evaluate(model, auth, scope, output, 'before', guard, initial=True)
        baseline = summarize(before)
        expected = {'rows': 32, 'first': 29, 'top5': 29, 'pool_covered': 30} if scope == 'pilot' else {
            'rows': 1642, 'first': 1453, 'top5': 1529, 'pool_covered': 1571,
            'no_pool': 13, 'unsupported': 31}
        require(all(baseline[k] == v for k,v in expected.items()), 'matched baseline differs')
        optimizer = torch.optim.AdamW(model.parameters(), lr=RECIPE['learning_rate'],
                                      weight_decay=RECIPE['weight_decay'])
        history = []
        for epoch in range(1, RECIPE['epochs']+1):
            model.train()
            losses = []
            for index, row in enumerate(rows_for(auth, scope)):
                started = time.monotonic()
                if row['positive_indices']:
                    optimizer.zero_grad(set_to_none=True)
                    losses.append(backward_listwise_chunks(model, row))
                    torch.nn.utils.clip_grad_norm_(model.parameters(), RECIPE['clip_norm'],
                                                   error_if_nonfinite=True)
                    optimizer.step()
                guard.require_row_time(started)
                if (index+1) % 32 == 0:
                    observe(guard, f'epoch-{epoch}:row-{index+1}', scope)
            require(len(losses) == baseline['pool_covered'], 'training denominator differs')
            history.append({'epoch': epoch, 'updates': len(losses),
                            'mean_training_loss': sum(losses)/len(losses)})
            observe(guard, f'epoch-{epoch}:complete', scope)
            print(json.dumps(history[-1]), flush=True)
        after, final_loss = evaluate(model, auth, scope, output, 'after', guard)
        require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'nonfinite weights')
        torch.save({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},
                   output/'model.pt')
        verify_pins(auth['pins'])
        observe(guard, 'complete', scope)
        report = {
            'completed': True, 'scope': scope, 'rows': len(before), 'recipe': RECIPE,
            'parameter_count': 31681, 'updates': sum(x['updates'] for x in history),
            'before': baseline, 'after': summarize(after), 'initial_loss': initial_loss,
            'final_loss': final_loss, 'history': history,
            'corrections': sum(not a['first'] and b['first'] for a,b in zip(before,after)),
            'regressions': sum(a['first'] and not b['first'] for a,b in zip(before,after)),
            'top5_losses': sum(a['top5'] and not b['top5'] for a,b in zip(before,after)),
            'pins': auth['pins'], 'hashes': {name:digest(output/name) for name in
                ('before.jsonl', 'after.jsonl', 'model.pt', 'resources.jsonl')},
            'model_source_sha256': digest(MODEL), 'runner_sha256': digest(Path(__file__)),
            'resources': {'seconds': time.monotonic()-guard.started,
                          'max_rss_bytes': guard.peak_rss,
                          'max_swap_growth_bytes': guard.peak_swap_growth},
            'dev_opened': False, 'test_opened': False, 'production_changed': False,
        }
        (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k not in ('pins','hashes')},
                         ensure_ascii=False), flush=True)
    except Exception as error:
        (output/'failure.json').write_text(json.dumps({'completed':False,
            'error_type':type(error).__name__, 'error':str(error)}, indent=2)+'\n')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scope', choices=('pilot','full'), required=True)
    args = parser.parse_args()
    execute(args.output, args.scope)
