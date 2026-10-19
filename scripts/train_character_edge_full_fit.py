"""Stream the frozen 1,642-row FIT corpus through character-edge training."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

from scripts.audit_recombination_tokenization import digest, require
from scripts.current_recombination_selection import calibrated_score, render_public


SCHEMA = 'character-edge-full-fit-v1'
SEED = 20260923
RECIPE = {'epochs': 4, 'learning_rate': .001, 'weight_decay': .01,
          'clip_norm': 1., 'threads': 2, 'interop_threads': 1, 'seed': SEED}
LIMITS = {'whole_seconds': 1800., 'row_seconds': 30.,
          'rss_bytes': 2 * 1024**3, 'swap_growth_bytes': 512 * 1024**2}
SELECTION = ROOT / '.omx/artifacts/current-fit-recombination-selection-v1'
INTAKE = ROOT / '.omx/artifacts/current-fit-recombination-selection-intake-v1.json'
LABELS = ROOT / '.omx/artifacts/current-phonetic-train-features-v1/rows.jsonl'
CALIBRATION = ROOT / '.omx/artifacts/phonetic-score-scale-training-v1/model.json'
MODEL_SOURCE = ROOT / 'hanglish/models/character_edge_residual.py'
PLAN = ROOT / '.omx/plans/character-edge-full-fit-v1.md'
PILOT = ROOT / '.omx/artifacts/character-edge-residual-pilot-v1'
EXPECTED_BEFORE = {'rows': 1642, 'first': 1453, 'top5': 1529,
                   'pool_covered': 1571}


def read_json(path: Path):
    return json.loads(path.read_text())


def append_jsonl(path: Path, value: dict) -> None:
    with path.open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n')


def sequence_digest(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()
    return hashlib.sha256(payload).hexdigest()


def swap_bytes() -> int:
    result = subprocess.run(['/usr/sbin/sysctl', '-n', 'vm.swapusage'],
                            capture_output=True, text=True, check=True)
    match = re.search(r'used\s*=\s*([0-9.]+)M', result.stdout)
    require(match is not None, 'swap telemetry unavailable')
    return round(float(match.group(1)) * 1024**2)


class ResourceGuard:
    def __init__(self, output: Path):
        self.output = output
        self.started = time.monotonic()
        self.baseline_swap = swap_bytes()
        self.peak_rss = 0
        self.peak_swap_growth = 0

    def observe(self, stage: str) -> dict:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss *= 1 if sys.platform == 'darwin' else 1024
        elapsed = time.monotonic() - self.started
        growth = swap_bytes() - self.baseline_swap
        row = {'stage': stage, 'seconds': elapsed, 'rss_bytes': rss,
               'swap_growth_bytes': growth}
        append_jsonl(self.output / 'resources.jsonl', row)
        self.peak_rss = max(self.peak_rss, rss)
        self.peak_swap_growth = max(self.peak_swap_growth, growth)
        require(elapsed <= LIMITS['whole_seconds']
                and rss <= LIMITS['rss_bytes']
                and growth <= LIMITS['swap_growth_bytes'],
                'full FIT resource limit exceeded')
        return row

    def require_row_time(self, started: float) -> None:
        require(time.monotonic() - started <= LIMITS['row_seconds'],
                'single row time limit exceeded')


def load_labels() -> dict[str, dict]:
    labels = {}
    with LABELS.open() as stream:
        for line in stream:
            row = json.loads(line)
            minimal = {key: row[key] for key in ('id', 'input', 'expected', 'component_id')}
            require(minimal['id'] not in labels, 'duplicate FIT label ID')
            labels[minimal['id']] = minimal
    return labels


def iter_captures(shards):
    """Yield one scoring row at a time in frozen shard/file order."""
    for expected_shard, shard in enumerate(shards):
        require(shard['shard'] == expected_shard, 'capture shard order differs')
        with (Path(shard['directory']) / 'capture.jsonl').open() as stream:
            for line in stream:
                captured = json.loads(line)
                require(isinstance(captured, dict) and isinstance(captured.get('id'), str)
                        and isinstance(captured.get('scoring'), dict),
                        'captured scoring row differs')
                yield captured['id'], captured['scoring']


def authenticate_inputs() -> dict:
    report = read_json(SELECTION / 'report.json')
    proof = read_json(SELECTION / 'independent-verification.json')
    intake = read_json(INTAKE)
    pilot = read_json(PILOT / 'report.json')
    pilot_proof = read_json(PILOT / 'checkpoint-verification.json')
    require(report.get('schema') == 'current-fit-recombination-selection-v1'
            and report.get('completed') is True and report.get('rows') == 1642
            and proof.get('verified') is True
            and proof.get('report_sha256') == digest(SELECTION / 'report.json')
            and proof.get('outcomes_sha256') == report.get('outcomes_sha256')
            and proof.get('closure_sha256') == report.get('closure_sha256'),
            'selection report or proof differs')
    require(pilot.get('completed') is True
            and pilot_proof.get('verified') is True
            and pilot_proof.get('report_sha256') == digest(PILOT / 'report.json')
            and pilot.get('model_source_sha256') == digest(MODEL_SOURCE),
            'verified pilot model source differs')
    require(digest(INTAKE) == report.get('intake_sha256')
            and digest(LABELS) == report.get('features_sha256')
            and digest(CALIBRATION) == report.get('model_sha256'),
            'FIT intake, labels, or calibration differs')
    shards = intake.get('shards')
    require(isinstance(shards, list) and len(shards) == 6,
            'six capture shards required')
    pins = {
        str(SELECTION / 'report.json'): digest(SELECTION / 'report.json'),
        str(SELECTION / 'independent-verification.json'):
            digest(SELECTION / 'independent-verification.json'),
        str(INTAKE): digest(INTAKE), str(LABELS): digest(LABELS),
        str(CALIBRATION): digest(CALIBRATION), str(MODEL_SOURCE): digest(MODEL_SOURCE),
        str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        str(PLAN): digest(PLAN), str(PILOT / 'report.json'): digest(PILOT / 'report.json'),
        str(PILOT / 'checkpoint-verification.json'):
            digest(PILOT / 'checkpoint-verification.json'),
    }
    for shard in shards:
        hashes = shard.get('hashes')
        require(isinstance(hashes, dict) and len(hashes) == 7,
                'seven closed files required per shard')
        for name, expected in hashes.items():
            path = Path(shard['directory']) / name
            require(digest(path) == expected, 'closed shard file differs')
            pins[str(path)] = expected

    labels = load_labels()
    ordered_ids, source_order, label_order = [], [], []
    for identity, scored in iter_captures(shards):
        require(identity in labels, 'capture label is missing')
        label = labels[identity]
        source = scored.get('request', {}).get('input')
        require(source == label['input'], 'capture source and label differ')
        ordered_ids.append(identity)
        source_order.append((identity, source))
        label_order.append((identity, source, label['expected'], label['component_id']))
    require(ordered_ids == report.get('ids')
            and len(ordered_ids) == len(set(ordered_ids)) == 1642
            and set(labels) == set(ordered_ids), 'canonical FIT ID order differs')
    calibration = read_json(CALIBRATION)
    return {'report': report, 'proof': proof, 'shards': shards, 'labels': labels,
            'calibration': calibration, 'pins': pins,
            'ordered_ids_sha256': sequence_digest(ordered_ids),
            'source_order_sha256': sequence_digest(source_order),
            'label_order_sha256': sequence_digest(label_order)}


def build_row(identity: str, scored: dict, label: dict, calibration: dict) -> dict:
    source = scored.get('request', {}).get('input')
    require(identity == label['id'] and source == label['input'],
            'score source alignment differs')
    original, novel = scored.get('original'), scored.get('novel')
    require(isinstance(original, list) and isinstance(novel, list)
            and type(scored.get('ranked')) is bool, 'scored candidate domain differs')
    unsupported = [row for row in novel if row.get('status') == 'unsupported']
    scored_novel = [row for row in novel if row.get('status') == 'scored']
    require(len(scored_novel) + len(unsupported) == len(novel),
            'novel candidate status differs')
    if not scored['ranked']:
        require(not original and not novel and isinstance(scored.get('fallback_public'), list)
                and bool(scored['fallback_public']), 'no-pool fallback differs')
        candidates, base_scores = [], []
    else:
        require(bool(original) and scored.get('fallback_public') is None,
                'ranked candidate pool differs')
        pool = original + scored_novel
        candidates = [{'text': row['text'], 'segments': row['segments']} for row in pool]
        base_scores = [calibrated_score(row, calibration) for row in pool]
        require(len({row['text'] for row in candidates}) == len(candidates),
                'ranked candidate texts differ')
    expected = set(label['expected'])
    positive = [index for index, row in enumerate(candidates)
                if row['text'] in expected]
    return {'id': identity, 'input': source, 'expected': label['expected'],
            'component_id': label['component_id'], 'candidates': candidates,
            'base_scores': base_scores, 'positive_indices': positive,
            'fallback': scored.get('fallback_public'),
            'unsupported': len(unsupported), 'no_pool': not scored['ranked']}


def row_metrics(row: dict, values) -> dict:
    if row['no_pool']:
        public = row['fallback']
    else:
        require(len(values) == len(row['candidates']), 'candidate scores differ')
        public = render_public(row['input'], [
            {**candidate, 'score': float(score)}
            for candidate, score in zip(row['candidates'], values)
        ])
    texts = [candidate['text'] for candidate in public]
    expected = set(row['expected'])
    return {'id': row['id'], 'first': texts[0] in expected,
            'top5': bool(expected.intersection(texts)),
            'pool_covered': bool(row['positive_indices']), 'public': texts}


def listwise_loss(scores, positive_indices):
    import torch
    require(bool(positive_indices), 'gold-present row required')
    return torch.logsumexp(scores, 0) - torch.logsumexp(scores[positive_indices], 0)


def _summary(statuses: dict, losses: list[float]) -> dict:
    return {'rows': len(statuses),
            'first': sum(row['first'] for row in statuses.values()),
            'top5': sum(row['top5'] for row in statuses.values()),
            'pool_covered': sum(row['pool_covered'] for row in statuses.values()),
            'mean_loss': sum(losses) / len(losses)}


def evaluate(model, authenticated: dict, output: Path, name: str,
             guard: ResourceGuard, exact_baseline: bool = False):
    import torch
    statuses, losses = {}, []
    counts = {'ranked_rows': 0, 'no_pool_rows': 0, 'unsupported_candidates': 0,
              'gold_present_rows': 0, 'missing_gold_rows': 0}
    model.eval()
    path = output / f'{name}.jsonl'
    require(not path.exists(), f'{name} outcomes already exist')
    with torch.no_grad():
        for index, (identity, scored) in enumerate(iter_captures(authenticated['shards'])):
            row_started = time.monotonic()
            row = build_row(identity, scored, authenticated['labels'][identity],
                            authenticated['calibration'])
            counts['unsupported_candidates'] += row['unsupported']
            counts['no_pool_rows' if row['no_pool'] else 'ranked_rows'] += 1
            if row['no_pool']:
                values = []
            else:
                prepared = model.prepare(row['input'], row['candidates'], row['base_scores'])
                values = model(prepared)['scores']
                if exact_baseline:
                    require(torch.equal(values, torch.tensor(
                        row['base_scores'], dtype=torch.float64)),
                        'initial baseline parity differs')
                del prepared
            if row['positive_indices']:
                counts['gold_present_rows'] += 1
                losses.append(float(listwise_loss(values, row['positive_indices'])))
            else:
                counts['missing_gold_rows'] += 1
            metric = row_metrics(row, values)
            append_jsonl(path, metric)
            statuses[identity] = {key: metric[key]
                                  for key in ('first', 'top5', 'pool_covered')}
            guard.require_row_time(row_started)
            if (index + 1) % 32 == 0:
                guard.observe(f'{name}:row-{index + 1}')
            del row, scored, values
    guard.observe(f'{name}:complete')
    return _summary(statuses, losses), statuses, counts


def train_epoch(model, optimizer, authenticated: dict, epoch: int,
                guard: ResourceGuard) -> tuple[float, int]:
    import torch
    model.train()
    total, updates = 0., 0
    for index, (identity, scored) in enumerate(iter_captures(authenticated['shards'])):
        row_started = time.monotonic()
        row = build_row(identity, scored, authenticated['labels'][identity],
                        authenticated['calibration'])
        if row['positive_indices']:
            prepared = model.prepare(row['input'], row['candidates'], row['base_scores'])
            optimizer.zero_grad(set_to_none=True)
            scores = model(prepared)['scores']
            loss = listwise_loss(scores, row['positive_indices'])
            require(bool(torch.isfinite(loss)), 'nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), RECIPE['clip_norm'],
                                           error_if_nonfinite=True)
            optimizer.step()
            total += loss.item()
            updates += 1
            del prepared, scores, loss
        guard.require_row_time(row_started)
        if (index + 1) % 32 == 0:
            guard.observe(f'epoch-{epoch}:row-{index + 1}')
        del row, scored
    require(updates > 0, 'training pass has no gold-present rows')
    guard.observe(f'epoch-{epoch}:complete')
    return total / updates, updates


def verify_pins(pins: dict[str, str]) -> None:
    require(all(digest(Path(path)) == expected for path, expected in pins.items()),
            'frozen full FIT inputs or code changed during run')


def execute(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    guard = None
    try:
        authenticated = authenticate_inputs()
        guard = ResourceGuard(output)
        guard.observe('authenticated')
        import torch
        from hanglish.models.character_edge_residual import CharacterEdgeResidualRanker

        torch.set_num_threads(RECIPE['threads'])
        torch.set_num_interop_threads(RECIPE['interop_threads'])
        torch.manual_seed(SEED)
        torch.use_deterministic_algorithms(True)
        model = CharacterEdgeResidualRanker(byte_dim=16, hidden=32, head_hidden=32)
        require(sum(parameter.numel() for parameter in model.parameters()) == 31681,
                'frozen model parameter count differs')
        before, before_status, counts = evaluate(
            model, authenticated, output, 'before', guard, exact_baseline=True)
        require({key: before[key] for key in EXPECTED_BEFORE} == EXPECTED_BEFORE
                and counts == {'ranked_rows': 1629, 'no_pool_rows': 13,
                               'unsupported_candidates': 31, 'gold_present_rows': 1571,
                               'missing_gold_rows': 71},
                'initial full FIT metrics or denominators differ')
        optimizer = torch.optim.AdamW(model.parameters(), lr=RECIPE['learning_rate'],
                                      weight_decay=RECIPE['weight_decay'])
        history, total_updates = [], 0
        for epoch in range(1, RECIPE['epochs'] + 1):
            mean_loss, updates = train_epoch(
                model, optimizer, authenticated, epoch, guard)
            history.append({'epoch': epoch, 'mean_training_loss': mean_loss,
                            'updates': updates})
            total_updates += updates
            print(json.dumps(history[-1]), flush=True)
        require(total_updates == 1571 * RECIPE['epochs']
                and all(bool(torch.isfinite(parameter).all())
                        for parameter in model.parameters()),
                'training updates or checkpoint tensors differ')
        after, after_status, after_counts = evaluate(
            model, authenticated, output, 'after', guard)
        require(after_counts == counts, 'final denominators differ')
        checkpoint = {name: tensor.detach().cpu().contiguous()
                      for name, tensor in model.state_dict().items()}
        torch.save(checkpoint, output / 'model.pt')
        guard.observe('checkpoint-saved')
        verify_pins(authenticated['pins'])
        guard.observe('inputs-reverified')
        corrections = sum(not before_status[i]['first'] and after_status[i]['first']
                          for i in authenticated['report']['ids'])
        regressions = sum(before_status[i]['first'] and not after_status[i]['first']
                          for i in authenticated['report']['ids'])
        top5_losses = sum(before_status[i]['top5'] and not after_status[i]['top5']
                          for i in authenticated['report']['ids'])
        outputs = ('before.jsonl', 'after.jsonl', 'model.pt', 'resources.jsonl')
        result = {
            'schema': SCHEMA, 'completed': True, 'rows': 1642,
            'recipe': RECIPE, 'limits': LIMITS, 'parameter_count': 31681,
            'updates': total_updates, 'history': history,
            'initial_baseline_exact': True, 'initial_loss': before['mean_loss'],
            'final_loss': after['mean_loss'], 'before': before, 'after': after,
            'counts': counts, 'corrections': corrections, 'regressions': regressions,
            'top5_losses': top5_losses,
            'ordered_ids_sha256': authenticated['ordered_ids_sha256'],
            'source_order_sha256': authenticated['source_order_sha256'],
            'label_order_sha256': authenticated['label_order_sha256'],
            'hashes': {name: digest(output / name) for name in outputs},
            'model_source_sha256': digest(MODEL_SOURCE),
            'runner_sha256': digest(Path(__file__).resolve()),
            'plan_sha256': digest(PLAN),
            'selection_report_sha256': authenticated['proof']['report_sha256'],
            'resources': {'seconds': time.monotonic() - guard.started,
                          'max_rss_bytes': guard.peak_rss,
                          'max_swap_growth_bytes': guard.peak_swap_growth},
            'dev_opened': False, 'test_opened': False, 'production_changed': False,
        }
        (output / 'report.json').write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(result, ensure_ascii=False), flush=True)
    except Exception as error:
        failure = {'schema': SCHEMA, 'completed': False,
                   'error_type': type(error).__name__, 'error': str(error)}
        if guard is not None:
            failure['seconds'] = time.monotonic() - guard.started
            try:
                guard.observe('failed')
            except Exception:
                pass
        (output / 'failure.json').write_text(
            json.dumps(failure, ensure_ascii=False, indent=2) + '\n')
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    execute(args.output)


if __name__ == '__main__':
    main()
