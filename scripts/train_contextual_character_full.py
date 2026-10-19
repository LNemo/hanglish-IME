"""Train the frozen full-candidate contextual ranker on all matched FIT rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_character_edge_full_fit import (
    append_jsonl,
    authenticate_inputs,
    build_row,
    digest,
    iter_captures,
    listwise_loss,
    row_metrics,
    swap_bytes,
    verify_pins,
)
from scripts.train_contextual_character import (
    RECIPE,
    backward_listwise_chunks,
    score_chunks,
)


SCHEMA = 'contextual-character-full-fit-v2'
LIMITS = {
    'whole_seconds': 7200.,
    'row_seconds': 30.,
    'rss_bytes': 2 * 1024**3,
    'swap_growth_bytes': 512 * 1024**2,
}
EXPECTED_BEFORE = {
    'rows': 1642,
    'first': 1453,
    'top5': 1529,
    'pool_covered': 1571,
    'no_pool': 13,
    'unsupported': 31,
}
EXPECTED_COUNTS = {
    'ranked_rows': 1629,
    'no_pool_rows': 13,
    'unsupported_candidates': 31,
    'gold_present_rows': 1571,
    'missing_gold_rows': 71,
}
EXPECTED_UPDATES = 1571 * RECIPE['epochs']

PLAN = ROOT / '.omx/plans/contextual-character-full-fit-v2.md'
MODEL = ROOT / 'hanglish/models/contextual_character_residual.py'
PARENT_MODEL = ROOT / 'hanglish/models/character_edge_residual.py'
TRAINING_MATH = ROOT / 'scripts/train_contextual_character.py'
ORIGINAL_GUARDS = ROOT / 'scripts/train_character_edge_full_fit.py'
SELECTION_HELPER = ROOT / 'scripts/current_recombination_selection.py'
SOURCE_CORRESPONDENCE = ROOT / 'hanglish/core/source_correspondence.py'
PILOT = ROOT / '.omx/artifacts/contextual-character-pilot-v1'


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def current_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == 'darwin' else value * 1024


class FullResourceGuard:
    """Independent two-hour guard for the measured full contextual run."""

    def __init__(self, output: Path, *, clock=None, swap_reader=None,
                 rss_reader=None):
        self.output = output
        self.clock = clock or time.monotonic
        self.swap_reader = swap_reader or swap_bytes
        self.rss_reader = rss_reader or current_rss_bytes
        self.started = self.clock()
        self.baseline_swap = self.swap_reader()
        self.peak_rss = 0
        self.peak_swap_growth = 0

    def observe(self, stage: str) -> dict:
        elapsed = self.clock() - self.started
        rss = self.rss_reader()
        growth = self.swap_reader() - self.baseline_swap
        row = {
            'stage': stage,
            'seconds': elapsed,
            'rss_bytes': rss,
            'swap_growth_bytes': growth,
        }
        append_jsonl(self.output / 'resources.jsonl', row)
        self.peak_rss = max(self.peak_rss, rss)
        self.peak_swap_growth = max(self.peak_swap_growth, growth)
        require(elapsed <= LIMITS['whole_seconds']
                and rss <= LIMITS['rss_bytes']
                and growth <= LIMITS['swap_growth_bytes'],
                'full contextual FIT resource limit exceeded')
        return row

    def require_row_time(self, started: float) -> None:
        require(self.clock() - started <= LIMITS['row_seconds'],
                'single row time limit exceeded')


def authenticate_full() -> dict:
    """Authenticate the closed FIT pool and independently verified pilot."""
    authenticated = authenticate_inputs()
    report_path = PILOT / 'report.json'
    proof_path = PILOT / 'independent-verification.json'
    model_path = PILOT / 'model.pt'
    before_path = PILOT / 'before.jsonl'
    after_path = PILOT / 'after.jsonl'
    resources_path = PILOT / 'resources.jsonl'
    report = read_json(report_path)
    proof = read_json(proof_path)
    require(
        report.get('completed') is True
        and report.get('scope') == 'pilot'
        and report.get('rows') == 32
        and report.get('recipe') == RECIPE
        and report.get('parameter_count') == 31681
        and report.get('updates') == 120
        and report.get('before') == {
            'rows': 32, 'first': 29, 'top5': 29, 'pool_covered': 30,
            'no_pool': 1, 'unsupported': 0,
        }
        and report.get('after') == report.get('before')
        and report.get('model_source_sha256') == digest(MODEL)
        and report.get('runner_sha256') == digest(TRAINING_MATH)
        and report.get('hashes') == {
            'before.jsonl': digest(before_path),
            'after.jsonl': digest(after_path),
            'model.pt': digest(model_path),
            'resources.jsonl': digest(resources_path),
        },
        'frozen contextual pilot report differs',
    )
    require(
        proof.get('verified') is True
        and proof.get('scope') == 'pilot'
        and proof.get('rows') == 32
        and proof.get('report_sha256') == digest(report_path)
        and proof.get('checkpoint_sha256') == digest(model_path)
        and proof.get('before_sha256') == digest(before_path)
        and proof.get('after_sha256') == digest(after_path)
        and proof.get('resources_sha256') == digest(resources_path)
        and proof.get('before') == report.get('before')
        and proof.get('after') == report.get('after')
        and proof.get('corrections') == report.get('corrections')
        and proof.get('regressions') == report.get('regressions')
        and proof.get('top5_losses') == report.get('top5_losses'),
        'independently verified contextual pilot required',
    )

    pins = authenticated['pins']
    for path in (
        PLAN,
        MODEL,
        PARENT_MODEL,
        TRAINING_MATH,
        ORIGINAL_GUARDS,
        SELECTION_HELPER,
        SOURCE_CORRESPONDENCE,
        Path(__file__).resolve(),
        report_path,
        proof_path,
        model_path,
        before_path,
        after_path,
        resources_path,
    ):
        pins[str(path)] = digest(path)
    return authenticated


def rows_for(authenticated: dict):
    """Rebuild one matched row at a time in the closed FIT order."""
    for identity, scored in iter_captures(authenticated['shards']):
        yield build_row(
            identity,
            scored,
            authenticated['labels'][identity],
            authenticated['calibration'],
        )


def summarize(records: list[dict]) -> dict:
    return {
        'rows': len(records),
        **{
            key: sum(int(row[key]) for row in records)
            for key in (
                'first', 'top5', 'pool_covered', 'no_pool', 'unsupported'
            )
        },
    }


def denominator_counts(records: list[dict]) -> dict:
    return {
        'ranked_rows': sum(not row['no_pool'] for row in records),
        'no_pool_rows': sum(row['no_pool'] for row in records),
        'unsupported_candidates': sum(row['unsupported'] for row in records),
        'gold_present_rows': sum(row['pool_covered'] for row in records),
        'missing_gold_rows': sum(not row['pool_covered'] for row in records),
    }


def evaluate(model, authenticated: dict, output: Path, phase: str,
             guard: FullResourceGuard, *, initial: bool = False):
    import torch

    path = output / f'{phase}.jsonl'
    require(not path.exists(), f'{phase} outcomes already exist')
    records, losses = [], []
    model.eval()
    with torch.no_grad():
        for index, row in enumerate(rows_for(authenticated)):
            started = guard.clock()
            values = (score_chunks(
                model, row, RECIPE['candidate_chunk'])
                if row['candidates'] else [])
            if initial and row['candidates']:
                expected = torch.tensor(row['base_scores'], dtype=torch.float64)
                require(torch.equal(values, expected),
                        'initial exact baseline changed')
            if row['positive_indices']:
                loss = listwise_loss(values, row['positive_indices'])
                require(bool(torch.isfinite(loss)), 'nonfinite evaluation loss')
                losses.append(float(loss))
            metric = row_metrics(row, values)
            metric.update(no_pool=row['no_pool'], unsupported=row['unsupported'])
            records.append(metric)
            append_jsonl(path, metric)
            guard.require_row_time(started)
            if (index + 1) % 32 == 0:
                guard.observe(f'{phase}:row-{index + 1}')
    guard.observe(f'{phase}:complete')
    require(bool(losses), 'evaluation has no gold-present rows')
    return records, sum(losses) / len(losses)


def train_epoch(model, optimizer, authenticated: dict, epoch: int,
                guard: FullResourceGuard) -> tuple[float, int]:
    import torch

    model.train()
    losses = []
    for index, row in enumerate(rows_for(authenticated)):
        started = guard.clock()
        if row['positive_indices']:
            optimizer.zero_grad(set_to_none=True)
            losses.append(backward_listwise_chunks(
                model, row, RECIPE['candidate_chunk']))
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                RECIPE['clip_norm'],
                error_if_nonfinite=True,
            )
            optimizer.step()
        guard.require_row_time(started)
        if (index + 1) % 32 == 0:
            guard.observe(f'epoch-{epoch}:row-{index + 1}')
    require(len(losses) == EXPECTED_COUNTS['gold_present_rows'],
            'training denominator differs')
    guard.observe(f'epoch-{epoch}:complete')
    return sum(losses) / len(losses), len(losses)


def execute(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    guard = None
    try:
        authenticated = authenticate_full()
        guard = FullResourceGuard(output)
        guard.observe('authenticated')

        import torch
        from hanglish.models.contextual_character_residual import (
            ContextualCharacterResidualRanker,
        )

        torch.set_num_threads(RECIPE['threads'])
        torch.set_num_interop_threads(RECIPE['interop_threads'])
        torch.manual_seed(RECIPE['seed'])
        torch.use_deterministic_algorithms(True)
        model = ContextualCharacterResidualRanker(
            byte_dim=16, hidden=32, head_hidden=32)
        require(sum(parameter.numel() for parameter in model.parameters()) == 31681,
                'frozen model parameter count differs')

        before, initial_loss = evaluate(
            model, authenticated, output, 'before', guard, initial=True)
        before_summary = summarize(before)
        counts = denominator_counts(before)
        require(before_summary == EXPECTED_BEFORE and counts == EXPECTED_COUNTS,
                'initial full FIT metrics or denominators differ')

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=RECIPE['learning_rate'],
            weight_decay=RECIPE['weight_decay'],
        )
        history = []
        for epoch in range(1, RECIPE['epochs'] + 1):
            mean_loss, updates = train_epoch(
                model, optimizer, authenticated, epoch, guard)
            history.append({
                'epoch': epoch,
                'updates': updates,
                'mean_training_loss': mean_loss,
            })
            print(json.dumps(history[-1]), flush=True)
        total_updates = sum(row['updates'] for row in history)
        require(total_updates == EXPECTED_UPDATES
                and all(bool(torch.isfinite(parameter).all())
                        for parameter in model.parameters()),
                'training updates or checkpoint tensors differ')

        after, final_loss = evaluate(
            model, authenticated, output, 'after', guard)
        after_summary = summarize(after)
        require(denominator_counts(after) == counts,
                'final denominators differ')
        checkpoint = {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in model.state_dict().items()
        }
        torch.save(checkpoint, output / 'model.pt')
        guard.observe('checkpoint-saved')
        verify_pins(authenticated['pins'])
        guard.observe('inputs-reverified')

        corrections = sum(
            not first['first'] and second['first']
            for first, second in zip(before, after))
        regressions = sum(
            first['first'] and not second['first']
            for first, second in zip(before, after))
        top5_losses = sum(
            first['top5'] and not second['top5']
            for first, second in zip(before, after))
        completed_resources = guard.observe('complete')
        output_names = (
            'before.jsonl', 'after.jsonl', 'model.pt', 'resources.jsonl')
        report = {
            'schema': SCHEMA,
            'completed': True,
            'scope': 'full',
            'rows': len(before),
            'recipe': RECIPE,
            'limits': LIMITS,
            'parameter_count': 31681,
            'updates': total_updates,
            'initial_baseline_exact': True,
            'before': before_summary,
            'after': after_summary,
            'initial_loss': initial_loss,
            'final_loss': final_loss,
            'history': history,
            'counts': counts,
            'corrections': corrections,
            'regressions': regressions,
            'top5_losses': top5_losses,
            'pins': authenticated['pins'],
            'hashes': {
                name: digest(output / name) for name in output_names
            },
            'ordered_ids_sha256': authenticated['ordered_ids_sha256'],
            'source_order_sha256': authenticated['source_order_sha256'],
            'label_order_sha256': authenticated['label_order_sha256'],
            'model_source_sha256': digest(MODEL),
            'runner_sha256': digest(Path(__file__).resolve()),
            'plan_sha256': digest(PLAN),
            'resources': {
                'seconds': completed_resources['seconds'],
                'max_rss_bytes': guard.peak_rss,
                'max_swap_growth_bytes': guard.peak_swap_growth,
            },
            'dev_opened': False,
            'test_opened': False,
            'production_changed': False,
        }
        (output / 'report.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({
            key: value for key, value in report.items()
            if key not in ('pins', 'hashes')
        }, ensure_ascii=False), flush=True)
    except Exception as error:
        failure = {
            'schema': SCHEMA,
            'completed': False,
            'error_type': type(error).__name__,
            'error': str(error),
        }
        if guard is not None:
            failure['seconds'] = guard.clock() - guard.started
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
