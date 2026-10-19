"""FIT-only algebraic audit of unchanged-span score contributions.

This script reads closed aggregate training artifacts.  It loads no tokenizer or
neural model and writes no row text, identifier, score, or label to its report.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import struct


ROOT = Path(__file__).resolve().parents[1]
PREFERENCES = ROOT / '.omx/artifacts/conditional-span-preferences-fit-v1/preferences.jsonl'
FIT_PATHS = ROOT / '.omx/artifacts/source-correspondence-fit-v2/fit-paths.jsonl'
CONTROL_AFTER = ROOT / '.omx/artifacts/source-span-paired-training-v1/source_local/after.jsonl'
CONTROL_PARITY = ROOT / '.omx/artifacts/local-span-objective-training-v1/control-parity.jsonl'
AUX_AFTER = ROOT / '.omx/artifacts/local-span-objective-training-v1/after.jsonl'
AUX_REPORT = ROOT / '.omx/artifacts/local-span-objective-training-v1/report.json'
AUX_PROOF = ROOT / '.omx/artifacts/local-span-objective-training-v1/independent-verification.json'
EXPECTED = {
    PREFERENCES: '9d4939ce9b23466320a8e9080339629b2577d362cc77e6f4404e62cbd73df285',
    FIT_PATHS: 'f1cd66da23cacc6025e8aa2203944f664f0d74d916a0bde135cac65ef56afd8c',
    CONTROL_AFTER: '9a66769aabe7be8e974acc1c6445468a4159192d59f4cd54aa9b0d57cccfcb67',
    CONTROL_PARITY: '730b3047685948071a06e6f530e27742ace6ee28cdbc333b25910657e8dd48f8',
    AUX_AFTER: '59123c4f26701f2742d12d7b77ddca94854ca6470a173142099a6befd5dfba88',
    AUX_REPORT: 'e3e62c9fcc8ffed97ddc79dcaa5eaa559968365f0fe1a3fb46b593838cabbd46',
    AUX_PROOF: '94b6705455b76b6a0c41ad9651de2e8752b488a9f0c4cedb18da3c0d04bbda5a',
}
SCHEMA = 'local-span-outside-contributions-fit-v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2,
                  allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def float32(value):
    return struct.unpack('>f', struct.pack('>f', float(value)))[0]


def float32_ulp(value):
    value = float32(abs(value))
    require(math.isfinite(value), 'nonfinite score scale')
    bits = struct.unpack('>I', struct.pack('>f', value))[0]
    following = struct.unpack('>f', struct.pack('>I', bits + 1))[0]
    return following - value


def summary(values):
    values = sorted(values)
    require(bool(values) and all(math.isfinite(value) for value in values),
            'empty or nonfinite aggregate')
    at = lambda quantile: values[round((len(values) - 1) * quantile)]
    return {
        'count': len(values), 'mean': sum(values) / len(values),
        'median': statistics.median(values), 'minimum': values[0],
        'p10_nearest_index': at(.1), 'p90_nearest_index': at(.9),
        'maximum': values[-1],
    }


def run(output):
    require(all(sha(path) == digest for path, digest in EXPECTED.items()),
            'frozen FIT diagnostic input changed')
    report, proof = read(AUX_REPORT), read(AUX_PROOF)
    require(report['completed'] is True and report['contracts'] == {
        'dev_opened': False, 'fit_only': True, 'hold_opened': False,
        'production_changed': False, 'test_opened': False},
        'local auxiliary report scope differs')
    require(proof['verified'] is True and proof['report_sha256'] == sha(AUX_REPORT)
            and proof['contracts']['fit_only'] is True
            and proof['contracts']['dev_opened'] is False
            and proof['contracts']['hold_opened'] is False
            and proof['contracts']['test_opened'] is False,
            'local auxiliary proof differs')

    preference_rows = list(rows(PREFERENCES))
    control_after = {row['id']: row for row in rows(CONTROL_AFTER)}
    control_parity = {row['id']: row for row in rows(CONTROL_PARITY)}
    auxiliary = {row['id']: row for row in rows(AUX_AFTER)}
    require(len(preference_rows) == len(control_after) == len(control_parity)
            == len(auxiliary) == 1483, 'FIT row denominator differs')
    ordered_ids = [row['id'] for row in preference_rows]
    require(set(ordered_ids) == set(control_after) == set(control_parity)
            == set(auxiliary) and len(set(ordered_ids)) == len(ordered_ids),
            'FIT row identity differs')

    needed = {row['id']: set(row['candidates']) for row in preference_rows}
    path_counts = {}; selected_path_counts = Counter()
    for record in rows(FIT_PATHS):
        if record['id'] not in needed:
            continue
        by_text = defaultdict(list)
        for path in record['paths']:
            if path['text'] in needed[record['id']]:
                by_text[path['text']].append(path)
        require(set(by_text) == needed[record['id']], 'selected candidate path missing')
        for variants in by_text.values():
            selected_path_counts['candidate_texts'] += 1
            selected_path_counts['paths'] += len(variants)
            selected_path_counts['multi_path_candidate_texts'] += len(variants) > 1
        path_counts[record['id']] = {
            path['path_id']: len(path['edge_indices'])
            for variants in by_text.values() for path in variants}
    require(set(path_counts) == set(ordered_ids)
            and selected_path_counts == Counter(
                candidate_texts=11793, paths=11793,
                multi_path_candidate_texts=0),
            'single-path selected domain differs')

    lost = {row_id for row_id in ordered_ids
            if control_after[row_id]['correct'] and not auxiliary[row_id]['correct']}
    gained = {row_id for row_id in ordered_ids
              if not control_after[row_id]['correct'] and auxiliary[row_id]['correct']}
    pair_records = []; witness_counts = Counter()
    transition_counts = Counter()
    lost_pair_count = 0
    decisive = []
    for row in preference_rows:
        row_id = row['id']; preferences = row['preferences']
        require(len(row['positive_indices']) == 1
                and len(preferences) == len(control_parity[row_id]['local_margins'])
                == len(auxiliary[row_id]['local_margins']),
                'preference/local-margin domain differs')
        wrong_first = max(range(len(auxiliary[row_id]['scores'])),
                          key=lambda index: auxiliary[row_id]['scores'][index])
        positive = row['positive_indices'][0]
        for index, preference in enumerate(preferences):
            witnesses = preference['witnesses']
            witness_counts['preferences'] += 1
            witness_counts['witnesses'] += len(witnesses)
            witness_counts['multiple_witness_preferences'] += len(witnesses) > 1
            require(len(witnesses) == 1, 'single-witness domain differs')
            witness = witnesses[0]
            preferred_index = preference['preferred_index']
            dispreferred_index = preference['dispreferred_index']
            preferred_path = witness['preferred']['path_id']
            dispreferred_path = witness['dispreferred']['path_id']
            complete_preferred = path_counts[row_id][preferred_path]
            complete_dispreferred = path_counts[row_id][dispreferred_path]
            changed_preferred = len(witness['preferred']['span_indices'])
            changed_dispreferred = len(witness['dispreferred']['span_indices'])
            same_counts = (complete_preferred == complete_dispreferred
                           and changed_preferred == changed_dispreferred)
            require((complete_preferred == complete_dispreferred)
                    == (changed_preferred == changed_dispreferred),
                    'only one span-count equality changed')
            record = {
                'row_id': row_id,
                'same_counts': same_counts,
                'complete_count': complete_preferred if same_counts else None,
                'changed_count': changed_preferred if same_counts else None,
                'lost': row_id in lost,
                'decisive': (row_id in lost and preferred_index == positive
                             and dispreferred_index == wrong_first),
            }
            for arm, source, residual_source in (
                    ('control', control_parity[row_id], control_after[row_id]),
                    ('local_aux', auxiliary[row_id], auxiliary[row_id])):
                local_margin = source['local_margins'][index]
                residual_delta = (residual_source['residuals'][preferred_index]
                                  - residual_source['residuals'][dispreferred_index])
                global_delta = (residual_source['scores'][preferred_index]
                                - residual_source['scores'][dispreferred_index])
                arm_value = {
                    'local_margin': local_margin,
                    'residual_delta': residual_delta,
                    'global_delta': global_delta,
                    'base_delta': global_delta - residual_delta,
                }
                if same_counts:
                    count, changed = complete_preferred, changed_preferred
                    outside = count * residual_delta - changed * local_margin
                    scale = max(abs(residual_source['residuals'][preferred_index]),
                                abs(residual_source['residuals'][dispreferred_index]),
                                abs(local_margin), 1.)
                    bound = 8 * (count + changed + 2) * float32_ulp(scale)
                    arm_value.update(outside_sum_delta=outside,
                                     recorded_fp32_ulp_bound=bound,
                                     outside_exceeds_bound=abs(outside) > bound)
                record[arm] = arm_value
            control_global = record['control']['global_delta']
            auxiliary_global = record['local_aux']['global_delta']
            control_local = record['control']['local_margin']
            auxiliary_local = record['local_aux']['local_margin']
            transition_counts['local_improved_global_worsened'] += (
                control_local <= 0 < auxiliary_local
                and control_global > 0 >= auxiliary_global)
            transition_counts['local_still_positive_global_worsened'] += (
                control_local > 0 and auxiliary_local > 0
                and control_global > 0 >= auxiliary_global)
            lost_pair_count += row_id in lost
            if record['decisive']:
                decisive.append(record)
            pair_records.append(record)

    require(witness_counts == Counter(preferences=6889, witnesses=6889,
                                      multiple_witness_preferences=0)
            and len(lost) == 15 and len(gained) == 1 and lost_pair_count == 45,
            'FIT preference or choice transition count differs')
    eligible = [record for record in pair_records if record['same_counts']]
    require(len(eligible) == 6737 and len(pair_records) - len(eligible) == 152,
            'equal-span-count denominator differs')

    arms = {}
    for arm in ('control', 'local_aux'):
        outside = [record[arm]['outside_sum_delta'] for record in eligible]
        bounds = [record[arm]['recorded_fp32_ulp_bound'] for record in eligible]
        arms[arm] = {
            'outside_sum_delta': summary(outside),
            'outside_negative': sum(value < 0 for value in outside),
            'outside_positive': sum(value > 0 for value in outside),
            'local_positive_residual_nonpositive': sum(
                record[arm]['local_margin'] > 0
                and record[arm]['residual_delta'] <= 0 for record in eligible),
            'local_positive_global_nonpositive': sum(
                record[arm]['local_margin'] > 0
                and record[arm]['global_delta'] <= 0 for record in eligible),
            'fp32_bound': {
                'maximum': max(bounds),
                'minimum_absolute_outside': min(map(abs, outside)),
                'all_outside_values_exceed_bound': all(
                    record[arm]['outside_exceeds_bound'] for record in eligible),
                'minimum_absolute_value_to_bound_ratio': min(
                    abs(record[arm]['outside_sum_delta'])
                    / record[arm]['recorded_fp32_ulp_bound'] for record in eligible),
            },
        }
    changes = [record['local_aux']['outside_sum_delta']
               - record['control']['outside_sum_delta'] for record in eligible]
    lost_eligible = [record for record in eligible if record['lost']]
    decisive_eligible = [record for record in decisive if record['same_counts']]
    decisive_values = {
        key: summary([record['local_aux'][key] for record in decisive_eligible])
        for key in ('local_margin', 'residual_delta', 'base_delta', 'global_delta',
                    'outside_sum_delta')}
    decisive_values['outside_change_from_control'] = summary([
        record['local_aux']['outside_sum_delta']
        - record['control']['outside_sum_delta'] for record in decisive_eligible])

    result = {
        'schema': SCHEMA, 'completed': True, 'scope': 'in_sample_fit_only',
        'inputs': {str(path.relative_to(ROOT)): digest for path, digest in EXPECTED.items()},
        'code_sha256': sha(Path(__file__).resolve()),
        'formula': {
            'eligibility': ('single selected path per candidate, one witness, equal complete '
                            'span count N, equal changed-block span count K'),
            'outside_sum_delta': 'N * (preferred_residual - dispreferred_residual) - K * local_margin',
            'recorded_fp32_ulp_bound': (
                '8 * (N + K + 2) * float32_ulp(max(abs(preferred_residual), '
                'abs(dispreferred_residual), abs(local_margin), 1))'),
            'bound_interpretation': (
                'deliberately padded diagnostic bound over recorded FP32 aggregates; '
                'raw per-span reduction intermediates are unavailable'),
        },
        'qualification': {
            'fit_rows': len(preference_rows),
            'selected_candidate_texts': selected_path_counts['candidate_texts'],
            'selected_paths': selected_path_counts['paths'],
            'multi_path_selected_candidates': selected_path_counts['multi_path_candidate_texts'],
            'preferences': witness_counts['preferences'],
            'witnesses': witness_counts['witnesses'],
            'multiple_witness_preferences': witness_counts['multiple_witness_preferences'],
            'equal_complete_and_changed_span_counts': len(eligible),
            'excluded_unequal_complete_and_changed_span_counts': len(pair_records) - len(eligible),
        },
        'fit_choice': {
            'control_correct': sum(row['correct'] for row in control_after.values()),
            'local_aux_correct': sum(row['correct'] for row in auxiliary.values()),
            'gained_rows': len(gained), 'lost_rows': len(lost),
        },
        'arms': arms,
        'outside_change_local_aux_minus_control': {
            **summary(changes),
            'improved': sum(value > 0 for value in changes),
            'worsened': sum(value < 0 for value in changes),
        },
        'objective_sign_transitions': dict(sorted(transition_counts.items())),
        'lost_rows': {
            'rows': len(lost), 'preference_pairs': lost_pair_count,
            'equal_count_pairs': len(lost_eligible),
            'equal_count_rows': len({record['row_id'] for record in lost_eligible}),
            'outside_worsened': sum(
                record['local_aux']['outside_sum_delta']
                < record['control']['outside_sum_delta'] for record in lost_eligible),
            'local_improved_outside_worsened': sum(
                record['local_aux']['local_margin'] > record['control']['local_margin']
                and record['local_aux']['outside_sum_delta']
                < record['control']['outside_sum_delta'] for record in lost_eligible),
            'local_positive_residual_nonpositive': sum(
                record['local_aux']['local_margin'] > 0
                and record['local_aux']['residual_delta'] <= 0 for record in lost_eligible),
            'local_positive_global_nonpositive': sum(
                record['local_aux']['local_margin'] > 0
                and record['local_aux']['global_delta'] <= 0 for record in lost_eligible),
        },
        'decisive_positive_vs_new_argmax': {
            'lost_rows': len(lost), 'locally_supervised': len(decisive),
            'equal_count_pairs': len(decisive_eligible),
            'local_positive': sum(record['local_aux']['local_margin'] > 0
                                  for record in decisive),
            'frozen_base_negative': sum(record['local_aux']['base_delta'] < 0
                                        for record in decisive),
            'equal_count_residual_positive': sum(
                record['local_aux']['residual_delta'] > 0 for record in decisive_eligible),
            'equal_count_outside_negative': sum(
                record['local_aux']['outside_sum_delta'] < 0 for record in decisive_eligible),
            'equal_count_outside_worsened': sum(
                record['local_aux']['outside_sum_delta']
                < record['control']['outside_sum_delta'] for record in decisive_eligible),
            'equal_count_local_improved_outside_worsened': sum(
                record['local_aux']['local_margin'] > record['control']['local_margin']
                and record['local_aux']['outside_sum_delta']
                < record['control']['outside_sum_delta'] for record in decisive_eligible),
            'aggregates': decisive_values,
        },
        'interpretation': {
            'evidence': (
                'Identical outside producer edges do not cancel in the recorded candidate '
                'residuals, even when N and K match. Local preference improvement therefore '
                'does not determine the candidate residual or frozen-base-plus-residual order.'),
            'architectural_hypothesis': (
                'Candidate-dependent frozen output states and output-relative features let '
                'an unchanged edge receive a different score in each complete candidate; '
                'this aggregate audit cannot isolate their individual contributions.'),
            'next_invariant': (
                'Identical outside edges must share a score or be omitted from the pair delta; '
                'aggregate edge weights should be normalized by canonical source length rather '
                'than producer-span count.'),
            'source_length_limit': (
                'Source-length weighting conserves total weight when an interval is split, but '
                'does not by itself guarantee equal scores for differently segmented paths; '
                'the edge scorer/decomposition must also be segmentation-consistent.'),
        },
        'contracts': {
            'models_or_tokenizers_loaded': False,
            'torch_imported': False,
            'dev_hold_test_or_canonical_labels_opened': False,
            'individual_ids_or_texts_written': False,
            'fitting_or_threshold_selection_performed': False,
            'production_changed': False,
        },
        'limitations': [
            'This is an in-sample FIT algebraic audit and makes no held-out quality claim.',
            'The formula is used only for single-path, single-witness pairs with equal N and K.',
            'Recorded residuals and margins do not expose individual outside-span logits, so the '
            'outside sum is aggregate attribution rather than per-span causal evidence.',
            'The padded ULP bound is a reproducible diagnostic bound, not a formal reconstruction '
            'of unavailable FP32 reduction intermediates.',
            'Frozen base scores still affect global candidate order; three of fifteen lost rows '
            'have no local preference for the positive-versus-new-argmax pair.',
            'No exception, loss weight, threshold, hyperparameter, or deployment decision is selected.',
        ],
    }
    # This intentionally contains aggregates only. Recursively guard against the
    # row fields most likely to leak an individual training example.
    def object_keys(value):
        if isinstance(value, dict):
            return set(value) | set().union(*(object_keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(object_keys(item) for item in value)) if value else set()
        return set()
    require(not object_keys(result).intersection({
        'id', 'input', 'text', 'candidates', 'preference_id', 'path_id'}),
            'aggregate report contains a row-level field')
    write(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    value = run(parser.parse_args().output)
    print(json.dumps({
        'qualified_pairs': value['qualification']['equal_complete_and_changed_span_counts'],
        'lost_rows': value['fit_choice']['lost_rows'],
        'decisive_pairs': value['decisive_positive_vs_new_argmax']['locally_supervised'],
    }, sort_keys=True))
