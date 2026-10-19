"""Source-only finite selector evaluation with object and raw preservation checks."""
from copy import deepcopy

from scripts.finite_final_choice_policy import select_candidates


def evaluate_case(ranker, row, suite):
    candidates = row['candidates']
    before = deepcopy(candidates)
    chosen = select_candidates(ranker,row['input'],row['context'],candidates)
    index, ordered = chosen['chosen_index'], chosen['candidates']
    order = [candidates[index],*candidates[:index],*candidates[index+1:]]
    if candidates != before or len(order) != len(ordered) or any(a is not b for a,b in zip(order,ordered)):
        raise RuntimeError('finite selector mutated candidate objects or content')
    raw = row['raw_bypass_response']
    raw_before = deepcopy(raw)
    bypass = select_candidates(None,row['input'],row['context'],raw['candidates'],raw=True)
    if (raw != raw_before or bypass['chosen_index'] != 0 or bypass['reason'] != 'raw_mode'
            or bypass['candidates'] is not raw['candidates'] or bypass['result'] != row['input']
            or raw['result'] != row['input']):
        raise RuntimeError('finite selector changed raw bypass')
    # The new tokenizer budget comes from the finite view. Old capture budget
    # flags describe the parent/all-candidate tokenizer and are not reused.
    expected = row['expected']
    return {'id':row['id'],'group_id':row['group_id'],'suite':suite,
            'chosen_index':index,'reason':chosen['reason'],
            'selector_scored':chosen['selector_scored'],
            'scored_source_indices':chosen['scored_source_indices'],
            'selector_scores':chosen['selector_scores'],
            'before_candidates':before,'candidates':deepcopy(ordered),
            'raw_bypass_response':raw_before,'raw_bypass_preserved':True,
            'before_top1':before[0]['text'] in expected,'top1':ordered[0]['text'] in expected,
            'before_top5':any(c['text'] in expected for c in before),
            'top5':any(c['text'] in expected for c in ordered),
            'candidate_objects_and_content_preserved':True,'accepted_action':index != 0}


def summarize(rows):
    return {'rows':len(rows),'before_top1':sum(r['before_top1'] for r in rows),
            'top1':sum(r['top1'] for r in rows),
            'before_top5':sum(r['before_top5'] for r in rows),'top5':sum(r['top5'] for r in rows),
            'corrections':[r['id'] for r in rows if r['top1'] and not r['before_top1']],
            'regressions':[r['id'] for r in rows if r['before_top1'] and not r['top1']],
            'top5_losses':[r['id'] for r in rows if r['before_top5'] and not r['top5']],
            'accepted_actions':sum(r['accepted_action'] for r in rows),
            'scored_rows':sum(r['selector_scored'] for r in rows)}


def evaluation_gate(suites):
    if set(suites) != {'natural_evaluation_v3','challenge_preservation'}:
        raise ValueError('both finite selector evaluation suites are required')
    natural = suites['natural_evaluation_v3']
    gain = natural['top1'] - natural['before_top1']
    losses = any(s['regressions'] or s['top5_losses'] for s in suites.values())
    actions = sum(s['accepted_actions'] for s in suites.values())
    passed = gain >= 2 and not losses and actions > 0
    return {'passed':passed,'eligible_for_development':passed,
            'natural_net_top1_gain':gain,'zero_row_regressions':not losses,
            'accepted_actions':actions,'inconclusive':actions == 0}
