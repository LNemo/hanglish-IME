"""Pure FIT example and schedule helpers for finite-candidate selector v3."""
from __future__ import annotations

from collections import Counter
import random

from scripts.finite_final_choice_policy import candidate_view
from scripts.train_final_choice_selector import SelectorExample


ROLES = ("natural_fit", "challenge_fit")
EXPECTED = {
    "examples": 2791,
    "parent_correct": 2712,
    "parent_error": 79,
    "optimizer_steps": 904,
}
SEED = 23
PRESERVE_PER_BATCH = 3


def _build_example(row, role, ranker):
    if role not in ROLES:
        raise ValueError("finite selector training accepts FIT roles only")
    source = row.get("input")
    candidates = row.get("candidates")
    if not isinstance(source, str) or not source or not isinstance(candidates, list):
        raise ValueError("finite selector FIT row is malformed")

    # This source-only view must be complete before any target alignment.
    finite, reason = candidate_view(ranker, source, candidates)
    if row.get("finite_indices") != finite:
        raise ValueError("stored finite candidate indices differ from the current view")
    if reason is not None:
        return None, reason

    accepted = row.get("accepted_positive_indices")
    expected = row.get("expected")
    if (not isinstance(accepted, list)
            or any(type(index) is not int for index in accepted)
            or len(accepted) != len(set(accepted))
            or not isinstance(expected, list) or not expected
            or any(not isinstance(value, str) or not value for value in expected)):
        raise ValueError("stored FIT target alignment is malformed")
    actual_accepted = [index for index in finite
                       if candidates[index].get("text") in expected]
    if accepted != actual_accepted or any(index not in finite for index in accepted):
        raise ValueError("stored positive alignment differs from the finite view")
    if not accepted:
        return None, "no_eligible_positive"

    compact = {source_index: position for position, source_index in enumerate(finite)}
    scores = tuple(float(candidates[index]["score"]) for index in finite)
    return SelectorExample(
        case_id=row["id"], group_id=row["group_id"], role=role,
        context=row["context"],
        candidates=tuple(candidates[index]["text"] for index in finite),
        positives=tuple(compact[index] for index in accepted),
        teacher_scores=scores,
        finite_source_indices=tuple(finite),
        parent_correct=0 in accepted,
        loss_weight=1.,
    ), None


def build_example(row, role, ranker):
    """Build one compact finite example without mutating its captured row."""
    return _build_example(row, role, ranker)[0]


def build_all_fit(natural_rows, challenge_rows, ranker, *, enforce_expected=True):
    examples = []
    exclusions = Counter()
    role_counts = Counter()
    for role, rows in (("natural_fit", natural_rows),
                       ("challenge_fit", challenge_rows)):
        for row in rows:
            example, reason = _build_example(row, role, ranker)
            if example is None:
                exclusions[reason] += 1
                continue
            examples.append(example)
            role_counts[role] += 1
    summary = {
        "examples": len(examples),
        "parent_correct": sum(example.parent_correct for example in examples),
        "parent_error": sum(not example.parent_correct for example in examples),
        "roles": dict(sorted(role_counts.items())),
        "excluded": dict(sorted(exclusions.items())),
    }
    if enforce_expected and any(summary[key] != EXPECTED[key]
                                for key in ("examples", "parent_correct", "parent_error")):
        raise ValueError("finite selector FIT eligibility counts differ from the frozen plan")
    return examples, summary


def balanced_schedule(examples, *, seed=SEED,
                      preserve_per_batch=PRESERVE_PER_BATCH):
    """Use every preservation case once and cycle one shuffled repair per batch."""
    if type(seed) is not int or type(preserve_per_batch) is not int or preserve_per_batch <= 0:
        raise ValueError("finite selector schedule parameters are invalid")
    identities = [(example.role, example.case_id) for example in examples]
    if len(identities) != len(set(identities)):
        raise ValueError("finite selector example identities are duplicated")
    preserve = [example for example in examples if example.parent_correct]
    repairs = [example for example in examples if not example.parent_correct]
    if not preserve or not repairs:
        raise ValueError("finite selector schedule requires preserve and repair examples")
    preserve_rng, repair_rng = random.Random(seed), random.Random(seed)
    preserve_rng.shuffle(preserve)
    batches = []
    repair_cycle, repair_cursor = [], 0
    for start in range(0, len(preserve), preserve_per_batch):
        if repair_cursor == len(repair_cycle):
            repair_cycle = list(repairs)
            repair_rng.shuffle(repair_cycle)
            repair_cursor = 0
        batches.append((repair_cycle[repair_cursor],
                        *preserve[start:start + preserve_per_batch]))
        repair_cursor += 1
    observed_preserve = [example for batch in batches for example in batch[1:]]
    if (Counter((example.role, example.case_id) for example in observed_preserve)
            != Counter((example.role, example.case_id) for example in preserve)
            or any(batch[0].parent_correct for batch in batches)
            or any(not example.parent_correct for batch in batches for example in batch[1:])
            or any(len(batch) != preserve_per_batch + 1 for batch in batches[:-1])):
        raise RuntimeError("finite selector balanced schedule invariants failed")
    return batches


__all__ = ["EXPECTED", "build_example", "build_all_fit", "balanced_schedule"]
