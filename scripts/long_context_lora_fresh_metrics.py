"""Fixed paired document bootstrap for the reviewed fresh evaluation gate."""
from __future__ import annotations

from collections import defaultdict
import math
import random


def paired_document_bootstrap(rows):
    """Consume completed row outcomes, retaining within-document correlation.

    Each row has a unique id, group_id and boolean before_correct/after_correct.
    This function belongs after inference and must never influence prediction.
    """
    groups, ids = defaultdict(lambda: [0, 0]), set()
    corrections = regressions = 0
    for row in rows:
        identity, group = row.get("id"), row.get("group_id")
        before, after = row.get("before_correct"), row.get("after_correct")
        if (not isinstance(identity, str) or not identity or identity in ids
                or not isinstance(group, str) or not group
                or type(before) is not bool or type(after) is not bool):
            raise ValueError("bootstrap requires unique rows, document groups and boolean outcomes")
        ids.add(identity)
        groups[group][0] += int(after) - int(before)
        groups[group][1] += 1
        corrections += int(after and not before)
        regressions += int(before and not after)
    if not groups:
        raise ValueError("bootstrap requires nonempty completed outcomes")
    ordered = [groups[group] for group in sorted(groups)]
    rng = random.Random(20260915)
    deltas = []
    for _ in range(2000):
        selected = rng.choices(ordered, k=len(ordered))
        deltas.append(sum(group[0] for group in selected) / sum(group[1] for group in selected))
    deltas.sort()
    lower_index = math.floor(0.025 * (2000 - 1))
    gain = corrections - regressions
    return {"rows": len(ids), "groups": len(groups), "corrections": corrections,
            "regressions": regressions, "net_gain": gain,
            "accuracy_delta": gain / len(ids), "resamples": 2000,
            "seed": 20260915, "lower_index": lower_index,
            "lower_2_5_percentile": deltas[lower_index],
            "gain_and_interval_passed": gain >= 8 and deltas[lower_index] > 0}
