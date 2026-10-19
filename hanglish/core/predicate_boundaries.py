"""Bounded POS-backed recognition of predicate ㄺ + ㄱ boundaries."""
from functools import lru_cache
import json
from pathlib import Path

INVENTORY = Path(__file__).with_name('data')/'predicate_boundary_inventory.json'


@lru_cache(maxsize=1)
def _inventory():
    data = json.loads(INVENTORY.read_text(encoding='utf-8'))
    if data['schema'] != 'predicate-boundary-inventory-v1':
        raise ValueError('unsupported predicate boundary inventory')
    return (frozenset(data['stems']), data['endings'], data['particles'])


def _particle_tail(text, particles, nominal):
    # A nominalized predicate can take case markers; other endings only take
    # auxiliary particles. Bound the analysis to two attached particles.
    positions = {0}
    for _ in range(2):
        positions = {start+len(particle) for start in positions
                     for particle,tags in particles.items()
                     if (nominal or 'JX' in tags) and text.startswith(particle,start)}
        if len(text) in positions:
            return True
    return False


@lru_cache(maxsize=4096)
def lateral_boundaries(run):
    """Return coda indices justified by a complete stem/ending analysis.

    Unknown stems, embedded compounds and unanalysed suffixes retain the
    ordinary phonology path. This is not a general morphological analyser.
    """
    boundaries = [i for i in range(len(run)-1)
                  if (ord(run[i])-0xAC00)%28==9 and (ord(run[i+1])-0xAC00)//588==0]
    if not boundaries:
        return ()
    stems,endings,particles = _inventory()
    result = []
    for index in boundaries:
        if run[:index+1] not in stems:
            continue
        suffix = run[index+1:]
        if suffix in endings or any(
            suffix.startswith(ending) and _particle_tail(suffix[len(ending):],particles,'ETN' in tags)
            for ending,tags in endings.items()
        ):
            result.append(index)
    return tuple(result)
