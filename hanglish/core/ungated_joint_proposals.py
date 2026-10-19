"""Compose captured phonetic edges with the frozen source-edge search.

Both seed modes build one identical pool from parent paths and already captured
single-span proposals.  ``roman_repaired`` changes only the first four search
seeds, and only with beam outputs that are also present as edges in that pool.
No model, ranker, label, or score is consulted here.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import re

from hanglish.core.source_correspondence import align
from hanglish.core.source_edge_recombination import MAX_PROPOSALS, propose
from hanglish.core.ungated_phonetic_proposals import _validate_beam


_ELIGIBLE = re.compile(r"[a-z]{2,40}\Z")
_ASCII_OUTPUT = re.compile(r"[A-Za-z]")
_SEED_MODES = {"original", "roman_repaired"}
_MAX_SINGLE_PROPOSALS = 96


def _stable_id(prefix: str, value) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return prefix + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _rows(value, name, *, maximum=None):
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence")
    rows = list(value)
    if maximum is not None and len(rows) > maximum:
        raise ValueError(f"{name} exceeds {maximum}")
    return rows


def _normalize(source: str, rows: list, role: str) -> list[dict]:
    result = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"{role} row {index} must be a mapping")
        text = row.get("text")
        segments = row.get("segments")
        path = align(source, text, segments)
        identity = _stable_id("ungated-joint-candidate-v1-", {
            "source": source,
            "role": role,
            "row_index": index,
            "path_id": path.path_id,
        })
        origin = f"ungated_joint_{role}"
        result.append({
            "candidate_id": identity,
            "path_id": path.path_id,
            "text": text,
            "segments": deepcopy(segments),
            "origin": origin,
            "origins": [origin, f"{role}_row_{index}"],
        })
    return result


def _validated_beams(generated_by_source) -> dict[str, list[tuple[str, float]]]:
    if not isinstance(generated_by_source, Mapping):
        raise TypeError("generated_by_source must be a mapping")
    result = {}
    for source_word, beam in generated_by_source.items():
        if not isinstance(source_word, str) or _ELIGIBLE.fullmatch(source_word) is None:
            raise ValueError("generated_by_source has an invalid Roman source")
        result[source_word] = _validate_beam(source_word, beam)
    return result


def _edge_lookup(pool: list[dict]):
    result = {}
    for row in pool:
        for segment in row["segments"]:
            key = (segment["input_start"], segment["input_end"], segment["text"])
            result.setdefault(key, deepcopy(segment))
    return result


def _seed_rows(source: str, parents: list[dict], beams, edges, seed_mode: str, repair_mixed: bool):
    seeds = []
    for parent_index, parent in enumerate(parents[:4]):
        segments = deepcopy(parent["segments"])
        replacements = []
        if seed_mode == "roman_repaired":
            for segment_index, segment in enumerate(segments):
                source_word = segment["source"]
                unrepaired = (bool(_ASCII_OUTPUT.search(segment["text"])) if repair_mixed
                              else segment["text"] == source_word)
                if not unrepaired or _ELIGIBLE.fullmatch(source_word) is None:
                    continue
                admitted = None
                for target, _ in beams.get(source_word, ()):
                    key = (segment["input_start"], segment["input_end"], target)
                    if key in edges:
                        admitted = edges[key]
                        break
                if admitted is None:
                    continue
                segments[segment_index] = deepcopy(admitted)
                replacements.append({
                    "segment_index": segment_index,
                    "source_start": segment["input_start"],
                    "source_end": segment["input_end"],
                    "source": source_word,
                    "replacement": admitted["text"],
                })
        text = "".join(segment["text"] for segment in segments)
        # Revalidate every seed independently before handing it to the search.
        align(source, text, segments)
        seeds.append({
            "seed_index": len(seeds),
            "parent_index": parent_index,
            "original_text": parent["text"],
            "text": text,
            "segments": segments,
            "replacements": replacements,
        })
    return seeds


def _edge_ids(edge_inventory):
    return {
        (row["structural_segment"]["input_start"],
         row["structural_segment"]["input_end"],
         row["structural_segment"]["text"]): row["edge_id"]
        for row in edge_inventory
    }


def _repair_seed_proposal(source: str, seed: dict, edge_ids: dict) -> dict:
    identifiers = [edge_ids[(segment["input_start"], segment["input_end"],
                             segment["text"])]
                   for segment in seed["segments"]]
    return {
        "proposal_id": _stable_id("ungated-joint-seed-v1-", {
            "source": source, "edge_ids": identifiers,
        }),
        "text": seed["text"],
        "segments": deepcopy(seed["segments"]),
        "edge_ids": identifiers,
        "seed_index": seed["seed_index"],
        "seed_text": seed["text"],
        "changed_edges": len(seed["replacements"]),
        "score": None,
        "score_type": None,
        "origin": "ungated_joint_repair_seed",
        "selection_eligible": False,
    }


def _beam_priority_candidates(source, seeds, beams, edges, edge_ids):
    candidates = []
    for seed in seeds:
        for segment_index, segment in enumerate(seed["segments"]):
            source_word = segment["source"]
            if _ELIGIBLE.fullmatch(source_word) is None:
                continue
            for beam_index, (target, _) in enumerate(beams.get(source_word, ())):
                key = (segment["input_start"], segment["input_end"], target)
                admitted = edges.get(key)
                if admitted is None or admitted["text"] == segment["text"]:
                    continue
                replacement = deepcopy(seed["segments"])
                replacement[segment_index] = deepcopy(admitted)
                text = "".join(row["text"] for row in replacement)
                align(source, text, replacement)
                identifiers = [edge_ids[(row["input_start"], row["input_end"],
                                         row["text"])]
                               for row in replacement]
                proposal = {
                    "proposal_id": _stable_id("ungated-joint-priority-v1-", {
                        "source": source, "edge_ids": identifiers,
                    }),
                    "text": text,
                    "segments": replacement,
                    "edge_ids": identifiers,
                    "seed_index": seed["seed_index"],
                    "seed_text": seed["text"],
                    "changed_edges": 1,
                    "score": None,
                    "score_type": None,
                    "origin": "ungated_joint_beam_priority",
                    "selection_eligible": False,
                    "beam_index": beam_index,
                    "source_start": segment["input_start"],
                    "source_end": segment["input_end"],
                    "source": source_word,
                    "replacement": target,
                    "segment_index": segment_index,
                    "parent_index": seed["parent_index"],
                }
                order = (beam_index, segment["input_start"],
                         seed["seed_index"], segment_index)
                candidates.append((order, proposal))
    candidates.sort(key=lambda row: row[0])
    return [row[1] for row in candidates]


def propose_joint(
        source: str,
        parents,
        single_proposals,
        generated_by_source,
        *,
        seed_mode: str = "original",
        repair_mixed: bool = False,
        prioritize_beams: bool = False,
) -> dict:
    """Return at most 96 joint paths from one captured, source-only graph."""
    if not isinstance(source, str):
        raise TypeError("source must be a string")
    if seed_mode not in _SEED_MODES:
        raise ValueError("seed_mode must be 'original' or 'roman_repaired'")
    if type(repair_mixed) is not bool:
        raise TypeError("repair_mixed must be boolean")
    if type(prioritize_beams) is not bool:
        raise TypeError("prioritize_beams must be boolean")
    parent_rows = _rows(parents, "parents")
    single_rows = _rows(
        single_proposals, "single_proposals", maximum=_MAX_SINGLE_PROPOSALS)
    normalized_parents = _normalize(source, parent_rows, "parent")
    normalized_singles = _normalize(source, single_rows, "single_proposal")
    pool = [*normalized_parents, *normalized_singles]
    beams = _validated_beams(generated_by_source)
    edges = _edge_lookup(pool)
    seeds = _seed_rows(source, normalized_parents, beams, edges, seed_mode, repair_mixed)

    recombined = propose(source, pool, seeds)
    inventory_ids = _edge_ids(recombined["edge_inventory"])
    old_texts = {source, *(row["text"] for row in pool)}
    seen = set(old_texts)
    proposals = []
    repair_seed_candidates = 0
    if seed_mode == "roman_repaired":
        for seed in seeds:
            if not seed["replacements"] or seed["text"] in seen:
                continue
            repair_seed_candidates += 1
            proposal = _repair_seed_proposal(source, seed, inventory_ids)
            proposals.append(proposal)
            seen.add(seed["text"])
            if len(proposals) == MAX_PROPOSALS:
                break

    beam_priority_examined = 0
    beam_priority_retained = 0
    use_beam_priority = prioritize_beams and seed_mode == "roman_repaired"
    if use_beam_priority:
        priority = _beam_priority_candidates(
            source, seeds, beams, edges, inventory_ids)
        beam_priority_examined = len(priority)
        for proposal in priority:
            if len(proposals) == MAX_PROPOSALS:
                break
            if proposal["text"] in seen:
                continue
            proposals.append(proposal)
            seen.add(proposal["text"])
            beam_priority_retained += 1

    recombination_retained = 0
    for proposal in recombined["proposals"]:
        if len(proposals) == MAX_PROPOSALS:
            break
        if proposal["text"] in seen:
            continue
        proposals.append(deepcopy(proposal))
        seen.add(proposal["text"])
        recombination_retained += 1

    diagnostics = {
        "limits": deepcopy(recombined["diagnostics"]["limits"]),
        "pool_rows": len(pool),
        "parent_rows": len(normalized_parents),
        "single_proposal_rows": len(normalized_singles),
        "edge_inventory": deepcopy(recombined["edge_inventory"]),
        "edge_count": recombined["diagnostics"]["edge_count"],
        "repair_seed_candidate_count": repair_seed_candidates,
        "repair_seed_retained_count": sum(
            row["origin"] == "ungated_joint_repair_seed" for row in proposals),
        "recombination_generated_count": len(recombined["proposals"]),
        "recombination_retained_count": recombination_retained,
        "recombination_truncated_count": (
            len(recombined["proposals"]) - recombination_retained),
        "proposal_count": len(proposals),
        "proposal_limit_reached": len(proposals) == MAX_PROPOSALS,
        "score_fields_consumed": False,
        "recombination": deepcopy(recombined["diagnostics"]),
    }
    if use_beam_priority:
        diagnostics.update({
            "beam_priority_examined_count": beam_priority_examined,
            "beam_priority_retained_count": beam_priority_retained,
        })
    return {
        "proposals": proposals,
        "seeds": deepcopy(seeds),
        "diagnostics": diagnostics,
        "seed_mode": seed_mode,
    }


__all__ = ["propose_joint"]
