"""Bounded source-only recombination of edges already present in a pool."""
from __future__ import annotations

from copy import deepcopy
from collections import defaultdict, deque
import hashlib
import heapq
import json
import math


SCHEMA_VERSION = "source-edge-recombination-v1"
MAX_PROPOSALS = 96
MAX_POPS_PER_SEED = 20_000
MAX_FRONTIER_PER_SEED = 4_000
MAX_SEEDS = 4


def _stable_id(prefix, value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"))
    return prefix + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _validate_segments(source, text, segments, label):
    _require(isinstance(segments, list), f"{label} segments are malformed")
    cursor = 0
    output = []
    for segment in segments:
        _require(isinstance(segment, dict)
                 and {"input_start", "input_end", "source", "text", "kind"}
                 <= set(segment), f"{label} segment fields are malformed")
        start, end = segment["input_start"], segment["input_end"]
        _require(not isinstance(start, bool) and isinstance(start, int)
                 and not isinstance(end, bool) and isinstance(end, int)
                 and start == cursor and start < end <= len(source),
                 f"{label} source coverage is not forward and contiguous")
        _require(isinstance(segment["source"], str)
                 and segment["source"] == source[start:end]
                 and isinstance(segment["text"], str)
                 and isinstance(segment["kind"], str),
                 f"{label} source coverage differs")
        cursor = end
        output.append(segment["text"])
    _require(cursor == len(source) and "".join(output) == text,
             f"{label} source coverage or output differs")


def _validate_pool(source, pool):
    _require(isinstance(source, str), "source must be a string")
    _require(isinstance(pool, list), "pool must be a list")
    for index, row in enumerate(pool):
        _require(isinstance(row, dict)
                 and isinstance(row.get("candidate_id"), str)
                 and isinstance(row.get("path_id"), str)
                 and isinstance(row.get("text"), str)
                 and isinstance(row.get("origin"), str)
                 and isinstance(row.get("origins"), list)
                 and all(isinstance(value, str) for value in row["origins"]),
                 f"pool row {index} is malformed")
        _validate_segments(source, row["text"], row.get("segments"),
                           f"pool row {index}")


def _validate_bases(source, base_candidates):
    _require(isinstance(base_candidates, list),
             "base_candidates must be a list")
    for index, row in enumerate(base_candidates):
        _require(isinstance(row, dict) and isinstance(row.get("text"), str),
                 f"base candidate {index} is malformed")
        _validate_segments(source, row["text"], row.get("segments"),
                           f"base candidate {index}")


def _edge_graph(source, pool):
    edges = []
    by_key = {}
    occurrences = 0
    for pool_index, row in enumerate(pool):
        for segment_index, segment in enumerate(row["segments"]):
            occurrences += 1
            key = (segment["input_start"], segment["input_end"], segment["text"])
            provenance = {
                "candidate_id": row["candidate_id"],
                "path_id": row["path_id"],
                "pool_index": pool_index,
                "segment_index": segment_index,
                "origin": row["origin"],
                "origins": list(row["origins"]),
                "kind": segment["kind"],
            }
            edge = by_key.get(key)
            if edge is None:
                copied = segment["source"] == segment["text"]
                edge = {
                    "edge_id": _stable_id("edge-v1-", {
                        "source": source, "input_start": key[0],
                        "input_end": key[1], "text": key[2],
                    }),
                    "order": len(edges),
                    "input_start": key[0],
                    "input_end": key[1],
                    "source": segment["source"],
                    "text": key[2],
                    "action": "copy" if copied else "convert",
                    "copy_exact": True if copied else None,
                    "segment": deepcopy(segment),
                    "provenance": [],
                }
                by_key[key] = edge
                edges.append(edge)
            edge["provenance"].append(provenance)
    adjacency = defaultdict(list)
    for edge in edges:
        adjacency[edge["input_start"]].append(edge)
    return edges, adjacency, occurrences


def _seed_rows(source, base_candidates):
    seeds = []
    for base_index, row in enumerate(base_candidates):
        if row["text"] == source:
            continue
        if len(seeds) == MAX_SEEDS:
            break
        keys = {(segment["input_start"], segment["input_end"], segment["text"])
                for segment in row["segments"]}
        seeds.append({"seed_index": len(seeds),
                      "base_candidate_index": base_index,
                      "seed_text": row["text"], "edge_keys": keys})
    return seeds


class _SeedSearch:
    def __init__(self, source, adjacency, seed):
        self.source = source
        self.adjacency = adjacency
        self.seed = seed
        self.cost = lambda edge: int((edge["input_start"], edge["input_end"],
                                      edge["text"]) not in seed["edge_keys"])
        self.heuristic = self._suffix_costs()
        self.frontier = []
        self.counter = 0
        self.pops = 0
        self.max_frontier = 0
        self.complete_paths_examined = 0
        self.accepted_proposals = 0
        self.termination_reason = None
        if math.isinf(self.heuristic[0]):
            self.termination_reason = "no_complete_path"
        else:
            self._push((self.heuristic[0], 0, (), self.counter, 0, ()))

    def _suffix_costs(self):
        result = [math.inf] * (len(self.source) + 1)
        result[len(self.source)] = 0
        for position in range(len(self.source) - 1, -1, -1):
            values = [self.cost(edge) + result[edge["input_end"]]
                      for edge in self.adjacency.get(position, ())
                      if not math.isinf(result[edge["input_end"]])]
            if values:
                result[position] = min(values)
        return result

    def _push(self, value):
        heapq.heappush(self.frontier, value)
        self.max_frontier = max(self.max_frontier, len(self.frontier))

    def _terminate(self, reason):
        self.termination_reason = reason
        self.frontier.clear()

    def next_complete(self):
        while self.termination_reason is None:
            if not self.frontier:
                self._terminate("exhausted")
                return None
            if self.pops >= MAX_POPS_PER_SEED:
                self._terminate("max_pops")
                return None
            _, changed, order_path, _, position, path = heapq.heappop(
                self.frontier)
            self.pops += 1
            if position == len(self.source):
                self.complete_paths_examined += 1
                return path, changed
            for edge in self.adjacency.get(position, ()):
                suffix = self.heuristic[edge["input_end"]]
                if math.isinf(suffix):
                    continue
                if len(self.frontier) >= MAX_FRONTIER_PER_SEED:
                    self._terminate("max_frontier")
                    return None
                next_changed = changed + self.cost(edge)
                next_orders = order_path + (edge["order"],)
                self.counter += 1
                self._push((next_changed + suffix, next_changed, next_orders,
                            self.counter, edge["input_end"], path + (edge,)))
        return None

    def diagnostics(self):
        return {
            "seed_index": self.seed["seed_index"],
            "base_candidate_index": self.seed["base_candidate_index"],
            "seed_text": self.seed["seed_text"],
            "pops": self.pops,
            "max_frontier": self.max_frontier,
            "complete_paths_examined": self.complete_paths_examined,
            "accepted_proposals": self.accepted_proposals,
            "termination_reason": self.termination_reason,
        }


def _proposal(source, search, path, changed, text):
    segments = [deepcopy(edge["segment"]) for edge in path]
    _validate_segments(source, text, segments, "recombined proposal")
    edge_ids = [edge["edge_id"] for edge in path]
    return {
        "proposal_id": _stable_id("proposal-v1-", {
            "source": source, "edge_ids": edge_ids,
        }),
        "text": text,
        "segments": segments,
        "edge_ids": edge_ids,
        "seed_index": search.seed["seed_index"],
        "seed_text": search.seed["seed_text"],
        "changed_edges": changed,
        "score": None,
        "score_type": None,
        "origin": "source_edge_recombination",
        "selection_eligible": False,
    }


def propose(source, pool, base_candidates):
    """Return novel complete paths built only from validated existing edges."""
    _validate_pool(source, pool)
    _validate_bases(source, base_candidates)
    edges, adjacency, occurrences = _edge_graph(source, pool)
    seeds = _seed_rows(source, base_candidates)
    searches = [_SeedSearch(source, adjacency, seed) for seed in seeds]
    active = deque(search for search in searches
                   if search.termination_reason is None)
    original_texts = {source}
    original_texts.update(row["text"] for row in pool)
    original_texts.update(row["text"] for row in base_candidates)
    seen = set(original_texts)
    proposals = []
    while active and len(proposals) < MAX_PROPOSALS:
        search = active.popleft()
        value = search.next_complete()
        if value is None:
            continue
        active.append(search)
        path, changed = value
        text = "".join(edge["text"] for edge in path)
        if text in seen:
            continue
        seen.add(text)
        proposal = _proposal(source, search, path, changed, text)
        proposals.append(proposal)
        search.accepted_proposals += 1
    global_limit = len(proposals) == MAX_PROPOSALS and bool(active)
    if global_limit:
        for search in active:
            if search.termination_reason is None:
                search._terminate("global_proposal_limit")
    # Searches that were initialized as unreachable never entered the queue;
    # all others have a concrete terminal reason after the loop.
    for search in searches:
        if search.termination_reason is None:
            search._terminate("exhausted")
    diagnostics = {
        "limits": {"max_proposals": MAX_PROPOSALS,
                   "max_pops_per_seed": MAX_POPS_PER_SEED,
                   "max_frontier_per_seed": MAX_FRONTIER_PER_SEED,
                   "max_seeds": MAX_SEEDS},
        "source_length": len(source),
        "pool_rows": len(pool),
        "edge_occurrences": occurrences,
        "edge_count": len(edges),
        "seed_count": len(seeds),
        "seeds": [search.diagnostics() for search in searches],
        "proposal_count": len(proposals),
        "global_limit_reached": global_limit,
        "original_text_count": len(original_texts),
        "score_fields_consumed": False,
    }
    edge_inventory = [{
        "edge_id": edge["edge_id"],
        "action": edge["action"],
        "copy_exact": edge["copy_exact"],
        "structural_segment": deepcopy(edge["segment"]),
        "provenance": deepcopy(edge["provenance"]),
    } for edge in edges]
    return {"schema_version": SCHEMA_VERSION,
            "edge_inventory": edge_inventory,
            "proposals": proposals, "diagnostics": diagnostics}


__all__ = ["MAX_FRONTIER_PER_SEED", "MAX_POPS_PER_SEED", "MAX_PROPOSALS",
           "MAX_SEEDS", "SCHEMA_VERSION", "propose"]
