"""Pure preparation and sentence scoring for frozen recombination rows.

This module performs no file, model, or label access.  Callers inject the
already-loaded frozen SentenceRanker and independently authenticated source and
census rows.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import re

from hanglish.models.phonetic_choice_features import _path_features, _validate_edge


PREPARED_SCHEMA = "current-recombination-scoring-prepared-v1"
SCORED_SCHEMA = "current-recombination-scoring-row-v1"
PARITY_TOLERANCE = 1e-5
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def require(value, message):
    if not value:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def canonical_sha256(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _segments(source, text, segments, label):
    require(isinstance(segments, list) and segments, label + " segments missing")
    cursor, rendered = 0, []
    for segment in segments:
        start, end = segment.get("input_start"), segment.get("input_end")
        require(isinstance(segment, dict)
                and type(start) is int and type(end) is int
                and start == cursor and start < end <= len(source)
                and segment.get("source") == source[start:end]
                and isinstance(segment.get("text"), str)
                and isinstance(segment.get("kind"), str),
                label + " source edge differs")
        cursor = end; rendered.append(segment["text"])
    require(cursor == len(source) and "".join(rendered) == text,
            label + " reconstruction differs")


def _score_records(choice):
    records = choice.get("score_records")
    require(isinstance(records, list), "captured score records missing")
    by_pair = {}
    for record in records:
        pair_id = record.get("pair_id") if isinstance(record, dict) else None
        require(isinstance(pair_id, str) and pair_id and pair_id not in by_pair,
                "captured score record ID differs")
        by_pair[pair_id] = record
    return by_pair


def _original_rows(source, original, choice, scores):
    paths, rows = original.get("base_paths"), choice.get("rows")
    require(isinstance(paths, list) and isinstance(rows, list)
            and len(paths) == len(rows), "captured original pool/feature rows differ")
    result, locations = [], {}
    for pool_index, (path, row) in enumerate(zip(paths, rows)):
        require(isinstance(path, list) and len(path) == 3
                and isinstance(path[0], str) and finite(path[1])
                and isinstance(path[2], list) and isinstance(row, dict),
                "captured original path differs")
        text, base_score, segments = path
        _segments(source, text, segments, "captured original path")
        require(row.get("text") == text and row.get("base_score") == base_score
                and row.get("segments") == segments
                and isinstance(row.get("edges"), list)
                and len(row["edges"]) == len(segments)
                and isinstance(row.get("features"), list)
                and len(row["features"]) == 4
                and all(finite(value) for value in row["features"]),
                "captured original feature alignment differs")
        edges = row["edges"]
        for segment_index, (segment, edge) in enumerate(zip(segments, edges)):
            _validate_edge(edge, source)
            require(isinstance(edge, dict)
                    and edge.get("input_start") == segment["input_start"]
                    and edge.get("input_end") == segment["input_end"]
                    and edge.get("source") == segment["source"]
                    and edge.get("target") == segment["text"],
                    "captured original edge differs")
            locations[(pool_index, segment_index)] = edge
        path_spec = {"path_id": f"captured-original-{pool_index}",
                     "edge_indices": list(range(len(edges)))}
        recomputed = _path_features(source, text, path_spec, edges, scores)
        require(recomputed == row["features"],
                "captured original feature vector differs")
        result.append({"text": text, "sentence_score": float(base_score),
                       "features": list(recomputed), "segments": deepcopy(segments),
                       "status": "preserved", "score_source": "captured_original"})
    require(len({row["text"] for row in result}) == len(result),
            "captured original texts are duplicated")
    return result, locations


def _inventory(identifier, source, census, original, locations):
    used = census.get("used_edge_inventory")
    require(isinstance(used, list), "used edge inventory missing")
    inventory, exact_edges = {}, {}
    paths = original["base_paths"]
    for item in used:
        edge_id = item.get("edge_id") if isinstance(item, dict) else None
        segment = item.get("structural_segment") if isinstance(item, dict) else None
        provenance = item.get("provenance") if isinstance(item, dict) else None
        require(isinstance(edge_id, str) and edge_id and edge_id not in inventory
                and isinstance(segment, dict) and isinstance(provenance, list)
                and provenance, "used edge inventory differs")
        key = (segment.get("input_start"), segment.get("input_end"), segment.get("text"))
        chosen = None
        for evidence in provenance:
            pool_index = evidence.get("pool_index") if isinstance(evidence, dict) else None
            segment_index = evidence.get("segment_index") if isinstance(evidence, dict) else None
            identity = (f"current-phonetic-fit:{identifier}:{pool_index}"
                        if type(pool_index) is int else None)
            require(type(pool_index) is int and type(segment_index) is int
                    and 0 <= pool_index < len(paths)
                    and 0 <= segment_index < len(paths[pool_index][2])
                    and evidence.get("candidate_id") == identity
                    and evidence.get("path_id") == identity
                    and evidence.get("origin") == "captured_full_union"
                    and evidence.get("origins") == ["captured_full_union"]
                    and evidence.get("kind")
                        == paths[pool_index][2][segment_index].get("kind")
                    and all(paths[pool_index][2][segment_index].get(field)
                            == segment.get(field)
                            for field in ("input_start", "input_end", "source", "text")),
                    "used edge provenance differs")
            edge = locations.get((pool_index, segment_index))
            require(edge is not None
                    and (edge["input_start"], edge["input_end"], edge["target"]) == key,
                    "used edge has no captured score edge")
            require(chosen is None or chosen == edge,
                    "used edge maps to inconsistent captured score edges")
            chosen = edge
        inventory[edge_id] = item
        exact_edges[edge_id] = chosen
    return inventory, exact_edges


def prepare_row(original, census, ranker):
    """Validate one frozen source/census pair and prepare every novel candidate."""
    require(isinstance(original, dict) and isinstance(census, dict),
            "source and census rows must be objects")
    identifier = original.get("id")
    request = original.get("request")
    source = request.get("input") if isinstance(request, dict) else None
    require(isinstance(identifier, str) and identifier
            and request == {"input": source, "context": ""}
            and census.get("schema") == "current-fit-recombination-census-row-v1"
            and census.get("id") == identifier
            and census.get("component_id") == original.get("component_id")
            and census.get("request") == request,
            "source/census identity differs")
    response, paths = original.get("response"), original.get("base_paths")
    require(isinstance(response, dict) and isinstance(paths, list)
            and response.get("input") == source and response.get("context") == ""
            and response.get("external_context_used") is False
            and type(response.get("sentence_ranking_applied")) is bool,
            "captured source response differs")
    public = response.get("candidates")
    require(isinstance(public, list) and public, "captured public candidates missing")
    for candidate in public:
        require(isinstance(candidate, dict) and isinstance(candidate.get("text"), str),
                "captured public candidate differs")
        _segments(source, candidate["text"], candidate.get("segments"),
                  "captured public candidate")
    reference = census.get("source_reference")
    ranked = response["sentence_ranking_applied"]
    require(isinstance(reference, dict)
            and type(reference.get("source_shard")) is int
            and type(reference.get("source_row_index")) is int
            and _DIGEST.fullmatch(str(reference.get("source_capture_sha256"))) is not None
            and reference.get("base_paths_sha256") == canonical_sha256(paths)
            and reference.get("public_candidates_sha256") == canonical_sha256(public)
            and reference.get("pool_rows") == len(paths)
            and reference.get("ranked") is ranked
            and ((ranked and paths and census.get("status") == "generated")
                 or (not ranked and not paths and census.get("status") == "no_pool")),
            "census source reference differs")
    choice = response.get("phonetic_choice")
    require(isinstance(choice, dict), "captured phonetic choice trace missing")
    scores = _score_records(choice)
    originals, locations = _original_rows(source, original, choice, scores)
    require(bool(originals) is ranked, "captured ranked pool differs")
    supported_ids = {edge["pair_id"] for edge in locations.values()
                     if edge.get("status") == "supported"}
    require(set(scores) == supported_ids,
            "captured score record coverage differs")

    inventory, exact_edges = _inventory(identifier, source, census, original, locations)
    proposals = census.get("proposals")
    require(isinstance(proposals, list), "census proposals missing")
    old_texts = {source, *(row["text"] for row in originals),
                 *(row["text"] for row in public)}
    seen, referenced, novel = set(), set(), []
    for index, proposal in enumerate(proposals):
        text = proposal.get("text") if isinstance(proposal, dict) else None
        edge_ids = proposal.get("edge_ids") if isinstance(proposal, dict) else None
        segments = proposal.get("segments") if isinstance(proposal, dict) else None
        require(isinstance(text, str) and text and text not in old_texts and text not in seen
                and isinstance(edge_ids, list) and len(edge_ids) == len(segments)
                and all(edge_id in inventory for edge_id in edge_ids),
                "census proposal novelty or edge coverage differs")
        _segments(source, text, segments, "census proposal")
        edges = []
        for segment, edge_id in zip(segments, edge_ids):
            require(inventory[edge_id]["structural_segment"] == segment,
                    "census proposal edge identity differs")
            edges.append(exact_edges[edge_id]); referenced.add(edge_id)
        features = _path_features(
            source, text, {"path_id": f"recombined-{index}",
                           "edge_indices": list(range(len(edges)))}, edges, scores)
        supported = ranker.can_score(text)
        require(type(supported) is bool, "sentence ranker can_score result differs")
        novel.append({"proposal_id": proposal.get("proposal_id"), "text": text,
                      "segments": deepcopy(segments), "edge_ids": list(edge_ids),
                      "features": features,
                      "status": "ready" if supported else "unsupported",
                      "reason": None if supported else "sentence_ranker_can_score_false",
                      "sentence_score": None})
        seen.add(text)
    require(referenced == set(inventory), "used edge inventory has unreferenced edges")
    if not ranked:
        require(not proposals and not inventory
                and census.get("fallback_public") == public,
                "no-pool fallback or proposals differ")
    else:
        require(census.get("fallback_public") is None,
                "ranked row unexpectedly has a fallback")
    return {"schema": PREPARED_SCHEMA, "id": identifier,
            "component_id": original["component_id"], "request": deepcopy(request),
            "ranked": ranked, "source_reference": deepcopy(reference),
            "original": originals, "novel": novel,
            "fallback_public": deepcopy(public) if not ranked else None,
            "score_texts": [row["text"] for row in novel if row["status"] == "ready"]}


def score_prepared(prepared, ranker, *, verify_original_parity=False,
                   parity_tolerance=PARITY_TOLERANCE):
    """Score supported novel texts and optionally rescore originals as a probe."""
    require(isinstance(prepared, dict) and prepared.get("schema") == PREPARED_SCHEMA
            and type(verify_original_parity) is bool
            and finite(parity_tolerance) and parity_tolerance == PARITY_TOLERANCE,
            "prepared scoring contract or parity tolerance differs")
    result = deepcopy(prepared)
    texts = result.get("score_texts")
    require(isinstance(texts, list)
            and texts == [row["text"] for row in result.get("novel", [])
                          if row.get("status") == "ready"],
            "prepared score text order differs")
    values = list(ranker.score_pairs([""] * len(texts), texts)) if texts else []
    require(len(values) == len(texts) and all(finite(value) for value in values),
            "sentence ranker novel scores differ")
    by_text = dict(zip(texts, map(float, values)))
    for row in result["novel"]:
        if row["status"] == "ready":
            row["status"] = "scored"; row["sentence_score"] = by_text[row["text"]]
    parity = None
    if verify_original_parity:
        original_texts = [row["text"] for row in result["original"]]
        require(all(ranker.can_score(text) for text in original_texts),
                "original candidate is unsupported by frozen sentence ranker")
        rescored = (list(ranker.score_pairs([""] * len(original_texts), original_texts))
                    if original_texts else [])
        require(len(rescored) == len(original_texts) and all(finite(value) for value in rescored),
                "sentence ranker original scores differ")
        differences = [abs(float(value) - row["sentence_score"])
                       for value, row in zip(rescored, result["original"])]
        require(all(value <= PARITY_TOLERANCE for value in differences),
                "frozen sentence score parity differs")
        parity = {"checked": True, "tolerance": PARITY_TOLERANCE,
                  "max_absolute_difference": max(differences, default=0.0),
                  "rows": len(differences)}
    result["schema"] = SCORED_SCHEMA
    result["original_score_parity"] = parity
    return result


def score_row(original, census, ranker, *, check_original_parity=False):
    """Prepare and score one authenticated row without reading external state."""
    return score_prepared(prepare_row(original, census, ranker), ranker,
                          verify_original_parity=check_original_parity)


__all__ = ["PARITY_TOLERANCE", "PREPARED_SCHEMA", "SCORED_SCHEMA",
           "canonical_sha256", "prepare_row", "score_prepared", "score_row"]
