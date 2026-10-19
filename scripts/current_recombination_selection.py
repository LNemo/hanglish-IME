"""Pure four-arm selection for a closed recombination scoring row."""
from __future__ import annotations

from copy import deepcopy
import math


ARMS = ("old_sentence", "expanded_sentence", "old_calibrated",
        "expanded_calibrated")


def require(value, message):
    if not value:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_model(model):
    require(isinstance(model, dict), "calibration model must be an object")
    parameters, scales = model.get("parameters"), model.get("scale")
    require(isinstance(parameters, list) and len(parameters) == 5
            and isinstance(scales, list) and len(scales) == 4
            and all(finite(value) for value in parameters + scales)
            and parameters[0] >= 0 and all(value > 0 for value in scales),
            "invalid calibration model")
    return tuple(map(float, parameters)), tuple(map(float, scales))


def calibrated_score(row, model):
    parameters, scales = validate_model(model)
    base, features = row.get("sentence_score"), row.get("features")
    require(finite(base) and isinstance(features, list) and len(features) == 4
            and all(finite(value) for value in features), "candidate score row differs")
    value = parameters[0] * base + sum(
        weight * (feature / scale)
        for weight, feature, scale in zip(parameters[1:], features, scales))
    require(math.isfinite(value), "calibrated score is not finite")
    return value


def _candidate(row, status):
    require(isinstance(row, dict) and isinstance(row.get("text"), str) and row["text"]
            and isinstance(row.get("segments"), list)
            and isinstance(row.get("features"), list) and len(row["features"]) == 4
            and all(finite(value) for value in row["features"])
            and row.get("status") == status, "candidate row differs")
    if status in ("preserved", "scored"):
        require(finite(row.get("sentence_score")), "candidate sentence score differs")
    return row


def render_public(source, pool):
    """Render NeuralConverter's ranked top five with its reserved raw slot."""
    require(isinstance(source, str) and source and isinstance(pool, list) and pool,
            "ranked public pool differs")
    ordered = sorted(pool, key=lambda row: (-row["score"], row["text"]))
    require(len({row["text"] for row in ordered}) == len(ordered),
            "ranked pool contains duplicate texts")
    public = [{"text": row["text"], "score": round(row["score"], 4),
               "segments": deepcopy(row["segments"])} for row in ordered[:5]]
    if source not in {row["text"] for row in public}:
        public = public[:4] + [{"text": source, "score": None,
                               "segments": [{"input_start": 0, "input_end": len(source),
                                             "source": source, "text": source, "kind": "raw"}]}]
    return public


def _outcome(public, pool_texts, expected):
    texts = [row["text"] for row in public]
    return {"first": texts[0] in expected, "top5": bool(expected.intersection(texts)),
            "public": public, "pool_coverage": bool(expected.intersection(pool_texts))}


def _transition(before, after):
    return {"corrections": int(not before["first"] and after["first"]),
            "regressions": int(before["first"] and not after["first"]),
            "top5_gains": int(not before["top5"] and after["top5"]),
            "top5_losses": int(before["top5"] and not after["top5"])}


def compare_row(scored, expected, calibration_model):
    """Evaluate four selection arms after expected strings are supplied late."""
    require(isinstance(scored, dict)
            and scored.get("schema") == "current-recombination-scoring-row-v1",
            "scored row contract differs")
    request = scored.get("request")
    source = request.get("input") if isinstance(request, dict) else None
    require(request == {"input": source, "context": ""}
            and isinstance(source, str) and source
            and isinstance(expected, list) and expected
            and all(isinstance(text, str) and text for text in expected),
            "evaluation input differs")
    validate_model(calibration_model)
    original, novel = scored.get("original"), scored.get("novel")
    fallback, ranked = scored.get("fallback_public"), scored.get("ranked")
    require(isinstance(original, list) and isinstance(novel, list) and type(ranked) is bool,
            "scored candidate domain differs")
    originals = [_candidate(row, "preserved") for row in original]
    scored_novel, unsupported, reasons = [], [], {}
    for row in novel:
        require(isinstance(row, dict), "novel candidate row differs")
        status = row.get("status")
        if status == "scored":
            scored_novel.append(_candidate(row, "scored"))
        elif status == "unsupported":
            _candidate(row, "unsupported")
            require(row.get("sentence_score") is None
                    and isinstance(row.get("reason"), str) and row["reason"],
                    "unsupported candidate differs")
            unsupported.append(row); reasons[row["reason"]] = reasons.get(row["reason"], 0) + 1
        else:
            raise ValueError("novel candidate status differs")
    all_texts = [row["text"] for row in originals + scored_novel + unsupported]
    require(len(all_texts) == len(set(all_texts)), "candidate texts are duplicated")
    counts = {"original_candidates": len(originals), "novel_candidates": len(novel),
              "scored_novel": len(scored_novel), "unsupported_novel": len(unsupported),
              "unsupported_reasons": reasons}
    gold = set(expected)
    if not ranked:
        require(not originals and not novel and isinstance(fallback, list) and fallback,
                "no-pool fallback differs")
        for candidate in fallback:
            require(isinstance(candidate, dict) and isinstance(candidate.get("text"), str)
                    and candidate["text"] and isinstance(candidate.get("segments"), list),
                    "fallback candidate differs")
        public = deepcopy(fallback); arms = {
            name: _outcome(deepcopy(public), set(), gold) for name in ARMS}
    else:
        require(originals and fallback is None, "ranked scored row differs")
        old_sentence = [{"text": row["text"], "score": float(row["sentence_score"]),
                         "segments": row["segments"]} for row in originals]
        novel_sentence = [{"text": row["text"], "score": float(row["sentence_score"]),
                           "segments": row["segments"]} for row in scored_novel]
        old_calibrated = [{"text": row["text"], "score": calibrated_score(row, calibration_model),
                           "segments": row["segments"]} for row in originals]
        novel_calibrated = [{"text": row["text"], "score": calibrated_score(row, calibration_model),
                             "segments": row["segments"]} for row in scored_novel]
        pools = {"old_sentence": old_sentence,
                 "expanded_sentence": old_sentence + novel_sentence,
                 "old_calibrated": old_calibrated,
                 "expanded_calibrated": old_calibrated + novel_calibrated}
        arms = {name: _outcome(render_public(source, pool),
                               {row["text"] for row in pool}, gold)
                for name, pool in pools.items()}
    transitions = {
        "candidate_expansion_sentence": _transition(arms["old_sentence"], arms["expanded_sentence"]),
        "calibration_old_pool": _transition(arms["old_sentence"], arms["old_calibrated"]),
        "calibration_expanded_pool": _transition(arms["expanded_sentence"], arms["expanded_calibrated"]),
        "candidate_expansion_calibrated": _transition(arms["old_calibrated"], arms["expanded_calibrated"]),
        "overall": _transition(arms["old_sentence"], arms["expanded_calibrated"]),
    }
    return {"id": scored.get("id"), "request": deepcopy(request), "arms": arms,
            "transitions": transitions, "counts": counts,
            "limitations": {"source_preservation": "exact raw source fallback only",
                            "english_semantics_evaluated": False}}


__all__ = ["ARMS", "calibrated_score", "compare_row", "render_public", "validate_model"]
