"""Experiment-only union of parent and preserved-student word proposals."""
from __future__ import annotations

from hanglish.core.neural_converter import beam_paths


def _unique_paths(*collections):
    unique = {}
    for paths in collections:
        for path in paths:
            unique.setdefault(path[0], path)
    return list(unique.values())


def propose(converter, student, text, context=""):
    """Build a source-only parent-prefix union without changing runtime state."""
    parent_pieces, parent_static, *_ = converter._prepare_candidates(
        text, context, learned=True, raw=False
    )
    parent_word_paths = beam_paths(parent_pieces)
    parent_static_paths = beam_paths(parent_static)
    parent_paths = _unique_paths(parent_word_paths, parent_static_paths)
    if len(parent_paths) > 48:
        raise ValueError("parent proposal pool exceeds runtime bound")

    sentence_ranker = converter.sentence_ranker
    sentence_eligible = (
        len(text.split()) > 1
        and len(parent_word_paths) > 1
        and sentence_ranker is not None
        and all(sentence_ranker.can_score(path[0]) for path in parent_paths)
    )
    if not sentence_eligible:
        return {
            "parent_word_paths": parent_word_paths,
            "parent_paths": parent_paths,
            "union_paths": parent_paths,
            "sentence_eligible": False,
            "additions": [],
            "filtered_long": [],
        }

    parent_ranker = converter.ranker
    try:
        converter.ranker = student
        student_pieces, student_static, *_ = converter._prepare_candidates(
            text, context, learned=True, raw=False
        )
    finally:
        converter.ranker = parent_ranker
    if student_static != parent_static:
        raise ValueError("student changed the source-only static lattice")

    known = {path[0] for path in parent_paths}
    additions = []
    filtered_long = []
    filtered_texts = set()
    for path in beam_paths(student_pieces):
        candidate = path[0]
        if candidate in known:
            continue
        known.add(candidate)
        if not sentence_ranker.can_score(candidate):
            if candidate not in filtered_texts:
                filtered_texts.add(candidate)
                filtered_long.append(path)
            continue
        additions.append(path)
    union_paths = parent_paths + additions
    if len(union_paths) > 72:
        raise ValueError("combined proposal pool exceeds experiment bound")
    return {
        "parent_word_paths": parent_word_paths,
        "parent_paths": parent_paths,
        "union_paths": union_paths,
        "sentence_eligible": True,
        "additions": additions,
        "filtered_long": filtered_long,
    }


def choose(proposal, sentence_ranker, context=""):
    """Apply the experiment's exact sentence selection or parent fallback."""
    if not proposal["sentence_eligible"]:
        return {"paths": proposal["parent_word_paths"],
                "sentence_ranking_applied": False}
    paths = proposal["union_paths"]
    scores = sentence_ranker.score_pairs([context] * len(paths), [path[0] for path in paths])
    if len(scores) != len(paths):
        raise ValueError("sentence ranker returned an invalid score count")
    ranked = sorted(zip(paths, map(float, scores)), key=lambda item: (-item[1], item[0][0]))
    scored_paths = [(path[0], score, path[2]) for path, score in ranked]
    return {"paths": scored_paths,
            "sentence_ranking_applied": True}


__all__ = ["choose", "propose"]
