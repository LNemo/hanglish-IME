"""Source-only admissibility for new lexical-boundary sentence paths."""
from __future__ import annotations

import re

from hanglish.core.phonology import romanize_word
from scripts.final_choice_policy import proposal_allowed

LOWER_ROMAN = re.compile(r"[a-z]+\Z")
HANGUL = re.compile(r"[가-힣]+\Z")


def _direct_provenance(record, source, text):
    if not isinstance(record, dict):
        return False
    if record.get("source") != source or record.get("text") != text:
        return False
    left = record.get("left_lexeme")
    right = record.get("right_lexeme")
    left_key = record.get("left_key")
    right_key = record.get("right_key")
    boundary = record.get("boundary")
    suffix = record.get("suffix")
    if (not isinstance(left, str) or HANGUL.fullmatch(left) is None
            or not isinstance(right, str) or HANGUL.fullmatch(right) is None
            or not isinstance(left_key, str) or LOWER_ROMAN.fullmatch(left_key) is None
            or not isinstance(right_key, str) or LOWER_ROMAN.fullmatch(right_key) is None
            or romanize_word(left) != left_key
            or romanize_word(right) != right_key
            or boundary not in ("changed", "unchanged")
            or suffix is not None and (
                not isinstance(suffix, str) or HANGUL.fullmatch(suffix) is None
            )):
        return False
    if text != left + right + (suffix or "") or romanize_word(text) != source:
        return False

    isolated = left_key + right_key
    if suffix is None:
        surface_changed = source != isolated
    else:
        suffix_surface = record.get("suffix_surface")
        restored = record.get("suffix_restored")
        stem_surface = romanize_word(left + right)
        if (not isinstance(suffix_surface, str) or not suffix_surface
                or not isinstance(restored, str)
                or not source.endswith(suffix_surface)
                or source[:-len(suffix_surface)] + restored != stem_surface):
            return False
        surface_changed = stem_surface != isolated
    return boundary == ("changed" if surface_changed else "unchanged")


def boundary_proposal_allowed(source, parent_candidate, extra_path_record,
                              derived_options):
    """Allow only directly derived changes from the exact parent candidate.

    The function consumes source-derived candidate records only.  It does not
    accept row identities, expected spellings, labels, or learned cutoffs.
    """
    if (not isinstance(source, str) or not isinstance(parent_candidate, dict)
            or not isinstance(extra_path_record, dict)
            or not isinstance(derived_options, list)):
        return False
    if not proposal_allowed(source, parent_candidate, extra_path_record):
        return False

    parent_segments = parent_candidate.get("segments")
    extra_segments = extra_path_record.get("segments")
    if not isinstance(parent_segments, list) or not isinstance(extra_segments, list):
        return False
    for parent, extra in zip(parent_segments, extra_segments):
        if parent.get("text") == extra.get("text"):
            continue
        # A recognized whole lexeme is stronger evidence than a newly
        # composed split, even when both reproduce the same romanization.
        if parent.get("kind") == "lexicon":
            return False
        if not any(_direct_provenance(
                record, extra.get("source"), extra.get("text"))
                for record in derived_options):
            return False
    return True
