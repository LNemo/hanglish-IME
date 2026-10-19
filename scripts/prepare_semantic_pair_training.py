"""Prepare immutable, role-free semantic pair inputs from frozen TRAIN FIT data."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.final_choice_policy import finite_candidate_indices, proposal_allowed
from scripts.probe_symmetric_pair_selector import RECIPE as FOLD_RECIPE, assign_folds
from scripts.semantic_pair_contract import canonical_candidates, ordered_views
from scripts.train_final_choice_selector import load_fit_role, validate_fit_pair


NATURAL_FIT = ROOT / "data/natural_context/final-choice-v1/natural_fit"
CHALLENGE_FIT = ROOT / "data/natural_context/final-choice-v1/challenge_fit"
NATURAL_MANIFEST_SHA256 = "fc22b1635cc496ed70aa757750944726d583d640f2d9f3a598e380f3e262a1c9"
CHALLENGE_MANIFEST_SHA256 = "2b8a6349e0ad091e797d86694c2a5165b81fa906a91c3d540ece9042b1b66194"
TOKENIZER_DIR = ROOT / "data/external/klue-roberta-small"
OUTPUT = ROOT / ".omx/artifacts/semantic-pair-training-input-v1"
EXPECTED_ROWS = {"natural_fit": 2048, "challenge_fit": 1444}
MAX_TOKENS = 512
FOLD_COUNT = 4
FOLD_SALT = "symmetric-pair-fit-v1"
TOKENIZER_FILES = (
    "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt",
)
DEPENDENCIES = (
    "hanglish/core/phonology.py",
    "scripts/final_choice_policy.py",
    "scripts/probe_symmetric_pair_selector.py",
    "scripts/semantic_pair_contract.py",
    "scripts/train_final_choice_selector.py",
)


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def file_digest(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def path_key(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def tokenize_views(tokenizer, views):
    lengths = []
    for view in views:
        if not isinstance(view, str):
            raise TypeError("neural views must be UTF-8 text")
        token_ids = tokenizer.encode(
            view, add_special_tokens=True, truncation=False
        )
        if (not isinstance(token_ids, (list, tuple))
                or any(type(token) is not int for token in token_ids)):
            raise ValueError("tokenizer must return an integer token sequence")
        lengths.append(len(token_ids))
    return lengths


def build_pair_envelope(row_index, row_meta, current_index, alternative_index,
                        current, alternative, tokenizer=None):
    current_text = current.get("text") if isinstance(current, dict) else None
    alternative_text = alternative.get("text") if isinstance(alternative, dict) else None
    candidate_a, candidate_b, current_was_swapped = canonical_candidates(
        current_text, alternative_text
    )
    view_bytes = ordered_views(
        row_meta["input"], row_meta["context"], candidate_a, candidate_b
    )
    views = [value.decode("utf-8") for value in view_bytes]
    envelope = {
        "row_index": row_index,
        "row_id": row_meta["id"],
        "fold": row_meta["fold"],
        "component": row_meta["component"],
        "current_source_index": current_index,
        "alternative_source_index": alternative_index,
        "current": deepcopy(current),
        "alternative": deepcopy(alternative),
        "canonical_orientation": {
            "candidate_a_text": candidate_a,
            "candidate_b_text": candidate_b,
            "current_is_a": not current_was_swapped,
        },
        "neural_views": views,
    }
    if tokenizer is not None:
        lengths = tokenize_views(tokenizer, views)
        envelope["token_lengths"] = lengths
        envelope["eligible_le_512"] = all(length <= MAX_TOKENS
                                            for length in lengths)
    return envelope


def _source_row(row, role, index, fold):
    finite_indices = finite_candidate_indices(row["candidates"])
    return {
        "index": index,
        "id": row["id"],
        "group_id": row["group_id"],
        "source_role": role,
        "input": row["input"],
        "context": row["context"],
        "current_result": row["result"],
        "finite_candidate_indices": finite_indices,
        "finite_candidates": [
            {"source_index": source_index,
             "object": deepcopy(row["candidates"][source_index])}
            for source_index in finite_indices
        ],
        "fold": fold["fold"],
        "component": fold["component"],
        "pair_indices": [],
    }


def build_records(natural_rows, challenge_rows, tokenizer):
    tagged = [("natural_fit", row) for row in natural_rows]
    tagged.extend(("challenge_fit", row) for row in challenge_rows)
    fold_inputs = [
        {"id": row["id"], "group_id": row["group_id"],
         "input": row["input"], "context": row["context"]}
        for _, row in tagged
    ]
    folds = assign_folds(fold_inputs)
    rows, pairs = [], []
    for index, ((role, source), fold) in enumerate(zip(tagged, folds)):
        row = _source_row(source, role, index, fold)
        rows.append(row)
        finite = row["finite_candidate_indices"]
        if 0 not in finite:
            continue
        current = source["candidates"][0]
        for alternative_index in finite:
            if alternative_index == 0:
                continue
            alternative = source["candidates"][alternative_index]
            if not proposal_allowed(source["input"], current, alternative):
                continue
            pair = build_pair_envelope(
                index, row, 0, alternative_index, current, alternative, tokenizer
            )
            pair["index"] = len(pairs)
            pairs.append(pair)
            row["pair_indices"].append(pair["index"])
    if len(rows) != len(tagged):
        raise RuntimeError("all FIT rows must remain in the denominator")
    return rows, pairs


def _write_jsonl(path, values):
    raw = b"".join(canonical(value) + b"\n" for value in values)
    with path.open("xb") as stream:
        stream.write(raw)
    return digest(raw)


def source_pins(natural, challenge):
    pins = {
        path_key(NATURAL_FIT / "manifest.json"): digest(natural["manifest_raw"]),
        path_key(NATURAL_FIT / "cases.jsonl"): digest(natural["cases_raw"]),
        path_key(NATURAL_FIT / "selection.jsonl"): digest(natural["selection_raw"]),
        path_key(CHALLENGE_FIT / "manifest.json"): digest(challenge["manifest_raw"]),
        path_key(CHALLENGE_FIT / "cases.jsonl"): digest(challenge["cases_raw"]),
        path_key(CHALLENGE_FIT / "selection.jsonl"): digest(challenge["selection_raw"]),
    }
    paths = [ROOT / name for name in DEPENDENCIES]
    paths.extend(TOKENIZER_DIR / name for name in TOKENIZER_FILES)
    paths.append(Path(__file__))
    pins.update({path_key(path): file_digest(path) for path in paths})
    return dict(sorted(pins.items()))


def verify_source_pins(pins):
    for name, expected in pins.items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        if not path.is_file() or file_digest(path) != expected:
            raise ValueError(f"semantic pair preparation source changed: {name}")


def load_local_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(
        TOKENIZER_DIR, local_files_only=True
    )


def prepare(output=OUTPUT, tokenizer_loader=load_local_tokenizer):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    natural = load_fit_role(
        NATURAL_FIT, NATURAL_MANIFEST_SHA256, "natural_fit"
    )
    challenge = load_fit_role(
        CHALLENGE_FIT, CHALLENGE_MANIFEST_SHA256, "challenge_fit"
    )
    validate_fit_pair(natural, challenge)
    if (FOLD_RECIPE.get("folds") != FOLD_COUNT
            or FOLD_RECIPE.get("fold_salt") != FOLD_SALT):
        raise ValueError("frozen four-fold assignment recipe changed")
    if (len(natural["rows"]) != EXPECTED_ROWS["natural_fit"]
            or len(challenge["rows"]) != EXPECTED_ROWS["challenge_fit"]):
        raise ValueError("complete TRAIN FIT captures are required")
    pins_before = source_pins(natural, challenge)
    tokenizer = tokenizer_loader()
    rows, pairs = build_records(natural["rows"], challenge["rows"], tokenizer)
    if len(rows) != sum(EXPECTED_ROWS.values()):
        raise RuntimeError("complete 3492-row denominator is required")
    eligible_per_row = [
        sum(pairs[index]["eligible_le_512"] for index in row["pair_indices"])
        for row in rows
    ]
    pair_lengths = [
        {"pair_index": pair["index"], "ab": pair["token_lengths"][0],
         "ba": pair["token_lengths"][1],
         "eligible_le_512": pair["eligible_le_512"]}
        for pair in pairs
    ]
    verify_source_pins(pins_before)
    output.mkdir()
    rows_sha = _write_jsonl(output / "rows.jsonl", rows)
    pairs_sha = _write_jsonl(output / "pairs.jsonl", pairs)
    manifest = {
        "schema": "semantic-pair-training-input-v1",
        "completed": True,
        "scope": "TRAIN-only source input qualification; no model training or evaluation",
        "rows": len(rows),
        "role_rows": dict(EXPECTED_ROWS),
        "pairs": len(pairs),
        "rows_with_pairs": sum(bool(row["pair_indices"]) for row in rows),
        "eligible_pairs_le_512": sum(pair["eligible_le_512"] for pair in pairs),
        "eligible_per_row": eligible_per_row,
        "pair_token_lengths": pair_lengths,
        "rows_sha256": rows_sha,
        "pairs_sha256": pairs_sha,
        "source_hashes": pins_before,
        "config": {
            "folds": FOLD_COUNT,
            "fold_salt": FOLD_SALT,
            "fold_assignment": "scripts.probe_symmetric_pair_selector.assign_folds",
            "max_tokens": MAX_TOKENS,
            "tokenizer": path_key(TOKENIZER_DIR),
            "tokenizer_files": list(TOKENIZER_FILES),
            "candidate_policy": "scripts.final_choice_policy.proposal_allowed",
        },
        "all_rows_retained_including_rows_without_pairs": True,
        "labels_or_expected_saved": False,
        "neural_views_are_role_free": True,
        "tokenizer_truncation": False,
        "encoder_or_model_weights_loaded": False,
        "training_run": False,
        "evaluation_outcomes_computed": False,
        "development_or_reserve_data_read": False,
    }
    with (output / "manifest.json").open("xb") as stream:
        stream.write(canonical(manifest) + b"\n")
    verify_source_pins(pins_before)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    prepare(args.output)


if __name__ == "__main__":
    main()
