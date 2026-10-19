"""Source-only candidate sets for a TRAIN diagnostic with an explicit NONE class."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from scripts.listwise_semantic_contract import serialize_candidate
from scripts.probe_role_free_semantic_head import load_inputs, verify_model_lineage
from scripts.probe_symmetric_pair_selector import canonical, sha


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / ".omx/artifacts/listwise-semantic-input-v1"
MODEL = ROOT / "data/external/klue-roberta-small"
MAX_TOKENS = 512


def candidate_set(row, pairs):
    """Preserve the current object and every source-policy-admitted alternative."""
    finite = {entry["source_index"]: entry["object"]
              for entry in row["finite_candidates"]}
    if 0 not in finite or finite[0]["text"] != row["current_result"]:
        raise ValueError("current candidate is missing")
    indices = [0]
    for index in row["pair_indices"]:
        pair = pairs[index]
        alternative = pair["alternative_source_index"]
        if (pair["row_index"] != row["index"] or pair["row_id"] != row["id"]
                or pair["current"] != finite[0]
                or pair["alternative"] != finite.get(alternative)
                or alternative in indices):
            raise ValueError("candidate pair source differs")
        indices.append(alternative)
    candidates = [finite[index] for index in indices]
    if len({candidate["text"] for candidate in candidates}) != len(candidates):
        raise ValueError("duplicate candidate strings need an explicit source mapping")
    return indices, candidates


def prepare_rows(rows, pairs, tokenizer):
    result = []
    for row in rows:
        source_indices, candidates = candidate_set(row, pairs)
        views = [serialize_candidate(row["input"], row["context"], candidate["text"])
                 for candidate in candidates]
        tokens = tokenizer(views, truncation=False, padding=False,
                           add_special_tokens=True)
        lengths = [len(value) for value in tokens["input_ids"]]
        if len(lengths) != len(candidates):
            raise ValueError("tokenizer omitted a candidate")
        # A long candidate never silently disappears from its competition set.
        eligible = all(0 < length <= MAX_TOKENS for length in lengths)
        result.append({
            **{key: row[key] for key in ("index", "id", "source_role", "group_id",
                                       "component", "fold", "input", "context",
                                       "current_result")},
            "current_index": 0, "source_indices": source_indices,
            "candidates": candidates, "neural_views": views,
            "token_lengths": lengths, "eligible": eligible,
        })
    return result


def prepare(output=OUTPUT):
    from transformers import AutoTokenizer

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    manifest, source_rows, pairs, pins = load_inputs()
    source = verify_model_lineage(pins)
    for relative in ("scripts/prepare_listwise_semantic_training.py",
                     "scripts/listwise_semantic_contract.py"):
        pins[relative] = sha(ROOT / relative)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False)
    rows = prepare_rows(source_rows, pairs, tokenizer)
    payload = b"".join(canonical(row) + b"\n" for row in rows)
    summary = {
        "schema": "listwise-semantic-training-input-v1", "completed": True,
        "rows": len(rows), "eligible_rows": sum(row["eligible"] for row in rows),
        "candidates": sum(len(row["candidates"]) for row in rows),
        "eligible_candidates": sum(len(row["candidates"]) for row in rows if row["eligible"]),
        "candidate_count_histogram": dict(Counter(len(row["candidates"]) for row in rows)),
        "maximum_tokens": max(max(row["token_lengths"]) for row in rows),
        "max_tokens": MAX_TOKENS, "long_candidate_policy": "bypass entire row",
        "labels_or_expected_saved": False, "new_targets_read": False,
        "candidate_selection_uses_labels": False,
        "source_validation_reads_previously_exposed_train_labels": True,
        "folds_reused_from_group_closed_train_partition": True,
        "neural_model_loaded": False, "source_hashes": dict(sorted(pins.items())),
        "base_model_source": source,
    }
    for name, digest in pins.items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"source changed: {name}")
    output.mkdir(parents=True)
    (output / "rows.jsonl").write_bytes(payload)
    summary["rows_sha256"] = sha(output / "rows.jsonl")
    (output / "manifest.json").write_bytes(canonical(summary) + b"\n")
    return summary


if __name__ == "__main__":
    result = prepare()
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"source_hashes", "base_model_source"}}, ensure_ascii=False))
