"""Measure information loss in closed FIT candidate pools; never change ranking."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def summarize(candidates, encodings, gold, winner):
    """Count exact token-ID collisions separately from unknown-token presence."""
    require(len(candidates) == len(encodings), "tokenization count differs")
    groups = defaultdict(list)
    counts = Counter(candidates=0, candidates_with_unknown=0, gold_candidates=0,
                     gold_with_unknown=0, collision_groups=0, mixed_gold_groups=0,
                     mixed_gold_candidates=0, winner_unknown=0,
                     wrong_winner_collides_with_gold=0)
    examples = []
    for candidate, encoded in zip(candidates, encodings):
        text = candidate["text"]
        unknown = "[UNK]" in encoded.tokens
        positive = text in gold
        counts["candidates"] += 1
        counts["candidates_with_unknown"] += unknown
        counts["gold_candidates"] += positive
        counts["gold_with_unknown"] += positive and unknown
        counts["winner_unknown"] += text == winner and unknown
        groups[tuple(encoded.ids)].append((text, positive, unknown))
    for values in groups.values():
        if len(values) < 2:
            continue
        counts["collision_groups"] += 1
        positives = [text for text, positive, _ in values if positive]
        negatives = [text for text, positive, _ in values if not positive]
        if positives and negatives:
            counts["mixed_gold_groups"] += 1
            counts["mixed_gold_candidates"] += len(values)
            counts["wrong_winner_collides_with_gold"] += winner in negatives
            if len(examples) < 3:
                examples.append({"gold": positives[:2], "other": negatives[:2],
                                 "has_unknown": any(v[2] for v in values)})
    return {"counts": dict(counts), "examples": examples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    selection = Path(".omx/artifacts/current-fit-recombination-selection-v1")
    intake_path = Path(".omx/artifacts/current-fit-recombination-selection-intake-v1.json")
    label_path = Path(".omx/artifacts/current-phonetic-train-features-v1/rows.jsonl")
    tokenizer_path = Path("checkpoints/sentence_ranker/tokenizer.json")
    proof = json.loads((selection / "independent-verification.json").read_text())
    report = json.loads((selection / "report.json").read_text())
    require(proof["verified"] and digest(selection / "report.json") == proof["report_sha256"],
            "selection proof differs")
    require(digest(intake_path) == report["intake_sha256"]
            and digest(label_path) == report["features_sha256"]
            and digest(selection / "outcomes.jsonl") == proof["outcomes_sha256"],
            "selection inputs differ")
    intake = json.loads(intake_path.read_text())
    for shard in intake["shards"]:
        directory = Path(shard["directory"])
        for name, expected in shard["hashes"].items():
            require(digest(directory / name) == expected, "closed shard differs")
        require(json.loads((directory / "independent-verification.json").read_text())["verified"],
                "unverified scoring shard")
    from tokenizers import Tokenizer

    tokenizer_hash = digest(tokenizer_path)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    # Token IDs of these complete normalized strings are the sentence ranker's
    # sequence A. Empty external context and special tokens are identical.
    # Saved tokenizer settings reflect its last model call. Match can_score's
    # explicit truncation=False and do not retain per-batch padding in ID keys.
    tokenizer.no_truncation()
    tokenizer.no_padding()
    labels, winners = {}, {}
    for line in label_path.open():
        row = json.loads(line)
        require(row["id"] not in labels, "duplicate label ID")
        labels[row["id"]] = (row["input"], set(row["expected"]))
    for line in (selection / "outcomes.jsonl").open():
        row = json.loads(line)
        require(row["id"] not in winners, "duplicate outcome ID")
        winners[row["id"]] = {arm: value["public"][0]["text"]
                              for arm, value in row["arms"].items()}
    args.output.mkdir(parents=True, exist_ok=False)
    totals = {arm: Counter() for arm in ("original", "novel", "expanded")}
    seen, no_pool = set(), 0
    max_rss = 0
    with (args.output / "rows.jsonl").open("x") as destination:
        for shard in intake["shards"]:
            capture = Path(shard["directory"]) / "capture.jsonl"
            for line in capture.open():
                row = json.loads(line); scored = row["scoring"]; identity = row["id"]
                require(identity not in seen and identity in labels and identity in winners,
                        "candidate ID differs")
                require(row["request"]["input"] == labels[identity][0], "source differs")
                seen.add(identity)
                original, novel = scored["original"], scored["novel"]
                candidates = original + novel
                encodings = tokenizer.encode_batch(
                    [" ".join(c["text"].strip().split()) for c in candidates],
                    add_special_tokens=False) if candidates else []
                result = {"id": identity, "ranked": scored["ranked"], "arms": {}}
                no_pool += not scored["ranked"]
                split = len(original)
                for arm, cs, es, chosen in (
                    ("original", original, encodings[:split], "old_calibrated"),
                    ("novel", novel, encodings[split:], "expanded_calibrated"),
                    ("expanded", candidates, encodings, "expanded_calibrated"),
                ):
                    value = summarize(cs, es, labels[identity][1], winners[identity][chosen])
                    result["arms"][arm] = value
                    totals[arm].update(value["counts"])
                    totals[arm]["rows_with_mixed_gold_collision"] += bool(value["counts"]["mixed_gold_groups"])
                destination.write(json.dumps(result, ensure_ascii=False) + "\n")
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                rss *= 1 if sys.platform == "darwin" else 1024
                max_rss = max(max_rss, rss)
                require(rss <= 1024**3 and time.monotonic() - started <= 300,
                        "audit exceeded 1GiB/300s bound")
            require(digest(capture) == shard["hashes"]["capture.jsonl"], "capture changed")
            print(json.dumps({"shard": shard["shard"], "rows": len(seen)}), flush=True)
    require(seen == set(labels) == set(winners) and len(seen) == report["rows"],
            "FIT denominator differs")
    require(digest(tokenizer_path) == tokenizer_hash, "tokenizer changed")
    output = {"completed": True, "rows": len(seen), "no_pool_rows": no_pool,
              "counts": {k: dict(v) for k, v in totals.items()},
              "elapsed_seconds": time.monotonic() - started, "max_rss_bytes": max_rss,
              "rows_sha256": digest(args.output / "rows.jsonl"),
              "tokenizer_sha256": tokenizer_hash,
              "selection_report_sha256": proof["report_sha256"],
              "intake_sha256": report["intake_sha256"],
              "script_sha256": digest(__file__),
              "limits": {"seconds": 300, "rss_bytes": 1024**3},
              "scope": "All FIT candidates including unsupported; exact normalized sequence-A IDs. No neural scores or ranking changed; UNK is not an error label.",
              "production_changed": False, "dev_opened": False, "test_opened": False}
    (args.output / "report.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
