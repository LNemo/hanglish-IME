#!/usr/bin/env python3
"""Train paired sentence-ranker controls on frozen TRAIN recombination pools."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
from types import SimpleNamespace
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import score_recombination_training_pools as base


SCHEMA = "recombination-fresh-paired-training-v1"
CENSUS_SCHEMA = "recombination-fresh-training-parent-census-v1"
CENSUS_PROOF_SCHEMA = "fresh-recombination-census-independent-verification-v1"
EXPECTED_CENSUS_HASHES = {
    "report.json": "2d1dd39b8645fc9f7a75e4e7e95b53f3b67aa01a3754ec2498002e91e649fe49",
    "capture.jsonl": "847a8d6603dbd66ed87d2f9863b4e20581989887da76679b1a5586bb6d552154",
    "capture.complete.json": "934b832bccf7b89c6f5a9700d2b3468dcf886caeb92be5a5a9ee0f88f5a06dcb",
    "independent-verification.json": "a744c8324fca8c2a1ae170fb9e51aa0ce06f9e0a6310446aecbe93b11e5deb29"
}
EXPECTED_ROWS = 2_048
FIT_ROWS = 1_642
HOLD_ROWS = 406
SEED = 20_260_921
MAX_CANDIDATES = 8
ACCUMULATION = 4
MAX_LENGTH = 256
PLAN = ROOT / ".omx/plans/recombination-fresh-paired-training-v1.md"
CENSUS_VERIFIER = ROOT / ".omx/artifacts/verify-fresh-recombination-census-v1.py"
EXPECTED_CENSUS_VERIFIER_SHA256 = (
    "8b16c25b0658d593c795a3a766eb6fdb262b5cbffed7866facd4f414c3e04de0")
LIMITS = {"whole_seconds": 3_600.0, "update_seconds": 120.0,
          "rss_bytes": 3 * 1024**3, "swap_growth_bytes": 512 * 1024**2}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def path_key(path: Path) -> str:
    return str(Path(path).resolve())


def iter_jsonl(path: Path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{number}") from error


def write_json(path: Path, value) -> None:
    temporary = Path(path).with_name(Path(path).name + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2,
                  allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def append_jsonl(path: Path, value) -> None:
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def literal_signature(text: str) -> str:
    """Match the predeclared uppercase/digit/punctuation diagnostic exactly."""
    return "".join(character for character in text
                   if not ("a" <= character <= "z")
                   and not ("가" <= character <= "힣")
                   and not ("ᄀ" <= character <= "ᇿ")
                   and not ("㄰" <= character <= "㆏"))


def signature_preserved(source: str, result: str) -> bool:
    return literal_signature(source) == literal_signature(result)


def score_map(census: dict) -> dict[str, float]:
    texts, scores = census.get("score_texts"), census.get("scores")
    require(isinstance(texts, list) and isinstance(scores, list)
            and len(texts) == len(scores) and len(set(texts)) == len(texts)
            and all(isinstance(text, str) and text for text in texts)
            and all(type(score) in (int, float) and math.isfinite(score)
                    for score in scores), "census score domain differs")
    return dict(zip(texts, map(float, scores)))


def selected_candidates(pool: list[str], expected: list[str], frozen: dict[str, float],
                        cap: int = MAX_CANDIDATES) -> dict:
    """Keep every pool positive, then the highest frozen-score real negatives."""
    positives = [value for value in pool if value in set(expected)]
    negatives = [value for value in pool if value not in set(expected)]
    require(len(set(pool)) == len(pool) and all(value in frozen for value in pool),
            "training pool or frozen score domain differs")
    if not positives or not negatives:
        return {"eligible": False, "reason": "missing_positive_or_negative"}
    if len(positives) >= cap:
        return {"eligible": False, "reason": "positive_cap_leaves_no_negative",
                "positive_count": len(positives)}
    negatives.sort(key=lambda value: (-frozen[value], value))
    values = positives + negatives[:cap - len(positives)]
    return {"eligible": True, "candidates": values,
            "positive_indices": [index for index, value in enumerate(values)
                                 if value in set(expected)],
            "positive_count": len(positives),
            "negative_count": len(values) - len(positives)}


def build_shared_selection(source_rows: list[dict], census_rows: list[dict],
                           fit_labels: dict[str, list[str]], seed: int = SEED) -> dict:
    require(len(source_rows) == len(census_rows), "source/census denominator differs")
    selected, exclusions = [], Counter()
    for source, census in zip(source_rows, census_rows):
        require(source["id"] == census.get("id")
                and source["input"] == census.get("input")
                and source["split"] == census.get("split")
                and source["component_id"] == census.get("component_id"),
                "source/census identity differs")
        if source["split"] != "fit":
            continue
        expected = fit_labels.get(source["id"])
        require(isinstance(expected, list) and expected, "FIT labels are incomplete")
        if not (census.get("original", {}).get("applied") is True
                and census.get("recombined", {}).get("applied") is True):
            exclusions["unsupported_pool"] += 1
            continue
        frozen = score_map(census)
        arms = {
            "a_original": selected_candidates(
                source["original_candidates"], expected, frozen),
            "b_union": selected_candidates(
                source["recombined_candidates"], expected, frozen),
        }
        failed = next((record["reason"] for record in arms.values()
                       if not record["eligible"]), None)
        if failed:
            exclusions[failed] += 1
            continue
        novel = set(source["novel_candidates"])
        b_values = arms["b_union"]["candidates"]
        b_positive = set(arms["b_union"]["positive_indices"])
        selected.append({"id": source["id"], "input": source["input"],
                         "component_id": source["component_id"], "arms": arms,
                         "selected_novel_candidates": [value for value in b_values
                                                       if value in novel],
                         "selected_novel_negatives": [value for index, value in
                             enumerate(b_values) if value in novel and index not in b_positive]})
    selected.sort(key=lambda row: row["id"])
    rng = random.Random(seed)
    rng.shuffle(selected)
    require(len({row["id"] for row in selected}) == len(selected),
            "shared selection IDs are not unique")
    return {"seed": seed, "cap": MAX_CANDIDATES,
            "gradient_accumulation": ACCUMULATION,
            "rows": selected, "row_ids": [row["id"] for row in selected],
            "excluded": dict(sorted(exclusions.items())),
            "selected_novel_candidate_count": sum(
                len(row["selected_novel_candidates"]) for row in selected),
            "selected_novel_negative_count": sum(
                len(row["selected_novel_negatives"]) for row in selected),
            "updates_per_arm": math.ceil(len(selected) / ACCUMULATION) if selected else 0}


def accumulation_groups(rows: list[dict], size: int = ACCUMULATION) -> list[list[dict]]:
    require(type(size) is int and size > 0, "accumulation size differs")
    return [rows[offset:offset + size] for offset in range(0, len(rows), size)]


def multi_positive_listwise_loss(scores, positive_indices: list[int], torch_module):
    require(scores.ndim == 1 and len(scores) > 1, "listwise scores differ")
    require(positive_indices and len(set(positive_indices)) == len(positive_indices)
            and all(type(index) is int and 0 <= index < len(scores)
                    for index in positive_indices), "positive indices differ")
    positive = torch_module.tensor(positive_indices, device=scores.device)
    return (torch_module.logsumexp(scores, 0)
            - torch_module.logsumexp(scores[positive], 0))


def metric_summary(predictions: list[dict], labels: dict[str, list[str]],
                   branch: str, policy: str) -> dict:
    counts = Counter(); outcomes = []
    for row in predictions:
        expected = labels[row["id"]]
        response = row[branch][policy]
        candidates = response["candidates"]
        top1 = response["result"] in expected
        top5 = bool(set(candidates).intersection(expected))
        literal = signature_preserved(row["input"], response["result"])
        counts.update(rows=1, top1=int(top1), top5=int(top5),
                      literal_signature_preserved=int(literal))
        outcomes.append({"id": row["id"], "top1": top1, "top5": top5,
                         "literal_signature_preserved": literal})
    return {"counts": dict(counts), "outcomes": outcomes}


def compare_metrics(candidate: dict, reference: dict) -> dict:
    candidate_by_id = {row["id"]: row for row in candidate["outcomes"]}
    reference_by_id = {row["id"]: row for row in reference["outcomes"]}
    require(candidate_by_id.keys() == reference_by_id.keys(),
            "paired metric IDs differ")
    fields = ("top1", "top5", "literal_signature_preserved")
    return {
        field + "_gain_ids": [row_id for row_id in sorted(candidate_by_id)
                               if candidate_by_id[row_id][field]
                               and not reference_by_id[row_id][field]]
        for field in fields
    } | {
        field + "_loss_ids": [row_id for row_id in sorted(candidate_by_id)
                               if reference_by_id[row_id][field]
                               and not candidate_by_id[row_id][field]]
        for field in fields
    }


def paired_gate(metrics: dict) -> dict:
    frozen_union = metrics["frozen"]["recombined"]
    arm_a_union = metrics["a_original"]["recombined"]
    arm_b_union = metrics["b_union"]["recombined"]
    frozen_original = metrics["frozen"]["original"]
    arm_b_original = metrics["b_union"]["original"]
    versus_frozen = compare_metrics(arm_b_union, frozen_union)
    versus_a = compare_metrics(arm_b_union, arm_a_union)
    original = compare_metrics(arm_b_original, frozen_original)
    b_top1 = arm_b_union["counts"]["top1"]
    frozen_top1 = frozen_union["counts"]["top1"]
    a_top1 = arm_a_union["counts"]["top1"]
    passed = (b_top1 >= frozen_top1 + 2 and b_top1 >= a_top1 + 2
              and not versus_frozen["top1_loss_ids"]
              and not versus_frozen["top5_loss_ids"]
              and not versus_frozen["literal_signature_preserved_loss_ids"]
              and not original["top1_loss_ids"])
    return {"passed": passed, "b_union_top1": b_top1,
            "frozen_union_top1": frozen_top1, "a_union_top1": a_top1,
            "minimum_net_gain": 2, "b_union_vs_frozen": versus_frozen,
            "b_union_vs_a": versus_a, "b_original_vs_frozen": original,
            "interpretation": "TRAIN-internal follow-up gate; it is not a DEV or promotion result."}


def _expected_hashes_ready() -> bool:
    return all(value != "PENDING" for value in EXPECTED_CENSUS_HASHES.values())


def validate_authorization(census_dir: Path, proof_path: Path) -> dict:
    require(_expected_hashes_ready(), "fresh census authorization hashes are pending")
    census_dir, proof_path = Path(census_dir), Path(proof_path)
    paths = {name: census_dir / name for name in EXPECTED_CENSUS_HASHES}
    require(proof_path.resolve() == paths["independent-verification.json"].resolve(),
            "census proof path differs")
    require(all(path.is_file() for path in paths.values()),
            "fresh census authorization is incomplete")
    for name, expected in EXPECTED_CENSUS_HASHES.items():
        require(sha256_file(paths[name]) == expected,
                f"fresh census artifact differs: {name}")
    report = json.loads(paths["report.json"].read_text(encoding="utf-8"))
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    closure = json.loads(paths["capture.complete.json"].read_text(encoding="utf-8"))
    gate = report.get("adequacy_gate", {})
    contracts = proof.get("contracts", {})
    require(report.get("schema") == CENSUS_SCHEMA and report.get("completed") is True
            and report.get("counts") == {"rows": EXPECTED_ROWS,
                                         "fit": FIT_ROWS, "inner_hold": HOLD_ROWS}
            and gate.get("passed") is True
            and gate.get("inner_hold_novel_union_positive_fully_supported", 0) >= 64
            and gate.get("inner_hold_parent_wrong_among_eligible", 0) >= 16,
            "fresh census report or adequacy gate differs")
    require(proof.get("schema") == CENSUS_PROOF_SCHEMA and proof.get("verified") is True
            and proof.get("report_sha256") == EXPECTED_CENSUS_HASHES["report.json"]
            and proof.get("capture_sha256") == EXPECTED_CENSUS_HASHES["capture.jsonl"]
            and proof.get("closure_sha256") == EXPECTED_CENSUS_HASHES["capture.complete.json"]
            and proof.get("verifier_sha256") == EXPECTED_CENSUS_VERIFIER_SHA256
            and sha256_file(CENSUS_VERIFIER) == EXPECTED_CENSUS_VERIFIER_SHA256
            and proof.get("partition_counts") == {"fit": FIT_ROWS, "inner_hold": HOLD_ROWS}
            and proof.get("metrics", {}).get("fit", {}).get("rows") == FIT_ROWS
            and proof.get("metrics", {}).get("inner_hold", {}).get("rows") == HOLD_ROWS
            and proof.get("adequacy_gate") == gate
            and contracts == {"model_loaded": False, "tokenizer_loaded": True,
                              "source_scores_checked_before_labels": True,
                              "dev_accessed": False, "test_accessed": False,
                              "production_changed": False},
            "fresh census independent proof differs")
    require(closure.get("rows") == EXPECTED_ROWS
            and closure.get("capture_sha256") == EXPECTED_CENSUS_HASHES["capture.jsonl"]
            and closure.get("source_only") is True,
            "fresh census source closure differs")
    prepared_matches = []
    for raw, digest in report.get("pins", {}).items():
        path = Path(raw)
        path = path if path.is_absolute() else ROOT / path
        require(path.is_file() and sha256_file(path) == digest,
                f"fresh census pin differs: {raw}")
        if path.name == "prepared.jsonl":
            prepared_matches.append(path)
    require(len(prepared_matches) == 1, "fresh prepared TRAIN pin is ambiguous")
    pins = {path_key(path): digest for path, digest in
            ((paths[name], expected) for name, expected in EXPECTED_CENSUS_HASHES.items())}
    pins.update({path_key(Path(raw) if Path(raw).is_absolute() else ROOT / raw): digest
                 for raw, digest in report["pins"].items()})
    return {"report": report, "proof": proof, "closure": closure,
            "capture": paths["capture.jsonl"], "prepared": prepared_matches[0],
            "pins": pins}


def load_source_rows(prepared_path: Path) -> list[dict]:
    rows = [base.source_projection(row) for row in iter_jsonl(prepared_path)]
    require(len(rows) == EXPECTED_ROWS
            and sum(row["split"] == "fit" for row in rows) == FIT_ROWS
            and sum(row["split"] == "inner_hold" for row in rows) == HOLD_ROWS
            and len({row["id"] for row in rows}) == EXPECTED_ROWS,
            "fresh prepared TRAIN partition differs")
    return rows


def load_labels(prepared_path: Path, split: str) -> dict[str, list[str]]:
    labels = {}
    for row in iter_jsonl(prepared_path):
        if row.get("split") != split:
            continue
        expected = row.get("expected")
        require(isinstance(expected, list) and expected
                and all(isinstance(value, str) and value for value in expected),
                f"{split} labels differ")
        labels[row["id"]] = expected
    expected_count = FIT_ROWS if split == "fit" else HOLD_ROWS
    require(len(labels) == expected_count, f"{split} label denominator differs")
    return labels


def parameter_fingerprint(model) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def parameter_hashes(model) -> dict[str, str]:
    return {name: hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for name, value in model.state_dict().items()}


def runtime_rng_hashes(torch) -> dict[str, str]:
    python_payload = repr(random.getstate()).encode()
    torch_payload = torch.get_rng_state().detach().cpu().contiguous().numpy().tobytes()
    return {"python": hashlib.sha256(python_payload).hexdigest(),
            "torch": hashlib.sha256(torch_payload).hexdigest()}


def checkpoint_metadata(checkpoint: Path, parent_metadata: dict, arm: str,
                        selection_sha: str) -> dict:
    from hanglish.models.sentence_ranker import REQUIRED_ARTIFACTS
    hashes = {name: sha256_file(checkpoint / name) for name in sorted(REQUIRED_ARTIFACTS)}
    return {"format_version": 1, "max_length": MAX_LENGTH,
            "inference_batch_size": parent_metadata["inference_batch_size"],
            "artifact_hashes": hashes, "training_recipe": SCHEMA,
            "arm": arm, "selection_sha256": selection_sha, "seed": SEED,
            "encoder_lr": 1e-5, "head_lr": 5e-5, "weight_decay": 0.01,
            "clip_norm": 1.0, "unfrozen_top_encoder_layers": 2,
            "gradient_accumulation": ACCUMULATION, "epochs": 1}


def save_and_verify_checkpoint(directory: Path, ranker, arm: str,
                               selection_sha: str, probe: dict,
                               expected_parameter_fingerprint: str, watchdog) -> dict:
    checkpoint = directory / "checkpoint"
    checkpoint.mkdir()
    with watchdog.operation(f"{arm}:save_checkpoint", update=True):
        ranker.model.save_pretrained(checkpoint, safe_serialization=True)
        ranker.tokenizer.save_pretrained(checkpoint)
        metadata = checkpoint_metadata(checkpoint, ranker.metadata, arm, selection_sha)
        write_json(checkpoint / "metadata.json", metadata)
    with watchdog.operation(f"{arm}:score_saved_probe", update=True):
        before = ranker.score_pairs(
            [""] * len(probe["candidates"]), probe["candidates"])
    from hanglish.models.sentence_ranker import SentenceRanker
    with watchdog.operation(f"{arm}:reload_checkpoint", update=True):
        reloaded = SentenceRanker.load(checkpoint)
        after = reloaded.score_pairs(
            [""] * len(probe["candidates"]), probe["candidates"])
    reloaded_fingerprint = parameter_fingerprint(reloaded.model)
    delta = max((abs(left - right) for left, right in zip(before, after)), default=0.0)
    require(reloaded_fingerprint == expected_parameter_fingerprint
            and delta <= 1e-5 and max(range(len(before)), key=before.__getitem__)
            == max(range(len(after)), key=after.__getitem__),
            "checkpoint reload score parity differs")
    verification = {"probe_id": probe["id"], "scores_before": before,
                    "scores_after": after, "max_score_delta": delta,
                    "argmax_same": True,
                    "parameter_fingerprint": reloaded_fingerprint,
                    "trained_parameter_fingerprint": expected_parameter_fingerprint,
                    "files": {path.name: sha256_file(path)
                              for path in sorted(checkpoint.iterdir()) if path.is_file()}}
    write_json(directory / "checkpoint-verification.json", verification)
    return {"ranker": reloaded, "verification": verification,
            "checkpoint_path": path_key(checkpoint)}


def train_arm(arm: str, selection: dict, parent_checkpoint: Path, output: Path,
              watchdog, torch, training_module) -> dict:
    from hanglish.models.sentence_ranker import SentenceRanker
    random.seed(SEED); torch.manual_seed(SEED)
    with watchdog.operation(f"{arm}:load_parent", update=True):
        loaded = SentenceRanker.load(parent_checkpoint)
    initial_fingerprint = parameter_fingerprint(loaded.model)
    trainable = training_module._set_trainable_layers(loaded.model, 2)
    trainable_names = {name for name, parameter in loaded.model.named_parameters()
                       if parameter.requires_grad}
    before_parameters = parameter_hashes(loaded.model)
    args = SimpleNamespace(encoder_lr=1e-5, head_lr=5e-5, weight_decay=0.01,
                           max_length=MAX_LENGTH, device="cpu")
    optimizer = training_module._optimizer(loaded.model, args, head_only=False)
    initial_rng = runtime_rng_hashes(torch)
    require(not optimizer.state, "new arm optimizer state is not empty")
    loaded.model.train()
    arm_dir = output / arm
    arm_dir.mkdir()
    updates_path = arm_dir / "updates.jsonl"
    rows = selection["rows"]
    started = time.monotonic(); update_count = 0
    for group in accumulation_groups(rows):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        with watchdog.operation(f"{arm}:update:{update_count + 1}", update=True):
            for row in group:
                chosen = row["arms"][arm]
                example = training_module.SentenceExample(
                    "", tuple(chosen["candidates"]),
                    tuple(chosen["positive_indices"]), row["id"], "recombination")
                loss = training_module._loss(loaded.model, loaded.tokenizer, [example], args)
                require(bool(torch.isfinite(loss)), "non-finite training loss")
                (loss / len(group)).backward()
                losses.append(float(loss.detach()))
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(
                loaded.model.parameters(), 1.0))
            require(math.isfinite(gradient_norm), "non-finite gradient norm")
            optimizer.step()
        update_count += 1
        elapsed = time.monotonic() - started
        record = {"arm": arm, "update": update_count,
                  "row_ids": [row["id"] for row in group],
                  "microsteps": len(group), "loss_mean": sum(losses) / len(losses),
                  "losses": losses, "gradient_norm_before_clip": gradient_norm,
                  "elapsed_s": elapsed}
        if update_count == 3:
            record["projected_training_s"] = elapsed / 3 * selection["updates_per_arm"]
            require(record["projected_training_s"] <= 1_200,
                    "first-three-update training estimate exceeds 1200 seconds")
        append_jsonl(updates_path, record)
    require(update_count == selection["updates_per_arm"], "arm update count differs")
    final_fingerprint = parameter_fingerprint(loaded.model)
    after_parameters = parameter_hashes(loaded.model)
    changed = sorted(name for name in before_parameters
                     if before_parameters[name] != after_parameters[name])
    require(changed and set(changed).issubset(trainable_names),
            "changed parameters exceed the declared trainable set")
    loaded.model.eval()
    wrapped = SentenceRanker(loaded.model, loaded.tokenizer, max_length=MAX_LENGTH,
                             batch_size=loaded.batch_size, metadata=loaded.metadata)
    probe = {"id": rows[0]["id"],
             "candidates": rows[0]["arms"][arm]["candidates"]}
    checkpoint = save_and_verify_checkpoint(
        arm_dir, wrapped, arm, sha256_file(output / "selection.json"), probe,
        final_fingerprint, watchdog)
    return {"arm": arm, "initial_parameter_fingerprint": initial_fingerprint,
            "final_parameter_fingerprint": final_fingerprint,
            "trainable": trainable, "changed_parameter_names": changed,
            "initial_rng_sha256": initial_rng,
            "initial_optimizer_state_entries": 0,
            "updates": update_count,
            "updates_sha256": sha256_file(updates_path),
            "training_elapsed_s": time.monotonic() - started,
            "ranker": checkpoint["ranker"],
            "checkpoint_path": checkpoint["checkpoint_path"],
            "checkpoint_verification": checkpoint["verification"]}


def capture_hold_predictions(source_rows: list[dict], census_rows: list[dict],
                             rankers: dict, output: Path, watchdog) -> list[dict]:
    predictions = []
    with (output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
        for source, census in zip(source_rows, census_rows):
            if source["split"] != "inner_hold":
                continue
            row = {"id": source["id"], "input": source["input"],
                   "component_id": source["component_id"],
                   "frozen": {policy: {"applied": census[policy]["applied"],
                                        "result": census[policy]["result"],
                                        "candidates": list(census[policy]["candidates"])}
                              for policy in ("original", "recombined")}}
            for arm, ranker in rankers.items():
                with watchdog.operation(f"hold:{arm}:{source['id']}", update=True):
                    scored = base.score_row(source, ranker)
                    row[arm] = {key: scored[key] for key in
                                ("score_texts", "scores", "original", "recombined")}
            predictions.append(row)
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False) + "\n")
            stream.flush()
        os.fsync(stream.fileno())
    require(len(predictions) == HOLD_ROWS, "HOLD prediction denominator differs")
    capture_sha = sha256_file(output / "predictions.jsonl")
    closure = {"schema": SCHEMA + "-source-only-hold-closure",
               "rows": HOLD_ROWS, "capture_sha256": capture_sha,
               "source_only": True, "targets_joined": False,
               "context": "", "input_truncation": False}
    write_json(output / "predictions.complete.json", closure)
    return predictions


def pool_coverage(source_rows: list[dict], labels: dict[str, list[str]]) -> dict:
    result = {}
    hold = [row for row in source_rows if row["split"] == "inner_hold"]
    for policy, key in (("original", "original_candidates"),
                        ("recombined", "recombined_candidates")):
        covered = [row["id"] for row in hold
                   if (set(row[key]) | {row["input"]}).intersection(labels[row["id"]])]
        result[policy] = {"rows": len(hold), "covered": len(covered),
                          "covered_ids": covered}
    return result


def run(census_dir: Path, proof_path: Path, output: Path) -> dict:
    authorization = validate_authorization(census_dir, proof_path)
    from scripts import preflight_pko_seq2seq as preflight
    from scripts import train_sentence_ranker as training_module
    import torch

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); baseline_swap = preflight.system_swap_bytes()
    pins = dict(authorization["pins"])
    tracked = [Path(__file__).resolve(), PLAN, CENSUS_VERIFIER,
               ROOT / "scripts/score_recombination_training_pools.py",
               ROOT / "scripts/train_sentence_ranker.py",
               ROOT / "scripts/preflight_pko_seq2seq.py",
               ROOT / "hanglish/models/sentence_ranker.py"]
    pins.update({path_key(path): sha256_file(path) for path in tracked})
    model_pins = base.frozen_model_pins()
    pins.update(model_pins)
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    source_rows = load_source_rows(authorization["prepared"])
    census_rows = list(iter_jsonl(authorization["capture"]))
    require(len(census_rows) == EXPECTED_ROWS, "fresh census denominator differs")
    # Only FIT labels are exposed while forming the training selection.
    fit_labels = load_labels(authorization["prepared"], "fit")
    selection = build_shared_selection(source_rows, census_rows, fit_labels)
    require(selection["updates_per_arm"] >= 3,
            "shared FIT selection cannot support the three-update timing gate")
    write_json(output / "selection.json", selection)

    class TrainingWatchdog(preflight.ResourceWatchdog):
        def _monitor(self):
            while not self.stop_event.wait(self.interval_s):
                try:
                    preflight.append_jsonl(self.trace, self._sample("watchdog"), self.lock)
                except BaseException as error:
                    self.failure = f"{type(error).__name__}: {error}"
                    if not (output / "report.json").exists():
                        preflight.write_json(output / "report.json", {
                            "schema": SCHEMA, "completed": False,
                            "error": self.failure, "dev_opened": False,
                            "test_opened": False, "production_changed": False})
                    os.kill(os.getpid(), signal.SIGUSR1)
                    if not self.stop_event.wait(5.0):  # pragma: no cover
                        os._exit(124)
                    return

    try:
        with TrainingWatchdog(started, baseline_swap, output / "resources.jsonl",
                              interval_s=5.0, limits=LIMITS) as watchdog:
            arms = {}
            for arm in ("a_original", "b_union"):
                arms[arm] = train_arm(arm, selection, base.CHECKPOINT, output,
                                      watchdog, torch, training_module)
                require(base.frozen_model_pins() == model_pins,
                        "frozen parent changed during paired training")
            require(arms["a_original"]["initial_parameter_fingerprint"]
                    == arms["b_union"]["initial_parameter_fingerprint"],
                    "paired arms did not start from identical parameters")
            require(arms["a_original"]["initial_rng_sha256"]
                    == arms["b_union"]["initial_rng_sha256"],
                    "paired arms did not start from identical RNG states")
            rankers = {arm: record["ranker"]
                       for arm, record in arms.items()}
            predictions = capture_hold_predictions(
                source_rows, census_rows, rankers, output, watchdog)
            require(sha256_file(authorization["prepared"])
                    == authorization["pins"][path_key(authorization["prepared"])],
                    "TRAIN labels changed before HOLD late join")
            hold_labels = load_labels(authorization["prepared"], "inner_hold")
            metrics = {branch: {policy: metric_summary(
                predictions, hold_labels, branch, policy)
                for policy in ("original", "recombined")}
                       for branch in ("frozen", "a_original", "b_union")}
            coverage = pool_coverage(source_rows, hold_labels)
            gate = paired_gate(metrics)
            final_resources = watchdog._sample("final")
        require(base.frozen_model_pins() == model_pins,
                "frozen parent changed after paired training")
        for path, digest in pins.items():
            require(Path(path).is_file() and sha256_file(Path(path)) == digest,
                    f"paired-training pin changed: {path}")
        report = {"schema": SCHEMA, "completed": True,
                  "rows": EXPECTED_ROWS,
                  "partition_counts": {"fit": FIT_ROWS, "inner_hold": HOLD_ROWS},
                  "selection": {key: value for key, value in selection.items()
                                if key != "rows"},
                  "arms": {arm: {key: value for key, value in record.items()
                                  if key != "ranker"}
                           for arm, record in arms.items()},
                  "metrics": metrics, "pool_coverage": coverage,
                  "paired_gate": gate,
                  "outputs": {
                      "selection.json": sha256_file(output / "selection.json"),
                      "predictions.jsonl": sha256_file(output / "predictions.jsonl"),
                      "predictions.complete.json": sha256_file(
                          output / "predictions.complete.json"),
                      "a_original/updates.jsonl": sha256_file(
                          output / "a_original/updates.jsonl"),
                      "a_original/checkpoint-verification.json": sha256_file(
                          output / "a_original/checkpoint-verification.json"),
                      "b_union/updates.jsonl": sha256_file(
                          output / "b_union/updates.jsonl"),
                      "b_union/checkpoint-verification.json": sha256_file(
                          output / "b_union/checkpoint-verification.json")},
                  "pins": dict(sorted(pins.items())), "model_pins": model_pins,
                  "resources": {"limits": LIMITS,
                                "baseline_swap_bytes": baseline_swap,
                                "final": final_resources,
                                "trace_sha256": sha256_file(
                                    output / "resources.jsonl")},
                  "elapsed_s": time.monotonic() - started,
                  "contracts": {"fit_labels_used_for_training_only": True,
                                "hold_predictions_closed_before_label_join": True,
                                "dev_opened": False, "test_opened": False,
                                "production_changed": False,
                                "runtime_promotion": False},
                  "finished_at_utc": datetime.now(timezone.utc).isoformat()}
        write_json(output / "report.json", report)
        return report
    except BaseException as error:
        if not (output / "report.json").exists():
            write_json(output / "report.json", {
                "schema": SCHEMA, "completed": False,
                "error_type": type(error).__name__, "error": str(error),
                "dev_opened": False, "test_opened": False,
                "production_changed": False})
        raise


def self_test() -> dict:
    source = {"id": "fit-1", "input": "ABC 1!", "split": "fit",
              "component_id": "c1", "parent_result": "가",
              "parent_candidates": ["가", "ABC 1!"],
              "original_candidates": ["가", "나"],
              "recombined_candidates": ["가", "나", "다"],
              "novel_candidates": ["다"]}
    census = {"id": "fit-1", "input": "ABC 1!", "split": "fit",
              "component_id": "c1",
              "score_texts": ["가", "나", "다"], "scores": [0.0, 2.0, 1.0],
              "original": {"applied": True, "result": "나",
                           "candidates": ["나", "가", "ABC 1!"]},
              "recombined": {"applied": True, "result": "나",
                             "candidates": ["나", "다", "가", "ABC 1!"]}}
    selection = build_shared_selection([source], [census], {"fit-1": ["가"]})
    require(selection["row_ids"] == ["fit-1"]
            and selection["rows"][0]["arms"]["b_union"]["candidates"]
            == ["가", "나", "다"], "paired selection self-test differs")
    return {"schema": SCHEMA + "-self-test", "passed": True,
            "model_loaded": False, "data_opened": False}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census", type=Path)
    parser.add_argument("--proof", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.self_test:
        print(json.dumps(self_test(), ensure_ascii=False, sort_keys=True))
        return 0
    require(args.census is not None and args.proof is not None and args.output is not None,
            "--census, --proof, and --output are required")
    print(json.dumps(run(args.census, args.proof, args.output),
                     ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
