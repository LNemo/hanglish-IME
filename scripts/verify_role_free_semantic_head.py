"""Independent verifier for the completed role-free semantic head v2 run."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import unicodedata

import torch

from scripts.symmetric_pair_head import SymmetricPairHead


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / ".omx/artifacts/role-free-semantic-head-v2"
SUPERVISOR = ROOT / ".omx/artifacts/role-free-semantic-head-v2-supervisor.json"
INPUT = ROOT / ".omx/artifacts/semantic-pair-training-input-v2"
OUTPUT = ROOT / ".omx/artifacts/role-free-semantic-head-v2-independent.json"
PREFLIGHT = ROOT / ".omx/artifacts/role-free-semantic-head-v2-preflight.json"
NATURAL_CASES = ROOT / "data/natural_context/final-choice-v1/natural_fit/cases.jsonl"
CHALLENGE_CASES = ROOT / "data/natural_context/final-choice-v1/challenge_fit/cases.jsonl"

TRAINER_SHA = "f9504d0d7ae34aa21c9f7a3f0967f41c438c156f74f4e9aeb2a7a9768827ea45"
PLAN_SHA = "6869945f2636e807b31b483567d41d992ff8aa28f282c2a60e65fee8f30e0556"
INPUT_MANIFEST_SHA = "3ab561d1a767e7a65b4706c2cd0f668366693536849e17f2f3e620fbcf5550a6"
INPUT_ROWS_SHA = "6a1f7f61a7f7ffeb9209772559412931137d71ef0173d7429264d9cb53aed396"
INPUT_PAIRS_SHA = "80854223a30e90e8ecc1daaa85b4598f747232db61fd82adc5dd9e1eeb411abf"
MODEL_SOURCE_SHA = "5166197923917d229b81d58f0e3561de3b141222d8597b7d0313bab4bb0b7d9c"
MODEL_WEIGHT_SHA = "737e7dbcb9f159d4719a06095268836a2634a673ac925de96c13c9d0bf4d895b"
PREFLIGHT_SHA = "ee41cfd6b7a87a7695456ec333049fd531ca2323c1c1228aad9c4e4bf3a7f7a6"
PHONOLOGY_SHA = "9e1f26cc153571c5a102577530a772f1288fc24294f573d3a470af45166e8b49"
FOLD_SALT = "symmetric-pair-fit-v1"
ROWS = 3492
PAIRS = 7677
ELIGIBLE = 7652
MAX_SECONDS = 2700
MAX_SUPERVISOR_SECONDS = 2800
MAX_RSS = 3 * 1024**3


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_bytes().splitlines() if line]


def verify_pins(pins):
    require(isinstance(pins, dict) and pins, "source pins missing")
    for name, expected in pins.items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        require(path.is_file() and sha(path) == expected, f"source pin differs: {name}")


def validate_feature_vectors(features, eligible):
    selected = features[eligible]
    require(bool(torch.isfinite(selected).all()), "non-finite eligible feature")
    exact_zero = torch.all(selected == 0, dim=1)
    norms = selected.norm(dim=1)
    unit = torch.isclose(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)
    require(bool((exact_zero | unit).all()),
            "eligible feature must be exactly zero or unit normalized")
    return int(exact_zero.sum())


def normalized(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def independent_assign_folds(rows):
    """Rebuild connected components from group and normalized source fields."""
    parent = list(range(len(rows)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    seen = {}
    for index, row in enumerate(rows):
        keys = (("group", row["group_id"]),
                ("source", normalized(row["input"]), normalized(row["context"])))
        for key in keys:
            if key in seen:
                parent[find(index)] = find(seen[key])
            seen[key] = index
    components = {}
    for index, row in enumerate(rows):
        components.setdefault(find(index), []).append(row["id"])
    digests = {root: hashlib.sha256(canonical(sorted(ids))).hexdigest()
               for root, ids in components.items()}
    assignments = []
    for index, row in enumerate(rows):
        component = digests[find(index)]
        fold = int(hashlib.sha256((FOLD_SALT + component).encode()).hexdigest(), 16) % 4
        require(row.get("component") == component and row.get("fold") == fold,
                "independent fold/component replay differs")
        assignments.append((component, fold))
    return assignments


def alternative_margin(canonical_margin, current_is_a):
    require(type(current_is_a) is bool and math.isfinite(canonical_margin),
            "finite margin and current role required")
    return -canonical_margin if current_is_a else canonical_margin


def decide(pair_indices, canonical_margins, pairs):
    require(len(pair_indices) == len(canonical_margins), "pair margin count differs")
    scores = [alternative_margin(float(margin),
              pairs[index]["canonical_orientation"]["current_is_a"])
              for index, margin in zip(pair_indices, canonical_margins)]
    if not scores:
        return None, scores
    maximum = max(scores)
    winners = [offset for offset, score in enumerate(scores) if score == maximum]
    chosen = pair_indices[winners[0]] if maximum > 0 and len(winners) == 1 else None
    return chosen, scores


def replay_row(row, pairs, eligible, margins):
    indices = [index for index in row["pair_indices"] if eligible[index]]
    canonical_margins = [float(margins[index]) for index in indices]
    chosen, alternative_margins = decide(indices, canonical_margins, pairs)
    chosen_text = (row["current_result"] if chosen is None
                   else pairs[chosen]["alternative"]["text"])
    return {
        "row_index": row["index"], "id": row["id"],
        "source_role": row["source_role"], "fold": row["fold"],
        "component": row["component"], "current_text": row["current_result"],
        "chosen_text": chosen_text, "chosen_pair_index": chosen,
        "eligible_pair_indices": indices, "canonical_margins": canonical_margins,
        "alternative_margins": alternative_margins,
    }


def read_targets(rows):
    result = {}
    for role, path, offset, count in (
        ("natural_fit", NATURAL_CASES, 0, 2048),
        ("challenge_fit", CHALLENGE_CASES, 2048, 1444),
    ):
        with path.open() as stream:
            for local_index, line in enumerate(stream):
                if local_index >= count:
                    break
                source = json.loads(line)
                index = offset + local_index
                require(rows[index]["source_role"] == role
                        and rows[index]["id"] == source["id"],
                        "target source order differs")
                result[index] = source["expected"]
    require(set(result) == set(range(ROWS)), "complete TRAIN labels required")
    return result


def label_name(pair, expected):
    accepted = set(expected)
    orientation = pair["canonical_orientation"]
    a = orientation["candidate_a_text"] in accepted
    b = orientation["candidate_b_text"] in accepted
    if a and not b:
        return "A_BETTER"
    if b and not a:
        return "B_BETTER"
    return "TIE" if a else "UNKNOWN"


def compute_metrics(rows, predictions, targets):
    require(len(rows) == len(predictions) == ROWS
            and set(targets) == set(range(ROWS)), "all denominator rows required")
    metrics = {"before_top1": 0, "top1": 0, "natural_before_top1": 0,
               "natural_top1": 0, "correction_ids": [],
               "natural_correction_ids": [], "regression_ids": []}
    for index, (row, prediction) in enumerate(zip(rows, predictions)):
        require(prediction["row_index"] == index and prediction["id"] == row["id"],
                "prediction order differs")
        accepted = set(targets[index])
        before = row["current_result"] in accepted
        after = prediction["chosen_text"] in accepted
        metrics["before_top1"] += before
        metrics["top1"] += after
        if row["source_role"] == "natural_fit":
            metrics["natural_before_top1"] += before
            metrics["natural_top1"] += after
        if after and not before:
            metrics["correction_ids"].append(row["id"])
            if row["source_role"] == "natural_fit":
                metrics["natural_correction_ids"].append(row["id"])
        if before and not after:
            metrics["regression_ids"].append(row["id"])
    metrics["mechanism_gate_passed"] = (
        bool(metrics["natural_correction_ids"]) and not metrics["regression_ids"])
    return metrics


def validate_fold_record(fold, record, checkpoint, rows, pairs, eligible):
    pair_folds = [pair["fold"] for pair in pairs]
    pair_rows = [pair["row_index"] for pair in pairs]
    held_pairs = [index for index in range(len(pairs))
                  if eligible[index] and pair_folds[index] == fold]
    train_pairs = [index for index in range(len(pairs))
                   if eligible[index] and pair_folds[index] != fold]
    held_rows = sorted({pair_rows[index] for index in held_pairs})
    train_rows = sorted({pair_rows[index] for index in train_pairs})
    require(not set(held_rows) & set(train_rows), "train/held source row leak")
    require(not ({rows[index]["component"] for index in held_rows}
                 & {rows[index]["component"] for index in train_rows}),
            "train/held component leak")
    require(record["fold"] == checkpoint["fold"] == fold,
            "checkpoint fold differs")
    require(record["train_source_indices"] == checkpoint["train_source_indices"] == train_rows,
            "checkpoint train source indices differ")
    require(record["held_source_indices"] == checkpoint["held_source_indices"] == held_rows,
            "checkpoint held source indices differ")
    require(record["train_pairs"] == len(train_pairs)
            and record["held_pairs"] == len(held_pairs)
            and record["trainable_parameters"] == 768
            and record["optimizer_updates"] == 100,
            "fold training counts differ")
    return held_pairs


def verify(run=RUN, supervisor_path=SUPERVISOR, output=OUTPUT):
    run, supervisor_path, output = Path(run), Path(supervisor_path), Path(output)
    require(not output.exists(), "independent verification output already exists")
    supervisor = json.loads(supervisor_path.read_bytes())
    require(supervisor.get("exit_code") == 0 and supervisor.get("timed_out") is False,
            "successful supervisor receipt required")
    require(type(supervisor.get("elapsed_seconds")) in (int, float)
            and supervisor["elapsed_seconds"] <= MAX_SUPERVISOR_SECONDS,
            "supervisor time cap exceeded")

    report_path = run / "report.json"
    require(sha(report_path) == supervisor.get("report_sha256"),
            "supervisor report hash differs")
    report = json.loads(report_path.read_bytes())
    require(report.get("completed") is True, "completed run report required")
    require(report.get("decision_only") is True
            and report.get("generalization_claim_allowed") is False
            and report.get("deployed") is False
            and report.get("production_changed") is False
            and report.get("promotion_allowed") is False,
            "diagnostic scope flags differ")
    require(report.get("rows") == ROWS and report.get("pairs") == PAIRS
            and report.get("eligible_pairs") == ELIGIBLE
            and report.get("skipped_pairs") == PAIRS - ELIGIBLE,
            "fixed denominator or eligibility differs")
    require(report.get("elapsed_seconds") <= MAX_SECONDS
            and report.get("peak_rss_bytes") <= MAX_RSS,
            "worker resource cap exceeded")

    configuration = json.loads((run / "configuration.json").read_bytes())
    require(sha(PREFLIGHT) == PREFLIGHT_SHA, "v2 preflight changed")
    preflight = json.loads(PREFLIGHT.read_bytes())
    require(preflight.get("trainer_sha256") == TRAINER_SHA
            and preflight.get("plan_sha256") == PLAN_SHA
            and preflight.get("input_manifest_sha256") == INPUT_MANIFEST_SHA
            and preflight.get("input_rows_pairs_unchanged") is True
            and preflight.get("model_loaded") is False
            and preflight.get("training_started") is False,
            "v2 preflight contract differs")
    recipe = configuration.get("recipe", {})
    require(recipe.get("max_seconds") == MAX_SECONDS
            and recipe.get("max_rss_bytes") == MAX_RSS
            and recipe.get("folds") == 4 and recipe.get("epochs") == 100,
            "fixed v2 recipe differs")
    pins = configuration.get("pins")
    require(pins == report.get("source_hashes"), "configuration/report source pins differ")
    require(pins.get("scripts/probe_role_free_semantic_head.py") == TRAINER_SHA
            and pins.get(".omx/plans/role-free-semantic-head-v2.md") == PLAN_SHA
            and pins.get(".omx/artifacts/semantic-pair-training-input-v2/manifest.json")
            == INPUT_MANIFEST_SHA
            and pins.get(".omx/artifacts/semantic-pair-training-input-v2/rows.jsonl")
            == INPUT_ROWS_SHA
            and pins.get(".omx/artifacts/semantic-pair-training-input-v2/pairs.jsonl")
            == INPUT_PAIRS_SHA
            and pins.get("data/external/klue-roberta-small/SOURCE.json") == MODEL_SOURCE_SHA
            and pins.get("data/external/klue-roberta-small/model.safetensors") == MODEL_WEIGHT_SHA,
            "fixed execution lineage differs")
    require(pins.get("hanglish/core/phonology.py") == PHONOLOGY_SHA,
            "phonology dependency pin differs")
    verify_pins(pins)
    model_source = json.loads((ROOT / "data/external/klue-roberta-small/SOURCE.json").read_bytes())
    require(configuration.get("base_model_source") == model_source
            and report.get("base_model_source") == model_source,
            "base model provenance differs")

    manifest = json.loads((INPUT / "manifest.json").read_bytes())
    require(sha(INPUT / "manifest.json") == INPUT_MANIFEST_SHA
            and sha(INPUT / "rows.jsonl") == INPUT_ROWS_SHA
            and sha(INPUT / "pairs.jsonl") == INPUT_PAIRS_SHA,
            "fixed input hashes differ")
    rows, pairs = read_jsonl(INPUT / "rows.jsonl"), read_jsonl(INPUT / "pairs.jsonl")
    require(len(rows) == ROWS and len(pairs) == PAIRS, "fixed input counts differ")
    independent_assign_folds(rows)

    features_path = run / "encoder-features.pt"
    predictions_path = run / "source-predictions.jsonl"
    folds_path = run / "folds.json"
    require(sha(features_path) == report.get("features_sha256")
            and sha(predictions_path) == report.get("predictions_sha256")
            and sha(folds_path) == report.get("folds_sha256"),
            "reported output hash differs")
    saved = torch.load(features_path, map_location="cpu", weights_only=True)
    require(set(saved) == {"features", "eligible", "pair_source_indices"},
            "feature payload keys differ")
    features, eligible, source_indices = (
        saved["features"], saved["eligible"], saved["pair_source_indices"])
    require(features.shape == (PAIRS, 768) and features.dtype == torch.float32
            and eligible.shape == (PAIRS,) and eligible.dtype == torch.bool
            and source_indices.tolist() == [pair["row_index"] for pair in pairs]
            and eligible.tolist() == [pair["eligible_le_512"] for pair in pairs]
            and int(features[~eligible].count_nonzero()) == 0,
            "feature or eligibility payload differs")
    zero_feature_pairs = validate_feature_vectors(features, eligible)

    fold_records = json.loads(folds_path.read_bytes())
    require(isinstance(fold_records, list) and len(fold_records) == 4,
            "four fold reports required")
    torch.set_num_threads(2)
    margins = torch.full((PAIRS,), float("nan"))
    for fold in range(4):
        record = fold_records[fold]
        checkpoint_path = run / f"fold-{fold}.pt"
        require(sha(checkpoint_path) == record.get("checkpoint_sha256"),
                "checkpoint hash differs")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        held_pairs = validate_fold_record(
            fold, record, checkpoint, rows, pairs, eligible.tolist())
        head = SymmetricPairHead(768)
        head.load_state_dict(checkpoint["state_dict"])
        require(sum(parameter.numel() for parameter in head.parameters()) == 768,
                "head parameter count differs")
        index = torch.tensor(held_pairs, dtype=torch.long)
        with torch.no_grad():
            predicted = head.score_features(features.index_select(0, index))
            require(torch.equal(predicted, -head.score_features(
                -features.index_select(0, index))), "head swap sign differs")
            margins[index] = predicted
    require(not bool(torch.isnan(margins[eligible]).any()), "eligible OOF margins missing")

    # Read and close all source predictions before accessing aggregate TRAIN labels.
    stored_predictions = read_jsonl(predictions_path)
    require(len(stored_predictions) == ROWS, "all source predictions required")
    replayed = [replay_row(row, pairs, eligible.tolist(), margins)
                for row in rows]
    require(stored_predictions == replayed, "independent margin/action replay differs")
    targets = read_targets(rows)
    metrics = compute_metrics(rows, replayed, targets)
    for fold, record in enumerate(fold_records):
        train_pairs = [index for index, pair in enumerate(pairs)
                       if eligible[index] and pair["fold"] != fold]
        labels = Counter(label_name(pairs[index], targets[pairs[index]["row_index"]])
                         for index in train_pairs)
        informative = labels["A_BETTER"] + labels["B_BETTER"]
        require(record.get("label_counts") == dict(labels)
                and record.get("informative_train_pairs") == informative,
                "fold label counts differ")
        first_loss, last_loss = record.get("first_loss"), record.get("last_loss")
        require(type(first_loss) in (int, float) and type(last_loss) in (int, float)
                and math.isfinite(first_loss) and math.isfinite(last_loss)
                and math.isclose(first_loss, math.log(2), rel_tol=0, abs_tol=1e-6)
                and 0 <= last_loss <= first_loss,
                "fold loss evidence differs")
    for key, value in metrics.items():
        require(report.get(key) == value, f"reported metric differs: {key}")
    require(report.get("switches") == sum(
        prediction["chosen_pair_index"] is not None for prediction in replayed)
        and report.get("rows_without_eligible_pairs") == sum(
            not prediction["eligible_pair_indices"] for prediction in replayed),
            "reported action counts differ")
    require(report.get("mechanism_gate_passed")
            == (len(metrics["natural_correction_ids"]) >= 1
                and len(metrics["regression_ids"]) == 0),
            "fixed mechanism gate differs")
    verify_pins(pins)

    result = {
        "schema": "role-free-semantic-head-v2-independent-verification",
        "verified": True, "rows": ROWS, "pairs": PAIRS, "eligible_pairs": ELIGIBLE,
        "folds_replayed": 4, "all_oof_margins_replayed": True,
        "zero_feature_pairs": zero_feature_pairs,
        "all_actions_replayed": True, "metrics_recomputed_after_prediction_close": True,
        "before_top1": metrics["before_top1"], "top1": metrics["top1"],
        "natural_corrections": len(metrics["natural_correction_ids"]),
        "regressions": len(metrics["regression_ids"]),
        "mechanism_gate_passed": metrics["mechanism_gate_passed"],
        "generalization_claim_allowed": False, "deployed": False,
        "report_sha256": sha(report_path), "predictions_sha256": sha(predictions_path),
        "features_sha256": sha(features_path), "source_hashes": pins,
        "verifier_sha256": sha(Path(__file__)),
    }
    output.write_bytes(canonical(result) + b"\n")
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), ensure_ascii=False))
