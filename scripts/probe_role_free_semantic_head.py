"""Fixed TRAIN-only diagnostic for a role-free semantic comparison head."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.probe_symmetric_pair_selector import (
    MODEL_SOURCE_SHA, MODEL_WEIGHT_SHA, canonical, rss_bytes, sha,
)
from scripts.semantic_pair_contract import (
    LabelState, SymmetricPairHead, orient_margin, semantic_pair_loss,
)


INPUT = ROOT / ".omx/artifacts/semantic-pair-training-input-v2"
MODEL = ROOT / "data/external/klue-roberta-small"
OUTPUT = ROOT / ".omx/artifacts/role-free-semantic-head-v2"
PLAN = ROOT / ".omx/plans/role-free-semantic-head-v2.md"
PRIOR_PROBE = ROOT / ".omx/artifacts/role-free-semantic-head-v1/resource-probe.json"
PRIOR_PROBE_SHA = "a02ae82bf33c5e7d6766090bd952f406c747888c6f3e02ce0c044b67d79b9a81"
NATURAL_CASES = ROOT / "data/natural_context/final-choice-v1/natural_fit/cases.jsonl"
CHALLENGE_CASES = ROOT / "data/natural_context/final-choice-v1/challenge_fit/cases.jsonl"
INPUT_MANIFEST_SHA = "3ab561d1a767e7a65b4706c2cd0f668366693536849e17f2f3e620fbcf5550a6"
INPUT_ROWS_SHA = "6a1f7f61a7f7ffeb9209772559412931137d71ef0173d7429264d9cb53aed396"
INPUT_PAIRS_SHA = "80854223a30e90e8ecc1daaa85b4598f747232db61fd82adc5dd9e1eeb411abf"
PLAN_SHA = "6869945f2636e807b31b483567d41d992ff8aa28f282c2a60e65fee8f30e0556"
EXPECTED_ROWS = 3492
EXPECTED_PAIRS = 7677
EXPECTED_ELIGIBLE_PAIRS = 7652
EXPECTED_ROWS_WITH_ELIGIBLE_PAIRS = 3155
RECIPE = {
    "seed": 23, "folds": 4, "epochs": 100, "learning_rate": 0.01,
    "weight_decay": 0.01, "max_tokens": 512, "batch_size": 4,
    "threads": 2, "max_seconds": 2700, "max_rss_bytes": 3 * 1024**3,
    "positive_threshold": 0.0,
    "features": "L2-normalized half difference of frozen CLS in canonical AB/BA order",
}


def require_resources(started):
    if (rss_bytes() > RECIPE["max_rss_bytes"]
            or time.monotonic() - started > RECIPE["max_seconds"]):
        raise RuntimeError("role-free semantic resource cap exceeded")


def write_json(path, value):
    Path(path).write_bytes(canonical(value) + b"\n")


def write_jsonl(path, values):
    with Path(path).open("xb") as stream:
        for value in values:
            stream.write(canonical(value) + b"\n")


def verify_pins(pins):
    for name, expected in pins.items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"pinned source differs: {name}")


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_bytes().splitlines() if line]


def validate_fold_closure(rows):
    components = {}
    groups = {}
    for row in rows:
        fold = row.get("fold")
        if type(fold) is not int or not 0 <= fold < RECIPE["folds"]:
            raise ValueError("invalid row fold")
        for mapping, key in ((components, "component"), (groups, "group_id")):
            value = row.get(key)
            if not isinstance(value, str) or not value:
                raise ValueError(f"invalid {key}")
            if value in mapping and mapping[value] != fold:
                raise ValueError(f"{key} crosses folds")
            mapping[value] = fold
    return True


def load_inputs():
    if sha(INPUT / "manifest.json") != INPUT_MANIFEST_SHA:
        raise ValueError("semantic pair input manifest changed")
    manifest = json.loads((INPUT / "manifest.json").read_bytes())
    if (manifest.get("schema") != "semantic-pair-training-input-v1"
            or manifest.get("completed") is not True
            or manifest.get("rows") != EXPECTED_ROWS
            or manifest.get("pairs") != EXPECTED_PAIRS
            or manifest.get("eligible_pairs_le_512") != EXPECTED_ELIGIBLE_PAIRS
            or manifest.get("role_rows") != {"natural_fit": 2048, "challenge_fit": 1444}
            or manifest.get("labels_or_expected_saved") is not False
            or manifest.get("neural_views_are_role_free") is not True):
        raise ValueError("semantic pair input contract differs")
    rows_sha, pairs_sha = sha(INPUT / "rows.jsonl"), sha(INPUT / "pairs.jsonl")
    if (rows_sha != manifest.get("rows_sha256") or rows_sha != INPUT_ROWS_SHA
            or pairs_sha != manifest.get("pairs_sha256") or pairs_sha != INPUT_PAIRS_SHA):
        raise ValueError("semantic pair input payload changed")
    if (manifest.get("rows_sha256") != INPUT_ROWS_SHA
            or manifest.get("pairs_sha256") != INPUT_PAIRS_SHA
            or sum(value > 0 for value in manifest.get("eligible_per_row", []))
            != EXPECTED_ROWS_WITH_ELIGIBLE_PAIRS):
        raise ValueError("semantic pair input qualification facts differ")
    verify_pins(manifest["source_hashes"])
    rows, pairs = _read_jsonl(INPUT / "rows.jsonl"), _read_jsonl(INPUT / "pairs.jsonl")
    if len(rows) != EXPECTED_ROWS or len(pairs) != EXPECTED_PAIRS:
        raise ValueError("complete semantic pair input required")
    if [row.get("index") for row in rows] != list(range(EXPECTED_ROWS)):
        raise ValueError("row indices differ")
    if [pair.get("index") for pair in pairs] != list(range(EXPECTED_PAIRS)):
        raise ValueError("pair indices differ")
    validate_fold_closure(rows)
    for pair in pairs:
        row = rows[pair["row_index"]]
        orientation = pair.get("canonical_orientation", {})
        current_is_a = orientation.get("current_is_a")
        candidate_a = orientation.get("candidate_a_text")
        candidate_b = orientation.get("candidate_b_text")
        views = [json.loads(view) for view in pair.get("neural_views", [])]
        expected_view_keys = {"roman", "context", "candidate_a_text", "candidate_b_text"}
        if (pair.get("row_id") != row["id"] or pair.get("fold") != row["fold"]
                or pair.get("component") != row["component"]
                or pair["index"] not in row["pair_indices"]
                or set(orientation) != {"candidate_a_text", "candidate_b_text", "current_is_a"}
                or type(current_is_a) is not bool
                or pair.get("current_source_index") != 0
                or pair["current"].get("text") != (candidate_a if current_is_a else candidate_b)
                or pair["alternative"].get("text") != (candidate_b if current_is_a else candidate_a)
                or len(views) != 2 or any(set(view) != expected_view_keys for view in views)
                or (views[0]["candidate_a_text"], views[0]["candidate_b_text"])
                != (candidate_a, candidate_b)
                or (views[1]["candidate_a_text"], views[1]["candidate_b_text"])
                != (candidate_b, candidate_a)
                or views[0]["roman"] != views[1]["roman"]
                or views[0]["context"] != views[1]["context"]
                or len(pair.get("token_lengths", [])) != 2
                or pair.get("eligible_le_512")
                != (max(pair["token_lengths"]) <= RECIPE["max_tokens"])):
            raise ValueError("pair envelope differs")
    if [index for row in rows for index in row["pair_indices"]] != list(range(EXPECTED_PAIRS)):
        raise ValueError("row pair index closure differs")
    pins = dict(manifest["source_hashes"])
    pins.update({
        str((INPUT / name).relative_to(ROOT)): sha(INPUT / name)
        for name in ("manifest.json", "rows.jsonl", "pairs.jsonl")
    })
    pins[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    for path in (ROOT / "scripts/semantic_pair_contract.py",
                 ROOT / "scripts/symmetric_pair_head.py"):
        pins[str(path.relative_to(ROOT))] = sha(path)
    if sha(PLAN) != PLAN_SHA:
        raise ValueError("frozen role-free experiment plan changed")
    pins[str(PLAN.relative_to(ROOT))] = PLAN_SHA
    if sha(PRIOR_PROBE) != PRIOR_PROBE_SHA:
        raise ValueError("prior resource qualification changed")
    pins[str(PRIOR_PROBE.relative_to(ROOT))] = PRIOR_PROBE_SHA
    verify_pins(pins)
    return manifest, rows, pairs, dict(sorted(pins.items()))


def verify_model_lineage(pins):
    if sha(MODEL / "SOURCE.json") != MODEL_SOURCE_SHA:
        raise ValueError("fixed upstream provenance changed")
    source = json.loads((MODEL / "SOURCE.json").read_bytes())
    if (source.get("repository") != "klue/roberta-small"
            or source.get("files", {}).get("model.safetensors") != MODEL_WEIGHT_SHA):
        raise ValueError("fixed upstream model lineage differs")
    for name, expected in source["files"].items():
        path = MODEL / name
        if sha(path) != expected:
            raise ValueError(f"upstream artifact changed: {name}")
        pins[str(path.relative_to(ROOT))] = expected
    pins[str((MODEL / "SOURCE.json").relative_to(ROOT))] = MODEL_SOURCE_SHA
    return source


def semantic_label(pair, expected):
    orientation = pair["canonical_orientation"]
    accepted = set(expected)
    a = orientation["candidate_a_text"] in accepted
    b = orientation["candidate_b_text"] in accepted
    if a and not b:
        return LabelState.A_BETTER
    if b and not a:
        return LabelState.B_BETTER
    return LabelState.TIE if a else LabelState.UNKNOWN


def alternative_margin(canonical_margin, current_is_a):
    if type(current_is_a) is not bool or not math.isfinite(canonical_margin):
        raise ValueError("finite canonical margin and current role required")
    return -orient_margin(canonical_margin, not current_is_a)


def choose_alternative(pair_indices, canonical_margins, pairs):
    if len(pair_indices) != len(canonical_margins):
        raise ValueError("complete pair margins required")
    scored = [(index, alternative_margin(float(margin),
               pairs[index]["canonical_orientation"]["current_is_a"]))
              for index, margin in zip(pair_indices, canonical_margins)]
    if not scored:
        return None, []
    maximum = max(score for _, score in scored)
    winners = [index for index, score in scored if score == maximum]
    chosen = winners[0] if maximum > RECIPE["positive_threshold"] and len(winners) == 1 else None
    return chosen, scored


def read_targets(indices, rows):
    wanted = set(indices)
    if any(type(index) is not int or not 0 <= index < len(rows) for index in wanted):
        raise ValueError("invalid target source index")
    result = {}
    roles = (("natural_fit", NATURAL_CASES, 0, 2048),
             ("challenge_fit", CHALLENGE_CASES, 2048, 1444))
    for role, path, offset, count in roles:
        local = {index - offset for index in wanted if offset <= index < offset + count}
        if not local:
            continue
        with path.open() as stream:
            for local_index, line in enumerate(stream):
                if local_index in local:
                    row = json.loads(line)
                    global_index = offset + local_index
                    if row["id"] != rows[global_index]["id"] or rows[global_index]["source_role"] != role:
                        raise ValueError("target source order differs")
                    result[global_index] = row["expected"]
                if local_index >= max(local):
                    break
    if set(result) != wanted:
        raise ValueError("target source rows missing")
    return result


def score_predictions(rows, predictions, targets):
    if len(rows) != len(predictions) or set(targets) != set(range(len(rows))):
        raise ValueError("all rows and targets are required for metrics")
    result = {"before_top1": 0, "top1": 0, "natural_before_top1": 0,
              "natural_top1": 0, "correction_ids": [], "natural_correction_ids": [],
              "regression_ids": []}
    for index, (row, prediction) in enumerate(zip(rows, predictions)):
        if prediction.get("row_index") != index or prediction.get("id") != row["id"]:
            raise ValueError("closed prediction order differs")
        expected = targets[index]
        before = row["current_result"] in expected
        after = prediction["chosen_text"] in expected
        result["before_top1"] += before
        result["top1"] += after
        if row["source_role"] == "natural_fit":
            result["natural_before_top1"] += before
            result["natural_top1"] += after
        if after and not before:
            result["correction_ids"].append(row["id"])
            if row["source_role"] == "natural_fit":
                result["natural_correction_ids"].append(row["id"])
        if before and not after:
            result["regression_ids"].append(row["id"])
    result["mechanism_gate_passed"] = (
        len(result["natural_correction_ids"]) >= 1 and not result["regression_ids"])
    return result


def _probe_positions(pairs, eligible_indices):
    ordered = sorted(eligible_indices, key=lambda index: (
        max(pairs[index]["token_lengths"]), index))
    if not ordered:
        raise ValueError("no eligible semantic pairs")
    return [ordered[index] for index in sorted(set(
        round(i * (len(ordered) - 1) / 15) for i in range(16)))]


def validate_runtime_lengths(attention_mask, pairs, indices):
    actual = attention_mask.sum(dim=1).tolist()
    expected = [length for index in indices for length in pairs[index]["token_lengths"]]
    if actual != expected or not actual or max(actual) > RECIPE["max_tokens"]:
        raise ValueError("runtime token lengths differ from qualified input")


def run(output=OUTPUT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {"completed": False, "decision_only": True,
              "generalization_claim_allowed": False, "deployed": False,
              "production_changed": False, "promotion_allowed": False}
    try:
        manifest, rows, pairs, pins = load_inputs()
        model_source = verify_model_lineage(pins)
        write_json(output / "configuration.json", {
            "recipe": RECIPE, "pins": dict(sorted(pins.items())),
            "base_model_source": model_source,
            "input_manifest_sha256": INPUT_MANIFEST_SHA,
            "decision_only": True, "generalization_claim_allowed": False,
            "deployed": False,
        })
        import torch
        from torch.nn import functional as F
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(RECIPE["threads"])
        torch.manual_seed(RECIPE["seed"])
        tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        eligible_indices = [p["index"] for p in pairs if p["eligible_le_512"]]
        sample_indices = _probe_positions(pairs, eligible_indices)
        model = AutoModel.from_pretrained(MODEL, local_files_only=True, add_pooling_layer=False)
        model.eval().requires_grad_(False)
        if (model.config.hidden_size != 768 or model.config.num_hidden_layers != 6
                or any(parameter.requires_grad for parameter in model.parameters())):
            raise ValueError("frozen upstream 6-layer 768-dimensional encoder required")

        def embed(indices):
            sequences = [view for index in indices for view in pairs[index]["neural_views"]]
            encoded = tokenizer(sequences, padding=True, truncation=False,
                                return_token_type_ids=False, return_tensors="pt")
            validate_runtime_lengths(encoded["attention_mask"], pairs, indices)
            with torch.inference_mode():
                hidden = model(**encoded, return_dict=True).last_hidden_state[:, 0].float()
                delta = F.normalize((hidden[0::2] - hidden[1::2]) * 0.5, p=2, dim=1)
            if not bool(torch.isfinite(delta).all()):
                raise ValueError("non-finite frozen encoder feature")
            return delta.clone()

        tick = time.monotonic()
        warm = embed(sample_indices[:1])
        for offset in range(0, len(sample_indices), RECIPE["batch_size"]):
            embed(sample_indices[offset:offset + RECIPE["batch_size"]])
            require_resources(started)
        probe_seconds = time.monotonic() - tick
        projected = probe_seconds / len(sample_indices) * len(eligible_indices) * 1.5 + 120
        write_json(output / "resource-probe.json", {
            "sample_pair_indices": sample_indices, "seconds": probe_seconds,
            "estimated_total_seconds": projected, "rss_peak_bytes": rss_bytes(),
            "eligible_pairs": len(eligible_indices), "is_estimate": True,
        })
        if projected > RECIPE["max_seconds"]:
            raise RuntimeError("projected encoding exceeds fixed time budget")

        features = torch.zeros(len(pairs), 768)
        eligible = torch.zeros(len(pairs), dtype=torch.bool)
        ordered = sorted(eligible_indices, key=lambda index: (
            max(pairs[index]["token_lengths"]), index))
        with (output / "embedding-progress.jsonl").open("xb") as progress:
            for offset in range(0, len(ordered), RECIPE["batch_size"]):
                indices = ordered[offset:offset + RECIPE["batch_size"]]
                features[indices] = embed(indices)
                eligible[indices] = True
                require_resources(started)
                if offset % 128 == 0:
                    event = {"embedded_pairs": offset + len(indices),
                             "eligible_pairs": len(ordered),
                             "elapsed_seconds": time.monotonic() - started,
                             "peak_rss_bytes": rss_bytes()}
                    progress.write(canonical(event) + b"\n")
                    progress.flush()
                    print(json.dumps(event), flush=True)
        if not torch.equal(warm, embed(sample_indices[:1])):
            raise ValueError("frozen encoder replay differs")
        torch.save({"features": features, "eligible": eligible,
                    "pair_source_indices": torch.tensor([p["row_index"] for p in pairs])},
                   output / "encoder-features.pt")
        del model

        pair_folds = torch.tensor([pair["fold"] for pair in pairs])
        pair_rows = torch.tensor([pair["row_index"] for pair in pairs])
        canonical_margins = torch.full((len(pairs),), float("nan"))
        fold_reports = []
        for fold in range(RECIPE["folds"]):
            train_mask = eligible & (pair_folds != fold)
            held_mask = eligible & (pair_folds == fold)
            train_rows = sorted(set(pair_rows[train_mask].tolist()))
            held_rows = sorted(set(pair_rows[held_mask].tolist()))
            if not train_rows or not held_rows:
                raise ValueError("each fold requires train and held source rows")
            if ({rows[index]["component"] for index in train_rows}
                    & {rows[index]["component"] for index in held_rows}):
                raise ValueError("held component leaked into optimizer source rows")
            targets = read_targets(train_rows, rows)
            train_indices = train_mask.nonzero().flatten().tolist()
            labels = [semantic_label(pairs[index], targets[pairs[index]["row_index"]])
                      for index in train_indices]
            del targets
            informative = sum(label in (LabelState.A_BETTER, LabelState.B_BETTER)
                              for label in labels)
            if not informative:
                raise ValueError("training fold has no informative semantic pairs")
            head = SymmetricPairHead(768)
            if (sum(parameter.numel() for parameter in head.parameters()) != 768
                    or any(bool(parameter.detach().count_nonzero())
                           for parameter in head.parameters())):
                raise ValueError("exactly 768 zero-initialized trainable scalars required")
            optimizer = torch.optim.AdamW(head.parameters(), lr=RECIPE["learning_rate"],
                                          weight_decay=RECIPE["weight_decay"])
            losses = []
            train_tensor = torch.tensor(train_indices)
            for _ in range(RECIPE["epochs"]):
                optimizer.zero_grad(set_to_none=True)
                margins = head.score_features(features.index_select(0, train_tensor))
                loss = semantic_pair_loss(margins, labels,
                                          pair_rows.index_select(0, train_tensor))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0,
                                               error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
                require_resources(started)
            held_indices = held_mask.nonzero().flatten()
            with torch.no_grad():
                held_features = features.index_select(0, held_indices)
                margins = head.score_features(held_features)
                if not torch.equal(margins, -head.score_features(-held_features)):
                    raise ValueError("canonical swap does not negate learned margins")
                canonical_margins[held_indices] = margins
            checkpoint = output / f"fold-{fold}.pt"
            torch.save({"state_dict": head.state_dict(), "fold": fold,
                        "train_source_indices": train_rows,
                        "held_source_indices": held_rows}, checkpoint)
            restored = SymmetricPairHead(768)
            restored.load_state_dict(torch.load(checkpoint, weights_only=True)["state_dict"])
            with torch.no_grad():
                if not torch.equal(margins, restored.score_features(held_features)):
                    raise ValueError("reloaded semantic head differs")
            fold_reports.append({
                "fold": fold, "train_source_indices": train_rows,
                "held_source_indices": held_rows, "train_pairs": len(train_indices),
                "held_pairs": int(held_mask.sum()), "informative_train_pairs": informative,
                "label_counts": dict(Counter(label.name for label in labels)),
                "first_loss": losses[0], "last_loss": losses[-1],
                "optimizer_updates": RECIPE["epochs"], "trainable_parameters": 768,
                "checkpoint_sha256": sha(checkpoint),
            })
            print(json.dumps({"fold_completed": fold,
                              "informative_train_pairs": informative,
                              "last_loss": losses[-1]}), flush=True)
        if bool(torch.isnan(canonical_margins[eligible]).any()):
            raise ValueError("missing held-fold semantic margins")

        predictions = []
        for row in rows:
            indices = [index for index in row["pair_indices"] if eligible[index]]
            margins = [float(canonical_margins[index]) for index in indices]
            chosen, scored = choose_alternative(indices, margins, pairs)
            predictions.append({
                "row_index": row["index"], "id": row["id"], "source_role": row["source_role"],
                "fold": row["fold"], "component": row["component"],
                "current_text": row["current_result"],
                "chosen_text": row["current_result"] if chosen is None else pairs[chosen]["alternative"]["text"],
                "chosen_pair_index": chosen, "eligible_pair_indices": indices,
                "canonical_margins": margins,
                "alternative_margins": [score for _, score in scored],
            })
        write_jsonl(output / "source-predictions.jsonl", predictions)
        write_json(output / "folds.json", fold_reports)
        prediction_sha = sha(output / "source-predictions.jsonl")
        targets = read_targets(range(len(rows)), rows)
        metrics = score_predictions(rows, predictions, targets)
        del targets
        verify_pins(pins)
        require_resources(started)
        result.update({
            "completed": True, "rows": len(rows), "pairs": len(pairs),
            "eligible_pairs": int(eligible.sum()),
            "skipped_pairs": int((~eligible).sum()),
            "rows_without_eligible_pairs": sum(not p["eligible_pair_indices"] for p in predictions),
            "switches": sum(p["chosen_pair_index"] is not None for p in predictions),
            **metrics, "source_hashes": dict(sorted(pins.items())),
            "input_manifest_sha256": INPUT_MANIFEST_SHA,
            "input_rows_sha256": manifest["rows_sha256"],
            "input_pairs_sha256": manifest["pairs_sha256"],
            "features_sha256": sha(output / "encoder-features.pt"),
            "predictions_sha256": prediction_sha,
            "folds_sha256": sha(output / "folds.json"),
            "base_model_source": model_source,
            "recipe": RECIPE,
            "note": "TRAIN-only out-of-fold decision diagnostic; no generalization or deployment claim.",
        })
    except Exception as error:
        result["failure"] = f"{type(error).__name__}: {error}"
    result.update({"elapsed_seconds": time.monotonic() - started,
                   "peak_rss_bytes": rss_bytes()})
    write_json(output / "report.json", result)
    print(json.dumps({
        "completed": result["completed"], "rows": result.get("rows"),
        "pairs": result.get("pairs"), "eligible_pairs": result.get("eligible_pairs"),
        "before_top1": result.get("before_top1"), "top1": result.get("top1"),
        "corrections": len(result.get("correction_ids", [])),
        "regressions": len(result.get("regression_ids", [])),
        "mechanism_gate_passed": result.get("mechanism_gate_passed"),
        "failure": result.get("failure"),
    }, ensure_ascii=False), flush=True)
    return result


if __name__ == "__main__":
    outcome = run()
    raise SystemExit(0 if outcome["completed"] else 1)
