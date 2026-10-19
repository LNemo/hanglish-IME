"""One fixed, grouped out-of-fold FIT experiment; never changes runtime output.

The boundary union was already diagnosed on this FIT set. Consequently these
are mechanism results, not an untouched generalization estimate. The upstream
encoder is frozen and only a zero-initialized comparison head is optimized.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.final_choice_policy import proposal_allowed

CAPTURE = ROOT / ".omx/artifacts/lexical-boundary-runtime-fit-v1"
FIT = ROOT / "data/natural_context/final-choice-v1/natural_fit"
MODEL = ROOT / "data/external/klue-roberta-small"
OUTPUT = ROOT / ".omx/artifacts/symmetric-pair-selector-fit-v1"
REPORT_SHA = "5c8d1f053960f870b747936a2b0e33e3ee7296198d5f5557c0d7407eda091cd7"
FIT_MANIFEST_SHA = "fc22b1635cc496ed70aa757750944726d583d640f2d9f3a598e380f3e262a1c9"
MODEL_WEIGHT_SHA = "737e7dbcb9f159d4719a06095268836a2634a673ac925de96c13c9d0bf4d895b"
MODEL_SOURCE_SHA = "5166197923917d229b81d58f0e3561de3b141222d8597b7d0313bab4bb0b7d9c"
RECIPE = {"seed": 23, "folds": 4, "epochs": 100, "learning_rate": .01,
          "weight_decay": .01, "context_chars": 128, "max_tokens": 512,
          "batch_size": 4, "threads": 2, "max_seconds": 1800,
          "max_rss_bytes": 3 * 1024**3, "fold_salt": "symmetric-pair-fit-v1",
          "positive_threshold": 0., "encoder": "upstream klue/roberta-small",
          "features": "L2-normalized half difference of frozen CLS in two orders"}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_bytes(canonical(value) + b"\n")


def normalized(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def assign_folds(rows):
    """Union documents and identical sources before deterministic component split."""
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, row in enumerate(rows):
        for key in (("group", row["group_id"]),
                    ("source", normalized(row["input"]), normalized(row["context"]))):
            if key in seen:
                parent[find(i)] = find(seen[key])
            seen[key] = i
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(find(i), []).append(row["id"])
    keys = {i: hashlib.sha256(canonical(sorted(ids))).hexdigest() for i, ids in groups.items()}
    return [{"component": keys[find(i)], "fold": int(hashlib.sha256(
        (RECIPE["fold_salt"] + keys[find(i)]).encode()).hexdigest(), 16) % RECIPE["folds"]}
        for i in range(len(rows))]


def candidate_union(capture):
    """Deduplicate by spelling, retaining each origin's complete candidate object."""
    found = {}
    for origin in ("parent", "child"):
        for rank, candidate in enumerate(capture[origin]["candidates"]):
            score = candidate["score"]
            if score is None:
                continue
            if type(score) not in (int, float) or not math.isfinite(score):
                raise ValueError("finite original candidate score required")
            entry = found.setdefault(candidate["text"], {"text": candidate["text"], "origins": {}})
            entry["origins"][origin] = {"rank": rank, "score": score,
                "sha256": hashlib.sha256(canonical(candidate)).hexdigest(),
                "object": deepcopy(candidate)}
    current = capture["parent"]["candidates"][0]
    if current["score"] is None:
        return None, []
    current_entry = found[current["text"]]
    alternatives = []
    for text, entry in found.items():
        if text == current["text"]:
            continue
        preferred = entry["origins"].get("parent", entry["origins"].get("child"))
        if proposal_allowed(capture["input"], current, preferred["object"]):
            alternatives.append(entry)
    return current_entry, alternatives


def candidate_payload(entry, role):
    return {"policy_role": role, "text": entry["text"], "origins": {
        origin: {"rank": data["rank"], "score": data["score"],
                 "kinds": [s["kind"] for s in data["object"]["segments"]]}
        for origin, data in entry["origins"].items()}}


def pair_views(row, current, alternative):
    """Only source fields; candidate role/rank metadata move with the candidate."""
    context = row["context"]
    if len(context) > RECIPE["context_chars"]:
        half = RECIPE["context_chars"] // 2
        context = context[:half] + "[…중략…]" + context[-half:]
    common = {"roman_input": row["input"], "context": context}
    a, b = candidate_payload(alternative, "alternative"), candidate_payload(current, "current")
    return (canonical({**common, "first": a, "second": b}).decode(),
            canonical({**common, "first": b, "second": a}).decode())


def relation(current, alternative, expected):
    a, b = alternative in expected, current in expected
    return int(a) - int(b)


def choose_pair(pair_indices, scores):
    if len(pair_indices) != len(scores) or any(not math.isfinite(s) for s in scores):
        raise ValueError("complete finite pair scores required")
    if not scores:
        return None
    best = max(range(len(scores)), key=lambda i: (scores[i], -pair_indices[i]))
    if scores.count(scores[best]) > 1:
        return None
    return pair_indices[best] if scores[best] > RECIPE["positive_threshold"] else None


def rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def require_resources(start):
    if rss_bytes() > RECIPE["max_rss_bytes"] or time.monotonic() - start > RECIPE["max_seconds"]:
        raise RuntimeError("fixed experiment resource limit exceeded")


def prepare(output):
    report = json.loads((CAPTURE / "report.json").read_bytes())
    if sha(CAPTURE / "report.json") != REPORT_SHA or not report["completed"]:
        raise ValueError("complete fixed boundary capture required")
    if sha(CAPTURE / "captures.jsonl") != report["captures_sha256"]:
        raise ValueError("boundary capture changed")
    if sha(FIT / "manifest.json") != FIT_MANIFEST_SHA:
        raise ValueError("fixed FIT manifest changed")
    manifest = json.loads((FIT / "manifest.json").read_bytes())
    if sha(FIT / "cases.jsonl") != manifest["cases_sha256"]:
        raise ValueError("FIT capture changed")
    # Parse source metadata only into the modeling view. Targets are used later,
    # exclusively for training folds and the closed out-of-fold metric pass.
    with (FIT / "cases.jsonl").open() as stream:
        rows = [{key: row[key] for key in ("id", "group_id", "input", "context")}
                for row in (json.loads(line) for _, line in zip(range(512), stream))]
    folds = assign_folds(rows)
    for row, fold in zip(rows, folds):
        row.update(fold)
    pairs = []
    with (CAPTURE / "captures.jsonl").open() as stream:
        for i, line in enumerate(stream):
            capture = json.loads(line)
            row = rows[i]
            if any(row[k] != capture[k] for k in ("id", "input", "context")):
                raise ValueError("capture source order differs")
            if capture["parent"]["result"] != capture["parent"]["candidates"][0]["text"]:
                raise ValueError("captured parent result differs from first candidate")
            current, alternatives = candidate_union(capture)
            row["parent_text"] = capture["parent"]["result"]
            row["pair_indices"] = []
            for alternative in alternatives:
                index = len(pairs)
                views = pair_views(row, current, alternative)
                pairs.append({"index": index, "row_index": i, "row_id": row["id"],
                    "fold": row["fold"], "component": row["component"],
                    "current": current, "alternative": alternative, "views": list(views)})
                row["pair_indices"].append(index)
    if len(rows) != 512 or i + 1 != 512:
        raise ValueError("all fixed FIT512 rows required")
    write_json(output / "sources.json", rows)
    with (output / "pairs.jsonl").open("xb") as stream:
        for pair in pairs:
            stream.write(canonical(pair) + b"\n")
    pins = {str(path.relative_to(ROOT)): sha(path) for path in (
        CAPTURE / "report.json", CAPTURE / "captures.jsonl", FIT / "manifest.json",
        FIT / "cases.jsonl", Path(__file__), ROOT / "scripts/symmetric_pair_head.py",
        ROOT / "scripts/final_choice_policy.py", ROOT / "hanglish/core/phonology.py",
        MODEL / "SOURCE.json")}
    if sha(MODEL / "SOURCE.json") != MODEL_SOURCE_SHA:
        raise ValueError("fixed upstream provenance changed")
    provenance = json.loads((MODEL / "SOURCE.json").read_bytes())
    if (provenance["repository"] != "klue/roberta-small"
            or provenance["files"]["model.safetensors"] != MODEL_WEIGHT_SHA):
        raise ValueError("upstream base model lineage differs")
    for name, expected in provenance["files"].items():
        path = MODEL / name
        if sha(path) != expected:
            raise ValueError(f"upstream artifact changed: {name}")
        pins[str(path.relative_to(ROOT))] = expected
    write_json(output / "configuration.json", {"recipe": RECIPE, "pins": pins,
        "base_model_source": provenance, "scope": "exposed FIT mechanism diagnostic",
        "rows": len(rows), "pairs": len(pairs), "production_changed": False,
        "decision_only": True, "generalization_claim_allowed": False})
    return rows, pairs, pins


def read_targets(indices):
    wanted = set(indices)
    result = {}
    with (FIT / "cases.jsonl").open() as stream:
        for i, line in enumerate(stream):
            if i >= 512:
                break
            if i in wanted:
                result[i] = json.loads(line)["expected"]
    if set(result) != wanted:
        raise ValueError("training/evaluation source rows missing")
    return result


def run(output=OUTPUT):
    output.mkdir(exist_ok=False, parents=True)
    started = time.monotonic()
    result = {"completed": False, "production_changed": False, "decision_only": True,
              "generalization_claim_allowed": False, "promotion_allowed": False}
    try:
        rows, pairs, pins = prepare(output)
        import torch
        from torch.nn import functional as F
        from transformers import AutoModel, AutoTokenizer
        from scripts.symmetric_pair_head import SymmetricPairHead, relation_loss
        torch.set_num_threads(RECIPE["threads"])
        torch.manual_seed(RECIPE["seed"])
        tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        encoded = []
        for pair in pairs:
            views = tokenizer(pair["views"], truncation=False, padding=False,
                              return_token_type_ids=False)
            token_ids = views["input_ids"]
            pair["token_lengths"] = [len(v) for v in token_ids]
            if max(pair["token_lengths"]) <= RECIPE["max_tokens"]:
                encoded.append((pair["index"], token_ids))
        if not encoded:
            raise ValueError("no complete candidate pairs fit the encoder")
        write_json(output / "token_lengths.json", [{"index": p["index"], "lengths": p["token_lengths"]} for p in pairs])
        model = AutoModel.from_pretrained(MODEL, local_files_only=True, add_pooling_layer=False)
        model.eval().requires_grad_(False)
        if (model.config.hidden_size != 768 or model.config.num_hidden_layers != 6
                or any(p.requires_grad for p in model.parameters())):
            raise ValueError("frozen upstream 768-dimensional encoder required")
        require_resources(started)

        def embed(batch):
            ids = [seq for _, pair in batch for seq in pair]
            tensors = tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt")
            with torch.inference_mode():
                h = model(**tensors, return_dict=True).last_hidden_state[:, 0].float()
                delta = F.normalize((h[0::2] - h[1::2]) * .5, p=2, dim=1)
            if not bool(torch.isfinite(delta).all()):
                raise ValueError("non-finite frozen features")
            return delta.clone()

        ordered = sorted(encoded, key=lambda item: max(map(len, item[1])))
        sample_positions = sorted(set(round(i * (len(ordered)-1) / 15) for i in range(16)))
        sample = [ordered[i] for i in sample_positions]
        tick = time.monotonic()
        warm = embed(sample[:1])
        for offset in range(0, len(sample), RECIPE["batch_size"]):
            embed(sample[offset:offset+RECIPE["batch_size"]])
            require_resources(started)
        probe_seconds = time.monotonic() - tick
        projected = probe_seconds / len(sample) * len(encoded) * 1.5 + 120
        write_json(output / "resource-probe.json", {"sample_pair_indices": [p[0] for p in sample],
            "seconds": probe_seconds, "estimated_total_seconds": projected,
            "rss_peak_bytes": rss_bytes(), "pairs": len(pairs), "eligible_pairs": len(encoded),
            "rows": len(rows), "is_estimate": True})
        if projected > RECIPE["max_seconds"]:
            raise RuntimeError("projected encoding exceeds fixed time budget")
        features = torch.zeros(len(pairs), 768)
        eligible = torch.zeros(len(pairs), dtype=torch.bool)
        with (output / "embedding-progress.jsonl").open("x") as progress:
            for offset in range(0, len(ordered), RECIPE["batch_size"]):
                batch = ordered[offset:offset+RECIPE["batch_size"]]
                delta = embed(batch)
                index = [item[0] for item in batch]
                features[index] = delta
                eligible[index] = True
                require_resources(started)
                if offset % 128 == 0:
                    event = {"embedded": offset + len(batch), "total": len(ordered),
                             "seconds": time.monotonic()-started, "rss": rss_bytes()}
                    progress.write(canonical(event).decode()+"\n"); progress.flush()
                    print(json.dumps(event), flush=True)
        # Verify deterministic eval before freeing the encoder.
        replay = embed(sample[:1])
        if not torch.equal(warm, replay):
            raise ValueError("frozen encoder repeat is not deterministic")
        torch.save({"features": features, "eligible": eligible}, output / "features.pt")
        del model
        margins = torch.full((len(pairs),), float("nan"))
        pair_folds = torch.tensor([p["fold"] for p in pairs])
        pair_rows = torch.tensor([p["row_index"] for p in pairs])
        fold_reports = []
        for fold in range(RECIPE["folds"]):
            train = eligible & (pair_folds != fold)
            held = eligible & (pair_folds == fold)
            if not bool(train.any()) or not bool(held.any()):
                raise ValueError("each fold needs nonempty eligible training and held-out pairs")
            train_rows = sorted(set(pair_rows[train].tolist()))
            if {rows[i]["component"] for i in train_rows} & {p["component"] for p in pairs if p["fold"]==fold}:
                raise ValueError("held-out component leaked into training")
            targets = read_targets(train_rows)
            train_indices = train.nonzero().flatten().tolist()
            relations = torch.tensor([relation(pairs[i]["current"]["text"],
                pairs[i]["alternative"]["text"], targets[pairs[i]["row_index"]]) for i in train_indices])
            del targets
            head = SymmetricPairHead(768)
            if sum(p.numel() for p in head.parameters()) != 768 or any(
                    bool(p.detach().count_nonzero()) for p in head.parameters()):
                raise ValueError("exactly 768 zero-initialized trainable scalars required")
            optimizer = torch.optim.AdamW(head.parameters(), lr=RECIPE["learning_rate"],
                                          weight_decay=RECIPE["weight_decay"])
            losses = []
            for epoch in range(RECIPE["epochs"]):
                optimizer.zero_grad(set_to_none=True)
                loss = relation_loss(head.score_features(features[train]), relations, pair_rows[train])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
                require_resources(started)
            with torch.no_grad():
                predicted = head.score_features(features[held])
                reverse = head.score_features(-features[held])
                if not torch.equal(predicted, -reverse) or not bool(torch.isfinite(predicted).all()):
                    raise ValueError("learned head antisymmetry/finite check failed")
                margins[held] = predicted
            checkpoint = output / f"fold-{fold}.pt"
            torch.save(head.state_dict(), checkpoint)
            restored = SymmetricPairHead(768)
            restored.load_state_dict(torch.load(checkpoint, weights_only=True))
            with torch.no_grad():
                if not torch.equal(predicted, restored.score_features(features[held])):
                    raise ValueError("head reload differs")
            fold_reports.append({"fold": fold, "train_rows": train_rows,
                "train_pairs": int(train.sum()), "held_pairs": int(held.sum()),
                "relation_counts": dict(Counter(relations.tolist())),
                "first_loss": losses[0], "last_loss": losses[-1],
                "optimizer_updates": RECIPE["epochs"], "trainable_parameters": 768,
                "nonzero_parameters_after": sum(int(p.detach().count_nonzero()) for p in head.parameters()),
                "checkpoint_sha256": sha(checkpoint)})
            print(json.dumps({"fold_completed": fold, "last_loss": losses[-1]}), flush=True)
        predictions = []
        for i, row in enumerate(rows):
            indices = [j for j in row["pair_indices"] if eligible[j]]
            scores = [float(margins[j]) for j in indices]
            chosen = choose_pair(indices, scores)
            predictions.append({"row_index": i, "id": row["id"], "fold": row["fold"],
                "component": row["component"], "parent_text": row["parent_text"],
                "chosen_text": row["parent_text"] if chosen is None else pairs[chosen]["alternative"]["text"],
                "chosen_pair": chosen, "pair_indices": indices, "margins": scores})
        write_json(output / "predictions.json", predictions)
        write_json(output / "folds.json", fold_reports)
        # Only after the entire OOF prediction artifact is closed: metric labels.
        targets = read_targets(range(512))
        corrections, regressions = [], []
        before = after = 0
        for p in predictions:
            expected = targets[p["row_index"]]
            b, a = p["parent_text"] in expected, p["chosen_text"] in expected
            before += b; after += a
            if a and not b: corrections.append(p["id"])
            if b and not a: regressions.append(p["id"])
        for name, expected in pins.items():
            if sha(ROOT / name) != expected:
                raise ValueError(f"source changed during experiment: {name}")
        require_resources(started)
        result.update({"completed": True, "rows": 512, "pairs": len(pairs),
            "eligible_pairs": int(eligible.sum()), "skipped_pairs": int((~eligible).sum()),
            "rows_without_eligible_pairs": sum(not p["pair_indices"] for p in predictions),
            "before_top1": before, "top1": after, "correction_ids": corrections,
            "regression_ids": regressions, "switches": sum(p["chosen_pair"] is not None for p in predictions),
            "mechanism_gate_passed": after > before and not regressions,
            "source_hashes": pins, "predictions_sha256": sha(output / "predictions.json"),
            "features_sha256": sha(output / "features.pt"),
            "note": "Exposed FIT OOF mechanism only. No candidate-list integration or fresh evaluation."})
    except Exception as error:
        result["failure"] = f"{type(error).__name__}: {error}"
    result.update({"elapsed_seconds": time.monotonic()-started, "peak_rss_bytes": rss_bytes()})
    write_json(output / "report.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


if __name__ == "__main__":
    result = run()
    raise SystemExit(0 if result["completed"] else 1)
