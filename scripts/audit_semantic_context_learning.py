"""TRAIN-only context interventions on two existing semantic checkpoints."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hanglish.core.neural_converter import NeuralConverter, file_sha256
from hanglish.models.semantic_ranker import SemanticRanker, context_neighborhood
from scripts.evaluate_ensemble_regression import rss_bytes, validate_response, verify_pins

DATA = ROOT / "data/challenge/context-augmentation/train.jsonl"
DATA_SHA = "001388eb3408d0cb9ad782f9dc67a47cd3893a4070b5744d4b44cf964cc81a31"
CHECKPOINTS = {"parent": ROOT / "checkpoints/semantic_ranker",
               "augmented": ROOT / "checkpoints/semantic_ranker_augmented_nll"}


def interventions(rows):
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate TRAIN IDs")
    result = []
    for row in rows:
        if row["split"] != "train" or len(row["expected"]) != 1:
            raise ValueError("expected decisive TRAIN rows")
        donors = [r for r in rows if r["input"] == row["input"] and r["expected"] != row["expected"]]
        if not donors:
            raise ValueError("missing contrastive TRAIN context")
        donors.sort(key=lambda r: hashlib.sha256((row["id"] + "\0" + r["id"]).encode()).hexdigest())
        donor = donors[0]
        result.append({"id": row["id"], "input": row["input"], "expected": row["expected"],
                       "context": row["context"], "donor_id": donor["id"],
                       "donor_context": donor["context"], "donor_expected": donor["expected"]})
    return result


class Recorder:
    def __init__(self, ranker):
        self.ranker, self.calls = ranker, []

    def score_pairs(self, contexts, candidates):
        scores = self.ranker.score_pairs(contexts, candidates)
        self.calls.append({"contexts": list(contexts), "candidates": list(candidates), "scores": list(scores)})
        return scores


def summarize(captures):
    metrics = {}
    for arm in CHECKPOINTS:
        rows = [r for r in captures if r["arm"] == arm]
        counts = {"rows": len(rows), "own_top1": 0, "donor_top1": 0,
                  "both_contexts_correct": 0, "own_differs_from_blank": 0,
                  "own_logits_only_top1": 0, "bias_fixes": 0, "bias_losses": 0,
                  "own_target_missing_scored_candidates": 0, "own_context_crop_changed": 0,
                  "own_token_budget_exceeded": 0}
        by_target = {}
        for row in rows:
            own = row["own"]["response"]["result"] in row["expected"]
            donor = row["donor"]["response"]["result"] in row["donor_expected"]
            call = row["own"]["calls"][0]
            choice = min(range(len(call["scores"])), key=lambda i: (-call["scores"][i], call["candidates"][i]))
            logits_correct = call["candidates"][choice] in row["expected"]
            counts["own_top1"] += own
            counts["donor_top1"] += donor
            counts["both_contexts_correct"] += own and donor
            counts["own_differs_from_blank"] += row["own"]["response"]["result"] != row["blank"]["response"]["result"]
            counts["own_logits_only_top1"] += logits_correct
            counts["bias_fixes"] += own and not logits_correct
            counts["bias_losses"] += logits_correct and not own
            counts["own_target_missing_scored_candidates"] += not any(t in row["expected"] for t in call["candidates"])
            counts["own_context_crop_changed"] += row["own"]["context_crop_changed"]
            counts["own_token_budget_exceeded"] += row["own"]["token_budget_exceeded"]
            target = row["expected"][0]
            entry = by_target.setdefault(target, {"rows": 0, "correct": 0})
            entry["rows"] += 1
            entry["correct"] += own
        metrics[arm] = {**counts, "by_target": by_target}
    return metrics


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    def guard():
        if time.monotonic() - started > 300 or rss_bytes() > 2 * 1024**3:
            raise RuntimeError("audit exceeded300seconds/2GiB")
    try:
        if file_sha256(DATA) != DATA_SHA:
            raise ValueError("changed TRAIN augmentation")
        rows = interventions([json.loads(line) for line in DATA.read_text().splitlines()])
        if len(rows) != 240:
            raise ValueError("expected TRAIN240")
        files = [DATA, Path(__file__), ROOT / "scripts/evaluate_ensemble_regression.py"]
        files.extend((ROOT / "hanglish/core").glob("*.py"))
        files.extend((ROOT / "hanglish/models").glob("*.py"))
        files.extend(ROOT / name for name in ("data/challenge/train_lexicon.json",
            "data/morphology_lexicon.json", "data/workbench_lexicon.json",
            "data/common_english.txt", "data/it_terms.txt", "data/external/mecab-ko-dic/SOURCE.json"))
        for path in CHECKPOINTS.values():
            files.extend(p for p in path.iterdir() if p.is_file())
        pins = {str(p.relative_to(ROOT)): file_sha256(p) for p in files}
        verify_pins(pins)
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        captures = []
        engine = NeuralConverter(ranker_kind="semantic")
        engine.load_models()
        production_version = engine.version
        generator_fingerprint = engine.generator.external_fingerprint
        # Keep the production transducer/generator; replace only the semantic model.
        phonetic = {k: v for k, v in engine.model_artifacts.items() if "phonetic_transducer" in k}
        pins.update(phonetic)
        with (output / "capture.jsonl").open("x") as stream:
            for arm, checkpoint in CHECKPOINTS.items():
                ranker = SemanticRanker.load(checkpoint)
                recorder = Recorder(ranker)
                engine.ranker = recorder
                engine.model_artifacts = {**phonetic, **{str(p.relative_to(ROOT)): file_sha256(p)
                    for p in checkpoint.iterdir() if p.is_file()}}
                digest = hashlib.sha256(json.dumps(engine.model_artifacts, sort_keys=True).encode()).hexdigest()
                engine.version = "audit-semantic-" + arm + "-" + digest[:12]
                for row in rows:
                    record = {**row, "arm": arm, "options": [vars(o) for o in engine.word_options(row["input"], True)]}
                    for mode, context in (("own", row["context"]), ("blank", ""), ("donor", row["donor_context"])):
                        recorder.calls.clear()
                        response = engine.convert(row["input"], context)
                        validate_response(response, row["input"], context)
                        if len(recorder.calls) != 1:
                            raise ValueError("expected one actual word-ranker call per single input")
                        call = recorder.calls[0]
                        neighborhoods = [context_neighborhood(c, t) for c, t in zip(call["contexts"], call["candidates"])]
                        changed = any(context and context not in n for n in neighborhoods)
                        lengths = [len(ids) for ids in ranker.tokenizer(call["candidates"], neighborhoods,
                                   truncation=False, padding=False)["input_ids"]]
                        record[mode] = {"response": response, "calls": deepcopy(recorder.calls),
                                        "neighborhoods": neighborhoods, "context_crop_changed": changed,
                                        "untruncated_token_lengths": lengths,
                                        "token_budget_exceeded": max(lengths) > ranker.max_length}
                    captures.append(record)
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    guard()
                print(json.dumps({"arm": arm, "rows": len(rows), "elapsed_s": time.monotonic() - started}), flush=True)
        verify_pins(pins)
        report = {"schema": "semantic-context-learning-audit-v1", "completed": True,
                  "training_run": False, "production_changed": False, "new_generalization": False,
                  "production_version": production_version, "generator_fingerprint": generator_fingerprint,
                  "scope": "TRAIN240; donor contexts intentionally selected using TRAIN sense labels;1440 actual single-word conversions",
                  "pins": pins, "capture_sha256": file_sha256(output / "capture.jsonl"),
                  "metrics": summarize(captures), "elapsed_s": time.monotonic() - started,
                  "peak_rss_bytes": rss_bytes()}
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({k: report[k] for k in ("completed", "metrics", "elapsed_s", "peak_rss_bytes")}, ensure_ascii=False), flush=True)
    except BaseException as exc:
        (output / "failure.json").write_text(json.dumps({"completed": False, "error": repr(exc)}, indent=2))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args().output)
