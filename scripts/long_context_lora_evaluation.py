"""Source-only finite selection; labels are accessed only by final scoring."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

from scripts.final_choice_policy import proposal_allowed, reorder_candidates
from scripts.consensus_context_policy import prompt_choices
from scripts.prepare_long_context_lora_data import tokenize_prompt, validate_digit_boundary
from scripts.probe_consensus_context_resources import prompt_record, system_message


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def prepare_diagnostic(items, tokenizer):
    system = system_message()
    records, identities = [], set()
    for item in items:
        record = prompt_record(item, system)
        identity = (record["role"], record["id"])
        if (not all(isinstance(part, str) and part for part in identity)
                or record["id"] in identities):
            raise ValueError("diagnostic identity missing or duplicated")
        identities.add(record["id"])
        tokens = tokenize_prompt(tokenizer, record["messages"])
        reason = ("insufficient_choices" if len(record["choices"]) < 2 else
                  "prompt_over_1024" if len(tokens) > 1024 else None)
        digit_ids = (validate_digit_boundary(tokenizer, record["messages"], tokens,
                     len(record["choices"])) if reason is None else [])
        raw = record["raw_bypass_response"]
        if (raw.get("result") != record["source"] or not raw.get("candidates")
                or raw["candidates"][0].get("text") != record["source"]):
            raise ValueError("raw bypass is not the original input")
        records.append({**record, "tokens": tokens, "digit_ids": digit_ids,
                        "eligible": reason is None, "fallback_reason": reason,
                        "_candidate_snapshot": canonical(record["candidates"]),
                        "_raw_snapshot": canonical(raw)})
    return records


def apply_logits(record, logits_or_None):
    candidates, raw = record["candidates"], record["raw_bypass_response"]
    if (canonical(candidates) != record["_candidate_snapshot"]
            or canonical(raw) != record["_raw_snapshot"]):
        raise ValueError("candidate or raw snapshot changed during inference")
    chosen, proposed, logits = 0, None, None
    if logits_or_None is None:
        reason = record["fallback_reason"] or "parent_baseline"
    else:
        logits = list(logits_or_None)
        if (not record["eligible"] or len(logits) != len(record["choices"])
                or not 2 <= len(logits) <= 5
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in logits)):
            raise ValueError("logits must match all eligible choices and be finite")
        position = max(range(len(logits)), key=lambda index: logits[index])
        proposed = record["choices"][position][0]
        if proposed == 0:
            reason = "parent_selected"
        elif proposal_allowed(record["source"], candidates[0], candidates[proposed]):
            chosen, reason = proposed, "lora_proposal"
        else:
            reason = "protected_or_unaligned"
    ordered = reorder_candidates(candidates, chosen)
    return {"id": record["id"], "role": record["role"], "source": record["source"],
            "context": record["context"], "chosen_index": chosen,
            "proposed_index": proposed, "reason": reason, "logits": logits,
            "choices": deepcopy(record["choices"]),
            "result": ordered[0]["text"], "candidates": ordered,
            "raw_bypass_response": deepcopy(raw),
            "raw_canonical_json_before": record["_raw_snapshot"],
            "raw_canonical_json_after": canonical(raw),
            "raw_sha256_before": sha(record["_raw_snapshot"]),
            "raw_sha256_after": sha(canonical(raw))}


def score_predictions(items, predictions):
    """Score completed predictions after releasing models; retain every row."""
    if not items or len(items) != len(predictions):
        raise ValueError("diagnostic rows must all be accounted")
    metrics, strata, identities = {}, {}, set()
    for item, prediction in zip(items, predictions):
        row, role = item["row"], item["role"]
        identity = (role, row["id"])
        if row["id"] in identities or identity != (prediction["role"], prediction["id"]):
            raise ValueError("diagnostic prediction identity mismatch")
        identities.add(row["id"])
        candidates = row["candidates"]
        choices = prompt_choices(row["input"], row["context"], candidates)
        if canonical(prediction["choices"]) != canonical(choices):
            raise ValueError("saved prompt-to-candidate mapping differs")
        logits = prediction["logits"]
        proposed = None
        expected_chosen = 0
        if logits is not None:
            if (not 2 <= len(choices) <= 5 or len(logits) != len(choices)
                    or any(type(x) not in (int, float) or not math.isfinite(x) for x in logits)):
                raise ValueError("saved logits are not finite or complete")
            proposed = choices[max(range(len(logits)), key=lambda index: logits[index])][0]
            if proposed == 0 or proposal_allowed(row["input"], candidates[0], candidates[proposed]):
                expected_chosen = proposed
        if prediction["proposed_index"] != proposed or prediction["chosen_index"] != expected_chosen:
            raise ValueError("saved selection differs from finite logits and policy")
        chosen = prediction["chosen_index"]
        ordered = reorder_candidates(candidates, chosen)
        raw_json = canonical(row["raw_bypass_response"])
        if (prediction["source"] != row["input"] or prediction["context"] != row["context"]
                or row["result"] != candidates[0]["text"]
                or canonical(prediction["candidates"]) != canonical(ordered)
                or prediction["result"] != ordered[0]["text"]
                or (chosen != 0 and not proposal_allowed(row["input"], candidates[0], candidates[chosen]))):
            raise ValueError("candidate preservation or allowed selection invariant failed")
        if (canonical(prediction["raw_bypass_response"]) != raw_json
                or prediction["raw_canonical_json_before"] != raw_json
                or prediction["raw_canonical_json_after"] != raw_json
                or prediction["raw_sha256_before"] != sha(raw_json)
                or prediction["raw_sha256_after"] != sha(raw_json)):
            raise ValueError("raw preservation invariant failed")
        expected = row["expected"]
        if not isinstance(expected, list) or not expected or any(not isinstance(x, str) for x in expected):
            raise ValueError("allowed answers must be a nonempty string list")
        before, after = row["result"] in expected, prediction["result"] in expected
        before5 = any(c["text"] in expected for c in candidates[:5])
        after5 = any(c["text"] in expected for c in ordered[:5])
        stratum = item.get("stratum")
        if not isinstance(stratum, str) or not stratum:
            raise ValueError("diagnostic stratum is missing")
        for collection, key in ((metrics, "all"), (metrics, role), (strata, stratum)):
            counts = collection.setdefault(key, dict.fromkeys(("rows", "before_top1", "top1",
                "before_top5", "top5", "corrections", "regressions", "actions", "fallbacks"), 0))
            increments = (1, before, after, before5, after5, after and not before,
                          before and not after, chosen != 0, prediction["logits"] is None)
            for name, value in zip(counts, increments):
                counts[name] += int(value)
    all_rows = metrics["all"]
    net = all_rows["top1"] - all_rows["before_top1"]
    return {"metrics": metrics, "strata": strata, "gate": {"passed": len(items) == 640 and net >= 5,
            "net_top1_gain": net, "corrections": all_rows["corrections"],
            "regressions": all_rows["regressions"], "all_rows_accounted": True,
            "zero_top5_losses": all_rows["top5"] == all_rows["before_top5"],
            "raw_preserved": True, "candidates_preserved": True}}
