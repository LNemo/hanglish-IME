"""Independently verify the closed source-only mixed-boundary union score capture."""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OLD_SCORE = ROOT / ".omx/artifacts/ungated-joint-sentence-score-fit-v1"
MIXED = ROOT / ".omx/artifacts/ungated-mixed-boundary-fit-v1"
PARENT = ROOT / ".omx/artifacts/ungated-phonetic-fit-v1"
JOINT = ROOT / ".omx/artifacts/ungated-joint-ranked-fit-v3"
CHECKPOINT = ROOT / "checkpoints/sentence_ranker"
PLAN = ROOT / ".omx/plans/ungated-mixed-union-fit-v1.md"
ROWS = 13
LIMITS = {"whole_seconds": 600.0, "row_seconds": 60.0,
          "rss_bytes": 2 * 1024**3, "swap_growth_bytes": 512 * 1024**2,
          "output_bytes": 128 * 1024**2}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def iter_jsonl(path: Path):
    with path.open() as stream:
        for line in stream:
            require(bool(line.strip()), f"blank row in {path.name}")
            yield json.loads(line)


def close(left, right, tolerance=1e-5):
    return (not isinstance(left, bool) and isinstance(left, (int, float))
            and not isinstance(right, bool) and isinstance(right, (int, float))
            and math.isfinite(left) and math.isfinite(right)
            and math.isclose(float(left), float(right), rel_tol=0.0,
                             abs_tol=tolerance))


def token_support(tokenizer, text):
    require(isinstance(text, str) and text, "candidate text differs")
    encoded = tokenizer(" ".join(text.strip().split()), add_special_tokens=False,
                        truncation=False)
    ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else None
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    require(isinstance(ids, list) and all(type(value) is int for value in ids),
            "untruncated tokenizer output differs")
    reserved = tokenizer.num_special_tokens_to_add(pair=True)
    require(type(reserved) is int and reserved == 3,
            "sentence tokenizer pair reserve differs")
    count = len(ids) + reserved
    return {"token_count_with_pair_specials": count, "max_tokens": 96,
            "supported": count <= 96}


def _source_group(candidates, provenance, expected, tokenizer):
    require(isinstance(candidates, list) and len(candidates) == expected,
            "source candidate group count differs")
    result, seen = [], set()
    for candidate in candidates:
        require(isinstance(candidate, dict)
                and isinstance(candidate.get("text"), str)
                and candidate["text"] not in seen,
                "source candidate uniqueness differs")
        support = token_support(tokenizer, candidate["text"])
        status = "supported" if support["supported"] else "unsupported"
        require(candidate.get("sentence_tokens") == support["token_count_with_pair_specials"]
                and candidate.get("sentence_support") == status,
                "source candidate token annotation differs")
        result.append({"text": candidate["text"], "support": status,
                       "provenance": [provenance]})
        seen.add(candidate["text"])
    return result


def _merge(groups):
    values, order = {}, []
    for group in groups:
        for candidate in group:
            text = candidate["text"]
            if text not in values:
                values[text] = deepcopy(candidate)
                order.append(text)
            else:
                require(values[text]["support"] == candidate["support"],
                        "duplicate candidate support differs")
                for origin in candidate["provenance"]:
                    if origin not in values[text]["provenance"]:
                        values[text]["provenance"].append(origin)
    return [values[text] for text in order]


def derive_baseline(parent, joint, tokenizer):
    require(parent.get("id") == joint.get("id")
            and parent.get("request") == joint.get("request")
            and parent.get("request", {}).get("context") == "",
            "source row identity differs")
    fallback = parent.get("fallback_public")
    require(isinstance(fallback, list) and len(fallback) == 5,
            "five-slot fallback differs")
    source = parent["request"]["input"]
    raw = [candidate for candidate in fallback
           if candidate.get("text") == source and candidate.get("segments")
           and all(segment.get("kind") == "raw"
                   for segment in candidate["segments"])]
    require(len(raw) == 1 and fallback[-1] == raw[0],
            "exact raw reserve differs")
    parents = _source_group(parent.get("parents"), "parent", 4, tokenizer)
    singles = _source_group(parent.get("proposals"), "single", 96, tokenizer)
    proposals = _source_group(
        joint.get("arms", {}).get("roman_repaired", {}).get("proposals"),
        "joint_roman_repaired", 96, tokenizer)
    return {"id": parent["id"], "request": deepcopy(parent["request"]),
            "fallback_public": deepcopy(fallback), "raw_reserve": deepcopy(raw[0]),
            "pool": _merge((parents, singles, proposals))}


def mixed_group(row, expected_id, expected_request, tokenizer):
    if row is None:
        return []
    require(row.get("id") == expected_id and row.get("request") == expected_request,
            "matching mixed source identity differs")
    proposals = row.get("proposals")
    require(isinstance(proposals, list)
            and row.get("proposal_count") == len(proposals),
            "matching mixed proposal count differs")
    result, seen = [], set()
    for proposal in proposals:
        text = proposal.get("text") if isinstance(proposal, dict) else None
        require(isinstance(text, str) and text and text not in seen,
                "matching mixed proposal uniqueness differs")
        support = token_support(tokenizer, text)
        require(proposal.get("token_support") == support,
                "matching mixed token annotation differs")
        result.append({"text": text,
                       "support": "supported" if support["supported"] else "unsupported",
                       "provenance": ["mixed_boundary"]})
        seen.add(text)
    return result


def _arm(pool, score_by_text, fallback, raw):
    supported = sorted((candidate["text"] for candidate in pool
                        if candidate["support"] == "supported"),
                       key=lambda text: (-score_by_text[text], text))
    unsupported = [candidate["text"] for candidate in pool
                   if candidate["support"] == "unsupported"]
    if supported:
        public = [{"text": text, "sentence_score": score_by_text[text]}
                  for text in supported[:4]] + [deepcopy(raw)]
        fallback_used = False
    else:
        public, fallback_used = deepcopy(fallback), True
    return {"pool": deepcopy(pool), "supported_order": supported,
            "unsupported_texts": unsupported, "public": public,
            "fallback_used": fallback_used}


def recreate_row(parent, joint, old, mixed, saved, tokenizer):
    baseline = derive_baseline(parent, joint, tokenizer)
    require(old.get("id") == baseline["id"] == saved.get("id")
            and old.get("request") == baseline["request"] == saved.get("request")
            and old.get("fallback_public") == baseline["fallback_public"]
            and old.get("arms", {}).get("roman_repaired", {}).get("pool")
                == baseline["pool"],
            "old baseline source projection differs")
    additions = mixed_group(mixed, baseline["id"], baseline["request"], tokenizer)
    expanded = _merge((baseline["pool"], additions))
    union = _merge((baseline["pool"], expanded))
    old_records = old.get("supported_scores")
    require(isinstance(old_records, list)
            and len({row.get("text") for row in old_records}) == len(old_records)
            and all(isinstance(row, dict) and set(row) == {"text", "score"}
                    and close(row["score"], row["score"], 0.0)
                    for row in old_records), "old supported scores differ")
    old_order = [row["text"] for row in old_records]
    old_scores = {row["text"]: float(row["score"]) for row in old_records}
    additive = [candidate["text"] for candidate in expanded
                if candidate["support"] == "supported"
                and candidate["text"] not in old_scores]
    expected_supported = old_order + additive
    expected_unsupported = [candidate["text"] for candidate in union
                            if candidate["support"] == "unsupported"]
    captured_scores = saved.get("supported_scores")
    require(isinstance(captured_scores, list)
            and [row.get("text") for row in captured_scores] == expected_supported
            and saved.get("unsupported_texts") == expected_unsupported,
            "captured supported/unsupported union differs")
    score_by_text = {}
    for record in captured_scores:
        require(isinstance(record, dict) and set(record) == {"text", "score"}
                and close(record["score"], record["score"], 0.0),
                "captured sentence score is nonfinite")
        score_by_text[record["text"]] = float(record["score"])
    baseline_supported = [candidate["text"] for candidate in baseline["pool"]
                          if candidate["support"] == "supported"]
    require(set(baseline_supported).issubset(old_scores)
            and set(baseline_supported).issubset(score_by_text),
            "fresh baseline score coverage differs")
    require(all(abs(score_by_text[text] - old_scores[text]) <= 1e-5
                for text in old_order), "old score replay parity differs")
    deltas = [abs(score_by_text[text] - old_scores[text])
              for text in baseline_supported]
    maximum = max(deltas, default=0.0)
    require(maximum <= 1e-5 and close(saved.get("baseline_max_score_delta"), maximum),
            "fresh baseline score parity differs")
    arms = {
        "baseline": _arm(baseline["pool"], score_by_text,
                         baseline["fallback_public"], baseline["raw_reserve"]),
        "mixed_union": _arm(expanded, score_by_text,
                            baseline["fallback_public"], baseline["raw_reserve"]),
    }
    require(saved.get("fallback_public") == baseline["fallback_public"]
            and saved.get("mixed_source_present") is (mixed is not None)
            and saved.get("old_supported_score_count") == len(old_order)
            and saved.get("additive_supported_score_count") == len(additive),
            "saved source diagnostics differ")
    for name, expected in arms.items():
        actual = saved.get("arms", {}).get(name)
        require(isinstance(actual, dict)
                and actual.get("pool") == expected["pool"]
                and actual.get("supported_order") == expected["supported_order"]
                and actual.get("unsupported_texts") == expected["unsupported_texts"]
                and actual.get("fallback_used") is expected["fallback_used"]
                and len(actual.get("public", [])) == len(expected["public"]),
                f"captured {name} arm differs")
        for left, right in zip(actual["public"], expected["public"]):
            if "sentence_score" in right:
                require(left.get("text") == right["text"]
                        and set(left) == {"text", "sentence_score"}
                        and close(left["sentence_score"], right["sentence_score"]),
                        f"captured {name} public score differs")
            else:
                require(left == right, f"captured {name} raw/fallback differs")
    old_arm = old["arms"]["roman_repaired"]
    require(arms["baseline"]["supported_order"] == old_arm["supported_order"]
            and [row["text"] for row in arms["baseline"]["public"]]
                == [row["text"] for row in old_arm["public"]],
            "fresh baseline order or public differs")
    return {"total": {"rows": 1, "mixed_source_rows": int(mixed is not None),
                       "unique_supported_scored": len(expected_supported),
                       "unique_unsupported_unscored": len(expected_unsupported)},
            "arms": {name: {"rows": 1, "pool": len(arm["pool"]),
                             "supported": len(arm["supported_order"]),
                             "unsupported": len(arm["unsupported_texts"]),
                             "fallback_rows": int(arm["fallback_used"])}
                     for name, arm in arms.items()}}


def _closed(directory, schema, rows, proof_prefix=""):
    report_path = directory / "report.json"
    capture_path = directory / "capture.jsonl"
    closure_path = directory / "capture.complete.json"
    proof_path = directory / "independent-verification.json"
    report, closure, proof = map(read_json, (report_path, closure_path, proof_path))
    capture_hash = report.get("hashes", {}).get("capture.jsonl",
                                                 report.get("capture_sha256"))
    resource_hash = report.get("hashes", {}).get("resources.jsonl",
                                                  report.get("resources_sha256"))
    require(report.get("schema") == schema and report.get("completed") is True
            and report.get("rows", report.get("counts", {}).get("rows")) == rows
            and report.get("labels_opened") is False
            and report.get("dev_opened") is False and report.get("test_opened") is False
            and capture_hash == digest(capture_path)
            and resource_hash == digest(directory / "resources.jsonl")
            and closure.get("completed") is True and closure.get("rows") == rows
            and closure.get("capture_sha256") == capture_hash
            and closure.get("report_sha256") == digest(report_path)
            and closure.get("labels_opened") is False
            and proof.get("verified") is True and proof.get("rows") == rows
            and proof.get(proof_prefix + "report_sha256") == digest(report_path)
            and proof.get(proof_prefix + "capture_sha256") == capture_hash,
            f"closed verified source differs: {directory.name}")
    pins = report.get("pins")
    require(isinstance(pins, dict) and pins
            and all(isinstance(path, str) and len(expected) == 64
                    and digest(Path(path)) == expected
                    for path, expected in pins.items()),
            f"source pins differ: {directory.name}")
    return report


def authenticate(directory):
    proof = directory / "independent-verification.json"
    require(directory.is_dir() and not proof.exists(),
            "capture directory missing or proof already exists")
    old_report = _closed(OLD_SCORE, "ungated-joint-sentence-scoring-v1", ROWS,
                         "score_")
    mixed_report = _closed(MIXED, "ungated-mixed-boundary-fit-capture-v1", 66)
    report_path, closure_path = directory / "report.json", directory / "capture.complete.json"
    report, closure = read_json(report_path), read_json(closure_path)
    require(report.get("schema") == "ungated-mixed-union-sentence-scoring-v1"
            and report.get("completed") is True and report.get("rows") == ROWS
            and report.get("limits") == LIMITS
            and report.get("settings") == {"max_length": 96, "batch_size": 8,
                "threads": 1, "interop_threads": 1, "external_context": ""}
            and report.get("score_tolerance") == 1e-5
            and report.get("labels_opened") is False
            and report.get("dev_opened") is False and report.get("test_opened") is False
            and report.get("sentence_inference") is True
            and report.get("generation_inference") is False
            and report.get("training_run") is False
            and report.get("production_changed") is False
            and report.get("hashes", {}).get("capture.jsonl")
                == digest(directory / "capture.jsonl")
            and report.get("hashes", {}).get("resources.jsonl")
                == digest(directory / "resources.jsonl")
            and closure == {"completed": True, "rows": ROWS,
                "capture_sha256": report["hashes"]["capture.jsonl"],
                "resources_sha256": report["hashes"]["resources.jsonl"],
                "report_sha256": digest(report_path), "labels_opened": False},
            "closed mixed-union capture differs")
    pins = report.get("pins")
    require(isinstance(pins, dict) and pins
            and all(digest(Path(path)) == expected for path, expected in pins.items()),
            "mixed-union pins differ")
    resources = list(iter_jsonl(directory / "resources.jsonl"))
    require(resources and report.get("resources") == resources[-1]
            and report.get("max_rss_bytes") == max(row["rss_bytes"] for row in resources)
            and report.get("max_swap_growth_bytes")
                == max(0, max(row["swap_growth_bytes"] for row in resources)),
            "mixed-union resources differ")
    for row in resources:
        require(isinstance(row.get("seconds"), (int, float))
                and math.isfinite(row["seconds"])
                and row["seconds"] <= LIMITS["whole_seconds"]
                and row["rss_bytes"] <= LIMITS["rss_bytes"]
                and row["swap_growth_bytes"] <= LIMITS["swap_growth_bytes"],
                "mixed-union resource bound differs")
    return report, closure, pins, {"report_sha256": digest(report_path),
        "capture_sha256": digest(directory / "capture.jsonl"),
        "closure_sha256": digest(closure_path),
        "resources_sha256": digest(directory / "resources.jsonl")}


def verify(directory: Path):
    started = time.monotonic()
    directory = directory.resolve()
    report, closure, pins, frozen = authenticate(directory)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT, local_files_only=True, trust_remote_code=False)
    require(tokenizer.num_special_tokens_to_add(pair=True) == 3,
            "actual tokenizer contract differs")
    parents = list(iter_jsonl(PARENT / "capture.jsonl"))
    joints = list(iter_jsonl(JOINT / "capture.jsonl"))
    olds = list(iter_jsonl(OLD_SCORE / "capture.jsonl"))
    saved = list(iter_jsonl(directory / "capture.jsonl"))
    mixed_rows = list(iter_jsonl(MIXED / "capture.jsonl"))
    mixed_by_id = {row["id"]: row for row in mixed_rows}
    require(len(parents) == len(joints) == len(olds) == len(saved) == ROWS
            and len(mixed_by_id) == 66, "source capture denominator differs")
    totals = Counter(rows=0, mixed_source_rows=0,
                     unique_supported_scored=0,
                     unique_unsupported_unscored=0)
    arms = {name: Counter(rows=0, pool=0, supported=0, unsupported=0,
                          fallback_rows=0)
            for name in ("baseline", "mixed_union")}
    ordered_ids = []
    for parent, joint, old, actual in zip(parents, joints, olds, saved):
        row_started = time.monotonic()
        values = recreate_row(parent, joint, old, mixed_by_id.get(old["id"]),
                              actual, tokenizer)
        totals.update(values["total"])
        for name in arms:
            arms[name].update(values["arms"][name])
        ordered_ids.append(actual["id"])
        require(time.monotonic() - row_started <= LIMITS["row_seconds"],
                "independent row verification exceeded")
    counts = {"total": dict(totals),
              "arms": {name: dict(value) for name, value in arms.items()}}
    require(report.get("counts") == counts, "mixed-union counters differ")
    closing = authenticate(directory)
    require(closing == (report, closure, pins, frozen),
            "mixed-union evidence changed during verification")
    elapsed = time.monotonic() - started
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss *= 1 if sys.platform == "darwin" else 1024
    require(elapsed <= LIMITS["whole_seconds"] and rss <= LIMITS["rss_bytes"],
            "independent verifier resource bound exceeded")
    result = {"schema": "ungated-mixed-union-fit-independent-verification-v1",
              "verified": True, "rows": ROWS, "ordered_ids": ordered_ids,
              "counts": counts, **frozen,
              "verifier_sha256": digest(Path(__file__).resolve()),
              "pins_verified": len(pins), "actual_tokenizer_replayed": True,
              "no_truncation": True, "unsupported_retained": True,
              "baseline_score_tolerance": 1e-5,
              "independent_neural_replay": False,
              "saved_inference_scores_finite": True,
              "labels_opened": False, "dev_opened": False,
              "test_opened": False, "production_changed": False,
              "resources": {"seconds": elapsed, "max_rss_bytes": rss,
                            "limits": LIMITS}}
    with (directory / "independent-verification.json").open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    verify(parser.parse_args().directory)
