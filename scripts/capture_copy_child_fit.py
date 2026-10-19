"""Capture source-only COPY-child phonetic proposals over the frozen FIT pool."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from scripts.audit_copy_boundaries_fit import authenticate as authenticate_fit
from scripts.audit_recombination_tokenization import digest, require
from scripts.train_character_edge_full_fit import ResourceGuard


SCHEMA = "copy-child-phonetic-fit-v1"
INTAKE = ROOT / ".omx/artifacts/current-fit-recombination-selection-intake-v1.json"
PLAN = ROOT / ".omx/plans/copy-child-phonetic-fit-v1.md"
PRIOR = ROOT / ".omx/artifacts/copy-boundary-fit-v1"
MODEL = ROOT / ".omx/artifacts/observed-phonetic-augmented-training-v1/phonetic_transducer.pt"
HELPER = ROOT / "hanglish/core/copy_child_proposals.py"
PHONETIC_SOURCE = ROOT / "hanglish/models/phonetic_transducer.py"
DECODE_SOURCE = ROOT / "hanglish/models/batched_phonetic_decode.py"
MODEL_SHA = "f06b0a7cd6c5ccbdfaa376cc29f59e90b3fcf0ae3a157fd2c4f7efaa73b4180d"
METADATA_SHA = "c7dcd4ad5aabb5849d89dc8847697c3eab66eaac1679126b06df0c37573f6e9d"
PRIOR_REPORT_SHA = "0b6f79e01868cce0236a88217b6ce882e32f9b169a433c42c3b696178a31e39c"
PRIOR_PROOF_SHA = "0a57e285e6f47ff9e4fefcc33d2c7154f78e7dc912dd2e8c8e0da07848819f40"
BEAM_SIZE = 8
MAX_NEW_PER_ROW = 96
MAX_OUTPUT_BYTES = 256 * 1024**2
_ASCII_OUTPUT = re.compile(r"[A-Za-z0-9]")
_ROMAN = re.compile(r"[a-z]+\Z")
_HANGUL = re.compile(r"[\uac00-\ud7a3]+\Z")


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def write_json(path: Path, value) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True,
                  separators=(",", ":"), allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_json_bounded(output: Path, path: Path, value) -> None:
    payload = canonical(value) + "\n"
    require(output_bytes(output) + len(payload.encode()) <= MAX_OUTPUT_BYTES,
            "COPY-child capture output limit exceeded before write")
    with path.open("x") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def output_bytes(output: Path) -> int:
    return sum(path.stat().st_size for path in output.iterdir() if path.is_file())


def require_output_bound(output: Path) -> None:
    require(output_bytes(output) <= MAX_OUTPUT_BYTES,
            "COPY-child capture output limit exceeded")


def coarse_eligible(candidate: dict) -> bool:
    """Cheap, semantic-free gate for a possible mixed non-COPY parent."""
    segments = candidate.get("segments")
    require(isinstance(segments, list), "parent segments differ")
    return any(
        isinstance(segment, dict)
        and segment.get("kind") == "mixed"
        and segment.get("source") != segment.get("text")
        and isinstance(segment.get("text"), str)
        and _ASCII_OUTPUT.search(segment["text"]) is not None
        for segment in segments
    )


def validate_beam(roman: str, values) -> list[list]:
    """Validate one decoder result while preserving its stable tie order."""
    from hanglish.core.phonology import romanize_word

    require(isinstance(roman, str) and _ROMAN.fullmatch(roman) is not None,
            "beam Roman source differs")
    require(isinstance(values, (list, tuple)) and len(values) <= BEAM_SIZE,
            "beam size differs")
    result, seen = [], set()
    previous = math.inf
    for item in values:
        require(isinstance(item, (list, tuple)) and len(item) == 2,
                "beam entry differs")
        target, score = item
        require(isinstance(target, str) and _HANGUL.fullmatch(target) is not None,
                "beam output is not complete Hangul")
        require(not isinstance(score, bool) and isinstance(score, (int, float))
                and math.isfinite(score), "beam score is not finite")
        score = float(score)
        require(target not in seen and score <= previous,
                "beam is duplicated or not in stable score order")
        require(romanize_word(target) == roman,
                "beam output fails forward romanization")
        seen.add(target)
        previous = score
        result.append([target, score])
    return result


def parent_candidates(scoring: dict):
    """Return frozen parent order and its truthful source."""
    original = scoring.get("original")
    novel = scoring.get("novel")
    fallback = scoring.get("fallback_public")
    ranked = scoring.get("ranked")
    require(isinstance(original, list) and isinstance(novel, list)
            and isinstance(ranked, bool), "scoring parent schema differs")
    if ranked:
        require(bool(original) and fallback is None,
                "ranked scoring parent source differs")
        return "scored_pool", [
            ("original", candidate) for candidate in original
        ] + [("novel", candidate) for candidate in novel]
    require(not original and not novel and isinstance(fallback, list) and fallback,
            "no-pool fallback parent source differs")
    return "fallback_public", [("fallback_public", candidate)
                                for candidate in fallback]


def capture_row(row: dict, *, generator, generated: dict,
                proposal_fn=None, maximum_new: int = MAX_NEW_PER_ROW) -> dict:
    """Pure per-row capture; it never mutates or records the full parent pool."""
    if proposal_fn is None:
        from hanglish.core.copy_child_proposals import propose_children
        proposal_fn = propose_children
    require(type(maximum_new) is int and 1 <= maximum_new <= MAX_NEW_PER_ROW,
            "row proposal budget differs")
    identity = row.get("id")
    request = row.get("request")
    scoring = row.get("scoring")
    require(isinstance(identity, str) and identity.startswith("s3-train-")
            and isinstance(request, dict) and request.get("context") == ""
            and isinstance(request.get("input"), str)
            and isinstance(scoring, dict), "FIT scoring row differs")
    require(scoring.get("id") == identity and scoring.get("request") == request,
            "nested scoring identity differs")
    source = request["input"]
    parent_source, parents = parent_candidates(scoring)
    existing_texts = {candidate.get("text") for _, candidate in parents}
    require(None not in existing_texts
            and all(isinstance(value, str) for value in existing_texts),
            "parent text differs")
    frozen = canonical(parents)
    accepted = set()
    proposals = []
    requests = []
    parent_diagnostics = []
    counts = Counter(
        parents=len(parents),
        original_parents=len(scoring["original"]),
        novel_parents=len(scoring["novel"]),
        fallback_parents=(len(parents) if parent_source == "fallback_public" else 0),
        scorable_parents=(len(scoring["original"]) + sum(
            candidate.get("status") == "scored" for candidate in scoring["novel"])),
        unsupported_parents=sum(
            candidate.get("status") == "unsupported" for candidate in scoring["novel"]),
    )

    for parent_index, (parent_pool, candidate) in enumerate(parents):
        if not coarse_eligible(candidate):
            counts["cheap_skipped_parents"] += 1
            continue
        counts["coarse_eligible_parents"] += 1
        if len(proposals) >= maximum_new:
            counts["budget_skipped_parents"] += 1
            continue
        result = proposal_fn(
            source, candidate["text"], candidate["segments"],
            generator=generator, generated=generated,
            existing_texts=existing_texts | accepted,
            limit=maximum_new - len(proposals),
        )
        require(isinstance(result, dict)
                and isinstance(result.get("requests"), list)
                and isinstance(result.get("proposals"), list)
                and isinstance(result.get("diagnostics"), dict),
                "COPY-child helper result differs")
        counts["helper_called_parents"] += 1
        counts["requests"] += len(result["requests"])
        counts["parents_with_requests"] += bool(result["requests"])
        counts["parents_with_proposals"] += bool(result["proposals"])
        counts["empty_beam_requests"] += sum(
            value.get("status") == "empty_beam"
            for value in result["diagnostics"].get("generation", []))
        captured_requests = []
        for request_row in result["requests"]:
            captured = deepcopy(request_row)
            captured["parent_candidate_index"] = parent_index
            captured["parent_pool"] = parent_pool
            captured_requests.append(captured)
            requests.append(deepcopy(captured))
        for proposed in result["proposals"]:
            captured = deepcopy(proposed)
            captured["parent_candidate_index"] = parent_index
            captured["parent_pool"] = parent_pool
            require(captured["text"] not in existing_texts
                    and captured["text"] not in accepted,
                    "helper emitted a duplicate candidate")
            accepted.add(captured["text"])
            proposals.append(captured)
        parent_diagnostics.append({
            "parent_candidate_index": parent_index,
            "parent_pool": parent_pool,
            "parent_text": candidate["text"],
            "requests": captured_requests,
            "diagnostics": deepcopy(result["diagnostics"]),
        })

    require(canonical(parents) == frozen, "source parent candidates mutated")
    require(len(proposals) == len(accepted) <= maximum_new,
            "row proposal budget or uniqueness differs")
    counts["new_proposals"] = len(proposals)
    return {
        "schema": SCHEMA + "-row",
        "id": identity,
        "request": deepcopy(request),
        "source_shard": row.get("source_shard"),
        "source_row_index": row.get("source_row_index"),
        "parent_source": parent_source,
        "parent_counts": dict(counts),
        "existing_text_count": len(existing_texts),
        "requests": requests,
        "proposals": proposals,
        "parent_diagnostics": parent_diagnostics,
        "budgets": {
            "maximum_new": maximum_new,
            "accepted_new": len(proposals),
            "remaining_new": maximum_new - len(proposals),
            "limit_reached": len(proposals) == maximum_new,
        },
    }


def authenticate() -> tuple[dict, dict, dict]:
    intake, pins = authenticate_fit()
    report_path = PRIOR / "report.json"
    proof_path = PRIOR / "independent-verification.json"
    require(digest(report_path) == PRIOR_REPORT_SHA
            and digest(proof_path) == PRIOR_PROOF_SHA,
            "prior COPY closure changed")
    prior = json.loads(report_path.read_text())
    proof = json.loads(proof_path.read_text())
    require(prior.get("completed") is True and prior.get("rows") == 1642
            and prior.get("labels_opened") is False
            and proof.get("verified") is True
            and proof.get("report_sha256") == PRIOR_REPORT_SHA,
            "prior COPY report or proof differs")
    require(digest(MODEL) == MODEL_SHA
            and digest(MODEL.with_suffix(".json")) == METADATA_SHA,
            "phonetic checkpoint or metadata changed")
    metadata = json.loads(MODEL.with_suffix(".json").read_text())
    require(metadata.get("checkpoint_sha256") == MODEL_SHA,
            "phonetic metadata checkpoint identity differs")
    for path in (Path(__file__), HELPER, PLAN, report_path, proof_path, MODEL,
                 MODEL.with_suffix(".json"), PHONETIC_SOURCE, DECODE_SOURCE):
        require(path.is_file(), "required COPY-child source missing: " + str(path))
        pins[str(path.resolve())] = digest(path)
    return intake, pins, prior


def run(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    guard = ResourceGuard(output)
    try:
        intake, pins, prior = authenticate()
        guard.observe("authenticated")
        require_output_bound(output)

        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        from hanglish.core.lattice import CandidateGenerator
        from hanglish.models.batched_phonetic_decode import generate_batched
        from hanglish.models.phonetic_transducer import PhoneticTransducer

        generator = CandidateGenerator(external_lexicon=True)
        expected_generator = prior.get("generator")
        actual_generator = {
            "external_lexicon": True,
            "external_version": generator.external_version,
            "external_fingerprint": generator.external_fingerprint,
            "dictionary_fingerprint": generator.dictionary_fingerprint,
            "english_suffixes": [list(value) for value in generator.english_suffixes],
        }
        require(actual_generator == expected_generator,
                "runtime generator differs from prior COPY closure")
        romans = [pair[0] for pair in generator.english_suffixes]
        require(len(romans) == len(set(romans)) == 66
                and all(_ROMAN.fullmatch(value) for value in romans),
                "English suffix model-source inventory differs")
        model = PhoneticTransducer.load(MODEL, device="cpu")
        guard.observe("model-loaded")
        require_output_bound(output)

        generated = {}
        for roman in sorted(romans):
            started = time.monotonic()
            generated[roman] = validate_beam(
                roman, generate_batched(model, roman, BEAM_SIZE))
            guard.require_row_time(started)
            guard.observe("decoded-" + roman)
            require_output_bound(output)
        write_json_bounded(output, output / "beams.json", generated)
        require_output_bound(output)

        counts = Counter(rows=0, no_pool_rows=0, old_candidates=0,
                         novel_candidates=0, unsupported_candidates=0,
                         fallback_parents=0, total_new=0, new_rows=0)
        identity_hash = hashlib.sha256()
        with (output / "capture.jsonl").open("x") as destination:
            for expected_shard, shard in enumerate(intake["shards"]):
                require(shard.get("shard") == expected_shard,
                        "FIT scoring shard order differs")
                path = Path(shard["directory"]) / "capture.jsonl"
                with path.open() as stream:
                    for line in stream:
                        source_row = json.loads(line)
                        started = time.monotonic()
                        captured = capture_row(
                            source_row, generator=generator, generated=generated)
                        encoded = canonical(captured) + "\n"
                        require(output_bytes(output) + len(encoded.encode()) <= MAX_OUTPUT_BYTES,
                                "COPY-child capture output limit exceeded before row write")
                        destination.write(encoded)
                        destination.flush()
                        counts["rows"] += 1
                        counts["no_pool_rows"] += captured["parent_source"] == "fallback_public"
                        counts["old_candidates"] += captured["parent_counts"]["original_parents"]
                        counts["novel_candidates"] += captured["parent_counts"]["novel_parents"]
                        counts["unsupported_candidates"] += captured["parent_counts"]["unsupported_parents"]
                        counts["fallback_parents"] += captured["parent_counts"]["fallback_parents"]
                        counts["total_new"] += len(captured["proposals"])
                        counts["new_rows"] += bool(captured["proposals"])
                        identity_hash.update((captured["id"] + "\n").encode())
                        guard.require_row_time(started)
                        guard.observe("row-" + str(counts["rows"]))
                        require_output_bound(output)
                        if counts["rows"] % 256 == 0:
                            print(json.dumps({"rows": counts["rows"],
                                              "total_new": counts["total_new"]}),
                                  flush=True)

        require(counts["rows"] == 1642 and counts["no_pool_rows"] == 13
                and counts["old_candidates"] == 249177
                and counts["novel_candidates"] == 149885
                and counts["unsupported_candidates"] == 31,
                "complete FIT source denominator differs")
        for path, expected in pins.items():
            require(digest(Path(path)) == expected,
                    "source changed during COPY-child capture")
        guard.observe("closed")
        require_output_bound(output)
        hashes = {name: digest(output / name)
                  for name in ("capture.jsonl", "beams.json", "resources.jsonl")}
        report = {
            "schema": SCHEMA,
            "completed": True,
            "rows": 1642,
            "counts": dict(counts),
            "ordered_id_sha256": identity_hash.hexdigest(),
            "beam": {"size": BEAM_SIZE, "sources": len(generated),
                     "empty_sources": sum(not values for values in generated.values()),
                     "threads": 2, "interop_threads": 1,
                     "checkpoint_sha256": MODEL_SHA},
            "generator": actual_generator,
            "budgets": {"maximum_new_per_row": MAX_NEW_PER_ROW,
                        "maximum_output_bytes": MAX_OUTPUT_BYTES},
            "pins": pins,
            "hashes": hashes,
            "resources": {"seconds": time.monotonic() - guard.started,
                          "max_rss_bytes": guard.peak_rss,
                          "max_swap_growth_bytes": guard.peak_swap_growth,
                          "output_bytes": output_bytes(output)},
            "labels_opened": False,
            "dev_opened": False,
            "test_opened": False,
            "production_changed": False,
            "neural_training": False,
            "model_inference": True,
            "scope": ("Source-only FIT COPY-child generation; no labels, sentence "
                      "scores, ranking, accuracy, DEV, TEST, or production change."),
        }
        write_json_bounded(output, output / "report.json", report)
        closure = {
            "schema": SCHEMA + "-closure",
            "completed": True,
            "rows": 1642,
            "capture_sha256": hashes["capture.jsonl"],
            "beams_sha256": hashes["beams.json"],
            "resources_sha256": hashes["resources.jsonl"],
            "report_sha256": digest(output / "report.json"),
            "labels_opened": False,
            "dev_opened": False,
            "test_opened": False,
            "production_changed": False,
            "source_only": True,
        }
        write_json_bounded(output, output / "capture.complete.json", closure)
        require_output_bound(output)
        print(json.dumps({"completed": True, "counts": dict(counts),
                          "resources": report["resources"]},
                         ensure_ascii=False), flush=True)
    except Exception as error:
        failure = output / "failure.json"
        if not failure.exists():
            write_json(failure, {"schema": SCHEMA + "-failure",
                                 "completed": False,
                                 "error_type": type(error).__name__,
                                 "error": str(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args().output)
