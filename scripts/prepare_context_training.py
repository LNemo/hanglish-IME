"""Prepare joint word/sentence features from source-only runtime proposals."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hanglish.core.lexical_data import FILES
from hanglish.core.neural_converter import NeuralConverter, beam_paths, file_sha256
from scripts.prepare_natural_ranker import (
    SALT as NATURAL_SELECTION_SALT,
    frozen_hashes,
    read_rows,
    select_rows,
    selected_sources,
    source_fields,
    validate_row,
)
from scripts.prepare_natural_word_ranker import (
    Recorder,
    add_targets,
    capture_groups,
    validate_case,
)


OUTPUT = ROOT / "data/natural_context/context-training-v1"
RUNTIME_VERSION = "neural-1-5e38f95d2a2a"
ROLES = ("natural_train", "natural_dev", "retention_train")
PROFILE = {
    "name": "joint-context-source-only-v1",
    "natural_train": 512,
    "natural_dev": 128,
    "retention_train": 1820,
    "learned_beam": 24,
    "static_beam": 24,
    "candidate_limit": 48,
    "threads": 2,
    "rss_limit_bytes": 3 * 1024**3,
    "time_limit_seconds": 1800,
    "progress_interval": 64,
    "natural_selection_salt": NATURAL_SELECTION_SALT,
}
FEATURE_POLICY = (
    "One runtime _prepare_candidates call; capture word calls and learned/static "
    "beams before target alignment"
)


def load_sources():
    """Return fixed natural selections and every pinned retention row in order."""
    selected, guard = selected_sources()
    challenge_train = ROOT / "data/challenge/train.jsonl"
    sources = {
        "natural_train": selected["natural_train"],
        "natural_dev": selected["natural_dev"],
        "retention_train": read_rows(challenge_train),
    }
    for role in ("natural_train", "natural_dev"):
        if sources[role] != select_rows(sources[role], role, PROFILE[role]):
            raise ValueError(f"{role} differs from fixed salted ordering")
    for role in ROLES:
        if len(sources[role]) != PROFILE[role]:
            raise ValueError(f"{role} count differs from fixed profile")
    return sources, guard


def source_digest(sources):
    payload = {
        role: [source_fields(row, role) for row in sources[role]]
        for role in ROLES
    }
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def mixed_split_guard(rows):
    training = rows["natural_train"] + rows["retention_train"]
    dev_targets = {text for row in rows["natural_dev"] for text in row["expected"]}
    overlaps = [row["id"] for row in training if any(
        target in row["context"] or any(target in expected for expected in row["expected"])
        for target in dev_targets
    )]
    train_groups = {row["group_id"] for row in rows["natural_train"]}
    dev_groups = {row["group_id"] for row in rows["natural_dev"]}
    group_overlap = sorted(train_groups & dev_groups)
    if overlaps or group_overlap:
        raise ValueError("natural dev data appears in mixed training sources")
    return {
        "dev_target_in_mixed_train_overlap_rows": 0,
        "natural_group_overlap_count": 0,
    }


def proposal_paths(pieces, static_pieces):
    """Union the runtime's learned and static 24-path beams without labels."""
    unique = {}
    for path in beam_paths(pieces):
        unique.setdefault(path[0], path)
    for path in beam_paths(static_pieces):
        unique.setdefault(path[0], path)
    paths = list(unique.values())
    if not 1 <= len(paths) <= PROFILE["candidate_limit"]:
        raise ValueError("joint sentence candidate pool exceeds fixed profile")
    return paths


def prepare_case(raw_source, role, converter, recorder):
    """Capture runtime features first, then align the selected source labels."""
    natural = role.startswith("natural_")
    text = raw_source["input"] if not natural else None
    context = raw_source["left_context"] if natural else raw_source["context"]
    if natural:
        from hanglish.core.phonology import romanize_word
        text = romanize_word(raw_source["text"])

    recorder.records = []
    pieces, static_pieces, _, _, _ = converter._prepare_candidates(
        text, context, learned=True, raw=False
    )
    groups = capture_groups(static_pieces, recorder.records, context)
    paths = proposal_paths(pieces, static_pieces)

    # This is the first target-bearing access in the proposal path.
    source = source_fields(raw_source, role)
    groups = add_targets(source, groups)
    candidates = [path[0] for path in paths]
    case = {
        **source,
        "groups": groups,
        "candidates": candidates,
        "positive_indices": [
            index for index, candidate in enumerate(candidates)
            if candidate in source["expected"]
        ],
    }
    validate_joint_case(case, raw_source, role)
    return case


def validate_joint_case(case, raw_source, role):
    validate_row(case, raw_source, role)
    validate_case(case, source_fields(raw_source, role))


def source_hashes():
    hashes = frozen_hashes()
    dictionary = ROOT / "data/external/mecab-ko-dic"
    paths = [
        Path(__file__).resolve(),
        ROOT / "scripts/prepare_natural_ranker.py",
        ROOT / "scripts/prepare_natural_word_ranker.py",
        dictionary / "NNBC.csv",
        dictionary / "SOURCE.json",
    ]
    paths.extend(dictionary / f"{name}.csv" for name in FILES)
    paths.extend(path for path in (ROOT / "checkpoints/sentence_ranker").iterdir()
                 if path.is_file())
    for path in sorted(set(paths)):
        hashes[str(path.relative_to(ROOT))] = file_sha256(path)
    return dict(sorted(hashes.items()))


def validate_context_data(directory, manifest_sha256):
    manifest_path = directory / "manifest.json"
    if not isinstance(manifest_sha256, str) or file_sha256(manifest_path) != manifest_sha256:
        raise ValueError("context manifest differs from explicitly pinned hash")
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("schema_version") != 1
            or manifest.get("profile") != PROFILE
            or manifest.get("runtime_version") != RUNTIME_VERSION
            or manifest.get("no_gold_injection") is not True
            or manifest.get("test_data_read") is not False
            or manifest.get("uncovered_rows_retained") is not True
            or manifest.get("feature_policy") != FEATURE_POLICY):
        raise ValueError("invalid joint preparation policy")
    sources, guard = load_sources()
    prepared_sources = {
        role: [source_fields(row, role) for row in sources[role]]
        for role in ROLES
    }
    expected_guard = mixed_split_guard(prepared_sources)
    if (manifest.get("challenge_guard") != guard
            or manifest.get("mixed_split_guard") != expected_guard
            or manifest.get("source_digest") != source_digest(sources)
            or manifest.get("source_hashes") != source_hashes()
            or set(manifest.get("outputs", {})) != set(ROLES)):
        raise ValueError("joint preparation sources changed")
    result = {}
    for role in ROLES:
        path = directory / f"{role}.jsonl"
        spec = manifest["outputs"][role]
        if file_sha256(path) != spec["sha256"] or path.stat().st_size != spec["bytes"]:
            raise ValueError("joint preparation output integrity mismatch")
        rows = read_rows(path)
        if len(rows) != PROFILE[role] or len(rows) != spec["rows"]:
            raise ValueError("joint preparation row count mismatch")
        for case, raw_source in zip(rows, sources[role]):
            validate_joint_case(case, raw_source, role)
        result[role] = rows
    return manifest, result


def rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def prepare(directory):
    if directory.exists():
        raise FileExistsError(directory)
    sources, guard = load_sources()
    prepared_sources = {
        role: [source_fields(row, role) for row in sources[role]]
        for role in ROLES
    }
    split_guard = mixed_split_guard(prepared_sources)
    before = source_hashes()
    converter = NeuralConverter(ranker_kind="semantic")
    converter.load_models()
    if converter.version != RUNTIME_VERSION:
        raise ValueError("runtime differs from fixed joint preparation profile")
    import torch
    torch.set_num_threads(PROFILE["threads"])
    recorder = Recorder(converter.ranker)
    converter.ranker = recorder
    directory.mkdir(parents=True)
    started = time.monotonic()
    completed = 0
    outputs = {}
    for role in ROLES:
        path = directory / f"{role}.jsonl"
        counts = {"rows": len(sources[role]), "sentence_covered": 0,
                  "word_groups": 0, "word_groups_covered": 0}
        with path.open("x") as stream:
            for raw_source in sources[role]:
                case = prepare_case(raw_source, role, converter, recorder)
                stream.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
                counts["sentence_covered"] += bool(case["positive_indices"])
                counts["word_groups"] += len(case["groups"])
                counts["word_groups_covered"] += sum(
                    bool(group["positive_indices"]) for group in case["groups"]
                )
                completed += 1
                if (rss_bytes() > PROFILE["rss_limit_bytes"]
                        or time.monotonic() - started > PROFILE["time_limit_seconds"]):
                    raise RuntimeError("joint preparation resource bound exceeded")
                if completed % PROFILE["progress_interval"] == 0:
                    print(json.dumps({"event": "prepare_context", "completed": completed,
                                      "elapsed_seconds": round(time.monotonic() - started, 2)}),
                          flush=True)
        outputs[role] = {**counts, "bytes": path.stat().st_size,
                         "sha256": file_sha256(path)}
    if before != source_hashes():
        raise ValueError("joint preparation sources changed during capture")
    manifest = {
        "schema_version": 1,
        "profile": PROFILE,
        "runtime_version": converter.version,
        "source_hashes": before,
        "source_digest": source_digest(sources),
        "challenge_guard": guard,
        "mixed_split_guard": split_guard,
        "outputs": outputs,
        "no_gold_injection": True,
        "test_data_read": False,
        "feature_policy": FEATURE_POLICY,
        "uncovered_rows_retained": True,
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": rss_bytes(),
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    digest = file_sha256(manifest_path)
    validate_context_data(directory, digest)
    print(json.dumps({"manifest_sha256": digest, "outputs": outputs,
                      "peak_rss_bytes": manifest["peak_rss_bytes"]}, ensure_ascii=False),
          flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    prepare(parser.parse_args().output)
