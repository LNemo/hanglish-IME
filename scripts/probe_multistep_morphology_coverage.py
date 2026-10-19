"""Model-free coverage probe for fixed source-only multi-step morphology."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hanglish.core.lattice import CandidateGenerator, WordOption
from hanglish.core.neural_converter import LETTERS, PROTECTED, TOKENS
from hanglish.core.phonology import romanize_word
from scripts.multistep_morphology_proposals import MAX_PROPOSALS, MultiStepMorphology


INPUT = ROOT / ".omx/artifacts/semantic-pair-training-input-v2"
INPUT_MANIFEST_SHA = "3ab561d1a767e7a65b4706c2cd0f668366693536849e17f2f3e620fbcf5550a6"
INPUT_ROWS_SHA = "6a1f7f61a7f7ffeb9209772559412931137d71ef0173d7429264d9cb53aed396"
OUTPUT = ROOT / ".omx/artifacts/multistep-morphology-coverage-v2"
NATURAL_CASES = ROOT / "data/natural_context/final-choice-v1/natural_fit/cases.jsonl"
NATURAL_CASES_SHA = "789e45a3b6028e7416ca3b82739c906c558ec414320796c53b0b76ed251b9a18"
EXPECTED_ROWS = 2048
OFFICIAL_GRAMMAR_SOURCES = (
    "https://www.korean.go.kr/front/onlineQna/onlineQnaView.do?mn_id=216&pageIndex=1&qna_seq=324971",
    "https://www.korean.go.kr/front/mcfaq/mcfaqView.do?mcfaq_seq=8559",
    "https://www.korean.go.kr/front/onlineQna/onlineQnaView.do?mn_id=&pageIndex=1&qna_seq=320510",
    "https://www.korean.go.kr/front/onlineQna/onlineQnaView.do?mn_id=&pageIndex=1&qna_seq=324214",
)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def verify_pins(pins):
    for name, expected in pins.items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"source changed during morphology probe: {name}")


def load_source_rows():
    if sha(INPUT / "manifest.json") != INPUT_MANIFEST_SHA:
        raise ValueError("semantic pair input manifest changed")
    manifest = json.loads((INPUT / "manifest.json").read_bytes())
    if (manifest.get("completed") is not True or manifest.get("rows") != 3492
            or manifest.get("role_rows") != {"natural_fit": 2048,
                                               "challenge_fit": 1444}
            or manifest.get("labels_or_expected_saved") is not False
            or manifest.get("rows_sha256") != INPUT_ROWS_SHA
            or sha(INPUT / "rows.jsonl") != INPUT_ROWS_SHA):
        raise ValueError("source-only semantic pair rows differ")
    all_rows = [json.loads(line) for line in (INPUT / "rows.jsonl").read_bytes().splitlines()
                if line]
    if len(all_rows) != 3492 or any("expected" in row for row in all_rows):
        raise ValueError("source-only rows contain labels or are incomplete")
    return [
        {"id": row["id"], "input": row["input"], "context": row["context"]}
        for row in all_rows if row["source_role"] == "natural_fit"
    ]


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def composable(target, option_groups):
    positions = {0}
    for options in option_groups:
        positions = {
            offset + len(text) for offset in positions for text in options
            if target.startswith(text, offset)
        }
        if not positions:
            return False
    return len(target) in positions


def source_pieces(text, generator, morphology):
    pieces = []
    for match in TOKENS.finditer(text):
        chunk = match.group()
        if chunk.isspace() or PROTECTED.fullmatch(chunk):
            pieces.append({
                "source": chunk, "start": match.start(), "end": match.end(),
                "baseline": [asdict(WordOption(chunk, 0.0, "raw"))],
                "proposals": [],
            })
            continue
        for part in LETTERS.finditer(chunk):
            source = part.group()
            start = match.start() + part.start()
            if source.isascii() and source.isalpha():
                baseline_options = tuple(generator.options(source))
                baseline_texts = {option.text for option in baseline_options}
                proposed = tuple(
                    option for option in morphology.propose(source)
                    if option.text not in baseline_texts
                )
            else:
                baseline_options = (WordOption(source, 0.0, "raw"),)
                proposed = ()
            if len(proposed) > MAX_PROPOSALS:
                raise RuntimeError("proposal cap exceeded")
            pieces.append({
                "source": source, "start": start, "end": start + len(source),
                "baseline": [asdict(option) for option in baseline_options],
                "proposals": [asdict(option) for option in proposed],
            })
    if "".join(piece["source"] for piece in pieces) != text:
        raise ValueError("source pieces do not reproduce input")
    return pieces


def coverage_metrics(source_rows, captures, target_rows):
    if (len(source_rows) != len(captures) or len(captures) != len(target_rows)
            or [row["id"] for row in source_rows] != [row["id"] for row in captures]
            or [row["id"] for row in source_rows] != [row["id"] for row in target_rows]):
        raise ValueError("source capture and targets differ")
    counts = Counter()
    outcomes = []
    nonspace = re.compile(r"\S+")
    for source_row, capture, target_row in zip(source_rows, captures, target_rows):
        pieces = capture["pieces"]
        baseline_groups = [tuple(option["text"] for option in piece["baseline"])
                           for piece in pieces]
        after_groups = [
            (*baseline, *(option["text"] for option in piece["proposals"]))
            for baseline, piece in zip(baseline_groups, pieces)
        ]
        expected = target_row.get("expected")
        if not isinstance(expected, list) or not expected:
            raise ValueError("expected target list is required after capture")
        before_full = any(composable(target, baseline_groups) for target in expected)
        after_full = any(composable(target, after_groups) for target in expected)
        counts["before_full_composable"] += before_full
        counts["after_full_composable"] += after_full
        counts["new_full_composable"] += after_full and not before_full
        counts["lost_full_composable"] += before_full and not after_full

        target = expected[0]
        source_chunks = list(nonspace.finditer(source_row["input"]))
        target_chunks = nonspace.findall(target)
        if (len(source_chunks) != len(target_chunks)
                or any(romanize_word(gold) != match.group()
                       for match, gold in zip(source_chunks, target_chunks))):
            counts["unaligned_rows"] += 1
            outcomes.append({"id": source_row["id"], "aligned": False,
                             "before_full": before_full, "after_full": after_full})
            continue
        row_word_gain = False
        for match, gold in zip(source_chunks, target_chunks):
            selected = [index for index, piece in enumerate(pieces)
                        if match.start() <= piece["start"] and piece["end"] <= match.end()]
            before = composable(gold, [baseline_groups[index] for index in selected])
            after = composable(gold, [after_groups[index] for index in selected])
            counts["gold_chunks"] += 1
            counts["before_gold_chunk_covered"] += before
            counts["after_gold_chunk_covered"] += after
            counts["new_gold_chunk_covered"] += after and not before
            counts["lost_gold_chunk_covered"] += before and not after
            row_word_gain |= after and not before
        counts["rows_with_new_gold_chunk"] += row_word_gain
        outcomes.append({"id": source_row["id"], "aligned": True,
                         "before_full": before_full, "after_full": after_full,
                         "word_gain": row_word_gain})
    return dict(counts), outcomes


def _resources(started):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss = rss if sys.platform == "darwin" else rss * 1024
    if rss > 3 * 1024**3 or time.monotonic() - started > 900:
        raise RuntimeError("model-free morphology probe resource cap exceeded")
    return rss


def main(output=OUTPUT):
    started = time.monotonic()
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    source_rows = load_source_rows()
    if len(source_rows) != EXPECTED_ROWS or len({row["id"] for row in source_rows}) != EXPECTED_ROWS:
        raise ValueError("complete source-only natural FIT rows required")

    direct_paths = [
        INPUT / "manifest.json", INPUT / "rows.jsonl",
        ROOT / "scripts/multistep_morphology_proposals.py",
        ROOT / "scripts/probe_multistep_morphology_coverage.py",
        ROOT / "hanglish/core/lexical_data.py", ROOT / "hanglish/core/lattice.py",
        ROOT / "hanglish/core/neural_converter.py", ROOT / "hanglish/core/phonology.py",
        ROOT / "hanglish/core/workbench.py", ROOT / "data/morphology_lexicon.json",
        ROOT / "data/workbench_lexicon.json", ROOT / "data/challenge/train_lexicon.json",
        ROOT / "data/common_english.txt", ROOT / "data/it_terms.txt",
    ]
    direct_paths.extend((ROOT / "data/external/mecab-ko-dic" / f"{name}.csv")
                        for name in ("NNG", "NNP", "NP", "NR", "MAG", "VV", "VA",
                                     "VX", "VCP", "VCN", "XR", "Inflect"))
    input_pins = {str(path.relative_to(ROOT)): sha(path) for path in direct_paths}
    verify_pins(input_pins)
    generator = CandidateGenerator(external_lexicon=True)
    morphology = MultiStepMorphology.from_data()
    output.mkdir(parents=True, exist_ok=False)
    captures = []
    with (output / "source-capture.jsonl").open("xb") as stream:
        for index, row in enumerate(source_rows):
            record = {"id": row["id"], "input": row["input"],
                      "pieces": source_pieces(row["input"], generator, morphology)}
            stream.write(canonical(record) + b"\n")
            captures.append(record)
            if (index + 1) % 256 == 0:
                _resources(started)
    capture_sha = sha(output / "source-capture.jsonl")
    generated = sum(len(piece["proposals"]) for row in captures for piece in row["pieces"])
    capture_receipt = {
        "schema": "multistep-morphology-source-capture-v2",
        "closed_before_target_join": True,
        "rows": len(captures), "generated_proposals": generated,
        "source_capture_sha256": capture_sha,
        "cap_per_query": MAX_PROPOSALS,
        "ordering": "family priority, MeCab cost, text, POS, constituents",
        "target_fields_read": False,
        "source_hashes_before_target_join": dict(sorted(input_pins.items())),
    }
    (output / "source-capture-receipt.json").write_bytes(
        canonical(capture_receipt) + b"\n"
    )

    if sha(NATURAL_CASES) != NATURAL_CASES_SHA:
        raise ValueError("natural FIT target bytes changed")
    target_rows = [json.loads(line) for line in NATURAL_CASES.read_bytes().splitlines() if line]
    counts, outcomes = coverage_metrics(source_rows, captures, target_rows)
    if counts.get("lost_gold_chunk_covered", 0) or counts.get("lost_full_composable", 0):
        raise RuntimeError("additive proposal coverage cannot regress")
    pins = dict(input_pins)
    pins[str(NATURAL_CASES.relative_to(ROOT))] = NATURAL_CASES_SHA
    verify_pins(pins)
    report = {
        "schema": "multistep-morphology-coverage-v2", "completed": True,
        "scope": "all source-only natural_fit2048 chunks; target join only after closed capture",
        "counts": counts, "rows": EXPECTED_ROWS,
        "generated_proposals": generated, "source_capture_sha256": capture_sha,
        "source_capture_receipt_sha256": sha(output / "source-capture-receipt.json"),
        "source_hashes": dict(sorted(pins.items())),
        "wordfreq_external_version": generator.external_version,
        "external_lexicon_fingerprint": generator.external_fingerprint,
        "mecab_lexicon_fingerprint": morphology.lexicon_fingerprint,
        "official_grammar_sources": list(OFFICIAL_GRAMMAR_SOURCES),
        "grammar_policy": {
            "plural_requires_attested_noun_plus_deul_xsn": True,
            "nominalizer_cross_product_is_fixed_compatible_subset": True,
            "stacked_postpositions_are_fixed_attested_whitelist": True,
            "forward_identity_required": True,
            "cap_per_query": MAX_PROPOSALS,
        },
        "outcomes": outcomes,
        "word_coverage_only": True, "complete_composability_only": True,
        "runtime_or_top1_claimed": False, "candidate_scores_assigned": False,
        "model_inference_run": False, "training_run": False,
        "development_data_read": False, "reserve_data_read": False,
        "final_data_read": False, "production_modified": False,
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": _resources(started),
    }
    (output / "report.json").write_bytes(canonical(report) + b"\n")
    print(json.dumps({"counts": counts, "generated_proposals": generated,
                      "report_sha256": sha(output / "report.json")},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
