"""Publish the independently audited v2.2 answer sets beside immutable v2.1."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

PARENT_TEST_SHA = "2f902fc1076dc8fc620a0b9ad3ad32e0d60aec02dc619b9d7424b28f77943aed"
AMENDMENTS = (
    (11, 15, "그녀는 의사입니다", "그년은 의사입니다"),
    (61, 65, "그녀는 한국어를 공부합니다", "그년은 한국어를 공부합니다"),
    (116, 120, "있는 것을 고른다", "잇는 것을 고른다"),
    (156, 160, "대학로를 걷는다", "대항로를 걷는다"),
    (176, 180, "칼날을 간다", "칼라를 간다"),
    (366, 370, "걷힌 서류를 받았다", "거친 서류를 받았다"),
    (391, 395, "숯이 남아 있다", "숱이 남아 있다"),
)


def publish(source, destination):
    if destination.exists():
        raise FileExistsError("Preserve the existing corpus revision")
    raw = (source / "test.jsonl").read_bytes()
    if hashlib.sha256(raw).hexdigest() != PARENT_TEST_SHA:
        raise ValueError("The audited v2.1 test source does not match")
    manifest = json.loads((source / "manifest.json").read_text())
    for name, checksum in manifest["sha256"].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != checksum:
            raise ValueError(f"Parent source mismatch: {name}")
    rows = [json.loads(line) for line in raw.splitlines()]
    changes = []
    for start, end, old, new in AMENDMENTS:
        for number in range(start, end + 1):
            row = rows[number - 1]
            if row["id"] != f"test-{number:04d}" or len(row["expected"]) != 1 or row["expected"][0].count(old) != 1:
                raise ValueError("Audited row identity or expected sentence changed")
            previous = list(row["expected"])
            row["expected"].append(row["expected"][0].replace(old, new))
            changes.append({"id": row["id"], "before": previous, "after": row["expected"]})
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ambiguity-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "v2.2"
        staged.mkdir()
        for name in ("train.jsonl", "dev.jsonl", "train_lexicon.json"):
            shutil.copyfile(source / name, staged / name)
        (staged / "test.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        manifest["corpus_version"] = "2.2"
        manifest["parent_test_sha256"] = PARENT_TEST_SHA
        manifest["sentence_ambiguity_policy"] = "Include independently audited standard same-RR/same-spacing readings when supplied context cannot distinguish them; register and rarity alone do not exclude a reading."
        manifest["sha256"]["test.jsonl"] = hashlib.sha256((staged / "test.jsonl").read_bytes()).hexdigest()
        (staged / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        (staged / "amendments.json").write_text(json.dumps({
            "parent_test_sha256": PARENT_TEST_SHA,
            "audit": ".omx/artifacts/independent-ambiguity-audit.md",
            "timing": "post-first-exposure independent audit without prediction access",
            "changed_fields": ["expected"], "changes": changes,
        }, ensure_ascii=False, indent=2) + "\n")
        staged.rename(destination)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/challenge"))
    parser.add_argument("--output", type=Path, default=Path("data/challenge/v2.2"))
    args = parser.parse_args()
    print(json.dumps(publish(args.source, args.output), ensure_ascii=False, indent=2))
