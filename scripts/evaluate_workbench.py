"""Small transparent probe suite, not a held-out accuracy benchmark."""
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hanglish.core.workbench import WorkbenchEngine


PROBES = [
    ("supported", "annyeonghaseyo", ["안녕하세요"]),
    ("supported", "gachi", ["같이", "가치"]),
    ("supported", "testreul sihaeng", ["test를 시행"]),
    ("supported", "Testreul sihaeng!", ["Test를 시행!"]),
    ("supported", "bap", ["밥"]),
    ("supported", "gungnip", ["국립"]),
    ("supported", "baengma", ["백마"]),
    ("supported", "APIreul test", ["API를 test"]),
    ("preservation", "Hello World!", ["Hello World!"]),
    ("preservation", "https://example.com/test?q=1", ["https://example.com/test?q=1"]),
    ("preservation", "hello@example.com", ["hello@example.com"]),
    ("preservation", "\tannyeong  hangeul\n", ["\t안녕  한글\n"]),
    ("unlisted_phonology", "eomneun", ["없는"]),
    ("unlisted_phonology", "gachihaeyo", ["같이해요"]),
    ("unlisted_phonology", "hangugeul", ["한국을"]),
    ("unlisted_phonology", "joneun", ["조는"]),
    ("unlisted_phonology", "gungmullon", ["국물론"]),
]


def main():
    engine = WorkbenchEngine()
    rows, timings = [], []
    for category, source, expected in PROBES:
        result = engine.convert(source)
        rows.append({"category": category, "input": source, "expected": expected,
                     "result": result["result"],
                     "top1": result["result"] in expected,
                     "top5": any(c["text"] in expected for c in result["candidates"])})
    for _ in range(20):
        for _, source, _ in PROBES:
            started = time.perf_counter()
            engine.convert(source)
            timings.append((time.perf_counter() - started) * 1000)
    long_started = time.perf_counter()
    engine.convert("a" * 200)
    report = {"version": engine.version,
              "scope": "Authored supported examples and unlisted probes; not a held-out general accuracy estimate. No neural model was trained.",
              "rows": rows,
              "timing": {"samples": len(timings), "warm_p50_ms": round(statistics.median(timings), 3),
                         "warm_p95_ms": round(sorted(timings)[int(len(timings) * .95)], 3),
                         "long_200_chars_ms": round((time.perf_counter() - long_started) * 1000, 3)}}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
