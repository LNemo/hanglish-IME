import hashlib
import json

from scripts.probe_proposal_union import ARMS, select_sources, summarize


def test_selection_ignores_targets_and_file_order(tmp_path):
    path = tmp_path / "rows.jsonl"
    rows = [{"id": str(i), "input": "roman", "context": "ctx", "expected": ["a"]}
            for i in range(512)]
    def select():
        path.write_text("\n".join(json.dumps(row) for row in rows))
        return select_sources(path, hashlib.sha256(path.read_bytes()).hexdigest())
    first = select()
    rows.reverse()
    for row in rows:
        row["expected"] = ["changed"]
    assert first == select()
    assert len(first) == 64 and all(set(row) == {"id", "input", "context"} for row in first)


def test_metrics_keep_missing_targets_and_report_regressions():
    captures = []
    for i in range(3):
        row = {"id": str(i), "trace": {"sentence_pool": ["gold"]}}
        for arm in ARMS:
            text = "gold" if (i == 0 and arm != "proposal_union") or (i == 1 and arm == "proposal_union") else "wrong"
            row[arm] = {"result": text, "candidates": [{"text": text}], "engine_ms": 10.}
        captures.append(row)
    targets = {str(i): {"expected": ["gold"]} for i in range(3)}
    report = summarize(captures, targets)
    assert report["metrics"]["proposal_union"] == {"rows": 3, "top1": 1, "top5": 1}
    assert report["changes"]["baseline"]["top1"] == {"fix_ids": ["1"], "loss_ids": ["0"]}
