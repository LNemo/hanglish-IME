import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_natural_context_corpus import (
    EXPECTED_REVISION,
    canonicalize_occurrences,
    extract_dp_text,
    extract_mrc,
    extract_nli,
    normalize_text,
    passage_windows,
    select_and_split,
    split_isolation_assertions,
    verify_sources,
)
from collections import Counter


def test_mrc_paragraphs_share_article_source_and_title_group():
    document = {"data": [{
        "title": "  같은   제목 ", "source": "신문사", "news_category": None,
        "paragraphs": [
            {"context": "첫 번째 문장은 충분히 긴 자연어 문장입니다."},
            {"context": "두 번째 문장도 충분히 긴 자연어 문장입니다."},
        ],
    }]}
    windows = extract_mrc(document, Counter())
    assert len(windows) == 2
    assert len({window["parent_group"] for window in windows}) == 1
    assert [window["provenance"]["paragraph_index"] for window in windows] == [0, 1]
    assert windows[0]["provenance"]["category"] == ""


def test_duplicate_text_unions_parent_groups_and_cannot_cross_splits():
    shared = "중복된 자연어 문장은 그룹들을 안전하게 연결합니다."
    occurrences = [
        {"text": shared, "left_context": "", "parent_group": "a",
         "provenance": {"dataset": "mrc", "source_id": "one", "normalized_start": 0}},
        {"text": shared, "left_context": "앞 문장.", "parent_group": "b",
         "provenance": {"dataset": "nli", "source_id": "two", "normalized_start": 4}},
        {"text": "첫 그룹의 별도 자연어 문장입니다.", "left_context": "", "parent_group": "a",
         "provenance": {"dataset": "mrc", "source_id": "three", "normalized_start": 0}},
        {"text": "둘째 그룹의 별도 자연어 문장입니다.", "left_context": "", "parent_group": "b",
         "provenance": {"dataset": "nli", "source_id": "four", "normalized_start": 0}},
    ]
    rows = canonicalize_occurrences(occurrences)
    assert len(rows) == 3
    assert len({row["group_id"] for row in rows}) == 1
    shared_row = next(row for row in rows if row["text"] == shared)
    assert shared_row["duplicate_occurrences"] == 2
    assert shared_row["source_ids"] == ["mrc:one", "nli:two"]
    split_rows = select_and_split(rows)
    assert len({row["split"] for row in split_rows}) == 1


def test_left_context_uses_only_preceding_text_and_preserves_punctuation_latin_digits():
    text = "앞 문장에는 ABC-123 표기가 그대로 남습니다! 뒤 문장에도 한글이 여덟 글자 이상 있습니다. 미래 문장은 절대 보이면 안 됩니다."
    windows = passage_windows(text, "group", {
        "dataset": "fixture", "source_id": "id", "category": "test"
    }, Counter())
    assert windows[0]["left_context"] == ""
    assert windows[0]["text"] == "앞 문장에는 ABC-123 표기가 그대로 남습니다!"
    assert "ABC-123" in windows[0]["text"]
    assert windows[1]["left_context"].endswith("그대로 남습니다!")
    assert "미래 문장" not in windows[1]["left_context"]
    assert len(windows[1]["left_context"]) <= 240


def test_nfc_whitespace_and_nli_dp_allowed_fields():
    assert normalize_text("  A\u0301\t  문장  42! ") == "Á 문장 42!"
    nli = extract_nli([{
        "guid": "nli-1", "genre": "뉴스", "premise": "NLI premise 42와 한글 문장이 충분히 깁니다.",
        "hypothesis": "사용하면 안 됩니다", "gold_label": "never-use",
    }], Counter())
    assert nli[0]["provenance"]["source_id"] == "nli-1"
    assert "hypothesis" not in nli[0]["provenance"]
    dp = extract_dp_text(
        "## 칼럼명 : INDEX\\tWORD_FORM\n"
        "## klue-dp-v1_train_00001_news\tDP Header 7과 자연어 문장이 충분히 깁니다.\n"
        "1\t의존\t의존\tNNG\t0\tROOT\n", Counter())
    assert dp[0]["text"].startswith("DP Header 7")
    assert dp[0]["provenance"]["source_id"] == "klue-dp-v1_train_00001_news"


def test_selection_is_deterministic_and_happens_before_group_split():
    rows = [{
        "text": f"결정적인 자연어 선택 문장 번호 {index}입니다.",
        "left_context": "", "group_id": f"group-{index % 4}",
        "provenance": [{"dataset": "fixture", "source_id": str(index), "normalized_start": 0}],
        "source_ids": [f"fixture:{index}"], "duplicate_occurrences": 1,
    } for index in range(30)]
    first = select_and_split(rows, limit=10)
    second = select_and_split(list(reversed(rows)), limit=10)
    assert first == second
    assert len(first) == 10
    splits_by_group = {}
    for row in first:
        assert splits_by_group.setdefault(row["group_id"], row["split"]) == row["split"]


def test_split_isolation_rejects_dev_text_inside_train_left_context():
    shared = {
        "provenance": [], "source_ids": [], "duplicate_occurrences": 1,
    }
    train = [{**shared, "id": "train", "text": "훈련 자연어 문장입니다.",
              "left_context": "개발 자연어 문장입니다. 다음 내용", "group_id": "train-group", "split": "train"}]
    dev = [{**shared, "id": "dev", "text": "개발 자연어 문장입니다.",
            "left_context": "", "group_id": "dev-group", "split": "dev"}]
    with pytest.raises(AssertionError, match="dev_text_embedded_in_train_left_context_count"):
        split_isolation_assertions(train, dev)


def test_source_integrity_rejects_changed_content(monkeypatch, tmp_path):
    import scripts.build_natural_context_corpus as module

    files = {}
    total = 0
    for name in (
        "klue-dp-v1.1_train.tsv", "klue-nli-v1.1_train.json", "klue-mrc-v1.1_train.json"
    ):
        content = (name + " fixture").encode()
        (tmp_path / name).write_bytes(content)
        total += len(content)
        files[name] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    manifest = {
        **module.TRUSTED_ATTRIBUTION,
        "revision": EXPECTED_REVISION,
        "files": files,
        "transferred_bytes_this_run": total,
    }
    (tmp_path / "SOURCE.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(module, "TRUSTED_SOURCE_FILES", json.loads(json.dumps(files)))
    monkeypatch.setattr(module, "EXPECTED_SOURCE_BYTES", total)
    verify_sources(tmp_path)
    manifest["license"] = "incorrect license"
    (tmp_path / "SOURCE.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="attribution"):
        verify_sources(tmp_path)
    manifest.update(module.TRUSTED_ATTRIBUTION)
    changed = b"coordinated tampering"
    changed_name = "klue-nli-v1.1_train.json"
    (tmp_path / changed_name).write_bytes(changed)
    manifest["files"][changed_name] = {
        "bytes": len(changed), "sha256": hashlib.sha256(changed).hexdigest()
    }
    changed_total = sum(item["bytes"] for item in manifest["files"].values())
    manifest["transferred_bytes_this_run"] = changed_total
    (tmp_path / "SOURCE.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(module, "EXPECTED_SOURCE_BYTES", changed_total)
    with pytest.raises(ValueError, match="trusted pin"):
        verify_sources(tmp_path)
