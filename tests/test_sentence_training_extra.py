import json
import hashlib

from scripts.build_sentence_training_extra import (
    NearDuplicateIndex, eligible_strata, row_features, select_growth,
)
from scripts.sentence_data import read_split


def candidate(identity, group, source, target, strata, family="fam"):
    return {"source_row_id": identity, "source_row_ids": [identity], "group_id": group,
            "source_ids": [source], "focus_families": [family], "input": identity + " roman sentence",
            "target": target, "features": {}, "eligible_strata": strata}


def test_near_duplicate_index_rejects_normalized_and_contained_sentences():
    index = NearDuplicateIndex(["기존 문장은, 문장 부호가 있습니다."])
    assert index.contains("기존 문장은 문장 부호가 있습니다")
    assert index.contains("앞부분 뒤에 기존 문장은, 문장 부호가 있습니다. 뒷부분")
    assert not index.contains("내용과 길이가 충분히 다른 새로운 학습 문장입니다.")


def test_near_duplicate_index_rejects_character_fivegram_jaccard_point_eight():
    original = "가나다라마바사아자차카타파하문장입니다"
    index = NearDuplicateIndex([original])
    assert index.contains("가나다라마바사아자차카타파하문장입니댜")
    assert not index.contains("전혀다른표현으로구성된별개의학습문장입니다")


def test_feature_strata_separate_mixed_and_plain_oov_growth():
    known = {"이미", "있는"}
    mixed = row_features("이미 ABC 2026년에 새로운 표현 넷을 기록했다.", known)
    plain = row_features("낯선표현 하나와 둘과 셋과 넷을 새롭게 기록하였다.", known)
    assert "mixed_oov" in eligible_strata(mixed)
    assert "plain_oov" in eligible_strata(plain)
    assert mixed["has_ascii"] and mixed["has_digits"]


def test_selection_is_deterministic_bounded_and_document_source_disjoint():
    rows = [
        candidate("a", "g1", "src1", "봄철 강가에서 철새 무리가 북쪽 하늘로 날아갑니다.", ["structural_oov"]),
        candidate("b", "g2", "src2", "연구팀은 새로운 계산 방식을 국제 학회에서 발표했습니다.", ["mixed_oov"]),
        candidate("c", "g3", "src3", "마을 도서관은 주말마다 어린이를 위한 강좌를 엽니다.", ["plain_oov"]),
        candidate("d", "g2", "src4", "해안 기상대는 거센 바람과 높은 파도를 관측했습니다.", ["plain_oov"]),
        candidate("e", "g4", "src2", "농업 연구소는 가뭄에 강한 새 품종을 공개했습니다.", ["plain_oov"]),
    ]
    quotas = {"structural_oov": 1, "mixed_oov": 1, "plain_oov": 1}
    first, counts = select_growth(rows, quotas, NearDuplicateIndex())
    second, _ = select_growth(rows, quotas, NearDuplicateIndex())
    assert [row["source_row_id"] for row in first] == [row["source_row_id"] for row in second]
    assert counts == quotas and len(first) == 3
    assert len({row["group_id"] for row in first}) == 3
    assert len({source for row in first for source in row["source_ids"]}) == 3


def test_dataset_contract_fixture_has_source_only_model_fields(tmp_path):
    row = {"id": "s3x-train-00001", "input": "roman full sentence", "context": "",
           "expected": ["완전한 한국어 문장"], "split": "train", "input_contract": "roman_sentence_only",
           "context_role": "none"}
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps({"input_contract": "roman_sentence_only",
        "model_input_fields": ["input"], "counts": {"train": 1},
        "sha256": {"train.jsonl": digest}}))
    loaded = read_split(tmp_path, "train")[0]
    assert loaded["input_contract"] == "roman_sentence_only"
    assert loaded["context"] == ""
    assert loaded["context_role"] == "none"
