#!/usr/bin/env python3
"""Write the immutable, manually authored reasoning development set."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


PROVENANCE = "authored-reasoning-development-manual-v1.1-context-correction"
PARENT_CASES_SHA256 = "8e1c5d87c45b045d44bb36e2cffb30971ca66a388ad03b84ceeff998d1e16ba2"
SCHEMA_FIELDS = (
    "id",
    "input",
    "context",
    "expected",
    "category",
    "family",
    "split",
    "provenance",
    "context_role",
    "boundary",
)


def reasoning_case(
    source: str,
    target: str,
    context: str,
    sense: str,
    variant: int,
    *,
    stem: bool = False,
) -> dict[str, object]:
    return {
        "input": source,
        "context": context,
        "expected": [target],
        "category": "homophone_context",
        "family": f"homophone_reasoning:{sense}:definition-{variant}",
        "split": "selection_dev",
        "provenance": PROVENANCE,
        "context_role": "semantic_hint",
        "boundary": "stem_fragment" if stem else "isolated_pronunciation",
    }


# Inputs, hints, and gold strings are manually authored. Gold strings are not
# produced by the IME or any converter.
CASES: list[dict[str, object]] = [
    reasoning_case(
        "nat",
        "낫",
        "초승달처럼 휜 쇠붙이를 몸쪽으로 당기는 동작으로 대상을 끊는다.",
        "nat-sickle",
        1,
    ),
    reasoning_case(
        "nat",
        "낫",
        "자르는 부분이 안쪽 곡선에 있어 밀기보다 당길 때 작동한다.",
        "nat-sickle",
        2,
    ),
    reasoning_case(
        "nat",
        "낮",
        "하루 기온이 최고점에 가까워지는 무렵이다.",
        "nat-daytime",
        1,
    ),
    reasoning_case(
        "nat",
        "낮",
        "오전과 오후를 아울러 사람들이 주로 활동하는 시간대다.",
        "nat-daytime",
        2,
    ),
    reasoning_case(
        "nat",
        "낯",
        "사람을 알아보게 하는 눈·코·입의 생김새다.",
        "nat-face",
        1,
    ),
    reasoning_case(
        "nat",
        "낯",
        "처음 온 동네라 마주치는 인상이 모두 생소하게 느껴진다.",
        "nat-face",
        2,
    ),
    reasoning_case(
        "bit",
        "빗",
        "여러 개의 가는 살이 한 줄로 촘촘히 박힌 생활용품이다.",
        "bit-comb",
        1,
    ),
    reasoning_case(
        "bit",
        "빗",
        "엉킨 가닥 사이를 훑어 가지런한 방향으로 정돈한다.",
        "bit-comb",
        2,
    ),
    reasoning_case(
        "bit",
        "빛",
        "물체를 보게 하고 서로 다른 색을 구별하게 하는 자극이다.",
        "bit-light",
        1,
    ),
    reasoning_case(
        "bit",
        "빛",
        "진공에서 초당 약 삼십만 킬로미터로 전파된다.",
        "bit-light",
        2,
    ),
    reasoning_case(
        "bit",
        "빚",
        "현재 소비 때문에 미래 소득 일부를 내줘야 하는 상태다.",
        "bit-debt",
        1,
    ),
    reasoning_case(
        "bit",
        "빚",
        "미래에 벌 돈의 일부가 이미 다른 사람 차지가 된 셈이다.",
        "bit-debt",
        2,
    ),
    reasoning_case(
        "got",
        "곧",
        "현재와 예정된 사건 사이의 시간 간격이 매우 짧음을 나타낸다.",
        "got-soon",
        1,
    ),
    reasoning_case(
        "got",
        "곧",
        "기다림이 길지 않을 것임을 알리는 시간 부사다.",
        "got-soon",
        2,
    ),
    reasoning_case(
        "got",
        "곳",
        "어떤 일이 벌어지거나 사람이 머무는 장소나 위치를 뜻하는 명사다.",
        "got-place",
        1,
    ),
    reasoning_case(
        "got",
        "곳",
        "‘이·그·저’ 뒤에 붙어 위치 범위를 가리키는 명사다.",
        "got-place",
        2,
    ),
    reasoning_case(
        "mat",
        "맛",
        "먹을 때 달고 시고 쓰다고 구별하는 감각이다.",
        "mat-flavor",
        1,
    ),
    reasoning_case(
        "mat",
        "맛",
        "코를 막으면 일부가 덜 느껴지는 섭취 감각이다.",
        "mat-flavor",
        2,
    ),
    reasoning_case(
        "mat",
        "맞",
        "‘정답이다’라는 판단을 동사로 표현할 때 쓰는 어간이다.",
        "mat-correct-stem",
        1,
        stem=True,
    ),
    reasoning_case(
        "mat",
        "맞",
        "과녁의 중심에 화살이 적중했음을 말하는 용언의 어간이다.",
        "mat-correct-stem",
        2,
        stem=True,
    ),
    reasoning_case(
        "gachi",
        "같이",
        "‘각자’와 반대로 여럿이 동일한 행동에 참여함을 뜻한다.",
        "gachi-together",
        1,
    ),
    reasoning_case(
        "gachi",
        "같이",
        "주어들이 한 무리를 이루어 움직였음을 나타내는 부사다.",
        "gachi-together",
        2,
    ),
    reasoning_case(
        "gachi",
        "가치",
        "선택을 위해 대상이 얼마나 바람직한지 매기는 기준이다.",
        "gachi-value",
        1,
    ),
    reasoning_case(
        "gachi",
        "가치",
        "비용을 치르고도 보존할 이유가 있다고 판단하게 하는 성질이다.",
        "gachi-value",
        2,
    ),
    reasoning_case(
        "sut",
        "숯",
        "산소 공급을 제한해 유기물을 가열한 뒤 남는 탄소질 덩어리다.",
        "sut-charcoal",
        1,
    ),
    reasoning_case(
        "sut",
        "숯",
        "바비큐 열원은 되지만 연탄처럼 모양이 일정하지 않은 탄소 재료다.",
        "sut-charcoal",
        2,
    ),
    reasoning_case(
        "sut",
        "숱",
        "손으로 한 번에 잡히는 가닥의 많고 적음을 이른다.",
        "sut-hair-density",
        1,
    ),
    reasoning_case(
        "sut",
        "숱",
        "같은 면적에서 가닥 사이 빈틈이 적을수록 이것이 많다고 한다.",
        "sut-hair-density",
        2,
    ),
    reasoning_case(
        "bak",
        "밖",
        "닫힌 문 너머에서 실내에 속하지 않는 쪽이다.",
        "bak-outside",
        1,
    ),
    reasoning_case(
        "bak",
        "밖",
        "울타리를 기준으로 보호 범위에 들지 않는 쪽이다.",
        "bak-outside",
        2,
    ),
    reasoning_case(
        "bak",
        "박",
        "한자로 朴이라 표기하며 영어로 Park라고도 적는 성씨다.",
        "bak-surname",
        1,
    ),
    reasoning_case(
        "bak",
        "박",
        "찬호·세리·지성이 공통으로 가진 성이다.",
        "bak-surname",
        2,
    ),
    reasoning_case(
        "mok",
        "몫",
        "공동 성과가 열이면 기여도가 이십 퍼센트인 사람에게 둘이 돌아간다.",
        "mok-share",
        1,
    ),
    reasoning_case(
        "mok",
        "몫",
        "나눗셈 식에서 피제수와 제수 다음에 얻는 셋째 값이다.",
        "mok-share",
        2,
    ),
    reasoning_case(
        "mok",
        "목",
        "고개를 좌우로 돌릴 때 함께 움직이는 관절 부위다.",
        "mok-neck",
        1,
    ),
    reasoning_case(
        "mok",
        "목",
        "경추가 지나며 뇌와 몸 사이 신경을 연결하는 부위다.",
        "mok-neck",
        2,
    ),
    reasoning_case(
        "ot",
        "옷",
        "소매·깃·단추가 달릴 수 있고 치수에 따라 고른다.",
        "ot-clothing",
        1,
    ),
    reasoning_case(
        "ot",
        "옷",
        "계절과 격식에 따라 바깥에 드러낼 천 제품을 선택한다.",
        "ot-clothing",
        2,
    ),
    reasoning_case(
        "ot",
        "옻",
        "그릇을 오래 쓰게 하는 천연 도막의 원료다.",
        "ot-lacquer",
        1,
    ),
    reasoning_case(
        "ot",
        "옻",
        "산소와 습기에 닿아 굳으면 방수성 막이 되는 천연 재료다.",
        "ot-lacquer",
        2,
    ),
]


def encoded_cases() -> bytes:
    lines = []
    for index, case in enumerate(CASES, start=1):
        row = {"id": f"reasoning-dev-{index:04d}", **case}
        lines.append(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
    return ("\n".join(lines) + "\n").encode()


def build(output_dir: Path) -> None:
    cases_path = output_dir / "cases.jsonl"
    manifest_path = output_dir / "manifest.json"
    existing = [path for path in (cases_path, manifest_path) if path.exists()]
    if existing:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"refusing to overwrite immutable output: {names}")

    payload = encoded_cases()
    senses = Counter(
        (str(case["input"]), str(case["expected"][0])) for case in CASES
    )
    manifest = {
        "name": output_dir.name,
        "split": "selection_dev",
        "case_count": len(CASES),
        "sense_count": len(senses),
        "cases_per_sense": 2,
        "schema_fields": list(SCHEMA_FIELDS),
        "cases_sha256": hashlib.sha256(payload).hexdigest(),
        "provenance": PROVENANCE,
        "claim": "fresh authored development for harder reranker selection; not a blind or naturally sampled test set",
        "correction": {
            "kind": "context-only",
            "parent": "reasoning-dev-v1",
            "parent_cases_sha256": PARENT_CASES_SHA256,
            "corrected_case_ids": ["reasoning-dev-0015", "reasoning-dev-0029"],
            "input_changes": 0,
            "expected_korean_changes": 0,
        },
        "authoring": {
            "romanization": "nine manually authored standalone RR inputs",
            "expected_korean": "manually specified; not generated by a converter",
            "hints": "independently authored paraphrased definitions and small implications",
        },
        "source_boundaries": {
            "consulted": ["data/challenge/train.jsonl target-sense inventory only"],
            "excluded": [
                "development and test splits",
                "predictions and model reports",
                "context-augmentation data",
                "selection development baselines",
            ],
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    cases_path.write_bytes(payload)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/challenge/reasoning-dev-v1.1"),
    )
    args = parser.parse_args()
    try:
        build(args.output_dir)
    except FileExistsError as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
