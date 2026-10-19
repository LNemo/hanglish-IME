#!/usr/bin/env python3
"""Write the immutable, manually authored candidate-selection development set."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


PROVENANCE = "authored-selection-development-manual-v1.1-rr-correction"
PARENT_CASES_SHA256 = "697e22a6c029495b7c4670fe43b44f7b0ffb3e351d5e341f0a8aa3fb0f4c28fb"
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


def semantic_case(
    source: str, target: str, context: str, sense: str, variant: int, *, stem: bool = False
) -> dict[str, object]:
    return {
        "input": source,
        "context": context,
        "expected": [target],
        "category": "homophone_context",
        "family": f"homophone_context:{sense}:dev-{variant}",
        "split": "selection_dev",
        "provenance": PROVENANCE,
        "context_role": "semantic_hint",
        "boundary": "stem_fragment" if stem else "isolated_pronunciation",
    }


def sentence_case(
    source: str,
    expected: str,
    category: str,
    family: str,
    *,
    boundary: str = "word_spaced",
) -> dict[str, object]:
    return {
        "input": source,
        "context": "",
        "expected": [expected],
        "category": category,
        "family": family,
        "split": "selection_dev",
        "provenance": PROVENANCE,
        "context_role": "none",
        "boundary": boundary,
    }


# Both the romanized inputs and the Korean gold strings below were written by hand.
# In particular, the gold strings are not produced by the IME or a converter.
CASES: list[dict[str, object]] = [
    semantic_case("nat", "낫", "추수철 농부가 벼를 베려고 손잡이 긴 농기구를 챙겼다.", "nat-sickle", 1),
    semantic_case("nat", "낫", "날이 무뎌 숫돌에 갈아야 하는 전통 농기구다.", "nat-sickle", 2),
    semantic_case("nat", "낮", "해가 가장 높이 뜬 시간이라 거리의 그림자가 짧다.", "nat-daytime", 1),
    semantic_case("nat", "낮", "점심을 먹고 해 지기 전에 일을 마쳤다.", "nat-daytime", 2),
    semantic_case("nat", "낯", "처음 만난 사람의 얼굴이 아직 익숙하지 않다.", "nat-face", 1),
    semantic_case("nat", "낯", "사진을 보고도 누구의 얼굴인지 알아보지 못했다.", "nat-face", 2),
    semantic_case("bit", "빗", "엉킨 머리카락을 풀려고 화장대 서랍에서 도구를 꺼냈다.", "bit-comb", 1),
    semantic_case("bit", "빗", "촘촘한 이 사이에 빠진 머리카락을 털어 냈다.", "bit-comb", 2),
    semantic_case("bit", "빛", "정전이 끝나자 전구에서 밝은 기운이 퍼졌다.", "bit-light", 1),
    semantic_case("bit", "빛", "구름 사이 햇살이 호수 표면에 반짝였다.", "bit-light", 2),
    semantic_case("bit", "빚", "은행 대출금을 매달 조금씩 갚고 있다.", "bit-debt", 1),
    semantic_case("bit", "빚", "외상값과 이자가 쌓여 상환 부담이 커졌다.", "bit-debt", 2),
    semantic_case("got", "곧", "기차가 몇 분 뒤 도착하니 승강장에서 기다리자.", "got-soon", 1),
    semantic_case("got", "곧", "회의가 잠시 후 시작된다는 안내가 나왔다.", "got-soon", 2),
    semantic_case("got", "곳", "지도에서 캠핑할 장소를 골라 표시했다.", "got-place", 1),
    semantic_case("got", "곳", "사람이 드문 조용한 장소를 찾고 있다.", "got-place", 2),
    semantic_case("mat", "맛", "국을 한 숟갈 먹고 간이 싱거운지 살폈다.", "mat-flavor", 1),
    semantic_case("mat", "맛", "레몬을 베어 물자 혀에 신 느낌이 퍼졌다.", "mat-flavor", 2),
    semantic_case("mat", "맞", "답이 옳거나 서로 일치함을 나타내는 용언의 어간이다.", "mat-correct-stem", 1, stem=True),
    semantic_case("mat", "맞", "표적에 적중함을 나타내는 용언에서 어미를 뗀 형태다.", "mat-correct-stem", 2, stem=True),
    semantic_case("gachi", "같이", "혼자 가지 않고 친구와 함께 전시회를 보았다.", "gachi-together", 1),
    semantic_case("gachi", "같이", "두 사람은 나란히 우산 하나를 쓰고 걸었다.", "gachi-together", 2),
    semantic_case("gachi", "가치", "낡은 그림의 경제적 쓸모와 중요성을 감정했다.", "gachi-value", 1),
    semantic_case("gachi", "가치", "그 연구가 사회에 주는 중요성을 평가했다.", "gachi-value", 2),
    semantic_case("sut", "숯", "나무를 태워 만든 검은 연료를 화로에 넣었다.", "sut-charcoal", 1),
    semantic_case("sut", "숯", "고기를 굽기 전에 검은 연료에 불을 붙였다.", "sut-charcoal", 2),
    semantic_case("sut", "숱", "미용사가 머리카락 양이 많아 손질이 어렵다고 했다.", "sut-hair-density", 1),
    semantic_case("sut", "숱", "털이 촘촘한 정도를 보고 건강 상태를 살폈다.", "sut-hair-density", 2),
    semantic_case("bak", "밖", "실내가 답답해 문을 열고 바깥 공기로 나갔다.", "bak-outside", 1),
    semantic_case("bak", "밖", "울타리 너머 외부 공간에서 아이들이 기다린다.", "bak-outside", 2),
    semantic_case("bak", "박", "한자로 朴이라 쓰는 한국 성씨다.", "bak-surname", 1),
    semantic_case("bak", "박", "야구 선수 찬호의 성씨를 적는다.", "bak-surname", 2),
    semantic_case("mok", "몫", "공동 수익에서 각자에게 배분된 부분을 계산했다.", "mok-share", 1),
    semantic_case("mok", "몫", "나눗셈 문제에서 피제수를 제수로 나눈 결과다.", "mok-share", 2),
    semantic_case("mok", "목", "셔츠 깃이 닿는 신체 부위가 뻐근하다.", "mok-neck", 1),
    semantic_case("mok", "목", "고개를 돌릴 때 머리 아래 신체 부위가 아팠다.", "mok-neck", 2),
    semantic_case("ot", "옷", "외출 전에 몸에 걸칠 셔츠와 바지를 골랐다.", "ot-clothing", 1),
    semantic_case("ot", "옷", "세탁기에 넣은 의류를 햇볕에 말렸다.", "ot-clothing", 2),
    semantic_case("ot", "옻", "전통 목가구 표면에 나무 진액으로 검은 칠을 했다.", "ot-lacquer", 1),
    semantic_case("ot", "옻", "그 나무 진액이 피부에 닿자 심한 발진이 생겼다.", "ot-lacquer", 2),
    sentence_case("naje haetsari ttatteutaeseo gongwone georeogatda", "낮에 햇살이 따뜻해서 공원에 걸어갔다", "sentence_homophone", "sentence_homophone:daytime"),
    sentence_case("natseon eolgureul bogo ireumi saenggangnatda", "낯선 얼굴을 보고 이름이 생각났다", "sentence_homophone", "sentence_homophone:face"),
    sentence_case("bicheul gariryeogo changmune ammak keoteuneul daratda", "빛을 가리려고 창문에 암막 커튼을 달았다", "sentence_homophone", "sentence_homophone:light"),
    sentence_case("bijeul da gapgo eunhaeng aebeul jiunda", "빚을 다 갚고 은행 앱을 지운다", "sentence_homophone", "sentence_homophone:debt"),
    sentence_case("hwaroe sutbureul piwotda", "화로에 숯불을 피웠다", "sentence_homophone", "sentence_homophone:charcoal"),
    sentence_case("miyongsaga meori suteul mani chyeonaetda", "미용사가 머리 숱을 많이 쳐냈다", "sentence_homophone", "sentence_homophone:hair-density"),
    sentence_case("suigeun se sarami gateun mokseuro nanwotda", "수익은 세 사람이 같은 몫으로 나눴다", "sentence_homophone", "sentence_homophone:share"),
    sentence_case("chuwoseo eoduun oseul du gyeop ibeotda", "추워서 어두운 옷을 두 겹 입었다", "sentence_homophone", "sentence_homophone:clothing"),
    sentence_case("jeomsim hu got chulbalhandaneun bangsongi nawatda", "점심 후 곧 출발한다는 방송이 나왔다", "sentence_homophone", "sentence_homophone:soon"),
    sentence_case("yaksokagi joeun goseul jidoeseo chajatda", "약속하기 좋은 곳을 지도에서 찾았다", "sentence_homophone", "sentence_homophone:place"),
    sentence_case("gyesan dabi manneunji dasi hwaginhaetda", "계산 답이 맞는지 다시 확인했다", "sentence_homophone", "sentence_homophone:correct"),
    sentence_case("chingurang gachi jeonyeok yeonghwareul bwatda", "친구랑 같이 저녁 영화를 봤다", "sentence_homophone", "sentence_homophone:together"),
    sentence_case("reviewreul oneul kkeunnaego teamgwa gongyuhaja", "review를 오늘 끝내고 team과 공유하자", "english_particle", "english_particle:review-team", boundary="latin_suffix"),
    sentence_case("APIga oneul bam dasi yeollyeotda", "API가 오늘 밤 다시 열렸다", "english_particle", "english_particle:api", boundary="latin_suffix"),
    sentence_case("meetingeun se siro yeongidoeeotda", "meeting은 세 시로 연기되었다", "english_particle", "english_particle:meeting", boundary="latin_suffix"),
    sentence_case("featureman teukjeong brancheseo hwalseonghwahaetda", "feature만 특정 branch에서 활성화했다", "english_particle", "english_particle:feature-branch", boundary="latin_suffix"),
    sentence_case("`https://example.com/docs`eseo annae munseoreul hwaginhae", "`https://example.com/docs`에서 안내 문서를 확인해", "protected_token", "protected_token:url", boundary="protected_backtick"),
    sentence_case("`support@example.com`euro yeongsujeungeul bonaetda", "`support@example.com`으로 영수증을 보냈다", "protected_token", "protected_token:email", boundary="protected_backtick"),
    sentence_case("`v2.4.1`eun naeil baepohal yejeongida", "`v2.4.1`은 내일 배포할 예정이다", "protected_token", "protected_token:version", boundary="protected_backtick"),
    sentence_case("`README.md`e seolchi bangbeobi jeokyeo itda", "`README.md`에 설치 방법이 적혀 있다", "protected_token", "protected_token:filename", boundary="protected_backtick"),
]


def encoded_cases() -> bytes:
    rows = []
    for index, case in enumerate(CASES, start=1):
        row = {"id": f"selection-dev-{index:04d}", **case}
        rows.append(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
    return ("\n".join(rows) + "\n").encode()


def build(output_dir: Path) -> None:
    cases_path = output_dir / "cases.jsonl"
    manifest_path = output_dir / "manifest.json"
    existing = [path for path in (cases_path, manifest_path) if path.exists()]
    if existing:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"refusing to overwrite immutable output: {names}")

    payload = encoded_cases()
    categories = Counter(str(case["category"]) for case in CASES)
    manifest = {
        "name": output_dir.name,
        "split": "selection_dev",
        "case_count": len(CASES),
        "semantic_hint_count": sum(case["context_role"] == "semantic_hint" for case in CASES),
        "sentence_count": sum(case["context_role"] == "none" for case in CASES),
        "category_counts": dict(sorted(categories.items())),
        "schema_fields": list(SCHEMA_FIELDS),
        "cases_sha256": hashlib.sha256(payload).hexdigest(),
        "provenance": PROVENANCE,
        "claim": "authored-development; not a blind or naturally sampled evaluation set",
        "correction": {
            "kind": "romanization-only",
            "parent": "selection-dev-v1",
            "parent_cases_sha256": PARENT_CASES_SHA256,
            "corrected_case_ids": [
                "selection-dev-0041",
                "selection-dev-0042",
                "selection-dev-0046",
                "selection-dev-0049",
                "selection-dev-0050",
                "selection-dev-0051",
                "selection-dev-0053",
                "selection-dev-0060",
            ],
            "expected_korean_changes": 0,
        },
        "authoring": {
            "romanization": "manually authored per-word-space RR source",
            "expected_korean": "manually specified; not generated by a converter",
        },
        "source_boundaries": {
            "consulted": ["data/challenge/train.jsonl homophone target senses only"],
            "excluded": [
                "test and dev splits",
                "model reports and predictions",
                "corpus builders",
                "context-augmentation data",
            ],
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    cases_path.write_bytes(payload)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/challenge/selection-dev-v1.1"),
    )
    args = parser.parse_args()
    try:
        build(args.output_dir)
    except FileExistsError as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
