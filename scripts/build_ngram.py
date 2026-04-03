"""
Korean syllable n-gram builder for Hanglish IME.

Generates unigram.json and bigram.json from Korean text input.
Uses a built-in minimal corpus if no external corpus is provided.

Usage:
    python scripts/build_ngram.py                     # Use built-in corpus
    python scripts/build_ngram.py --input corpus.txt  # Use external corpus file
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

HANGUL_SYLLABLE_RE = re.compile(r"[\uAC00-\uD7A3]")
DATA_DIR = Path(__file__).parent.parent / "data"

# Built-in minimal Korean corpus — common sentences covering frequent syllables.
# This provides a reasonable baseline; replace with a larger corpus for production.
BUILTIN_CORPUS = """
안녕하세요 반갑습니다 감사합니다 죄송합니다
오늘 날씨가 좋습니다 내일 비가 올 것 같습니다
저는 학생입니다 선생님 감사합니다
사랑합니다 행복합니다 건강하세요
한국어를 공부하고 있습니다 열심히 하겠습니다
이것은 무엇입니까 저것은 책입니다
어디에 가십니까 학교에 갑니다
무엇을 하고 있습니까 공부를 하고 있습니다
맛있는 음식을 먹었습니다 배가 부릅니다
친구와 함께 영화를 봤습니다 재미있었습니다
서울에서 부산까지 기차로 세 시간 걸립니다
주말에 뭐 하세요 쉬고 싶습니다
커피 한 잔 주세요 물도 주세요
이름이 뭐예요 나이가 어떻게 되세요
전화번호가 뭐예요 이메일 주소를 알려주세요
지금 몇 시예요 약속 시간에 늦었습니다
어제 회의가 있었습니다 결과가 좋았습니다
내일 시험이 있습니다 준비를 열심히 해야 합니다
가족은 몇 명이에요 부모님과 형이 있습니다
취미가 뭐예요 독서와 운동을 좋아합니다
한국 음식 중에서 김치를 제일 좋아합니다
서울은 대한민국의 수도입니다
봄에는 벚꽃이 피고 가을에는 단풍이 아름답습니다
겨울에는 눈이 오고 여름에는 비가 많이 옵니다
컴퓨터를 사용하여 프로그래밍을 합니다
인터넷으로 정보를 검색합니다
대학교에서 공학을 전공했습니다
회사에서 일하고 있습니다 직장 생활이 바쁩니다
주말에 산에 올라갔습니다 경치가 아름다웠습니다
맛있는 한국 음식을 만들어 봤습니다
우리 가족은 모두 건강합니다
아버지는 회사원이고 어머니는 선생님입니다
동생은 대학생이고 저는 직장인입니다
한국의 전통 문화는 매우 아름답습니다
설날과 추석은 한국의 큰 명절입니다
한복은 한국의 전통 의상입니다
태권도는 한국의 전통 무술입니다
불고기와 비빔밥은 유명한 한국 음식입니다
한강은 서울을 가로지르는 큰 강입니다
제주도는 아름다운 섬입니다
경복궁은 조선시대의 궁궐입니다
지하철은 편리한 교통수단입니다
버스를 타고 출근합니다
택시는 비싸지만 편합니다
비행기를 타고 여행을 갑니다
호텔에서 숙박합니다
식당에서 저녁을 먹습니다
카페에서 커피를 마십니다
서점에서 책을 삽니다
병원에 가서 진료를 받습니다
약국에서 약을 삽니다
은행에서 돈을 찾습니다
우체국에서 편지를 보냅니다
도서관에서 책을 빌립니다
공원에서 산책합니다
시장에서 과일을 삽니다
마트에서 장을 봅니다
빵집에서 빵을 삽니다
꽃집에서 꽃을 삽니다
미용실에서 머리를 잘랐습니다
세탁소에서 옷을 맡겼습니다
영화관에서 영화를 봅니다
체육관에서 운동합니다
수영장에서 수영합니다
도움이 필요하면 말씀하세요
문제가 있으면 알려주세요
시간이 있으면 같이 가요
날씨가 좋으면 소풍 가요
배가 고프면 밥을 먹어요
피곤하면 쉬세요
모르면 물어보세요
필요하면 사세요
좋아하면 하세요
싫으면 하지 마세요
한국어는 아름다운 언어입니다
한국어 문법은 영어와 많이 다릅니다
한국어에는 존댓말과 반말이 있습니다
한국어를 배우는 외국인이 많아지고 있습니다
한국 문화가 세계적으로 인기를 얻고 있습니다
한국 드라마와 영화가 해외에서 큰 인기를 끌고 있습니다
한국 가수들의 음악이 전 세계에서 사랑받고 있습니다
대한민국은 아시아의 동쪽에 위치한 나라입니다
한국의 인구는 약 오천만 명입니다
한국의 수도는 서울이고 가장 큰 도시입니다
부산은 한국에서 두 번째로 큰 도시입니다
인천은 국제공항이 있는 도시입니다
대구와 광주는 한국의 주요 도시입니다
한국은 사계절이 뚜렷한 나라입니다
한국의 여름은 덥고 습합니다
한국의 겨울은 춥고 건조합니다
감사합니다 정말 감사합니다 대단히 감사합니다
고맙습니다 정말 고맙습니다
미안합니다 정말 미안합니다 죄송합니다
실례합니다 잠깐만요 잠시만요
알겠습니다 이해했습니다 네 알겠습니다
모르겠습니다 잘 모르겠습니다
괜찮습니다 괜찮아요 아무 문제 없습니다
좋습니다 좋아요 아주 좋습니다
나쁩니다 나빠요 별로입니다
맞습니다 맞아요 그렇습니다
아닙니다 아니에요 아닌 것 같습니다
그렇습니다 그래요 네 그렇습니다
오늘은 월요일입니다 내일은 화요일입니다
이번 주는 바쁜 한 주가 될 것 같습니다
지난주에 친구를 만났습니다
다음 달에 여행을 갈 계획입니다
작년에 일본에 다녀왔습니다
올해는 새로운 것을 많이 배웠습니다
아침에 일찍 일어나는 것이 건강에 좋습니다
점심은 보통 열두 시에 먹습니다
저녁에는 가족과 함께 시간을 보냅니다
밤에는 일찍 자는 것이 좋습니다
한국 사람들은 밥을 주식으로 먹습니다
김치는 한국의 대표적인 음식입니다
된장찌개와 김치찌개는 한국인이 좋아하는 음식입니다
삼겹살은 한국에서 인기 있는 고기입니다
떡볶이는 한국의 대표적인 길거리 음식입니다
비빔밥은 밥 위에 여러 가지 나물을 올린 음식입니다
냉면은 차가운 국수로 여름에 많이 먹습니다
삼계탕은 닭에 인삼을 넣고 끓인 보양식입니다
한국에서는 어른에게 존댓말을 사용합니다
한국 사람들은 나이를 중요하게 생각합니다
한국에서는 신발을 벗고 집에 들어갑니다
한국의 교육열은 세계적으로 유명합니다
한국 학생들은 열심히 공부합니다
대학 입시는 한국에서 매우 중요합니다
한국의 기술 산업은 빠르게 성장하고 있습니다
삼성과 현대는 한국의 대표적인 기업입니다
한국은 반도체와 자동차 산업이 강합니다
인터넷 속도가 한국은 세계에서 가장 빠릅니다
한국에서 스마트폰 사용률이 매우 높습니다
서울의 지하철은 매우 편리합니다
한국의 고속철도는 서울에서 부산까지 두 시간 반이면 갑니다
한국어를 잘하고 싶습니다
매일 한국어 단어를 외우고 있습니다
한국어 발음이 어렵습니다
한국어 문법을 공부하는 중입니다
한국 친구와 한국어로 대화합니다
한국 노래를 들으면서 한국어를 배웁니다
한국 영화를 보면서 듣기 연습을 합니다
한국어 시험을 준비하고 있습니다
한국어 실력이 많이 늘었습니다
한국어로 편지를 쓸 수 있습니다
프로그래밍은 컴퓨터에게 명령을 내리는 것입니다
소프트웨어 개발자는 코드를 작성합니다
파이썬은 인기 있는 프로그래밍 언어입니다
자바스크립트는 웹 개발에 많이 사용됩니다
인공지능 기술이 빠르게 발전하고 있습니다
기계학습은 데이터에서 패턴을 찾는 기술입니다
앱 개발은 모바일 시대의 핵심 기술입니다
클라우드 컴퓨팅은 서버를 원격으로 사용하는 것입니다
데이터베이스는 정보를 저장하고 관리합니다
네트워크는 컴퓨터를 연결하는 기술입니다
"""


def extract_syllables(text: str) -> list[str]:
    return HANGUL_SYLLABLE_RE.findall(text)


def build_ngrams(text: str) -> tuple[dict, dict]:
    syllables = extract_syllables(text)

    unigram_counts = Counter(syllables)

    bigram_counts: Counter = Counter()
    for i in range(len(syllables) - 1):
        bigram_key = f"{syllables[i]}|{syllables[i+1]}"
        bigram_counts[bigram_key] += 1

    return dict(unigram_counts.most_common()), dict(bigram_counts.most_common())


def main():
    parser = argparse.ArgumentParser(description="Build Korean syllable n-gram data")
    parser.add_argument("--input", "-i", type=str, help="Path to Korean text corpus file")
    args = parser.parse_args()

    if args.input:
        corpus_path = Path(args.input)
        if not corpus_path.exists():
            print(f"Error: File not found: {args.input}", file=sys.stderr)
            sys.exit(1)
        text = corpus_path.read_text(encoding="utf-8")
        print(f"Using external corpus: {args.input}")
    else:
        text = BUILTIN_CORPUS
        print("Using built-in minimal corpus")

    unigrams, bigrams = build_ngrams(text)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    unigram_path = DATA_DIR / "unigram.json"
    bigram_path = DATA_DIR / "bigram.json"

    with open(unigram_path, "w", encoding="utf-8") as f:
        json.dump(unigrams, f, ensure_ascii=False, indent=2)

    with open(bigram_path, "w", encoding="utf-8") as f:
        json.dump(bigrams, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(unigrams)} unigrams -> {unigram_path}")
    print(f"Generated {len(bigrams)} bigrams -> {bigram_path}")


if __name__ == "__main__":
    main()
