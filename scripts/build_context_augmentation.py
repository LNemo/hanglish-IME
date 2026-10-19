#!/usr/bin/env python3
"""Build independently authored, train-only semantic-context augmentation.

The output is a sidecar: this script never modifies the frozen challenge corpus.
Every label is checked against homophone-context senses present in train.jsonl.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = ROOT / "data/challenge/train.jsonl"
DEFAULT_OUTPUT = ROOT / "data/challenge/context-augmentation/train.jsonl"
PROVENANCE = "independently-authored-train-only-post-first-test-exposure"
GENERATOR_VERSION = 1


# These clues were written from the meanings represented in the frozen train
# split.  They are intentionally complete sentences and do not paraphrase the
# controlled templates already present there.
CONTEXTS: dict[tuple[str, str], tuple[str, ...]] = {
    ("bak", "밖"): (
        "비가 그치자 아이들이 현관문을 밀고 마당으로 뛰어나갔다.",
        "실내 공기가 답답해서 잠깐 건물 외부로 나가 바람을 쐬었다.",
        "택배 상자는 집 안으로 들이지 않고 대문 앞에 두었다.",
        "창 너머에서는 새들이 전깃줄 위를 오가고 있었다.",
        "회의실을 벗어나 복도 건너 야외 휴게 공간에서 통화했다.",
        "강아지는 산책하자는 말을 듣자 출입문 쪽으로 달려갔다.",
        "매장 내부보다 거리 쪽이 훨씬 밝고 시원했다.",
        "텐트 안에 있던 사람들이 하나둘 들판으로 나왔다.",
        "운동화에 흙이 묻어서 현관 바깥에 벗어 두었다.",
        "소방 훈련이 시작되자 모두 건물을 빠져나와 공터에 모였다.",
        "유리문 건너편의 야외 좌석은 아직 비어 있었다.",
        "고양이가 집을 나가 골목을 돌아다니다 저녁에 돌아왔다.",
    ),
    ("bak", "박"): (
        "그 선수의 성은 한 글자이고 이름은 두 글자다.",
        "출석부에서 성씨가 김 다음에 적힌 학생을 찾았다.",
        "명함에는 한국에서 널리 쓰이는 외자 성이 먼저 인쇄되어 있었다.",
        "새 담임의 성함 가운데 맨 앞 글자를 확인했다.",
        "족보를 펼쳐 같은 성을 가진 조상들의 이름을 살폈다.",
        "예약자는 성을 먼저 말한 뒤 자신의 이름을 덧붙였다.",
        "야구 중계에서 해설자는 그 투수의 성만 짧게 불렀다.",
        "논문 저자 목록에는 같은 성씨를 가진 연구자가 셋 있었다.",
        "병원 접수대에서 환자의 성과 생년월일을 물었다.",
        "반장은 새 전학생의 성을 칠판에 한 글자로 적었다.",
        "그 가문의 한 글자 성씨를 주민 명부에서 확인했다.",
        "도서관은 저자의 성을 기준으로 책을 다시 분류했다.",
    ),
    ("bit", "빛"): (
        "해 질 무렵 등대에서 밝은 광선이 바다 위로 길게 뻗었다.",
        "커튼을 열자 아침 햇살이 방 안을 환하게 채웠다.",
        "정전이 끝나고 전구에서 다시 환한 기운이 퍼졌다.",
        "손전등을 켜 어두운 계단의 발밑을 밝혔다.",
        "구름 사이로 내려온 햇살이 호수 표면에서 반짝였다.",
        "무대 조명이 배우의 얼굴을 선명하게 비추었다.",
        "멀리 있는 별에서 출발한 광선이 지구에 닿았다.",
        "휴대전화 화면의 밝기를 낮추자 눈이 한결 편해졌다.",
        "촛불이 흔들리며 벽에 따뜻한 밝음을 드리웠다.",
        "사진가는 역광을 피하려고 태양의 방향을 확인했다.",
        "프리즘을 통과한 햇살이 여러 색으로 갈라졌다.",
        "터널 끝에서 희미한 밝음이 보여 출구를 알아챘다.",
    ),
    ("bit", "빚"): (
        "사업 자금을 빌린 사람은 매달 원금과 이자를 갚고 있다.",
        "신용카드 대금을 미루면서 갚아야 할 금액이 커졌다.",
        "은행 대출을 모두 상환한 뒤 담보 설정이 해지되었다.",
        "친구에게 빌린 돈을 월급날에 돌려주기로 약속했다.",
        "가계부에는 아직 갚지 못한 채무가 따로 기록되어 있다.",
        "높은 이자 때문에 대출 잔액이 좀처럼 줄지 않았다.",
        "상속인은 재산뿐 아니라 남아 있는 채무도 확인했다.",
        "식당 외상값을 치르고 장부에서 이름을 지웠다.",
        "회사는 차입금을 조기에 상환해 재정 부담을 덜었다.",
        "보증을 잘못 서서 다른 사람의 채무까지 떠안게 되었다.",
        "농부는 수확물을 판 돈으로 농기계 대금을 갚았다.",
        "상담사는 여러 대출을 하나로 합치는 방법을 설명했다.",
    ),
    ("bit", "빗"): (
        "미용사는 엉킨 머리카락을 촘촘한 살로 천천히 풀었다.",
        "아이는 등교 전에 머리를 가지런히 정돈할 도구를 찾았다.",
        "젖은 머리에는 살이 넓게 벌어진 손질 도구를 사용했다.",
        "화장대 서랍에는 머리 정돈용품과 거울이 함께 들어 있다.",
        "반려견의 털을 정리하려고 전용 손질 도구를 꺼냈다.",
        "긴 머리가 엉키지 않도록 끝부분부터 조심스럽게 정돈했다.",
        "촘촘한 이 사이에 빠진 머리카락을 깨끗이 제거했다.",
        "무대에 오르기 전 분장사가 배우의 머리를 단정하게 다듬었다.",
        "나무 손잡이가 달린 머리 손질 도구가 세면대 옆에 놓여 있다.",
        "정전기가 덜 생기는 재질의 머리 정돈용품을 골랐다.",
        "머리카락의 가르마를 반듯하게 내기 위해 가는 살을 사용했다.",
        "여행 가방에 칫솔과 머리 손질 도구를 챙겨 넣었다.",
    ),
    ("gachi", "가치"): (
        "감정사는 오래된 그림이 지닌 경제적 중요성을 평가했다.",
        "이 선택이 공동체에 얼마나 쓸모 있는지 따져 보았다.",
        "낡은 시계는 돈으로 환산하기 어려운 추억의 의미가 있다.",
        "연구 결과의 학문적 중요성이 국제 학회에서 인정받았다.",
        "소비자는 가격에 견주어 제품이 주는 효용을 살폈다.",
        "문화재의 역사적 의미는 시장 가격만으로 정할 수 없다.",
        "봉사 활동에서 얻은 경험의 소중함을 뒤늦게 깨달았다.",
        "투자자는 재무 자료를 바탕으로 기업이 지닌 경제적 중요성을 분석했다.",
        "재활용은 버려진 물건에 새로운 쓸모를 더하는 일이다.",
        "두 제안 가운데 장기적으로 더 의미 있는 쪽을 골랐다.",
        "희소성과 보존 상태가 골동품의 값어치를 좌우한다.",
        "작은 약속을 지키는 태도의 중요성을 아이에게 설명했다.",
    ),
    ("gachi", "같이"): (
        "친구와 나란히 앉아 공연이 시작되기를 기다렸다.",
        "두 사람은 한 팀이 되어 무거운 책상을 옮겼다.",
        "가족 모두가 함께 저녁을 준비하고 식탁에 둘러앉았다.",
        "동료와 동행하여 낯선 거래처를 방문했다.",
        "아이들은 한데 모여 같은 노래를 큰 소리로 불렀다.",
        "우산 하나를 함께 쓰고 정류장까지 걸어갔다.",
        "선생님과 학생들이 힘을 모아 교실을 꾸몄다.",
        "두 연주자는 호흡을 맞추며 한 무대에서 공연했다.",
        "이웃들과 다 함께 눈 쌓인 골목을 치웠다.",
        "출장길에는 통역사가 동행해 회의를 도왔다.",
        "형제는 한집에서 지내며 집안일을 서로 나누었다.",
        "연구팀이 공동으로 실험 결과를 검토했다.",
    ),
    ("got", "곧"): (
        "안내 방송은 열차가 잠시 후 도착한다고 알렸다.",
        "먹구름이 몰려와 머지않아 비가 내릴 듯했다.",
        "공연 시작 시간이 얼마 남지 않아 관객들이 자리에 앉았다.",
        "주전자가 끓기 직전이라 불을 약하게 줄였다.",
        "해가 지려 하자 산책하던 사람들이 발길을 돌렸다.",
        "신호등이 금방 바뀔 것 같아 횡단보도 앞에서 기다렸다.",
        "마감이 코앞으로 다가와 모두 작업을 서둘렀다.",
        "택배 기사가 이내 도착한다는 문자를 보냈다.",
        "새 학기가 조만간 시작되어 교재를 미리 준비했다.",
        "의사는 환자가 머지않아 퇴원할 수 있다고 말했다.",
        "영화가 바로 시작된다는 안내가 화면에 나타났다.",
        "겨울이 다가오면 이 호수도 얼마 안 있어 얼어붙는다.",
    ),
    ("got", "곳"): (
        "지도에서 우리가 만나기로 한 장소를 다시 확인했다.",
        "햇볕이 잘 드는 자리를 골라 화분을 옮겨 놓았다.",
        "여행자는 조용히 쉴 수 있는 숙소를 찾았다.",
        "사고가 난 지점 주변에 안전선을 설치했다.",
        "도서관에서 가장 조용한 구역은 맨 안쪽 열람실이다.",
        "주소를 잘못 입력해 전혀 다른 장소에 도착했다.",
        "강물이 얕은 지점을 골라 조심스럽게 건넜다.",
        "빈자리가 없어 가방을 내려놓을 자리조차 없었다.",
        "사진 속 풍경이 촬영된 장소를 현지인에게 물었다.",
        "대피소는 위험할 때 사람들이 머무를 수 있는 공간이다.",
        "시장 입구에서 물건을 보관할 자리를 안내받았다.",
        "산 정상에는 도시 전체를 내려다볼 수 있는 지점이 있다.",
    ),
    ("mat", "맛"): (
        "요리사는 국물을 한 숟갈 떠서 간과 풍미를 확인했다.",
        "잘 익은 복숭아를 베어 물자 달콤한 느낌이 퍼졌다.",
        "레몬을 넣은 차에서 새콤한 풍미가 났다.",
        "소금을 너무 많이 넣어 음식이 지나치게 짰다.",
        "오래 끓인 육수는 깊고 진한 풍미를 냈다.",
        "감기에 걸려 혀로 음식의 풍미를 거의 느끼지 못했다.",
        "새 원두로 내린 커피는 고소하면서 약간 쌉쌀했다.",
        "아이는 처음 먹어 본 나물의 향미가 낯설다고 했다.",
        "양념을 숙성하자 매운 느낌이 한결 부드러워졌다.",
        "제철 딸기는 설탕을 넣지 않아도 충분히 달았다.",
        "심사위원들은 소스의 향과 풍미를 차례로 평가했다.",
        "찬 음식은 따뜻할 때보다 간이 약하게 느껴졌다.",
    ),
    ("mat", "맞"): (
        "학생이 고른 답이 옳다는 뜻의 용언에서 어간만 적는다.",
        "두 부품의 크기가 일치한다는 뜻의 용언에서 어간을 적는다.",
        "목격자의 설명이 현장 기록과 일치함을 나타내는 용언의 어간을 쓴다.",
        "계산 결과가 정답표의 수치와 같다는 뜻의 용언에서 어간을 고른다.",
        "주장의 근거가 사실에 부합함을 나타내는 용언의 어간을 적는다.",
        "열쇠가 자물쇠의 홈에 꼭 들어가는 상황을 나타내는 용언의 어간을 쓴다.",
        "예보가 실제 날씨와 일치했다는 뜻의 용언에서 어간만 남긴다.",
        "퍼즐 조각이 빈 공간에 정확히 들어간다는 뜻의 용언에서 어간을 적는다.",
        "서류의 주소가 신분증 정보와 동일함을 나타내는 용언의 어간을 쓴다.",
        "추리한 범인이 실제 범인이라 판단이 옳았음을 나타내는 용언의 어간을 적는다.",
        "시계 두 개의 시간이 일치한다는 뜻의 용언에서 어간을 고른다.",
        "실험값이 연구자의 가설에 부합함을 나타내는 용언의 어간만 쓴다.",
    ),
    ("mok", "목"): (
        "오래 고개를 숙이고 일했더니 머리 아래가 뻐근했다.",
        "가수는 공연을 마친 뒤 발성 기관이 피로해져 말을 아꼈다.",
        "추운 날에는 머리와 몸통 사이를 두꺼운 천으로 감쌌다.",
        "의사는 삼킬 때 아픈 신체 부위를 살펴보았다.",
        "셔츠 칼라가 닿는 부분에 땀이 차서 단추를 풀었다.",
        "잠을 잘못 자서 고개를 좌우로 돌리기 힘들었다.",
        "기린은 머리 아래의 긴 부위를 뻗어 잎을 먹었다.",
        "찬바람을 오래 맞은 뒤 음성이 쉬고 기침이 났다.",
        "사진을 찍으려고 고개 아래에 카메라 끈을 걸었다.",
        "긴장한 배우가 침을 삼키며 발성을 가다듬었다.",
        "운동 전에는 고개와 어깨 사이 근육도 충분히 풀어야 한다.",
        "간호사는 환자의 머리 바로 아래에서 맥박을 확인했다.",
    ),
    ("mok", "몫"): (
        "케이크를 똑같이 나눈 뒤 각자에게 돌아갈 부분을 건넸다.",
        "공동 수익에서 자신의 배당분이 얼마인지 계산했다.",
        "열두 개를 세 명에게 나누니 한 사람당 네 개가 돌아갔다.",
        "팀원들은 각자 맡은 책임과 할 일을 끝까지 수행했다.",
        "유산을 분배한 뒤 상속인마다 받을 부분이 정해졌다.",
        "나눗셈 문제에서 피제수를 제수로 나눈 결과를 적었다.",
        "공동 구매 비용 중 내가 부담할 금액을 송금했다.",
        "수확한 감자를 가족 수에 맞춰 같은 분량으로 나누었다.",
        "그는 성공에 기여한 만큼 정당한 지분을 요구했다.",
        "각 부서에 배정된 예산의 비율이 서로 달랐다.",
        "형제들은 집안일에서 자신에게 돌아온 역할을 맡았다.",
        "매출을 투자 비율대로 나누어 각자의 배당분을 정했다.",
    ),
    ("nat", "낫"): (
        "농부는 굽은 날이 달린 도구로 논둑의 풀을 베었다.",
        "추수를 앞두고 벼를 자를 농기구의 날을 갈았다.",
        "손잡이를 단단히 잡고 잡초 줄기를 밑동 가까이 잘라 냈다.",
        "곳간 벽에는 반달처럼 휘어진 풀베기 도구가 걸려 있었다.",
        "농기구의 날이 무뎌져 마른 줄기가 잘 끊어지지 않았다.",
        "일꾼들은 황금빛 벼를 한 움큼씩 잡아 베기 시작했다.",
        "할아버지는 풀을 벨 때 쓰는 전통 연장을 숫돌에 갈았다.",
        "휘어진 쇠날에 손이 닿지 않도록 덮개를 씌웠다.",
        "산소 주변의 무성한 잡초를 손도구로 정리했다.",
        "수확철 농부의 허리춤에는 벼 베는 연장이 매달려 있었다.",
        "풀숲을 정리한 뒤 날붙이를 닦아 창고에 보관했다.",
        "짧은 손잡이 끝의 굽은 칼날로 보리 줄기를 잘랐다.",
    ),
    ("nat", "낮"): (
        "해가 떠 있는 동안에는 거리의 간판 불이 꺼져 있었다.",
        "정오 무렵 기온이 올라 사람들은 그늘을 찾았다.",
        "밤샘 근무를 마친 직원은 밝은 시간에 잠을 잤다.",
        "태양이 머리 위에 오는 정오에는 그림자가 짧았다.",
        "겨울에는 해가 비치는 시간이 여름보다 짧다.",
        "올빼미는 밝은 동안 쉬고 어두워지면 활동한다.",
        "아침과 저녁 사이에 박물관을 관람하기로 했다.",
        "환한 시간대의 거리 풍경을 사진에 담았다.",
        "밤에는 닫히는 산책로가 해가 뜨자 다시 열렸다.",
        "정오가 지나면서 햇볕이 조금씩 기울기 시작했다.",
        "야간 요금보다 밝은 시간에 적용되는 요금이 저렴했다.",
        "해가 지기 전에 일을 마치려고 오전부터 서둘렀다.",
    ),
    ("nat", "낯"): (
        "오랜만에 만난 친구의 얼굴 모습이 많이 달라져 있었다.",
        "처음 보는 사람들 앞이라 어색하고 긴장되었다.",
        "사진을 보고도 수염을 기른 지인의 얼굴을 알아보지 못했다.",
        "새로운 환경에서 처음 마주친 얼굴들이 익숙하지 않았다.",
        "배우의 얼굴 생김새가 어디선가 본 듯 친숙했다.",
        "아이는 모르는 사람을 보자 부모 뒤로 숨었다.",
        "매일 마주치던 이웃의 얼굴이라 금방 알아차렸다.",
        "변장한 사람의 겉모습을 유심히 살펴 신원을 확인했다.",
        "전학 첫날에는 교실의 모든 얼굴이 생소하게 느껴졌다.",
        "기억 속 인물의 얼굴 모습과 몽타주를 비교했다.",
        "몇 번 인사하고 나니 처음의 어색함이 사라졌다.",
        "접수 직원은 방문객의 얼굴과 신분증 사진을 대조했다.",
    ),
    ("ot", "옷"): (
        "기온이 내려가자 두꺼운 외투를 몸에 걸쳤다.",
        "세탁기에 셔츠와 바지를 색깔별로 나누어 넣었다.",
        "면접을 앞두고 단정한 복장을 골라 입었다.",
        "비에 젖은 의복을 벗어 건조대에 널었다.",
        "재단사는 손님의 몸에 맞게 소매 길이를 줄였다.",
        "계절이 바뀌어 여름 의류를 상자에 정리했다.",
        "아이는 단추를 채우고 겉에 재킷을 입었다.",
        "여행 가방에 갈아입을 티셔츠와 여벌 의류를 챙겼다.",
        "축제 참가자들은 전통 복장을 갖춰 입었다.",
        "매장에서 몸에 잘 맞는 바지를 입어 보았다.",
        "요리할 때 소매가 더러워지지 않도록 앞치마를 둘렀다.",
        "헌 의류를 깨끗이 세탁해 기부함에 넣었다.",
    ),
    ("ot", "옻"): (
        "장인은 나무 그릇에 천연 수액을 여러 번 칠해 광택을 냈다.",
        "칠기 표면의 검붉은 막이 벗겨져 다시 손질했다.",
        "산에서 채취한 나무 진액이 피부에 닿자 심하게 가려웠다.",
        "전통 공예가는 목재를 보호하려고 천연 도료를 발랐다.",
        "칠감으로 쓰는 나무 수액에 알레르기가 있어 채취 현장에 가지 않았다.",
        "검은색 칠을 입힌 함은 습기에도 쉽게 상하지 않았다.",
        "수액을 바른 밥상은 시간이 지나며 깊은 윤이 났다.",
        "장갑 없이 칠 작업을 한 뒤 팔에 붉은 발진이 생겼다.",
        "박물관에는 천연 도료로 마감한 오래된 공예품이 전시되었다.",
        "나무에서 받은 진액을 정제해 칠감으로 사용했다.",
        "쟁반에 얇게 도료를 올리고 완전히 마를 때까지 기다렸다.",
        "전통 칠은 목기의 결을 살리면서 표면을 단단하게 보호한다.",
    ),
    ("sut", "숯"): (
        "화로에 검은 연료를 넣자 고기가 천천히 익기 시작했다.",
        "나무를 공기 없이 구워 만든 덩어리가 붉게 달아올랐다.",
        "야영객은 불씨가 오래가는 연료로 저녁을 준비했다.",
        "재가 날리지 않도록 화덕 속 검은 덩어리를 조심히 옮겼다.",
        "식당에서는 연기 향을 내려고 참나무 연료를 사용했다.",
        "불이 약해지자 집게로 새 연료 조각을 화로에 보탰다.",
        "장인은 가마 안에서 목재를 천천히 탄화시켰다.",
        "검게 구운 나무는 습기와 냄새를 흡수하는 데도 쓰인다.",
        "불판 아래의 연료가 하얀 재로 변할 때까지 탔다.",
        "대장간에서는 높은 열을 얻기 위해 검은 연료를 피웠다.",
        "캠핑을 마친 뒤 남은 불씨에 물을 부어 완전히 껐다.",
        "상자는 나무를 태워 만든 연료 조각으로 가득 차 있었다.",
    ),
    ("sut", "숱"): (
        "미용사는 머리카락이 너무 빽빽해 안쪽을 가볍게 쳐 냈다.",
        "나이가 들면서 머리털의 양이 예전보다 줄었다.",
        "진찰 결과 머리카락이 얼마나 촘촘한지 이전 기록과 비교했다.",
        "머리카락이 촘촘한 사람은 말리는 데 시간이 오래 걸린다.",
        "의사는 갑자기 털의 양이 감소한 원인을 검사했다.",
        "가발을 고를 때 머리털의 밀도와 굵기를 함께 살폈다.",
        "아이의 머리가 빽빽해서 작은 머리핀으로는 고정되지 않았다.",
        "탈모 치료 뒤 빈 부분에 잔머리가 촘촘히 자라났다.",
        "미용사는 전체 길이는 두고 머리털의 양만 줄였다.",
        "풍성한 털을 가진 강아지는 여름마다 털을 짧게 다듬는다.",
        "조명 아래에서 두피가 얼마나 드러나는지 확인했다.",
        "머리카락의 개수와 분포가 사람마다 크게 다르다.",
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _train_contracts(path: Path) -> dict[tuple[str, str], set[tuple[str, str]]]:
    contracts: dict[tuple[str, str], set[tuple[str, str]]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            if row.get("split") != "train":
                raise ValueError(f"non-train row at {path}:{line_number}")
            if row.get("category") == "homophone_context":
                for expected in row["expected"]:
                    sense = (row["input"], expected)
                    contract = (row["boundary"], row["context_role"])
                    contracts.setdefault(sense, set()).add(contract)
    return contracts


def build_rows(train_path: Path) -> list[dict]:
    train_contracts = _train_contracts(train_path)
    train_senses = set(train_contracts)
    authored_senses = set(CONTEXTS)
    if authored_senses != train_senses:
        missing = sorted(train_senses - authored_senses)
        unknown = sorted(authored_senses - train_senses)
        raise ValueError(f"authored/train sense mismatch; missing={missing}, unknown={unknown}")

    rows = []
    for sense_number, ((roman, expected), contexts) in enumerate(sorted(CONTEXTS.items()), 1):
        contracts = train_contracts[(roman, expected)]
        if len(contracts) != 1:
            raise ValueError(f"ambiguous train contract for {(roman, expected)!r}: {contracts}")
        boundary, context_role = next(iter(contracts))
        if not 12 <= len(contexts) <= 20 or len(contexts) != len(set(contexts)):
            raise ValueError(f"invalid context inventory for {(roman, expected)!r}")
        for context_number, context in enumerate(contexts, 1):
            if expected in context:
                raise ValueError(f"context leaks its label for {(roman, expected)!r}")
            rows.append({
                "id": f"context-augmentation-{sense_number:02d}-{context_number:02d}",
                "input": roman,
                "context": context,
                "expected": [expected],
                "category": "homophone_context",
                "family": "homophone_context:semantic-usage",
                "split": "train",
                "provenance": PROVENANCE,
                "context_role": context_role,
                "boundary": boundary,
            })
    return rows


def write_sidecar(train_path: Path, output_path: Path) -> dict:
    rows = build_rows(train_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    output_path.write_text(serialized, encoding="utf-8")
    by_sense = {
        f"{roman}:{expected}": len(contexts)
        for (roman, expected), contexts in sorted(CONTEXTS.items())
    }
    try:
        source_train = str(train_path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        source_train = str(train_path)
    manifest = {
        "schema_version": 1,
        "split": "train",
        "generator": "scripts/build_context_augmentation.py",
        "generator_version": GENERATOR_VERSION,
        "generator_policy": "deterministic-independent-semantic-context-inventory",
        "count": len(rows),
        "target_count": len(CONTEXTS),
        "counts_by_target": by_sense,
        "augmentation_sha256": _sha256(output_path),
        "source_train": source_train,
        "source_train_sha256": _sha256(train_path),
        "source_scope": ["data/challenge/train.jsonl"],
        "authorship_timing": "post-first-test-exposure",
        "authorship_policy": "independent train-only authorship; no held-out examples or reports used",
        "provenance_prefix": PROVENANCE,
        "held_out_disjointness_verification": "requires-independent-attestation",
        "independent_review": "pending",
        "frozen_corpus_modified": False,
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = write_sidecar(args.train, args.output)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
