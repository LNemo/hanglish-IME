#!/usr/bin/env python3
"""Build the authored Hanglish challenge corpus.

The romanized sides in this file are source data.  They are intentionally not
produced by either the workbench converter or the forward romanizer under test.
Templates vary ordinary sentence context; they do not manufacture diversity by
changing numbers or punctuation.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "challenge"
SCHEMA_VERSION = 2
CORPUS_VERSION = "2.1"


# Complete predicates/clauses whose transcription was reviewed as a pair.  The
# first twenty items in every bank are evaluation items.  Items 20..31 provide
# extra train-only vocabulary; 32..35 are dev-only compositions.
BANKS: dict[str, list[tuple[str, str]]] = {
    "copula_assimilation": [
        ("jeoneun haksaengimnida", "저는 학생입니다"), ("geuneun seonsaengnimimnida", "그는 선생님입니다"),
        ("geunyeoneun uisaimnida", "그녀는 의사입니다"), ("uri chinguneun yeonguwonimnida", "우리 친구는 연구원입니다"),
        ("uri ius-eun gijaimnida", "우리 이웃은 기자입니다"), ("sae hoewoneun gongmuwonimnida", "새 회원은 공무원입니다"),
        ("nae dongsaeng-eun daehaksaengimnida", "내 동생은 대학생입니다"), ("geu sarameun hoewonimnida", "그 사람은 회원입니다"),
        ("uri chinguneun danwonimnida", "우리 친구는 단원입니다"), ("damdangjaneun gwallijaimnida", "담당자는 관리자입니다"),
        ("jeoneun seoryu-reul geomtohamnida", "저는 서류를 검토합니다"), ("geuneun gyeolgwa-reul hwaginhamnida", "그는 결과를 확인합니다"),
        ("geunyeoneun hangugeo-reul gongbuhamnida", "그녀는 한국어를 공부합니다"), ("geu yeonguwon-eun achimmada undonghamnida", "그 연구원은 아침마다 운동합니다"),
        ("geu gijaneun jigeum sijakamnida", "그 기자는 지금 시작합니다"), ("damdangjaneun jaryo-reul jeongnihamnida", "담당자는 자료를 정리합니다"),
        ("haksaengdeureun gyosireseo silseupamnida", "학생들은 교실에서 실습합니다"), ("urineun gajogeul saranghamnida", "우리는 가족을 사랑합니다"),
        ("jeoneun doum-e gamsahamnida", "저는 도움에 감사합니다"), ("chamgajaneun seonsaengnimkke jilmunhamnida", "참가자는 선생님께 질문합니다"),
        ("je chinguneun beonyeokgaimnida", "제 친구는 번역가입니다"), ("geubuneun seolgyejaimnida", "그분은 설계자입니다"),
        ("sae dongnyoneun sangdamwonimnida", "새 동료는 상담원입니다"), ("geu jageopjaneun jeonmungaimnida", "그 작업자는 전문가입니다"),
        ("jeoneun oneul balpyohamnida", "저는 오늘 발표합니다"), ("geuneun maeil yeonseupamnida", "그는 매일 연습합니다"),
        ("geunyeoneun jaryo-reul bogwanhamnida", "그녀는 자료를 보관합니다"), ("geu yeonguwon-eun munje-reul haegyeolhamnida", "그 연구원은 문제를 해결합니다"),
        ("geu gijaneun gogaeg-eul bangmunhamnida", "그 기자는 고객을 방문합니다"), ("damdangjaneun sajin-eul chwaryeonghamnida", "담당자는 사진을 촬영합니다"),
        ("hoegye damdangjaneun yogeum-eul hwaginhamnida", "회계 담당자는 요금을 확인합니다"), ("timjangeun gyehoeg-eul seolmyeonghamnida", "팀장은 계획을 설명합니다"),
        ("sae daepyoneun danche-reul daepyohamnida", "새 대표는 단체를 대표합니다"), ("geu sarameun bangmunjaimnida", "그 사람은 방문자입니다"),
        ("geu sarameun pyeonggajaimnida", "그 사람은 평가자입니다"), ("jeonhwa sangdamwoneun annaewonimnida", "전화 상담원은 안내원입니다"),
    ],
    "nasal_liquid_assimilation": [
        ("gungmureul meongneunda", "국물을 먹는다"), ("bammureul mandeunda", "밥물을 만든다"),
        ("eomneun jaryoreul channeunda", "없는 자료를 찾는다"), ("inneun geoseul goreunda", "있는 것을 고른다"),
        ("meongneun bangbeobeul baeunda", "먹는 방법을 배운다"), ("damneun geureuseul ssinneunda", "담는 그릇을 씻는다"),
        ("gungnip bangmulgwan-e ganda", "국립 박물관에 간다"), ("dongnibundong-eul gieokanda", "독립운동을 기억한다"),
        ("simni geomsa-reul banneunda", "심리 검사를 받는다"), ("hamni-jeogeuro pandanhamnida", "합리적으로 판단합니다"),
        ("seongnyu-reul meongneunda", "석류를 먹는다"), ("daehangno-reul geonneunda", "대학로를 걷는다"),
        ("daetongnyeong-eul mannanda", "대통령을 만난다"), ("silla yeoksa-reul baeunda", "신라 역사를 배운다"),
        ("nallo-e bureul piunda", "난로에 불을 피운다"), ("kallareul ganda", "칼날을 간다"),
        ("gangneung-euro tteonanda", "강릉으로 떠난다"), ("jeongnyujang-eseo gidarinda", "정류장에서 기다린다"),
        ("eomneun ireul mandeulji anneunda", "없는 일을 만들지 않는다"), ("jibe gamnida", "집에 갑니다"),
        ("gungmin-ege allyeotda", "국민에게 알렸다"), ("bangmun-eul yeolda", "방문을 열다"),
        ("eomma-wa mannanda", "엄마와 만난다"), ("jangnyeo-reul chingchanhaetda", "장녀를 칭찬했다"),
        ("geumnyeon-eul gieokanda", "금년을 기억한다"), ("gwalli bangbeobeul ikinda", "관리 방법을 익힌다"),
        ("seongneung-eul cheukjeonghanda", "성능을 측정한다"), ("imnyeongman heoyonghanda", "입력만 허용한다"),
        ("beomnyureul geomtohanda", "법률을 검토한다"), ("geumnyeon gyehoegeul sewotda", "금년 계획을 세웠다"),
        ("eomnyeogeul jurinda", "업력을 줄인다"), ("hamnyang-eul jaenda", "함량을 잰다"),
        ("gungnae-ro bonaenda", "국내로 보낸다"), ("seomni-reul seolmyeonghanda", "섭리를 설명한다"),
        ("jangnyeogeul kiunda", "장력을 키운다"), ("beobineul hwaginhanda", "법인을 확인한다"),
    ],
    "liaison": [
        ("hangugeoreul baeunda", "한국어를 배운다"), ("eumageul deunneunda", "음악을 듣는다"),
        ("jibe ganda", "집에 간다"), ("mun-eul yeonda", "문을 연다"),
        ("oseul ibeotda", "옷을 입었다"), ("kkocheul simneunda", "꽃을 심는다"),
        ("bateuro naganda", "밭으로 나간다"), ("apeseo gidarinda", "앞에서 기다린다"),
        ("goseul chajatda", "곳을 찾았다"), ("nameul doumyeo sanda", "남을 도우며 산다"),
        ("gapseul chireotda", "값을 치렀다"), ("ilg-eoyo", "읽어요"),
        ("anja swieoyo", "앉아 쉬어요"), ("jeolmeunireul dowatda", "젊은이를 도왔다"),
        ("nuneul balba georeotda", "눈을 밟아 걸었다"), ("neokseul ireotda", "넋을 잃었다"),
        ("kkocheul ssiseotda", "꽃을 씻었다"), ("naje chaegeul ilgeotda", "낮에 책을 읽었다"),
        ("bicheul bogo georeotda", "빛을 보고 걸었다"), ("bakkeuro nagatda", "밖으로 나갔다"),
        ("myeoch-eul gollatda", "몇을 골랐다"), ("dareul barabwatda", "달을 바라봤다"),
        ("nuneul tteotda", "눈을 떴다"), ("mureul masyeotda", "물을 마셨다"),
        ("saneul ollatda", "산을 올랐다"), ("gireul georeotda", "길을 걸었다"),
        ("son-eul jabasseoyo", "손을 잡았어요"), ("maeumeul yeoreotda", "마음을 열었다"),
        ("byeoreul boatda", "별을 보았다"), ("ireul kkeunnaetda", "일을 끝냈다"),
        ("jangeul yeoreotda", "장을 열었다"), ("mareul ieotda", "말을 이었다"),
        ("sumeul goreunda", "숨을 고른다"), ("georeumeul meomchwotda", "걸음을 멈췄다"),
        ("kkumeul irueotda", "꿈을 이루었다"), ("dabeul jeogeotda", "답을 적었다"),
    ],
    "palatalization": [
        ("gachi georeotda", "같이 걸었다"), ("gachireul pyeonggahaetda", "가치를 평가했다"),
        ("guji malhaji anatda", "굳이 말하지 않았다"), ("haedoji-reul bwatda", "해돋이를 봤다"),
        ("midaji-reul yeoreotda", "미닫이를 열었다"), ("bachi neolbeotda", "밭이 넓었다"),
        ("bateuro nagatda", "밭으로 나갔다"), ("gojideutji anatda", "곧이듣지 않았다"),
        ("ttambaji-reul mallinda", "땀받이를 말린다"), ("byeohulchi-reul sseunda", "벼훑이를 쓴다"),
        ("guchineun gwajeongida", "굳히는 과정이다"), ("dachin mun-eul yeoreotda", "닫힌 문을 열었다"),
        ("muchin sangja-reul chajatda", "묻힌 상자를 찾았다"), ("geochin seoryu-reul badatda", "걷힌 서류를 받았다"),
        ("chae bachin jajeongeoreul omgyeotda", "차에 받힌 자전거를 옮겼다"), ("kkochi areumdawotda", "꽃이 아름다웠다"),
        ("naji gireotda", "낮이 길었다"), ("bichi balgatda", "빛이 밝았다"),
        ("suchi nama itda", "숯이 남아 있다"), ("gapsi keotda", "값이 컸다"),
        ("gachi ilhaetda", "같이 일했다"), ("gachireul bigyohaetda", "가치를 비교했다"),
        ("guji dasi mutji anatda", "굳이 다시 묻지 않았다"), ("haedoji-ga balgatda", "해돋이가 밝았다"),
        ("midaji-ga dachyeotda", "미닫이가 닫혔다"), ("bachi meoreotda", "밭이 멀었다"),
        ("bate kkochi pieotda", "밭에 꽃이 피었다"), ("gojideureotda", "곧이들었다"),
        ("guchyeoseo ganghaejyeotda", "굳혀서 강해졌다"), ("dachyeoseo yeolliji anatda", "닫혀서 열리지 않았다"),
        ("muchyeotdeon geosi boyeotda", "묻혔던 것이 보였다"), ("geochyeoseo dasi boyeotda", "걷혀서 다시 보였다"),
        ("kkochi pieotda", "꽃이 피었다"), ("naji jjalbatda", "낮이 짧았다"),
        ("bichi beonjjeogyeotda", "빛이 번쩍였다"), ("gapsi ollatda", "값이 올랐다"),
    ],
    "tense_consonants": [
        ("kkocheul satda", "꽃을 샀다"), ("ttareul mannanda", "딸을 만난다"),
        ("ppang-eul meongneunda", "빵을 먹는다"), ("ssareul ssinneunda", "쌀을 씻는다"),
        ("jjigae-reul kkeurinda", "찌개를 끓인다"), ("kkum-eul kkunda", "꿈을 꾼다"),
        ("tteogeul nanunda", "떡을 나눈다"), ("ppuri-reul naerinda", "뿌리를 내린다"),
        ("ssaum-eul meomchunda", "싸움을 멈춘다"), ("jjageul channeunda", "짝을 찾는다"),
        ("kkae-reul bongneunda", "깨를 볶는다"), ("ttwieo naganda", "뛰어 나간다"),
        ("ppalli geonneunda", "빨리 걷는다"), ("sseuregi-reul beorinda", "쓰레기를 버린다"),
        ("jjokjireul bonaenda", "쪽지를 보낸다"), ("kkok yaksokanda", "꼭 약속한다"),
        ("ttokttoki malhanda", "똑똑히 말한다"), ("ppeonhan dabeul pihanda", "뻔한 답을 피한다"),
        ("ssi-reul ppurinda", "씨를 뿌린다"), ("jjolji anneunda", "쫄지 않는다"),
        ("kkori-reul gamchunda", "꼬리를 감춘다"), ("ttaereul gidarinda", "때를 기다린다"),
        ("ppyeoreul matchunda", "뼈를 맞춘다"), ("ssugeul kaenda", "쑥을 캔다"),
        ("jjim-eul mandeunda", "찜을 만든다"), ("kkeut-eul hwaginhanda", "끝을 확인한다"),
        ("ttangeul panda", "땅을 판다"), ("ppalgansaegeul chilhaetda", "빨간색을 칠했다"),
        ("ssang-eul jeongnihanda", "쌍을 정리한다"), ("jjalbeun geureul sseunda", "짧은 글을 쓴다"),
        ("kkwong-eul bwatda", "꿩을 봤다"), ("ttukkeong-eul deopda", "뚜껑을 덮다"),
        ("ppikkeut neomeojyeotda", "삐끗 넘어졌다"), ("ssikssikage georeotda", "씩씩하게 걸었다"),
        ("jjogeuro doratda", "쪽으로 돌았다"), ("kkureul damatda", "꿀을 담았다"),
    ],
    "vowel_sequences": [
        ("uija-e anjatda", "의자에 앉았다"), ("uimi-reul saenggakanda", "의미를 생각한다"),
        ("hoeui-reul sijakanda", "회의를 시작한다"), ("yeui-reul jikinda", "예의를 지킨다"),
        ("gwaireul meongneunda", "과일을 먹는다"), ("goyangi-reul kiunda", "고양이를 키운다"),
        ("oegug-euro tteonanda", "외국으로 떠난다"), ("wiro ollaganda", "위로 올라간다"),
        ("uyeonhi mannanda", "우연히 만난다"), ("yeyageul chwisohaetda", "예약을 취소했다"),
        ("yeohaeng-eul junbihanda", "여행을 준비한다"), ("gyoyugeul banneunda", "교육을 받는다"),
        ("wonin-eul bunseokanda", "원인을 분석한다"), ("wae geureonji munneunda", "왜 그런지 묻는다"),
        ("oeroum-eul dallaenda", "외로움을 달랜다"), ("wiheom-eul allyeotda", "위험을 알렸다"),
        ("uiroe-reul badatda", "의뢰를 받았다"), ("uijeongbu-e ganda", "의정부에 간다"),
        ("huimang-eul malhanda", "희망을 말한다"), ("gwiyeoun gangaji-reul bwatda", "귀여운 강아지를 봤다"),
        ("oetu-reul georeotda", "외투를 걸었다"), ("wisaeng-eul jeomgeomhanda", "위생을 점검한다"),
        ("yuyeonhage daeeunghanda", "유연하게 대응한다"), ("yuilhan dabeul chajatda", "유일한 답을 찾았다"),
        ("gwanggoreul deunneunda", "광고를 듣는다"), ("hoewon-eul bureunda", "회원을 부른다"),
        ("oebu-reul hwaginhanda", "외부를 확인한다"), ("wiim-eul cheolhoehanda", "위임을 철회한다"),
        ("uiwonege munneunda", "의원에게 묻는다"), ("uihageul gongbuhanda", "의학을 공부한다"),
        ("yuuihal jeomeul jeongnihanda", "유의할 점을 정리한다"), ("yeonguui mokjeogeul sseotda", "연구의 목적을 썼다"),
        ("hoeuiui gyeolgwa-reul allyeotda", "회의의 결과를 알렸다"), ("gwiyeoun goyangi-ga janda", "귀여운 고양이가 잔다"),
        ("yuwore yeohaenghanda", "유월에 여행한다"), ("wiheom-eul pihaetda", "위험을 피했다"),
    ],
}


ENGLISH_TERMS = [
    "test", "GraphQL", "Rust", "Swift", "Docker", "Kubernetes", "TypeScript", "OpenAI", "WebGPU", "GraphRAG",
    "Redis", "PostgreSQL", "WebSocket", "JSONL", "README", "Tailscale", "DuckDB", "SentencePiece", "PyTorch", "ONNX",
    "LangGraph", "FastAPI", "SQLite", "Svelte", "Claude", "Playwright", "NumPy", "Pandas", "HuggingFace", "Gemini",
    "MLX", "GitHub", "Cloudflare", "Figma", "WebRTC", "Notion",
]

PROTECTED_TERMS = [
    "https://example.com/api/v1", "https://openai.com/research", "http://127.0.0.1:8765/", "www.example.org/docs",
    "hello@example.com", "dev.team+ime@example.org", "snake_case", "camelCase", "HTTPServer", "parse_json()",
    "user_id", "feature/ime-ranker", "v1.2.3", "README.md", "data/train.jsonl", "C:\\Users\\guest\\notes.txt",
    "SELECT_name", "apiKey", "foo/bar/baz", "model.onnx", "ws://localhost:9000", "git@github.com:team/repo.git",
    "pytest::test_ranker", "application/json", "ko-KR", "SHA-256", "x-request-id", "build_2026_09",
    "docs/challenge-corpus.md", "npm_test", "0.0.0.0:8765", "127.0.0.1", "https://example.net?q=hanglish",
    "main.py:42", "user@example.kr", "src/model/ranker.py", "GET_/api/meta",
]

# Finite ambiguity policy: include manually audited, complete standard-dictionary
# headwords whose exact isolated standard pronunciation has this RR form.  Verb
# stems, affixes, bound grammatical forms, and particle-attached inflections are
# outside this bank.  Korean vowel length does not change RR spelling.
HOMOPHONES = [
    ("nat", ["낫", "낮", "낯", "낟", "낱"]), ("bit", ["빗", "빛", "빚"]),
    ("got", ["곧", "곳", "곶"]),
    ("jeot", ["젖", "젓"]), ("sut", ["숯", "숱"]),
    ("bak", ["박", "밖"]), ("mok", ["목", "몫"]), ("ot", ["옷", "옻"]),
    ("geot", ["것", "겉"]), ("gachi", ["같이", "가치"]),
]

HOMOPHONE_CLUES = [
    ("nat", "낫"), ("nat", "낮"), ("nat", "낯"),
    ("bit", "빗"), ("bit", "빛"), ("bit", "빚"),
    ("got", "곧"), ("got", "곳"), ("mat", "맛"), ("mat", "맞"),
    ("gachi", "같이"), ("gachi", "가치"), ("sut", "숯"), ("sut", "숱"),
    ("bak", "밖"), ("bak", "박"), ("mok", "몫"), ("mok", "목"),
    ("ot", "옷"), ("ot", "옻"),
]

# Three train, two dev, and five test clues per spelling.  Every surface is
# split-specific, decisive without the gold spelling, and a complete semantic
# hint rather than a fake left-context prefix.
SEMANTIC_CLUES = {
    "낫": ("농부가 풀베기 도구를 챙긴다.", "벼를 벨 날이 선 농기구다.", "손잡이에 굽은 날이 달렸다.", "추수 때 풀을 자르는 연장이다.", "농기구 창고에서 풀 베는 도구를 찾는다.", "아버지가 논두렁의 잡초를 벤다.", "사전은 곡식을 베는 연장으로 풀이한다.", "굽은 쇠날과 손잡이가 붙은 농기구다.", "벼 수확에 쓰는 전통 도구를 고른다.", "대장간에서 풀 베는 연장의 날을 간다."),
    "낮": ("해가 떠 있는 시간에 출발한다.", "정오가 포함되는 밝은 때다.", "밤의 반대 시간대를 말한다.", "해가 진 뒤가 아닌 시간이다.", "점심을 먹는 밝은 시간이었다.", "사전에서 밤과 대비되는 때를 찾는다.", "해가 중천에 뜬 시간대를 가리킨다.", "여름에는 밝은 시간이 길어진다.", "오전과 오후를 아우르는 때다.", "태양이 떠 있는 동안 문을 연다."),
    "낯": ("처음 보는 얼굴이라 어색하다.", "사람의 얼굴 생김새를 기억한다.", "익숙하지 않은 얼굴을 마주했다.", "얼굴이 익지 않아 머뭇거렸다.", "오랜만에 본 사람의 얼굴이 달라졌다.", "사전은 사람의 얼굴을 이르는 말로 풀이한다.", "처음 보는 얼굴 앞에서 긴장했다.", "그 배우의 얼굴이 눈에 익었다.", "사진 속 사람의 얼굴을 알아보았다.", "타인의 얼굴을 함부로 평가하지 않는다."),
    "빗": ("머리카락을 가지런히 하는 도구다.", "촘촘한 살로 엉킨 머리를 푼다.", "화장대에서 머리 손질 도구를 찾는다.", "머리를 정돈할 때 쓰는 물건이다.", "긴 머리의 엉킴을 풀 도구를 집었다.", "사전은 머리털을 다듬는 도구로 풀이한다.", "촘촘한 이가 달린 손질 도구다.", "미용사가 머리 정돈 도구를 소독한다.", "화장대 서랍의 머리 도구가 부러졌다.", "아이의 엉킨 머리를 풀 도구를 건넨다."),
    "빛": ("전등에서 밝은 기운이 퍼진다.", "어둠을 밝히는 에너지가 들어온다.", "햇살이 창문 사이로 비친다.", "깜깜한 방을 밝히는 것이 필요하다.", "등대에서 밝은 줄기가 바다로 뻗는다.", "사전은 시각을 일으키는 전자기파로 풀이한다.", "새벽의 밝은 기운이 산 너머로 번진다.", "손전등이 어두운 길을 밝혔다.", "별에서 온 밝음이 밤하늘에 남았다.", "무대 조명이 배우를 환하게 비춘다."),
    "빚": ("은행에서 빌린 돈을 갚는다.", "외상으로 생긴 채무가 남았다.", "갚아야 할 금액이 늘어났다.", "빌린 돈의 상환 기한이 다가왔다.", "사업 실패로 갚을 돈이 쌓였다.", "사전은 남에게 갚아야 할 돈으로 풀이한다.", "대출로 생긴 채무를 모두 정리했다.", "외상값을 갚지 못해 걱정한다.", "이자까지 붙은 채무를 계산한다.", "친구에게 빌린 돈이 아직 남았다."),
    "곧": ("머지않아 기차가 도착한다.", "잠시 뒤 시작된다는 뜻이다.", "시간이 얼마 남지 않았음을 나타낸다.", "조금 있으면 문을 닫는다고 알렸다.", "머지않은 때에 다시 만나기로 했다.", "사전에서 잠시 후라는 뜻의 부사를 찾는다.", "이윽고 비가 그칠 것이라고 말했다.", "잠깐 뒤 결과가 나온다고 한다.", "멀지 않은 미래를 가리키는 부사다.", "조만간 새 소식이 올 예정이다."),
    "곳": ("사람이 머무는 장소를 찾는다.", "약속한 지점을 지도에 표시한다.", "어느 장소인지 주소를 묻는다.", "물건을 둘 자리를 정한다.", "행사가 열리는 장소로 이동했다.", "사전은 일정한 자리나 지역으로 풀이한다.", "쉴 만한 장소를 주변에서 찾았다.", "태어난 지역을 소개해 달라고 했다.", "지도에서 목적지가 있는 자리를 찍는다.", "사람이 많이 모이는 장소를 피했다."),
    "맛": ("국물의 짠 정도를 느낀다.", "음식을 혀로 느끼는 감각이다.", "요리의 풍미가 깊어졌다.", "찌개의 간과 풍미를 평가했다.", "과일을 먹고 단 느낌을 말했다.", "사전은 혀로 느끼는 감각으로 풀이한다.", "새 음식의 풍미를 천천히 음미했다.", "소금이 많아 짠 느낌이 강했다.", "셰프가 요리의 간을 확인했다.", "차에서 은은한 풍미가 느껴졌다."),
    "맞": ("정답과 일치한다는 동사의 어간이다.", "옳다는 뜻의 용언에서 어미를 뗐다.", "둘이 서로 일치한다는 말의 어간이다.", "정답이라는 뜻으로 활용할 동사 어간이다.", "채점 결과 옳다고 말할 때 쓰는 어간이다.", "두 수가 일치한다는 용언의 어간을 적는다.", "사실과 부합한다는 동사에서 어미를 가렸다.", "규격에 부합한다는 말의 어간만 남겼다.", "대답이 옳음을 나타내는 용언의 어간이다.", "짝이 서로 일치한다는 동사의 어간을 고른다."),
    "같이": ("친구와 함께 운동장으로 나간다.", "두 동료가 나란히 길을 걷는다.", "일행과 동행해서 전시장에 들어간다.", "친구와 함께 가려고 한다.", "동료와 나란히 앉기로 했다.", "두 사람이 동행할 예정이다.", "가족 모두 한자리에 모였다.", "팀원이 함께 작업한다.", "친구와 어깨를 나란히 하고 선다.", "우리 조가 힘을 모아 상자를 옮긴다."),
    "가치": ("물건의 가격과 쓸모를 평가한다.", "선택이 지닌 중요성을 따져 본다.", "작품의 값어치를 논한다.", "물건의 가격과 쓸모를 평가하고 있다.", "연구가 지닌 공익적 의미를 살핀다.", "사회적 의미와 효용을 판단한다.", "투자 대상의 장기적인 쓸모를 비교한다.", "문화유산의 소중함을 다음 세대에 알린다.", "새 기술의 경제적 효용을 계산한다.", "보존할 만한 역사적 중요성을 검토한다."),
    "숯": ("불을 피울 검은 연료를 꺼낸다.", "나무를 구워 만든 연료다.", "화로에 넣는 검은 덩어리다.", "바비큐용 검은 연료를 샀다.", "나무를 탄화해 만든 연료를 고른다.", "화로 속 검은 연료가 붉게 달아올랐다.", "사전은 나무를 구워 만든 연료로 풀이한다.", "고기를 구울 때 쓰는 검은 연료다.", "가마에서 만든 탄소 덩어리를 포장한다.", "불씨를 붙일 검은 연료를 집었다."),
    "숱": ("머리카락의 양이 아주 많다.", "털의 촘촘한 정도를 살핀다.", "미용사가 머리털의 양을 묻는다.", "머리카락 양이 줄어 걱정한다.", "빽빽한 머리털의 정도를 표현한다.", "사전은 머리털 따위의 부피나 분량으로 풀이한다.", "아이의 머리카락 양이 풍성하다.", "미용사가 머리털의 많고 적음을 확인한다.", "나이가 들며 머리카락 양이 줄었다.", "가발의 털이 얼마나 촘촘한지 비교한다."),
    "밖": ("문을 열고 실외로 나간다.", "건물 안이 아닌 공간이다.", "경계의 바깥쪽을 가리킨다.", "집 안의 반대편 공간으로 이동한다.", "교실 바깥에서 친구가 기다린다.", "사전은 일정한 한도나 범위의 외부로 풀이한다.", "텐트 외부에 신발을 벗어 두었다.", "창문 너머 실외에 눈이 내린다.", "울타리 바깥쪽으로 공이 굴러갔다.", "건물 외부에서 통화를 마쳤다."),
    "박": ("한국에서 흔한 성씨를 적는다.", "성명에서 성에 해당하는 글자다.", "국립 전시관을 뜻하는 세 글자 말의 첫 음절과 같은 성씨다.", "명단에서 김 다음의 성씨를 찾는다.", "한 글자로 된 한국 성씨를 고른다.", "사전에서 넓다는 뜻의 한자 성씨를 찾는다.", "야구 선수의 성만 적어 달라고 했다.", "둥근 덩굴 열매를 이르는 두 글자 말의 마지막 음절이다.", "그 사람의 성은 이 씨가 아니라 다른 한 글자다.", "명함에 적힌 한국 성씨를 읽는다."),
    "몫": ("각자에게 돌아갈 분량을 나눈다.", "수익 중 자기에게 배정된 부분이다.", "나눗셈에서 얻은 결과를 적는다.", "공동 작업 뒤 각자의 배당분을 계산한다.", "유산에서 자신에게 돌아온 부분을 확인한다.", "사전은 나눗셈의 결과나 배당분으로 풀이한다.", "세 사람이 받을 분량을 공평하게 나눈다.", "공동 수익에서 자기 배당을 챙긴다.", "나눈 결과로 나온 수를 답란에 적는다.", "각자 책임져야 할 부분이 따로 있다."),
    "목": ("머리와 몸통 사이가 뻐근하다.", "셔츠 깃이 닿는 신체 부위다.", "노래를 많이 불러 말소리가 쉬었다.", "고개를 돌릴 때 아픈 부위다.", "의사가 머리 아래의 신체 부위를 살핀다.", "사전은 머리와 몸통을 잇는 부분으로 풀이한다.", "겨울 스카프를 두르는 신체 부위가 시리다.", "긴장해서 말소리가 잘 나오지 않았다.", "베개가 높아 머리 아래가 결렸다.", "가수가 발성 전에 성대를 푼다."),
    "옷": ("추워서 두꺼운 의복을 입는다.", "몸에 걸치는 물건을 세탁한다.", "외출할 때 입을 의복을 고른다.", "장롱에서 입을 것을 꺼냈다.", "비에 젖은 의복을 말린다.", "사전은 몸을 싸서 가리는 물건으로 풀이한다.", "계절에 맞는 의복을 준비했다.", "세탁기에 몸에 입는 물건을 넣었다.", "공연용 의상을 갈아입었다.", "단추가 떨어진 의복을 수선한다."),
    "옻": ("칠기에 바르는 나무 진액이다.", "피부 알레르기를 일으킬 수 있는 나무다.", "전통 가구에 검은 칠을 올린다.", "나무 수액으로 그릇을 칠한다.", "닿으면 피부가 오를 수 있는 식물이다.", "사전은 칠감으로 쓰는 나무의 진으로 풀이한다.", "장인이 목기에 천연 칠을 여러 번 입힌다.", "산에서 알레르기를 조심해야 하는 나무다.", "전통 공예용 검붉은 진액을 채취한다.", "피부가 민감해 그 나무 근처를 피한다."),
}

CLUE_SLICES = {"train": (0, 3), "dev": (3, 5), "test": (5, 10)}
CLUE_FAMILIES = {
    "train": ("semantic-usage", "semantic-function", "semantic-domain"),
    "dev": ("semantic-contrast", "semantic-editorial"),
    "test": ("semantic-narrative", "semantic-dictionary", "semantic-practical", "semantic-question", "semantic-domain-test"),
}


CLAUSE_PREFIXES = {
    "train": [
        (left_r + manner_r, left_h + manner_h, f"{left_id}-{manner_id}")
        for left_r, left_h, left_id in (
            ("eoje ", "어제 ", "yesterday"), ("geunal achim ", "그날 아침 ", "that-morning"),
            ("pyeongso ", "평소 ", "ordinarily"), ("hyeonjang-eseo ", "현장에서 ", "on-site"),
            ("jibeseo ", "집에서 ", "at-home"),
        )
        for manner_r, manner_h, manner_id in (
            ("", "", "plain"), ("joyonghi ", "조용히 ", "quietly"),
            ("cheoncheonhi ", "천천히 ", "slowly"), ("yeoreo beon ", "여러 번 ", "repeatedly"),
            ("honja ", "혼자 ", "alone"),
        )
    ],
    "dev": [("geumbang ", "금방 ", "recently"), ("ibeone ", "이번에 ", "occasion")],
    "test": [("oneul ", "오늘 ", "today"), ("jigeum ", "지금 ", "now"), ("dasi ", "다시 ", "again"),
             ("meonjeo ", "먼저 ", "first"), ("geuttae ", "그때 ", "at-that-time")],
}

COPULA_PREFIXES = {
    "train": [
        ("giroge ttareumyeon ", "기록에 따르면 ", "record"), ("gwangyejaui mallo-neun ", "관계자의 말로는 ", "source"),
        ("balpyo naeyongdaero ", "발표 내용대로 ", "presentation"), ("deungnok jeongbosang ", "등록 정보상 ", "registration"),
        ("inteobyu-e ttareumyeon ", "인터뷰에 따르면 ", "interview"),
    ],
    "dev": [("oneul achim ", "오늘 아침 ", "this-morning"), ("banggeum ", "방금 ", "just-now")],
    "test": [("", "", "plain"), ("oneul ", "오늘 ", "today"), ("hyeonjae ", "현재 ", "current"),
             ("jeongbo-e ttareumyeon ", "정보에 따르면 ", "information"),
             ("hwagin gyeolgwa ", "확인 결과 ", "finding")],
}

PALATAL_PREFIXES = {
    "train": [("geu hu ", "그 후 ", "afterward"), ("sasil ", "사실 ", "fact"),
              ("jeonmungaui seolmyeong-e ttareumyeon ", "전문가의 설명에 따르면 ", "expert"),
              ("mokgyeokjaui jeung-eon-e ttareumyeon ", "목격자의 증언에 따르면 ", "witness"),
              ("jaryo-reul hwaginhae boni ", "자료를 확인해 보니 ", "inspection")],
    "dev": [("bogoseo-e ttareumyeon ", "보고서에 따르면 ", "report"), ("geu sungan ", "그 순간 ", "moment")],
    "test": [("", "", "plain"), ("geuttae ", "그때 ", "then"), ("dangsi ", "당시 ", "at-time"),
             ("giroge ttareumyeon ", "기록에 따르면 ", "record"),
             ("hwagin gyeolgwa ", "확인 결과 ", "finding")],
}

ENGLISH_FRAMES = {
    "train": [
        ("{term}reul seontaekaetda", "{term}를 선택했다", "select"),
        ("{term}eseo gyeolgwareul hwaginhaetda", "{term}에서 결과를 확인했다", "source-result"),
        ("{term}do geomtohaetda", "{term}도 검토했다", "also-review"),
        ("{term}man jegeohaetda", "{term}만 제거했다", "only-remove"),
        ("{term}wa dareun dogureul hamkke sayonghaetda", "{term}와 다른 도구를 함께 사용했다", "use-with-tool"),
        ("{term}buteo baeunda", "{term}부터 배운다", "learn-first"),
        ("{term}kkaji jiwonhanda", "{term}까지 지원한다", "support-through"),
        ("{term}cheoreom dongjakanda", "{term}처럼 동작한다", "acts-like"),
        ("{term}ro jeonhwanhaetda", "{term}로 전환했다", "switch-to"),
        ("{term}e gwanhan jaryoreul jeonsonghaetda", "{term}에 관한 자료를 전송했다", "send-material-about"),
    ],
    "dev": [
        ("{term}reul jeoehaetda", "{term}를 제외했다", "exclude"),
        ("{term}reul jiwo beoryeotda", "{term}를 지워 버렸다", "delete"),
    ],
    "test": [
        ("{term}reul imnyeokaetda", "{term}를 입력했다", "enter-term"),
        ("{term}reul boksahaetda", "{term}를 복사했다", "copy-term"),
        ("{term}do geudaero pyosihaetda", "{term}도 그대로 표시했다", "display-term"),
        ("{term}reul munjangeseo eongeupaetda", "{term}를 문장에서 언급했다", "mention-term"),
        ("{term}reul dareun yongeowa bigyohaetda", "{term}를 다른 용어와 비교했다", "compare-term"),
    ],
}

PROTECTED_FRAMES = {
    "train": [
        ("{term}reul wonbon geudaero boksahaetda", "{term}를 원본 그대로 복사했다", "copy-exact"),
        ("{term}reul wonbongwa bigyohaetda", "{term}를 원본과 비교했다", "compare-original"),
        ("{term}reul byeongyeong eopsi jeojanghaetda", "{term}를 변경 없이 저장했다", "save-unchanged"),
        ("{term}reul rogeue namgyeotda", "{term}를 로그에 남겼다", "log-literal"),
        ("{term}reul dasi boksahae ilchireul hwaginhaetda", "{term}를 다시 복사해 일치를 확인했다", "round-trip"),
        ("{term}reul byeoldo paire boksahaetda", "{term}를 별도 파일에 복사했다", "copy-file"),
        ("{term}reul geudaero hwamyeone pyosihaetda", "{term}를 그대로 화면에 표시했다", "display-literal"),
        ("{term}reul wonmungwa daejohaetda", "{term}를 원문과 대조했다", "collate-source"),
        ("{term}reul mugyeolseong geomsaro hwaginhaetda", "{term}를 무결성 검사로 확인했다", "integrity-check"),
        ("{term}reul han geuljado bakkuji anatda", "{term}를 한 글자도 바꾸지 않았다", "no-change"),
    ],
    "dev": [("{term}reul chekeuseomeuro daejohaetda", "{term}를 체크섬으로 대조했다", "checksum-dev")],
    "test": [
        ("{term}reul geudaero boksahaetda", "{term}를 그대로 복사했다", "literal-copy-test"),
        ("{term}reul wonbon hyeongtaero yujihaetda", "{term}를 원본 형태로 유지했다", "preserve-form-test"),
        ("{term}reul jeojangbongwa daejohaetda", "{term}를 저장본과 대조했다", "collate-saved-test"),
        ("{term}reul haesi gapseuro hwaginhaetda", "{term}를 해시 값으로 확인했다", "hash-check-test"),
        ("{term}reul byeongyeong eopsi girokaetda", "{term}를 변경 없이 기록했다", "record-unchanged-test"),
    ],
}

AMBIGUOUS_CONTEXTS = {
    "train": [
        (f"{source} {detail}:", f"ambiguous_multi:{source_id}-{detail_id}")
        for source, source_id in (("라디오에서", "radio"), ("복도에서", "hallway"), ("통화 중에", "call"),
                                  ("강의실에서", "classroom"), ("시장 안에서", "market"))
        for detail, detail_id in (("뜻 설명 없이 들은 말", "no-meaning"), ("화면 없이 들은 낱말", "no-screen"),
                                  ("한 음절만 녹음된 말", "clipped-audio"), ("철자를 모른 채 받아쓴 말", "dictation"))
    ],
    "dev": [("잡음 섞인 방송에서 들린 말:", "ambiguous_multi:noisy-broadcast"),
            ("발음만 제시된 퀴즈의 답:", "ambiguous_multi:sound-only-quiz"),
            ("철자 선택지 없이 재생한 발음:", "ambiguous_multi:no-spelling-options"),
            ("뜻을 밝히지 않은 녹음 속 낱말:", "ambiguous_multi:meaning-withheld")],
    "test": [("철자를 보여 주지 않고 읽어 준 말:", "ambiguous_multi:hidden-spelling"),
             ("문맥 없이 녹음된 낱말:", "ambiguous_multi:context-free-recording"),
             ("동음어 받아쓰기에서 들은 말:", "ambiguous_multi:homophone-dictation"),
             ("화자가 뜻을 설명하지 않고 말한 낱말:", "ambiguous_multi:unexplained-speech"),
             ("음성 인식기가 짧게 기록한 말:", "ambiguous_multi:asr-fragment"),
             ("스피커에서 뜻 설명 없이 들린 말:", "ambiguous_multi:speaker-no-meaning"),
             ("받아쓰기 파일에 소리만 남은 낱말:", "ambiguous_multi:audio-only-file"),
             ("입 모양을 보지 못하고 들은 말:", "ambiguous_multi:no-lip-reading"),
             ("철자 단서가 없는 음성 표본:", "ambiguous_multi:no-spelling-clue"),
             ("전화 녹음에서 따로 잘린 낱말:", "ambiguous_multi:clipped-call")],
}


def row(case_id: str, source: str, context: str, expected: list[str], category: str,
        family: str, split: str, provenance: str, context_role: str, boundary: str) -> dict:
    return {"id": case_id, "input": source, "context": context, "expected": expected,
            "category": category, "family": family, "split": split,
            "provenance": provenance, "context_role": context_role, "boundary": boundary}


def normal_rows(category: str, split: str, pairs: list[tuple[str, str]]) -> list[dict]:
    if category == "copula_assimilation":
        prefixes = COPULA_PREFIXES[split]
    elif category == "palatalization":
        prefixes = PALATAL_PREFIXES[split]
    else:
        prefixes = CLAUSE_PREFIXES[split]
    selected = pairs[:20] if split == "test" else (pairs[20:32] if split == "train" else pairs[32:36])
    made = []
    for pair_index, (source, target) in enumerate(selected):
        for source_prefix, target_prefix, template in prefixes:
            # Hyphens in the authored bank are review-only morpheme markers.
            # Shipped Korean input uses ordinary spaces and attached particles.
            shipped_source = (source_prefix + source).replace("-", "")
            made.append(row("", shipped_source, "", [target_prefix + target], category,
                            f"{category}:{template}", split,
                            f"authored-{split}-rr-pair+controlled-context-template",
                            "none", "word_spaced"))
    return made


def english_rows(split: str) -> list[dict]:
    selected = ENGLISH_TERMS[:20] if split == "test" else (ENGLISH_TERMS[20:32] if split == "train" else ENGLISH_TERMS[32:36])
    made = []
    for term in selected:
        for source, target, family in ENGLISH_FRAMES[split]:
            made.append(row("", source.format(term=term), "", [target.format(term=term)], "english_particle",
                            f"english_particle:{family}", split,
                            f"authored-{split}-mixed-script-template", "none", "latin_suffix"))
    return made


def protected_rows(split: str) -> list[dict]:
    selected = PROTECTED_TERMS[:20] if split == "test" else (PROTECTED_TERMS[20:32] if split == "train" else PROTECTED_TERMS[32:36])
    made = []
    for term in selected:
        wrapped = f"`{term}`"
        for source, target, family in PROTECTED_FRAMES[split]:
            made.append(row("", source.format(term=wrapped), "", [target.format(term=wrapped)], "protected_token",
                            f"protected_token:{family}", split,
                            f"authored-{split}-protected-token-template", "none", "protected_backtick"))
    return made


def ambiguous_rows(split: str) -> list[dict]:
    made = []
    for source, expected in HOMOPHONES:
        for context, family in AMBIGUOUS_CONTEXTS[split]:
            made.append(row("", source, context, expected, "ambiguous_multi",
                            family, split, f"authored-{split}-pronunciation-equivalence-set",
                            "non_decisive_hint", "isolated_pronunciation"))
    return made


def homophone_rows(split: str) -> list[dict]:
    made = []
    start, stop = CLUE_SLICES[split]
    for source, target in HOMOPHONE_CLUES:
        clues = SEMANTIC_CLUES[target][start:stop]
        boundary = "stem_fragment" if target == "맞" else "isolated_pronunciation"
        for context, clue_family in zip(clues, CLUE_FAMILIES[split]):
            made.append(row("", source, context, [target], "homophone_context",
                            f"homophone_context:{clue_family}", split,
                            f"authored-{split}-decisive-semantic-hint",
                            "semantic_hint", boundary))
    return made


def build_split(split: str) -> list[dict]:
    rows = []
    for category in ("copula_assimilation", "nasal_liquid_assimilation", "liaison", "palatalization",
                     "tense_consonants", "vowel_sequences"):
        rows.extend(normal_rows(category, split, BANKS[category]))
    rows.extend(english_rows(split))
    rows.extend(protected_rows(split))
    rows.extend(ambiguous_rows(split))
    rows.extend(homophone_rows(split))
    for index, item in enumerate(rows, 1):
        item["id"] = f"{split}-{index:04d}"
    return rows


def lexical_coverage(train_rows: list[dict], train_sha256: str) -> dict:
    """Export train-only Hangul forms for candidate generation without test leakage."""
    tokens = Counter()
    for item in train_rows:
        for train_text in [item["context"], *item["expected"]]:
            for token in re.findall(r"[가-힣]+", train_text):
                tokens[token] += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "provenance": {
            "source_split": "train",
            "policy": "Unique Hangul runs extracted only from train.jsonl context and expected values",
            "counts": {text: count for text, count in sorted(tokens.items())},
        },
        "entries": [{"text": text, "count": count} for text, count in sorted(tokens.items())],
        "train_sha256": train_sha256,
    }


def write_jsonl(path: Path, items: list[dict]) -> None:
    path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    splits = {name: build_split(name) for name in ("train", "dev", "test")}
    assert len(splits["test"]) == 1000
    for split, items in splits.items():
        write_jsonl(OUT / f"{split}.jsonl", items)
    train_bytes = (OUT / "train.jsonl").read_bytes()
    (OUT / "train_lexicon.json").write_text(
        json.dumps(lexical_coverage(splits["train"], hashlib.sha256(train_bytes).hexdigest()),
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "corpus_version": CORPUS_VERSION,
        "counts": {split: len(items) for split, items in splits.items()},
        "category_counts": {split: dict(sorted(Counter(x["category"] for x in items).items()))
                            for split, items in splits.items()},
        "family_counts": {split: len({x["family"] for x in items}) for split, items in splits.items()},
        "generation": "authored phrase banks expanded with split-specific semantic/context templates",
        "input_policy": "Romanized inputs are source data; no project converter generated them.",
        "ambiguity_policy": ("Manually audited complete standard-dictionary headwords with the exact "
                             "isolated RR form; excludes stems, affixes, bound grammatical forms, and "
                             "particle-attached inflections."),
        "sha256": {
            name: hashlib.sha256((OUT / name).read_bytes()).hexdigest()
            for name in ("train.jsonl", "dev.jsonl", "test.jsonl", "train_lexicon.json")
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
