import json
import os
from pathlib import Path

# Hangul unicode offsets
HANGUL_BASE = 0xAC00
CHOSUNG_COUNT = 588
JUNGSUNG_COUNT = 28

# Chosung, Jungsung, Jongsung lists in Unicode sequence order
CHOSUNG = ["ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
JUNGSUNG = ["ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ"]
JONGSUNG = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]


class Romanizer:
    def __init__(self, rules_path: str = None):
        if not rules_path:
            # 기본 경로 (프로젝트 루트의 data/romanization_rules.json)
            base_dir = Path(__file__).parent.parent.parent
            rules_path = base_dir / "data" / "romanization_rules.json"
            
        with open(rules_path, "r", encoding="utf-8") as f:
            self.rules = json.load(f)
            
        # 매핑 캐시
        self.initial_consonants = self.rules["consonants"]["initial"]
        self.vowels = self.rules["vowels"]
        self.final_consonants = self.rules["consonants"]["final"]

    def _compose_hangul(self, cho: str, jung: str, jong: str = "") -> str:
        """초성, 중성, 종성을 합쳐서 하나의 한글 문자로 만듭니다."""
        if cho not in CHOSUNG or jung not in JUNGSUNG:
            return ""

        cho_idx = CHOSUNG.index(cho)
        jung_idx = JUNGSUNG.index(jung)
        jong_idx = JONGSUNG.index(jong) if jong in JONGSUNG else 0

        unicode_val = HANGUL_BASE + (cho_idx * CHOSUNG_COUNT) + (jung_idx * JUNGSUNG_COUNT) + jong_idx
        return chr(unicode_val)

    def romanize(self, text: str) -> str:
        """
        간략화된 예시: 'annyeong' -> '안녕' 
        기본적으로 `get_candidates` 중 첫 번째 결과를 반환합니다.
        """
        candidates = self.get_candidates(text)
        return candidates[0] if candidates else text

    def get_candidates(self, text: str) -> list[str]:
        """
        입력된 로마자 문자열 가능한 모든 한글 조합(후보 리스트)을 반환합니다.
        DFS/백트래킹을 통해 중의성(예: 'eobs' + 'eo' vs 'eob' + 'seo')을 모두 탐색 가능합니다.
        """
        text = text.lower()
        n = len(text)
        results = []

        def dfs(index: int, current_path: str):
            if index == n:
                results.append(current_path)
                return

            # 하이픈(-) 등 구분자 처리
            if text[index] == '-':
                dfs(index + 1, current_path)
                return

            # 초성 탐색 (최대 2글자)
            possible_chos = []  # (cho_jamo, length)
            for l in (2, 1):
                if index + l <= n and text[index:index+l] in self.initial_consonants:
                    possible_chos.append((self.initial_consonants[text[index:index+l]], l))
            
            # 모음으로 시작하는 경우의 초성 (ㅇ)
            if any(text[index:].startswith(v) for v in sorted(self.vowels.keys(), key=len, reverse=True)):
                possible_chos.append(("ㅇ", 0))

            if not possible_chos:
                # 아예 매칭이 안되는 경우 알파벳 그대로 출력하고 다음으로
                dfs(index + 1, current_path + text[index])
                return

            for cho_jamo, cho_len in possible_chos:
                next_idx_cho = index + cho_len
                
                # 중성 탐색 (최대 3글자)
                possible_jungs = []
                for l in (3, 2, 1):
                    if next_idx_cho + l <= n and text[next_idx_cho:next_idx_cho+l] in self.vowels:
                        possible_jungs.append((self.vowels[text[next_idx_cho:next_idx_cho+l]], l))
                
                if not possible_jungs:
                    # 초성만 있고 모음이 없는 경우, 초성 그대로 붙이고 종료 (또는 다음 글자로)
                    dfs(next_idx_cho, current_path + cho_jamo)
                    continue

                for jung_jamo, jung_len in possible_jungs:
                    next_idx_jung = next_idx_cho + jung_len
                    
                    # 1. 기본 분절 (종성 없이 바로 합성)
                    dfs(next_idx_jung, current_path + self._compose_hangul(cho_jamo, jung_jamo, ""))
                    
                    # 2. 종성 탐색 (최대 2글자)
                    for l in (2, 1):
                        if next_idx_jung + l <= n and text[next_idx_jung:next_idx_jung+l] in self.final_consonants:
                            jong_jamo = self.final_consonants[text[next_idx_jung:next_idx_jung+l]]
                            
                            # 만약 종성을 가져갔는데 뒤에 모음이 바로 오면 연음에 의해 초성이 뺏겨서 유효하지 않을 확률이 크지만,
                            # 1:1 매핑 관점의 사용자 의도를 위해 (eobs + eo -> 없어) 모두 후보로 추가합니다.
                            # 즉, '업서'(종성 1개 양보) 와 '없어'(종성 모두 유지) 두 방향 모두 생성합니다.
                            dfs(next_idx_jung + l, current_path + self._compose_hangul(cho_jamo, jung_jamo, jong_jamo))

        dfs(0, "")
        
        # Heuristic 정렬
        # 1. 길이가 짧은 문자열 조합(합칠 수 있는 건 다 합친 것) 우선
        # 2. 'ㅇ' 초성이 적게 들어간 조합 우선 (자연스러운 연음 선호)
        unique_results = list(dict.fromkeys(results))
        
        def score_candidate(hangul_str: str) -> tuple[int, int]:
            cnt_empty_cho = 0
            for char in hangul_str:
                # 한글 유니코드 범위 확인
                if 0xAC00 <= ord(char) <= 0xD7A3:
                    # 초성 인덱스 추출
                    cho_idx = (ord(char) - 0xAC00) // 588
                    if cho_idx == 11:  # 'ㅇ'의 인덱스는 11
                        cnt_empty_cho += 1
            return (len(hangul_str), cnt_empty_cho)

        unique_results.sort(key=score_candidate)
        return unique_results
