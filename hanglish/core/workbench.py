"""Bounded, inspectable RR baseline for the web experiment (not a complete IME).

The small authored lexicon restores selected phonological spellings. The syllable
parser is a bounded fallback, not a general inverse of Korean phonology.
"""

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import threading
import time


DATA = Path(__file__).resolve().parents[2] / "data"
SCORE_SCALE_MODEL = DATA.parent / ".omx/artifacts/phonetic-score-scale-training-v1/model.json"
PHONETIC_MODEL = (
    DATA.parent
    / ".omx/artifacts/rr-retraining-20261020/training-cpu1/phonetic_transducer.pt"
)
SCORE_SCALE_ARTIFACTS = (
    (SCORE_SCALE_MODEL, "e07462d50da0b67304a82a9a261068ccc84b8055a99b5d70af56db530d245838"),
    (PHONETIC_MODEL, "ec544c15203bb7fac49f981c1ca1f26769305d413b1a187135affbc8c0c34d62"),
    (PHONETIC_MODEL.with_suffix(".json"), "fbbf2b26b9fb501a18acb73090be7935e6b50e76a21bbad35adfa52698949109"),
)
VERSION = "workbench-1"
MAX_INPUT = 2000
BEAM = 24
ONSET = {"": 11, "g": 0, "kk": 1, "n": 2, "d": 3, "tt": 4,
         "r": 5, "l": 5, "m": 6, "b": 7, "pp": 8, "s": 9,
         "ss": 10, "j": 12, "jj": 13, "ch": 14, "k": 15,
         "t": 16, "p": 17, "h": 18}
VOWELS = dict(zip(
    "a ae ya yae eo e yeo ye o wa wae oe yo u wo we wi yu eu ui i".split(),
    range(21),
))
# Neutralized final sounds have several possible original spellings.
CODA = {"": (0,), "k": (1, 2, 3, 9, 24), "n": (4,),
        "t": (7, 19, 20, 22, 23, 25, 27), "l": (8,),
        "m": (16, 10), "p": (17, 18, 26), "ng": (21,)}
PARTICLES = {"reul": "를", "eul": "을", "neun": "는", "eun": "은",
             "eseo": "에서", "e": "에", "wa": "와", "gwa": "과",
             "ro": "로", "euro": "으로", "do": "도", "man": "만",
             "ga": "가", "i": "이", "ui": "의"}
PROTECTED = re.compile(r"`[^`\n]+`[A-Za-z]*[.,!?]?|(?:https?://|www\.)\S+|\S+@\S+|\S*[_/\\\d]\S*", re.I)
TOKENS = re.compile(r"`[^`\n]+`[A-Za-z]*[.,!?]?|\s+|[^\s]+")
LETTERS = re.compile(r"[A-Za-z]+|[^A-Za-z]+")


@dataclass(frozen=True)
class Option:
    text: str
    score: float
    kind: str


def best(options, limit=BEAM):
    unique = {}
    for option in options:
        previous = unique.get(option.text)
        if previous is None or previous.score < option.score:
            unique[option.text] = option
    return sorted(unique.values(), key=lambda item: (-item.score, item.text))[:limit]


@lru_cache(maxsize=1024)
def syllables(token):
    """Position beam avoids the legacy parser's exponential full enumeration."""
    if not token or len(token) > 40:
        return ()
    paths = [[] for _ in range(len(token) + 1)]
    paths[0] = [Option("", 0, "syllable")]
    for pos in range(len(token)):
        active = best(paths[pos])
        if not active:
            continue
        for onset, cho in ONSET.items():
            if not token.startswith(onset, pos):
                continue
            mid = pos + len(onset)
            for vowel, jung in VOWELS.items():
                if not token.startswith(vowel, mid):
                    continue
                end = mid + len(vowel)
                for coda, finals in CODA.items():
                    if not token.startswith(coda, end):
                        continue
                    stop = end + len(coda)
                    for preference, jong in enumerate(finals):
                        char = chr(0xAC00 + cho * 588 + jung * 28 + jong)
                        for item in active:
                            paths[stop].append(Option(
                                item.text + char,
                                item.score - 1 - preference * .15,
                                "syllable",
                            ))
                    paths[stop] = best(paths[stop])
    return tuple(best(paths[-1], 12))


def _neural_readiness():
    from hanglish.core.neural_converter import neural_readiness
    available, reason = neural_readiness()
    if not available:
        return available, reason
    try:
        valid = all(hashlib.sha256(path.read_bytes()).hexdigest() == digest
                    for path, digest in SCORE_SCALE_ARTIFACTS)
    except OSError:
        valid = False
    if not valid:
        return False, "검증된 음운·점수 보정 모델 파일이 필요합니다."
    return True, reason


class WorkbenchEngine:
    def __init__(self):
        files = {name: (DATA / name).read_bytes() for name in
                 ("workbench_lexicon.json", "common_english.txt", "it_terms.txt")}
        self.lexicon = json.loads(files["workbench_lexicon.json"])
        self.english = set(files["common_english.txt"].decode().split())
        self.english.update(files["it_terms.txt"].decode().lower().split())
        self.english.update({"test", "gain", "api", "hello", "world", "openai"})
        fingerprint = hashlib.sha256(Path(__file__).read_bytes())
        for content in files.values():
            fingerprint.update(content)
        self.version = VERSION + "-" + fingerprint.hexdigest()[:12]
        self._neural = None
        self._morphology = None
        self._neural_lock = threading.Lock()
        self.neural_error = None

    def metadata(self):
        neural_available, neural_reason = _neural_readiness()
        if self.neural_error is not None:
            neural_available, neural_reason = False, self.neural_error
        return {
            "version": self.version,
            "max_input": MAX_INPUT,
            "engines": [
                {"id": "baseline", "name": "규칙 + 작은 사전", "available": True},
                {"id": "rules", "name": "음절 규칙만", "available": True},
                {"id": "morphology", "name": "음운 + 형태 조합", "available": True},
                {"id": "neural", "name": "학습 문맥 + 음운·형태", "available": neural_available,
                 "reason": neural_reason},
                {"id": "byt5", "name": "ByT5-small", "available": False,
                 "reason": "미학습 · 메모리 여유와 학습 데이터 확보 후 검토"},
            ],
            "limitations": [
                "학습 엔진과 음운·형태 엔진은 생산적인 조합을 지원하지만 전체 표준 발음·형태 분석기를 완성한 것은 아닙니다.",
                "학습 엔진만 문맥으로 후보 순서를 정합니다. 규칙 기준선은 문맥을 사용하지 않습니다.",
                "미등록 어휘·발음 예외·미완성 입력은 정확하지 않을 수 있습니다. 점수는 정답 확률이 아닙니다.",
            ],
        }

    def word(self, original, engine):
        token = original.lower()
        options = [Option(original, 0, "raw")]
        if token in self.english:
            options[0] = Option(original, 5, "raw")
        if engine == "baseline":
            for index, text in enumerate(self.lexicon.get(token, [])):
                options.append(Option(text, 9 - index * .2, "lexicon"))
            for suffix, hangul in PARTICLES.items():
                if token.endswith(suffix) and len(token) > len(suffix):
                    prefix = original[:-len(suffix)]
                    if len(prefix) >= 2 and prefix.lower() in self.english:
                        options.append(Option(prefix + hangul, 8, "mixed"))
        # Unknown complete syllables may be offered; incomplete/invalid tokens
        # stay intact rather than silently dropping unmatched Latin characters.
        for item in syllables(token):
            options.append(Option(item.text, 3 + item.score * .12, "syllable"))
        chosen = best(options, 6)
        if not any(item.text == original for item in chosen):
            chosen[-1] = options[0]
        return chosen

    def convert(self, text, context="", engine="baseline", mode="batch", raw=False):
        start = time.perf_counter()
        if not isinstance(text, str) or len(text) > MAX_INPUT:
            raise ValueError(f"입력은 {MAX_INPUT}자 이하의 문자열이어야 합니다.")
        if not isinstance(context, str) or len(context) > 500:
            raise ValueError("왼쪽 문맥은 500자 이하의 문자열이어야 합니다.")
        if engine not in ("baseline", "rules", "morphology", "neural"):
            raise ValueError("선택한 모델은 아직 준비되지 않았습니다.")
        if mode not in ("batch", "live") or type(raw) is not bool:
            raise ValueError("지원하지 않는 입력 모드입니다.")
        if any(0xD800 <= ord(char) <= 0xDFFF for char in text + context):
            raise ValueError("입력과 문맥에는 올바른 유니코드 문자열이 필요합니다.")
        if engine in ("morphology", "neural"):
            if engine == "neural" and self.neural_error is not None:
                raise ValueError(self.neural_error)
            if engine == "neural" and (self._neural is None or not self._neural.models_loaded):
                available, reason = _neural_readiness()
                if not available:
                    raise ValueError(reason)
            with self._neural_lock:
                if engine == "neural":
                    if self._neural is None:
                        from hanglish.core.score_scale_converter import ScoreScaleConverter
                        self._neural = ScoreScaleConverter(
                            score_scale_checkpoint=SCORE_SCALE_MODEL,
                            phonetic_checkpoint=PHONETIC_MODEL)
                    converter = self._neural
                else:
                    if self._morphology is None:
                        from hanglish.core.ctc_union_converter import CTCUnionConverter
                        self._morphology = CTCUnionConverter()
                    converter = self._morphology
            return converter.convert(text, context, engine, mode, raw)
        sequences = [("", 0., [])]
        warnings = []
        fragmented = len(LETTERS.findall(text)) > 128
        if fragmented:
            warnings.append("128개를 넘는 문자 구간은 빠른 응답을 위해 전체 원문으로 보존했습니다.")
        if raw or fragmented:
            segments = [{"input_start": 0, "input_end": len(text), "source": text, "text": text, "kind": "raw"}]
            sequences = [(text, 0., segments)]
        for match in (() if raw or fragmented else TOKENS.finditer(text)):
            chunk = match.group()
            if raw or chunk.isspace() or PROTECTED.fullmatch(chunk):
                pieces = [(chunk, match.start(), [Option(chunk, 0, "raw")])]
            else:
                pieces = []
                for piece in LETTERS.finditer(chunk):
                    value = piece.group()
                    options = self.word(value, engine) if value.isascii() and value.isalpha() else [Option(value, 0, "raw")]
                    if len(value) > 40 and value.isascii() and value.isalpha():
                        warnings.append("40자를 넘는 연속 영문 토큰은 원문으로 보존했습니다.")
                    pieces.append((value, match.start() + piece.start(), options))
            for value, offset, options in pieces:
                expanded = []
                for current, score, segments in sequences:
                    for option in options:
                        segment = {"input_start": offset, "input_end": offset + len(value),
                                   "source": value, "text": option.text, "kind": option.kind}
                        expanded.append((current + option.text, score + option.score, segments + [segment]))
                unique = {}
                for item in sorted(expanded, key=lambda item: (-item[1], item[0])):
                    unique.setdefault(item[0], item)
                sequences = list(unique.values())[:BEAM]
        candidates = [{"text": value, "score": round(score, 3), "segments": segments}
                      for value, score, segments in sequences[:5]]
        if not any(item["text"] == text for item in candidates):
            original = {"text": text, "score": None,
                        "segments": [{"input_start": 0, "input_end": len(text),
                                      "source": text, "text": text, "kind": "raw"}]}
            candidates = candidates[:4] + [original]
        return {"input": text, "context": context, "engine": engine, "mode": mode,
                "raw": raw, "version": self.version, "candidates": candidates,
                "result": candidates[0]["text"], "context_used": False,
                "warnings": list(dict.fromkeys(warnings)),
                "engine_ms": round((time.perf_counter() - start) * 1000, 3)}
