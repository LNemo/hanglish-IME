import json
from pathlib import Path
import time

import pytest

from hanglish.core.workbench import MAX_INPUT, WorkbenchEngine, syllables


@pytest.fixture(scope="module")
def engine():
    return WorkbenchEngine()


@pytest.mark.parametrize("source,target", [
    ("annyeonghaseyo", "안녕하세요"), ("testreul sihaeng", "test를 시행"),
    ("Testreul sihaeng!", "Test를 시행!"), ("bap", "밥"),
    ("gungnip baengma silla", "국립 백마 신라"), ("gamsahamnida", "감사합니다"),
    ("oneureun chinguwa gachi", "오늘은 친구와 같이"),
    ("Hello World!", "Hello World!"), ("APIreul test", "API를 test"),
    ("\tannyeong  hangeul\n", "\t안녕  한글\n"),
    ("한글 😀", "한글 😀"), ("", ""),
])
def test_supported_examples(engine, source, target):
    assert engine.convert(source)["result"] == target


@pytest.mark.parametrize("text", ["https://example.com/test?q=1", "hello@example.com",
                                   "some_variable", "v1.2.3", "test/file.py", "192.168.0.1"])
def test_protected_tokens(engine, text):
    assert engine.convert(text)["result"] == text


@pytest.mark.parametrize("source", ["gachi", "gain", "nat", "bap", "Testreul", "kk", "x", "annyeong"])
def test_original_candidate_and_alignment(engine, source):
    result = engine.convert(source)
    assert source in [c["text"] for c in result["candidates"]]
    assert len(result["candidates"]) <= 5
    for candidate in result["candidates"]:
        assert "".join(s["text"] for s in candidate["segments"]) == candidate["text"]
        assert "".join(s["source"] for s in candidate["segments"]) == source
        for segment in candidate["segments"]:
            assert source[segment["input_start"]:segment["input_end"]] == segment["source"]


def test_ambiguity_and_comparison(engine):
    values = [item["text"] for item in engine.convert("gachi")["candidates"]]
    assert {"같이", "가치", "gachi"}.issubset(values)
    assert engine.convert("gachi", engine="rules")["result"] != "같이"
    assert engine.convert("gachi", context="친구와 ")["context_used"] is False


@pytest.mark.parametrize("source", ["TESTreul 한글!?\n😀", "annyeong", "<script>alert(1)</script>"])
def test_force_raw_is_exact(engine, source):
    assert engine.convert(source, raw=True)["result"] == source


def test_bounded_incomplete_and_long_input(engine):
    for source in ("g", "kk", "xyz", "annyeong" * 25, "a" * 40, "x" * MAX_INPUT):
        start = time.perf_counter()
        result = engine.convert(source)
        assert source in [c["text"] for c in result["candidates"]]
        assert time.perf_counter() - start < 2
    assert engine.convert("x" * 200)["warnings"]
    fragmented = engine.convert("a." * 1000)
    assert fragmented["result"] == "a." * 1000
    assert fragmented["warnings"]
    assert len(json.dumps(fragmented)) < 20000


def test_reject_invalid_unicode(engine):
    with pytest.raises(ValueError, match="유니코드"):
        engine.convert("a", context="\ud800")


def test_version_covers_behavioral_files(engine):
    import hashlib
    import hanglish.core.workbench as module
    digest = hashlib.sha256(Path(module.__file__).read_bytes())
    for name in ("workbench_lexicon.json", "common_english.txt", "it_terms.txt"):
        digest.update((module.DATA / name).read_bytes())
    assert engine.metadata()["version"] == module.VERSION + "-" + digest.hexdigest()[:12]


def test_neural_facade_uses_score_scale_engine_and_reuses_instance(monkeypatch):
    from hanglish.core import score_scale_converter, neural_converter, workbench
    created=[]
    calls=[]
    class Converter:
        models_loaded=True
        def __init__(self, **kwargs):
            assert kwargs == dict(score_scale_checkpoint=workbench.SCORE_SCALE_MODEL,
                                  phonetic_checkpoint=workbench.PHONETIC_MODEL)
            created.append(self)
        def convert(self, *args):
            calls.append(args)
            return {'result': 'fixture'}
    monkeypatch.setattr(neural_converter, 'neural_readiness', lambda: (True, 'fixture'))
    monkeypatch.setattr(score_scale_converter, 'ScoreScaleConverter', Converter)
    facade=WorkbenchEngine()
    assert facade.convert('je23hoe imnida', engine='neural')['result']=='fixture'
    facade.convert('je23hoe imnida', engine='neural', mode='live', raw=True)
    assert len(created)==1
    assert calls==[('je23hoe imnida', '', 'neural', 'batch', False),
                   ('je23hoe imnida', '', 'neural', 'live', True)]


@pytest.mark.parametrize('state', ['missing', 'changed'])
def test_score_scale_artifact_failure_disables_only_neural(monkeypatch, tmp_path, state):
    from hanglish.core import neural_converter, workbench, ctc_union_converter
    artifact = tmp_path / 'model'
    if state == 'changed':
        artifact.write_bytes(b'changed')
    monkeypatch.setattr(neural_converter, 'neural_readiness', lambda: (True, 'base ready'))
    monkeypatch.setattr(workbench, 'SCORE_SCALE_ARTIFACTS', ((artifact, 'expected'),))
    calls = []
    class Morphology:
        def __init__(self):
            calls.append('created')
        def convert(self, text, context, engine, mode, raw):
            assert engine == 'morphology'
            return {'result': text}
    monkeypatch.setattr(ctc_union_converter, 'CTCUnionConverter', Morphology)
    facade = WorkbenchEngine()
    metadata = {item['id']: item for item in facade.metadata()['engines']}
    assert not metadata['neural']['available']
    assert metadata['morphology']['available']
    with pytest.raises(ValueError, match='모델 파일'):
        facade.convert('imnida', engine='neural')
    for _ in range(2):
        assert facade.convert('TEST', engine='morphology', raw=True)['result'] == 'TEST'
    assert calls == ['created'] and facade._neural is None


@pytest.mark.parametrize("kwargs", [
    {"text": "x" * (MAX_INPUT + 1)}, {"text": []}, {"text": "a", "context": None},
    {"text": "a", "context": "x" * 501}, {"text": "a", "engine": "byt5"},
    {"text": "a", "mode": "unknown"}, {"text": "a", "raw": "false"},
])
def test_reject_invalid_requests(engine, kwargs):
    with pytest.raises(ValueError):
        engine.convert(**kwargs)


def test_no_claim_of_full_standard_or_trained_model(engine):
    metadata = engine.metadata()
    assert len(metadata["limitations"]) >= 3
    assert all(not item["available"] for item in metadata["engines"] if item["id"] in ("byt5", "ranker"))
    # A held-out phonological spelling demonstrates a real, recorded limitation.
    assert engine.convert("eomneun")["result"] != "없는"


def test_syllable_coverage_is_bounded():
    for onset in ["g", "n", "d", "m", "b"]:
        for vowel in ["a", "eo", "o", "u", "i"]:
            for coda in ["", "n", "m", "ng"]:
                results = syllables(onset + vowel + coda)
                assert 0 < len(results) <= 12


def test_lexicon_keys_are_single_roman_words():
    lexicon = json.loads(Path("data/workbench_lexicon.json").read_text())
    assert all(key.isascii() and key.isalpha() for key in lexicon)
