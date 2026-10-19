import pytest

from hanglish.core.lattice import CandidateGenerator, WordOption, keep_strongest
from hanglish.core.phonology import romanize_word
from hanglish.core.workbench import WorkbenchEngine


@pytest.fixture(scope="module")
def generator():
    return CandidateGenerator()


@pytest.mark.parametrize("roman,expected", [
    ("haksaengimnida", "학생입니다"), ("seonsaengnimimnida", "선생님입니다"),
    ("gungmureul", "국물을"), ("eomneun", "없는"), ("ilgeoyo", "읽어요"),
    ("hangugeo", "한국어"), ("jibeuro", "집으로"), ("gillo", "길로"),
    ("NovelAppeseo", "NovelApp에서"), ("testreul", "test를"),
    ("beobineul", "법인을"), ("daepyohamnida", "대표합니다"),
    ("jangnyeogeul", "장력을"),
    # Independent analogues for productive predicate and coda behavior.
    ("meokneunda", "먹는다"), ("sanda", "산다"),
    ("meongneun", "먹는"), ("saneun", "사는"),
    ("balba", "밟아"), ("gapsi", "값이"), ("ilgeo", "읽어"),
    ("ttareul", "딸을"), ("jjageul", "짝을"),
    # Nominalization must retain aspiration at productive noun + 하기 boundaries.
    ("seoldeukagi", "설득하기"), ("yeyakagi", "예약하기"),
    ("meokgi", "먹기"), ("sseugi", "쓰기"),
    ("balbeuni", "밟으니"), ("neulgeona", "늙거나"), ("eolgeodeun", "얽거든"),
    ("ilgeuni", "읽으니"), ("meogeunikka", "먹으니까"),
    ("ilgoseo", "읽고서"), ("malgie", "맑기에"), ("sanikka", "사니까"),
])
def test_productive_candidates(generator, roman, expected):
    assert expected in [option.text for option in generator.options(roman)]


def test_morphology_integration_preserves_source_and_spaces():
    result = WorkbenchEngine().convert("haksaengimnida.\nNovelAppeseo  testreul", engine="morphology")
    assert result["result"] == "학생입니다.\nNovelApp에서  test를"
    for candidate in result["candidates"]:
        assert "".join(segment["source"] for segment in candidate["segments"]) == result["input"]
        assert "".join(segment["text"] for segment in candidate["segments"]) == candidate["text"]


def test_morphology_protected_and_unknown_bounded():
    engine = WorkbenchEngine()
    for text in ("https://example.com/a?b=1", "foo@example.com", "some_code", "x" * 2000):
        assert engine.convert(text, engine="morphology")["result"] == text


def test_whole_word_is_not_misread_as_english_particle(generator):
    assert "on을" not in [option.text for option in generator.options("oneul")]


def test_explicit_protected_boundary_only():
    engine = WorkbenchEngine()
    for token in ("https://example.net?q=hanglish", "user@example.kr", "main.py:42"):
        assert engine.convert(token + "man", engine="morphology")["result"] == token + "man"
        source = "`" + token + "`man"
        result = engine.convert(source, engine="morphology")
        assert result["result"] == "`" + token + "`만"
        assert "".join(part["source"] for part in result["candidates"][0]["segments"]) == source


def test_neural_evidence_promotes_duplicate_syllable_candidate():
    from hanglish.core.lattice import WordOption
    from hanglish.core.neural_converter import NeuralConverter
    from types import SimpleNamespace

    converter = object.__new__(NeuralConverter)
    converter.generator = SimpleNamespace(options=lambda source: (
        WordOption('여행한다', .6, 'syllable'), WordOption('여하엥한다', .5, 'syllable'),
        WordOption(source, 0., 'raw')))
    converter.transducer = SimpleNamespace(generate_constrained=lambda *args, **kwargs: [], generate=lambda source, beam_size: [('여행한다', -.2)])
    import threading
    converter.lock = threading.RLock()
    options = converter._word_options('yeohaenghanda', True)
    assert options[0].text == '여행한다'
    assert options[0].kind == 'neural'
    assert options[0].score > 3
    assert [o.text for o in options] == ['여행한다', 'yeohaenghanda', '여하엥한다']
    assert options[-1].kind == 'syllable'
    assert romanize_word(options[-1].text) == 'yeohaenghanda'


def test_morphology_only_pool_keeps_two_neural_alternatives_after_pruning():
    from hanglish.core.neural_converter import NeuralConverter
    from types import SimpleNamespace
    import threading

    calls = []
    converter = object.__new__(NeuralConverter)
    converter.generator = SimpleNamespace(options=lambda source: (
        WordOption('배우바께', 6., 'morphology'),
        WordOption('배우밖에', .5, 'syllable'),
        WordOption(source, 0., 'raw')))
    converter.transducer = SimpleNamespace(generate_constrained=lambda *args, **kwargs: [], generate=lambda source, beam_size: (
        calls.append((source, beam_size)) or
        [('배우밖에', -.2), ('배웁아께', -.4), ('배우바께', -.6)]))
    converter.lock = threading.RLock()

    options = converter._word_options('baeubakke', True)

    assert calls == [('baeubakke', 8)]
    assert [option.text for option in options] == [
        '배우바께', 'baeubakke', '배우밖에', '배웁아께'
    ]
    assert [option.kind for option in options[-2:]] == ['neural', 'neural']


@pytest.mark.parametrize("blocker", [
    WordOption('배우', 6., 'lexicon'),
    WordOption('baeubakke', 7.5, 'raw'),
    WordOption('Actor밖에', 8., 'mixed'),
])
def test_strong_lexicon_english_or_mixed_option_blocks_morphology_supplement(blocker):
    from hanglish.core.neural_converter import NeuralConverter
    from types import SimpleNamespace
    import threading

    calls = []
    source = 'baeubakke'
    raw = blocker if blocker.kind == 'raw' else WordOption(source, 0., 'raw')
    converter = object.__new__(NeuralConverter)
    converter.generator = SimpleNamespace(options=lambda _: (
        WordOption('배우바께', 6., 'morphology'), blocker, raw))
    converter.transducer = SimpleNamespace(generate_constrained=lambda *args, **kwargs: [], generate=lambda token, beam_size: (
        calls.append((token, beam_size)) or [('배우밖에', -.2)]))
    converter.lock = threading.RLock()

    options = converter._word_options(source, True)

    assert calls == []
    assert all(option.kind != 'neural' for option in options)
    assert blocker in options
    assert raw in options


def test_morphology_supplement_preserves_stronger_duplicate_and_source():
    from hanglish.core.neural_converter import NeuralConverter
    from types import SimpleNamespace
    import threading

    morphology = WordOption('배우바께', 6., 'morphology')
    raw = WordOption('baeubakke', 0., 'raw')
    converter = object.__new__(NeuralConverter)
    converter.generator = SimpleNamespace(options=lambda source: (morphology, raw))
    converter.transducer = SimpleNamespace(generate_constrained=lambda *args, **kwargs: [], generate=lambda source, beam_size: [
        ('배우바께', -.1), ('배우밖에', -.2), ('배웁아께', -.3)])
    converter.lock = threading.RLock()

    options = converter._word_options('baeubakke', True)

    assert next(option for option in options if option.text == morphology.text) == morphology
    assert sum(option.text == morphology.text for option in options) == 1
    assert raw in options
    assert '배우밖에' in [option.text for option in options]
    assert '배웁아께' not in [option.text for option in options]


@pytest.mark.parametrize("source", ['Baeubakke', '배우밖에', 'x', 'a' * 41])
def test_morphology_supplement_requires_bounded_lowercase_ascii_word(source):
    from hanglish.core.neural_converter import NeuralConverter
    from types import SimpleNamespace
    import threading

    calls = []
    converter = object.__new__(NeuralConverter)
    converter.generator = SimpleNamespace(options=lambda token: (
        WordOption('형태', 6., 'morphology'), WordOption(token, 0., 'raw')))
    converter.transducer = SimpleNamespace(generate_constrained=lambda *args, **kwargs: [], generate=lambda token, beam_size: (
        calls.append((token, beam_size)) or [('한글', -.2)]))
    converter.lock = threading.RLock()

    converter._word_options(source, True)

    assert calls == []


def test_longest_english_particle_boundary(generator):
    options = [option.text for option in generator.options('asdfreul')]
    assert 'asdf를' in options
    assert 'asdfr을' not in options


@pytest.mark.parametrize("source,expected,wrong", [
    ("HTTPdo", "HTTP도", None),
    ("MyNamedo", "MyName도", "MyNam에도"),
    ("camelCasedo", "camelCase도", "camelCas에도"),
    ("UPPERCASEdo", "UPPERCASE도", "UPPERCAS에도"),
    ("ProjectNamedo", "ProjectName도", "ProjectNam에도"),
    ("JSONParserdo", "JSONParser도", "JSONPars에도"),
])
def test_named_english_token_keeps_overlapping_particle_boundaries(generator, source, expected, wrong):
    options = [option.text for option in generator.options(source)]
    assert expected in options
    if wrong:
        assert wrong not in options


def test_known_english_prefix_preserves_real_boundary_ambiguity(generator):
    options = [option.text for option in generator.options("NOTEdo")]
    assert options[0] == "NOTE도"
    assert "NOT에도" in options


@pytest.mark.parametrize("source,expected", [
    ("NovelAppeseo", "NovelApp에서"),
    ("camelCasereul", "camelCase를"),
])
def test_named_english_long_particles_keep_the_longest_boundary(generator, source, expected):
    assert generator.options(source)[0].text == expected


def test_direct_word_does_not_hide_productive_noun_candidate(generator):
    options = [option.text for option in generator.options("sareul")]
    assert "사를" in options
    assert "살을" in options


def test_union_keeps_strongest_evidence_for_same_spelling():
    options = {}
    keep_strongest(options, WordOption("보기", 7.0, "lexicon"))
    keep_strongest(options, WordOption("보기", 5.8, "morphology"))
    assert options["보기"] == WordOption("보기", 7.0, "lexicon")
    options = {}
    keep_strongest(options, WordOption("보기", 5.8, "morphology"))
    keep_strongest(options, WordOption("보기", 7.0, "lexicon"))
    assert options["보기"] == WordOption("보기", 7.0, "lexicon")


def test_stem_and_compound_union_cannot_weaken_dictionary_evidence():
    generator = object.__new__(CandidateGenerator)
    generator.index = {'san': {'산': WordOption('산', 6., 'lexicon')},
                       'mul': {'물': WordOption('물', 5., 'lexicon')}}
    generator.dictionary_index = {'san': {'산': WordOption('산', 8., 'lexicon')},
                                  'mul': {'물': WordOption('물', 7., 'lexicon')}}
    generator.derivations = []
    assert generator._stem_options('san')[0].score == 8.
    compound = next(option for option in generator._stem_options('sanmul') if option.text == '산물')
    assert compound.score == pytest.approx(6.6)
    generator.index, generator.dictionary_index = generator.dictionary_index, generator.index
    assert generator._stem_options('san')[0].score == 8.
    assert next(option for option in generator._stem_options('sanmul') if option.text == '산물').score == compound.score


def test_two_lexeme_compound_survives_direct_homophone(generator):
    # General compound mechanism: 학교 + 로, followed by an object particle.
    roman = romanize_word("학교로를")
    options = [option.text for option in generator.options(roman)]
    assert "학교로를" in options


def test_neural_hypothesis_keeps_unknown_english_particle_candidate(generator):
    from hanglish.core.neural_converter import NeuralConverter
    from types import SimpleNamespace
    import threading

    converter = object.__new__(NeuralConverter)
    converter.generator = generator
    converter.transducer = SimpleNamespace(generate_constrained=lambda *args, **kwargs: [], generate=lambda source, beam_size: [('아스드프를', -.2)])
    converter.lock = threading.RLock()
    options = converter._word_options('asdfreul', True)
    assert 'asdf를' in [option.text for option in options]
    assert '아스드프를' in [option.text for option in options]
    assert 'asdfreul' in [option.text for option in options]
    assert len({option.text for option in options}) == len(options)


def test_neural_readiness_requires_dependency_and_both_models(monkeypatch, tmp_path):
    import hanglish.core.neural_converter as module
    checkpoint = tmp_path / 'context.pt'
    checkpoint.touch()
    monkeypatch.setattr(module, 'CHECKPOINT', checkpoint)
    monkeypatch.setattr(module, 'TRANSDUCER', tmp_path / 'missing.pt')
    assert module.neural_readiness()[0] is False
    monkeypatch.setattr(module, 'find_spec', lambda name: None)
    assert '의존성' in module.neural_readiness()[1]


def test_quoted_code_preserves_spaces_and_romanized_identifiers():
    engine = WorkbenchEngine()
    for selected in ['baseline', 'morphology']:
        source = '`haksaeng = gachi`.'
        assert engine.convert(source, engine=selected)['result'] == source
    source = '`haksaeng = gachi`reul'
    result = engine.convert(source, engine='morphology')
    assert result['result'] == '`haksaeng = gachi`를'
    assert ''.join(s['source'] for s in result['candidates'][0]['segments']) == source


def test_semantic_context_can_rank_generated_unknown_candidates():
    from hanglish.core.lattice import WordOption
    from hanglish.core.neural_converter import NeuralConverter
    from types import SimpleNamespace
    import threading

    converter = object.__new__(NeuralConverter)
    converter.generator = SimpleNamespace(english_suffixes=[])
    converter.load_models = lambda: None
    converter.context_ranking = True
    converter.version = 'test'
    converter.lock = threading.RLock()
    converter.word_options = lambda source, learned: (
        WordOption('같이', 3.9, 'neural'), WordOption('가치', 3., 'neural'),
        WordOption(source, 0., 'raw'))
    converter.ranker = SimpleNamespace(score_pairs=lambda contexts, candidates: [0., 2.])
    converter.ranker_kind = 'semantic'
    result = converter.convert('gachi', context='물건의 쓸모')
    assert result['result'] == '가치'
    assert result['context_used'] is True
    converter.ranker_kind = 'mlp'
    result = converter.convert('gachi', context='물건의 쓸모')
    assert result['result'] == '같이'
    assert result['context_used'] is False
