from types import SimpleNamespace

import pytest

from hanglish.core.lattice import WordOption
import scripts.probe_phonetic_fallback_union as module


def test_validate_option_union_preserves_prefix_and_checks_additions():
    baseline = (WordOption("밥", 3., "neural"), WordOption("bap", 0., "raw"))
    added = WordOption("밮", .9, "syllable")

    assert module.validate_option_union("bap", baseline, baseline + (added,)) == (added,)
    with pytest.raises(ValueError, match="baseline"):
        module.validate_option_union("bap", baseline, (baseline[1], baseline[0]))
    with pytest.raises(ValueError, match="invalid mechanical"):
        module.validate_option_union(
            "bap", baseline, baseline + (WordOption("밤", .8, "syllable"),)
        )


def test_installed_union_is_cached_source_only_and_audited(monkeypatch):
    baseline = (WordOption("밥", 3., "neural"), WordOption("bap", 0., "raw"))
    converter = SimpleNamespace(
        _word_options=lambda source, learned: baseline,
        generator=SimpleNamespace(options=lambda source: (
            WordOption("밥", 1., "syllable"), WordOption("밮", .9, "syllable")
        )),
    )
    calls = []

    def source_only(delegate, source, learned, limit):
        calls.append((source, learned, limit))
        return delegate._word_options(source, learned) + (
            WordOption("밮", .9, "syllable"),
        )

    monkeypatch.setattr(module, "preserved_options", source_only)
    audits = module.install_fallback_union(converter, 4)

    first = converter.word_options("bap", True)
    second = converter.word_options("bap", True)
    assert first is second
    assert calls == [("bap", True, 4)]
    assert audits[("bap", True)]["additions"] == ["밮"]


def test_summary_keeps_fallback_pool_coverage_nullable():
    cases = [
        {"before_top1": True, "new_top1": True, "before_top5": True,
         "new_top5": True, "before_pool_covered": None, "new_pool_covered": None},
        {"before_top1": False, "new_top1": True, "before_top5": False,
         "new_top5": True, "before_pool_covered": False, "new_pool_covered": True},
    ]

    assert module.summarize(cases) == {
        "before_top1": 1, "new_top1": 2, "before_top5": 1, "new_top5": 2,
        "before_pool_covered": 0, "new_pool_covered": 1,
        "rows": 2, "sentence_scored_rows": 1,
    }


def test_p95_uses_nearest_rank_and_handles_empty_values():
    assert module.p95([]) is None
    assert module.p95(list(range(1, 21))) == 19
