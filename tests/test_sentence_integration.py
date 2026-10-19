import threading
from types import SimpleNamespace

import pytest

from hanglish.core.lattice import WordOption
from hanglish.core.neural_converter import NeuralConverter


def test_default_ranker_selection_requires_available_bundles(tmp_path, monkeypatch):
    import hanglish.core.neural_converter as module
    semantic = tmp_path / 'semantic'
    sentence = tmp_path / 'sentence'
    monkeypatch.setattr(module, 'SEMANTIC_CHECKPOINT', semantic)
    monkeypatch.setattr(module, 'SENTENCE_CHECKPOINT', sentence)
    assert module.default_ranker_kind() == 'mlp'
    sentence.mkdir()
    assert module.default_ranker_kind() == 'mlp'
    semantic.mkdir()
    assert module.default_ranker_kind() == 'hybrid'
    # Existing incomplete bundles still select their declared model: loading
    # must fail explicitly instead of silently reporting a different model.
    monkeypatch.setattr(module, 'find_spec', lambda name: object())
    monkeypatch.setattr(module, 'TRANSDUCER', tmp_path / 'phonetic.pt')
    assert module.neural_readiness()[0] is False
    sentence.rmdir()
    assert module.default_ranker_kind() == 'semantic'


def test_artifact_version_never_reads_unexpected_bundle_files(tmp_path, monkeypatch):
    import hanglish.core.neural_converter as module
    from pathlib import Path
    model_dir = tmp_path / 'checkpoints' / 'semantic'
    model_dir.mkdir(parents=True)
    unexpected = model_dir / 'unverified-large.bin'
    unexpected.touch()
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, '__file__', str(tmp_path / 'converter.py'))
    monkeypatch.setattr(module, 'SEMANTIC_CHECKPOINT', model_dir)
    monkeypatch.setattr(module, 'SENTENCE_CHECKPOINT', tmp_path / 'sentence')
    monkeypatch.setattr(module, 'CHECKPOINT', tmp_path / 'context.pt')
    monkeypatch.setattr(module, 'TRANSDUCER', tmp_path / 'phonetic.pt')
    original_open = Path.open
    def guarded_open(path, *args, **kwargs):
        assert path != unexpected, 'must not hash unverified bundle data'
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded_open)
    before = module.artifact_version()
    unexpected.unlink()
    assert module.artifact_version() != before


@pytest.mark.parametrize('fits', [True, False])
def test_sentence_model_compares_complete_alternatives_and_respects_length(fits):
    converter = object.__new__(NeuralConverter)
    converter.generator = SimpleNamespace(english_suffixes=[])
    converter.load_models = lambda: None
    converter.context_ranking = True
    converter.version = 'test'
    converter.lock = threading.RLock()
    choices = {'gachi': ('가치', '같이'), 'ireul': ('이를', '일을')}
    converter.word_options = lambda source, learned: tuple(
        [WordOption(value, 7 - i * .1, 'lexicon') for i, value in enumerate(choices[source])]
        + [WordOption(source, 0, 'raw')])
    converter.ranker = SimpleNamespace(score_pairs=lambda contexts, candidates: [
        20. if value in ('가치', '이를') else 0. for value in candidates])
    converter.ranker_kind = 'sentence'
    scored = []
    def score_sentences(contexts, candidates):
        scored.extend(candidates)
        assert all(context == '동료와 협력한다' for context in contexts)
        return [10. if value == '같이 일을' else 0. for value in candidates]
    converter.sentence_ranker = SimpleNamespace(can_score=lambda value: fits, score_pairs=score_sentences)
    converter.ranker_kind = 'semantic'
    proposal_texts = [path[0] for path in converter.sentence_proposals(
        'gachi ireul', context='동료와 협력한다')['paths']]
    converter.ranker_kind = 'sentence'
    result = converter.convert('gachi ireul', context='동료와 협력한다')
    assert result['result'] == ('같이 일을' if fits else '가치 이를')
    assert result['sentence_ranking_applied'] is fits
    assert any(candidate['text'] == 'gachi ireul' for candidate in result['candidates'])
    if fits:
        assert '같이 일을' in scored and scored == proposal_texts and len(scored) <= 48
    else:
        assert not scored and any('길이' in warning for warning in result['warnings'])
    for candidate in result['candidates']:
        assert ''.join(segment['source'] for segment in candidate['segments']) == 'gachi ireul'
        assert ''.join(segment['text'] for segment in candidate['segments']) == candidate['text']
