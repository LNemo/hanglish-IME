import hashlib
import json
import math
import pytest
import torch

from hanglish.models.sentence_transducer import (
    BOS, EOS, PAD, ONSET_OFFSET, SentenceTransducer, SentenceTransducerConfig,
    encode_source, encode_target, decode_target, training_batch,
)


@pytest.mark.parametrize('text', ['입니다', 'test를 시행한다.', '값이 12㎞이다 😀', '“API_v2”에서\n확인한다.', '힣갂뷁'])
def test_exact_unicode_target_roundtrip(text):
    assert decode_target(encode_target(text)) == text


@pytest.mark.parametrize('text', ['imnida', 'jeoneun 학생입니다', 'ᄀ abc', 'ㄱ abc'])
def test_source_contract_rejects_single_word_and_korean_hints(text):
    with pytest.raises(ValueError):
        encode_source(text)


def test_target_shift_no_truncation_and_malformed_output():
    source, shifted, expected = training_batch([('jeoneun haksaengimnida', '저는 학생입니다')])
    assert source[0, 0] == BOS and shifted[0, 0] == BOS and expected[0, -1] == EOS
    assert torch.equal(shifted[0, 1:], expected[0, :-1])
    with pytest.raises(ValueError):
        training_batch([('jeoneun haksaengimnida', '저는 학생입니다')], max_tokens=5)
    for tokens in [[ONSET_OFFSET], [BOS], [255 + 3]]:
        with pytest.raises((ValueError, UnicodeError)):
            decode_target(tokens)


def test_causal_teacher_forcing_and_checkpoint_reload(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(71)
    model = SentenceTransducer(SentenceTransducerConfig(width=16, heads=2, encoder_layers=1,
                               decoder_layers=1, feedforward=32, dropout=0., max_tokens=64)).eval()
    source, shifted, _ = training_batch([('jeoneun haksaengimnida', '저는 학생입니다')])
    with torch.inference_mode():
        before = model(source, shifted)
        changed = shifted.clone(); changed[:, 5:] = EOS
        after = model(source, changed)
    assert torch.equal(before[:, :5], after[:, :5])
    path = tmp_path/'model.pt'
    model.save(path, {'input_contract': 'roman_sentence_only'})
    restored = SentenceTransducer.load(path)
    with torch.inference_mode():
        assert torch.equal(before, restored(source, shifted))
    with pytest.raises(FileExistsError):
        model.save(path, {})


def test_incomplete_generation_is_not_returned_as_success(monkeypatch):
    model = SentenceTransducer(SentenceTransducerConfig(width=16, heads=2, encoder_layers=1,
                               decoder_layers=1, feedforward=32, dropout=0., max_tokens=8))
    def repeat_byte(target, memory, mask):
        logits = torch.zeros(target.shape[0], target.shape[1], model.output.out_features)
        logits[:, :, ord('a') + 3] = 10
        return logits
    monkeypatch.setattr(model, 'decode', repeat_byte)
    assert model.generate(['a a'])[0] == {'text': None, 'reason': 'token_limit', 'generated_tokens': 8}


@pytest.mark.parametrize('scale', [0., -1., float('nan'), float('inf'), True])
def test_embedding_scale_rejects_invalid_configuration(scale):
    with pytest.raises(ValueError, match='embedding scale'):
        SentenceTransducer(SentenceTransducerConfig(embedding_scale=scale))


@pytest.mark.parametrize('scale', [None, 1.])
def test_embedding_scale_is_checkpointed_and_reloaded(tmp_path, scale):
    config = SentenceTransducerConfig(width=16, heads=2, encoder_layers=1,
        decoder_layers=1, feedforward=32, dropout=0., max_tokens=64, embedding_scale=scale)
    model = SentenceTransducer(config).eval()
    source, shifted, _ = training_batch([('jeoneun haksaengimnida', '저는 학생입니다')])
    multiplier = math.sqrt(config.width) if scale is None else scale
    assert torch.equal(model.embedded(source, model.source_embedding),
                       model.source_embedding(source) * multiplier + model.positions[:, :source.shape[1]])
    path = tmp_path/'model.pt'
    model.save(path, {})
    reloaded = SentenceTransducer.load(path)
    assert reloaded.config.embedding_scale == scale
    with torch.inference_mode():
        assert torch.equal(model(source, shifted), reloaded(source, shifted))


def test_checkpoint_without_scale_keeps_legacy_logits(tmp_path):
    config = SentenceTransducerConfig(width=16, heads=2, encoder_layers=1,
        decoder_layers=1, feedforward=32, dropout=0., max_tokens=64)
    model = SentenceTransducer(config).eval()
    path = tmp_path/'model.pt'
    model.save(path, {})
    # Simulate the previous saved schema, which has no embedding_scale field.
    payload = torch.load(path, weights_only=True)
    del payload['config']['embedding_scale']
    torch.save(payload, path)
    metadata = json.loads(path.with_suffix('.json').read_text())
    del metadata['config']['embedding_scale']
    metadata['checkpoint_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix('.json').write_text(json.dumps(metadata))
    reloaded = SentenceTransducer.load(path)
    source, shifted, _ = training_batch([('jeoneun haksaengimnida', '저는 학생입니다')])
    assert reloaded.config.embedding_scale is None
    with torch.inference_mode():
        assert torch.equal(model(source, shifted), reloaded(source, shifted))
