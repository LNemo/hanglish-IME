import hashlib
import json
from pathlib import Path

import pytest
import torch

from hanglish.core.phonology import romanize_word
from hanglish.models.phonetic_transducer import (
    BOS,
    CODAS,
    EOS,
    INPUT_TO_ID,
    OUTPUT_TO_ID,
    PAD,
    PhoneticTransducer,
    ONSETS,
    TransducerConfig,
    VOWELS,
    _Network,
    _allowed_output_ids,
    checkpoint_payload,
    hangul_to_jamo,
    jamo_to_hangul,
)
from scripts.train_phonetic_transducer import require_unchanged_sources, run_with_failure_status


def _write_checkpoint(path, config=None):
    config = config or TransducerConfig(dropout=0.0)
    torch.save(checkpoint_payload(_Network(config), config), path)
    metadata = {
        "format_version": 1,
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    path.with_suffix(".json").write_text(json.dumps(metadata))
    return config


@pytest.mark.parametrize("word", ["가", "입니다", "읽어요", "학생입니다"])
def test_jamo_representation_roundtrips(word):
    assert jamo_to_hangul(hangul_to_jamo(word)) == word


def test_jamo_codec_rejects_partial_or_non_hangul_data():
    assert jamo_to_hangul([]) is None
    assert jamo_to_hangul(["ㄱ", "ㅏ"]) is None
    with pytest.raises(ValueError, match="precomposed Hangul"):
        hangul_to_jamo("한글A")


def test_decoder_masks_each_jamo_position_and_only_ends_complete_syllables():
    onset = {OUTPUT_TO_ID[token] for token in ONSETS}
    vowel = {OUTPUT_TO_ID[token] for token in VOWELS}
    coda = {OUTPUT_TO_ID[token] for token in CODAS}
    assert set(_allowed_output_ids(0)) == onset
    assert set(_allowed_output_ids(1)) == vowel
    assert set(_allowed_output_ids(2)) == coda
    assert set(_allowed_output_ids(3)) == onset | {OUTPUT_TO_ID[EOS]}
    assert OUTPUT_TO_ID[EOS] not in _allowed_output_ids(1)
    assert OUTPUT_TO_ID[EOS] not in _allowed_output_ids(2)


def test_network_is_small_and_forward_shape_is_stable():
    config = TransducerConfig(dropout=0.0)
    network = _Network(config)
    parameters = sum(parameter.numel() for parameter in network.parameters())
    assert 800_000 <= parameters <= 1_300_000
    source = torch.tensor([[INPUT_TO_ID[BOS], INPUT_TO_ID["g"], INPUT_TO_ID["a"], INPUT_TO_ID[EOS]]])
    target = torch.tensor([[OUTPUT_TO_ID[BOS], *[OUTPUT_TO_ID[t] for t in hangul_to_jamo("가")]]])
    assert network(source, target).shape == (1, 4, len(OUTPUT_TO_ID))


def test_encoder_is_invariant_to_padding_from_a_longer_batch_peer():
    torch.manual_seed(11)
    network = _Network(TransducerConfig(dropout=0.0)).eval()
    short = [INPUT_TO_ID[BOS], INPUT_TO_ID["g"], INPUT_TO_ID["a"], INPUT_TO_ID[EOS]]
    long = [INPUT_TO_ID[BOS], *(INPUT_TO_ID[c] for c in "haksaeng"), INPUT_TO_ID[EOS]]
    alone = torch.tensor([short])
    batched = torch.tensor([
        short + [INPUT_TO_ID[PAD]] * (len(long) - len(short)),
        long,
    ])
    with torch.inference_mode():
        alone_encoded, alone_hidden, _ = network.encode(alone)
        batch_encoded, batch_hidden, _ = network.encode(batched)
    assert torch.allclose(alone_encoded[0], batch_encoded[0, :len(short)], atol=1e-6)
    assert torch.allclose(alone_hidden[:, 0], batch_hidden[:, 0], atol=1e-6)


def test_checkpoint_loads_with_weights_only_and_generation_never_returns_partial(tmp_path):
    torch.manual_seed(7)
    config = TransducerConfig(dropout=0.0)
    path = tmp_path / "model.pt"
    _write_checkpoint(path, config)
    payload = torch.load(path, weights_only=True)
    assert payload["format_version"] == 1
    model = PhoneticTransducer.load(path)
    for hangul, _ in model.generate("imnida", beam_size=4):
        assert hangul
        assert romanize_word(hangul) == "imnida"
    assert isinstance(model.generate("imnida", beam_size=16), list)


@pytest.mark.parametrize("value", ["", "Imnida", "한글", "with space", None])
def test_generate_rejects_invalid_roman_input(tmp_path, value):
    config = TransducerConfig(dropout=0.0)
    path = tmp_path / "model.pt"
    _write_checkpoint(path, config)
    model = PhoneticTransducer.load(path)
    with pytest.raises(ValueError, match="lowercase ASCII"):
        model.generate(value)


def test_checkpoint_load_requires_companion_metadata(tmp_path):
    path = tmp_path / "model.pt"
    torch.save(checkpoint_payload(_Network(TransducerConfig()), TransducerConfig()), path)
    with pytest.raises(FileNotFoundError, match="metadata is missing"):
        PhoneticTransducer.load(path)


def test_checkpoint_load_rejects_checksum_mismatch(tmp_path):
    path = tmp_path / "model.pt"
    _write_checkpoint(path)
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    metadata["checkpoint_sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="SHA-256"):
        PhoneticTransducer.load(path)


def test_checkpoint_load_rejects_unsupported_metadata_format(tmp_path):
    path = tmp_path / "model.pt"
    _write_checkpoint(path)
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    metadata["format_version"] = 2
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="format_version"):
        PhoneticTransducer.load(path)


def test_checkpoint_load_rejects_unsupported_payload_format(tmp_path):
    path = tmp_path / "model.pt"
    config = TransducerConfig(dropout=0.0)
    payload = checkpoint_payload(_Network(config), config)
    payload["format_version"] = 2
    torch.save(payload, path)
    metadata = {
        "format_version": 1,
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    path.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="payload format_version"):
        PhoneticTransducer.load(path)


def test_training_failure_is_written_atomically_to_status(tmp_path):
    status = tmp_path / "status.json"

    def fail():
        raise RuntimeError("synthetic failure")

    with pytest.raises(RuntimeError, match="synthetic failure"):
        run_with_failure_status(status, fail)
    payload = json.loads(status.read_text())
    assert payload["state"] == payload["phase"] == "failed"
    assert payload["error_type"] == "RuntimeError"
    assert payload["error"] == "synthetic failure"
    assert not status.with_suffix(".json.tmp").exists()


def test_training_rejects_source_changes_before_publication():
    with pytest.raises(RuntimeError, match="sources changed"):
        require_unchanged_sources({"source": "before"}, {"source": "after"})


def test_trained_checkpoint_loads_and_outputs_are_forward_valid():
    path = Path("checkpoints/phonetic_transducer.pt")
    if not path.exists():
        pytest.skip("training artifact has not been built")
    model = PhoneticTransducer.load(path)
    for roman in ("imnida", "hamnida", "eomneun", "haksaengimnida"):
        assert all(romanize_word(word) == roman for word, _ in model.generate(roman))
    assert model.generate("imnida")[0][0] == "입니다"
    assert model.generate("haksaengimnida")[0][0] == "학생입니다"
    assert {word for word, _ in model.generate("gachi")} >= {"같이", "가치"}
