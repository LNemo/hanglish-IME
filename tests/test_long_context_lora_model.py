import math
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import long_context_lora_model as model


def test_reference_log_mass_loss_single_multiple_and_all_gold():
    scores = [0.0, math.log(2.0), math.log(3.0)]
    assert model.reference_log_mass_loss(scores, [2]) == pytest.approx(math.log(2.0))
    assert model.reference_log_mass_loss(scores, [0, 2]) == pytest.approx(math.log(1.5))
    assert model.reference_log_mass_loss(scores, [0, 1, 2]) == pytest.approx(0.0)


def test_mlx_loss_shape_uses_float32_and_matches_reference_with_fake_array_backend():
    fake_mx = SimpleNamespace(
        float32=np.float32,
        array=lambda value, dtype=None: np.array(value, dtype=dtype),
        logsumexp=lambda value: np.logaddexp.reduce(value),
    )
    scores = np.array([1000.0, 999.0, 997.0], dtype=np.float64)
    actual = model.mlx_log_mass_loss(fake_mx, scores, [0, 2])
    assert actual.dtype == np.float32
    assert float(actual) == pytest.approx(model.reference_log_mass_loss(scores, [0, 2]),
                                          abs=1e-5)


def test_reference_actual_count_average_and_global_clip_for_full_and_tail_batches():
    for count in (4, 3):
        gradients = [[3.0, 4.0] for _ in range(count)]
        clipped, norm = model.reference_average_and_clip(gradients)
        assert norm == pytest.approx(5.0)
        assert clipped == pytest.approx((0.6, 0.8))
    unclipped, norm = model.reference_average_and_clip([[0.3, 0.4]] * 3)
    assert norm == pytest.approx(0.5)
    assert unclipped == pytest.approx((0.3, 0.4))
    with pytest.raises(FloatingPointError):
        model.reference_average_and_clip([[1e308, 1e308]])


@pytest.mark.parametrize("scores,gold", [
    ([1.0], [0]),
    ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [0]),
    ([1.0, float("nan")], [0]),
    ([1.0, 2.0], []),
    ([1.0, 2.0], [2]),
    ([1.0, 2.0], [0, 0]),
    ([1.0, 2.0], [True]),
])
def test_reference_log_mass_loss_rejects_invalid_inputs(scores, gold):
    with pytest.raises(ValueError):
        model.reference_log_mass_loss(scores, gold)


def test_expected_trainable_contract_is_exact_last_four_qv_factors():
    names = model.expected_trainable_names()
    assert len(names) == 16
    assert names[0] == "model.layers.32.self_attn.q_proj.lora_a"
    assert names[-1] == "model.layers.35.self_attn.v_proj.lora_b"
    assert {int(name.split(".")[2]) for name in names} == {32, 33, 34, 35}
    assert {name.split(".")[4] for name in names} == {"q_proj", "v_proj"}


def test_final_hidden_tied_scorer_matches_ordinary_forward_on_selected_digits():
    class Embedding:
        @staticmethod
        def as_linear(hidden):
            return np.concatenate([hidden, hidden + 10.0], axis=-1)

    class Backbone:
        embed_tokens = Embedding()

        def __call__(self, tokens):
            values = np.asarray(tokens, dtype=np.float32)
            return np.stack([values, values + 0.5], axis=-1)

    class FakeModel:
        args = SimpleNamespace(tie_word_embeddings=True)
        model = Backbone()

        def __call__(self, tokens):
            return self.model.embed_tokens.as_linear(self.model(tokens))

    fake_mx = SimpleNamespace(array=np.array, float32=np.float32)
    selected = model.score_choices(fake_mx, FakeModel(), [7, 8, 9], [0, 2, 3])
    ordinary = model.ordinary_forward_choice_scores(
        fake_mx, FakeModel(), [7, 8, 9], [0, 2, 3])
    assert selected.tolist() == [9.0, 19.0, 19.5]
    assert selected.tolist() == ordinary.tolist()


def test_fixed_optimizer_and_lora_constants_match_approved_recipe():
    assert model.SEED == 20260915
    assert model.LORA_LAYERS == 4
    assert model.LORA_CONFIG == {"rank": 8, "scale": 20.0, "dropout": 0.0,
                                 "keys": ["self_attn.q_proj", "self_attn.v_proj"]}
    assert model.EXPECTED_TRAINABLE_TENSORS == 16
    assert model.EXPECTED_TRAINABLE_SCALARS == 327680
    assert model.LEARNING_RATE == 1e-5
    assert model.ADAM_BETAS == (0.9, 0.999)
    assert model.ADAM_EPS == 1e-8
    assert model.CLIP_NORM == 1.0


def test_module_import_is_mlx_lazy():
    assert "mlx.core" not in model.__dict__
    assert "torch" not in model.__dict__
