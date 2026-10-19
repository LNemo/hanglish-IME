import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'pair_verifier_v2', ROOT / '.omx/artifacts/verify-recombination-pair-v2.py')
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


@pytest.mark.parametrize('change', [
    {'scores_after': [1.0, 2.0]},
    {'scores_after': [float('nan'), 1.0]},
    {'scores_before': [], 'scores_after': []},
    {'scores_after': [2.0]},
    {'max_score_delta': 0.1},
    {'probe_id': 'wrong'},
])
def test_reload_rejects_inconsistent_recorded_parity(change):
    probe = {'id': 'fit-1', 'arms': {'a_original': {'candidates': ['가', '나']}}}
    check = {'probe_id': 'fit-1', 'scores_before': [2., 1.], 'scores_after': [2., 1.],
             'max_score_delta': 0., 'argmax_same': True}
    verifier.verify_reload_scores(check, probe, 'a_original')
    check.update(change)
    with pytest.raises(ValueError, match='checkpoint probe'):
        verifier.verify_reload_scores(check, probe, 'a_original')


@pytest.mark.parametrize('change', [
    {'seed': 0}, {'encoder_lr': 0.01}, {'unfrozen_top_encoder_layers': 12},
    {'training_recipe': 'different'}, {'inference_batch_size': 64},
    {'artifact_hashes': {}},
])
def test_recipe_rejects_changed_training_contract(change):
    metadata = {'format_version': 1, 'max_length': 256, 'inference_batch_size': 8,
                'training_recipe': verifier.SCHEMA, 'arm': 'a_original',
                'selection_sha256': 'selection', 'seed': 20260921,
                'encoder_lr': 1e-5, 'head_lr': 5e-5, 'weight_decay': .01,
                'clip_norm': 1., 'unfrozen_top_encoder_layers': 2,
                'gradient_accumulation': 4, 'epochs': 1,
                'artifact_hashes': dict.fromkeys([
                    'config.json', 'model.safetensors', 'special_tokens_map.json',
                    'tokenizer.json', 'tokenizer_config.json', 'vocab.txt'], 'hash')}
    verifier.verify_recipe(metadata, 'a_original', 'selection')
    metadata.update(change)
    with pytest.raises(ValueError, match='metadata'):
        verifier.verify_recipe(metadata, 'a_original', 'selection')
