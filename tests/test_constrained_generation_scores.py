"""Exercise the actual beam decoders against deterministic token probabilities."""
import pytest
import torch
from hanglish.models.phonetic_transducer import (
    PhoneticTransducer, TransducerConfig, OUTPUT_TO_ID, OUTPUT_TOKENS, EOS,
    hangul_to_jamo,
)
from scripts.constrained_phonetic_search import generate_constrained


class FixedNetwork(torch.nn.Module):
    def __init__(self, eos_logit):
        super().__init__()
        self.eos_logit = eos_logit
        self.target = [OUTPUT_TO_ID[t] for t in hangul_to_jamo('각')] + [OUTPUT_TO_ID[EOS]]

    def encode(self, source):
        return None, torch.tensor([0]), None

    def logits_at(self, step):
        logits = torch.full((1, len(OUTPUT_TOKENS)), -12.)
        logits[0, self.target[min(step, 3)]] = self.eos_logit if step >= 3 else 2.
        return logits

    def decode_step(self, token, hidden, encoded, mask):
        return self.logits_at(int(hidden.item())), hidden + 1


def model(eos_logit):
    return PhoneticTransducer(FixedNetwork(eos_logit), TransducerConfig())


def test_common_candidate_scores_include_every_token_and_eos():
    transducer = model(-2.)
    baseline = dict(transducer.generate('gak', beam_size=4))
    constrained = dict(generate_constrained(transducer, 'gak', beam_size=4))
    assert '각' in baseline and '각' in constrained
    for candidate in baseline.keys() & constrained.keys():
        assert constrained[candidate] == pytest.approx(baseline[candidate], abs=1e-7)
    network = transducer.network
    exact_score = sum(torch.log_softmax(network.logits_at(i)[0], dim=-1)[token].item()
                      for i, token in enumerate(network.target))
    assert constrained['각'] == pytest.approx(exact_score, abs=1e-7)


def test_eos_probability_changes_completed_score():
    low, high = model(-2.), model(2.)
    low_score = dict(generate_constrained(low, 'gak', beam_size=4))['각']
    high_score = dict(generate_constrained(high, 'gak', beam_size=4))['각']
    eos = OUTPUT_TO_ID[EOS]
    expected_difference = (torch.log_softmax(high.network.logits_at(3)[0], dim=-1)[eos]
                           - torch.log_softmax(low.network.logits_at(3)[0], dim=-1)[eos]).item()
    assert high_score-low_score == pytest.approx(expected_difference, abs=1e-7)
