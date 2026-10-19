"""Correct the current-production reference in the otherwise frozen DEV screen.

V1 confused the old parent's ASCII count (54) with the already verified deployed
model's count (55). This wrapper changes no prediction, model, metric or gate.
"""
import argparse
import json
from pathlib import Path

from scripts import evaluate_current_phonetic_choice_dev as base

PRODUCER_SHA = '6141379bc626b6c57c481243e7fd167f40f04658ac2c019dc9cd6dbbbd9cfb07'
REFERENCE = base.ROOT / '.omx/artifacts/phonetic-score-scale-dev-screen-v1/independent-verification.json'
REFERENCE_SHA = 'e1b23b2fc23fbacba83c56358207eaa810c4b80addab59481c3b04a47a8ef8f4'


def current_reference(proof):
    base.need(proof.get('verified') is True, 'current reference proof not verified')
    before = proof['comparisons']['parent']['english_ascii_signature']
    old_cal = proof['comparisons']['old_calibration']['english_ascii_signature']
    base.need(before['rows_with_reference_ascii_words'] == old_cal['rows_with_reference_ascii_words'] == 59
              and before['proposed_exact_signature'] == old_cal['proposed_exact_signature'] == 55,
              'deployed ASCII reference differs')
    return dict(rows=316, top1=267, top5=284, ascii_reference_rows=59, ascii_exact_first=55)


def run(output):
    base.need(base.sha256_file(Path(base.__file__)) == PRODUCER_SHA,
              'frozen v1 producer changed')
    base.need(base.sha256_file(REFERENCE) == REFERENCE_SHA, 'reference proof changed')
    base.CURRENT_REFERENCE = current_reference(base.read_json(REFERENCE))
    base.FROZEN = {**base.FROZEN, REFERENCE: REFERENCE_SHA,
                   Path(__file__).resolve(): base.sha256_file(Path(__file__))}
    return base.run(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    result = run(parser.parse_args().output)
    print(json.dumps({k: result[k] for k in ('completed', 'rows', 'quality_gate')}))
