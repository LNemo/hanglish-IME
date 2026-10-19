import hashlib
import json
from pathlib import Path

import pytest

from hanglish.core.phonology import romanize_word
from scripts.amend_challenge_ambiguity import PARENT_TEST_SHA, publish

ROOT = Path(__file__).resolve().parents[1] / 'data/challenge'


def test_v22_preserves_sources_and_only_changes_audited_accepted_sets(tmp_path):
    parent_bytes = (ROOT / 'test.jsonl').read_bytes()
    assert hashlib.sha256(parent_bytes).hexdigest() == PARENT_TEST_SHA
    destination = tmp_path / 'v2.2'
    manifest = publish(ROOT, destination)
    assert manifest['corpus_version'] == '2.2'
    for name in ['train.jsonl', 'dev.jsonl', 'train_lexicon.json']:
        assert (ROOT / name).read_bytes() == (destination / name).read_bytes()
    original = [json.loads(line) for line in parent_bytes.splitlines()]
    revised = [json.loads(line) for line in (destination / 'test.jsonl').read_bytes().splitlines()]
    changes = 0
    for before, after in zip(original, revised, strict=True):
        assert {key: value for key, value in before.items() if key != 'expected'} == {
            key: value for key, value in after.items() if key != 'expected'}
        assert set(before['expected']).issubset(after['expected'])
        if before['expected'] != after['expected']:
            changes += 1
            assert len(after['expected']) == 2
            assert romanize_word(after['expected'][0]) == romanize_word(after['expected'][1])
    assert changes == 35
    assert (ROOT / 'test.jsonl').read_bytes() == parent_bytes
    assert (ROOT / 'v2.2/test.jsonl').read_bytes() == (destination / 'test.jsonl').read_bytes()
    with pytest.raises(FileExistsError):
        publish(ROOT, destination)
