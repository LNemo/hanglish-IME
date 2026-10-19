"""Pinned regeneration preserves existing data on invalid or corrupt inputs."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.prepare_english_dictionary import normalize, write_verified


def lock_for(raw, output):
    return {'source': {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()},
            'output': {'bytes': len(output), 'sha256': hashlib.sha256(output).hexdigest(),
                       'words': len(output.splitlines())}}


class EnglishPreparationTests(unittest.TestCase):
    def test_ascii_normalization_and_inflections(self):
        # Inflected words already exist upstream: no lossy .dic flag stripping.
        raw = "Apple\r\napple\r\nrun\r\nrunning\r\ncan't\r\ncafé\r\nx-ray\r\nA1\r\n".encode()
        self.assertEqual(normalize(raw), b'apple\nrun\nrunning\n')

    def test_exact_regeneration_is_repeatable(self):
        raw, expected = b'Banana\nApple\napple\n', b'apple\nbanana\n'
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'english.txt'
            for _ in range(2):
                self.assertEqual(write_verified(raw, output, lock_for(raw, expected)), 2)
                self.assertEqual(output.read_bytes(), expected)
            self.assertEqual([p.name for p in output.parent.iterdir()], ['english.txt'])

    def test_corruption_or_wrong_generation_does_not_replace_existing_file(self):
        raw, expected = b'apple\n', b'apple\n'
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'english.txt'
            output.write_bytes(b'previous dictionary')
            for source, lock in [(b'wrong\n', lock_for(raw, expected)),
                                 (raw, lock_for(raw, b'wrong\n'))]:
                with self.subTest(source=source), self.assertRaises(ValueError):
                    write_verified(source, output, lock)
                self.assertEqual(output.read_bytes(), b'previous dictionary')
            self.assertEqual([p.name for p in output.parent.iterdir()], ['english.txt'])

    def test_tracked_dictionary_and_notice_match_the_source_record(self):
        root = Path(__file__).resolve().parents[1]
        lock = json.loads((root / 'licenses/scowl/SOURCE.json').read_text())
        raw = (root / 'data/english_dict.txt').read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), lock['output']['sha256'])
        self.assertEqual(len(raw.splitlines()), lock['output']['words'])
        words = raw.decode('ascii').splitlines()
        self.assertEqual(words, sorted(set(words)))
        notice = (root / lock['copyright']['path']).read_bytes()
        self.assertEqual(hashlib.sha256(notice).hexdigest(), lock['copyright']['sha256'])


if __name__ == '__main__':
    unittest.main()
