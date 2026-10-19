"""The shared native-service baseline must work with only tracked seed data."""
import unittest

from hanglish.core.workbench import WorkbenchEngine


class BaselineAssetsTests(unittest.TestCase):
    def test_authored_seed_restores_phonetic_spelling(self):
        result = WorkbenchEngine().convert('imnida', engine='baseline')
        self.assertEqual(result['candidates'][0]['text'], '입니다')

    def test_protected_english_and_particle(self):
        result = WorkbenchEngine().convert('testreul', engine='baseline')
        self.assertEqual(result['candidates'][0]['text'], 'test를')


if __name__ == '__main__':
    unittest.main()
