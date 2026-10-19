"""Manual keyboard diagnostics must never run during automated test collection."""
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
NAMES = ('intercept', 'backspace', 'shift_left')


class ManualDiagnosticsTests(unittest.TestCase):
    def test_not_in_pytest_discovery(self):
        for name in ('test_intercept.py', 'test_quartz_bs.py', 'test_quartz_shift_left.py'):
            self.assertFalse((ROOT / 'tests' / name).exists(), name)

    def test_import_does_not_load_platform_hooks(self):
        code = '''
import importlib.abc, runpy, sys
class DenyHooks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('Quartz', 'pynput'):
            raise AssertionError('Platform hook loaded during import: ' + fullname)
sys.meta_path.insert(0, DenyHooks())
runpy.run_path(sys.argv[1])
'''
        for name in NAMES:
            path = ROOT / 'scripts' / 'manual' / (name + '.py')
            with self.subTest(name=name):
                result = subprocess.run([sys.executable, '-I', '-c', code, str(path)],
                                        capture_output=True, text=True, timeout=5)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, '')

    def test_execution_requires_explicit_run_flag(self):
        for name in NAMES:
            path = ROOT / 'scripts' / 'manual' / (name + '.py')
            with self.subTest(name=name):
                result = subprocess.run([sys.executable, '-I', str(path)],
                                        capture_output=True, text=True, timeout=5)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn('--run', result.stderr)
                self.assertNotIn('ModuleNotFoundError', result.stderr)


if __name__ == '__main__':
    unittest.main()
