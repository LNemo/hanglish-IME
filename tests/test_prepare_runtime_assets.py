"""Runtime preparation uses pinned bytes without executing model contents."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.prepare_runtime_assets import asset_path, check_assets, import_assets, read_manifest


def entry(name, data=b'asset', profiles=None):
    return {'path': name, 'size': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
            'profiles': profiles or ['package']}


class RuntimeAssetsTests(unittest.TestCase):
    def test_repository_lock_covers_package_manifest(self):
        root = Path(__file__).resolve().parents[1]
        packaging = root / 'native/macos/Packaging'
        specification = json.loads((packaging / 'engine-files.json').read_text())
        locked = json.loads((packaging / 'runtime-assets.lock.json').read_text())
        expected = set(specification['required']['data'])
        expected.update(specification['required']['models_and_metadata'])
        expected.update(specification['optional_compatibility']['files'])
        expected.update(specification['licenses_and_provenance'])
        self.assertEqual({item['path'] for item in locked['files']}, expected)
        self.assertEqual(locked['engine_version'], specification['selected_version'])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.source = self.base / 'source'
        self.target = self.base / 'target'
        self.source.mkdir()
        self.target.mkdir()

    def test_import_is_exact_and_idempotent(self):
        item = entry('models/model.pt', b'not an executable model')
        (self.source / 'models').mkdir()
        (self.source / item['path']).write_bytes(b'not an executable model')
        self.assertEqual(import_assets(self.target, self.source, [item]), [item['path']])
        self.assertEqual(import_assets(self.target, self.source, [item]), [])
        self.assertEqual(check_assets(self.target, [item]), {'missing': [], 'mismatched': []})

    def test_preflight_checks_all_sources_before_any_copy(self):
        (self.source / 'first').write_bytes(b'asset')
        with self.assertRaises(ValueError):
            import_assets(self.target, self.source, [entry('first'), entry('missing')])
        self.assertEqual(list(self.target.iterdir()), [])

    def test_does_not_overwrite_existing_modified_asset(self):
        (self.source / 'file').write_bytes(b'asset')
        (self.target / 'file').write_bytes(b'user content')
        with self.assertRaisesRegex(ValueError, 'refusing to overwrite'):
            import_assets(self.target, self.source, [entry('file')])
        self.assertEqual((self.target / 'file').read_bytes(), b'user content')

    def test_rejects_same_size_corruption(self):
        (self.source / 'file').write_bytes(b'wrong')
        with self.assertRaises(ValueError):
            import_assets(self.target, self.source, [entry('file')])
        self.assertFalse((self.target / 'file').exists())

    def test_missing_and_mismatched_are_reported_separately(self):
        (self.target / 'bad').write_bytes(b'wrong')
        self.assertEqual(check_assets(self.target, [entry('missing'), entry('bad')]),
                         {'missing': ['missing'], 'mismatched': ['bad']})

    def test_rejects_path_traversal_and_absolute_paths(self):
        for path in ('../outside', '/absolute', 'nested/../../outside', r'nested\outside', ''):
            with self.subTest(path=path), self.assertRaises(ValueError):
                asset_path(self.target, path)

    def test_rejects_source_and_destination_symlinks(self):
        (self.base / 'outside').write_bytes(b'asset')
        (self.source / 'file').symlink_to(self.base / 'outside')
        with self.assertRaises(ValueError):
            import_assets(self.target, self.source, [entry('file')])
        (self.source / 'file').unlink()
        (self.source / 'file').write_bytes(b'asset')
        (self.target / 'file').symlink_to(self.base / 'outside')
        with self.assertRaises(ValueError):
            import_assets(self.target, self.source, [entry('file')])

    def test_manifest_profiles_and_duplicate_validation(self):
        manifest = self.base / 'manifest.json'
        baseline = entry('small.json', profiles=['baseline', 'package'])
        model = entry('model.pt')
        data = {'schema': 'hanglish-runtime-assets-v1', 'files': [baseline, model]}
        manifest.write_text(json.dumps(data))
        self.assertEqual(read_manifest(manifest, 'baseline'), [baseline])
        self.assertEqual(read_manifest(manifest, 'package'), [baseline, model])
        data['files'].append(baseline)
        manifest.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            read_manifest(manifest, 'package')


if __name__ == '__main__':
    unittest.main()
