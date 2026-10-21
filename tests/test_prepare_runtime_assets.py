"""Runtime preparation uses pinned bytes without executing model contents."""
import hashlib
import json
import plistlib
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

    def test_release_source_closure_excludes_rejected_context_ranker(self):
        root = Path(__file__).resolve().parents[1]
        specification = json.loads(
            (root / 'native/macos/Packaging/engine-files.json').read_text()
        )
        selected = {
            path.relative_to(root).as_posix()
            for pattern in specification['required']['source_packages']
            for path in root.glob(pattern)
            if path.is_file()
        }
        self.assertNotIn('hanglish/models/guarded_semantic_ranker.py', selected)
        self.assertIn('hanglish/models/sentence_ctc.py', selected)
        self.assertIn('hanglish/models/phonetic_transducer.py', selected)
        packaged_models = set(specification['required']['models_and_metadata'])
        packaged_models.update(specification['optional_compatibility']['files'])
        self.assertFalse(any('context-ranking-improvement' in path
                             for path in packaged_models))

    def test_release_model_selection_matches_locked_assets(self):
        root = Path(__file__).resolve().parents[1]
        packaging = root / 'native/macos/Packaging'
        specification = json.loads((packaging / 'engine-files.json').read_text())
        locked = json.loads((packaging / 'runtime-assets.lock.json').read_text())
        hashes = {entry['path']: entry['sha256'] for entry in locked['files']}
        selection = specification['release_model_selection']
        self.assertEqual(selection['release_version'], plistlib.loads(
            (root / 'native/macos/Info.plist').read_bytes()
        )['CFBundleShortVersionString'])
        self.assertTrue(selection['training_metadata_flags_are_historical'])
        self.assertEqual(
            selection['model_license_manifest'],
            'licenses/hanglish-models/models.json',
        )
        for model in selection['models'].values():
            self.assertEqual(hashes[model['checkpoint']], model['checkpoint_sha256'])
            self.assertEqual(hashes[model['metadata']], model['metadata_sha256'])
            self.assertNotIn('context-ranking-improvement', model['checkpoint'])

    def test_klue_trained_weights_have_separate_cc_by_sa_license(self):
        root = Path(__file__).resolve().parents[1]
        packaging = root / 'native/macos/Packaging'
        locked = json.loads((packaging / 'runtime-assets.lock.json').read_text())
        licenses = json.loads(
            (root / 'licenses/hanglish-models/models.json').read_text()
        )
        expected = {
            'checkpoints/sentence_ranker/model.safetensors':
                'a68f3bd26733e9bc897a182f78b1decafefa3582fa82b6b2cfcaa34672412ffa',
            '.omx/artifacts/ctc-rr-retraining-20261020/training-word/model.pt':
                'fb88a38dc8070fa0877b4d98d1a117940d94b132d33b847610f8bb2b7f89a9c7',
            '.omx/artifacts/ctc-rr-retraining-20261020/training-growth/model.pt':
                '1312683a4e3b3238a793ad2bea2a7427c98953a5bb4996ea91d5e43a6d22108e',
        }
        self.assertEqual(licenses['license'], 'CC-BY-SA-4.0')
        self.assertEqual(
            {item['path']: item['sha256'] for item in licenses['weights']}, expected
        )
        entries = {item['path']: item for item in locked['files']}
        self.assertEqual(
            {path for path, item in entries.items()
             if item['distribution'] == 'CC-BY-SA-4.0'},
            set(expected),
        )
        for path, digest in expected.items():
            self.assertEqual(entries[path]['sha256'], digest)
            self.assertEqual(entries[path]['distribution'], 'CC-BY-SA-4.0')
            self.assertGreater(entries[path]['size'], 0)
        self.assertIn('Attribution-ShareAlike 4.0 International',
                      (root / 'licenses/hanglish-models/LICENSE.txt').read_text())
        self.assertTrue((root / 'licenses/hanglish-models/NOTICE.txt').is_file())

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
