"""Canonical release metadata preserves contracts without machine-local paths."""
import ast
import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / 'native/macos/Packaging/release_metadata.py'
spec = importlib.util.spec_from_file_location('release_metadata', MODULE)
metadata = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metadata)


def test_provenance_paths_normalized_without_changing_inference_or_license():
    original = {'format_version': 1, 'max_length': 96, 'artifact_hashes': {'weights.bin': 'abc'},
                'base_model': 'monologg/koelectra-small-v3-discriminator',
                'source': 'https://example.org/data/License.md', 'license': 'CC BY-SA 4.0',
                'source_hashes': {'/Users/person/project/data/train.jsonl': '123'},
                'provenance': [{'path': '/home/researcher/work/scripts/train.py'}]}
    cleaned = metadata.sanitize_metadata(original)
    assert cleaned['source_hashes'] == {'data/train.jsonl': '123'}
    assert cleaned['provenance'] == [{'path': 'scripts/train.py'}]
    assert {k: cleaned[k] for k in original if k not in ('source_hashes', 'provenance')} == {
        k: original[k] for k in original if k not in ('source_hashes', 'provenance')}
    assert '/Users/person' in next(iter(original['source_hashes']))
    assert metadata.sanitize_metadata(cleaned) == cleaned


def test_normalization_collision_is_rejected():
    with pytest.raises(ValueError, match='collision'):
        metadata.sanitize_metadata({
            '/Users/a/repo/data/train.json': 'old',
            'data/train.json': 'new',
        })


@pytest.mark.parametrize('value', ['/Users/person/secret.txt', 'file:///Users/person/secrets',
                                  'trace: /private/var/folders/session/error',
                                  '/Users/person/repo/data/../../secret'])
def test_unrecognized_local_paths_fail_closed_without_echoing_them(value):
    with pytest.raises(ValueError) as error:
        metadata.sanitize_metadata({'provenance': value})
    assert value not in str(error.value)


@pytest.mark.parametrize('value', [
    'https://example.org/home/license',
    'https://example.org/private/source.json',
    'http://example.org/download?path=/tmp/public-fixture',
    'See https://example.org/Volumes/reference for attribution',
])
def test_http_urls_are_preserved_even_with_path_like_segments(value):
    assert metadata.sanitize_metadata({'source': value}) == {'source': value}


def test_private_path_scanner_distinguishes_urls_from_local_paths():
    assert not metadata.PRIVATE_PATH.search(
        json.dumps({'source': 'https://example.org/private/license'})
    )
    assert metadata.PRIVATE_PATH.search(json.dumps({'source': '/tmp/private/license'}))


@pytest.mark.parametrize('value', [
    '/tmp/session/model.json',
    '/Volumes/Build/release.json',
    '/opt/local/share/model.json',
    r'C:\Users\person\secret.json',
    'trace: /tmp/session/error.log',
    'loaded from /Volumes/Build/release.json',
    'prefix=/opt/local/share/model.json',
])
def test_other_absolute_and_embedded_private_paths_fail_closed(value):
    with pytest.raises(ValueError) as error:
        metadata.sanitize_metadata({'provenance': value})
    assert value not in str(error.value)


def test_windows_repository_path_is_normalized():
    value = r'C:\Users\person\hanglish\data\train.jsonl'
    assert metadata.sanitize_metadata({'source': value}) == {
        'source': 'hanglish/data/train.jsonl'
    }


def test_numerical_model_fields_and_shape_are_unchanged():
    model = {'format': 'ctc', 'config': {'width': 64}, 'checkpoint_sha256': 'def',
             'provenance': {'pins': {'/Users/person/repo/hanglish/models/ctc.py': '456'}},
             'parameters': [1., -0.1, 0., 2.3], 'scale': [1., 2.]}
    cleaned = metadata.sanitize_metadata(model)
    assert set(cleaned) == set(model)
    for key in model.keys() - {'provenance'}:
        assert cleaned[key] == model[key]


def assigned_string(path, name):
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            value = ast.literal_eval(node.value)
            assert isinstance(value, str)
            return value
    raise AssertionError(f'{name} is not assigned in {path}')


def test_ctc_runtime_metadata_constants_match_asset_lock_without_model_imports():
    lock = json.loads(
        (ROOT / 'native/macos/Packaging/runtime-assets.lock.json').read_text(encoding='utf-8')
    )
    hashes = {entry['path']: entry['sha256'] for entry in lock['files']}
    expected = [
        (
            ROOT / 'hanglish/core/ctc_word_converter.py',
            'EXPECTED_CTC_METADATA_SHA256',
            '.omx/artifacts/sentence-ctc-pilot-v1/model.json',
        ),
        (
            ROOT / 'hanglish/core/ctc_union_converter.py',
            'EXPECTED_GROWTH_METADATA_SHA256',
            '.omx/artifacts/sentence-ctc-growth-pilot-v1/model.json',
        ),
    ]
    for source, constant, asset in expected:
        assert assigned_string(source, constant) == hashes[asset]
