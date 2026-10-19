"""Check or import exact runtime assets from a trusted local engine/source tree.

This command never downloads or deserializes models, and never overwrites a
conflicting file. Hashes identify the development engine, not redistribution rights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'native/macos/Packaging/runtime-assets.lock.json'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def asset_path(root, name):
    relative = PurePosixPath(name)
    if not relative.parts or relative.is_absolute() or any(p in ('.', '..') for p in relative.parts) or '\\' in name:
        raise ValueError(f'Unsafe asset path: {name}')
    root = Path(root).resolve()
    path = root.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(root):
        raise ValueError(f'Asset path escapes root: {name}')
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f'Symlink is not accepted for an asset: {name}')
    return path


def read_manifest(path, profile):
    data = json.loads(Path(path).read_text())
    if data.get('schema') != 'hanglish-runtime-assets-v1':
        raise ValueError('Unsupported runtime asset manifest')
    entries = []
    seen = set()
    for entry in data['files']:
        name = entry['path']
        asset_path(Path.cwd(), name)
        digest = entry['sha256']
        if name in seen or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError(f'Invalid or duplicate asset entry: {name}')
        seen.add(name)
        if not isinstance(entry['size'], int) or entry['size'] < 0:
            raise ValueError(f'Invalid asset size: {name}')
        if profile == 'package' or 'baseline' in entry.get('profiles', []):
            entries.append(entry)
    if not entries:
        raise ValueError(f'Empty asset profile: {profile}')
    return entries


def matches(path, entry):
    return path.is_file() and path.stat().st_size == entry['size'] and sha256(path) == entry['sha256']


def check_assets(root, entries):
    missing, mismatched = [], []
    for entry in entries:
        path = asset_path(root, entry['path'])
        if not path.exists():
            missing.append(entry['path'])
        elif not matches(path, entry):
            mismatched.append(entry['path'])
    return {'missing': missing, 'mismatched': mismatched}


def import_assets(root, source, entries):
    root, source = Path(root).resolve(), Path(source).resolve()
    planned = []
    # Validate the entire import before creating any destination files.
    for entry in entries:
        destination = asset_path(root, entry['path'])
        if destination.exists():
            if not matches(destination, entry):
                raise ValueError(f'Existing file differs; refusing to overwrite: {entry["path"]}')
            continue
        origin = asset_path(source, entry['path'])
        if not matches(origin, entry):
            raise ValueError(f'Source file is missing or does not match the lock: {entry["path"]}')
        planned.append((entry, origin, destination))
    imported = []
    for entry, origin, destination in planned:
        # Recheck path components in case they changed after preflight.
        destination = asset_path(root, entry['path'])
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix='.hanglish-asset-', dir=destination.parent)
        temporary = Path(temporary)
        try:
            with os.fdopen(fd, 'wb') as output, origin.open('rb') as input_file:
                shutil.copyfileobj(input_file, output)
            if not matches(temporary, entry):
                raise ValueError(f'Source changed during import: {entry["path"]}')
            # Hard-link creation is atomic and refuses to overwrite a race-created file.
            os.link(temporary, destination)
            imported.append(entry['path'])
        finally:
            temporary.unlink(missing_ok=True)
    return imported


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('check', 'import'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--profile', choices=('baseline', 'package'), default='package')
    parser.add_argument('--from', dest='source', type=Path,
                        help='Trusted repository root or installed app Contents/Resources/Engine directory')
    args = parser.parse_args(argv)
    try:
        entries = read_manifest(args.manifest, args.profile)
        imported = []
        if args.command == 'import':
            if args.source is None:
                parser.error('import requires --from with a trusted local asset directory')
            imported = import_assets(args.root, args.source, entries)
        result = check_assets(args.root, entries)
        result.update(profile=args.profile, checked=len(entries), imported=imported)
        result['ready'] = not result['missing'] and not result['mismatched']
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result['ready'] else 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({'ready': False, 'error': str(error)}, ensure_ascii=False))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
