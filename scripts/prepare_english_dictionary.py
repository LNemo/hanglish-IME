"""Rebuild the optional English list from pinned, permissively licensed SCOWL.

The word list retains its upstream license; it is not relicensed as CC BY-NC.
Use --source for offline regeneration. No Hunspell runtime is needed because
the upstream file is already an expanded plain word list.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / 'licenses/scowl/SOURCE.json'
MAX_DOWNLOAD_BYTES = 5_000_000


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def normalize(raw):
    words = {line.lower() for line in raw.decode('utf-8').splitlines()
             if re.fullmatch('[a-z]+', line.lower())}
    return ('\n'.join(sorted(words)) + '\n').encode('ascii')


def write_verified(raw, destination, lock):
    if len(raw) != lock['source']['bytes'] or digest(raw) != lock['source']['sha256']:
        raise ValueError('Source checksum/size mismatch; existing dictionary was not changed.')
    result = normalize(raw)
    expected = lock['output']
    if (digest(result) != expected['sha256'] or len(result) != expected['bytes']
            or len(result.splitlines()) != expected['words']):
        raise ValueError('Generated dictionary differs from the lock; existing file was not changed.')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.english-dictionary-', dir=destination.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'wb') as output:
            output.write(result)
        temporary.chmod(0o644)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return expected['words']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, help='Previously downloaded pinned en_US.txt')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/english_dict.txt')
    args = parser.parse_args(argv)
    lock = json.loads(LOCK.read_text())
    try:
        notice = ROOT / lock['copyright']['path']
        if digest(notice.read_bytes()) != lock['copyright']['sha256']:
            raise ValueError('Pinned copyright notice is missing or changed.')
        if args.source:
            raw = args.source.read_bytes()
        else:
            with urllib.request.urlopen(lock['source']['url'], timeout=30) as response:
                raw = response.read(MAX_DOWNLOAD_BYTES + 1)
        count = write_verified(raw, args.output, lock)
    except (OSError, ValueError) as error:
        parser.exit(1, f'{error}\n')
    print(f'Wrote {count:,} SCOWL words to {args.output}; upstream terms: licenses/scowl/Copyright')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
