"""Read version-pinned MeCab lexical CSVs without a native MeCab runtime."""

import csv
import hashlib
from pathlib import Path
import re


HANGUL = re.compile(r'[가-힣]{1,12}\Z')
FILES = ('NNG', 'NNP', 'NP', 'NR', 'MAG', 'VV', 'VA', 'VX', 'VCP', 'VCN', 'XR', 'Inflect')


def read_attested_ni_forms(directory: Path):
    """Observed connective surfaces; never infer irregularity from numeric IDs."""
    path = directory / 'Inflect.csv'
    if not path.is_file():
        return {}
    forms = {}
    for row in csv.reader(path.read_text(encoding='utf-8').splitlines()):
        if len(row) < 12 or not HANGUL.fullmatch(row[0]):
            continue
        parts = [part.split('/') for part in row[11].split('+')]
        if (len(parts) == 2 and all(len(part) >= 2 for part in parts)
                and parts[0][1] in {'VV', 'VA', 'VX'}
                and HANGUL.fullmatch(parts[0][0])
                and parts[1][0] in {'으니', '니'} and parts[1][1] == 'EC'
                and row[0].endswith('니')):
            forms.setdefault(parts[0][0], set()).add(row[0])
    return forms


def read_counter_units(directory: Path):
    """Read quantity-unit surfaces separately from the general word index."""
    path = directory / 'NNBC.csv'
    if not path.is_file():
        return ()
    rows = csv.reader(path.read_text(encoding='utf-8').splitlines())
    units = {row[0] for row in rows if len(row) >= 12 and row[4] == 'NNBC' and HANGUL.fullmatch(row[0])}
    return tuple(sorted(units, key=lambda word: (-len(word), word)))


def invalid_past_duplicate(row, siblings):
    """Reject malformed 이+었 contractions only with a matching valid sibling.

    This checks morphology, not a word blacklist; a lexical noun 엿 stays valid.
    """
    if len(row) < 12 or not row[0].endswith('엿'):
        return False
    parts = [part.split('/') for part in row[11].split('+')]
    if len(parts) < 2 or any(len(part) < 2 for part in parts[-2:]):
        return False
    stem, ending = parts[-2:]
    return (stem[0].endswith('이') and stem[1] in {'VV', 'VA', 'VX', 'VCP'}
            and ending[:2] == ['었', 'EP']
            and (row[0][:-1] + '였', row[11]) in siblings)


def read_lexicon(directory: Path):
    """Return lexical surfaces/POS/cost and an exact source fingerprint.

    MeCab surfaces include bound stems and inflected forms. Keeping POS permits
    inflecting predicates rather than treating every noun as a verb.
    """
    digest = hashlib.sha256()
    entries = []
    rejected = 0
    for name in FILES:
        path = directory / (name + '.csv')
        if not path.is_file():
            continue
        raw = path.read_bytes()
        digest.update(path.name.encode())
        digest.update(raw)
        rows = list(csv.reader(raw.decode('utf-8').splitlines()))
        siblings = {(row[0], row[11]) for row in rows if len(row) >= 12} if name == 'Inflect' else set()
        for row in rows:
            if len(row) >= 12 and HANGUL.fullmatch(row[0]):
                if name == 'Inflect' and invalid_past_duplicate(row, siblings):
                    rejected += 1
                    continue
                entries.append((row[0], row[4], int(row[3])))
    return entries, digest.hexdigest() if entries else 'absent', rejected
