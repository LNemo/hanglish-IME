"""Package POS-backed boundary evidence from the pinned local MeCab dictionary."""
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/external/mecab-ko-dic'
OUTPUT = ROOT / 'hanglish/core/data/predicate_boundary_inventory.json'


def build():
    manifest = json.loads((SOURCE/'SOURCE.json').read_text())
    stems, endings, particles = set(), {}, {}
    hashes = {}
    for name in ('VV', 'VA', 'VX', 'Inflect'):
        path = SOURCE/(name+'.csv')
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != manifest['files'][path.name]:
            raise ValueError('dictionary source hash changed: '+path.name)
        hashes[path.name] = digest
        for row in csv.reader(raw.decode().splitlines()):
            if len(row)<12:
                continue
            if name!='Inflect':
                if row[4]==name and re.fullmatch('[가-힣]+',row[0]) and (ord(row[0][-1])-0xAC00)%28==9:
                    stems.add(row[0])
                continue
            for analysis in row[11].split('+'):
                parts = analysis.split('/')
                if len(parts)<2 or not re.fullmatch('[가-힣]+',parts[0]):
                    continue
                text,pos = parts[:2]
                if pos in {'EC','EF','ETN','ETM'} and (ord(text[0])-0xAC00)//588==0:
                    endings.setdefault(text,set()).add(pos)
                elif pos.startswith('J'):
                    particles.setdefault(text,set()).add(pos)
    value = {'schema':'predicate-boundary-inventory-v1','source':manifest['source'],
             'source_version':manifest['version'],'license':'Apache-2.0',
             'attribution':'Derived from mecab-ko-dic; see mecab-ko-dic.COPYING in this directory.',
             'source_hashes':hashes,'stems':sorted(stems),
             'endings':{s:sorted(tags) for s,tags in sorted(endings.items())},
             'particles':{s:sorted(tags) for s,tags in sorted(particles.items())}}
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    shutil.copyfile(SOURCE/'COPYING',OUTPUT.parent/'mecab-ko-dic.COPYING')
    print(json.dumps({'stems':len(stems),'endings':len(endings),'particles':len(particles)}))


if __name__=='__main__':
    build()
