"""Experimental, bounded Korean quantity/date annotations in numeric chunks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from hanglish.core.lattice import WordOption, suffix_allowed
from hanglish.core.neural_converter import NeuralConverter, NUMERIC_PARTICLES, file_sha256
from hanglish.core.phonology import romanize_word

NUMBER = r'[0-9]+(?:\.[0-9]+)?'
ANNOTATED = re.compile(r'('+NUMBER+r')([a-z]{1,40})\(('+NUMBER+r'[%％]|[a-z]{1,20})\)([a-z]{0,20})([.,!?]?)\Z')
PERCENTAGE = re.compile(NUMBER+r'[%％]\Z')
WEEKDAYS = {romanize_word(day):day for day in ('월','화','수','목','금','토','일',
                                              '월요일','화요일','수요일','목요일','금요일','토요일','일요일')}
PARTICLES = {}
for _particle in NUMERIC_PARTICLES:
    PARTICLES.setdefault(romanize_word(_particle),[]).append(_particle)


def raw_piece(source, offset):
    return source,offset,(WordOption(source,0.,'raw'),)


def annotation_pieces(chunk, offset, learned, parse_quantity):
    """Use an existing quantity parser; return None when this grammar cannot prove a split."""
    if len(chunk)>96:
        return None
    match=ANNOTATED.fullmatch(chunk)
    if match is None:
        return None
    number,unit_source,annotation,tail_source,punctuation=match.groups()
    is_percentage=PERCENTAGE.fullmatch(annotation) is not None
    if not is_percentage and (unit_source!='il' or not number.isdecimal()
                              or not 1<=int(number)<=31 or annotation not in WEEKDAYS):
        return None
    quantity=parse_quantity(number+unit_source,offset,learned)
    if quantity is None or len(quantity)!=2:
        return None
    number_piece,unit_piece=quantity
    if number_piece[0]!=number or unit_piece[:2]!=(unit_source,offset+len(number)):
        raise ValueError('quantity parser returned unexpected source alignment')
    unit_options=[o for o in unit_piece[2] if o.kind!='raw']
    # The quantity before parentheses must be a bare unit, not unit+particle.
    # This is checked below by the concrete converter's restricted parser.
    tails=PARTICLES.get(tail_source,[]) if tail_source else ['']
    if not tails:
        return None
    allowed_units=[o for o in unit_options if any(not tail or suffix_allowed(o.text,tail) for tail in tails)]
    if not allowed_units:
        return None
    allowed_tails=[tail for tail in tails if all(not tail or suffix_allowed(o.text,tail) for o in allowed_units)]
    if not allowed_tails:
        return None
    original_unit=next((o for o in unit_piece[2] if o.text==unit_source and o.kind=='raw'),WordOption(unit_source,0.,'raw'))
    pieces=[number_piece,(unit_source,unit_piece[1],tuple(allowed_units+[original_unit]))]
    cursor=offset+len(number)+len(unit_source)
    if is_percentage:
        literal='('+annotation+')'
        pieces.append(raw_piece(literal,cursor));cursor+=len(literal)
    else:
        pieces.append(raw_piece('(',cursor));cursor+=1
        pieces.append((annotation,cursor,(WordOption(WEEKDAYS[annotation],6.,'morphology'),
                                         WordOption(annotation,0.,'raw'))))
        cursor+=len(annotation)
        pieces.append(raw_piece(')',cursor));cursor+=1
    if tail_source:
        pieces.append((tail_source,cursor,tuple([WordOption(tail,6.,'morphology') for tail in allowed_tails]
                                               +[WordOption(tail_source,0.,'raw')])))
        cursor+=len(tail_source)
    if punctuation:
        pieces.append(raw_piece(punctuation,cursor))
    return pieces


class NumericParenthesesConverter(NeuralConverter):
    def __init__(self, baseline, *, experimental=False):
        if experimental is not True:
            raise ValueError('NumericParenthesesConverter requires experimental=True')
        super().__init__(external_lexicon=False,context_ranking=baseline.context_ranking,
                         ranker_kind=baseline.ranker_kind)
        self.baseline=baseline
        self.generator=baseline.generator
        self.numeric_units=baseline.numeric_units
        self._stamp(baseline.version,getattr(baseline,'model_artifacts',{}))

    def _stamp(self,parent_version,artifacts):
        source_sha=file_sha256(Path(__file__))
        digest=hashlib.sha256(json.dumps({'parent':parent_version,'source':source_sha},sort_keys=True).encode()).hexdigest()
        self.version='experimental-numeric-parentheses-1-'+digest[:12]
        self.model_artifacts=dict(artifacts)
        self.model_artifacts['experimental/numeric_parentheses.py']=source_sha

    def load_models(self):
        with self.lock,self.baseline.lock:
            if self.models_loaded:
                return
            self.baseline.load_models()
            for name in ('generator','numeric_units','ranker','sentence_ranker','transducer','ranker_kind'):
                setattr(self,name,getattr(self.baseline,name))
            self._stamp(self.baseline.version,self.baseline.model_artifacts)
            self.models_loaded=True

    def _bare_quantity(self,chunk,offset,learned):
        pieces=super()._numeric_unit_pieces(chunk,offset,learned)
        if pieces is None:
            return None
        source,start,options=pieces[1]
        # Use only the installed unit inventory here; suffix-bearing quantities
        # retain the original protected-chunk behavior for this experiment.
        units=[o for o in options if o.kind=='raw' or o.text in self.numeric_units]
        if not any(o.kind!='raw' for o in units):
            return None
        return [pieces[0],(source,start,tuple(units))]

    def _numeric_unit_pieces(self,chunk,offset,learned):
        ordinary=super()._numeric_unit_pieces(chunk,offset,learned)
        if ordinary is not None:
            return ordinary
        return annotation_pieces(chunk,offset,learned,self._bare_quantity)

    def convert(self,*args,**kwargs):
        with self.lock,self.baseline.lock:
            return super().convert(*args,**kwargs)
