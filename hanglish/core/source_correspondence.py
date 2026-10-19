"""Lossless source/output span correspondence; no labels, models, or scoring.

Offsets are Python Unicode code-point offsets, not bytes or browser UTF-16 units.
An existing producer segment is an alignment hypothesis, not phonetic evidence.
In particular, a mixed segment containing ASCII output is not silently split
into guessed COPY edges.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json


SCHEMA = 'source-correspondence-v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class Span:
    input_start: int
    input_end: int
    output_start: int
    output_end: int
    source: str
    text: str

    @property
    def action(self):
        return 'copy' if self.source == self.text else 'convert'

    @property
    def unresolved_ascii_output(self):
        """ASCII letters/digits inside a non-COPY span; not a copy failure."""
        return self.action == 'convert' and any(
            'a' <= c <= 'z' or 'A' <= c <= 'Z' or '0' <= c <= '9'
            for c in self.text)


@dataclass(frozen=True)
class Path:
    source: str
    text: str
    spans: tuple[Span, ...]

    @property
    def path_id(self):
        # Same output with different source boundaries remains a distinct path.
        value = [self.source, [(s.input_start, s.input_end, s.text) for s in self.spans]]
        payload = json.dumps(value, ensure_ascii=False, separators=(',', ':'))
        return 'alignment-v1-' + hashlib.sha256(payload.encode()).hexdigest()


def align(source, text, segments):
    require(isinstance(source, str) and isinstance(text, str)
            and isinstance(segments, list), 'source, output or segments malformed')
    input_cursor = output_cursor = 0
    spans = []
    for segment in segments:
        require(isinstance(segment, dict), 'segment malformed')
        start, end = segment.get('input_start'), segment.get('input_end')
        value = segment.get('text')
        require(type(start) is int and type(end) is int
                and start == input_cursor and start < end <= len(source),
                'source coverage must be forward and contiguous')
        require(segment.get('source') == source[start:end] and isinstance(value, str),
                'source slice or output malformed')
        spans.append(Span(start, end, output_cursor, output_cursor + len(value),
                          source[start:end], value))
        input_cursor = end
        output_cursor += len(value)
    require(input_cursor == len(source) and ''.join(s.text for s in spans) == text,
            'source coverage or complete output differs')
    return Path(source, text, tuple(spans))


def contrast(left, right):
    """Compare at shared source boundaries without guessing inside an edge.

    All blocks are returned, including unchanged context. Different segment
    boundaries with the same text are distinguished from actual output changes.
    Empty output spans are allowed; source spans always advance.
    """
    require(isinstance(left, Path) and isinstance(right, Path)
            and left.source == right.source, 'contrast requires the same source')
    boundaries = sorted(({0} | {s.input_end for s in left.spans})
                        & ({0} | {s.input_end for s in right.spans}))
    offsets = [{0: 0, **{s.input_end: s.output_end for s in path.spans}}
               for path in (left, right)]
    cursors = [0, 0]
    blocks = []
    for start, end in zip(boundaries, boundaries[1:]):
        sides = []
        signatures = []
        for index, path in enumerate((left, right)):
            members = []
            while cursors[index] < len(path.spans) and path.spans[cursors[index]].input_end <= end:
                members.append(path.spans[cursors[index]])
                cursors[index] += 1
            out_start, out_end = offsets[index][start], offsets[index][end]
            sides.append({'output_start': out_start, 'output_end': out_end,
                          'text': path.text[out_start:out_end],
                          'span_count': len(members),
                          'actions': [s.action for s in members]})
            signatures.append([(s.input_start, s.input_end, s.text) for s in members])
        relation = ('output_changed' if sides[0]['text'] != sides[1]['text'] else
                    'segmentation_only' if signatures[0] != signatures[1] else 'same')
        blocks.append({'input_start': start, 'input_end': end,
                       'source': left.source[start:end], 'left': sides[0], 'right': sides[1],
                       'relation': relation})
    return blocks


def compact_paths(paths):
    """Keep every distinct source path; share repeated edges to bound storage."""
    require(bool(paths) and all(p.source == paths[0].source for p in paths),
            'nonempty paths with the same source required')
    edges, edge_indices, records, seen = [], {}, [], set()
    for path in paths:
        path_id = path.path_id
        if path_id in seen:
            continue
        seen.add(path_id)
        indices = []
        for span in path.spans:
            key = (span.input_start, span.input_end, span.text)
            if key not in edge_indices:
                edge_indices[key] = len(edges)
                edges.append({'input_start': key[0], 'input_end': key[1], 'text': key[2]})
            indices.append(edge_indices[key])
        records.append({'path_id': path_id, 'text': path.text, 'edge_indices': indices})
    return {'schema': SCHEMA, 'input': paths[0].source, 'edges': edges, 'paths': records}


def expand_path(record, path_record):
    source, edges = record['input'], record['edges']
    segments = []
    for index in path_record['edge_indices']:
        require(type(index) is int and 0 <= index < len(edges), 'edge index malformed')
        edge = edges[index]
        segments.append(dict(edge, source=source[edge['input_start']:edge['input_end']]))
    path = align(source, path_record['text'], segments)
    require(path.path_id == path_record['path_id'], 'aligned path identity differs')
    return path
