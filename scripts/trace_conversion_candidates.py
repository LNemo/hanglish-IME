"""Trace one conversion's bounded candidate flow without reading target labels."""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hanglish.core.neural_converter import (  # noqa: E402
    NeuralConverter, PROTECTED, TOKENS, WordOption, beam_paths,
)


def option_record(option):
    return {"text": option.text, "score": float(option.score), "kind": option.kind}


def piece_records(pieces):
    return [
        {"source": source, "offset": offset,
         "options": [option_record(option) for option in options]}
        for source, offset, options in pieces
    ]


def path_records(paths):
    return [
        {"text": text, "score": float(score), "segments": segments}
        for text, score, segments in paths
    ]


def bounded_beam_trace(pieces):
    """Replay the runtime's 8-per-word/24-per-sentence bounded beam."""
    paths = [("", 0.0, [])]
    steps = []
    for piece_index, (source, offset, options) in enumerate(pieces):
        used = options[:8]
        expanded = []
        for prefix, score, segments in paths:
            for option in used:
                segment = {
                    "input_start": offset, "input_end": offset + len(source),
                    "source": source, "text": option.text, "kind": option.kind,
                }
                expanded.append((prefix + option.text, score + option.score,
                                 segments + [segment]))
        unique = {}
        for item in sorted(expanded, key=lambda item: (-item[1], item[0])):
            unique.setdefault(item[0], item)
        paths = list(unique.values())[:24]
        steps.append({
            "piece_index": piece_index, "source": source,
            "input_path_count": max(1, len(expanded) // max(1, len(used))),
            "available_options": len(options), "used_options": len(used),
            "expanded_count": len(expanded), "unique_count": len(unique),
            "retained_count": len(paths), "retained": path_records(paths),
        })
    canonical = beam_paths(pieces)
    if paths != canonical:
        raise RuntimeError("bounded beam replay differs from runtime beam_paths")
    return paths, steps


def displayed_candidates(paths, source):
    candidates = [
        {"text": output, "score": round(score, 4), "segments": segments}
        for output, score, segments in paths[:5]
    ]
    if not any(candidate["text"] == source for candidate in candidates):
        original = {
            "text": source, "score": None,
            "segments": [{
                "input_start": 0, "input_end": len(source), "source": source,
                "text": source, "kind": "raw",
            }],
        }
        candidates = candidates[:4] + [original]
    return candidates


def _cache_state(callable_):
    info = getattr(callable_, "cache_info", None)
    return info()._asdict() if info is not None else None


class PairRecorder:
    """Transparent ranker wrapper that records exact score_pairs traffic."""
    def __init__(self, delegate, stage):
        self.delegate = delegate
        self.stage = stage
        self.calls = []

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def score_pairs(self, contexts, candidates):
        contexts = list(contexts)
        candidates = list(candidates)
        returned = self.delegate.score_pairs(contexts, candidates)
        scores = list(returned)
        if (len(contexts) != len(candidates) or len(scores) != len(candidates)
                or any(type(value) not in (int, float) or not math.isfinite(value)
                       for value in scores)):
            raise RuntimeError(f"{self.stage} ranker returned malformed scores")
        self.calls.append({"contexts": contexts, "candidates": candidates,
                           "scores": [float(value) for value in scores]})
        return returned


class TraceConverter(NeuralConverter):
    """NeuralConverter with transparent, per-call source-only instrumentation."""
    def __init__(self, *args, **kwargs):
        self.generator_calls = []
        self.word_option_calls = []
        self.word_option_computations = []
        self.trace_pieces = None
        self.trace_static_pieces = None
        self.trace_numeric_extras = []
        self._generator_wrapped_object = None
        self._word_wrapper_installed = False
        self._ranker_wrapped_object = None
        self._sentence_wrapped_object = None
        super().__init__(*args, **kwargs)
        self._install_source_wrappers()

    def _word_options(self, source, learned):
        returned = super()._word_options(source, learned)
        self.word_option_computations.append({
            "source": source, "learned": bool(learned),
            "options": [option_record(option) for option in returned],
        })
        return returned

    def _install_source_wrappers(self):
        if self._generator_wrapped_object is not self.generator:
            generator_options = self.generator.options

            def traced_generator(source):
                before = _cache_state(generator_options)
                returned = generator_options(source)
                after = _cache_state(generator_options)
                self.generator_calls.append({
                    "source": source, "cache_before": before, "cache_after": after,
                    "options": [option_record(option) for option in returned],
                })
                return returned

            self.generator.options = traced_generator
            self._generator_wrapped_object = self.generator
        if not self._word_wrapper_installed:
            cached_word_options = self.word_options

            def traced_word(source, learned):
                before = _cache_state(cached_word_options)
                returned = cached_word_options(source, learned)
                after = _cache_state(cached_word_options)
                self.word_option_calls.append({
                    "source": source, "learned": bool(learned),
                    "cache_before": before, "cache_after": after,
                    "cache_hit": (before is not None and after is not None
                                  and after["hits"] == before["hits"] + 1),
                    "options": [option_record(option) for option in returned],
                })
                return returned

            self.word_options = traced_word
            self._word_wrapper_installed = True

    def _install_model_recorders(self):
        if self.ranker is not None and self._ranker_wrapped_object is not self.ranker:
            delegate = self.ranker.delegate if isinstance(self.ranker, PairRecorder) else self.ranker
            self.ranker = PairRecorder(delegate, "word")
            self._ranker_wrapped_object = self.ranker
        if (self.sentence_ranker is not None
                and self._sentence_wrapped_object is not self.sentence_ranker):
            delegate = (self.sentence_ranker.delegate
                        if isinstance(self.sentence_ranker, PairRecorder)
                        else self.sentence_ranker)
            self.sentence_ranker = PairRecorder(delegate, "sentence")
            self._sentence_wrapped_object = self.sentence_ranker

    def load_models(self):
        super().load_models()
        self._install_model_recorders()

    def _prepare_candidates(self, *args, **kwargs):
        result = super()._prepare_candidates(*args, **kwargs)
        self.trace_pieces, self.trace_static_pieces = result[:2]
        return result

    def _numeric_sentence_extras(self, pieces, static_pieces, complete):
        returned = super()._numeric_sentence_extras(pieces, static_pieces, complete)
        self.trace_numeric_extras = list(returned)
        return returned

    def reset_trace(self):
        self.generator_calls.clear()
        self.word_option_calls.clear()
        self.word_option_computations.clear()
        self.trace_pieces = None
        self.trace_static_pieces = None
        self.trace_numeric_extras = []
        for ranker in (self.ranker, self.sentence_ranker):
            if isinstance(ranker, PairRecorder):
                ranker.calls.clear()


def _protected_chunks(source, static_pieces):
    records = []
    for match in TOKENS.finditer(source):
        chunk = match.group()
        if not PROTECTED.fullmatch(chunk):
            continue
        matching = [piece for piece in static_pieces
                    if piece[0] == chunk and piece[1] == match.start()]
        if (len(matching) == 1 and len(matching[0][2]) == 1
                and matching[0][2][0] == WordOption(chunk, 0.0, "raw")):
            records.append({"source": chunk, "offset": match.start(),
                            "preserved_raw": True})
    return records


def trace_conversion(converter, source, context="", *, engine="neural", raw=False):
    if not isinstance(source, str) or not isinstance(context, str):
        raise TypeError("source and context must be strings")
    converter.reset_trace()
    response = converter.convert(source, context, engine=engine, raw=raw)
    pieces = converter.trace_pieces
    static_pieces = converter.trace_static_pieces
    if pieces is None or static_pieces is None:
        raise RuntimeError("converter did not expose prepared candidate pieces")

    word_paths, word_steps = bounded_beam_trace(pieces)
    static_paths, static_steps = bounded_beam_trace(static_pieces)
    proposals = {path[0]: path for path in word_paths}
    for path in static_paths:
        proposals.setdefault(path[0], path)
    complete = list(proposals.values())

    sentence_calls = (converter.sentence_ranker.calls
                      if isinstance(converter.sentence_ranker, PairRecorder) else [])
    if sentence_calls:
        if len(sentence_calls) != 1:
            raise RuntimeError("sentence ranker was called more than once")
        complete.extend(converter.trace_numeric_extras)
        observed = sentence_calls[0]
        if observed["candidates"] != [path[0] for path in complete]:
            raise RuntimeError("recorded sentence inputs differ from reconstructed bounded pool")
        final_paths = sorted(
            [(path[0], score, path[2])
             for path, score in zip(complete, observed["scores"])],
            key=lambda path: (-path[1], path[0]),
        )
    else:
        final_paths = word_paths
    reconstructed_display = displayed_candidates(final_paths, source)
    if reconstructed_display != response["candidates"]:
        raise RuntimeError("trace replay differs from actual displayed candidates")

    protected = _protected_chunks(source, static_pieces)
    protected_sources = {item["source"] for item in protected}
    if any(call["source"] in protected_sources
           for call in converter.generator_calls + converter.word_option_calls):
        raise RuntimeError("an explicitly protected raw chunk entered conversion")

    word_calls = (converter.ranker.calls
                  if isinstance(converter.ranker, PairRecorder) else [])
    trace = {
        "schema": "conversion-candidate-trace-v1",
        "source": source, "context": context, "engine": engine, "raw": bool(raw),
        "source_only": True, "target_fields_read": False,
        "limits": {"word_options_per_piece": 8, "sentence_paths": 24,
                   "displayed_candidates": 5, "unbounded_cartesian_enumerated": False},
        "mechanical_generator_calls": converter.generator_calls,
        "word_option_calls": converter.word_option_calls,
        "word_option_computations": converter.word_option_computations,
        "pieces_before_word_ranking": piece_records(static_pieces),
        "pieces_after_word_ranking": piece_records(pieces),
        "word_ranker_score_pairs": word_calls,
        "word_ranked_beam": {"steps": word_steps, "final": path_records(word_paths)},
        "static_beam": {"steps": static_steps, "final": path_records(static_paths)},
        "sentence_score_pairs": sentence_calls,
        "numeric_sentence_extras": path_records(converter.trace_numeric_extras),
        "protected_chunks": protected,
        "final_pre_display_paths": path_records(final_paths),
        "displayed_candidates": response["candidates"],
        "result": response["result"],
        "parity_verified": True,
        "runtime": {key: value for key, value in response.items()
                    if key not in {"input", "context", "candidates", "result", "engine_ms"}},
    }
    # A converter can trace another call later; returned evidence must not share
    # the mutable recorder lists that reset_trace clears on the next request.
    return copy.deepcopy(trace)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("--context", default="")
    parser.add_argument("--engine", choices=("neural", "mechanical"), default="neural")
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--ranker-kind", default="auto",
                        choices=("auto", "mlp", "semantic", "sentence", "hybrid"))
    parser.add_argument("--no-external-lexicon", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    converter = TraceConverter(external_lexicon=not args.no_external_lexicon,
                               ranker_kind=args.ranker_kind)
    trace = trace_conversion(converter, args.source, args.context,
                             engine=args.engine, raw=args.raw)
    payload = json.dumps(trace, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    else:
        sys.stdout.write(payload)


if __name__ == "__main__":
    main()
