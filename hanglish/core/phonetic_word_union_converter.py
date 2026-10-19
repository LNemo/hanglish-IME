"""Experimental union of production CTC paths and phonetic word proposals."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import threading
import time

from hanglish.core.ctc_union_converter import CTCUnionConverter
from hanglish.core.neural_converter import ROOT, file_sha256
from hanglish.core.phonology import romanize_word


_ELIGIBLE = re.compile(r"[a-z]{2,40}")
_MAX_BEAM = 8
_MAX_EXTRAS = 96
_BATCHED_DECODER = ROOT / "hanglish/models/batched_phonetic_decode.py"


class PhoneticWordUnionConverter(CTCUnionConverter):
    """Add single-span phonetic alternatives to the complete production pool.

    The parent CTC union runs exactly once. Its tuples and scores are retained;
    new paths only replace one already-existing lowercase ASCII segment and are
    scored once by the same frozen sentence ranker.
    """

    def __init__(self, *, phonetic_checkpoint, phonetic_model=None, **kwargs):
        super().__init__(**kwargs)
        self.phonetic_checkpoint = Path(phonetic_checkpoint)
        self.phonetic_model = phonetic_model
        self._phonetic_model_loaded = phonetic_model is not None
        self.phonetic_artifacts = self._phonetic_artifact_identity()
        self.phonetic_parent_version = self.version
        self._stamp_phonetic_version()
        self.phonetic_word_trace = self._empty_trace()
        self.phonetic_word_ms = 0.0

    @staticmethod
    def _empty_trace():
        return {
            "production_pool_texts": [],
            "generation_requests": 0,
            "skipped_known_english": [],
            "skipped_generation_sources": [],
            "generation_budget_exhausted": False,
            "generated_extras": [],
            "scoreable_extras": [],
            "union": [],
            "public": [],
            "selected_extras": [],
            "missing_extras": [],
        }

    def _phonetic_artifact_identity(self):
        artifacts = {
            "hanglish/core/phonetic_word_union_converter.py": file_sha256(Path(__file__)),
            "hanglish/models/batched_phonetic_decode.py": file_sha256(_BATCHED_DECODER),
        }
        for path in (self.phonetic_checkpoint,
                     self.phonetic_checkpoint.with_suffix(".json")):
            if path.is_file():
                try:
                    name = str(path.relative_to(ROOT))
                except ValueError:
                    name = "external/" + path.name
                artifacts[name] = file_sha256(path)
        return artifacts

    def _stamp_phonetic_version(self):
        payload = json.dumps({
            "parent": self.phonetic_parent_version,
            "artifacts": self.phonetic_artifacts,
        }, sort_keys=True)
        self.version = "experimental-phonetic-word-union-1-" + hashlib.sha256(
            payload.encode()).hexdigest()[:12]

    def load_models(self):
        with self.lock:
            was_loaded = self.models_loaded
            super().load_models()
            if not was_loaded:
                self.phonetic_parent_version = self.version
            self.model_artifacts = {
                **getattr(self, "model_artifacts", {}),
                **self.phonetic_artifacts,
            }
            self._stamp_phonetic_version()

    def load_phonetic_model(self):
        with self.lock:
            if self._phonetic_model_loaded:
                return self.phonetic_model
            current = self._phonetic_artifact_identity()
            if current != self.phonetic_artifacts:
                raise ValueError("phonetic checkpoint or source identity changed")
            if not self.phonetic_checkpoint.is_file():
                raise ValueError("phonetic checkpoint missing")
            from hanglish.models.batched_phonetic_decode import generate_batched
            from hanglish.models.phonetic_transducer import PhoneticTransducer

            class BatchedPhoneticTransducer(PhoneticTransducer):
                def generate(self, roman, beam_size=4):
                    return generate_batched(self, roman, beam_size)

            self.phonetic_model = BatchedPhoneticTransducer.load(
                self.phonetic_checkpoint)
            self._phonetic_model_loaded = True
            return self.phonetic_model

    @staticmethod
    def _eligible_segment(source: str, segment: dict) -> bool:
        # Keep unknown/raw Roman spans eligible. Protected chunks remain safe
        # because eligibility requires the complete source slice to be letters.
        start, end = segment.get("input_start"), segment.get("input_end")
        text = segment.get("source")
        return (type(start) is int and type(end) is int and 0 <= start < end <= len(source)
                and text == source[start:end]
                and isinstance(text, str) and _ELIGIBLE.fullmatch(text) is not None)

    def _known_english(self, source: str) -> bool:
        english = getattr(getattr(self, "generator", None), "english", ())
        return source.lower() in english

    @staticmethod
    def _validate_path(source: str, text: str, segments: list[dict]) -> None:
        cursor = 0
        for segment in segments:
            start, end = segment.get("input_start"), segment.get("input_end")
            if (type(start) is not int or type(end) is not int
                    or start != cursor or not start < end <= len(source)
                    or segment.get("source") != source[start:end]
                    or not isinstance(segment.get("text"), str)):
                raise RuntimeError("phonetic union path does not preserve source offsets")
            cursor = end
        if cursor != len(source) or text != "".join(row["text"] for row in segments):
            raise RuntimeError("phonetic union path text differs from its segments")

    @staticmethod
    def _validate_outputs(source: str, outputs) -> list[str]:
        if not isinstance(outputs, (list, tuple)) or len(outputs) > _MAX_BEAM:
            raise RuntimeError("phonetic model returned an invalid beam")
        result, seen, previous = [], set(), float("inf")
        for item in outputs:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise RuntimeError("phonetic model returned an invalid beam item")
            text, score = item
            if (not isinstance(text, str) or not text
                    or any(not "가" <= char <= "힣" for char in text)
                    or text in seen or romanize_word(text) != source
                    or isinstance(score, bool) or not isinstance(score, (int, float))
                    or not math.isfinite(score) or score > previous):
                raise RuntimeError("phonetic model returned an invalid candidate")
            result.append(text)
            seen.add(text)
            previous = float(score)
        return result

    def _extend_sentence_paths(self, text, context, paths):
        parent = super()._extend_sentence_paths(text, context, paths)
        parent_texts = [path[0] for path in parent]
        if len(parent_texts) != len(set(parent_texts)):
            raise RuntimeError("phonetic union parent pool contains duplicate text")
        self.phonetic_word_trace["production_pool_texts"] = list(parent_texts)
        for output, score, segments in parent:
            if (isinstance(score, bool) or not isinstance(score, (int, float))
                    or not math.isfinite(score)):
                raise RuntimeError("phonetic union parent score is invalid")
            self._validate_path(text, output, segments)

        eligible = {}
        for _, _, segments in parent:
            for index, segment in enumerate(segments):
                if self._eligible_segment(text, segment):
                    key = (segment["input_start"], segment["input_end"], segment["source"])
                    if self._known_english(segment["source"]):
                        skipped = self.phonetic_word_trace["skipped_known_english"]
                        audit = {"source_start": key[0], "source_end": key[1],
                                 "source": key[2]}
                        if audit not in skipped:
                            skipped.append(audit)
                        continue
                    eligible.setdefault(key, (index, segment["source"]))
        if not eligible:
            self.phonetic_word_trace["union"] = list(parent_texts)
            return parent

        started = time.perf_counter()
        generated = {}
        generated_by_source = {}
        model = self.load_phonetic_model()
        for key in eligible:
            source_word = key[2]
            if source_word not in generated_by_source:
                if len(generated_by_source) == _MAX_EXTRAS:
                    self.phonetic_word_trace["generation_budget_exhausted"] = True
                    self.phonetic_word_trace["skipped_generation_sources"].append({
                        "source_start": key[0], "source_end": key[1],
                        "source": source_word,
                    })
                    continue
                self.phonetic_word_trace["generation_requests"] += 1
                with self.lock:
                    outputs = model.generate(source_word, beam_size=_MAX_BEAM)
                generated_by_source[source_word] = self._validate_outputs(
                    source_word, outputs)
            if source_word in generated_by_source:
                generated[key] = generated_by_source[source_word]
        self.phonetic_word_ms = (time.perf_counter() - started) * 1000

        seen = set(parent_texts)
        extras = []
        extra_audit = []
        for parent_index, (_, _, segments) in enumerate(parent):
            for segment_index, segment in enumerate(segments):
                key = (segment.get("input_start"), segment.get("input_end"),
                       segment.get("source"))
                for replacement in generated.get(key, ()):
                    replacement_segments = deepcopy(segments)
                    replacement_segments[segment_index]["text"] = replacement
                    replacement_segments[segment_index]["kind"] = "phonetic_word_union"
                    output = "".join(row["text"] for row in replacement_segments)
                    self._validate_path(text, output, replacement_segments)
                    if output in seen:
                        continue
                    seen.add(output)
                    extras.append((output, replacement_segments))
                    extra_audit.append({
                        "text": output,
                        "source_start": key[0], "source_end": key[1],
                        "source": key[2], "replacement": replacement,
                        "parent_index": parent_index,
                    })
                    if len(extras) == _MAX_EXTRAS:
                        break
                if len(extras) == _MAX_EXTRAS:
                    break
            if len(extras) == _MAX_EXTRAS:
                break
        self.phonetic_word_trace["generated_extras"] = extra_audit
        accepted = [row for row in extras if self.sentence_ranker.can_score(row[0])]
        self.phonetic_word_trace["scoreable_extras"] = [row[0] for row in accepted]
        if not accepted:
            self.phonetic_word_trace["union"] = list(parent_texts)
            return parent
        scores = self.sentence_ranker.score_pairs(
            [context] * len(accepted), [row[0] for row in accepted])
        if (len(scores) != len(accepted)
                or any(isinstance(score, bool) or not isinstance(score, (int, float))
                       or not math.isfinite(score) for score in scores)):
            raise RuntimeError("phonetic union ranker returned invalid scores")
        additional = [(row[0], float(score), row[1])
                      for row, score in zip(accepted, scores)]
        union = sorted([*parent, *additional], key=lambda path: (-path[1], path[0]))
        self.phonetic_word_trace["union"] = [path[0] for path in union]
        return union

    def convert(self, text, context="", engine="neural", mode="batch", raw=False):
        with self.lock:
            self.phonetic_word_trace = self._empty_trace()
            self.phonetic_word_ms = 0.0
            response = super().convert(text, context, engine, mode, raw)
            public = [row["text"] for row in response.get("candidates", [])]
            generated = [row["text"] for row in self.phonetic_word_trace["generated_extras"]]
            self.phonetic_word_trace["public"] = public
            self.phonetic_word_trace["selected_extras"] = [row for row in generated if row in public]
            self.phonetic_word_trace["missing_extras"] = [row for row in generated if row not in public]
            response["phonetic_word_union"] = {
                "experimental": True,
                "generation_ms": round(self.phonetic_word_ms, 3),
                "parent_version": self.phonetic_parent_version,
                "artifacts": dict(self.phonetic_artifacts),
                **deepcopy(self.phonetic_word_trace),
            }
            return response


__all__ = ["PhoneticWordUnionConverter"]
