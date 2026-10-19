"""Isolated two-pass runtime for provenance-filtered boundary proposals."""
from __future__ import annotations

from copy import deepcopy
import time

from scripts.boundary_proposal_policy import boundary_proposal_allowed
from scripts.lexical_boundary_runtime import BoundaryCompositionRuntime

_MISSING = object()


def _path_record(path):
    return {
        "text": path[0],
        "score": path[1],
        "segments": [dict(segment) for segment in path[2]],
    }


class ProvenanceBoundaryRuntime(BoundaryCompositionRuntime):
    """Anchor new boundary paths to the exact production first candidate."""

    def _numeric_sentence_extras(self, pieces, static_pieces, complete):
        # The first pass disables the boundary overlay and must remain the
        # exact one-pass parent, including its existing numeric extras.
        if not getattr(self, "boundary_overlay_enabled", True):
            return super()._numeric_sentence_extras(
                pieces, static_pieces, complete
            )

        combined = list(super()._numeric_sentence_extras(
            pieces, static_pieces, complete
        ))
        trace = self.boundary_trace
        proposed_records = trace["accepted_new_extras"]
        if not proposed_records:
            return combined
        anchor = getattr(self, "_boundary_parent_anchor", None)
        if not isinstance(anchor, dict):
            raise ValueError("boundary provenance filter requires a parent anchor")

        records_by_text = {}
        for record in proposed_records:
            text = record.get("text") if isinstance(record, dict) else None
            if not isinstance(text, str) or text in records_by_text:
                raise ValueError("boundary trace contains malformed new paths")
            records_by_text[text] = record

        preserved = []
        accepted_paths = []
        accepted_records = []
        observed_new = set()
        for path in combined:
            record = records_by_text.get(path[0])
            if record is None:
                # Existing numeric extras are outside this policy and remain
                # exactly as returned by the parent runtime.
                preserved.append(path)
                continue
            if _path_record(path) != record:
                raise ValueError("boundary trace differs from returned new path")
            observed_new.add(path[0])
            if boundary_proposal_allowed(
                    self._boundary_current_text,
                    anchor,
                    record,
                    trace["derived_options"]):
                accepted_paths.append(path)
                accepted_records.append(record)
        if observed_new != set(records_by_text):
            raise ValueError("boundary trace does not match returned new paths")

        trace["accepted_new_extras"] = accepted_records
        trace["activated"] = bool(accepted_records)
        if accepted_records:
            trace["reason"] = "accepted_new_extras"
        else:
            trace["reason"] = "provenance_policy_rejected"
        return [*preserved, *accepted_paths]

    def convert(self, text, context="", engine="neural", mode="batch", raw=False):
        started = time.perf_counter()
        with self.lock:
            eligible = (
                engine == "neural"
                and not raw
                and getattr(self, "boundary_overlay_enabled", True)
            )
            if not eligible:
                return super().convert(text, context, engine, mode, raw)
            original_enabled = self.boundary_overlay_enabled
            original_anchor = getattr(self, "_boundary_parent_anchor", _MISSING)
            try:
                self.boundary_overlay_enabled = False
                parent = super().convert(text, context, engine, mode, raw)
                self.boundary_overlay_enabled = original_enabled
                if not parent.get("sentence_ranking_applied"):
                    return parent
                candidates = parent.get("candidates")
                if not isinstance(candidates, list) or not candidates:
                    raise ValueError("parent conversion has no first candidate")
                self._boundary_parent_anchor = deepcopy(candidates[0])
                child = super().convert(text, context, engine, mode, raw)
                child["engine_ms"] = round(
                    (time.perf_counter() - started) * 1000, 3
                )
                return child
            finally:
                self.boundary_overlay_enabled = original_enabled
                if original_anchor is _MISSING:
                    if hasattr(self, "_boundary_parent_anchor"):
                        del self._boundary_parent_anchor
                else:
                    self._boundary_parent_anchor = original_anchor
