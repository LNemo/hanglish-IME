"""Private stdio transport for the bundled IME. Never opens a network listener."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

MAX_FRAME = 32768


def validate_request(payload):
    if not isinstance(payload, dict):
        raise ValueError("request must be an object")
    request_id = payload.get("request_id")
    text = payload.get("input")
    if not isinstance(request_id, str) or not 1 <= len(request_id) <= 100:
        raise ValueError("invalid request id")
    if not isinstance(text, str) or not 1 <= len(text) <= 2000:
        raise ValueError("invalid input length")
    if payload.get("context", "") != "" or payload.get("engine", "neural") != "neural":
        raise ValueError("unsupported context or engine")
    if payload.get("mode", "batch") != "batch" or payload.get("raw", False) is not False:
        raise ValueError("unsupported conversion mode")
    return request_id, text


def serve(stream, output, make_engine):
    engine = None
    while True:
        line = stream.readline(MAX_FRAME + 1)
        if not line:
            break
        request_id = ""
        try:
            if len(line) > MAX_FRAME:
                while line and not line.endswith(b"\n"):
                    line = stream.readline(MAX_FRAME + 1)
                raise ValueError("request too large")
            payload = json.loads(line)
            if isinstance(payload, dict) and isinstance(payload.get("request_id"), str):
                request_id = payload["request_id"][:100]
            request_id, text = validate_request(payload)
            if engine is None:
                engine = make_engine()
            result = engine.convert(text, context="", engine="neural", mode="batch", raw=False)
            response = {key: result[key] for key in ("result", "candidates", "version")}
            response["candidates"] = [{"text": candidate["text"]} for candidate in result["candidates"]]
            response["request_id"] = request_id
        except Exception:
            # Avoid exception messages containing typed text, paths or model internals.
            response = {"request_id": request_id, "error": "conversion_unavailable"}
        try:
            output.write(json.dumps(response, ensure_ascii=False) + "\n")
            output.flush()
        except (BrokenPipeError, OSError):
            break


def main():
    # Keep the protocol on its own fd; native-library stdout/stderr cannot corrupt it.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8")
    with open(os.devnull, "w") as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
        sys.stdout = sink
        sys.stderr = sink
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                          TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="2",
                          MKL_NUM_THREADS="2")
        root = Path(__file__).resolve().parent
        sys.path.insert(0, str(root))

        def make_engine():
            import torch
            torch.set_num_threads(2)
            torch.set_num_interop_threads(1)
            from hanglish.core.workbench import WorkbenchEngine
            return WorkbenchEngine()

        serve(sys.stdin.buffer, protocol, make_engine)
    protocol.close()


if __name__ == "__main__":
    main()
