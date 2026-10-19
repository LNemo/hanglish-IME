import importlib.util
import io
import json
from pathlib import Path

FILE = Path(__file__).resolve().parents[1] / "native/macos/Packaging/engine_service.py"
spec = importlib.util.spec_from_file_location("engine_service", FILE)
service = importlib.util.module_from_spec(spec)
spec.loader.exec_module(service)


class Engine:
    def convert(self, text, **kwargs):
        if text == "crash":
            raise ValueError("sensitive typed payload")
        assert kwargs == dict(context="", engine="neural", mode="batch", raw=False)
        return {"result": "입니다", "candidates": [{"text": "입니다", "score": 10}], "version": "test"}


def run(lines):
    output = io.StringIO()
    service.serve(io.BytesIO(lines), output, Engine)
    return [json.loads(line) for line in output.getvalue().splitlines()]


def test_valid_request_reduced_response():
    response = run(b'{"input":"imnida","request_id":"1"}\n')[0]
    assert response == {"result": "입니다", "candidates": [{"text": "입니다"}], "version": "test", "request_id": "1"}


def test_invalid_and_exception_do_not_echo_input_and_next_request_survives():
    rows = run(b'{"input":"crash","request_id":"bad"}\nnot-json\n{"input":"imnida","request_id":"ok"}\n')
    assert rows[0] == {"error": "conversion_unavailable", "request_id": "bad"}
    assert rows[1] == {"error": "conversion_unavailable", "request_id": ""}
    assert rows[2]["result"] == "입니다"


def test_oversized_frame_is_drained_without_splitting_into_requests():
    rows = run(b"x" * (service.MAX_FRAME * 2) + b'\n{"input":"imnida","request_id":"ok"}\n')
    assert len(rows) == 2 and "error" in rows[0] and rows[1]["request_id"] == "ok"


def test_hints_and_invalid_types_rejected():
    for extra in [{"context": "힌트"}, {"input": 1}, {"raw": 1}, {"engine": "baseline"}, {"input": "a" * 2001}]:
        payload = {"input": "imnida", "request_id": "1", **extra}
        assert "error" in run((json.dumps(payload) + "\n").encode())[0]
