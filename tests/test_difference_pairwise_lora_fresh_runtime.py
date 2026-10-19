import hashlib
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from scripts import difference_pairwise_lora_fresh_runtime as runtime
from scripts.difference_pairwise_lora_inference import prepare_inference_selection, qualify_inference_tokens
from scripts.difference_pairwise_lora_evaluation import canonical


@pytest.fixture(autouse=True)
def synthetic_system_baseline(monkeypatch):
    monkeypatch.setattr(runtime, "_system_resources", lambda: {"free_percent": 50.0, "swap_used_bytes": 0})


class Tokenizer:
    chat_template = "synthetic"

    def encode(self, text, add_special_tokens=False):
        return [7] * 20 + ([16] if text.endswith("1") else [17] if text.endswith("2") else [])

    def apply_chat_template(self, messages, *, tokenize, **kwargs):
        return self.encode("prefix") if tokenize else "prefix"


def row(index=0):
    candidates = [{"text": text, "score": 2.0-i, "segments": [{
        "input_start": 0, "input_end": 6, "source": "imnida", "text": text,
        "kind": "morphology"}]} for i,text in enumerate(("임니다", "입니다"))]
    return {"id": f"r{index}", "group_id": f"g{index % 387}" if index < 512 else f"p{index}",
            "input": "imnida", "context": "", "candidates": candidates,
            "raw_bypass_response": {"result": "imnida"},
            "role": "fresh_natural" if index < 512 else "challenge_preservation"}


def pair_and_prompt():
    value = row()
    prepared = prepare_inference_selection(value["role"], value)
    pair = prepared["pairs"][0]
    prompt = qualify_inference_tokens(Tokenizer(), prepared)[0]
    prompt["pair_sha256"] = hashlib.sha256(canonical(pair)).hexdigest()
    return pair, prompt


def test_whole_cohort_qualification_keeps_every_pair():
    rows = [row(i) for i in range(888)]
    prompts = runtime.qualify_rows(rows, Tokenizer())
    assert len(prompts) == 888
    for source in rows:
        pair = prepare_inference_selection(source["role"], source)["pairs"][0]
        assert prompts[pair["pair_id"]]["pair_sha256"] == hashlib.sha256(canonical(pair)).hexdigest()
    with pytest.raises(ValueError, match="888"):
        runtime.qualify_rows(rows[:-1], Tokenizer())
    rows[-1]["id"] = rows[-2]["id"]
    with pytest.raises(ValueError, match="identity"):
        runtime.qualify_rows(rows, Tokenizer())


def test_train_rejection_precedes_tokenizer_or_mlx_setup(monkeypatch):
    def reject():
        raise ValueError("TRAIN pending")

    monkeypatch.setattr(runtime.bridge, "require_training_proof", reject)
    monkeypatch.setattr(runtime.importlib.metadata, "version", lambda *_: pytest.fail("packages checked before TRAIN"))
    with pytest.raises(ValueError, match="TRAIN pending"):
        runtime.load_verified_runtime()


def test_qualification_under_return_rejects_whole_cohort(monkeypatch):
    rows = [row(i) for i in range(888)]
    rows[0]["candidates"].append({**rows[0]["candidates"][1], "score": 0.0})
    original = runtime.qualify_inference_tokens
    monkeypatch.setattr(runtime, "qualify_inference_tokens", lambda tokenizer, selection: original(tokenizer, selection)[:-1])
    with pytest.raises(ValueError, match="omitted or reordered"):
        runtime.qualify_rows(rows, Tokenizer())


def test_resource_baseline_is_captured_before_qualification(monkeypatch, tmp_path):
    observed = {"free_percent": 50.0, "swap_used_bytes": 12}
    monkeypatch.setattr(runtime, "_system_resources", lambda: observed)
    backend = runtime.MLXRuntime(Tokenizer(), tmp_path, {}, {})
    observed["swap_used_bytes"] = 500
    assert backend.baseline["swap_used_bytes"] == 12
    assert backend.mx is None and backend.model is None


def test_accepted_factory_binds_files_before_local_tokenizer_without_model(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr(runtime.bridge, "ROOT", tmp_path)
    proof_path = tmp_path / "proof.json"
    monkeypatch.setattr(runtime.bridge, "TRAIN_PROOF", proof_path)
    directory = tmp_path / "train" / "adapter"
    directory.mkdir(parents=True)
    adapter, config = directory / "adapters.safetensors", directory / "adapter_config.json"
    adapter.write_bytes(b"synthetic adapter"); config.write_text("{}")
    report = {"adapter": {"path": "train/adapter/adapters.safetensors", "sha256": runtime.bridge.sha(adapter),
        "config_path": "train/adapter/adapter_config.json", "config_sha256": runtime.bridge.sha(config)},
        "final_trainable_tensor_hashes": {"tensor": "synthetic"}}
    report_path = directory.parent / "report.json"
    report_path.write_text(json.dumps(report))
    proof_path.write_text(json.dumps({"payload": {"sole_output": "train", "source_hashes": {},
        "artifacts": {"report.json": runtime.bridge.sha(report_path)}}}))

    def verified():
        events.append("TRAIN_verified")
        return {"sha256": runtime.bridge.sha(proof_path)}

    monkeypatch.setattr(runtime.bridge, "require_training_proof", verified)
    monkeypatch.setattr(runtime.importlib.metadata, "version", lambda name: runtime.VERSIONS[name])
    module = ModuleType("transformers")

    def tokenizer(path, **kwargs):
        assert events == ["TRAIN_verified"]
        assert kwargs == {"local_files_only": True, "trust_remote_code": False}
        events.append("local_tokenizer")
        return Tokenizer()

    module.AutoTokenizer = SimpleNamespace(from_pretrained=tokenizer)
    monkeypatch.setitem(sys.modules, "transformers", module)
    monkeypatch.setattr(runtime, "CHAT_TEMPLATE_SHA256", hashlib.sha256(Tokenizer.chat_template.encode()).hexdigest())
    backend = runtime.load_verified_runtime()
    assert events == ["TRAIN_verified", "local_tokenizer"]
    assert backend.model is None and backend.mx is None
    assert backend.source_pins["train/adapter/adapters.safetensors"] == runtime.bridge.sha(adapter)
    adapter.write_bytes(b"changed after verification")
    with pytest.raises(ValueError, match="source binding differs"):
        backend.validate_sources("before_model_load")


def test_tampered_tokens_cannot_trigger_model_loading(monkeypatch, tmp_path):
    backend = runtime.MLXRuntime(Tokenizer(), tmp_path, {}, {})
    monkeypatch.setattr(backend, "validate_sources", lambda *_: pytest.fail("model loading reached"))
    pair, prompt = pair_and_prompt()
    prompt["tokens"][0] = 123
    with pytest.raises(ValueError, match="prompt does not replay"):
        backend.score_pair(pair, prompt)
    assert backend.model is None and backend.mx is None


@pytest.fixture
def fake_backend(monkeypatch, tmp_path):
    from scripts import long_context_lora_model as shared
    events = []
    model = SimpleNamespace(freeze=lambda: events.append("freeze"), eval=lambda: events.append("eval_mode"))
    mx = SimpleNamespace(eval=lambda _: events.append("materialize"), clear_cache=lambda: events.append("clear_cache"))
    package, tuner, utils = (ModuleType(name) for name in ("mlx_lm", "mlx_lm.tuner", "mlx_lm.tuner.utils"))
    package.__path__ = []; tuner.__path__ = []
    package.load = lambda *args, **kwargs: (events.append("load") or model, Tokenizer())
    utils.load_adapters = lambda *args: events.append("adapter")
    for module in (package, tuner, utils):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    backend = runtime.MLXRuntime(Tokenizer(), tmp_path, {"tensor": "digest"}, {})
    backend.mx = mx
    monkeypatch.setattr(backend, "checkpoint", lambda stage: events.append(stage) or {"passed": True})
    monkeypatch.setattr(backend, "validate_sources", lambda stage: events.append("sources") or {"passed": True})
    monkeypatch.setattr(shared, "require_trainable_contract", lambda model: {"tensor": "array"})
    monkeypatch.setattr(shared, "tensor_hashes", lambda tensors: {"tensor": "digest"})
    monkeypatch.setattr(shared, "score_choices", lambda *args: SimpleNamespace(tolist=lambda: [1.0, 2.0]))
    return backend, events, shared


def test_native_backend_loads_once_checks_adapter_then_releases(fake_backend):
    backend, events, _ = fake_backend
    pair, prompt = pair_and_prompt()
    assert backend.score_pair(pair, prompt) == [1.0, 2.0]
    assert backend.score_pair(pair, prompt) == [1.0, 2.0]
    assert events.count("load") == 1
    assert events.index("adapter") < events.index("materialize")
    assert backend.release_model() == {"released": True, "model_loaded": True}
    assert backend.model is None and backend.tokenizer is None
    assert events[-2:] == ["clear_cache", "model_released"]
    with pytest.raises(ValueError, match="cannot restart"):
        backend.score_pair(pair, prompt)


def test_wrong_adapter_is_rejected_before_any_inference(fake_backend, monkeypatch):
    backend, events, shared = fake_backend
    monkeypatch.setattr(shared, "tensor_hashes", lambda _: {"tensor": "wrong"})
    with pytest.raises(ValueError, match="adapter tensors differ"):
        backend.score_pair(*pair_and_prompt())
    assert "materialize" not in events
    assert backend.model_loaded is True
    monkeypatch.setattr(shared, "tensor_hashes", lambda _: {"tensor": "digest"})
    with pytest.raises(ValueError, match="cannot restart"):
        backend.score_pair(*pair_and_prompt())
    assert events.count("load") == 1 and "materialize" not in events
    assert backend.release_model()["released"] is True
