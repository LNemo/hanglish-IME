"""TRAIN-gated local MLX backend for the pairwise fresh execution core.

The caller still owns exact fresh input provenance, an external process deadline
and the terminal execution verifier. Constructing this backend is not deployment
approval. No package/model download or optimization step is provided here.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import time

from scripts import prepare_difference_pairwise_lora_fresh as bridge
from scripts.difference_pairwise_lora_evaluation import canonical
from scripts.difference_pairwise_lora_inference import prepare_inference_selection, qualify_inference_tokens
from scripts.prepare_difference_pairwise_lora_data import MODEL_DIR, CHAT_TEMPLATE_SHA256, tokenize_pair

LIMITS = {"rss_bytes": 6 * 1024**3, "mlx_peak_bytes": 6 * 1024**3,
          "cache_bytes": 256 * 1024**2, "whole_seconds": 21600.0,
          "minimum_system_free_percent": 10.0, "maximum_swap_growth_bytes": 512 * 1024**2}
VERSIONS = {"mlx": "0.31.2", "mlx-lm": "0.31.3", "transformers": "5.12.1"}


def _system_resources():
    from scripts import probe_long_context_lora_resources as system
    return system.system_resources()


def qualify_rows(rows, tokenizer):
    """Build every pair before inference, rejecting overflow without row skips."""
    if not isinstance(rows, (list, tuple)) or len(rows) != 888:
        raise ValueError("exactly 888 fresh input rows required")
    seen, groups, prompts = set(), set(), {}
    for index, row in enumerate(rows):
        role = "fresh_natural" if index < 512 else "challenge_preservation"
        if row.get("role") != role or row.get("id") in seen:
            raise ValueError("fresh source identity or role order differs")
        selection = prepare_inference_selection(role, row)
        seen.add(row["id"])
        if index < 512:
            groups.add(row["group_id"])
        qualified = qualify_inference_tokens(tokenizer, selection)
        if (not isinstance(qualified, list) or len(qualified) != len(selection["pairs"])
                or any(not isinstance(q, dict) for q in qualified)
                or [q.get("pair_id") for q in qualified] != [p["pair_id"] for p in selection["pairs"]]):
            raise ValueError("token qualification omitted or reordered a permitted pair")
        for pair, record in zip(selection["pairs"], qualified):
            if record["pair_id"] in prompts:
                raise ValueError("duplicate qualified pair identity")
            prompts[record["pair_id"]] = {**record,
                "pair_sha256": hashlib.sha256(canonical(pair)).hexdigest()}
    if len(groups) != 387:
        raise ValueError("fresh document group count differs")
    return prompts


def load_verified_runtime():
    """Replay terminal TRAIN acceptance before any tokenizer or MLX import."""
    training = bridge.require_training_proof()
    proof = json.loads(bridge.TRAIN_PROOF.read_bytes())["payload"]
    if bridge.sha(bridge.TRAIN_PROOF) != training["sha256"]:
        raise ValueError("TRAIN proof changed before runtime construction")
    if {name: importlib.metadata.version(name) for name in VERSIONS} != VERSIONS:
        raise ValueError("MLX inference package versions differ")
    root = bridge.ROOT
    output = root / proof["sole_output"]
    report_path = output / "report.json"
    if bridge.sha(report_path) != proof["artifacts"]["report.json"]:
        raise ValueError("TRAIN report binding differs")
    report = json.loads(report_path.read_bytes())
    pins = dict(proof["source_hashes"])
    for path_key, hash_key in (("path", "sha256"), ("config_path", "config_sha256")):
        pins[report["adapter"][path_key]] = report["adapter"][hash_key]
    pins[str(bridge.TRAIN_PROOF.relative_to(root))] = training["sha256"]
    pins[str(report_path.relative_to(root))] = proof["artifacts"]["report.json"]
    bridge.verify_pins(pins)
    # Tokenization happens without loading the Qwen weights. Qualification of
    # the complete fresh cohort must finish before execute() calls score_pair.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True,
                                               trust_remote_code=False)
    if hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() != CHAT_TEMPLATE_SHA256:
        raise ValueError("frozen tokenizer chat template differs")
    return MLXRuntime(tokenizer, output / "adapter", report["final_trainable_tensor_hashes"], pins)


class MLXRuntime:
    """Backend callbacks; obtain through load_verified_runtime in real runs."""

    def __init__(self, tokenizer, adapter_dir, tensor_hashes, source_pins):
        self.tokenizer = tokenizer
        self.adapter_dir = adapter_dir
        self.tensor_hashes = tensor_hashes
        self.source_pins = source_pins
        self.started = time.monotonic()
        self.baseline = dict(_system_resources())
        self.mx = self.model = None
        self.state = "new"
        self.released = False
        self.resource_records = []
        self.model_loaded = False

    def _initialize_mlx(self):
        if self.mx is None:
            import mlx.core as mx
            self.mx = mx
            mx.set_memory_limit(LIMITS["mlx_peak_bytes"])
            mx.set_cache_limit(LIMITS["cache_bytes"])

    def checkpoint(self, stage):
        from scripts.long_context_lora_resources import observe_resources, ResourceObservationError
        from scripts import probe_long_context_lora_resources as system
        self._initialize_mlx()
        try:
            _, record = observe_resources(stage=stage, mx=self.mx, started=self.started,
                baseline=self.baseline, limits=LIMITS, system_reader=system.system_resources,
                rss_reader=system.rss_bytes)
        except ResourceObservationError as error:
            self.resource_records.append(error.record)
            raise
        self.resource_records.append(record)
        return {"passed": True, "observation": record, "baseline": self.baseline}

    def validate_sources(self, stage):
        bridge.verify_pins(self.source_pins)
        return {"passed": True, "stage": stage, "source_hashes": self.source_pins}

    def score_pair(self, pair, qualification):
        if self.state in ("failed", "released"):
            raise ValueError("failed or released inference runtime cannot restart")
        messages, tokens, digit_ids, rendered = tokenize_pair(self.tokenizer, pair)
        expected = {"pair_id": pair["pair_id"], "pair_sha256": hashlib.sha256(canonical(pair)).hexdigest(),
                    "messages": messages, "tokens": tokens, "prompt_tokens": len(tokens),
                    "digit_continuation_ids": digit_ids, "rendered_prefix": rendered}
        if (len(tokens) > 1024 or digit_ids != {"1": 16, "2": 17}
                or canonical(qualification) != canonical(expected)):
            raise ValueError("qualified model prompt does not replay exactly")
        from scripts import long_context_lora_model as model_tools
        try:
            if self.state == "new":
                self.state = "loading"
                self.validate_sources("before_model_load")
                self.checkpoint("before_model_load")
                from mlx_lm import load
                from mlx_lm.tuner.utils import load_adapters
                self.model, loaded_tokenizer = load(str(MODEL_DIR), tokenizer_config={"trust_remote_code": False})
                self.model_loaded = True
                if loaded_tokenizer.chat_template != self.tokenizer.chat_template:
                    raise ValueError("model-loaded tokenizer differs")
                self.model.freeze()
                load_adapters(self.model, str(self.adapter_dir))
                self.model.eval()
                actual = model_tools.tensor_hashes(model_tools.require_trainable_contract(self.model))
                if actual != self.tensor_hashes:
                    raise ValueError("loaded adapter tensors differ from verified training")
                self.checkpoint("adapter_loaded")
                self.state = "ready"
            if self.state != "ready" or self.model is None:
                raise ValueError("inference model is not ready")
            values = model_tools.score_choices(self.mx, self.model, qualification["tokens"], (16, 17))
            self.mx.eval(values)
            return values.tolist()
        except BaseException:
            self.state = "failed"
            raise

    def release_model(self):
        self.model = self.tokenizer = None
        self.state = "released"
        self.released = True
        gc.collect()
        if self.mx is not None:
            self.mx.clear_cache()
            self.checkpoint("model_released")
        return {"released": True, "model_loaded": self.model_loaded}
