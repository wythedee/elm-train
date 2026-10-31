"""Explicit inference profiles. This module does not import a model or GPU library."""
from __future__ import annotations

VERSIONS = {"legacy_v0": "0.9.2", "v1": "0.32.0"}


def profile(cfg):
    name = cfg.get("ar_backend", "legacy_v0")
    if name not in VERSIONS:
        raise ValueError("Unknown ar_backend: " + str(name))
    return name


def validate_version(cfg, installed):
    name = profile(cfg)
    if installed.split("+", 1)[0] != VERSIONS[name]:
        raise ValueError(f"{name} requires vLLM {VERSIONS[name]}, found {installed}")
    return name


def configure_environment(env, cfg):
    if profile(cfg) == "legacy_v0":
        env["VLLM_USE_V1"] = "0"
    else:
        env.pop("VLLM_USE_V1", None)
    return env


def engine_options(cfg, utilization):
    result = dict(model=cfg["model"], dtype="bfloat16", enable_prompt_embeds=True,
                  max_model_len=4096, max_num_seqs=16,
                  gpu_memory_utilization=float(utilization), enforce_eager=True,
                  disable_log_stats=True, seed=cfg.get("seed", 1014),
                  num_gpu_blocks_override=512, block_size=16)
    if profile(cfg) == "v1":
        # Match the legacy diagnostic's no-prefix-cache, unchunked policy first.
        result.update(enable_prefix_caching=False, enable_chunked_prefill=False)
    return result
