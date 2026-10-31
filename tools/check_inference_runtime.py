#!/usr/bin/env python3
"""Server CPU import/API/tokenizer checks; no model generation or optimization."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference-raw", type=Path)
    args = p.parse_args()
    if sys.platform != "linux" or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        p.error("Run on a server with CUDA_VISIBLE_DEVICES='' for CPU preflight")
    from elm_pipeline.io import config, read, sha, write
    from elm_pipeline.backends import validate_version, engine_options
    cfg = config(args.config)
    backend = validate_version(cfg, importlib.metadata.version("vllm"))
    import torch
    from safetensors import safe_open
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.engine.arg_utils import EngineArgs
    from elm_pipeline import tasks
    from elm_pipeline.training import oracle_prefix, tensor_state
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU preflight unexpectedly initialized CUDA")
    options = engine_options(cfg, .35)
    EngineArgs(**options)
    SamplingParams(temperature=0, max_tokens=384, seed=cfg.get("seed", 1014))
    requirements = read(REPO/"metadata/asset_requirements.json")["files"]
    model = Path(cfg["model"])
    checked = 0
    for name, info in requirements.items():
        if info["role"] == "model":
            if sha(model/Path(name).name) != info["sha256"]:
                raise ValueError("Base model identity mismatch: " + Path(name).name)
            checked += 1
    assert checked == 10
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    weight_map = read(model/"model.safetensors.index.json")["weight_map"]
    key = "model.embed_tokens.weight"
    with safe_open(str(model/weight_map[key]), framework="pt", device="cpu") as f:
        emb = torch.nn.Embedding.from_pretrained(f.get_tensor(key), freeze=True)
    records = read(Path(cfg["features"])/"val_index.json")["records"]
    if not records:
        raise ValueError("Empty Val")
    teacher_checks = []
    for record in records[:2]:
        for task in ["label", "feature:"+tasks.FIELDS[0], "report"]:
            item = tasks.example(record, task)
            text = tok.apply_chat_template(tasks.messages(item, teacher=True), tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False)
            ids = tok(text, add_special_tokens=False)["input_ids"]
            actual = tasks.render(tok, emb, item, teacher=True)
            if not torch.equal(actual, emb(torch.tensor(ids))):
                raise ValueError("Text teacher native IDs and embedding path differ")
            teacher_checks.append(len(ids))
    matched = 0
    if args.reference_raw:
        reference = {}
        with args.reference_raw.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["source"] == "oracle_positive_control":
                    if row["id"] in reference:
                        raise ValueError("Duplicate reference query")
                    reference[row["id"]] = row
        oracle = tensor_state(cfg["oracle"])
        for record in records:
            prefix = oracle_prefix([record], oracle, "cpu")
            for task in ["label"] + ["feature:"+f for f in tasks.FIELDS] + ["report"]:
                item = tasks.example(record, task)
                value = tasks.render(tok, emb, item, prefix=prefix).bfloat16()
                digest = hashlib.sha256(value.float().numpy().tobytes()).hexdigest()
                old = reference[item["id"]]
                if (old["prompt_sha256"] != digest or old["question"] != item["question"]
                        or old["expected"] != item["expected"]):
                    raise ValueError("Frozen reference input changed: " + item["id"])
                matched += 1
        if matched != len(reference):
            raise ValueError("Reference coverage differs from Val")
    assert not torch.cuda.is_initialized()
    result = {"status": "passed_CPU_inference_preflight", "profile": backend,
              "python": sys.version, "torch_CUDA_build": torch.version.cuda,
              "packages": {name: importlib.metadata.version(name) for name in
                           ["vllm", "torch", "transformers", "numpy", "safetensors"]},
              "model_files_full_SHA_verified": checked, "teacher_native_embedding_cases": len(teacher_checks),
              "reference_Oracle_prompts_bitwise_equal": matched,
              "engine_arguments_constructed": True, "CUDA_initialized": False,
              "GPU_generation_verified": False, "additional_optimizer_steps": 0}
    write(args.output, result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
