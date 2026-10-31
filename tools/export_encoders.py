#!/usr/bin/env python3
"""Export trusted historical Encoder checkpoints as inference-only safe tensors.

Run on a server. This never trains or changes the supplied checkpoints. Pickle
checkpoints must come from a trusted source. Classification heads and optimizers
are not needed by the feature extractor and are omitted.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--class-checkpoint", type=Path, required=True)
    p.add_argument("--statistics-checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or args.output.is_symlink():
        p.error("Output must be a new directory")
    args.output.parent.resolve(strict=True)
    if shutil.disk_usage(args.output.parent).free < 12 * 1024**3:
        p.error("Need at least 12 GiB free for safe checkpoint conversion")
    args.output.mkdir()
    import torch
    from safetensors.torch import save_file
    source = args.class_checkpoint
    if source.suffix == ".zst":
        tool = shutil.which("zstd")
        if not tool:
            p.error("Existing zstd executable required; this tool never installs it")
        source = args.output / "original_class_checkpoint.pt"
        with source.open("xb") as stream:
            subprocess.run([tool, "-dc", str(args.class_checkpoint)], stdout=stream, check=True)
    blob = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
    state = blob.get("model", blob.get("state_dict", blob))
    weights = {k.removeprefix("encoder."): v.detach().float().contiguous()
               for k, v in state.items() if k.startswith("encoder.")}
    required = ("experts.", "router.", "token_compressor.", "lowdim_adapters.")
    if not weights or any(not k.startswith(required) for k in weights):
        raise ValueError("Unexpected class Encoder checkpoint layout")
    num_experts = max(int(k.split(".")[1]) for k in weights if k.startswith("experts.")) + 1
    config = {
        "experts": num_experts,
        "router_patches": weights["router.1.weight"].shape[1] // (75 * 2),
        "router_hidden": weights["router.1.weight"].shape[0],
        "router_top_k": 3, "adapter_dim": weights["lowdim_adapters.0.down.weight"].shape[0],
        "temperature": 1.0,
    }
    class_target = args.output / "encoder_class.safetensors"
    save_file(weights, str(class_target))
    class_count = sum(v.numel() for v in weights.values())
    del blob, state, weights
    stats = torch.load(args.statistics_checkpoint, map_location="cpu", weights_only=False, mmap=True)
    state = stats.get("state_dict", stats)
    weights = {k: v.detach().float().contiguous() for k, v in state.items()
               if k.startswith(("backbone.", "compressor.", "adapter."))}
    if not all(any(k.startswith(prefix) for k in weights)
               for prefix in ("backbone.", "compressor.", "adapter.")):
        raise ValueError("Statistics Encoder components missing")
    stats_target = args.output / "encoder_statistics.safetensors"
    save_file(weights, str(stats_target))
    result = {
        "status": "exported_not_yet_feature_validated", "encoder_config": config,
        "class_parameters": class_count,
        "statistics_parameters": sum(v.numel() for v in weights.values()),
        "files": {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)}
                  for p in (class_target, stats_target)},
        "source_class_sha256": sha(args.class_checkpoint),
        "source_statistics_sha256": sha(args.statistics_checkpoint),
        "heads_and_optimizer_exported": False,
        "training_started": False,
        "compatibility": "Fixed raw-stat Top3 router, 75 canonical channels, two anchors, 50-dimensional compression.",
    }
    (args.output / "encoder_config.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
