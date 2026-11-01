#!/usr/bin/env python3
"""Write portable configs for the implemented Projector pilot and 6600 evaluation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re

REPO = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--gpu", type=int, required=True, help="One physical GPU explicitly assigned to this job")
    p.add_argument("--name", default="pilot35")
    p.add_argument("--mode", choices=("pilot35", "report6600"), default="pilot35")
    p.add_argument("--features", type=Path, help="Existing completed feature directory, for report6600")
    args = p.parse_args()
    if args.gpu < 0 or not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", args.name):
        p.error("Invalid GPU or output name")
    root = args.work.expanduser().resolve()
    ready = json.loads((root / "setup/assets_ready.json").read_text())
    if ready.get("status") != "assets_downloaded_and_checked":
        p.error("Run setup_server.sh and finish downloading the assets first")
    template = "pipeline_debug.example.json" if args.mode == "pilot35" else "pipeline_report6600.example.json"
    cfg = json.loads((REPO / "configs" / template).read_text())
    weights = Path(ready["elm_weights"])
    cfg.update(output=str(root / "runs" / args.name),
               features=str(args.features.expanduser().resolve() if args.features else root / "features" / args.name),
               tmpdir=str(root / "tmp" / args.name),
               manifest_root=ready["manifest_root"], data_root=ready["data_root"], path_map={},
               model=ready["model"], encoder_bundle=str(weights / "encoders"),
               oracle=str(weights / "oracle/oracle10200.safetensors"),
               schema=str(weights / "data_schema.json"), thresholds=str(weights / "thresholds.json"),
               train_python=str(root / "envs/train/bin/python"),
               ar_python=str(root / "envs/ar-v1/bin/python"), ar_backend="v1", gpus=[args.gpu],
               datasets=list(ready["dataset_counts"]), allow_known_legacy_overlap_for_debug=True)
    if args.mode == "report6600":
        cfg["evaluation_checkpoint"] = str(weights / "projector/projector_report6600.safetensors")
    path = root / "configs" / (args.name + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(cfg, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(path)
    print("Config created only; no training launched. Historical splits are for engineering/debug.")


if __name__ == "__main__":
    main()
