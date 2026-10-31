#!/usr/bin/env python3
"""Bounded server software acceptance: feature equivalence → 30+5 updates → AR/report.

This is an engineering smoke test, not a scientific comparison or full-data run.
All paths come from a private config; no assets are downloaded or published.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from elm_pipeline.io import config, read, sha, write


def compare_features(cfg):
    import numpy as np
    import torch
    from elm_pipeline.data import FeatureDataset
    from elm_pipeline.tasks import FIELDS
    results = []
    for split in ("train", "val"):
        current = FeatureDataset(cfg["features"], split)
        old_root = Path(cfg["reference_label_features"])
        old_records = read(old_root/(split+"_records.json"))
        lookup = {r["id"]: i for i, r in enumerate(old_records)}
        label = torch.load(old_root/(split+".pt"), map_location="cpu", weights_only=False, mmap=True)
        stats = torch.load(Path(cfg["reference_statistics_features"])/(split+".pt"),
                           map_location="cpu", weights_only=False, mmap=True)
        for i, r in enumerate(current.records):
            j = lookup[r["id"]]
            bb, vv, ss, sv = current.inputs(i, "cpu")
            mask_equal = torch.equal(vv[0], label["valid"][j]) and torch.equal(sv[0], stats["valid"][j])
            measures = {}
            for name, actual, expected in (("label", bb[0], label["bank"][j]),
                                            ("statistics", ss[0], stats["bank"][j])):
                a, b = actual.float().flatten(), expected.float().flatten()
                relative_rms = float((a-b).square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-8))
                cosine = float(torch.nn.functional.cosine_similarity(a[None], b[None]))
                measures[name] = {"relative_rms": relative_rms, "cosine": cosine,
                                  "max_abs_error": float((a-b).abs().max())}
            feature_equal = np.allclose([r["features"][f] for f in FIELDS],
                                        [old_records[j]["features"][f] for f in FIELDS],
                                        rtol=1e-4, atol=1e-6)
            bins_equal = r["bins"] == old_records[j]["bins"]
            results.append({"id": r["id"], "dataset": r["dataset"], "split": split,
                            "mask_equal": mask_equal, "numeric_features_equal": bool(feature_equal),
                            "bins_equal": bins_equal, **measures})
    passed = all(r["mask_equal"] and r["numeric_features_equal"] and r["bins_equal"]
                 and all(r[k]["relative_rms"] < 0.02 and r[k]["cosine"] > 0.999
                         for k in ("label", "statistics")) for r in results)
    result = {"status": "passed" if passed else "mismatch", "records": results,
              "scope": "bounded selection, native preprocessing and frozen Encoder reconstruction",
              "tolerance": "masks/bins exact; numeric 1e-4 rtol/1e-6 atol; bank relRMS<.02 and cosine>.999",
              "raw_adapter_coverage_complete": False}
    write(Path(cfg["output"])/"feature_equivalence.json", result)
    if not passed:
        raise AssertionError("Fresh Encoder output differs from reference cache; training not dispatched")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--compare-features", action="store_true")
    args = p.parse_args()
    cfg = config(args.config)
    if args.compare_features:
        compare_features(cfg); return
    if sys.platform != "linux" or not cfg.get("software_debug") or cfg["train_steps"] != 35:
        raise ValueError("Acceptance requires the bounded Linux software-debug 35-step profile")
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    log_dir = out/"coordinator_logs"; log_dir.mkdir(exist_ok=True)
    with (out/"coordinator.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out/"acceptance_dispatch.json").exists():
            raise FileExistsError("Acceptance already dispatched; preserve it")
        write(out/"acceptance_dispatch.json", {"pid": os.getpid(), "time": time.time(),
              "config_sha256": sha(args.config), "source_sha256": sha(__file__),
              "software_debug_only": True})
        commands = [
            ("configuration", [cfg["train_python"], "-B", "-m", "elm_pipeline.cli", "--config", str(args.config.resolve()), "check"]),
            ("runtime_guards", [cfg["train_python"], "-B", str(REPO/"tools/runtime_fixture.py"), "--root", str(out/"guard_fixture")]),
            ("features", [cfg["train_python"], "-B", "-m", "elm_pipeline.cli", "--config", str(args.config.resolve()), "features"]),
            ("feature_equivalence", [cfg["train_python"], "-B", str(Path(__file__).resolve()), "--config", str(args.config.resolve()), "--compare-features"]),
            ("train30", [cfg["train_python"], "-B", "-m", "elm_pipeline.cli", "--config", str(args.config.resolve()), "train", "--stop-at", "30"]),
            ("resume35", [cfg["train_python"], "-B", "-m", "elm_pipeline.cli", "--config", str(args.config.resolve()), "train", "--stop-at", "35"]),
            ("autoregressive", [cfg["train_python"], "-B", "-m", "elm_pipeline.cli", "--config", str(args.config.resolve()), "evaluate", "--tag", "smoke35"]),
            ("report", [cfg["train_python"], "-B", "-m", "elm_pipeline.cli", "--config", str(args.config.resolve()), "report"]),
        ]
        completed = []
        try:
            for stage, command in commands:
                write(out/"acceptance_status.json", {"stage": stage, "time": time.time(), "completed": completed})
                with (log_dir/(stage+".log")).open("a") as log:
                    process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
                    write(out/"acceptance_child.json", {"stage": stage, "pid": process.pid,
                          "command": command, "time": time.time()})
                    code = process.wait()
                if code:
                    raise RuntimeError(stage+" exit "+str(code)+"; preserved log "+str(log_dir/(stage+".log")))
                completed.append(stage)
            from elm_pipeline.runtime import final_guard
            final_guard(out)
            result = {"status": "passed_software_acceptance", "completed": completed,
                      "committed_steps": 35, "resume_from_step": 30,
                      "raw_to_Native200_adapters_verified": False,
                      "full_data_training_completed": False,
                      "H20_verified": False, "process_exit_check_pending": False,
                      "source_sha256": sha(__file__), "config_sha256": sha(args.config), "time": time.time()}
            write(out/"acceptance_result.json", result)
            write(out/"acceptance_status.json", result)
            print(json.dumps(result), flush=True)
        except BaseException:
            write(out/"acceptance_error.json", {"time": time.time(), "completed": completed,
                  "traceback": traceback.format_exc(), "logs_preserved": True})
            raise


if __name__ == "__main__":
    main()
