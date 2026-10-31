"""Explicit staged entry point. No downloading, package installation or auto-training."""
from __future__ import annotations
import argparse
import json
import importlib.util
from pathlib import Path
import re
import sys
import traceback
from .io import config, read, resolve_data_path, sha, write
from . import tasks
from .backends import profile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("stage", choices=["check", "features", "train", "evaluate", "report"])
    p.add_argument("--stop-at", type=int)
    p.add_argument("--tag", default="validation")
    args = p.parse_args()
    cfg = config(args.config)
    profile(cfg)
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.tag):
        p.error("Evaluation tag must contain only letters, digits, underscores or hyphens")
    if cfg.get("software_debug") and cfg.get("train_steps", 0) > 50:
        p.error("Software-debug profile permits at most 50 optimizer steps")
    if not cfg.get("gpus") or len(set(cfg["gpus"])) != len(cfg["gpus"]):
        p.error("Specify the GPU whitelist explicitly")
    if any(type(i) is not int or i < 0 for i in cfg["gpus"]):
        p.error("Invalid GPU whitelist")
    if args.stage == "check":
        for key in ("manifest_root", "encoder_bundle", "model", "oracle", "schema"):
            if not Path(cfg[key]).exists():
                raise FileNotFoundError(key + ": " + cfg[key])
        selection = read(cfg["debug_selection"]) if cfg.get("debug_selection") else None
        checked = set()
        for split in cfg.get("splits", ["train", "val"]):
            for dataset in cfg["datasets"]:
                wanted = set(selection[split][dataset]) if selection else None
                found = set()
                file = Path(cfg["manifest_root"])/dataset/(split+".jsonl")
                for line in file.open():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if wanted is not None and row["id"] not in wanted:
                        continue
                    found.add(row["id"])
                    value = row.get("lmdb_path") or row.get("h5_path")
                    if not value:
                        raise ValueError("Record has no storage path")
                    if value not in checked:
                        resolve_data_path(value, cfg.get("path_map"), cfg.get("data_root"))
                        checked.add(value)
                if wanted is not None and found != wanted:
                    raise ValueError("Selected IDs missing from manifest")
        print(json.dumps({"status": "configuration_paths_and_task_fixture_passed",
                          "task_fixture": tasks.no_leak_fixture(),
                          "unique_input_paths_verified": len(checked),
                          "GPU_used": False, "training_started": False}))
        return
    if args.stage == "report":
        from .runtime import final_guard
        from .evaluation import report
        final_guard(cfg["output"])
        result = report(cfg, sha(args.config))
        final_guard(cfg["output"])
        result["process_exit_check_pending"] = False
        write(Path(cfg["output"])/"results_summary.json", result)
        write(Path(cfg["output"])/"final_verification.json", {
            "status": "passed_for_available_artifacts", "process_exit_check_pending": False,
            "raw_parser_rescored": True, "role_GPU_locks_free": True,
            "all_predeclared_stages_complete": False})
        print(json.dumps(result)); return
    if not args.worker:
        from .runtime import run_stage
        if args.stage == "features":
            code = Path(__file__).resolve().parents[1]/"tools/validate_data.py"
            spec = importlib.util.spec_from_file_location("elm_data_identity_audit", code)
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            audit = module.audit(cfg["manifest_root"], cfg["datasets"])
            root = Path(cfg["output"]); root.mkdir(parents=True, exist_ok=True)
            write(root/"input_identity_audit.json", audit)
            if audit["status"] != "passed_manifest_checks":
                known_legacy_only = (cfg.get("software_debug") and cfg.get("allow_known_legacy_overlap_for_debug")
                                     and audit["invalid_rows"] == 0
                                     and audit["critical_violations"] == audit["cross_split_subject_overlap_count"])
                if not known_legacy_only:
                    raise ValueError("Data identity audit failed; no GPU dispatch. Inspect input_identity_audit.json")
        extra = ["--stop-at", str(args.stop_at)] if args.stop_at is not None else []
        if args.stage == "evaluate":
            extra += ["--tag", args.tag]
        run_stage(cfg, args.config, args.stage, extra)
        return
    if sys.platform != "linux":
        raise RuntimeError("Model execution is restricted to authorized Linux servers")
    digest = sha(args.config)
    if args.stage == "features":
        from .features import prepare
        result = prepare(cfg)
    elif args.stage == "train":
        from .training import train
        result = train(cfg, digest, args.stop_at)
    else:
        from .evaluation import evaluate
        result = evaluate(cfg, digest, args.tag)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
