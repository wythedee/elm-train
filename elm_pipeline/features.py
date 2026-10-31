"""Native200 records → frozen Encoder shards. Thresholds never fit Val/Test."""
from __future__ import annotations
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
import torch

from .data import SignalReader
from .encoders import load_encoders
from .io import read, sha, write
from .tasks import FIELDS, LEVELS


def retained_coverage(selected, accepted):
    before = Counter(r["label"] for r in selected)
    after = Counter(r["label"] for r in accepted)
    missing = sorted(set(before) - set(after))
    if missing or not accepted:
        raise ValueError("Rejected records removed a dataset/split or label: " + repr(missing))
    return {"selected": len(selected), "accepted": len(accepted),
            "selected_by_label": dict(before), "accepted_by_label": dict(after)}


def weighted_train_cuts(records):
    groups = Counter(r["dataset"] for r in records)
    values = np.array([[r["features"][f] for f in FIELDS] for r in records])
    weights = np.array([1 / groups[r["dataset"]] for r in records])
    cuts = []
    for i in range(6):
        order = np.argsort(values[:, i])
        cdf = np.cumsum(weights[order]); cdf /= cdf[-1]
        cuts.append([float(values[order[min(len(order)-1, np.searchsorted(cdf, q))], i])
                     for q in (1/3, 2/3)])
    return cuts


def prepare(cfg):
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    torch.cuda.set_per_process_memory_fraction(min(0.98, float(os.environ["ELM_ALLOCATOR_GIB"]) / total), 0)
    out = Path(cfg["features"])
    if out.exists():
        raise FileExistsError("Feature output must be new; partial artifacts are preserved: " + str(out))
    out.parent.resolve(strict=True)
    if shutil.disk_usage(out.parent).free < 20 * 1024**3:
        raise OSError("Need 20 GiB disk reserve")
    out.mkdir()
    schema = read(cfg["schema"])
    label_order = schema["labels"]
    bundle = Path(cfg["encoder_bundle"])
    meta = read(bundle / "encoder_config.json")
    for name, info in meta["files"].items():
        if sha(bundle / name) != info["sha256"]:
            raise ValueError("Encoder weight identity mismatch: " + name)
    label, stats = load_encoders(bundle / "encoder_class.safetensors",
                                 bundle / "encoder_statistics.safetensors",
                                 meta["encoder_config"])
    if any(p.requires_grad for net in (label, stats) for p in net.parameters()):
        raise AssertionError("Encoder must be frozen")
    reader = SignalReader(cfg.get("path_map"), cfg.get("data_root"))
    indices, rejected = {}, []
    source_counts = {}
    coverage = {}
    started = time.time()
    allowed_ids = None
    if cfg.get("debug_selection"):
        if not cfg.get("software_debug"):
            raise ValueError("debug_selection requires software_debug=true")
        allowed_ids = read(cfg["debug_selection"])
    cap = cfg.get("per_dataset_limit")
    if cfg.get("software_debug") and (not isinstance(cap, int) or not 1 <= cap <= 8):
        raise ValueError("Software debug must cap records per dataset at 1..8")
    if not cfg.get("software_debug") and cap is not None:
        raise ValueError("Full-data profile cannot silently cap records")
    try:
        for split in cfg.get("splits", ["train", "val"]):
            all_records = []
            for dataset in cfg["datasets"]:
                source = Path(cfg["manifest_root"]) / dataset / (split + ".jsonl")
                rows = [json.loads(line) for line in source.open() if line.strip()]
                source_counts[f"{dataset}/{split}"] = len(rows)
                if allowed_ids is not None:
                    wanted = set(allowed_ids[split][dataset])
                    rows = [r for r in rows if r["id"] in wanted]
                    if {r["id"] for r in rows} != wanted:
                        raise ValueError("Debug selection IDs missing from source")
                if cap is not None:
                    rows = rows[:cap]
                if not rows:
                    raise ValueError("Empty dataset/split: " + dataset + "/" + split)
                first_record = len(all_records)
                shard_rows, tensors, shard_no = [], [], 0

                def flush():
                    nonlocal shard_rows, tensors, shard_no
                    if not shard_rows:
                        return
                    rel = f"{split}/{dataset}/{shard_no:05d}.pt"
                    dest = out / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if shutil.disk_usage(out).free < 20 * 1024**3 + 256 * 1024**2:
                        raise OSError("Feature export disk reserve")
                    payload = {k: torch.cat([v[j] for v in tensors])
                               for j, k in enumerate(("label_bank", "label_valid", "statistics_bank", "statistics_valid"))}
                    temp = dest.with_suffix(".pt.partial")
                    torch.save(payload, temp); temp.replace(dest)
                    write(dest.with_suffix(".json"), {"records": shard_rows,
                          "source_shard": rel, "split": split, "dataset": dataset})
                    for i, r in enumerate(shard_rows):
                        all_records.append({**r, "shard": rel, "shard_index": i})
                    shard_rows, tensors = [], []
                    shard_no += 1

                for r in rows:
                    if time.time() - started > cfg.get("stage_seconds", 3600):
                        raise TimeoutError("Feature-stage original time budget")
                    try:
                        x, channels, _ = reader.source(r)
                        if r["label"] not in label_order:
                            raise ValueError("Unknown semantic label")
                        values, info = reader.statistics(r)
                        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                            b, mask, route = label(x[None].cuda(), channels[None].cuda())
                            s, smask = stats(x[None].cuda(), channels[None].cuda(), mask)
                        if b.shape != (1, 1, 150, 50) or s.shape != b.shape:
                            raise AssertionError("Wrong encoded shape")
                        if not torch.isfinite(b).all() or not torch.isfinite(s).all():
                            raise ValueError("Non-finite Encoder output")
                        tensors.append((b.half().cpu(), mask.cpu(), s.half().cpu(), smask.cpu()))
                        shard_rows.append({"id": r["id"], "dataset": dataset, "subject": r["subject"],
                                           "split": split, "label": r["label"],
                                           "global_label_id": label_order.index(r["label"]),
                                           "features": dict(zip(FIELDS, map(float, values))),
                                           "signal": info, "route": route[0].float().cpu().tolist()})
                    except (OSError, ValueError, KeyError) as exc:
                        rejected.append({"id": r["id"], "dataset": dataset, "split": split,
                                         "error": repr(exc)})
                        write(out / "rejected.json", rejected)
                        if len(rejected) > cfg.get("max_bad_records", 0):
                            raise
                    if len(shard_rows) >= cfg.get("shard_records", 1024):
                        flush()
                flush()
                coverage[f"{dataset}/{split}"] = retained_coverage(rows, all_records[first_record:])
                write(out / "accepted_coverage.json", coverage)
                print("FEATURES", split, dataset, len(rows), flush=True)
            if len({r["id"] for r in all_records}) != len(all_records):
                raise ValueError("Duplicate record IDs")
            indices[split] = all_records
            write(out / (split + "_index.partial.json"), {"records": all_records})
        if cfg.get("thresholds"):
            cuts = np.array(read(cfg["thresholds"])["cuts"])
            threshold_origin = {"mode": "reuse_frozen_train_thresholds", "sha256": sha(cfg["thresholds"])}
        else:
            if "train" not in indices:
                raise ValueError("Fit thresholds from Train first")
            cuts = np.array(weighted_train_cuts(indices["train"]))
            threshold_origin = {"mode": "fit_only_current_train_dataset_weighted_quantiles"}
        if cuts.shape != (6, 2) or not np.isfinite(cuts).all() or (cuts[:, 0] > cuts[:, 1]).any():
            raise ValueError("Invalid statistical thresholds")
        write(out / "thresholds.json", {"cuts": cuts.tolist(), **threshold_origin})
        for split, records in indices.items():
            for r in records:
                r["bins"] = {f: LEVELS[int(r["features"][f] > cuts[j, 0])
                                     + int(r["features"][f] > cuts[j, 1])]
                             for j, f in enumerate(FIELDS)}
            write(out / (split + "_index.json"), {"records": records})
        result = {"status": "complete", "splits": {k: len(v) for k, v in indices.items()},
                  "source_counts": source_counts, "rejected": len(rejected),
                  "accepted_coverage": coverage,
                  "software_debug": cfg.get("software_debug", False),
                  "full_raw_preprocessing_verified": False,
                  "input_stage": "already-preprocessed Native200 LMDB or declared lazy H5",
                  "encoder_files": meta["files"], "schema_sha256": sha(cfg["schema"]),
                  "source_code_sha256": sha(__file__), "frozen_encoder": True}
        write(out / "features_result.json", result)
        return result
    finally:
        reader.close()
