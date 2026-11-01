#!/usr/bin/env python3
"""Resample the six legacy H5 datasets to 200 Hz and merge them with native manifests."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pickle
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import lmdb
import numpy as np
from scipy.signal import resample_poly


LEGACY_DATASETS = (
    "PHYSIONET_4CLASS",
    "SEED_BINARY",
    "ATTEN_GONOGO",
    "MENTAL_ARITHMETIC",
    "SCHIRR_LR",
    "CHBMIT_SUBSET",
)
NATIVE_DATASETS = (
    "FACED_NATIVE_9",
    "THUEP_NATIVE_VALENCE2",
    "KOREAU_NATIVE_LR",
    "SHIN_NATIVE_LR",
    "DEAP_NATIVE_VALENCE2",
    "CHO_NATIVE_LR",
    "WEIBO_NATIVE_LR",
    "TSINGHUA_NATIVE_SSVEP40",
)
SPLITS = ("train", "val", "test")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_legacy_window(row: dict[str, Any], handles: dict[str, h5py.File]) -> tuple[np.ndarray, bool]:
    path = str(row["h5_path"])
    if path not in handles:
        handles[path] = h5py.File(path, "r")
    group = handles[path][str(row["subject"])]
    trial = int(row["trial_idx"])
    concat_count = int(row.get("time_concat_count") or 1)
    if concat_count > 1:
        chunks = np.asarray(group["X"][trial : trial + concat_count], dtype=np.float32)
        x = chunks.transpose(1, 0, 2).reshape(chunks.shape[1], -1)
    else:
        x = np.asarray(group["X"][trial], dtype=np.float32)

    start = int(row.get("window_start") or 0)
    length = row.get("window_length")
    if length is not None:
        x = x[:, start : start + int(length)]
    normalized = bool(row.get("normalize_before_padding"))
    if normalized:
        x = x - x.mean(axis=1, keepdims=True)
        x = x / (x.std(axis=1, keepdims=True) + 1e-6)
    target = row.get("pad_to_length")
    if target is not None:
        missing = int(target) - int(x.shape[-1])
        if missing < 0:
            raise ValueError(f"cannot pad {row['id']} from {x.shape[-1]} to {target}")
        left = missing // 2
        x = np.pad(x, ((0, 0), (left, missing - left)), mode="constant")
    if x.shape[-1] != 1000:
        raise ValueError(f"expected a 1000-sample legacy window for {row['id']}, got {x.shape}")
    return x, normalized


def process_subject(
    task: tuple[str, str, str, list[dict[str, Any]]]
) -> tuple[str, str, str, list[tuple[str, dict[str, Any], dict[str, Any]]]]:
    dataset, split, subject, rows = task
    handles: dict[str, h5py.File] = {}
    records: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    try:
        for row in rows:
            x, already_normalized = load_legacy_window(row, handles)
            x = resample_poly(x, 4, 5, axis=-1).astype(np.float32)
            if x.shape[-1] != 800:
                raise ValueError(f"expected 800 samples after resampling {row['id']}, got {x.shape}")
            key = str(row["id"])
            record = {
                "sample": x.reshape(x.shape[0], 4, 200),
                "label": row["label"],
                "subject": subject,
                "source_sampling_rate": 250,
                "sampling_rate": 200,
                "source_id": row["id"],
            }
            manifest = {
                key_: value
                for key_, value in row.items()
                if key_
                not in {
                    "h5_path",
                    "trial_idx",
                    "time_concat_count",
                    "window_start",
                    "window_length",
                    "pad_to_length",
                    "normalize_before_padding",
                    "stats_path",
                }
            }
            manifest.update(
                {
                    "id": key,
                    "dataset": dataset,
                    "input_shape": [int(x.shape[0]), 800],
                    "sampling_rate": 200,
                    "source_sampling_rate": 250,
                    "patch_size": 200,
                    "patch_count": 4,
                    "canonical_channel_indices": list(range(int(x.shape[0]))),
                    "skip_trial_normalization": already_normalized,
                    "input_scale": 1.0,
                }
            )
            records.append((key, record, manifest))
    finally:
        for handle in handles.values():
            handle.close()
    return dataset, split, subject, records


def build_legacy_dataset(
    dataset: str,
    legacy_root: Path,
    lmdb_root: Path,
    manifest_root: Path,
    workers: int,
) -> dict[str, Any]:
    output = lmdb_root / dataset / "processed_average"
    protocol_path = output / "protocol.json"
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Choose a new output directory: {output}")

    split_rows = {split: read_jsonl(legacy_root / dataset / f"{split}.jsonl") for split in SPLITS}
    tasks: list[tuple[str, str, str, list[dict[str, Any]]]] = []
    for split, rows in split_rows.items():
        by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if int(row.get("sampling_rate") or 0) != 250:
                raise ValueError(f"{dataset}/{split}/{row['id']} is not marked 250 Hz")
            by_subject[str(row["subject"])].append(row)
        tasks.extend((dataset, split, subject, selected) for subject, selected in by_subject.items())

    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"Inspect the existing partial output: {partial}")
    partial.parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(partial), map_size=80 << 30)
    keys: dict[str, list[str]] = {split: [] for split in SPLITS}
    manifests: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    subjects: dict[str, set[str]] = {split: set() for split in SPLITS}
    labels: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}

    with mp.Pool(processes=min(int(workers), len(tasks))) as pool:
        for _, split, subject, records in pool.imap_unordered(process_subject, tasks):
            with env.begin(write=True) as transaction:
                for key, record, manifest in records:
                    transaction.put(key.encode(), pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL))
                    manifest["lmdb_path"] = str(output)
                    manifest["lmdb_key"] = key
                    keys[split].append(key)
                    manifests[split].append(manifest)
                    labels[split][str(manifest["label"])] += 1
            subjects[split].add(subject)
            print(f"[{dataset}] {split} {subject}: {len(records)}", flush=True)

    protocol = {
        "dataset": dataset,
        "source_protocol": "legacy standardized 75x1000 H5 at 250 Hz",
        "output_protocol": "physical polyphase resampling to 75x800 at 200 Hz",
        "sampling_rate": 200,
        "source_sampling_rate": 250,
        "patch_size": 200,
        "patch_count": 4,
        "split_policy": "preserve the existing subject-disjoint train/val/test split",
        "splits": {
            split: {
                "samples": len(keys[split]),
                "subjects": sorted(subjects[split]),
                "label_distribution": dict(sorted(labels[split].items())),
            }
            for split in SPLITS
        },
    }
    with env.begin(write=True) as transaction:
        transaction.put(b"__keys__", pickle.dumps(keys, protocol=pickle.HIGHEST_PROTOCOL))
        transaction.put(b"__meta__", json.dumps(protocol, ensure_ascii=False).encode())
    env.sync()
    env.close()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output appeared during conversion: {output}")
    partial.rename(output)
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for split in SPLITS:
        manifests[split].sort(key=lambda row: str(row["id"]))
        write_jsonl(manifest_root / dataset / f"{split}.jsonl", manifests[split])
    return protocol


def audit_dataset(root: Path, dataset: str) -> dict[str, Any]:
    split_report: dict[str, Any] = {}
    subject_sets: dict[str, set[str]] = {}
    for split in SPLITS:
        rows = read_jsonl(root / dataset / f"{split}.jsonl")
        rates = sorted({int(row["sampling_rate"]) for row in rows})
        shapes = sorted({tuple(row["input_shape"]) for row in rows})
        subject_sets[split] = {str(row["subject"]) for row in rows}
        split_report[split] = {
            "samples": len(rows),
            "subjects": len(subject_sets[split]),
            "sampling_rates": rates,
            "input_shapes": [list(shape) for shape in shapes],
        }
        if rates != [200]:
            raise RuntimeError(f"{dataset}/{split} failed the 200 Hz audit: {rates}")
    overlaps = {
        "train_val": sorted(subject_sets["train"] & subject_sets["val"]),
        "train_test": sorted(subject_sets["train"] & subject_sets["test"]),
        "val_test": sorted(subject_sets["val"] & subject_sets["test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"{dataset} has cross-split subject overlap: {overlaps}")
    return {"splits": split_report, "subject_overlap": overlaps}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--legacy_root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--native_root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--lmdb_root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        required=True,
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=False)
    conversion_report = {}
    for dataset in LEGACY_DATASETS:
        conversion_report[dataset] = build_legacy_dataset(
            dataset, args.legacy_root, args.lmdb_root, args.output_root, args.workers
        )
    for dataset in NATIVE_DATASETS:
        destination = args.output_root / dataset
        destination.mkdir(parents=True, exist_ok=True)
        for split in SPLITS:
            shutil.copy2(args.native_root / dataset / f"{split}.jsonl", destination / f"{split}.jsonl")
        audit = args.native_root / dataset / "audit.json"
        if audit.is_file():
            shutil.copy2(audit, destination / "source_audit.json")

    datasets = [*LEGACY_DATASETS, *NATIVE_DATASETS]
    audit = {
        "status": "complete",
        "sampling_rate": 200,
        "dataset_count": len(datasets),
        "datasets": {dataset: audit_dataset(args.output_root, dataset) for dataset in datasets},
        "legacy_conversions": conversion_report,
    }
    (args.output_root / "merged14_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "datasets": datasets}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
