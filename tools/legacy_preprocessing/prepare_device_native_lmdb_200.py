#!/usr/bin/env python3
"""Build split-safe 200 Hz LMDBs for the first Device Expert experiment."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import h5py
import lmdb
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, resample_poly, sosfiltfilt


DEAP_CHANNELS = (
    "Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7", "CP5", "CP1",
    "P3", "P7", "PO3", "O1", "Oz", "Pz", "Fp2", "AF4", "Fz", "F4",
    "F8", "FC6", "FC2", "Cz", "C4", "T8", "CP6", "CP2", "P4", "P8",
    "PO4", "O2",
)

THU_EP_CHANNELS = (
    "Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5",
    "FC6", "Cz", "C3", "C4", "T7", "T8", "A1", "A2", "CP1", "CP2",
    "CP5", "CP6", "Pz", "P3", "P4", "P7", "P8", "PO3", "PO4", "Oz",
    "O1", "O2",
)

TSINGHUA_CHANNELS = (
    "Fp1", "Fpz", "Fp2", "AF3", "AF4", "F7", "F5", "F3", "F1", "Fz",
    "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2",
    "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "Cz", "C2", "C4",
    "C6", "T8", "M1", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4",
    "CP6", "TP8", "M2", "P7", "P5", "P3", "P1", "Pz", "P2", "P4",
    "P6", "P8", "PO7", "PO5", "PO3", "POz", "PO4", "PO6", "PO8", "CB1",
    "O1", "Oz", "O2", "CB2",
)


def split_subjects(subjects: list[str], train: int, val: int) -> dict[str, list[str]]:
    return {
        "train": subjects[:train],
        "val": subjects[train : train + val],
        "test": subjects[train + val :],
    }


def write_lmdb(
    output: Path,
    results: Iterable[tuple[str, str, list[tuple[str, dict[str, Any]]]]],
    metadata: dict[str, Any],
) -> None:
    if (output / "data.mdb").is_file():
        print(f"[skip] existing LMDB: {output}", flush=True)
        return
    partial = output.with_name(f"{output.name}.partial")
    if partial.exists():
        raise FileExistsError(f"partial output already exists: {partial}")
    partial.parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(partial), map_size=80 << 30)
    keys: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    class_counts: dict[str, Counter[int]] = {split: Counter() for split in keys}
    subject_counts: dict[str, Counter[str]] = {split: Counter() for split in keys}
    for split, subject, records in results:
        with env.begin(write=True) as transaction:
            for key, record in records:
                transaction.put(key.encode(), pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL))
                keys[split].append(key)
                class_counts[split][int(record["label"])] += 1
                subject_counts[split][subject] += 1
        print(f"[{metadata['dataset']}] {subject} {split}: {len(records)}", flush=True)
    metadata = {
        **metadata,
        "splits": {
            split: {
                "subjects": sorted(subject_counts[split]),
                "samples": len(keys[split]),
                "class_distribution": dict(sorted(class_counts[split].items())),
            }
            for split in keys
        },
    }
    with env.begin(write=True) as transaction:
        transaction.put(b"__keys__", pickle.dumps(keys, protocol=pickle.HIGHEST_PROTOCOL))
        transaction.put(b"__meta__", json.dumps(metadata, ensure_ascii=False).encode())
    env.sync()
    env.close()
    partial.rename(output)
    (output / "protocol.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def process_deap_subject(args: tuple[str, str, Path]) -> tuple[str, str, list[tuple[str, dict[str, Any]]]]:
    split, subject, source = args
    with source.open("rb") as handle:
        payload = pickle.load(handle, encoding="latin1")
    data = np.asarray(payload["data"], dtype=np.float32)[:, :32, 3 * 128 :]
    labels = np.asarray(payload["labels"], dtype=np.float32)[:, 0]
    data = resample_poly(data, 25, 16, axis=-1).astype(np.float32)
    data = data[..., : 60 * 200]
    records: list[tuple[str, dict[str, Any]]] = []
    for trial in range(data.shape[0]):
        label = int(labels[trial] > 5.0)
        for window in range(6):
            sample = data[trial, :, window * 2000 : (window + 1) * 2000]
            sample = sample.reshape(32, 10, 200)
            key = f"{subject}-trial{trial:02d}-win{window:02d}"
            records.append(
                (
                    key,
                    {
                        "sample": sample,
                        "label": label,
                        "subject": subject,
                        "trial": trial,
                        "window": window,
                        "channel_names": list(DEAP_CHANNELS),
                        "sampling_rate": 200,
                        "label_rule": "DEAP valence > 5 -> 1 else 0",
                    },
                )
            )
    return split, subject, records


def build_deap(raw_root: Path, output_root: Path, workers: int) -> None:
    subjects = [f"s{index:02d}" for index in range(1, 33)]
    splits = split_subjects(subjects, 22, 4)
    tasks = [
        (split, subject, raw_root / f"{subject}.dat")
        for split, selected in splits.items()
        for subject in selected
    ]
    for _, _, path in tasks:
        if not path.is_file():
            raise FileNotFoundError(path)
    with mp.Pool(processes=min(workers, len(tasks))) as pool:
        write_lmdb(
            output_root / "EMO_DEAP_200" / "processed_average",
            pool.imap_unordered(process_deap_subject, tasks),
            {
                "dataset": "DEAP_200",
                "source_sampling_rate": 128,
                "sampling_rate": 200,
                "channels": list(DEAP_CHANNELS),
                "window_seconds": 10,
                "patch_size": 200,
                "patch_count": 10,
                "split_policy": "subject-disjoint 22/4/6",
            },
        )


_THU_RATINGS: np.ndarray | None = None


def init_thu_pool(ratings_path: str) -> None:
    global _THU_RATINGS
    with h5py.File(ratings_path, "r") as source:
        _THU_RATINGS = np.asarray(source["ratings"], dtype=np.float32)


def process_thu_subject(args: tuple[str, str, int, Path]) -> tuple[str, str, list[tuple[str, dict[str, Any]]]]:
    split, subject, subject_index, source_path = args
    if _THU_RATINGS is None:
        raise RuntimeError("THU-EP ratings were not initialized")
    with h5py.File(source_path, "r") as source:
        data = np.asarray(source["data"][:, :, :, 5], dtype=np.float32)
    data = np.moveaxis(data, -1, 0)  # stimulus, time, channel
    data = np.moveaxis(data, -1, 1)  # stimulus, channel, time
    data = resample_poly(data, 4, 5, axis=-1).astype(np.float32)[..., :6000]
    ratings = _THU_RATINGS[9, :, subject_index]
    records: list[tuple[str, dict[str, Any]]] = []
    for trial in range(data.shape[0]):
        label = int(float(ratings[trial]) > 3.0)
        for window in range(3):
            sample = data[trial, :, window * 2000 : (window + 1) * 2000]
            sample = sample.reshape(32, 10, 200)
            key = f"{subject}-trial{trial:02d}-win{window:02d}"
            records.append(
                (
                    key,
                    {
                        "sample": sample,
                        "label": label,
                        "subject": subject,
                        "trial": trial,
                        "window": window,
                        "channel_names": list(THU_EP_CHANNELS),
                        "sampling_rate": 200,
                        "label_rule": "THU-EP subject valence > 3 -> 1 else 0",
                    },
                )
            )
    return split, subject, records


def build_thu_ep(raw_root: Path, output_root: Path, workers: int) -> None:
    subjects = [f"sub_{index}" for index in range(1, 81)]
    splits = split_subjects(subjects, 56, 8)
    tasks = [
        (split, subject, int(subject.split("_")[-1]) - 1, raw_root / "EEG_data" / f"{subject}.mat")
        for split, selected in splits.items()
        for subject in selected
    ]
    ratings_path = raw_root / "Ratings" / "ratings.mat"
    with mp.Pool(
        processes=min(workers, len(tasks)), initializer=init_thu_pool, initargs=(str(ratings_path),)
    ) as pool:
        write_lmdb(
            output_root / "EMO_THU_EP_200" / "processed_average",
            pool.imap_unordered(process_thu_subject, tasks),
            {
                "dataset": "THU_EP_200",
                "source_sampling_rate": 250,
                "sampling_rate": 200,
                "channels": list(THU_EP_CHANNELS),
                "window_seconds": 10,
                "patch_size": 200,
                "patch_count": 10,
                "split_policy": "subject-disjoint 56/8/16",
            },
        )


def process_tsinghua_subject(
    args: tuple[str, str, Path]
) -> tuple[str, str, list[tuple[str, dict[str, Any]]]]:
    split, subject, source_path = args
    data = np.asarray(loadmat(source_path)["data"], dtype=np.float32)
    data = data[:, 125:1125, :, :]
    sos = butter(4, (1.0, 40.0), btype="bandpass", fs=250.0, output="sos")
    data = sosfiltfilt(sos, data, axis=1).astype(np.float32)
    data = resample_poly(data, 4, 5, axis=1).astype(np.float32)
    data = data.transpose(2, 3, 0, 1).reshape(40 * 6, 64, 800)
    labels = np.repeat(np.arange(40, dtype=np.int64), 6)
    records: list[tuple[str, dict[str, Any]]] = []
    for trial, (sample, label) in enumerate(zip(data, labels)):
        key = f"{subject}-trial{trial:03d}"
        records.append(
            (
                key,
                {
                    "sample": sample.reshape(64, 4, 200),
                    "label": int(label),
                    "subject": subject,
                    "trial": trial,
                    "channel_names": list(TSINGHUA_CHANNELS),
                    "sampling_rate": 200,
                    "label_rule": "original target index 0-39",
                },
            )
        )
    return split, subject, records


def build_tsinghua(raw_root: Path, output_root: Path, workers: int) -> None:
    subjects = [f"S{index}" for index in range(1, 36)]
    splits = split_subjects(subjects, 24, 4)
    tasks = [
        (split, subject, raw_root / f"{subject}.mat")
        for split, selected in splits.items()
        for subject in selected
    ]
    with mp.Pool(processes=min(workers, len(tasks))) as pool:
        write_lmdb(
            output_root / "SSVEP_TSINGHUA_200" / "processed_average",
            pool.imap_unordered(process_tsinghua_subject, tasks),
            {
                "dataset": "TSINGHUA_SSVEP_200",
                "source_sampling_rate": 250,
                "sampling_rate": 200,
                "channels": list(TSINGHUA_CHANNELS),
                "window_seconds": 4,
                "patch_size": 200,
                "patch_count": 4,
                "split_policy": "subject-disjoint 24/4/7",
                "crop": "0.0-4.0 s after stimulus onset",
            },
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("deap", "thu_ep", "tsinghua_ssvep"))
    parser.add_argument(
        "--raw_root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        required=True,
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(f"Choose a new output directory: {args.output_root}")
    if args.dataset == "deap":
        build_deap(args.raw_root, args.output_root, int(args.workers))
    elif args.dataset == "thu_ep":
        build_thu_ep(args.raw_root, args.output_root, int(args.workers))
    else:
        build_tsinghua(args.raw_root, args.output_root, int(args.workers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
