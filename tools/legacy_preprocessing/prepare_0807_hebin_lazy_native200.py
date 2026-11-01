#!/usr/bin/env python3
"""Build Native200 HeBin manifests that lazily resample the existing 250 Hz H5."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import h5py


SPECS = {
    "HEBIN2021_NATIVE200_LR": {
        "filename": "MI_10_HeBin2021_LR.h5",
        "task": "motor_imagery",
        "labels": {0: "left_hand", 1: "right_hand"},
    },
    "HEBIN2021_NATIVE200_UD": {
        "filename": "MI_10_HeBin2021_UD.h5",
        "task": "motor_imagery_direction",
        "labels": {0: "up", 1: "down"},
    },
}
SPLITS = {
    "train": tuple(f"S{index}" for index in range(1, 41)),
    "val": tuple(f"S{index}" for index in range(41, 50)),
    "test": tuple(f"S{index}" for index in range(50, 59)),
}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def link_base_datasets(base_root: Path, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for source in sorted(base_root.iterdir()):
        if not source.is_dir():
            continue
        target = output_root / source.name
        if target.exists() or target.is_symlink():
            continue
        target.symlink_to(source.resolve(), target_is_directory=True)


def build_dataset(name: str, source_root: Path, output_root: Path) -> dict[str, Any]:
    spec = SPECS[name]
    h5_path = (source_root / spec["filename"]).resolve()
    if not h5_path.is_file():
        raise FileNotFoundError(h5_path)
    labels = spec["labels"]
    report: dict[str, Any] = {
        "dataset": name,
        "source_h5": str(h5_path),
        "source_sampling_rate": 250,
        "sampling_rate": 200,
        "source_crop": [125, 1125],
        "input_shape": [75, 800],
        "materialized_signal_copy": False,
        "splits": {},
    }
    with h5py.File(h5_path, "r") as source:
        for split, requested_subjects in SPLITS.items():
            rows: list[dict[str, Any]] = []
            counts: Counter[str] = Counter()
            present_subjects: list[str] = []
            missing_subjects: list[str] = []
            for subject in requested_subjects:
                if subject not in source:
                    missing_subjects.append(subject)
                    continue
                group = source[subject]
                shape = tuple(int(value) for value in group["X"].shape[1:])
                if shape != (75, 1250):
                    raise RuntimeError(f"{name}/{subject} has unexpected source shape {shape}")
                present_subjects.append(subject)
                for trial_index, raw_label in enumerate(group["Y"]):
                    raw_label = int(raw_label)
                    if raw_label not in labels:
                        continue
                    label = labels[raw_label]
                    counts[label] += 1
                    rows.append(
                        {
                            "id": f"{name.lower()}_{subject}_trial{trial_index:05d}",
                            "dataset": name,
                            "task": spec["task"],
                            "h5_path": str(h5_path),
                            "subject": subject,
                            "trial_idx": int(trial_index),
                            "raw_label": raw_label,
                            "label": label,
                            "source_sampling_rate": 250,
                            "sampling_rate": 200,
                            "resample_on_load": True,
                            "window_start": 125,
                            "window_length": 1000,
                            "input_shape": [75, 800],
                            "canonical_channel_indices": list(range(75)),
                        }
                    )
            write_jsonl(output_root / name / f"{split}.jsonl", rows)
            report["splits"][split] = {
                "samples": len(rows),
                "subjects": len(present_subjects),
                "present_subject_ids": present_subjects,
                "missing_subject_ids": missing_subjects,
                "label_distribution": dict(sorted(counts.items())),
            }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_root",
        type=Path,
        required=True,
    )
    parser.add_argument("--base_manifest_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument(
        "--datasets",
        default="HEBIN2021_NATIVE200_LR,HEBIN2021_NATIVE200_UD",
    )
    args = parser.parse_args()
    names = [name.strip() for name in args.datasets.split(",") if name.strip()]
    for name in names:
        if name not in SPECS:
            raise ValueError(f"Unknown dataset: {name}")
        if (args.base_manifest_root / name).exists():
            raise FileExistsError(f"Base manifests already contain {name}; refusing to write through a symlink")
    if args.base_manifest_root.resolve() in args.output_root.resolve().parents:
        raise ValueError("Output root must not be inside the base manifest root")
    args.output_root.mkdir(parents=True, exist_ok=False)
    link_base_datasets(args.base_manifest_root, args.output_root)
    reports = [
        build_dataset(name.strip(), args.source_root, args.output_root)
        for name in names
    ]
    audit = {
        "status": "complete",
        "base_manifest_root": str(args.base_manifest_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "datasets": reports,
    }
    (args.output_root / "hebin_lazy_native200_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
