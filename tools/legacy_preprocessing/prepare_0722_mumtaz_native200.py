#!/usr/bin/env python3
"""Prepare Mumtaz resting EEG as subject-disjoint Native200 LMDB manifests."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lmdb
import mne
import numpy as np

from native_channel_alignment import canonical_channel_indices


DATASET = "MUMTAZ_MDD"
SPLITS = ("train", "val", "test")
SOURCE_CHANNELS = (
    "EEG Fp1-LE",
    "EEG Fp2-LE",
    "EEG F3-LE",
    "EEG F4-LE",
    "EEG C3-LE",
    "EEG C4-LE",
    "EEG P3-LE",
    "EEG P4-LE",
    "EEG O1-LE",
    "EEG O2-LE",
    "EEG F7-LE",
    "EEG F8-LE",
    "EEG T3-LE",
    "EEG T4-LE",
    "EEG T5-LE",
    "EEG T6-LE",
    "EEG Fz-LE",
    "EEG Cz-LE",
    "EEG Pz-LE",
)
CHANNELS = (
    "Fp1",
    "Fp2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T3",
    "T4",
    "T5",
    "T6",
    "Fz",
    "Cz",
    "Pz",
)
LABELS = {
    "H": "healthy_control",
    "MDD": "major_depressive_disorder",
}
EDF_PATTERN = re.compile(r"^(H|MDD) S(\d+) (EC|EO)\.edf$", re.IGNORECASE)


@dataclass(frozen=True)
class Recording:
    path: Path
    cohort: str
    participant: str
    session: str
    split: str


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def discover_recordings(raw_root: Path) -> dict[str, list[tuple[Path, str]]]:
    by_participant: dict[str, list[tuple[Path, str]]] = defaultdict(list)
    for path in sorted(raw_root.glob("*.edf")):
        if "TASK" in path.name.upper():
            continue
        match = EDF_PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(f"unrecognized Mumtaz resting EDF name: {path.name}")
        cohort = match.group(1).upper()
        participant = f"{cohort}_S{int(match.group(2)):02d}"
        session = match.group(3).upper()
        by_participant[participant].append((path, session))
    if not by_participant:
        raise FileNotFoundError(f"no Mumtaz resting EDF files under {raw_root}")
    return dict(by_participant)


def subject_split(
    participants: list[str], seed: int, val_fraction: float, test_fraction: float
) -> dict[str, str]:
    by_cohort: dict[str, list[str]] = defaultdict(list)
    for participant in participants:
        by_cohort[participant.split("_", 1)[0]].append(participant)
    result: dict[str, str] = {}
    rng = np.random.default_rng(int(seed))
    for cohort, selected in sorted(by_cohort.items()):
        selected = sorted(selected)
        rng.shuffle(selected)
        n_test = max(1, int(math.ceil(len(selected) * float(test_fraction))))
        n_val = max(1, int(math.ceil(len(selected) * float(val_fraction))))
        if n_test + n_val >= len(selected):
            raise ValueError(f"not enough {cohort} subjects for a three-way split")
        for participant in selected[:n_test]:
            result[participant] = "test"
        for participant in selected[n_test : n_test + n_val]:
            result[participant] = "val"
        for participant in selected[n_test + n_val :]:
            result[participant] = "train"
    return result


def preprocess_recording(recording: Recording) -> dict[str, Any]:
    mne.set_log_level("ERROR")
    raw = mne.io.read_raw_edf(recording.path, preload=True, verbose="ERROR")
    missing = sorted(set(SOURCE_CHANNELS) - set(raw.ch_names))
    if missing:
        raise ValueError(f"{recording.path.name} is missing channels: {missing}")
    source_sampling_rate = float(raw.info["sfreq"])
    raw.pick_channels(list(SOURCE_CHANNELS), ordered=True)
    raw.resample(200, npad="auto", verbose="ERROR")
    raw.filter(l_freq=0.3, h_freq=75.0, verbose="ERROR")
    raw.notch_filter(freqs=[50.0], verbose="ERROR")

    # MNE stores EEG in volts; CBraMod's original to_data_frame path emits uV.
    signal = np.asarray(raw.get_data(), dtype=np.float32) * 1e6
    window_samples = 5 * 200
    usable = (signal.shape[-1] // window_samples) * window_samples
    if usable < window_samples:
        raise ValueError(f"{recording.path.name} is shorter than one 5-second window")
    signal = signal[:, :usable]
    windows = signal.reshape(len(CHANNELS), -1, 5, 200).transpose(1, 0, 2, 3)
    return {
        "path": str(recording.path),
        "cohort": recording.cohort,
        "participant": recording.participant,
        "session": recording.session,
        "split": recording.split,
        "source_sampling_rate": source_sampling_rate,
        "windows": windows.astype(np.float32, copy=False),
    }


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    by_participant = discover_recordings(args.raw_root)
    split_by_participant = subject_split(
        list(by_participant), args.seed, args.val_fraction, args.test_fraction
    )
    recordings = [
        Recording(
            path=path,
            cohort=participant.split("_", 1)[0],
            participant=participant,
            session=session,
            split=split_by_participant[participant],
        )
        for participant, files in sorted(by_participant.items())
        for path, session in sorted(files)
    ]

    output = args.lmdb_root / DATASET / "processed_average"
    manifest_dir = args.manifest_root / DATASET
    if output.exists() or manifest_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing Mumtaz output: {output} or {manifest_dir}"
        )
    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"remove or inspect stale partial output first: {partial}")
    partial.parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(partial), map_size=int(args.map_size_gb) << 30)

    keys: dict[str, list[str]] = {split: [] for split in SPLITS}
    rows: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    label_counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    participant_sets: dict[str, set[str]] = {split: set() for split in SPLITS}
    file_counts: Counter[str] = Counter()
    mapped_indices = canonical_channel_indices(CHANNELS)
    if any(index < 0 for index in mapped_indices):
        raise RuntimeError(f"Mumtaz channel mapping is incomplete: {mapped_indices}")

    context = mp.get_context("spawn")
    try:
        with context.Pool(processes=min(int(args.workers), len(recordings))) as pool:
            for processed in pool.imap_unordered(preprocess_recording, recordings):
                split = str(processed["split"])
                participant = str(processed["participant"])
                session = str(processed["session"])
                cohort = str(processed["cohort"])
                label = LABELS[cohort]
                windows = processed.pop("windows")
                with env.begin(write=True) as transaction:
                    for window_index, sample in enumerate(windows):
                        key = (
                            f"mumtaz_{participant.lower()}_{session.lower()}_"
                            f"window{window_index:03d}"
                        )
                        payload = {
                            "sample": sample,
                            "label": 0 if cohort == "H" else 1,
                            "subject": participant,
                            "session": session,
                            "source_id": Path(str(processed["path"])).name,
                        }
                        transaction.put(
                            key.encode(), pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
                        )
                        rows[split].append(
                            {
                                "id": key,
                                "dataset": DATASET,
                                "subject": participant,
                                "session": session,
                                "source_file": Path(str(processed["path"])).name,
                                "lmdb_path": str(output),
                                "lmdb_key": key,
                                "input_shape": [len(CHANNELS), 1000],
                                "sampling_rate": 200,
                                "source_sampling_rate": 256,
                                "patch_size": 200,
                                "patch_count": 5,
                                "channels": list(CHANNELS),
                                "canonical_channel_indices": mapped_indices,
                                "task": "major_depressive_disorder",
                                "label": label,
                                "raw_label": 0 if cohort == "H" else 1,
                                "skip_trial_normalization": False,
                                "input_scale": 1.0,
                            }
                        )
                        keys[split].append(key)
                        label_counts[split][label] += 1
                participant_sets[split].add(participant)
                file_counts[split] += 1
                print(
                    f"[mumtaz] split={split} participant={participant} "
                    f"session={session} windows={len(windows)}",
                    flush=True,
                )

        protocol = {
            "dataset": DATASET,
            "status": "complete",
            "source": str(args.raw_root),
            "source_files": len(recordings),
            "participants": len(by_participant),
            "sampling_rate": 200,
            "source_sampling_rate": 256,
            "window_seconds": 5,
            "patch_size": 200,
            "patch_count": 5,
            "channels": list(CHANNELS),
            "canonical_channel_indices": mapped_indices,
            "preprocessing": {
                "channel_selection": "CBraMod Mumtaz 19-channel order",
                "resample_hz": 200,
                "bandpass_hz": [0.3, 75.0],
                "notch_hz": 50.0,
                "stored_unit": "uV",
                "window_normalization": "deferred to the training loader",
            },
            "split_policy": (
                "stratified participant-disjoint split; EC and EO sessions from one "
                "participant always remain in the same split"
            ),
            "seed": int(args.seed),
            "splits": {
                split: {
                    "participants": sorted(participant_sets[split]),
                    "participant_count": len(participant_sets[split]),
                    "source_file_count": int(file_counts[split]),
                    "samples": len(keys[split]),
                    "label_distribution": dict(sorted(label_counts[split].items())),
                }
                for split in SPLITS
            },
        }
        overlap = {
            "train_val": sorted(participant_sets["train"] & participant_sets["val"]),
            "train_test": sorted(participant_sets["train"] & participant_sets["test"]),
            "val_test": sorted(participant_sets["val"] & participant_sets["test"]),
        }
        if any(overlap.values()):
            raise RuntimeError(f"Mumtaz participant leakage detected: {overlap}")
        protocol["subject_overlap"] = overlap
        with env.begin(write=True) as transaction:
            transaction.put(b"__keys__", pickle.dumps(keys, protocol=pickle.HIGHEST_PROTOCOL))
            transaction.put(b"__meta__", json.dumps(protocol, ensure_ascii=False).encode())
        env.sync()
    except BaseException:
        env.close()
        raise
    env.close()

    output.parent.mkdir(parents=True, exist_ok=True)
    partial.rename(output)
    manifest_dir.mkdir(parents=True, exist_ok=False)
    for split in SPLITS:
        rows[split].sort(key=lambda row: str(row["id"]))
        write_jsonl(manifest_dir / f"{split}.jsonl", rows[split])
    (manifest_dir / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return protocol


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--lmdb_root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--manifest_root",
        type=Path,
        required=True,
    )
    parser.add_argument("--seed", type=int, default=7221)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--test_fraction", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--map_size_gb", type=int, default=16)
    args = parser.parse_args()
    protocol = build_dataset(args)
    print(json.dumps(protocol, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
