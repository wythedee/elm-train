#!/usr/bin/env python3
"""Train an EEG-only expert router with prompt-equivalent label-schema heads."""

from __future__ import annotations

if __name__ == "__main__":
    raise SystemExit("Reference snapshot only. Use tools/workbench.py for inspection; configure a fresh run before training.")


import argparse
import inspect
import json
import math
import os
import pickle
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterator

import h5py
import lmdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler

from train_cbramod_cross_attention_classifier import normalize_signal, transform_signal
from train_cbramod_eeg_router import (
    ConcatHead,
    EEGOnlyRouterClassifier,
    inject_expert_lora,
    route_dict,
)
from native_channel_alignment import CANONICAL_CHANNELS, canonical_channel_indices


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA_ROOT = Path(
    os.environ.get(
        "VSTATS_NATIVE200_ROOT",
        "/home/workspace/datasets/yuqiudeng/merged_native200_manifests_0722",
    )
)
_LMDB_ENV_CACHE: dict[tuple[int, str], lmdb.Environment] = {}
FEATURE_BIN_TO_ID = {"low": 0, "mid": 1, "high": 2}

# Native200 LMDBs may be serialized by NumPy 2.x but read in a NumPy 1.x env.
try:
    __import__("numpy._core.numeric")
except ModuleNotFoundError:
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def csv_values(value: str) -> list[str]:
    values = [part.strip() for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("at least one comma-separated value is required")
    return values


def parse_schema_map(value: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in csv_values(value):
        dataset, separator, schema = item.partition(":")
        if not separator or not dataset.strip() or not schema.strip():
            raise ValueError(f"invalid schema mapping: {item}")
        result[dataset.strip()] = schema.strip()
    return result


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


class TaskRows(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        schema_to_id: dict[str, int],
        labels_by_schema: dict[str, list[str]],
        patch_size: int,
        canonical_channel_scatter: bool = False,
        feature_records: dict[str, dict[str, Any]] | None = None,
        feature_names: list[str] | None = None,
        teacher_targets: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
        teacher_target_width: int = 0,
    ):
        self.rows = rows
        self.schema_to_id = schema_to_id
        self.label_to_id = {
            schema: {label: index for index, label in enumerate(labels)}
            for schema, labels in labels_by_schema.items()
        }
        self.patch_size = int(patch_size)
        self.canonical_channel_scatter = bool(canonical_channel_scatter)
        self.feature_records = feature_records or {}
        self.feature_names = list(feature_names or [])
        self.teacher_targets = teacher_targets or {}
        self.teacher_target_width = int(teacher_target_width)
        self._h5: dict[str, h5py.File] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _get_h5(self, path: str) -> h5py.File:
        if path not in self._h5:
            self._h5[path] = h5py.File(path, "r")
        return self._h5[path]

    def _get_lmdb(self, path: str) -> lmdb.Environment:
        key = (os.getpid(), path)
        if key not in _LMDB_ENV_CACHE:
            _LMDB_ENV_CACHE[key] = lmdb.open(
                path, readonly=True, lock=False, readahead=False, meminit=False
            )
        return _LMDB_ENV_CACHE[key]

    def __getitem__(self, index: int):
        row = self.rows[int(index)]
        if row.get("lmdb_path"):
            env = self._get_lmdb(str(row["lmdb_path"]))
            with env.begin(write=False) as transaction:
                payload = transaction.get(str(row["lmdb_key"]).encode())
            if payload is None:
                raise KeyError(f"missing LMDB key for {row['id']}: {row['lmdb_key']}")
            record = pickle.loads(payload)
            x = np.asarray(record["sample"], dtype=np.float32)
            if x.ndim != 3 or int(x.shape[-1]) != self.patch_size:
                raise ValueError(f"{row['id']} has invalid LMDB shape {tuple(x.shape)}")
            x = x.reshape(x.shape[0], -1) * float(row.get("input_scale", 1.0))
            x, _ = transform_signal(x, row)
            normalized = bool(row.get("skip_trial_normalization"))
        else:
            group = self._get_h5(str(row["h5_path"]))[str(row["subject"])]
            trial_index = int(row["trial_idx"])
            concat_count = int(row.get("time_concat_count") or 1)
            if concat_count > 1:
                chunks = np.asarray(group["X"][trial_index : trial_index + concat_count], dtype=np.float32)
                if chunks.ndim != 3:
                    raise ValueError(f"{row['id']} has invalid concatenation source {tuple(chunks.shape)}")
                x = chunks.transpose(1, 0, 2).reshape(chunks.shape[1], -1)
            else:
                x = np.asarray(group["X"][trial_index], dtype=np.float32)
            x, normalized = transform_signal(x, row)
        if x.ndim != 2 or x.shape[-1] % self.patch_size:
            raise ValueError(f"{row['id']} has invalid transformed shape {tuple(x.shape)}")
        if not normalized:
            x = normalize_signal(x)
        x = x.reshape(x.shape[0], x.shape[-1] // self.patch_size, self.patch_size)
        schema = str(row["_schema"])
        result = (
            torch.from_numpy(x),
            torch.tensor(self.label_to_id[schema][str(row["label"])], dtype=torch.long),
            torch.tensor(self.schema_to_id[schema], dtype=torch.long),
        )
        if self.canonical_channel_scatter:
            indices = row.get("canonical_channel_indices")
            if indices is None:
                indices = canonical_channel_indices(row.get("channels") or ())
            if len(indices) != int(x.shape[0]):
                raise ValueError(
                    f"{row['id']} has {len(indices)} channel indices for {x.shape[0]} EEG channels"
                )
            result = (*result, torch.tensor(indices, dtype=torch.long))
        if self.feature_names:
            record = self.feature_records[str(row["id"])]
            values = record.get("features") or {}
            targets = torch.full((len(self.feature_names),), -100, dtype=torch.long)
            mask = torch.zeros(len(self.feature_names), dtype=torch.bool)
            for feature_index, name in enumerate(self.feature_names):
                if name in values:
                    targets[feature_index] = FEATURE_BIN_TO_ID[str(values[name])]
                    mask[feature_index] = True
            result = (*result, targets, mask)
        if self.teacher_targets:
            teacher_logits, teacher_hidden = self.teacher_targets[str(row["id"])]
            padded_logits = np.full((self.teacher_target_width,), -1e4, dtype=np.float32)
            padded_logits[: len(teacher_logits)] = teacher_logits
            result = (
                *result,
                torch.from_numpy(padded_logits),
                torch.from_numpy(np.asarray(teacher_hidden, dtype=np.float32)),
            )
        return result


class TaskAwareRouterClassifier(nn.Module):
    """The encoder receives only EEG; schema IDs select only output heads."""

    def __init__(
        self,
        schema_to_id: dict[str, int],
        labels_by_schema: dict[str, list[str]],
        mode: str,
        expert_checkpoints: list[Path],
        channels: int,
        patch_count: int,
        patch_size: int,
        compressed_dim: int,
        anchor_time_count: int,
        d_model: int,
        dim_feedforward: int,
        layers: int,
        heads: int,
        head_hidden: int,
        head_bottleneck: int,
        router_hidden: int,
        router_temperature: float,
        router_jitter: float,
        dropout: float,
        lowdim_adapter_dim: int = 0,
        task_expert_count: int = 0,
        task_expert_indices: list[int] | None = None,
        feature_expert_count: int = 0,
        device_expert_count: int = 0,
        task_bank_weight: float = 0.4,
        feature_bank_weight: float = 0.4,
        device_bank_weight: float = 0.2,
        router_top_k: int = 0,
        task_router_top_k: int = 1,
        feature_router_top_k: int = 0,
        device_router_top_k: int = 1,
        feature_names: list[str] | None = None,
        router_input_mode: str = "raw_stats",
        router_query_count: int = 4,
        router_query_heads: int = 4,
        router_reference_checkpoint: str = "",
    ):
        super().__init__()
        self.schema_to_id = dict(schema_to_id)
        self.schemas = [schema for schema, _ in sorted(schema_to_id.items(), key=lambda item: item[1])]
        self.labels_by_schema = labels_by_schema
        self.encoder = EEGOnlyRouterClassifier(
            mode=mode,
            expert_checkpoints=expert_checkpoints,
            channels=channels,
            patch_count=patch_count,
            patch_size=patch_size,
            compressed_dim=compressed_dim,
            anchor_time_count=anchor_time_count,
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            layers=layers,
            heads=heads,
            head_hidden=head_hidden,
            head_bottleneck=head_bottleneck,
            dropout=dropout,
            classes=max(len(labels) for labels in labels_by_schema.values()),
            router_hidden=router_hidden,
            router_temperature=router_temperature,
            router_jitter=router_jitter,
            lowdim_adapter_dim=lowdim_adapter_dim,
            task_expert_count=task_expert_count,
            task_expert_indices=task_expert_indices,
            feature_expert_count=feature_expert_count,
            device_expert_count=device_expert_count,
            task_bank_weight=task_bank_weight,
            feature_bank_weight=feature_bank_weight,
            device_bank_weight=device_bank_weight,
            router_top_k=router_top_k,
            task_router_top_k=task_router_top_k,
            feature_router_top_k=feature_router_top_k,
            device_router_top_k=device_router_top_k,
            router_input_mode=router_input_mode,
            router_query_count=router_query_count,
            router_query_heads=router_query_heads,
            router_reference_checkpoint=router_reference_checkpoint,
        )
        del self.encoder.head
        flat_dim = channels * anchor_time_count * compressed_dim
        self.heads = nn.ModuleDict(
            {
                schema: ConcatHead(
                    flat_dim,
                    head_hidden,
                    head_bottleneck,
                    dropout,
                    len(labels_by_schema[schema]),
                )
                for schema in self.schemas
            }
        )
        self.max_classes = max(len(labels) for labels in labels_by_schema.values())
        self.feature_names = list(feature_names or [])
        self.feature_trunk: nn.Module | None = None
        self.feature_heads = nn.ModuleList()
        if self.feature_names:
            self.feature_trunk = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat_dim, head_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, head_bottleneck),
                nn.GELU(),
            )
            self.feature_heads.extend(
                nn.Linear(head_bottleneck, 3) for _name in self.feature_names
            )

    @property
    def num_experts(self) -> int:
        return self.encoder.num_experts

    def forward(
        self,
        x: torch.Tensor,
        schema_ids: torch.Tensor,
        channel_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reduced, route = self.encoder.encode(x, channel_indices)
        return self._label_logits(reduced, schema_ids), route

    def _label_logits(
        self, reduced: torch.Tensor, schema_ids: torch.Tensor
    ) -> torch.Tensor:
        return self._label_logits_and_hidden(reduced, schema_ids)[0]

    def _label_logits_and_hidden(
        self, reduced: torch.Tensor, schema_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        padded_logits = []
        hidden_states = []
        for schema in self.schemas:
            head = self.heads[schema]
            hidden = head.net[:-1](reduced)
            logits = head.net[-1](hidden)
            if logits.shape[-1] < self.max_classes:
                padding = logits.new_full((logits.shape[0], self.max_classes - logits.shape[-1]), -1e4)
                logits = torch.cat((logits, padding), dim=-1)
            padded_logits.append(logits)
            hidden_states.append(hidden)
        all_logits = torch.stack(padded_logits, dim=1)
        all_hidden = torch.stack(hidden_states, dim=1)
        batch_indices = torch.arange(reduced.shape[0], device=reduced.device)
        return (
            all_logits[batch_indices, schema_ids],
            all_hidden[batch_indices, schema_ids],
        )

    def forward_distill(
        self,
        x: torch.Tensor,
        schema_ids: torch.Tensor,
        channel_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        reduced, route = self.encoder.encode(x, channel_indices)
        logits, hidden = self._label_logits_and_hidden(reduced, schema_ids)
        return logits, hidden, route

    def forward_with_features(
        self,
        x: torch.Tensor,
        schema_ids: torch.Tensor,
        channel_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        reduced, route = self.encoder.encode(x, channel_indices)
        if self.feature_trunk is None:
            raise RuntimeError("feature heads are not configured")
        hidden = self.feature_trunk(reduced)
        feature_logits = [head(hidden) for head in self.feature_heads]
        return self._label_logits(reduced, schema_ids), feature_logits, route

    def load_v1_warmstart(
        self, path: Path, head_schema: str | None, skip_experts: bool = False
    ) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        source = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if not isinstance(source, dict):
            raise RuntimeError(f"unsupported warmstart checkpoint: {path}")
        current = self.state_dict()
        compatible: dict[str, torch.Tensor] = {}
        expanded: list[str] = []
        remapped: list[str] = []
        for key, value in source.items():
            target = None
            if str(key) in current:
                target = str(key)
            elif str(key).startswith("encoder.router.task."):
                task_router_target = "encoder.router." + str(key)[len("encoder.router.task.") :]
                if task_router_target in current:
                    target = task_router_target
                    remapped.append(f"{key}->{target}")
            elif str(key).startswith(("experts.", "router.", "token_compressor.")):
                target = f"encoder.{key}"
            elif head_schema and str(key).startswith("head."):
                target = f"heads.{head_schema}.{str(key)[len('head.') :]}"
            if skip_experts and target and target.startswith("encoder.experts."):
                continue
            if target in current and isinstance(value, torch.Tensor) and tuple(value.shape) == tuple(current[target].shape):
                compatible[target] = value
            elif (
                target in current
                and isinstance(value, torch.Tensor)
                and "router" in target
                and value.ndim > 0
                and value.shape[0] < current[target].shape[0]
                and tuple(value.shape[1:]) == tuple(current[target].shape[1:])
            ):
                merged = current[target].clone()
                merged[: value.shape[0]].copy_(value)
                compatible[target] = merged
                expanded.append(target)
        result = self.load_state_dict(compatible, strict=False)
        return {
            "source": str(path),
            "loaded_keys": len(compatible),
            "expanded_keys": expanded,
            "remapped_keys": remapped,
            "missing_keys": len(result.missing_keys),
            "missing_key_names": list(result.missing_keys),
            "unexpected_keys": len(result.unexpected_keys),
            "head_schema": head_schema,
            "skip_experts": bool(skip_experts),
        }


def load_rows(
    data_root: Path,
    datasets: list[str],
    schema_map: dict[str, str],
    task_map: dict[str, str],
) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    for dataset in datasets:
        for split in ("train", "val", "test"):
            items = read_jsonl(data_root / dataset / f"{split}.jsonl")
            if not items:
                raise RuntimeError(f"{dataset}/{split} is empty")
            rows[f"{dataset}_{split}"] = [
                {
                    **item,
                    "_dataset": dataset,
                    "_schema": schema_map[dataset],
                    "_task": task_map[dataset],
                }
                for item in items
            ]
    return rows


def validate_contract(
    rows: dict[str, list[dict[str, Any]]],
    datasets: list[str],
    schema_map: dict[str, str],
    patch_size: int,
    canonical_channel_scatter: bool = False,
) -> tuple[int, dict[str, int], dict[str, list[str]]]:
    shapes_by_dataset: dict[str, set[tuple[int, int]]] = {}
    for dataset in datasets:
        shapes = {
            tuple(int(value) for value in row["input_shape"])
            for split in ("train", "val", "test")
            for row in rows[f"{dataset}_{split}"]
        }
        if not canonical_channel_scatter and len(shapes) != 1:
            raise RuntimeError(f"{dataset} mixes input shapes: {sorted(shapes)}")
        sample_counts = {shape[1] for shape in shapes}
        if len(sample_counts) != 1:
            raise RuntimeError(
                f"{dataset} mixes temporal input lengths: {sorted(shapes)}"
            )
        shapes_by_dataset[dataset] = shapes
    channels_set = {
        shape[0] for shapes in shapes_by_dataset.values() for shape in shapes
    }
    if canonical_channel_scatter:
        channels = len(CANONICAL_CHANNELS)
        for dataset in datasets:
            for split in ("train", "val", "test"):
                for row in rows[f"{dataset}_{split}"]:
                    indices = row.get("canonical_channel_indices")
                    if indices is None:
                        indices = canonical_channel_indices(row.get("channels") or ())
                    if len(indices) != int(row["input_shape"][0]) or not any(
                        int(value) >= 0 for value in indices
                    ):
                        raise RuntimeError(f"{row['id']} has invalid canonical channel metadata")
    else:
        if len(channels_set) != 1:
            raise RuntimeError(f"datasets do not share channel count: {shapes_by_dataset}")
        channels = next(iter(channels_set))
    patch_counts: dict[str, int] = {}
    for dataset, shapes in shapes_by_dataset.items():
        samples = next(iter(shapes))[1]
        if samples % int(patch_size):
            raise RuntimeError(
                f"{dataset} sample count {samples} is not divisible by patch size {patch_size}"
            )
        patch_counts[dataset] = samples // int(patch_size)
    labels_by_schema: dict[str, list[str]] = {}
    for schema in sorted(set(schema_map.values())):
        schema_datasets = [dataset for dataset in datasets if schema_map[dataset] == schema]
        label_sets = {
            tuple(sorted({str(row["label"]) for row in rows[f"{dataset}_train"]}))
            for dataset in schema_datasets
        }
        if len(label_sets) != 1:
            raise RuntimeError(f"schema {schema} has inconsistent dataset labels: {sorted(label_sets)}")
        labels_by_schema[schema] = list(next(iter(label_sets)))
    return channels, patch_counts, labels_by_schema


def hierarchical_sampling_weights(
    rows: list[dict[str, Any]], strategy: str
) -> torch.Tensor:
    """Return normalized row weights for a declared sampling hierarchy."""
    if not rows:
        raise ValueError("cannot build sampling weights for an empty row list")
    if strategy in {"proportional", "natural_full_pass"}:
        return torch.full((len(rows),), 1.0 / len(rows), dtype=torch.double)

    tasks = sorted({str(row["_task"]) for row in rows})
    datasets_by_task = {
        task: sorted({str(row["_dataset"]) for row in rows if str(row["_task"]) == task})
        for task in tasks
    }
    datasets = sorted({str(row["_dataset"]) for row in rows})
    labels_by_dataset = {
        dataset: sorted({str(row["label"]) for row in rows if str(row["_dataset"]) == dataset})
        for dataset in datasets
    }
    subjects_by_dataset = {
        dataset: sorted(
            {
                str(row.get("subject", "UNKNOWN"))
                for row in rows
                if str(row["_dataset"]) == dataset
            }
        )
        for dataset in datasets
    }
    labels_by_dataset_subject = {
        (dataset, subject): sorted(
            {
                str(row["label"])
                for row in rows
                if str(row["_dataset"]) == dataset
                and str(row.get("subject", "UNKNOWN")) == subject
            }
        )
        for dataset in datasets
        for subject in subjects_by_dataset[dataset]
    }
    dataset_label_counts = Counter(
        (str(row["_dataset"]), str(row["label"])) for row in rows
    )
    subject_label_counts = Counter(
        (
            str(row["_dataset"]),
            str(row.get("subject", "UNKNOWN")),
            str(row["label"]),
        )
        for row in rows
    )
    weights = []
    for row in rows:
        task = str(row["_task"])
        dataset = str(row["_dataset"])
        subject = str(row.get("subject", "UNKNOWN"))
        label = str(row["label"])
        if strategy == "task_dataset_label":
            denominator = (
                len(tasks)
                * len(datasets_by_task[task])
                * len(labels_by_dataset[dataset])
                * dataset_label_counts[(dataset, label)]
            )
        elif strategy == "dataset_label":
            denominator = (
                len(datasets)
                * len(labels_by_dataset[dataset])
                * dataset_label_counts[(dataset, label)]
            )
        elif strategy == "dataset_subject_label":
            denominator = (
                len(datasets)
                * len(subjects_by_dataset[dataset])
                * len(labels_by_dataset_subject[(dataset, subject)])
                * subject_label_counts[(dataset, subject, label)]
            )
        elif strategy == "task_dataset_subject_label":
            denominator = (
                len(tasks)
                * len(datasets_by_task[task])
                * len(subjects_by_dataset[dataset])
                * len(labels_by_dataset_subject[(dataset, subject)])
                * subject_label_counts[(dataset, subject, label)]
            )
        else:
            raise ValueError(f"unsupported sampling strategy: {strategy}")
        weights.append(1.0 / denominator)
    result = torch.as_tensor(weights, dtype=torch.double)
    return result / result.sum()


def task_balanced_weights(rows: list[dict[str, Any]]) -> torch.Tensor:
    """Backward-compatible name for the historical sampling protocol."""
    return hierarchical_sampling_weights(rows, "task_dataset_label")


def sampling_mass_audit(
    rows: list[dict[str, Any]], weights: torch.Tensor, strategy: str
) -> dict[str, Any]:
    by_dataset: dict[str, float] = defaultdict(float)
    by_task: dict[str, float] = defaultdict(float)
    by_subject: dict[str, float] = defaultdict(float)
    for row, weight in zip(rows, weights.tolist(), strict=True):
        dataset = str(row["_dataset"])
        task = str(row["_task"])
        subject = str(row.get("subject", "UNKNOWN"))
        by_dataset[dataset] += float(weight)
        by_task[task] += float(weight)
        by_subject[f"{dataset}/{subject}"] += float(weight)
    return {
        "strategy": strategy,
        "expected_task_mass": dict(sorted(by_task.items())),
        "expected_dataset_mass": dict(sorted(by_dataset.items())),
        "expected_subject_mass_min": min(by_subject.values()),
        "expected_subject_mass_max": max(by_subject.values()),
        "unique_subjects": len(by_subject),
    }


def task_balanced_sampler(rows: list[dict[str, Any]], num_samples: int, seed: int) -> WeightedRandomSampler:
    return WeightedRandomSampler(
        task_balanced_weights(rows),
        num_samples=int(num_samples),
        replacement=True,
        generator=torch.Generator().manual_seed(int(seed)),
    )


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Draw balanced replacement batches without mixing temporal tensor lengths."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        weights: torch.Tensor,
        batch_size: int,
        num_batches: int,
        seed: int,
    ):
        if len(rows) != int(weights.numel()):
            raise ValueError("row and sampling-weight counts differ")
        self.batch_size = int(batch_size)
        self.num_batches = int(num_batches)
        self.seed = int(seed)
        self.iteration = 0
        grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            grouped[tuple(int(value) for value in row["input_shape"])].append(index)
        self.shapes = sorted(grouped)
        self.indices = [torch.tensor(grouped[shape], dtype=torch.long) for shape in self.shapes]
        self.weights = [weights.index_select(0, indices).double() for indices in self.indices]
        self.bucket_weights = torch.tensor(
            [float(values.sum()) for values in self.weights], dtype=torch.double
        )

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.iteration)
        self.iteration += 1
        selected_buckets = torch.multinomial(
            self.bucket_weights,
            self.num_batches,
            replacement=True,
            generator=generator,
        )
        for bucket_index in selected_buckets.tolist():
            local = torch.multinomial(
                self.weights[bucket_index],
                self.batch_size,
                replacement=True,
                generator=generator,
            )
            yield self.indices[bucket_index].index_select(0, local).tolist()


class LengthBucketFullPassBatchSampler(Sampler[list[int]]):
    """Shuffle once per epoch while visiting every heterogeneous row exactly once."""

    def __init__(self, rows: list[dict[str, Any]], batch_size: int, seed: int):
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.iteration = 0
        grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            grouped[tuple(int(value) for value in row["input_shape"])].append(index)
        self.shapes = sorted(grouped)
        self.indices = [torch.tensor(grouped[shape], dtype=torch.long) for shape in self.shapes]
        self.num_batches = sum(
            math.ceil(len(indices) / self.batch_size) for indices in self.indices
        )

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.iteration)
        self.iteration += 1
        batches: list[list[int]] = []
        for indices in self.indices:
            shuffled = indices.index_select(
                0, torch.randperm(len(indices), generator=generator)
            )
            for start in range(0, len(shuffled), self.batch_size):
                batches.append(shuffled[start : start + self.batch_size].tolist())
        order = torch.randperm(len(batches), generator=generator).tolist()
        for index in order:
            yield batches[index]


def metric_dict(y_true: list[int], y_pred: list[int], labels: list[str]) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "prediction_histogram": {label: int(y_pred.count(index)) for index, label in enumerate(labels)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(labels)))).tolist(),
    }


def auxiliary_feature_loss(
    logits: list[torch.Tensor],
    targets: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    losses = []
    for feature_index, values in enumerate(logits):
        active = mask[:, feature_index]
        if bool(active.any()):
            losses.append(
                F.cross_entropy(
                    values[active],
                    targets[active, feature_index],
                    weight=weights[feature_index],
                )
            )
    if not losses:
        raise RuntimeError("batch contains no active feature targets")
    return torch.stack(losses).mean()


def feature_metric_dict(
    gold: dict[str, list[int]], pred: dict[str, list[int]]
) -> dict[str, Any]:
    per_feature: dict[str, Any] = {}
    scores: list[float] = []
    for name in gold:
        if not gold[name]:
            continue
        labels = [0, 1, 2]
        score = float(balanced_accuracy_score(gold[name], pred[name]))
        scores.append(score)
        per_feature[name] = {
            "accuracy": float(accuracy_score(gold[name], pred[name])),
            "balanced_accuracy": score,
            "macro_f1": float(
                f1_score(gold[name], pred[name], average="macro", zero_division=0)
            ),
            "gold_histogram": dict(Counter(gold[name])),
            "prediction_histogram": dict(Counter(pred[name])),
            "confusion_matrix": confusion_matrix(
                gold[name], pred[name], labels=labels
            ).tolist(),
        }
    return {
        "feature_macro_bacc": mean(scores) if scores else None,
        "per_feature": per_feature,
    }


def gradient_norm(parameters) -> float:
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().float().norm(2).item()
        squared += value * value
    return float(math.sqrt(squared))


def bank_probabilities(model: TaskAwareRouterClassifier) -> dict[str, torch.Tensor]:
    logits = model.encoder.last_router_logits
    if logits is None:
        return {}
    temperature = float(model.encoder.router_temperature)
    banks = {
        "task": model.encoder.task_expert_indices,
        "feature": model.encoder.feature_expert_indices,
        "device": model.encoder.device_expert_indices,
    }
    result: dict[str, torch.Tensor] = {}
    if model.encoder.mode == "soft_moe":
        result["all"] = torch.softmax(logits / temperature, dim=-1)
        return result
    for name, indices in banks.items():
        if not indices:
            continue
        index = torch.tensor(indices, dtype=torch.long, device=logits.device)
        result[name] = torch.softmax(logits.index_select(1, index) / temperature, dim=-1)
    return result


def supervised_task_group_loss(logits, schema_ids, schema_to_id, groups):
    """Negative log probability mass of compatible experts, before sparse routing."""
    log_probs = F.log_softmax(logits.float(), dim=-1)
    losses = torch.zeros(len(logits), device=logits.device)
    covered = torch.zeros(len(logits), device=logits.device, dtype=torch.bool)
    for schema, indices in groups.items():
        mask = schema_ids == schema_to_id[schema]
        if mask.any():
            losses[mask] = -torch.logsumexp(log_probs[mask][:, indices], dim=-1)
            covered |= mask
    if not covered.all():
        raise ValueError("Task group supervision must cover every training schema")
    return losses.mean()


def supervised_bank_loss(
    *,
    logits: torch.Tensor,
    schema_ids: torch.Tensor,
    schema_to_id: dict[str, int],
    supervision: dict[str, int],
    bank_indices: tuple[int, ...],
) -> torch.Tensor | None:
    if not supervision:
        return None
    absolute_to_local = {int(expert): index for index, expert in enumerate(bank_indices)}
    invalid = {
        schema: expert
        for schema, expert in supervision.items()
        if int(expert) not in absolute_to_local
    }
    if invalid:
        raise RuntimeError(f"router supervision targets experts outside its bank: {invalid}")
    targets = torch.full_like(schema_ids, -100)
    for schema, expert in supervision.items():
        targets[schema_ids == schema_to_id[schema]] = absolute_to_local[int(expert)]
    selected = targets >= 0
    if not bool(selected.any()):
        return None
    index = torch.tensor(bank_indices, dtype=torch.long, device=logits.device)
    bank_logits = logits.index_select(1, index)
    return F.cross_entropy(bank_logits[selected].float(), targets[selected])


def evaluate(
    model: TaskAwareRouterClassifier,
    loaders: dict[str, DataLoader],
    dataset_schemas: dict[str, str],
    dataset_tasks: dict[str, str],
    labels_by_schema: dict[str, list[str]],
    device: torch.device,
    amp_enabled: bool,
    feature_names: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    model.eval()
    results: dict[str, dict[str, Any]] = {}
    with torch.inference_mode():
        for dataset, loader in loaders.items():
            y_true: list[int] = []
            y_pred: list[int] = []
            routes: list[torch.Tensor] = []
            pre_hard_routes: dict[str, list[torch.Tensor]] = defaultdict(list)
            active_feature_names = list(feature_names or [])
            feature_gold = {name: [] for name in active_feature_names}
            feature_pred = {name: [] for name in active_feature_names}
            for batch in loader:
                x, y, schema_ids = batch[:3]
                has_features = bool(active_feature_names)
                has_channels = len(batch) == (6 if has_features else 4)
                channel_indices = batch[3] if has_channels else None
                feature_targets = batch[-2] if has_features else None
                feature_mask = batch[-1] if has_features else None
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                    if has_features:
                        logits, feature_logits, route = model.forward_with_features(
                            x.to(device, non_blocking=True),
                            schema_ids.to(device, non_blocking=True),
                            None
                            if channel_indices is None
                            else channel_indices.to(device, non_blocking=True),
                        )
                    else:
                        logits, route = model(
                            x.to(device, non_blocking=True),
                            schema_ids.to(device, non_blocking=True),
                            None
                            if channel_indices is None
                            else channel_indices.to(device, non_blocking=True),
                        )
                y_true.extend(y.tolist())
                y_pred.extend(logits.argmax(dim=-1).cpu().tolist())
                routes.append(route.float().cpu())
                for bank, probability in bank_probabilities(model).items():
                    pre_hard_routes[bank].append(probability.float().cpu())
                if has_features:
                    assert feature_targets is not None and feature_mask is not None
                    for feature_index, name in enumerate(active_feature_names):
                        active = feature_mask[:, feature_index]
                        if bool(active.any()):
                            feature_gold[name].extend(
                                feature_targets[active, feature_index].tolist()
                            )
                            feature_pred[name].extend(
                                feature_logits[feature_index][active]
                                .argmax(dim=-1)
                                .cpu()
                                .tolist()
                            )
            schema = dataset_schemas[dataset]
            results[dataset] = {
                "schema": schema,
                "task": dataset_tasks[dataset],
                "samples": len(y_true),
                "metrics": metric_dict(y_true, y_pred, labels_by_schema[schema]),
                "routing": route_dict(torch.cat(routes)),
                "routing_pre_hard": {
                    bank: route_dict(torch.cat(values))
                    for bank, values in pre_hard_routes.items()
                },
            }
            if has_features:
                results[dataset]["feature_metrics"] = feature_metric_dict(
                    feature_gold, feature_pred
                )
    return results


def aggregate_scores(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_schema: dict[str, list[float]] = defaultdict(list)
    by_task: dict[str, list[float]] = defaultdict(list)
    for value in results.values():
        score = float(value["metrics"]["balanced_accuracy"])
        by_schema[str(value["schema"])].append(score)
        by_task[str(value["task"])].append(score)
    schema_scores = {schema: mean(values) for schema, values in by_schema.items()}
    task_scores = {task: mean(values) for task, values in by_task.items()}
    return {
        "dataset_mean_bacc": mean(
            float(value["metrics"]["balanced_accuracy"]) for value in results.values()
        ),
        "schema_mean_bacc": schema_scores,
        "task_mean_bacc": task_scores,
        "task_macro_bacc": mean(task_scores.values()),
    }


def load_vft_teacher_targets(
    root: Path,
    datasets: list[str],
    rows: dict[str, list[dict[str, Any]]],
    labels_by_schema: dict[str, list[str]],
    schema_map: dict[str, str],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    targets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for dataset in datasets:
        path = root / dataset / "train.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as values:
            ids = values["ids"].astype(str).tolist()
            logits = np.asarray(values["logits"], dtype=np.float32)
            hidden = np.asarray(values["hidden"], dtype=np.float32)
            labels = values["labels"].astype(str).tolist()
        expected_labels = labels_by_schema[schema_map[dataset]]
        if labels != expected_labels:
            raise RuntimeError(
                f"teacher label order mismatch for {dataset}: {labels} != {expected_labels}"
            )
        if logits.shape != (len(ids), len(labels)) or hidden.shape != (len(ids), 200):
            raise RuntimeError(
                f"invalid teacher targets for {dataset}: logits={logits.shape}, hidden={hidden.shape}"
            )
        expected_ids = {str(row["id"]) for row in rows[f"{dataset}_train"]}
        if set(ids) != expected_ids:
            missing = sorted(expected_ids - set(ids))[:3]
            extra = sorted(set(ids) - expected_ids)[:3]
            raise RuntimeError(
                f"teacher target IDs mismatch for {dataset}: missing={missing}, extra={extra}"
            )
        overlap = set(targets).intersection(ids)
        if overlap:
            raise RuntimeError(f"teacher target IDs are not globally unique: {sorted(overlap)[:3]}")
        targets.update(
            {
                sample_id: (logit, representation)
                for sample_id, logit, representation in zip(ids, logits, hidden, strict=True)
            }
        )
    return targets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="PHYSIONET_4CLASS,KOREAU_LR,CHO_LR,SEED_BINARY")
    parser.add_argument(
        "--schema_map",
        default="PHYSIONET_4CLASS:physionet_mi4,KOREAU_LR:mi_lr,CHO_LR:mi_lr,SEED_BINARY:emotion_binary",
    )
    parser.add_argument(
        "--task_map",
        default="PHYSIONET_4CLASS:motor_imagery,KOREAU_LR:motor_imagery,CHO_LR:motor_imagery,SEED_BINARY:emotion",
    )
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--feature_manifest",
        type=Path,
        help="Optional train-fitted feature targets for an auxiliary Feature prediction head.",
    )
    parser.add_argument("--aux_feature_weight", type=float, default=0.0)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument(
        "--mode",
        choices=("single", "soft_moe", "hybrid_moe", "two_bank_moe", "three_bank_moe"),
        required=True,
    )
    parser.add_argument("--expert_checkpoints", required=True)
    parser.add_argument(
        "--task_expert_count",
        type=int,
        default=0,
        help="For hybrid_moe, the leading experts use hard Top-1 routing; remaining experts stay soft.",
    )
    parser.add_argument(
        "--task_expert_indices",
        default="",
        help="Optional explicit Task Expert indices for hybrid_moe; all remaining experts are Feature Experts.",
    )
    parser.add_argument(
        "--feature_expert_count",
        type=int,
        default=0,
        help="For three_bank_moe, Feature checkpoints follow Task checkpoints.",
    )
    parser.add_argument(
        "--device_expert_count",
        type=int,
        default=0,
        help="For three_bank_moe, Device checkpoints follow Feature checkpoints.",
    )
    parser.add_argument("--task_bank_weight", type=float, default=0.4)
    parser.add_argument("--feature_bank_weight", type=float, default=0.4)
    parser.add_argument("--device_bank_weight", type=float, default=0.2)
    parser.add_argument("--router_top_k", type=int, default=0)
    parser.add_argument("--task_router_top_k", type=int, default=1)
    parser.add_argument("--feature_router_top_k", type=int, default=0)
    parser.add_argument("--device_router_top_k", type=int, default=1)
    parser.add_argument("--warmstart", type=Path)
    parser.add_argument("--strict_warmstart", action="store_true")
    parser.add_argument("--allow_router_input_extension", action="store_true")
    parser.add_argument("--router_reference_checkpoint", default="")
    parser.add_argument("--warmstart_skip_experts", action="store_true")
    parser.add_argument("--warmstart_head_schema", default="")
    parser.add_argument("--freeze_experts", action="store_true")
    parser.add_argument(
        "--vft_teacher_target_root",
        type=Path,
        help="Optional per-dataset VFT logits and 200-d penultimate representations.",
    )
    parser.add_argument("--vft_logit_kl_weight", type=float, default=0.0)
    parser.add_argument("--vft_hidden_cosine_weight", type=float, default=0.0)
    parser.add_argument("--vft_distill_temperature", type=float, default=2.0)
    parser.add_argument("--expert_lora_last_n_layers", type=int, default=0)
    parser.add_argument("--expert_lora_rank", type=int, default=8)
    parser.add_argument("--expert_lora_alpha", type=float, default=16.0)
    parser.add_argument("--expert_lora_dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--steps_per_epoch", type=int, default=400)
    parser.add_argument(
        "--sampling_strategy",
        choices=(
            "task_dataset_label",
            "dataset_label",
            "dataset_subject_label",
            "task_dataset_subject_label",
            "proportional",
            "natural_full_pass",
        ),
        default="task_dataset_label",
        help=(
            "Training row allocation. natural_full_pass disables replacement and visits every "
            "eligible row exactly once per epoch; all other modes use fixed replacement steps."
        ),
    )
    parser.add_argument("--train_num_workers", type=int, default=0)
    parser.add_argument(
        "--checkpoint_trainable_only",
        action="store_true",
        help="Store only trainable parameters; frozen Expert weights are reconstructed from args.",
    )
    parser.add_argument("--eval_limit", type=int)
    parser.add_argument("--test_limit", type=int)
    parser.add_argument("--patch_size", type=int, default=200)
    parser.add_argument(
        "--canonical_channel_scatter",
        action="store_true",
        help="Keep native channels in CBraMod, then scatter encoder tokens to 75 named slots.",
    )
    parser.add_argument(
        "--router_reference_patch_count",
        type=int,
        default=0,
        help="Keep the Router input width at this patch count while experts use runtime length.",
    )
    parser.add_argument("--compressed_dim", type=int, default=25)
    parser.add_argument(
        "--identity_token_compressor",
        action="store_true",
        help="Initialize a square token compressor as identity for information-preserving controls.",
    )
    parser.add_argument("--lowdim_adapter_dim", type=int, default=0)
    parser.add_argument("--anchor_time_count", type=int, default=2)
    parser.add_argument("--d_model", type=int, default=200)
    parser.add_argument("--dim_feedforward", type=int, default=800)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head_hidden", type=int, default=800)
    parser.add_argument("--head_bottleneck", type=int, default=200)
    parser.add_argument("--router_hidden", type=int, default=128)
    parser.add_argument("--router_temperature", type=float, default=1.0)
    parser.add_argument(
        "--router_input_mode",
        choices=("raw_stats", "raw_waveform", "expert_query", "stats_spectral", "stats_encoder"),
        default="raw_stats",
        help=(
            "Use raw EEG statistics, complete canonical waveforms, or learned-query "
            "pooling over Expert 0 tokens."
        ),
    )
    parser.add_argument("--router_query_count", type=int, default=4)
    parser.add_argument("--router_query_heads", type=int, default=4)
    parser.add_argument("--router_balance_weight", type=float, default=0.05)
    parser.add_argument("--router_entropy_weight", type=float, default=0.0)
    parser.add_argument("--router_jitter", type=float, default=1e-4)
    parser.add_argument(
        "--router_supervision",
        default="",
        help="Optional training-only SCHEMA:EXPERT_INDEX mappings; the encoder still receives EEG only.",
    )
    parser.add_argument("--router_supervision_weight", type=float, default=0.0)
    parser.add_argument("--task_group_supervision", default="",
                        help="Training only: schema:expert|expert,...; no inference task input")
    parser.add_argument("--task_group_supervision_weight", type=float, default=0.0)
    parser.add_argument("--router_supervision_decay_epochs", type=int, default=0)
    parser.add_argument(
        "--feature_router_supervision",
        default="",
        help="Optional training-only SCHEMA:ABSOLUTE_FEATURE_EXPERT_INDEX mappings.",
    )
    parser.add_argument("--feature_router_supervision_weight", type=float, default=0.0)
    parser.add_argument("--feature_router_supervision_decay_epochs", type=int, default=0)
    parser.add_argument(
        "--device_router_supervision",
        default="",
        help="Optional training-only SCHEMA:ABSOLUTE_DEVICE_EXPERT_INDEX mappings.",
    )
    parser.add_argument("--device_router_supervision_weight", type=float, default=0.0)
    parser.add_argument("--device_router_supervision_decay_epochs", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--router_lr",
        type=float,
        default=0.0,
        help="Optional Router-only learning rate; 0 reuses --lr.",
    )
    parser.add_argument("--weight_decay", type=float, default=5e-2)
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)
    parser.add_argument("--separate_router_gradient_clip", action="store_true")
    parser.add_argument("--router_gradient_clip_norm", type=float, default=5.0)
    parser.add_argument("--cpu_threads", type=int, default=4)
    parser.add_argument("--disable_amp", action="store_true")
    parser.add_argument(
        "--selection_metric",
        choices=("task_macro_bacc", "dataset_mean_bacc"),
        default="task_macro_bacc",
    )
    args = parser.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    torch.set_num_threads(int(args.cpu_threads))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    amp_enabled = not bool(args.disable_amp)

    datasets = csv_values(args.datasets)
    schema_map = parse_schema_map(args.schema_map)
    task_map = parse_schema_map(args.task_map)
    task_groups = {
        schema: [int(i) for i in indices.split("|")]
        for schema, indices in parse_schema_map(args.task_group_supervision).items()
    } if args.task_group_supervision else {}
    if args.task_group_supervision_weight < 0:
        raise ValueError("Task group loss weight must be nonnegative")
    if args.task_group_supervision_weight > 0 and not task_groups:
        raise ValueError("Task group targets required")
    if task_groups:
        n_experts = len(csv_values(args.expert_checkpoints))
        if set(task_groups) != set(schema_map.values()):
            raise ValueError("Task group targets must cover exactly all schemas")
        if any(not ids or len(ids) != len(set(ids)) or min(ids) < 0 or max(ids) >= n_experts
               for ids in task_groups.values()):
            raise ValueError("Invalid task group expert indices")
    if set(schema_map) != set(datasets):
        raise RuntimeError(f"schema map datasets {sorted(schema_map)} do not match {sorted(datasets)}")
    if set(task_map) != set(datasets):
        raise RuntimeError(f"task map datasets {sorted(task_map)} do not match {sorted(datasets)}")
    rows = load_rows(args.data_root, datasets, schema_map, task_map)
    feature_records: dict[str, dict[str, Any]] = {}
    feature_names: list[str] = []
    active_features_by_dataset: dict[str, list[str]] = {}
    if args.feature_manifest:
        feature_records = {
            str(row["id"]): row for row in read_jsonl(args.feature_manifest)
        }
        for dataset in datasets:
            dataset_rows = [
                row
                for split in ("train", "val", "test")
                for row in rows[f"{dataset}_{split}"]
            ]
            missing = [
                str(row["id"])
                for row in dataset_rows
                if str(row["id"]) not in feature_records
            ]
            if missing:
                raise RuntimeError(
                    f"feature manifest misses {dataset} rows: {missing[:3]}"
                )
            schemas_for_dataset = {
                tuple((feature_records[str(row["id"])].get("features") or {}).keys())
                for row in dataset_rows
            }
            if len(schemas_for_dataset) != 1 or not next(iter(schemas_for_dataset)):
                raise RuntimeError(
                    f"feature manifest has an inconsistent schema for {dataset}: "
                    f"{schemas_for_dataset}"
                )
            active_features_by_dataset[dataset] = list(next(iter(schemas_for_dataset)))
        feature_names = list(
            dict.fromkeys(
                name
                for dataset in datasets
                for name in active_features_by_dataset[dataset]
            )
        )
    if float(args.aux_feature_weight) > 0 and not feature_names:
        raise RuntimeError("--aux_feature_weight requires --feature_manifest")
    channels, patch_counts_by_dataset, labels_by_schema = validate_contract(
        rows,
        datasets,
        schema_map,
        int(args.patch_size),
        canonical_channel_scatter=bool(args.canonical_channel_scatter),
    )
    router_reference_patch_count = int(args.router_reference_patch_count) or max(
        patch_counts_by_dataset.values()
    )
    if min(patch_counts_by_dataset.values()) < int(args.anchor_time_count):
        raise RuntimeError(
            f"runtime patch counts {patch_counts_by_dataset} cannot provide "
            f"{args.anchor_time_count} anchors"
        )
    if router_reference_patch_count < int(args.anchor_time_count):
        raise RuntimeError(
            f"router reference has {router_reference_patch_count} patches but "
            f"needs {args.anchor_time_count} anchors"
        )
    schemas = sorted(labels_by_schema)
    schema_to_id = {schema: index for index, schema in enumerate(schemas)}
    router_supervision = (
        {schema: int(expert) for schema, expert in parse_schema_map(args.router_supervision).items()}
        if args.router_supervision.strip()
        else {}
    )
    feature_router_supervision = (
        {
            schema: int(expert)
            for schema, expert in parse_schema_map(args.feature_router_supervision).items()
        }
        if args.feature_router_supervision.strip()
        else {}
    )
    device_router_supervision = (
        {
            schema: int(expert)
            for schema, expert in parse_schema_map(args.device_router_supervision).items()
        }
        if args.device_router_supervision.strip()
        else {}
    )
    unknown_supervision = sorted(
        (
            set(router_supervision)
            | set(feature_router_supervision)
            | set(device_router_supervision)
        )
        - set(schemas)
    )
    invalid_experts = {
        schema: expert
        for schema, expert in {
            **router_supervision,
            **feature_router_supervision,
            **device_router_supervision,
        }.items()
        if expert < 0
    }
    if unknown_supervision or invalid_experts:
        raise RuntimeError(
            f"invalid router supervision: unknown_schemas={unknown_supervision}, "
            f"invalid_experts={invalid_experts}"
        )
    dataset_kwargs = {
        "schema_to_id": schema_to_id,
        "labels_by_schema": labels_by_schema,
        "patch_size": int(args.patch_size),
        "canonical_channel_scatter": bool(args.canonical_channel_scatter),
        "feature_records": feature_records,
        "feature_names": feature_names,
    }
    train_rows = [row for dataset in datasets for row in rows[f"{dataset}_train"]]
    teacher_targets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if args.vft_teacher_target_root:
        teacher_targets = load_vft_teacher_targets(
            args.vft_teacher_target_root,
            datasets,
            rows,
            labels_by_schema,
            schema_map,
        )
    elif float(args.vft_logit_kl_weight) > 0 or float(args.vft_hidden_cosine_weight) > 0:
        raise RuntimeError("VFT distillation weights require --vft_teacher_target_root")
    if teacher_targets and feature_names:
        raise RuntimeError("VFT distillation and auxiliary feature targets are not combined yet")
    train_weights = hierarchical_sampling_weights(train_rows, args.sampling_strategy)
    train_sampling_audit = sampling_mass_audit(
        train_rows, train_weights, args.sampling_strategy
    )
    if args.sampling_strategy == "natural_full_pass":
        train_batch_sampler: Sampler[list[int]] = LengthBucketFullPassBatchSampler(
            train_rows,
            int(args.batch_size),
            int(args.seed),
        )
        examples_per_epoch = len(train_rows)
        replacement = False
    else:
        train_batch_sampler = LengthBucketBatchSampler(
            train_rows,
            train_weights,
            int(args.batch_size),
            int(args.steps_per_epoch),
            int(args.seed),
        )
        examples_per_epoch = int(args.batch_size) * int(args.steps_per_epoch)
        replacement = True
    train_sampling_audit.update(
        {
            "eligible_rows": len(train_rows),
            "replacement": replacement,
            "complete_pass": not replacement,
            "batches_per_epoch": len(train_batch_sampler),
            "examples_per_epoch": examples_per_epoch,
        }
    )
    train_num_workers = int(args.train_num_workers)
    train_loader = DataLoader(
        TaskRows(
            train_rows,
            **dataset_kwargs,
            teacher_targets=teacher_targets,
            teacher_target_width=max(len(labels) for labels in labels_by_schema.values()),
        ),
        batch_sampler=train_batch_sampler,
        num_workers=train_num_workers,
        pin_memory=True,
        persistent_workers=train_num_workers > 0,
    )
    val_loaders = {
        dataset: DataLoader(
            TaskRows(
                rows[f"{dataset}_val"][: int(args.eval_limit)]
                if args.eval_limit
                else rows[f"{dataset}_val"],
                **dataset_kwargs,
            ),
            batch_sampler=LengthBucketFullPassBatchSampler(
                rows[f"{dataset}_val"][: int(args.eval_limit)]
                if args.eval_limit
                else rows[f"{dataset}_val"],
                int(args.eval_batch_size),
                int(args.seed) + 10_000,
            ),
            num_workers=0,
            pin_memory=True,
        )
        for dataset in datasets
    }
    test_loaders = {
        dataset: DataLoader(
            TaskRows(
                rows[f"{dataset}_test"][: int(args.test_limit)]
                if args.test_limit
                else rows[f"{dataset}_test"],
                **dataset_kwargs,
            ),
            batch_sampler=LengthBucketFullPassBatchSampler(
                rows[f"{dataset}_test"][: int(args.test_limit)]
                if args.test_limit
                else rows[f"{dataset}_test"],
                int(args.eval_batch_size),
                int(args.seed) + 20_000,
            ),
            num_workers=0,
            pin_memory=True,
        )
        for dataset in datasets
    }

    model = TaskAwareRouterClassifier(
        schema_to_id=schema_to_id,
        labels_by_schema=labels_by_schema,
        mode=args.mode,
        expert_checkpoints=[Path(value) for value in csv_values(args.expert_checkpoints)],
        channels=channels,
        patch_count=router_reference_patch_count,
        patch_size=int(args.patch_size),
        compressed_dim=int(args.compressed_dim),
        anchor_time_count=int(args.anchor_time_count),
        d_model=int(args.d_model),
        dim_feedforward=int(args.dim_feedforward),
        layers=int(args.layers),
        heads=int(args.heads),
        head_hidden=int(args.head_hidden),
        head_bottleneck=int(args.head_bottleneck),
        router_hidden=int(args.router_hidden),
        router_temperature=float(args.router_temperature),
        router_jitter=float(args.router_jitter),
        dropout=float(args.dropout),
        lowdim_adapter_dim=int(args.lowdim_adapter_dim),
        task_expert_count=int(args.task_expert_count),
        task_expert_indices=(
            [int(value) for value in csv_values(args.task_expert_indices)]
            if args.task_expert_indices.strip()
            else None
        ),
        feature_expert_count=int(args.feature_expert_count),
        device_expert_count=int(args.device_expert_count),
        task_bank_weight=float(args.task_bank_weight),
        feature_bank_weight=float(args.feature_bank_weight),
        device_bank_weight=float(args.device_bank_weight),
        router_top_k=int(args.router_top_k),
        task_router_top_k=int(args.task_router_top_k),
        feature_router_top_k=int(args.feature_router_top_k),
        device_router_top_k=int(args.device_router_top_k),
        router_input_mode=str(args.router_input_mode),
        router_query_count=int(args.router_query_count),
        router_query_heads=int(args.router_query_heads),
        router_reference_checkpoint=args.router_reference_checkpoint,
        feature_names=feature_names,
    ).to(device)
    if args.identity_token_compressor:
        compressor = model.encoder.token_compressor
        if not isinstance(compressor, nn.Linear) or compressor.in_features != compressor.out_features:
            raise RuntimeError(
                "--identity_token_compressor requires a square Linear token compressor"
            )
        with torch.no_grad():
            nn.init.eye_(compressor.weight)
    warmstart_report = None
    if args.warmstart:
        head_schema = args.warmstart_head_schema.strip() or None
        if head_schema and head_schema not in schema_to_id:
            raise RuntimeError(f"unknown warmstart head schema: {head_schema}")
        warmstart_report = model.load_v1_warmstart(
            args.warmstart, head_schema, skip_experts=bool(args.warmstart_skip_experts)
        )
        missing_names = warmstart_report["missing_key_names"]
        if args.allow_router_input_extension:
            if args.router_input_mode not in {"stats_spectral", "stats_encoder"}:
                raise ValueError("Extension allowance requires an extended input mode")
            missing_names = [n for n in missing_names if not n.startswith(
                ("encoder.router.extra.", "encoder.router_reference."))]
        if args.strict_warmstart and (
            missing_names or warmstart_report["unexpected_keys"]
            or warmstart_report["expanded_keys"] or warmstart_report["remapped_keys"]
        ):
            raise RuntimeError(f"Strict continuation requires identical checkpoint: {warmstart_report}")
    expert_lora_report = inject_expert_lora(
        model.encoder.experts,
        last_n_layers=int(args.expert_lora_last_n_layers),
        rank=int(args.expert_lora_rank),
        alpha=float(args.expert_lora_alpha),
        dropout=float(args.expert_lora_dropout),
    )
    if args.freeze_experts:
        for parameter in model.encoder.experts.parameters():
            parameter.requires_grad = False
        if int(args.expert_lora_last_n_layers) > 0:
            for name, parameter in model.encoder.experts.named_parameters():
                if "lora_a." in name or "lora_b." in name:
                    parameter.requires_grad = True
    if list(inspect.signature(model.encoder.encode).parameters) != ["x", "channel_indices"]:
        raise RuntimeError("encoder contract must be encode(eeg, electrode_indices=None)")
    if list(inspect.signature(model.forward).parameters) != ["x", "schema_ids", "channel_indices"]:
        raise RuntimeError(
            "task model contract must be forward(eeg, label_schema, electrode_indices=None)"
        )
    out_of_range_experts = {
        schema: expert
        for schema, expert in {
            **router_supervision,
            **feature_router_supervision,
            **device_router_supervision,
        }.items()
        if expert >= model.num_experts
    }
    if out_of_range_experts:
        raise RuntimeError(f"router supervision expert index out of range: {out_of_range_experts}")
    wrong_task_bank = {
        schema: expert
        for schema, expert in router_supervision.items()
        if expert
        not in (
            model.encoder.task_expert_indices
            if model.encoder.task_expert_indices
            else tuple(range(model.num_experts))
        )
    }
    wrong_device_bank = {
        schema: expert
        for schema, expert in device_router_supervision.items()
        if expert not in model.encoder.device_expert_indices
    }
    wrong_feature_bank = {
        schema: expert
        for schema, expert in feature_router_supervision.items()
        if expert not in model.encoder.feature_expert_indices
    }
    if wrong_task_bank or wrong_feature_bank or wrong_device_bank:
        raise RuntimeError(
            "router supervision targets the wrong bank: "
            f"task={wrong_task_bank}, feature={wrong_feature_bank}, "
            f"device={wrong_device_bank}"
        )

    router_parameters = [
        parameter
        for parameter in ([] if model.encoder.router is None else model.encoder.router.parameters())
        if parameter.requires_grad
    ]
    router_parameter_ids = {id(parameter) for parameter in router_parameters}
    other_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in router_parameter_ids
    ]
    router_lr = float(args.router_lr) if float(args.router_lr) > 0 else float(args.lr)
    optimizer_groups = [{"params": other_parameters, "lr": float(args.lr)}]
    if router_parameters:
        optimizer_groups.append({"params": router_parameters, "lr": router_lr})
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        weight_decay=float(args.weight_decay),
    )
    loss_fn = nn.CrossEntropyLoss()
    feature_weights = torch.empty((0, 3), dtype=torch.float32, device=device)
    if feature_names:
        feature_counts = torch.zeros((len(feature_names), 3), dtype=torch.float32)
        for row in train_rows:
            values = feature_records[str(row["id"])].get("features") or {}
            for feature_index, name in enumerate(feature_names):
                if name in values:
                    feature_counts[feature_index, FEATURE_BIN_TO_ID[str(values[name])]] += 1
        feature_weights = feature_counts.sum(dim=1, keepdim=True) / feature_counts.clamp_min(1.0)
        feature_weights = (
            feature_weights / feature_weights.mean(dim=1, keepdim=True)
        ).clamp_max(5.0).to(device)
    run_dir = args.out_root / args.run_name
    best_score = -math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses: list[float] = []
        route_means: list[torch.Tensor] = []
        route_loss_values: dict[str, list[float]] = defaultdict(list)
        gradient_values: dict[str, list[float]] = defaultdict(list)
        feature_loss_values: list[float] = []
        distill_logit_values: list[float] = []
        distill_hidden_values: list[float] = []
        for step, batch in enumerate(train_loader, 1):
            if teacher_targets:
                teacher_logits = batch[-2]
                teacher_hidden = batch[-1]
                batch = batch[:-2]
            else:
                teacher_logits = None
                teacher_hidden = None
            x, y, schema_ids = batch[:3]
            has_features = bool(feature_names)
            has_channels = len(batch) == (6 if has_features else 4)
            channel_indices = batch[3] if has_channels else None
            feature_targets = batch[-2] if has_features else None
            feature_mask = batch[-1] if has_features else None
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                if teacher_targets:
                    logits, student_hidden, route = model.forward_distill(
                        x.to(device, non_blocking=True),
                        schema_ids.to(device, non_blocking=True),
                        None
                        if channel_indices is None
                        else channel_indices.to(device, non_blocking=True),
                    )
                elif has_features:
                    logits, feature_logits, route = model.forward_with_features(
                        x.to(device, non_blocking=True),
                        schema_ids.to(device, non_blocking=True),
                        None
                        if channel_indices is None
                        else channel_indices.to(device, non_blocking=True),
                    )
                else:
                    logits, route = model(
                        x.to(device, non_blocking=True),
                        schema_ids.to(device, non_blocking=True),
                        None
                        if channel_indices is None
                        else channel_indices.to(device, non_blocking=True),
                    )
                loss = loss_fn(logits, y.to(device, non_blocking=True))
                if teacher_targets:
                    assert teacher_logits is not None and teacher_hidden is not None
                    temperature = float(args.vft_distill_temperature)
                    logit_kl = F.kl_div(
                        F.log_softmax(logits.float() / temperature, dim=-1),
                        F.softmax(
                            teacher_logits.to(device, non_blocking=True).float() / temperature,
                            dim=-1,
                        ),
                        reduction="batchmean",
                    ) * (temperature * temperature)
                    hidden_cosine = (
                        1.0
                        - F.cosine_similarity(
                            student_hidden.float(),
                            teacher_hidden.to(device, non_blocking=True).float(),
                            dim=-1,
                        )
                    ).mean()
                    loss = loss + float(args.vft_logit_kl_weight) * logit_kl
                    loss = loss + float(args.vft_hidden_cosine_weight) * hidden_cosine
                    distill_logit_values.append(float(logit_kl.detach().cpu()))
                    distill_hidden_values.append(float(hidden_cosine.detach().cpu()))
                if has_features and float(args.aux_feature_weight) > 0:
                    assert feature_targets is not None and feature_mask is not None
                    aux_loss = auxiliary_feature_loss(
                        feature_logits,
                        feature_targets.to(device, non_blocking=True),
                        feature_mask.to(device, non_blocking=True),
                        feature_weights,
                    )
                    loss = loss + float(args.aux_feature_weight) * aux_loss
                    feature_loss_values.append(float(aux_loss.detach().cpu()))
                if args.mode in {
                    "soft_moe",
                    "hybrid_moe",
                    "two_bank_moe",
                    "three_bank_moe",
                }:
                    importance = route.mean(dim=0)
                    balance_target = model.encoder.routing_balance_target(
                        device=importance.device, dtype=importance.dtype
                    )
                    balance = (importance - balance_target).square().mean()
                    entropy = -(route * torch.log(route.clamp_min(1e-8))).sum(dim=-1).mean()
                    loss = loss + float(args.router_balance_weight) * balance
                    loss = loss + float(args.router_entropy_weight) * entropy
                    raw_router_logits = model.encoder.last_router_logits
                    if raw_router_logits is None:
                        raise RuntimeError("MoE forward pass did not expose pre-hard router logits")
                    if task_groups and args.task_group_supervision_weight > 0:
                        group_loss = supervised_task_group_loss(
                            raw_router_logits, schema_ids.to(device), schema_to_id, task_groups
                        )
                        loss = loss + args.task_group_supervision_weight * group_loss
                        route_loss_values["task_group_ce"].append(float(group_loss.detach().cpu()))
                    if router_supervision and float(args.router_supervision_weight) > 0:
                        decay_epochs = int(args.router_supervision_decay_epochs)
                        decay = (
                            max(0.0, 1.0 - float(epoch - 1) / float(decay_epochs))
                            if decay_epochs > 0
                            else 1.0
                        )
                        route_loss = supervised_bank_loss(
                            logits=raw_router_logits,
                            schema_ids=schema_ids.to(device, non_blocking=True),
                            schema_to_id=schema_to_id,
                            supervision=router_supervision,
                            bank_indices=(
                                tuple(model.encoder.task_expert_indices)
                                if model.encoder.task_expert_indices
                                else tuple(range(model.num_experts))
                            ),
                        )
                        if route_loss is not None and decay > 0:
                            loss = loss + float(args.router_supervision_weight) * decay * route_loss
                            route_loss_values["task_ce"].append(float(route_loss.detach().cpu()))
                    if (
                        feature_router_supervision
                        and float(args.feature_router_supervision_weight) > 0
                    ):
                        decay_epochs = int(args.feature_router_supervision_decay_epochs)
                        decay = (
                            max(0.0, 1.0 - float(epoch - 1) / float(decay_epochs))
                            if decay_epochs > 0
                            else 1.0
                        )
                        feature_route_loss = supervised_bank_loss(
                            logits=raw_router_logits,
                            schema_ids=schema_ids.to(device, non_blocking=True),
                            schema_to_id=schema_to_id,
                            supervision=feature_router_supervision,
                            bank_indices=tuple(model.encoder.feature_expert_indices),
                        )
                        if feature_route_loss is not None and decay > 0:
                            loss = (
                                loss
                                + float(args.feature_router_supervision_weight)
                                * decay
                                * feature_route_loss
                            )
                            route_loss_values["feature_ce"].append(
                                float(feature_route_loss.detach().cpu())
                            )
                    if (
                        device_router_supervision
                        and float(args.device_router_supervision_weight) > 0
                    ):
                        decay_epochs = int(args.device_router_supervision_decay_epochs)
                        decay = (
                            max(0.0, 1.0 - float(epoch - 1) / float(decay_epochs))
                            if decay_epochs > 0
                            else 1.0
                        )
                        device_route_loss = supervised_bank_loss(
                            logits=raw_router_logits,
                            schema_ids=schema_ids.to(device, non_blocking=True),
                            schema_to_id=schema_to_id,
                            supervision=device_router_supervision,
                            bank_indices=tuple(model.encoder.device_expert_indices),
                        )
                        if device_route_loss is not None and decay > 0:
                            loss = (
                                loss
                                + float(args.device_router_supervision_weight)
                                * decay
                                * device_route_loss
                            )
                            route_loss_values["device_ce"].append(
                                float(device_route_loss.detach().cpu())
                            )
            loss.backward()
            if isinstance(model.encoder.router, nn.ModuleDict):
                for bank, router in model.encoder.router.items():
                    gradient_values[f"{bank}_router_pre_clip"].append(
                        gradient_norm(router.parameters())
                    )
            if bool(args.separate_router_gradient_clip) and router_parameters:
                router_norm = torch.nn.utils.clip_grad_norm_(
                    router_parameters, float(args.router_gradient_clip_norm)
                )
                other_norm = torch.nn.utils.clip_grad_norm_(
                    other_parameters, float(args.gradient_clip_norm)
                )
                gradient_values["router_group_pre_clip"].append(float(router_norm))
                gradient_values["other_group_pre_clip"].append(float(other_norm))
                gradient_values["router_group_clip_scale"].append(
                    min(
                        1.0,
                        float(args.router_gradient_clip_norm) / max(float(router_norm), 1e-12),
                    )
                )
            else:
                total_gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(args.gradient_clip_norm)
                )
                gradient_values["global_pre_clip"].append(float(total_gradient_norm))
                gradient_values["global_clip_scale"].append(
                    min(
                        1.0,
                        float(args.gradient_clip_norm) / max(float(total_gradient_norm), 1e-12),
                    )
                )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            route_means.append(route.detach().float().mean(dim=0).cpu())
            if step % 50 == 0:
                print(
                    f"[train] {args.run_name} epoch={epoch} step={step}/{len(train_loader)} "
                    f"loss={mean(losses[-50:]):.4f}",
                    flush=True,
                )
        validation = evaluate(
            model,
            val_loaders,
            schema_map,
            task_map,
            labels_by_schema,
            device,
            amp_enabled,
            feature_names,
        )
        scores = aggregate_scores(validation)
        score = float(scores[str(args.selection_metric)])
        history.append(
            {
                "epoch": epoch,
                "loss": mean(losses),
                "train_route_mean": [float(value) for value in torch.stack(route_means).mean(dim=0).tolist()],
                "train_router_losses": {
                    name: mean(values) for name, values in route_loss_values.items()
                },
                "train_gradient_norms": {
                    name: mean(values) for name, values in gradient_values.items()
                },
                "train_aux_feature_loss": (
                    mean(feature_loss_values) if feature_loss_values else None
                ),
                "train_vft_logit_kl": (
                    mean(distill_logit_values) if distill_logit_values else None
                ),
                "train_vft_hidden_cosine": (
                    mean(distill_hidden_values) if distill_hidden_values else None
                ),
                "val_scores": scores,
                "val": validation,
            }
        )
        print(
            f"[val] {args.run_name} epoch={epoch} {args.selection_metric}={score:.4f} "
            f"per_dataset={ {name: round(value['metrics']['balanced_accuracy'], 4) for name, value in validation.items()} }",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            stale = 0
            run_dir.mkdir(parents=True, exist_ok=True)
            state = (
                {
                    name: parameter.detach().cpu()
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad
                }
                if args.checkpoint_trainable_only
                else model.state_dict()
            )
            torch.save(
                {
                    "model": state,
                    "args": serializable_args(args),
                    "trainable_only": bool(args.checkpoint_trainable_only),
                },
                run_dir / "best.pt",
            )
        else:
            stale += 1
            if stale >= int(args.patience):
                break

    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(
        checkpoint["model"], strict=not bool(checkpoint.get("trainable_only"))
    )
    del checkpoint
    test = evaluate(
        model,
        test_loaders,
        schema_map,
        task_map,
        labels_by_schema,
        device,
        amp_enabled,
        feature_names,
    )
    test_scores = aggregate_scores(test)
    summary = {
        "run_name": args.run_name,
        "status": "complete",
        "args": serializable_args(args),
        "datasets": datasets,
        "schema_map": schema_map,
        "task_map": task_map,
        "labels_by_schema": labels_by_schema,
        "feature_names": feature_names,
        "active_features_by_dataset": active_features_by_dataset,
        "input_shape": [channels, router_reference_patch_count, int(args.patch_size)],
        "runtime_input_shapes": {
            dataset: [
                int(rows[f"{dataset}_train"][0]["input_shape"][0]),
                patch_counts_by_dataset[dataset],
                int(args.patch_size),
            ]
            for dataset in datasets
        },
        "router_reference_patch_count": router_reference_patch_count,
        "encoder_input_contract": (
            ["eeg", "electrode_indices"]
            if args.canonical_channel_scatter
            else ["eeg"]
        ),
        "canonical_channels": list(CANONICAL_CHANNELS) if args.canonical_channel_scatter else None,
        "output_input_contract": ["label_schema"],
        "dataset_id_visible_to_encoder": False,
        "experts_frozen": bool(args.freeze_experts),
        "expert_lora": expert_lora_report,
        "vft_teacher_targets": (
            str(args.vft_teacher_target_root) if args.vft_teacher_target_root else None
        ),
        "expert_reports": model.encoder.expert_reports,
        "warmstart_report": warmstart_report,
        "parameters": {
            "total": sum(parameter.numel() for parameter in model.parameters()),
            "encoder_experts": sum(parameter.numel() for parameter in model.encoder.experts.parameters()),
            "router": sum(parameter.numel() for parameter in model.encoder.router.parameters()) if model.encoder.router else 0,
            "compressor": sum(parameter.numel() for parameter in model.encoder.token_compressor.parameters()),
            "lowdim_adapters": sum(
                parameter.numel() for parameter in model.encoder.lowdim_adapters.parameters()
            ),
            "heads": sum(parameter.numel() for parameter in model.heads.parameters()),
        },
        "rows": {
            split: {dataset: len(rows[f"{dataset}_{split}"]) for dataset in datasets}
            for split in ("train", "val", "test")
        },
        "train_sampling": train_sampling_audit,
        "best_epoch": best_epoch,
        "selection_metric": str(args.selection_metric),
        "best_val_selection_score": best_score,
        "best_val_task_macro_bacc": (
            best_score if args.selection_metric == "task_macro_bacc" else None
        ),
        "test_scores": test_scores,
        "test": test,
        "history": history,
    }
    write_json(run_dir / "summary.json", summary)
    print(
        json.dumps(
            {
                "run_name": args.run_name,
                "selection_metric": str(args.selection_metric),
                "best_val_selection_score": best_score,
                "test_scores": test_scores,
                "test_per_dataset": {
                    dataset: value["metrics"]["balanced_accuracy"] for dataset, value in test.items()
                },
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
