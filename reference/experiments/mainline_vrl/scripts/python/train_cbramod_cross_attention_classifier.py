#!/usr/bin/env python3
"""Compare fixed-length resampling strategies for CBraMod classifiers."""

from __future__ import annotations

if __name__ == "__main__":
    raise SystemExit("Reference snapshot only. Use tools/workbench.py for inspection; configure a fresh run before training.")


import argparse
import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample_poly
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA_ROOT = Path(
    os.environ.get(
        "VSTATS_NATIVE200_ROOT",
        "/home/workspace/datasets/yuqiudeng/merged_native200_manifests_0722",
    )
)
CBRAMOD_ROOT = Path(os.environ.get("CBRAMOD_ROOT", "/home/workspace/code/DEFT/CBraMod"))
if str(CBRAMOD_ROOT) not in sys.path:
    sys.path.insert(0, str(CBRAMOD_ROOT))

from models.cbramod import CBraMod  # noqa: E402
from models.criss_cross_transformer import (  # noqa: E402
    TransformerEncoder as CrissCrossTransformerEncoder,
)
from models.criss_cross_transformer import (  # noqa: E402
    TransformerEncoderLayer as CrissCrossTransformerEncoderLayer,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def normalize_signal(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=1, keepdims=True)
    return x / (x.std(axis=1, keepdims=True) + 1e-6)


def transform_signal(x: np.ndarray, row: dict[str, Any]) -> tuple[np.ndarray, bool]:
    start = int(row.get("window_start") or 0)
    length = row.get("window_length")
    if length is not None:
        end = start + int(length)
        if start < 0 or end > int(x.shape[-1]):
            raise ValueError(f"invalid window [{start}:{end}] for {row['id']}: {tuple(x.shape)}")
        x = x[:, start:end]
    source_rate = int(row.get("source_sampling_rate") or row.get("sampling_rate") or 0)
    target_rate = int(row.get("sampling_rate") or source_rate)
    if (
        bool(row.get("resample_on_load"))
        and source_rate > 0
        and target_rate > 0
        and source_rate != target_rate
    ):
        divisor = math.gcd(source_rate, target_rate)
        x = resample_poly(
            x,
            target_rate // divisor,
            source_rate // divisor,
            axis=-1,
        ).astype(np.float32, copy=False)
    normalized = False
    if bool(row.get("normalize_before_padding")):
        x = normalize_signal(x)
        normalized = True
    pad_to_length = row.get("pad_to_length")
    if pad_to_length is not None:
        target = int(pad_to_length)
        missing = target - int(x.shape[-1])
        if missing < 0:
            raise ValueError(f"cannot pad {row['id']} from {x.shape[-1]} to {target}")
        left = missing // 2
        x = np.pad(x, ((0, 0), (left, missing - left)), mode="constant")
    return x, normalized


class EEGRows(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, rows: list[dict[str, Any]], label_to_id: dict[str, int], patch_size: int):
        self.rows = rows
        self.label_to_id = label_to_id
        self.patch_size = int(patch_size)
        self._h5: dict[str, h5py.File] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _get_h5(self, path: str) -> h5py.File:
        if path not in self._h5:
            self._h5[path] = h5py.File(path, "r")
        return self._h5[path]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[int(index)]
        h5 = self._get_h5(str(row["h5_path"]))
        x = np.asarray(h5[str(row["subject"])]["X"][int(row["trial_idx"])], dtype=np.float32)
        x, normalized = transform_signal(x, row)
        if x.ndim != 2 or x.shape[-1] % self.patch_size:
            raise ValueError(f"{row['id']} shape {tuple(x.shape)} is not divisible into {self.patch_size}-sample patches")
        if not normalized:
            x = normalize_signal(x)
        x = x.reshape(x.shape[0], x.shape[1] // self.patch_size, self.patch_size)
        return torch.from_numpy(x), torch.tensor(self.label_to_id[str(row["label"])], dtype=torch.long)


class QueryResamplerLayer(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        normalized_query = self.query_norm(query)
        normalized_context = self.context_norm(context)
        update, _ = self.attention(normalized_query, normalized_context, normalized_context, need_weights=False)
        query = query + update
        return query + self.ff(self.ff_norm(query))


class LargeConcatHead(nn.Module):
    def __init__(self, input_dim: int, hidden: int, bottleneck: int, dropout: float, classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, bottleneck),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VariableTokenClassifier(nn.Module):
    @staticmethod
    def temporal_anchor_indices(
        patch_count: int,
        anchor_count: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        if anchor_count <= 0 or anchor_count > patch_count:
            raise ValueError(f"anchor_count={anchor_count} is invalid for patch_count={patch_count}")
        # Select the center of each interval in normalized time. This keeps the
        # output count fixed while adapting coordinates to the actual input.
        indices = [
            min(patch_count - 1, int((index + 0.5) * patch_count / anchor_count))
            for index in range(anchor_count)
        ]
        return torch.tensor(indices, dtype=torch.long, device=device)

    def __init__(
        self,
        channels: int,
        patch_count: int,
        patch_size: int,
        classes: int,
        variant: str,
        query_count: int,
        query_layers: int,
        pool_channels: int,
        pool_patches: int,
        d_model: int,
        dim_feedforward: int,
        backbone_layers: int,
        attention_heads: int,
        head_hidden: int,
        head_bottleneck: int,
        dropout: float,
        pretrained_backbone: Path,
        compressed_dim: int = 50,
        anchor_time_count: int = 2,
        anchor_layers: int = 1,
        anchor_attention_heads: int = 2,
        anchor_feedforward: int = 200,
    ):
        super().__init__()
        self.variant = str(variant)
        self.d_model = int(d_model)
        self.compressed_dim = int(compressed_dim)
        self.token_compressor: nn.Module | None = None
        self.anchor_context: nn.Module | None = None
        self.anchor_time_count = int(anchor_time_count)
        # Anchor coordinates are deterministic from patch_count, so checkpoints
        # do not need a new state-dict key and remain compatible with old runs.
        self.register_buffer("anchor_indices", torch.empty(0, dtype=torch.long), persistent=False)
        self.pool_shape = (int(pool_channels), int(pool_patches))
        pooled_tokens = self.pool_shape[0] * self.pool_shape[1]
        if pooled_tokens != int(query_count):
            raise ValueError(
                f"pool grid {self.pool_shape} produces {pooled_tokens} tokens, "
                f"but query_count={query_count}"
            )
        self.backbone = CBraMod(
            in_dim=patch_size,
            out_dim=d_model,
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            seq_len=patch_count,
            n_layer=backbone_layers,
            nhead=attention_heads,
        )
        state = torch.load(pretrained_backbone, map_location="cpu", weights_only=True)
        self.backbone.load_state_dict(state, strict=True)
        self.backbone.proj_out = nn.Identity()
        if self.variant == "concat":
            head_tokens = int(channels) * int(patch_count)
            head_feature_dim = d_model
            self.query_tokens = None
            self.resampler = None
        elif self.variant == "adaptive100":
            head_tokens = pooled_tokens
            head_feature_dim = d_model
            self.query_tokens = None
            self.resampler = None
        elif self.variant == "cross100":
            head_tokens = int(query_count)
            head_feature_dim = d_model
            self.query_tokens = nn.Parameter(torch.randn(1, int(query_count), d_model) * 0.02)
            self.resampler = nn.ModuleList(
                [QueryResamplerLayer(d_model, attention_heads, dropout) for _ in range(int(query_layers))]
            )
        elif self.variant == "pooled_cross100":
            head_tokens = pooled_tokens
            head_feature_dim = d_model
            self.query_tokens = nn.Parameter(torch.randn(1, pooled_tokens, d_model) * 0.02)
            self.resampler = nn.ModuleList(
                [QueryResamplerLayer(d_model, attention_heads, dropout) for _ in range(int(query_layers))]
            )
        elif self.variant == "token_linear":
            head_tokens = int(channels) * int(patch_count)
            head_feature_dim = self.compressed_dim
            self.query_tokens = None
            self.resampler = None
            self.token_compressor = nn.Linear(d_model, self.compressed_dim, bias=False)
        elif self.variant == "token_conv":
            head_tokens = int(channels) * int(patch_count)
            head_feature_dim = self.compressed_dim
            self.query_tokens = None
            self.resampler = None
            self.token_compressor = nn.Conv2d(
                d_model,
                self.compressed_dim,
                kernel_size=(1, 3),
                padding=(0, 1),
                bias=False,
            )
        elif self.variant in {"stride150", "crisscross150"}:
            if self.anchor_time_count <= 0 or self.anchor_time_count > int(patch_count):
                raise ValueError(
                    f"anchor_time_count={self.anchor_time_count} is invalid for patch_count={patch_count}"
                )
            self.anchor_indices = self.temporal_anchor_indices(
                int(patch_count),
                self.anchor_time_count,
            )
            head_tokens = int(channels) * self.anchor_time_count
            head_feature_dim = self.compressed_dim
            self.query_tokens = None
            self.resampler = None
            self.token_compressor = nn.Linear(d_model, self.compressed_dim, bias=False)
            if self.variant == "crisscross150":
                if self.compressed_dim % 2 or int(anchor_attention_heads) % 2:
                    raise ValueError("criss-cross dimensions and attention heads must both be even")
                half_dim = self.compressed_dim // 2
                half_heads = int(anchor_attention_heads) // 2
                if half_dim % half_heads:
                    raise ValueError(
                        f"compressed_dim/2={half_dim} must be divisible by anchor_attention_heads/2={half_heads}"
                    )
                anchor_layer = CrissCrossTransformerEncoderLayer(
                    d_model=self.compressed_dim,
                    nhead=int(anchor_attention_heads),
                    dim_feedforward=int(anchor_feedforward),
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,
                    activation=F.gelu,
                )
                self.anchor_context = CrissCrossTransformerEncoder(
                    anchor_layer,
                    num_layers=int(anchor_layers),
                    enable_nested_tensor=False,
                )
        else:
            raise ValueError(f"unsupported variant: {self.variant}")
        self.head = LargeConcatHead(
            head_tokens * head_feature_dim,
            head_hidden,
            head_bottleneck,
            dropout,
            classes,
        )
        self.encoder_output_tokens = int(channels) * int(patch_count)
        self.classifier_input_tokens = head_tokens
        self.classifier_input_dim = head_tokens * head_feature_dim

    def adaptive_pool(self, features: torch.Tensor) -> torch.Tensor:
        # Pool the channel/patch grid while preserving the feature dimension.
        pooled = F.adaptive_avg_pool2d(features.permute(0, 3, 1, 2), self.pool_shape)
        return pooled.permute(0, 2, 3, 1).reshape(features.shape[0], -1, self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_grid = self.backbone(x)
        features = feature_grid.reshape(x.shape[0], -1, self.d_model)
        if self.variant == "token_linear":
            features = self.token_compressor(feature_grid).reshape(x.shape[0], -1, self.compressed_dim)
        elif self.variant == "token_conv":
            compressed = self.token_compressor(feature_grid.permute(0, 3, 1, 2))
            features = compressed.permute(0, 2, 3, 1).reshape(x.shape[0], -1, self.compressed_dim)
        elif self.variant in {"stride150", "crisscross150"}:
            compressed_grid = self.token_compressor(feature_grid)
            if self.anchor_context is not None:
                compressed_grid = self.anchor_context(compressed_grid)
            anchor_indices = self.temporal_anchor_indices(
                int(compressed_grid.shape[2]),
                self.anchor_time_count,
                device=compressed_grid.device,
            )
            features = compressed_grid.index_select(2, anchor_indices)
            features = features.reshape(x.shape[0], -1, self.compressed_dim)
        elif self.variant == "adaptive100":
            features = self.adaptive_pool(feature_grid)
        elif self.variant == "pooled_cross100":
            query = self.adaptive_pool(feature_grid)
            query = query + self.query_tokens.expand(x.shape[0], -1, -1).to(dtype=features.dtype)
            for layer in self.resampler:
                query = layer(query, features)
            features = query
        elif self.query_tokens is not None and self.resampler is not None:
            query = self.query_tokens.expand(x.shape[0], -1, -1).to(dtype=features.dtype)
            for layer in self.resampler:
                query = layer(query, features)
            features = query
        return self.head(features)


def metrics(y_true: list[int], y_pred: list[int], labels: list[str]) -> dict[str, Any]:
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "prediction_histogram": {label: int(y_pred.count(index)) for index, label in enumerate(labels)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(labels)))).tolist(),
    }


def evaluate(model: nn.Module, loader: DataLoader, labels: list[str], device: torch.device, amp: bool) -> dict[str, Any]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.inference_mode():
        for x, y in loader:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                logits = model(x.to(device, non_blocking=True))
            y_true.extend(y.tolist())
            y_pred.extend(logits.argmax(dim=-1).cpu().tolist())
    return metrics(y_true, y_pred, labels)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--variant",
        choices=(
            "concat",
            "cross100",
            "adaptive100",
            "pooled_cross100",
            "token_linear",
            "token_conv",
            "stride150",
            "crisscross150",
        ),
        required=True,
    )
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--patch_size", type=int, default=200)
    parser.add_argument("--query_count", type=int, default=100)
    parser.add_argument("--query_layers", type=int, default=1)
    parser.add_argument("--pool_channels", type=int, default=25)
    parser.add_argument("--pool_patches", type=int, default=4)
    parser.add_argument("--compressed_dim", type=int, default=50)
    parser.add_argument("--anchor_time_count", type=int, default=2)
    parser.add_argument("--anchor_layers", type=int, default=1)
    parser.add_argument("--anchor_attention_heads", type=int, default=2)
    parser.add_argument("--anchor_feedforward", type=int, default=200)
    parser.add_argument("--d_model", type=int, default=200)
    parser.add_argument("--dim_feedforward", type=int, default=800)
    parser.add_argument("--backbone_layers", type=int, default=12)
    parser.add_argument("--attention_heads", type=int, default=8)
    parser.add_argument("--head_hidden", type=int, default=800)
    parser.add_argument("--head_bottleneck", type=int, default=200)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-2)
    parser.add_argument("--cpu_threads", type=int, default=4)
    parser.add_argument("--pretrained_backbone", type=Path, required=True)
    parser.add_argument("--disable_amp", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(int(args.cpu_threads))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    amp = not bool(args.disable_amp)

    data_dir = args.data_root / args.dataset
    rows = {split: read_jsonl(data_dir / f"{split}.jsonl") for split in ("train", "val", "test")}
    shapes = {tuple(int(value) for value in row["input_shape"]) for split_rows in rows.values() for row in split_rows}
    if len(shapes) != 1:
        raise RuntimeError(f"{args.dataset} mixes manifest input shapes: {sorted(shapes)}")
    channels, samples = next(iter(shapes))
    if samples % int(args.patch_size):
        raise RuntimeError(f"input length {samples} is not divisible by patch size {args.patch_size}")
    patch_count = samples // int(args.patch_size)
    labels = sorted({str(row["label"]) for split_rows in rows.values() for row in split_rows})
    label_to_id = {label: index for index, label in enumerate(labels)}
    datasets = {split: EEGRows(split_rows, label_to_id, int(args.patch_size)) for split, split_rows in rows.items()}
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True),
        "val": DataLoader(datasets["val"], batch_size=args.eval_batch_size, shuffle=False, num_workers=0, pin_memory=True),
        "test": DataLoader(datasets["test"], batch_size=args.eval_batch_size, shuffle=False, num_workers=0, pin_memory=True),
    }
    model = VariableTokenClassifier(
        channels=channels,
        patch_count=patch_count,
        patch_size=int(args.patch_size),
        classes=len(labels),
        variant=args.variant,
        query_count=int(args.query_count),
        query_layers=int(args.query_layers),
        pool_channels=int(args.pool_channels),
        pool_patches=int(args.pool_patches),
        d_model=int(args.d_model),
        dim_feedforward=int(args.dim_feedforward),
        backbone_layers=int(args.backbone_layers),
        attention_heads=int(args.attention_heads),
        head_hidden=int(args.head_hidden),
        head_bottleneck=int(args.head_bottleneck),
        dropout=float(args.dropout),
        pretrained_backbone=args.pretrained_backbone,
        compressed_dim=int(args.compressed_dim),
        anchor_time_count=int(args.anchor_time_count),
        anchor_layers=int(args.anchor_layers),
        anchor_attention_heads=int(args.anchor_attention_heads),
        anchor_feedforward=int(args.anchor_feedforward),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    run_dir = args.out_root / args.run_name
    best_score = -1.0
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses: list[float] = []
        for step, (x, y) in enumerate(loaders["train"], 1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                logits = model(x.to(device, non_blocking=True))
                loss = F.cross_entropy(logits, y.to(device, non_blocking=True))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if step % 50 == 0:
                print(f"train {args.run_name} epoch={epoch} step={step}/{len(loaders['train'])} loss={mean(losses[-50:]):.4f}", flush=True)
        validation = evaluate(model, loaders["val"], labels, device, amp)
        score = float(validation["balanced_accuracy"])
        history.append({"epoch": epoch, "loss": mean(losses), "val": validation})
        print(json.dumps({"run": args.run_name, "epoch": epoch, "val_bacc": score}), flush=True)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            stale = 0
            run_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "labels": labels, "args": serializable_args(args)}, run_dir / "best.pt")
        else:
            stale += 1
            if stale >= int(args.patience):
                break

    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    test = evaluate(model, loaders["test"], labels, device, amp)
    summary = {
        "run_name": args.run_name,
        "status": "complete",
        "dataset": args.dataset,
        "variant": args.variant,
        "args": serializable_args(args),
        "labels": labels,
        "manifest_input_shape": [channels, samples],
        "encoder_input_shape": [channels, patch_count, int(args.patch_size)],
        "encoder_output_tokens": model.encoder_output_tokens,
        "classifier_input_tokens": model.classifier_input_tokens,
        "classifier_input_dim": model.classifier_input_dim,
        "compressed_dim": (
            model.compressed_dim
            if args.variant in {"token_linear", "token_conv", "stride150", "crisscross150"}
            else None
        ),
        "anchor_indices": model.anchor_indices.tolist() if model.anchor_indices.numel() else None,
        "anchor_time_count": model.anchor_time_count if model.anchor_indices.numel() else None,
        "pool_shape": list(model.pool_shape),
        "train_rows": len(rows["train"]),
        "val_rows": len(rows["val"]),
        "test_rows": len(rows["test"]),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "encoder_parameters": sum(parameter.numel() for parameter in model.backbone.parameters()),
        "bridge_parameters": sum(parameter.numel() for parameter in model.parameters())
        - sum(parameter.numel() for parameter in model.backbone.parameters()),
        "best_epoch": best_epoch,
        "best_val_bacc": best_score,
        "test": test,
        "history": history,
    }
    write_json(run_dir / "summary.json", summary)
    print(json.dumps({"run": args.run_name, "test_bacc": test["balanced_accuracy"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
