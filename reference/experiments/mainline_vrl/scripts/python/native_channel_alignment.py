#!/usr/bin/env python3
"""Canonical post-encoder channel alignment for heterogeneous EEG montages."""

from __future__ import annotations

if __name__ == "__main__":
    raise SystemExit("Reference snapshot only. Use tools/workbench.py for inspection; configure a fresh run before training.")


from typing import Sequence

import torch


CANONICAL_CHANNELS = (
    "A1", "A2", "TP9", "TP10", "F9", "F10", "Fp1", "Fp2", "Fpz", "F7",
    "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "T3", "C3",
    "Cz", "C4", "T4", "CP5", "CP1", "CP2", "CP6", "T5", "P3", "Pz",
    "P4", "T6", "POz", "O1", "Oz", "O2", "AF7", "AF3", "AF4", "AF8",
    "F5", "F1", "F2", "F6", "FC3", "FCz", "FC4", "C5", "C1", "C2",
    "C6", "CP3", "CPz", "CP4", "P5", "P1", "P2", "P6", "PO5", "PO3",
    "PO4", "PO6", "FT7", "FT8", "TP7", "TP8", "PO7", "PO8", "FT9", "FT10",
    "PO9", "PO10", "P9", "P10", "AFz",
)

# Legacy 10-20 names and high-density half-step names are mapped to their
# nearest canonical electrode. The aliases are acquisition metadata, not a
# dataset or device identifier.
CHANNEL_ALIASES = {
    "T7": "T3",
    "T8": "T4",
    "P7": "T5",
    "P8": "T6",
    "M1": "A1",
    "M2": "A2",
    "HEOL": "A1",
    "HEOR": "A2",
    "CB1": "PO9",
    "CB2": "PO10",
    "AFP1": "Fp1",
    "AFP2": "Fp2",
    "AFF5H": "AF7",
    "AFF6H": "AF8",
    "AFF1H": "AF3",
    "AFF2H": "AF4",
    "FCC5H": "FC5",
    "FCC3H": "FC3",
    "FCC4H": "FC4",
    "FCC6H": "FC6",
    "CCP5H": "CP5",
    "CCP3H": "CP3",
    "CCP4H": "CP4",
    "CCP6H": "CP6",
    "PPO1H": "PO3",
    "PPO2H": "PO4",
    "POO1": "O1",
    "POO2": "O2",
    "FTT9H": "FT9",
    "FTT10H": "FT10",
    "TTP7H": "TP7",
    "TPP8H": "TP8",
    "TPP9H": "TP9",
    "TPP10H": "TP10",
}

_CANONICAL_INDEX = {name.upper(): index for index, name in enumerate(CANONICAL_CHANNELS)}


def canonical_channel_indices(channel_names: Sequence[str]) -> list[int]:
    """Map native channel names to canonical slots; unknown names become -1."""
    result: list[int] = []
    for value in channel_names:
        normalized = str(value).strip().upper()
        normalized = CHANNEL_ALIASES.get(normalized, normalized)
        result.append(_CANONICAL_INDEX.get(normalized.upper(), -1))
    return result


def scatter_channels(
    values: torch.Tensor,
    channel_indices: torch.Tensor,
    canonical_count: int = len(CANONICAL_CHANNELS),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter ``[B,C,...]`` values into canonical channels and return validity mask."""
    if values.ndim < 3:
        raise ValueError(f"channel scatter expects [B,C,...], got {tuple(values.shape)}")
    if channel_indices.ndim == 1:
        channel_indices = channel_indices.unsqueeze(0).expand(values.shape[0], -1)
    if tuple(channel_indices.shape) != tuple(values.shape[:2]):
        raise ValueError(
            f"channel index shape {tuple(channel_indices.shape)} does not match "
            f"values {tuple(values.shape[:2])}"
        )
    output = values.new_zeros((values.shape[0], int(canonical_count), *values.shape[2:]))
    counts = values.new_zeros((values.shape[0], int(canonical_count)))
    for batch_index in range(values.shape[0]):
        indices = channel_indices[batch_index].to(device=values.device, dtype=torch.long)
        valid = (indices >= 0) & (indices < int(canonical_count))
        if not bool(valid.any()):
            raise ValueError("sample has no channels mapped to the canonical montage")
        selected_indices = indices[valid]
        output[batch_index].index_add_(0, selected_indices, values[batch_index, valid])
        counts[batch_index].index_add_(
            0,
            selected_indices,
            torch.ones_like(selected_indices, dtype=counts.dtype),
        )
    divisor = counts.clamp_min(1).reshape(counts.shape[0], counts.shape[1], *([1] * (values.ndim - 2)))
    return output / divisor, counts > 0
