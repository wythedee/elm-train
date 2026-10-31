"""Portable Native200 path mapping and trusted LMDB/H5 loading."""
from __future__ import annotations
from pathlib import Path
import pickle
import h5py
import lmdb
import numpy as np
import torch
from . import signals
from .io import resolve_data_path


class SignalReader:
    def __init__(self, path_map=None, data_root=None):
        self.cache = {}
        self.path_map = path_map or {}
        self.data_root = Path(data_root) if data_root else None

    def resolve(self, row):
        row = dict(row)
        for field in ("lmdb_path", "h5_path"):
            if not row.get(field):
                continue
            row[field] = str(resolve_data_path(row[field], self.path_map, self.data_root))
        return row

    def source(self, row):
        row = self.resolve(row)
        if row.get("lmdb_path"):
            p = row["lmdb_path"]
            if p not in self.cache:
                self.cache[p] = lmdb.open(p, readonly=True, lock=False, readahead=False, meminit=False)
            with self.cache[p].begin(write=False) as txn:
                value = txn.get(str(row["lmdb_key"]).encode())
            if value is None:
                raise KeyError("Missing LMDB record " + row["id"])
            x = np.asarray(pickle.loads(value)["sample"], dtype=np.float32)
            if x.ndim != 3 or x.shape[-1] != 200:
                raise ValueError("LMDB sample must be channels×patches×200")
            x = x.reshape(x.shape[0], -1) * float(row.get("input_scale", 1.0))
            x, _ = signals.transform_signal(x, row)
            normalized = bool(row.get("skip_trial_normalization"))
        else:
            p = row["h5_path"]
            if p not in self.cache:
                self.cache[p] = h5py.File(p, "r")
            group = self.cache[p][str(row["subject"])]["X"]
            i, n = int(row["trial_idx"]), int(row.get("time_concat_count") or 1)
            chunks = np.asarray(group[i:i+n] if n > 1 else group[i], dtype=np.float32)
            x = chunks.transpose(1, 0, 2).reshape(chunks.shape[1], -1) if n > 1 else chunks
            x, normalized = signals.transform_signal(x, row)
        if x.ndim != 2 or x.shape[1] % 200 or not np.isfinite(x).all():
            raise ValueError("Invalid/non-finite transformed signal: " + row["id"])
        if not normalized:
            x = signals.normalize_signal(x)
        indices = row["canonical_channel_indices"]
        if len(indices) != x.shape[0]:
            raise ValueError("Channel map differs from waveform")
        return (torch.from_numpy(x.reshape(x.shape[0], -1, 200)),
                torch.tensor(indices, dtype=torch.long), row)

    def statistics(self, row):
        row = self.resolve(row)
        x, info = signals.load_signal(row, self.cache)
        return signals.numeric(x), info

    def close(self):
        for obj in self.cache.values():
            obj.close()
        self.cache.clear()


class FeatureDataset:
    """Index of independent shards; no requirement to hold the full corpus in RAM."""
    def __init__(self, path, split):
        import json
        self.root = Path(path).resolve(strict=True)
        meta = json.loads((self.root / (split + "_index.json")).read_text())
        self.records = meta["records"]
        self._path = None
        self._shard = None

    def inputs(self, index, device):
        row = self.records[index]
        path = (self.root / row["shard"]).resolve(strict=True)
        if self.root not in path.parents:
            raise ValueError("Feature shard escaped its root")
        if self._path != path:
            self._shard = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            self._path = path
        i = row["shard_index"]
        return tuple(self._shard[k][i:i+1].to(device) for k in
                     ("label_bank", "label_valid", "statistics_bank", "statistics_valid"))
