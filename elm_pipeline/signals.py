from __future__ import annotations
import math,pickle
from pathlib import Path
from typing import Any
import h5py,lmdb,numpy as np
import sys
try:
    __import__("numpy._core.numeric")
except ModuleNotFoundError:
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
from scipy.signal import resample_poly,welch
def normalize_signal(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=1, keepdims=True)
    return x / (x.std(axis=1, keepdims=True) + 1e-06)

def transform_signal(x: np.ndarray, row: dict[str, Any]) -> tuple[np.ndarray, bool]:
    start = int(row.get('window_start') or 0)
    length = row.get('window_length')
    if length is not None:
        end = start + int(length)
        if start < 0 or end > int(x.shape[-1]):
            raise ValueError(f"invalid window [{start}:{end}] for {row['id']}: {tuple(x.shape)}")
        x = x[:, start:end]
    source_rate = int(row.get('source_sampling_rate') or row.get('sampling_rate') or 0)
    target_rate = int(row.get('sampling_rate') or source_rate)
    if bool(row.get('resample_on_load')) and source_rate > 0 and (target_rate > 0) and (source_rate != target_rate):
        divisor = math.gcd(source_rate, target_rate)
        x = resample_poly(x, target_rate // divisor, source_rate // divisor, axis=-1).astype(np.float32, copy=False)
    normalized = False
    if bool(row.get('normalize_before_padding')):
        x = normalize_signal(x)
        normalized = True
    pad_to_length = row.get('pad_to_length')
    if pad_to_length is not None:
        target = int(pad_to_length)
        missing = target - int(x.shape[-1])
        if missing < 0:
            raise ValueError(f"cannot pad {row['id']} from {x.shape[-1]} to {target}")
        left = missing // 2
        x = np.pad(x, ((0, 0), (left, missing - left)), mode='constant')
    return (x, normalized)
def load_signal(row, cache):
    if row.get('lmdb_path'):
        p = row['lmdb_path']
        if p not in cache:
            cache[p] = lmdb.open(p, readonly=True, lock=False, readahead=False, meminit=False)
        with cache[p].begin() as t:
            v = t.get(str(row['lmdb_key']).encode())
        if v is None:
            raise KeyError(row['id'])
        x = np.asarray(pickle.loads(v)['sample'], dtype=np.float32)
        x = x.reshape(x.shape[0], -1) * float(row.get('input_scale', 1))
    else:
        p = row['h5_path']
        if p not in cache:
            cache[p] = h5py.File(p, 'r')
        g = cache[p][str(row['subject'])]['X']
        i = int(row['trial_idx'])
        n = int(row.get('time_concat_count') or 1)
        if n > 1:
            z = np.asarray(g[i:i + n], dtype=np.float32)
            x = z.transpose(1, 0, 2).reshape(z.shape[1], -1)
        else:
            x = np.asarray(g[i], dtype=np.float32)
    start = int(row.get('window_start') or 0)
    length = row.get('window_length')
    if length is not None:
        assert 0 <= start and start + int(length) <= x.shape[-1], row['id']
        x = x[:, start:start + int(length)]
    sr = int(row.get('source_sampling_rate') or row.get('sampling_rate'))
    tr = int(row['sampling_rate'])
    assert tr == 200, (row['id'], tr)
    if row.get('resample_on_load') and sr != tr:
        d = math.gcd(sr, tr)
        x = resample_poly(x, tr // d, sr // d, axis=-1).astype(np.float32)
    allzero = np.all(x == 0, axis=0)
    nz = np.flatnonzero(~allzero)
    assert len(nz) > 4, 'empty/constant EEG'
    (l, r) = (int(nz[0]), int(nz[-1]) + 1)
    x = x[:, l:r]
    indices = row.get('canonical_channel_indices')
    channel_valid = np.isfinite(x).all(1) & (x.std(1) > 1e-08)
    if indices is not None:
        assert len(indices) == x.shape[0]
        channel_valid &= (np.asarray(indices) >= 0) & (np.asarray(indices) < 75)
    x = x[channel_valid]
    assert len(x) > 0 and x.shape[-1] >= 100, 'no valid channels/short window'
    x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-06)
    info = {'valid_channel_count': len(x), 'source_channel_count': len(channel_valid), 'valid_channel_indices': np.flatnonzero(channel_valid).tolist(), 'valid_samples': x.shape[-1], 'zero_margin_left': l, 'zero_margin_right': int(len(allzero) - r), 'declared_pad_to_length': row.get('pad_to_length'), 'sampling_rate': 200}
    return (x.astype(np.float64), info)

def numeric(x):
    (freq, p) = welch(x, fs=200, nperseg=min(400, x.shape[-1]), noverlap=min(200, min(400, x.shape[-1]) // 2), axis=-1)
    p = p.mean(0)
    valid = (freq >= 1) & (freq < 45)
    total = p[valid].sum() + 1e-12

    def band(a, b):
        return p[(freq >= a) & (freq < b)].sum() / total
    dist = p[valid] / total
    dx = np.diff(x, axis=-1)
    dd = np.diff(dx, axis=-1)
    vx = x.var(1) + 1e-12
    vd = dx.var(1) + 1e-12
    mobility = np.sqrt(vd / vx)
    complexity = np.sqrt((dd.var(1) + 1e-12) / vd) / (mobility + 1e-12)
    out = np.array([band(8, 13), band(4, 8) / (band(13, 30) + 1e-08), -(dist * np.log(dist + 1e-12)).sum() / np.log(len(dist)), mobility.mean(), complexity.mean(), np.abs(dx).mean()])
    assert np.isfinite(out).all()
    return out
