from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def config(path):
    path = Path(path).resolve(strict=True)
    cfg = read(path)
    for key in ("output", "manifest_root", "encoder_bundle", "model", "oracle", "schema", "thresholds", "features", "initial_projector", "evaluation_checkpoint", "debug_selection", "data_root", "tmpdir", "reference_label_features", "reference_statistics_features"):
        if cfg.get(key):
            p = Path(os.path.expandvars(cfg[key])).expanduser()
            cfg[key] = str((path.parent / p).resolve() if not p.is_absolute() else p.resolve())
    return cfg


def resolve_data_path(value, path_map=None, data_root=None):
    path = Path(value)
    for old, new in sorted((path_map or {}).items(), key=lambda kv: -len(kv[0])):
        try:
            suffix = path.relative_to(old)
        except ValueError:
            continue
        path = Path(new) / suffix
        break
    if not path.is_absolute():
        if data_root is None:
            raise ValueError("Relative data path requires data_root")
        path = Path(data_root) / path
    return path.resolve(strict=True)
