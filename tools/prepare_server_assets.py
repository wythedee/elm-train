#!/usr/bin/env python3
"""Download assets on Linux, in parallel; no model execution or optimizer steps."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

REPO = Path(__file__).resolve().parents[1]
PREFIX = "releases/native200_20261101"


def safe_release_path(value):
    p = PurePosixPath(value)
    if p.is_absolute() or ".." in p.parts or not str(p).startswith(PREFIX + "/") or "\\" in value:
        raise ValueError("Unsafe dataset release path")
    return p


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def model_download(work, workers):
    from modelscope.hub.snapshot_download import snapshot_download
    requirements = json.loads((REPO / "metadata/asset_requirements.json").read_text())["files"]
    expected = {Path(k).name: v for k, v in requirements.items() if v["role"] == "model"}
    root = work / "models/Qwen3-4B"
    snapshot_download(model_id="Qwen/Qwen3-4B", revision="master", local_dir=str(root),
                      allow_patterns=list(expected), max_workers=workers)
    for name, item in expected.items():
        if digest(root / name) != item["sha256"]:
            raise ValueError("Original Qwen3-4B version mismatch: " + name)
    return {"model": str(root), "verified_model_files": len(expected)}


def components_download(work):
    root = work / "models/elm"
    subprocess.run([sys.executable, "-B", str(REPO / "tools/download_checkpoints.py"),
                    "--root", str(root)], check=True)
    return {"elm_weights": str(root / "releases/20261031")}


def dataset_download(work, workers, wait_seconds, stopped):
    from modelscope.hub.snapshot_download import dataset_snapshot_download
    marker = PREFIX + "/dataset_manifest.json"
    url = "https://www.modelscope.cn/api/v1/datasets/wythedee/elm-data/repo?" + urllib.parse.urlencode(
        {"Revision": "master", "FilePath": marker})
    deadline = time.monotonic() + wait_seconds
    while True:
        if stopped.is_set():
            raise RuntimeError("Another setup task failed; preserve completed downloads and rerun after repair")
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                inventory = json.load(response)
            if "files" not in inventory or len(inventory.get("datasets", {})) != 16:
                raise ValueError("The completed 16-source release is not available yet")
            break
        except (urllib.error.URLError, TimeoutError, ValueError):
            if time.monotonic() >= deadline:
                raise RuntimeError("Full dataset upload is not ready or cannot be reached; rerun setup later")
            print("Waiting for completed 16-source dataset release; other downloads continue", flush=True)
            time.sleep(min(60, max(1, deadline - time.monotonic())))
    files = inventory["files"]
    reference = json.loads((REPO / "metadata/native200_reference_counts.json").read_text())["datasets"]
    if set(inventory["datasets"]) != set(reference):
        raise ValueError("Dataset names differ from the declared 16-source version")
    for dataset, item in reference.items():
        if [inventory["datasets"][dataset][s] for s in ("train", "val", "test")] != item["records"]:
            raise ValueError("Published dataset version differs: " + dataset)
    seen = set()
    for item in files:
        name = str(safe_release_path(item["path"]))
        if name in seen or type(item["bytes"]) is not int or item["bytes"] < 0:
            raise ValueError("Invalid or duplicate dataset file entry")
        seen.add(name)
    root = work / "data"
    dataset_snapshot_download(dataset_id="wythedee/elm-data", revision="master", local_dir=str(root),
                              allow_patterns=[PREFIX + "/**"], max_workers=workers)
    for item in files:
        path = root / safe_release_path(item["path"])
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise ValueError("Missing/incomplete dataset file: " + item["path"])
    actual = {}
    for dataset, splits in inventory["datasets"].items():
        if Path(dataset).name != dataset or dataset in (".", ".."):
            raise ValueError("Invalid dataset name")
        actual[dataset] = {}
        for split in ("train", "val", "test"):
            path = root / PREFIX / "manifests" / dataset / (split + ".jsonl")
            with path.open() as stream:
                actual[dataset][split] = sum(bool(line.strip()) for line in stream)
            if actual[dataset][split] != splits[split]:
                raise ValueError("Dataset row count differs: " + dataset + "/" + split)
    return {"data_root": str(root / PREFIX), "manifest_root": str(root / PREFIX / "manifests"),
            "dataset_counts": actual, "dataset_file_sizes_checked": len(files),
            "dataset_full_SHA_repeated": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", required=True, type=Path)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--wait-data-seconds", type=int, default=21600)
    args = p.parse_args()
    if sys.platform != "linux":
        p.error("Actual asset downloads are restricted to the authorized Linux server")
    if args.workers < 1 or args.wait_data_seconds < 0:
        p.error("Invalid worker count or wait interval")
    work = args.work.expanduser().resolve()
    for name, path in {"MODELSCOPE_CACHE": work / "cache/modelscope", "HF_HOME": work / "cache/hf",
                       "TMPDIR": work / "tmp", "XDG_CACHE_HOME": work / "cache/xdg"}.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    result = {"training_started": False, "GPU_used": False, "work": str(work)}
    stopped = threading.Event()
    jobs = {"model": lambda: model_download(work, args.workers),
            "components": lambda: components_download(work),
            "data": lambda: dataset_download(work, args.workers, args.wait_data_seconds, stopped)}
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = {pool.submit(fn): name for name, fn in jobs.items()}
        for future in as_completed(pending):
            try:
                result.update(future.result())
            except BaseException as error:
                stopped.set()
                save(work / "setup/assets_error.json", {"phase": pending[future],
                     "exception_type": type(error).__name__, "time": time.time()})
                raise
            print("READY", pending[future], flush=True)
    result.update(status="assets_downloaded_and_checked", time=time.time())
    save(work / "setup/assets_ready.json", result)
    print(json.dumps({"status": result["status"], "manifest_root": result["manifest_root"]}), flush=True)


if __name__ == "__main__":
    main()
