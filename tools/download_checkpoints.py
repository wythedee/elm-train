#!/usr/bin/env python3
"""Download the published ELM components on a Linux server; no model execution."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

REPO = Path(__file__).resolve().parents[1]
COMMON = {"data_schema.json", "thresholds.json", "channels.json"}
COMPONENTS = ("encoders", "oracle", "projector")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def selected_files(catalog, components):
    if catalog["repo_id"] != "wythedee/elm":
        raise ValueError("This downloader accepts only the declared ELM repository")
    for rel, info in catalog["files"].items():
        path = PurePosixPath(rel)
        if (path.is_absolute() or ".." in path.parts or path.suffix not in (".json", ".safetensors")
                or type(info["bytes"]) is not int or info["bytes"] <= 0
                or not re.fullmatch(r"[a-f0-9]{64}", info["sha256"])):
            raise ValueError("Invalid release entry: " + rel)
        if rel not in COMMON and path.parts[0] not in COMPONENTS:
            raise ValueError("Unexpected release component: " + rel)
    return {rel: info for rel, info in catalog["files"].items()
            if rel in COMMON or PurePosixPath(rel).parts[0] in components}


def verified(path, info):
    return path.is_file() and path.stat().st_size == info["bytes"] and digest(path) == info["sha256"]


def download(url, target, info, reserve):
    if target.exists():
        if verified(target, info):
            return "already_verified"
        raise ValueError("Existing file differs; preserved without overwriting: " + str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    if partial.is_symlink():
        raise ValueError("Refusing a symlinked partial file")
    for attempt in range(3):
        start = partial.stat().st_size if partial.exists() else 0
        if start > info["bytes"]:
            raise ValueError("Oversized partial retained: " + str(partial))
        if start == info["bytes"]:
            break
        headers = {"Range": f"bytes={start}-{info['bytes']-1}"} if start else {}
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if start or response.status == 206:
                    wanted = f"bytes {start}-{info['bytes']-1}/{info['bytes']}"
                    if response.status != 206 or response.headers.get("Content-Range") != wanted:
                        raise ValueError("Range response differs; partial preserved")
                elif response.status != 200:
                    raise ValueError("Unexpected download response")
                received = start
                with partial.open("ab" if start else "wb") as stream:
                    while True:
                        if shutil.disk_usage(target.parent).free < reserve:
                            raise ValueError("Disk reserve reached; partial preserved")
                        block = response.read(1024**2)
                        if not block:
                            break
                        received += len(block)
                        if received > info["bytes"]:
                            raise ValueError("Server sent more bytes than declared")
                        stream.write(block)
                if received != info["bytes"]:
                    raise OSError("Incomplete download")
            break
        except (OSError, urllib.error.URLError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    if not verified(partial, info):
        raise ValueError("Full SHA mismatch; partial retained: " + str(partial))
    if target.exists():
        raise FileExistsError("Another writer created the target; partial retained")
    partial.replace(target)
    return "downloaded_full_SHA_verified"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="ModelScope local root, e.g. /work/elm-models/elm")
    parser.add_argument("--components", nargs="+", choices=COMPONENTS, default=list(COMPONENTS))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="Metadata-only; safe on a laptop")
    mode.add_argument("--verify-only", action="store_true", help="Verify existing files without a network request")
    parser.add_argument("--reserve-gib", type=float, default=20)
    args = parser.parse_args()
    catalog = json.loads((REPO / "metadata/modelscope_release.json").read_text())
    prefix = PurePosixPath(catalog["repo_prefix"])
    if prefix.is_absolute() or ".." in prefix.parts:
        parser.error("Unsafe release prefix")
    files = selected_files(catalog, args.components)
    if args.plan:
        print(json.dumps({"repo_id": catalog["repo_id"], "release": str(prefix),
                          "files": list(files), "bytes": sum(x["bytes"] for x in files.values()),
                          "base_4B_included": False, "network_used": False}))
        return
    if sys.platform != "linux":
        parser.error("Weights stay on the authorized Linux server; use --plan for laptop checks")
    if not args.root or args.reserve_gib < 12:
        parser.error("Specify --root and keep at least 12 GiB disk reserve")
    root = args.root.resolve()
    if root == REPO or REPO in root.parents:
        parser.error("Keep weights outside the Git checkout")
    root.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (root / ".elm_download.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reserve = int(args.reserve_gib * 1024**3)
        manifest = root.joinpath(*prefix.parts, "checkpoint_manifest.json")
        if manifest.exists() and json.loads(manifest.read_text()) != catalog:
            raise ValueError("Existing release manifest differs; preserved without overwriting")
        results = []
        for rel, info in files.items():
            target = root.joinpath(*prefix.parts, *PurePosixPath(rel).parts)
            if root not in target.resolve().parents:
                raise ValueError("Target escaped the private download root")
            if args.verify_only:
                if not verified(target, info):
                    raise ValueError("Missing or changed checkpoint: " + rel)
                status = "existing_full_SHA_verified"
            else:
                if shutil.disk_usage(root).free < reserve + info["bytes"] and not target.exists():
                    raise OSError("Insufficient disk reserve for the next file")
                query = urllib.parse.urlencode({"Revision": catalog["revision"],
                                                "FilePath": str(prefix / rel)})
                url = f"https://modelscope.cn/api/v1/models/{catalog['repo_id']}/repo?{query}"
                status = download(url, target, info, reserve)
            result = {"path": str(prefix / rel), "bytes": info["bytes"], "status": status}
            results.append(result)
            print(json.dumps(result), flush=True)
        if not args.verify_only and not manifest.exists():
            with manifest.open("x") as stream:
                stream.write(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"status": "all_selected_files_full_SHA_verified", "files": len(results),
                          "optimizer_or_GPU_used": False, "credentials_required": False}))


if __name__ == "__main__":
    main()
