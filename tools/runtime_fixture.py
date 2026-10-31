#!/usr/bin/env python3
"""Linux CPU-only tests of real role/lease/cache-owner report guards."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from elm_pipeline.runtime import cache_owned, final_guard, identity, same
from elm_pipeline.io import write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    args = p.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)

    def refused():
        try:
            final_guard(root)
        except RuntimeError:
            return
        raise AssertionError("Guard should have refused")

    final_guard(root)
    for name in ["stage.lock", "gpu6.lease"]:
        with (root/name).open("a+") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            refused()
        final_guard(root)
    env = os.environ.copy(); env["HF_HOME"] = str(root/"cache/hf")
    command = [sys.executable, "-B", "-c", "import time;time.sleep(30)"]
    process = subprocess.Popen(command, env=env)
    try:
        import time
        stamp = None
        for _ in range(100):
            current = identity(process.pid)
            if current and current["argv"] == command:
                stamp = current; break
            time.sleep(.01)
        assert stamp and same(stamp)
        changed = copy.deepcopy(stamp); changed["start_ticks"] += 1
        assert not same(changed)
        changed = copy.deepcopy(stamp); changed["argv"] = ["unknown"]
        assert not same(changed)
        assert any(x["pid"] == process.pid for x in cache_owned(root))
        refused()
    finally:
        if process.poll() is None:
            if not stamp or not same(stamp):
                raise RuntimeError("CPU fixture identity changed; no signal sent")
            process.terminate()
        process.wait()
    final_guard(root)
    result = {"status": "passed", "held_role_refused": True, "held_lease_refused": True,
              "actual_cache_owner_refused": True, "PID_reuse_and_unknown_argv_refused": True,
              "fixture_process_exited": True, "GPU_used": False, "optimizer_used": False}
    write(root/"result.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
