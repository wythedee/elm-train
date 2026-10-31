"""GPU admission and exact-owned cleanup for one explicitly invoked stage."""
from __future__ import annotations
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from .io import read, sha, write
from .backends import configure_environment

REPO = Path(__file__).resolve().parents[1]
GIB = 1024**3


def identity(pid):
    try:
        root = Path("/proc") / str(pid)
        line = (root / "stat").read_text()
        rest = line[line.rfind(")")+2:].split()
        argv = [s.decode(errors="replace") for s in (root / "cmdline").read_bytes().split(b"\0") if s]
        return {"pid": int(pid), "ppid": int(rest[1]), "state": rest[0],
                "start_ticks": int(rest[19]), "argv": argv}
    except (OSError, ValueError, IndexError):
        return None


def same(stamp):
    current = identity(stamp["pid"])
    return current is not None and all(current[k] == stamp[k] for k in ("pid", "ppid", "start_ticks", "argv"))


def cache_owned(root):
    wanted = str((Path(root)/"cache/hf").resolve()).encode()
    result = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            env = dict(v.split(b"=", 1) for v in (p/"environ").read_bytes().split(b"\0") if b"=" in v)
            if env.get(b"HF_HOME") == wanted:
                ident = identity(p.name)
                if ident and ident["state"] != "Z":
                    result.append(ident)
        except (OSError, ValueError):
            pass
    return result


def free_lock(path):
    with Path(path).open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            return False


def final_guard(root):
    root = Path(root)
    if not free_lock(root/"stage.lock"):
        raise RuntimeError("Active stage role lock; CPU report refused")
    for p in root.glob("gpu*.lease"):
        if not free_lock(p):
            raise RuntimeError("Held GPU lease; CPU report refused")
    owned = cache_owned(root)
    if owned:
        raise RuntimeError("Active cache-owned process; CPU report refused: " + json.dumps(owned))


def gpu_status():
    text = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.total,memory.free,utilization.gpu",
                                    "--format=csv,noheader,nounits"], text=True)
    result = []
    for row in text.splitlines():
        index, total, free, util = [float(x.strip()) for x in row.split(",")]
        result.append({"index": int(index), "total_GiB": total/1024,
                       "free_GiB": free/1024, "util": util})
    return result


def cleanup(root, stamp):
    # First ask only our exact worker to exit. Never signal a reused PID.
    if same(stamp):
        os.kill(stamp["pid"], signal.SIGTERM)
    for _ in range(20):
        if not cache_owned(root):
            return
        time.sleep(0.5)
    for item in cache_owned(root):
        argv = item["argv"]
        recognized = ("elm_pipeline.cli" in argv
                      or any("multiprocessing.spawn" in a or "vllm" in a or "EngineCore" in a for a in argv))
        if not recognized:
            raise RuntimeError("Unknown cache-owned argv retained, needs review: " + json.dumps(item))
        if same(item):
            os.kill(item["pid"], signal.SIGKILL)
    time.sleep(0.5)
    if cache_owned(root):
        raise RuntimeError("Owned processes remain after cleanup")


def run_stage(cfg, config_path, stage, extra):
    if sys.platform != "linux":
        raise RuntimeError("Model execution is server-only; laptop use is limited to static checks")
    def interrupted(signum, frame):
        raise KeyboardInterrupt("Stage interrupted by signal " + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    root = Path(cfg["output"])
    if REPO == root or REPO in root.parents:
        raise ValueError("Keep private outputs outside the Git checkout")
    root.mkdir(parents=True, exist_ok=True)
    disk_reserve = float(cfg.get("disk_reserve_gib", 20))*GIB
    if disk_reserve < 20*GIB:
        raise ValueError("Disk reserve must be at least 20 GiB")
    with (root/"stage.lock").open("a+") as role:
        fcntl.flock(role, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if cache_owned(root):
            raise RuntimeError("Previous worker still active; no duplicate dispatch")
        clock = root/"clock.json"
        if not clock.exists():
            write(clock, {"first_started": time.time(), "deadline": time.time()+cfg.get("total_seconds", 7200)})
        deadline = read(clock)["deadline"]
        if time.time() >= deadline:
            raise TimeoutError("Original pipeline budget expired; no clock reset")
        admission = 26 if stage in ("features", "train") else 24
        lease, candidate = None, None
        while time.time() < deadline:
            for item in sorted(gpu_status(), key=lambda x: -x["free_GiB"]):
                if item["index"] not in cfg["gpus"] or item["free_GiB"] < admission:
                    continue
                stream = (root/f"gpu{item['index']}.lease").open("a+")
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    stream.close(); continue
                samples = []
                for _ in range(3):
                    current = next(g for g in gpu_status() if g["index"] == item["index"])
                    samples.append(current)
                    if current["free_GiB"] < admission:
                        break
                    time.sleep(10)
                write(root/"admission_samples.json", {"stage": stage, "samples": samples})
                if len(samples) == 3 and min(x["free_GiB"] for x in samples) >= admission:
                    lease, candidate = stream, {**item, "min_free_GiB": min(x["free_GiB"] for x in samples)}
                    break
                stream.close()
            if candidate:
                break
            time.sleep(15)
        if candidate is None:
            raise TimeoutError("No GPU admitted within original queue budget")
        try:
            if shutil.disk_usage(root).free < disk_reserve:
                raise OSError("Configured disk reserve")
            env = os.environ.copy()
            env.pop("LD_LIBRARY_PATH", None); env.pop("CUDA_HOME", None)
            env.update(CUDA_VISIBLE_DEVICES=str(candidate["index"]),
                       OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONDONTWRITEBYTECODE="1",
                       TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1",
                       ELM_ALLOCATOR_GIB=str(candidate["min_free_GiB"]-10),
                       ELM_AR_UTIL=str(min(.35, (candidate["min_free_GiB"]-10)/candidate["total_GiB"])))
            configure_environment(env, cfg)
            for key, directory in {"HF_HOME": "hf", "HF_HUB_CACHE": "hf/hub",
                "TRANSFORMERS_CACHE": "hf/transformers", "TORCH_HOME": "torch",
                "TORCH_EXTENSIONS_DIR": "extensions", "TRITON_CACHE_DIR": "triton",
                "VLLM_CACHE_ROOT": "vllm", "NUMBA_CACHE_DIR": "numba",
                "CUDA_CACHE_PATH": "cuda", "XDG_CACHE_HOME": "xdg",
                "TORCHINDUCTOR_CACHE_DIR": "inductor", "CUPY_CACHE_DIR": "cupy",
                "FLASHINFER_WORKSPACE_BASE": "flashinfer", "TVM_FFI_CACHE_DIR": "tvm-ffi"}.items():
                path = root/"cache"/directory; path.mkdir(parents=True, exist_ok=True)
                env[key] = str(path.resolve())
            tmp = Path(cfg.get("tmpdir", root/"tmp"))
            if len(str(tmp)) > 60:
                raise ValueError("Configure a shorter project-local tmpdir for vLLM Unix sockets")
            tmp.mkdir(parents=True, exist_ok=True); env["TMPDIR"] = str(tmp)
            stage_id = stage + ("_"+"_".join(extra) if extra else "")
            stage_clock = root/("clock_"+stage_id.replace("/", "_")+".json")
            if not stage_clock.exists():
                write(stage_clock, {"first_started": time.time(), "deadline": min(deadline, time.time()+cfg.get("stage_seconds", 3600))})
            env["ELM_STAGE_DEADLINE"] = str(read(stage_clock)["deadline"])
            python = cfg["ar_python"] if stage == "evaluate" else cfg["train_python"]
            env["PATH"] = str(Path(python).parent)+":"+env.get("PATH", "")
            command = [python, "-u", "-B", "-m", "elm_pipeline.cli", "--config", str(Path(config_path).resolve()),
                       "--worker", stage] + extra
            logs = root/"logs"; logs.mkdir(exist_ok=True)
            with (logs/(stage_id.replace("/", "_")+".log")).open("a") as log:
                process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                stamp = None
                for _ in range(200):
                    ident = identity(process.pid)
                    if ident and ident["argv"] == command and ident["ppid"] == os.getpid():
                        stamp = ident; break
                    if process.poll() is not None:
                        break
                    time.sleep(.01)
                if stamp is None:
                    raise RuntimeError("Worker identity unresolved")
                write(root/"launch.json", {"stage": stage, "identity": stamp, "gpu": candidate,
                      "command": command, "config_sha256": sha(config_path), "time": time.time()})
                low = 0
                try:
                    while process.poll() is None:
                        if time.time() > float(env["ELM_STAGE_DEADLINE"]):
                            raise TimeoutError("Stage original budget expired")
                        if shutil.disk_usage(root).free < disk_reserve:
                            raise OSError("Configured disk reserve")
                        free = next(g["free_GiB"] for g in gpu_status() if g["index"] == candidate["index"])
                        low = low+1 if free < 8 else 0
                        if low >= 3:
                            raise RuntimeError("Physical GPU reserve below 8 GiB for 3 samples")
                        time.sleep(5)
                    if process.returncode:
                        raise RuntimeError(f"Worker exit {process.returncode}; inspect preserved log")
                finally:
                    cleanup(root, stamp)
                    process.wait()
            write(root/"last_stage_result.json", {"stage": stage, "exit_code": 0,
                  "owned_exited": not cache_owned(root), "time": time.time()})
        finally:
            if lease:
                lease.close()
