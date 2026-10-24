#!/usr/bin/env python3
"""Prepare an isolated copy of the archived 80-token baseline. No implicit launch."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import os
import shlex
import subprocess

REPO = Path(__file__).resolve().parents[1]
RUN = Path("experiments/mainline_vrl/runs/oracle4b80_simple_20261024_s1014")
ARM = "label32_stat8"
SOURCE = Path("experiments/mainline_vrl/scripts/python/oracle4b80_simple_1024_v1.py")
SCRIPTS = SOURCE.parent
DATA = Path("experiments/mainline_vrl/runs/stat_sleep_20261014_s1014/statistics/data")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def export_guard(node):
    return (isinstance(node, ast.If) and len(node.body) == 1
            and isinstance(node.body[0], ast.Raise)
            and "Reference snapshot only." in ast.unparse(node))


def portable_source(text, rel, workspace, gpus):
    tree = ast.parse(text)
    tree.body = [node for node in tree.body if not export_guard(node)]
    before = {node.name: ast.dump(node, include_attributes=False) for node in tree.body
              if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    if rel == SOURCE:
        assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "P" for t in node.targets)]
        assert len(assignments) == 1
        assignments[0].value = ast.parse("Path(" + repr(str(workspace)) + ")", mode="eval").body
        pipeline = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "pipeline")
        locks = [node for node in pipeline.body if isinstance(node, ast.With)]
        assert len(locks) == 1
        lock = locks[0]
        # Only remove waiting for the original machine's already-finished teacher process.
        assert isinstance(lock.body[1], ast.Assign) and ast.unparse(lock.body[1]).startswith("teacher =")
        assert isinstance(lock.body[2], ast.Assign) and ast.unparse(lock.body[2]).startswith("di =")
        assert isinstance(lock.body[3], ast.While)
        assert isinstance(lock.body[4], ast.Assert) and "process_exit_check_pending" in ast.unparse(lock.body[4])
        assert isinstance(lock.body[5], ast.Assert) and "teacher/summary.json" in ast.unparse(lock.body[5])
        lock.body = [lock.body[0]] + lock.body[5:]

        class Resources(ast.NodeTransformer):
            def visit_Call(self, node):
                self.generic_visit(node)
                if (isinstance(node.func, ast.Name) and node.func.id == "range"
                        and len(node.args) == 2
                        and all(isinstance(a, ast.Constant) for a in node.args)
                        and [a.value for a in node.args] == [2, 8]):
                    return ast.copy_location(ast.Tuple(elts=[ast.Constant(g) for g in gpus], ctx=ast.Load()), node)
                return node

        tree = Resources().visit(tree)
    ast.fix_missing_locations(tree)
    after = {node.name: ast.dump(node, include_attributes=False) for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    allowed = {"pipeline", "stage", "finalize", "gpus"} if rel == SOURCE else set()
    assert set(before) == set(after)
    assert all(before[name] == after[name] for name in before if name not in allowed)
    return ast.unparse(tree) + "\n"


def prepare(args):
    requested = Path(args.output).absolute()
    output = requested.parent.resolve(strict=True) / requested.name
    assets = Path(args.assets).resolve(strict=True)
    assert not output.exists() and not output.is_symlink(), "Output must be a NEW directory"
    assert output.parent.is_dir(), "Create the output parent directory first"
    assert REPO not in output.parents, "Keep data, environments and outputs outside the code repository"
    assert output != assets, "Never write into the supplied asset workspace"
    gpus = [int(x) for x in args.gpus.split(",")]
    assert gpus and len(set(gpus)) == len(gpus) and min(gpus) >= 0
    environments = {}
    for role, value in [("train", args.train_env), ("ar", args.ar_env)]:
        root = Path(value).resolve(strict=True)
        assert (root / "bin/python").is_file(), str(root / "bin/python")
        environments[role] = root
    manifest = read(REPO / "metadata/source_manifest.json")
    requirements = read(REPO / "metadata/asset_requirements.json")["files"]
    for row in manifest["source_files"]:
        assert sha(REPO / row["snapshot_path"]) == row["snapshot_sha256"], row["snapshot_path"]
    for rel, info in requirements.items():
        candidate = assets / rel
        assert candidate.is_file(), "Missing private asset: " + rel
        assert sha(candidate) == info["sha256"], "Asset SHA mismatch: " + rel
    output.mkdir()
    dependencies = {}
    for row in manifest["source_files"]:
        rel = Path(row["upstream_path"])
        source = REPO / row["snapshot_path"]
        target = output / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix == ".py":
            target.write_text(portable_source(source.read_text(), rel, output, gpus))
        else:
            target.write_bytes(source.read_bytes())
        if rel != SOURCE:
            dependencies[str(rel)] = sha(target)
    supplied = {}
    for rel, info in requirements.items():
        target = output / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        assert not target.exists() and not target.is_symlink()
        target.symlink_to(assets / rel)
        supplied[str(target)] = info["sha256"]
    (output / ".venv").symlink_to(environments["train"], target_is_directory=True)
    ar = output / "runtime/envs/eeglm-vllm"
    ar.parent.mkdir(parents=True, exist_ok=True)
    ar.symlink_to(environments["ar"], target_is_directory=True)
    protocol = {
        "source_sha256": sha(output / SOURCE), "dependencies": dependencies,
        "models": {str(output / rel): v["sha256"] for rel, v in requirements.items() if v["role"] == "model"},
        "assets": {str(output / rel): v["sha256"] for rel, v in requirements.items() if v["role"] == "initialization"},
        "data": {"record_file_sha256": {s: requirements[str(DATA / (s + "_records.json"))]["sha256"] for s in ["train", "val"]}},
        "training": {"steps_total_per_arm": 3600}, "arms": {ARM: {"total": 80}},
        "reproduction": {"baseline": manifest["baseline_run"], "seed": 1014, "gpus": gpus,
                         "asset_workspace": str(assets), "no_automatic_EEG": True},
    }
    write(output / RUN / "protocol.json", protocol)
    (output / RUN / "audit").mkdir(exist_ok=True)
    state = {"output": str(output), "source": str(SOURCE), "run": str(RUN),
             "source_sha256": sha(output / SOURCE), "protocol_sha256": sha(output / RUN / "protocol.json"),
             "supplied_assets": supplied, "gpus": gpus, "training_started": False,
             "transformations": ["Remove archive entry guards", "Relocate project root", "Use explicit GPU allowlist",
                                 "Use verified existing teacher assets instead of waiting for the historical process"],
             "algorithm_function_and_class_AST_unchanged": True,
             "train_environment": str(environments["train"]), "ar_environment": str(environments["ar"])}
    write(output / "reproduction.json", state)
    print(json.dumps({"status": "prepared_not_launched", "output": str(output), "asset_files_verified": len(supplied),
                      "next": "Run preflight before run; neither is executed by prepare"}))


def environment(output):
    run = output / RUN
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", VLLM_USE_V1="0",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               TOKENIZERS_PARALLELISM="false", TMPDIR=str(output / "t"))
    folders = {"HF_HOME": "hf", "XDG_CACHE_HOME": "xdg", "TORCH_HOME": "torch",
               "TORCH_EXTENSIONS_DIR": "extensions", "TRITON_CACHE_DIR": "triton", "VLLM_CACHE_ROOT": "vllm",
               "TORCHINDUCTOR_CACHE_DIR": "inductor", "NUMBA_CACHE_DIR": "numba", "CUDA_CACHE_PATH": "cuda"}
    for name, folder in folders.items():
        path = run / "cache" / folder
        path.mkdir(parents=True, exist_ok=True)
        env[name] = str(path)
    (output / "t").mkdir(exist_ok=True)
    env["PATH"] = str(output / ".venv/bin") + ":" + str(output / "runtime/envs/eeglm-vllm/bin") + ":" + env.get("PATH", "")
    for name in ["LD_LIBRARY_PATH", "CUDA_HOME", "FLASHINFER_WORKSPACE_BASE"]:
        env.pop(name, None)
    return env


def execute(args):
    output = Path(args.output).resolve(strict=True)
    state = read(output / "reproduction.json")
    assert state["output"] == str(output), "Prepared workspace cannot be moved; prepare a new one"
    assert sha(output / SOURCE) == state["source_sha256"]
    assert sha(output / RUN / "protocol.json") == state["protocol_sha256"]
    for path, expected in state["supplied_assets"].items():
        assert sha(path) == expected, "Supplied asset changed after preparation: " + path
    env = environment(output)
    python = str(output / ".venv/bin/python")
    if args.command == "preflight":
        code = ("import sys;sys.path.insert(0," + repr(str(output / SCRIPTS)) + ");"
                "import oracle4b80_simple_1024_v1 as m; m.prepare();"
                "assert not m.torch.cuda.is_initialized();print('CPU preflight passed; no GPU initialized')")
        subprocess.run([python, "-B", "-c", code], cwd=output, env=env, check=True)
        ar_code = ("import sys;sys.path.insert(0," + repr(str(output / SCRIPTS)) + ");"
                   "import oracle4b80_simple_1024_v1 as m;"
                   "import vllm;from transformers import AutoTokenizer;"
                   "AutoTokenizer.from_pretrained(str(m.base.MODEL),local_files_only=True);"
                   "assert not m.torch.cuda.is_initialized();"
                   "print('AR environment and tokenizer CPU preflight passed; no GPU initialized')")
        subprocess.run([str(output / "runtime/envs/eeglm-vllm/bin/python"), "-B", "-c", ar_code],
                       cwd=output, env=env, check=True)
        assert not (output / RUN / "first_launch.json").exists(), "A pipeline was already launched"
        write(output / "cpu_preflight.json", {"status": "passed", "source_sha256": state["source_sha256"], "GPU_used": False})
    else:
        passed = read(output / "cpu_preflight.json")
        assert passed["status"] == "passed" and passed["source_sha256"] == state["source_sha256"]
        startup = "\n".join([
            "import sys;sys.path.insert(0," + repr(str(output / SCRIPTS)) + ")",
            "import oracle4b80_simple_1024_v1 as m",
            "assert not m.cache_owned(), 'An owned worker is still alive; do not start a duplicate'",
            "(m.ROOT/'worker_master').mkdir(exist_ok=True)",
            "paths=[m.ROOT/'worker_master/pipeline.lock']+[m.ROOT/('gpu%d.lease'%g) for g in " + repr(state["gpus"]) + "]",
            "assert all(m.lock_free(p) for p in paths), 'An existing pipeline or GPU lease is held'",
            "p=m.R/'curriculum/oracle/last.pt'",
            "if p.exists():",
            " c=m.torch.load(p,map_location='cpu',weights_only=False)",
            " assert c['source_sha256']==m.sha(m.__file__) and c['init_sha256']==m.sha(m.R/'oracle_init.pt')",
            " assert all(k in c for k in ['optimizer','rng','torch_rng','cuda_rng','numpy_rng','all_cuda_rng','python_global_rng'])",
            " assert c['optimizer']['state'] and all(int(s['step'])==c['step'] for s in c['optimizer']['state'].values())",
            "assert not m.torch.cuda.is_initialized()",
        ])
        subprocess.run([python, "-B", "-c", startup], cwd=output, env=env, check=True)
        # This explicit command is the ONLY GPU entry point. It preserves the original stage clocks and locks.
        command = [python, "-B", str(output / SOURCE), "pipeline"]
        print("Starting: " + shlex.join(command), flush=True)
        os.execvpe(python, command, env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Verify supplied assets and create a NEW isolated workspace; no GPU")
    p.add_argument("--assets", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--train-env", required=True, help="Existing training venv root")
    p.add_argument("--ar-env", required=True, help="Existing vLLM venv root")
    p.add_argument("--gpus", required=True, help="Physical GPU indices to allow, e.g. 2,5")
    for name in ["preflight", "run"]:
        p = sub.add_parser(name)
        p.add_argument("--output", required=True)
    args = parser.parse_args()
    prepare(args) if args.command == "prepare" else execute(args)


if __name__ == "__main__":
    main()
