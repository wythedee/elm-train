"""Autoregressive Val generation and independent schema/fact scoring."""
from __future__ import annotations
from collections import Counter, defaultdict
import json
import hashlib
import importlib.metadata
import os
from pathlib import Path
import time
from .io import read, sha, write
from . import tasks
from .backends import engine_options, validate_version


def evaluation_identity(cfg, config_sha, checkpoint):
    """Freeze actual inference dependencies; paths alone are not identities."""
    model = Path(cfg["model"])
    files = {"checkpoint": Path(checkpoint), "oracle": Path(cfg["oracle"]),
             "feature_index": Path(cfg["features"]) / "val_index.json"}
    index = read(model / "model.safetensors.index.json")
    for name in sorted(set(index["weight_map"].values())):
        files["model/" + name] = model / name
    for f in sorted(model.iterdir()):
        if f.is_file() and f.suffix in (".json", ".jinja", ".txt"):
            files["model/" + f.name] = f
    root = Path(cfg["features"]).resolve()
    for name in sorted({r["shard"] for r in read(files["feature_index"])["records"]}):
        path = (root / name).resolve(strict=True)
        if root not in path.parents:
            raise ValueError("Feature shard escaped its root")
        files["feature/" + name] = path
    for f in sorted(Path(__file__).parent.glob("*.py")):
        files["source/" + f.name] = f
    return {"config_sha256": config_sha,
            "files": {name: {"path": str(path), "sha256": sha(path)}
                      for name, path in files.items()}}


def verify_identity(identity):
    for name, info in identity["files"].items():
        if sha(info["path"]) != info["sha256"]:
            raise ValueError("Inference dependency changed: " + name)


def metrics(rows):
    result = {}
    for task in sorted({r["task"] for r in rows}):
        rr = [r for r in rows if r["task"] == task]
        by_dataset = {}
        for dataset in sorted({r["dataset"] for r in rr}):
            ss = [r for r in rr if r["dataset"] == dataset]
            item = {"n": len(ss), "correct": sum(r["correct"] for r in ss),
                    "accuracy": sum(r["correct"] for r in ss)/len(ss),
                    "format_valid": sum(r["valid"] for r in ss)/len(ss)}
            if task != "report":
                support = Counter(tasks.dumps(r["expected"]) for r in ss)
                item["balanced_accuracy"] = sum(
                    sum(r["correct"] for r in ss if tasks.dumps(r["expected"]) == label) / n
                    for label, n in support.items()) / len(support)
                item["support"] = dict(support)
            by_dataset[dataset] = item
        value = {"n": len(rr), "correct": sum(r["correct"] for r in rr),
                 "accuracy": sum(r["correct"] for r in rr)/len(rr),
                 "format_valid": sum(r["valid"] for r in rr)/len(rr),
                 "datasets": by_dataset,
                 "dataset_macro_accuracy": sum(x["accuracy"] for x in by_dataset.values())/len(by_dataset)}
        if task == "report":
            value["components"] = {f: sum(r["components"][f] for r in rr)/len(rr)
                                   for f in ["label"] + tasks.FIELDS}
        else:
            value["dataset_macro_bacc"] = sum(x["balanced_accuracy"] for x in by_dataset.values())/len(by_dataset)
        result[task] = value
    return {"tasks": result, "n": len(rows), "no_task0_aggregate": True, "Test": "N/A"}


def evaluate(cfg, config_sha, tag="validation"):
    backend = validate_version(cfg, importlib.metadata.version("vllm"))
    began = time.perf_counter()
    import torch
    from .data import FeatureDataset
    from .projector import EEGProjector80
    from .training import oracle_prefix, seed, tensor_state
    from safetensors import safe_open
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    out = Path(cfg["output"])
    folder = out / "evaluations" / tag
    if folder.exists():
        raise FileExistsError("Preserve existing evaluation; use a new explicit tag")
    folder.mkdir(parents=True)
    seed(cfg.get("seed", 1014))
    ck = Path(cfg["evaluation_checkpoint"]) if cfg.get("evaluation_checkpoint") else out / "checkpoints/last.pt"
    ck_sha = sha(ck)
    frozen = evaluation_identity(cfg, config_sha, ck)
    write(folder / "identity.json", frozen)
    model = EEGProjector80().cuda().eval()
    model.load_state_dict(tensor_state(ck), strict=True); model.requires_grad_(False)
    data = FeatureDataset(cfg["features"], "val")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], local_files_only=True)
    index = read(Path(cfg["model"]) / "model.safetensors.index.json")
    key = "model.embed_tokens.weight"
    with safe_open(str(Path(cfg["model"]) / index["weight_map"][key]), framework="pt", device="cpu") as f:
        embedding = torch.nn.Embedding.from_pretrained(f.get_tensor(key), freeze=True)
    options = engine_options(cfg, os.environ["ELM_AR_UTIL"])
    startup_begin = time.perf_counter()
    engine = LLM(**options)
    startup_seconds = time.perf_counter() - startup_begin
    params = SamplingParams(temperature=0, max_tokens=384, seed=cfg.get("seed", 1014))
    queries = [(i, tasks.example(r, task)) for i, r in enumerate(data.records)
               for task in ["label"] + ["feature:"+f for f in tasks.FIELDS] + ["report"]]
    if cfg.get("eval_query_limit"):
        if not cfg.get("software_debug"):
            raise ValueError("A query cap is only allowed in explicit software debug")
        queries = queries[:cfg["eval_query_limit"]]
    if not queries:
        raise ValueError("No Val queries; evaluation refused")
    oracle = tensor_state(cfg["oracle"]) if cfg.get("oracle_positive_control") else None
    sources = ["projector", "oracle_positive_control"] if oracle is not None else ["projector"]
    if cfg.get("migration_text_controls"):
        # Fixed, accuracy-independent text controls: native IDs and the exact
        # same word embeddings must reach the same decoder path.
        selected, labels = [], set()
        for r in data.records:
            if r["label"] not in labels:
                selected.append(tasks.example(r, "label")); labels.add(r["label"])
            if len(selected) == 2:
                break
        if len(selected) < 2:
            raise ValueError("Migration control requires two distinct labels")
        token_ids = [tokenizer(tokenizer.apply_chat_template(tasks.messages(e, teacher=True),
                     tokenize=False, add_generation_prompt=True, enable_thinking=False),
                     add_special_tokens=False)["input_ids"] for e in selected]
        words = [tasks.render(tokenizer, embedding, e, teacher=True).bfloat16() for e in selected]
        with torch.no_grad():
            if not all(torch.equal(w, embedding(torch.tensor(ids)).bfloat16()) for w, ids in zip(words, token_ids)):
                raise ValueError("Native teacher IDs and text embeddings differ")
        native = engine.generate([{"prompt_token_ids": ids} for ids in token_ids], params, use_tqdm=False)
        embedded = engine.generate([{"prompt_embeds": w} for w in words], params, use_tqdm=False)
        controls = [{"id": e["id"], "expected": e["expected"],
                     "native_text": a.outputs[0].text, "embedded_text": b.outputs[0].text,
                     "token_ids_equal": list(a.outputs[0].token_ids) == list(b.outputs[0].token_ids),
                     "native_score": tasks.score(a.outputs[0].text, e),
                     "embedded_score": tasks.score(b.outputs[0].text, e)}
                    for e, a, b in zip(selected, native, embedded)]
        write(folder / "native_embedding_controls.json", controls)
        if len(controls) != 2 or not all(c["token_ids_equal"] for c in controls):
            raise ValueError("Native-ID versus embedding generation differs; review preserved controls")
    rows = []
    timings = []
    for source in sources:
        for offset in range(0, len(queries), 16):
            if time.time() > float(os.environ["ELM_STAGE_DEADLINE"]):
                raise TimeoutError("Original evaluation-stage deadline")
            part = queries[offset:offset+16]
            prompts = []
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for i, item in part:
                    prefix = model(*data.inputs(i, "cuda")) if source == "projector" else oracle_prefix([item["record"]], oracle, "cuda")
                    prompts.append({"prompt_embeds": tasks.render(tokenizer, embedding, item, prefix).cpu().bfloat16()})
            torch.cuda.synchronize()
            generation_begin = time.perf_counter()
            generated = engine.generate(prompts, params, use_tqdm=False)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - generation_begin
            if len(generated) != len(prompts):
                raise ValueError("Inference backend returned the wrong query count")
            timing = {"source": source, "offset": offset, "queries": len(part),
                      "seconds": seconds, "input_tokens": sum(len(p["prompt_embeds"]) for p in prompts),
                      "output_tokens": sum(len(g.outputs[0].token_ids) for g in generated)}
            timings.append(timing)
            with (folder / "generation_batches.jsonl").open("a") as stream:
                stream.write(json.dumps(timing) + "\n")
            for (_, item), prompt, generation in zip(part, prompts, generated):
                text = generation.outputs[0].text
                row = {"id": item["id"], "task": item["task"], "dataset": item["record"]["dataset"],
                       "question": item["question"], "expected": item["expected"],
                       "text": text, "source": source, "checkpoint_sha256": ck_sha,
                       "prompt_sha256": hashlib.sha256(prompt["prompt_embeds"].float().numpy().tobytes()).hexdigest(),
                       "config_sha256": config_sha, **tasks.score(text, item)}
                rows.append(row)
                with (folder / "raw.partial.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False)+"\n"); stream.flush()
            print("AR", source, min(offset+16, len(queries)), len(queries), flush=True)
    verify_identity(frozen)
    (folder / "raw.partial.jsonl").replace(folder / "raw.jsonl")
    result = {"status": "complete", "scope": "Val software debug" if cfg.get("software_debug") else "Val",
              "results": {source: metrics([r for r in rows if r["source"] == source]) for source in sources},
              "queries_per_source": len(queries), "unique_Val_EEG": len(data.records),
              "checkpoint_sha256": ck_sha, "Test": "N/A",
              "inference_profile": backend, "vllm_version": importlib.metadata.version("vllm"),
              "engine_options": options, "generation_timing": {
                  "engine_startup_seconds": startup_seconds,
                  "all_generate_calls_seconds": sum(t["seconds"] for t in timings),
                  "output_tokens": sum(t["output_tokens"] for t in timings),
                  "output_tokens_per_generate_second": sum(t["output_tokens"] for t in timings) / sum(t["seconds"] for t in timings),
                  "including_startup_hashing_and_scoring_seconds": time.perf_counter()-began,
                  "peak_allocated_GiB": torch.cuda.max_memory_allocated()/1024**3}}
    write(folder / "summary.json", result)
    return result


def report(cfg, config_sha=None):
    out = Path(cfg["output"])
    groups = {}
    folders = sorted(p for p in (out / "evaluations").iterdir() if p.is_dir())
    if not folders:
        raise ValueError("No evaluation artifacts; report refused")
    for folder in folders:
        file = folder / "raw.jsonl"
        if not file.exists():
            raise ValueError("Partial evaluation retained; no complete report: " + str(folder))
        frozen = read(folder / "identity.json")
        if config_sha is None or frozen["config_sha256"] != config_sha:
            raise ValueError("Report config differs from evaluated config")
        verify_identity(frozen)
        with file.open() as stream:
            raw = [json.loads(line) for line in stream if line.strip()]
        records = read(Path(cfg["features"]) / "val_index.json")["records"]
        expected = {tasks.example(r, t)["id"]: tasks.example(r, t)
                    for r in records
                    for t in ["label"] + ["feature:"+f for f in tasks.FIELDS] + ["report"]}
        if cfg.get("eval_query_limit"):
            if not cfg.get("software_debug"):
                raise ValueError("Query cap requires software debug")
            expected = dict(list(expected.items())[:cfg["eval_query_limit"]])
        if not expected:
            raise ValueError("Empty Val input")
        sources = {"projector", "oracle_positive_control"} if cfg.get("oracle_positive_control") else {"projector"}
        seen = set()
        for row in raw:
            identity = (row["source"], row["id"])
            if identity in seen or row["id"] not in expected or row["source"] not in sources:
                raise ValueError("Duplicate/unknown query ID")
            if row["config_sha256"] != config_sha or row["checkpoint_sha256"] != frozen["files"]["checkpoint"]["sha256"]:
                raise ValueError("Raw checkpoint/config identity mismatch")
            seen.add(identity)
            item = expected[row["id"]]
            if (row["expected"] != item["expected"] or row["question"] != item["question"]
                    or row["dataset"] != item["record"]["dataset"] or row["task"] != item["task"]):
                raise ValueError("Query/target mismatch")
            recomputed = tasks.score(row["text"], item)
            if any(row[k] != value for k, value in recomputed.items()):
                raise ValueError("Parser rescore mismatch")
        if seen != {(s, i) for s in sources for i in expected}:
            raise ValueError("Missing evaluations; retain partial raw without claiming completion")
        summary = read(folder / "summary.json")
        rescored = {s: metrics([r for r in raw if r["source"] == s]) for s in sources}
        if summary["status"] != "complete" or summary["results"] != rescored:
            raise ValueError("Stored summary differs from raw rescore")
        groups[file.parent.name] = {source: metrics([r for r in raw if r["source"] == source])
                                   for source in sorted({r["source"] for r in raw})}
    lines = ["# Pipeline debug result", "", "Val only; software acceptance is separate from model accuracy.", "",
             "|Evaluation|Input|Task|Correct|Accuracy|Valid JSON|", "|---|---|---|---:|---:|---:|"]
    for tag, sources in groups.items():
        for source, value in sources.items():
            for task, m in value["tasks"].items():
                lines.append(f"|{tag}|{source}|{task}|{m['correct']}/{m['n']}|{m['accuracy']:.2%}|{m['format_valid']:.2%}|")
    lines += ["", "Known historical cohort aliases must be resolved before formal generalization claims.",
              "This run starts at already-preprocessed Native200 input; it does not certify every raw-data adapter."]
    (out / "final_report.md").write_text("\n".join(lines)+"\n")
    result = {"status": "rescored_available_artifacts", "evaluations": groups,
              "software_debug": cfg.get("software_debug", False), "Test": "N/A",
              "process_exit_check_pending": True}
    write(out / "results_summary.json", result)
    return result
