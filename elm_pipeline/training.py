"""Single-GPU Projector SFT + frozen Oracle alignment, with strict local resume."""
from __future__ import annotations
import json
import hashlib
import os
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch
from torch.nn import functional as F

from .data import FeatureDataset
from .io import sha, write
from .projector import EEGProjector80, token_alignment
from . import tasks


def seed(value):
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def tensor_state(path):
    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(str(path), device="cpu")
    blob = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    return blob.get("state_dict", blob)


def oracle_prefix(records, state, device):
    outputs = []
    for r in records:
        label = state["label_codes"][r["global_label_id"]] + state["label_position"]
        stats = torch.cat([state["stat_codes"][j, tasks.LEVELS.index(r["bins"][f])]
                           for j, f in enumerate(tasks.FIELDS)])
        outputs.append(torch.cat([label, stats]))
    return torch.stack(outputs).to(device).detach()


def assignment(step, slot):
    if slot < 4:
        return "label"
    if slot < 6:
        return "feature:" + tasks.FIELDS[((step * 2) + slot - 4) % 6]
    return "report"


def task_loss(lm, tokenizer, embedding, item, prefix):
    x = tasks.render(tokenizer, embedding, item, prefix)
    ids = tokenizer(tasks.dumps(item["expected"]), add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    y = torch.tensor(ids, dtype=torch.long, device=x.device)
    xx = torch.cat([x, embedding(y).detach()])[None]
    logits = lm(inputs_embeds=xx, attention_mask=torch.ones(xx.shape[:2], device=x.device, dtype=torch.long),
                use_cache=False).logits[0, len(x)-1:len(x)+len(y)-1].float()
    return F.cross_entropy(logits, y)


def rng_pack(private):
    return {"python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
            "private_rng": private.getstate()}


def rng_restore(blob, private):
    random.setstate(blob["python_rng"]); np.random.set_state(blob["numpy_rng"])
    torch.set_rng_state(blob["torch_rng"]); torch.cuda.set_rng_state_all(blob["cuda_rng"])
    private.setstate(blob["private_rng"])


def train(cfg, config_sha, stop_at=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out = Path(cfg["output"])
    target_steps = stop_at if stop_at is not None else cfg["train_steps"]
    if not 0 < target_steps <= cfg["train_steps"]:
        raise ValueError("Stop point must be within the frozen training plan")
    seed(cfg.get("seed", 1014))
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    torch.cuda.set_per_process_memory_fraction(min(0.98, float(os.environ["ELM_ALLOCATOR_GIB"]) / total), 0)
    model = EEGProjector80().cuda().train()
    if sum(p.numel() for p in model.parameters()) != 253616432:
        raise AssertionError("Projector architecture changed")
    lm = AutoModelForCausalLM.from_pretrained(cfg["model"], local_files_only=True,
                    torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    lm.requires_grad_(False); lm.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], local_files_only=True)
    embedding = lm.get_input_embeddings()
    dataset = FeatureDataset(cfg["features"], "train")
    oracle = tensor_state(cfg["oracle"])
    for value in oracle.values():
        value.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-5), weight_decay=1e-4)
    private = random.Random(cfg.get("seed", 1014))
    last = out / "checkpoints/last.pt"
    last.parent.mkdir(parents=True, exist_ok=True)
    feature_identity = sha(Path(cfg["features"]) / "train_index.json")
    sources = {name: sha(Path(__file__).with_name(name))
               for name in ["training.py", "projector.py", "tasks.py", "data.py", "io.py"]}
    identity = {"config_sha256": config_sha, "feature_index_sha256": feature_identity,
                "oracle_sha256": sha(cfg["oracle"]), "source_sha256": sha(__file__),
                "implementation_sha256": hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest(),
                "output_scale": 0.08, "software_debug": cfg.get("software_debug", False)}
    start, order, cursor, epoch = 0, [], 0, 0
    if last.exists():
        d = torch.load(last, map_location="cpu", weights_only=False, mmap=True)
        if d["identity"] != identity:
            raise ValueError("Resume identity mismatch; preserve the old run")
        model.load_state_dict(d["state_dict"], strict=True); opt.load_state_dict(d["optimizer"])
        start = d["step"]
        if any(int(v["step"]) != start for v in opt.state.values()):
            raise AssertionError("Optimizer step differs from committed checkpoint")
        order, cursor, epoch = d["order"], d["cursor"], d["epoch"]
        rng_restore(d, private)
    elif cfg.get("initial_projector"):
        model.load_state_dict(tensor_state(cfg["initial_projector"]), strict=True)
    committed_ids, grad_names = set(), set()
    all_slots = torch.zeros(80, device="cuda")

    def save(step):
        if shutil.disk_usage(out).free < 20 * 1024**3:
            raise OSError("20 GiB disk reserve")
        blob = {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "optimizer": opt.state_dict(), "step": step, "identity": identity,
                "order": order, "cursor": cursor, "epoch": epoch, **rng_pack(private)}
        tmp = last.with_suffix(".pt.partial")
        with tmp.open("wb") as stream:
            torch.save(blob, stream); stream.flush(); os.fsync(stream.fileno())
        tmp.replace(last)
        write(out / "commit.json", {"step": step, "sha256": sha(last),
              "identity": identity, "optimizer_all_rng_saved": True, "time": time.time()})

    trace = out / "train_trace.jsonl"
    if trace.exists():
        old = [json.loads(line) for line in trace.open() if line.strip()]
        if [r["step"] for r in old] != list(range(1, start+1)):
            raise ValueError("Uncommitted/duplicate trace requires explicit recovery; no silent overwrite")
        committed_ids.update(i for r in old for i in r["sample_ids"])
    for step in range(start, target_steps):
        if time.time() > float(os.environ["ELM_STAGE_DEADLINE"]):
            raise TimeoutError("Original stage budget expired")
        opt.zero_grad(set_to_none=True)
        ces, aligned_values, batch_ids = [], [], []
        for slot in range(8):
            if cursor >= len(order):
                order = list(range(len(dataset.records))); private.shuffle(order)
                cursor = 0; epoch += 1
            index = order[cursor]; cursor += 1
            record = dataset.records[index]
            item = tasks.example(record, assignment(step, slot))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prefix = model(*dataset.inputs(index, "cuda"))
                if cfg.get("software_debug"):
                    prefix.retain_grad()
                alignment = token_alignment(prefix, oracle_prefix([record], oracle, "cuda"))
                ce = task_loss(lm, tokenizer, embedding, item, prefix)
                loss = (cfg.get("ce_weight", 1.0) * ce
                        + cfg.get("alignment_weight", 1.0) * alignment) / 8
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss")
            loss.backward()
            if cfg.get("software_debug"):
                all_slots += prefix.grad.detach().float().abs().sum((0, 2))
            ces.append(float(ce.detach())); aligned_values.append(float(alignment.detach()))
            batch_ids.append(record["id"])
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0:
                grad_names.add(name)
        if any(p.grad is not None or p.requires_grad for p in lm.parameters()):
            raise AssertionError("Frozen Decoder changed its gradient contract")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        row = {"step": step+1, "ce": sum(ces)/8, "alignment": sum(aligned_values)/8,
               "sample_ids": batch_ids, "tasks": [assignment(step, i) for i in range(8)],
               "applied": True, "time": time.time()}
        with trace.open("a") as stream:
            stream.write(json.dumps(row) + "\n"); stream.flush()
        committed_ids.update(batch_ids)
        if (step+1) % 10 == 0 or step+1 == target_steps:
            print(json.dumps(row), flush=True)
        if (step+1) % cfg.get("save_every", 50) == 0 or step+1 == target_steps:
            save(step+1)
    expected_names = set(dict(model.named_parameters()))
    if cfg.get("software_debug") and target_steps > start:
        if grad_names != expected_names or not bool((all_slots > 0).all()):
            raise AssertionError("Missing parameter or token-slot gradients")
        d = torch.load(last, map_location="cpu", weights_only=False, mmap=True)
        model.load_state_dict(d["state_dict"], strict=True); opt.load_state_dict(d["optimizer"])
        if not all(torch.equal(value.detach().cpu(), d["state_dict"][name])
                   for name, value in model.state_dict().items()):
            raise AssertionError("Saved Projector state did not reload exactly")
        rng_restore(d, private)
        first = (random.random(), np.random.random(), torch.rand(5), torch.rand(5, device="cuda"), private.random())
        rng_restore(d, private)
        second = (random.random(), np.random.random(), torch.rand(5), torch.rand(5, device="cuda"), private.random())
        if not (first[0:2] == second[0:2] and torch.equal(first[2], second[2])
                and torch.equal(first[3], second[3]) and first[4] == second[4]):
            raise AssertionError("RNG replay failed")
    result = {"status": "committed", "start_step": start, "step": target_steps,
              "all_parameters_received_gradients": grad_names == expected_names,
              "all_80_slots_received_gradients": bool((all_slots > 0).all()),
              "Decoder_frozen": True, "Encoder_frozen_feature_input": True,
              "unique_Train_EEG_seen": len(committed_ids), "Train_EEG_total": len(dataset.records),
              "data_passes_started": epoch, "sampling": "shuffled full passes, no replacement within a pass",
              "peak_allocated_GiB": torch.cuda.max_memory_allocated()/1024**3,
              "software_debug": cfg.get("software_debug", False), "Test": "N/A"}
    write(out / ("train_result_step" + str(target_steps) + ".json"), result)
    return result
