#!/usr/bin/env python3
"""Bounded statistical semantics: factorized Oracle -> raw-bank EEG -> optional DPO.
Only this new run is written. Gold statistics/labels are never student inputs.
"""
from __future__ import annotations

if __name__ == "__main__":
    raise SystemExit("Reference snapshot only. Use tools/workbench.py for inspection; configure a fresh run before training.")

import argparse, collections, copy, fcntl, hashlib, json, math, os, random
import signal, subprocess, sys, time, traceback, shutil
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

PROJECT = Path(__file__).resolve().parents[4]
MASTER = PROJECT / "experiments/mainline_vrl/runs/stat_sleep_20261014_s1014"
DATA = MASTER / "statistics/data"
MODEL = PROJECT / "experiments/eeg_vrl_demo/models/Qwen/Qwen3-1___7B"
BASE = Path("/home/workspace/datasets/yuqiudeng/vstats_experiments/universal_oracle31_0904/expanded_oracle/codebook.pt")
OLD = Path("/var/tmp/yuqiudeng_vstats/braincot16_icl_0907/sft/best.pt")
AR_PYTHON = "/home/workspace/yuqiudeng/envs/eeglm-vllm/bin/python"
FIELDS = ["alpha_relative_power", "theta_beta_ratio", "spectral_entropy",
          "hjorth_mobility", "hjorth_complexity", "line_length"]
VALUES = ["low", "mid", "high"]
SEED = 1014
TASKS = ["label"] + ["feature:" + f for f in FIELDS] + ["rule", "report"]

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for b in iter(lambda: stream.read(4 * 1024 * 1024), b""): h.update(b)
    return h.hexdigest()

def seed_all():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_num_threads(4)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

def token_ids(tok, text, eos=False):
    values = tok(text, add_special_tokens=False).input_ids
    return values + ([tok.eos_token_id] if eos else [])

def wrapper(tok):
    marker = "__STATSEM_INPUT__"
    value = tok.apply_chat_template([{"role": "user", "content": marker}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert value.count(marker) == 1
    return value.split(marker)

def combo_partition(bins):
    # A split of combinations, independent of label, dataset, question or arm.
    text = ",".join(str(int(v)) for v in bins)
    return int(hashlib.sha256(("statsem1014:" + text).encode()).hexdigest()[:8], 16) % 5

def synthetic_record(rng, labels, split, index):
    allowed = {2, 3, 4} if split == "train" else ({0} if split == "val" else {1})
    while True:
        bins = [rng.randrange(3) for _ in FIELDS]
        if combo_partition(bins) in allowed: break
    label_id = rng.randrange(len(labels))
    return {"id": "synthetic_" + split + "_" + str(index),
            "dataset": "synthetic_combinations", "subject": "synthetic",
            "split": split, "label": labels[label_id], "label_id": label_id,
            "global_label_id": label_id,
            "bins": dict(zip(FIELDS, bins)), "synthetic": True}

def bin_values(record):
    bins = record["bins"]
    raw = [bins[f] for f in FIELDS] if isinstance(bins, dict) else list(bins)
    return [VALUES.index(v) if isinstance(v, str) else int(v) for v in raw]

def label_id(record):
    return int(record.get("global_label_id", record["label_id"]))

def question(record, task, rng):
    values = dict(zip(FIELDS, [VALUES[v] for v in bin_values(record)]))
    if task == "label":
        return {"task": task, "question": 'Read the EEG evidence and return its observed class as {"label":"CLASS"}. Output only JSON.',
                "answer": {"label": record["label"]}}
    if task.startswith("feature:"):
        field = task.split(":", 1)[1]
        return {"task": task, "field": field,
                "question": 'Read the EEG evidence. What is the ' + field +
                 ' level? Return only a JSON object with key "' + field +
                 '" and a value low, mid or high.',
                "answer": {field: values[field]}}
    if task == "report":
        return {"task": task,
                "question": 'Write a factual machine-readable EEG report. Return only JSON with keys "label" and "statistics". '
                    '"statistics" must contain exactly these six fields: ' + ", ".join(FIELDS) +
                    '. Each statistic is low, mid or high. State the encoded facts, without diagnoses or extra fields.',
                "answer": {"label": record["label"], "statistics": values}}
    a, b = rng.sample(FIELDS, 2); op = rng.choice(["AND", "OR"]); desired = bool(rng.randrange(2))
    # Exactly balanced truth targets; literals are generated from the evidence.
    while True:
        av, bv = rng.choice(VALUES), rng.choice(VALUES)
        truth = (values[a] == av and values[b] == bv) if op == "AND" else (values[a] == av or values[b] == bv)
        if truth == desired: break
    return {"task": "rule", "operands": [a, b], "operator": op,
            "question": "Apply this stated rule to the EEG evidence: if " + a + " is " + av +
              " " + op + " " + b + " is " + bv +
              ', return {"decision":"review"}; otherwise return {"decision":"routine"}. Return only JSON.',
            "answer": {"decision": "review" if truth else "routine"}}

def fixed_questions(record):
    local_seed = int(hashlib.sha256((record["id"] + ":questions").encode()).hexdigest()[:12], 16)
    return [question(record, t, random.Random(local_seed + j)) for j, t in enumerate(TASKS)]

def prepare(root):
    base = torch.load(BASE, map_location="cpu", weights_only=False)
    labels = base["slot_values"]["label"]
    codes = base["codes"]["label"].float()
    assert codes.shape == (len(labels), 16, 2048)
    positions = base.get("sample_position_embeddings", torch.zeros(1, 2048)).float()
    torch.save({"codes": codes, "positions": positions, "labels": labels}, root / "base_codes.pt")
    for split, count in [("val", 96), ("smoke", 8)]:
        rng = random.Random(SEED + 500000 + (1 if split == "smoke" else 0))
        records = [synthetic_record(rng, labels, "val", i) for i in range(count)]
        write_json(root / ("synthetic_" + split + ".json"), records)
    write_json(root / "prepared.json", {"status": "complete", "base_sha256": sha(BASE),
        "labels": labels, "fields": FIELDS, "synthetic_split": "hash bins %5: train 2/3/4; val 0; test 1",
        "factorized_not_cartesian_lookup": True, "source_sha256": sha(__file__)})

class StatOracle(nn.Module):
    def __init__(self, base, initial_stats, arm):
        super().__init__(); self.arm = arm
        self.label_codes = nn.Parameter(base["codes"].float().clone())
        self.stat_codes = nn.Parameter(initial_stats.float().clone())
        self.register_buffer("initial_label", base["codes"].float().clone(), persistent=False)
        self.register_buffer("initial_stats", initial_stats.float().clone(), persistent=False)
        self.register_buffer("label_position", base["positions"][0].float().clone())
        self.tokens = 32 if arm == "shared32" else 112

    def forward(self, bins, labels):
        # The question is intentionally absent from this function.
        selected = torch.stack([self.stat_codes[f, bins[:, f]] for f in range(len(FIELDS))], dim=1)
        stats = selected.sum(1) / math.sqrt(len(FIELDS)) if self.arm == "shared32" else selected.flatten(1, 2)
        label = self.label_codes[labels] + self.label_position[None, None]
        return torch.cat([label, stats], dim=1)

    def anchor(self):
        a = (self.label_codes - self.initial_label).square().mean() / self.initial_label.square().mean().clamp_min(1e-8)
        b = (self.stat_codes - self.initial_stats).square().mean() / self.initial_stats.square().mean().clamp_min(1e-8)
        return .0001 * (a + b)

class StatStudent(nn.Module):
    """Matched-parameter field-conditioned shared head; raw bank only."""
    def __init__(self, tokens):
        super().__init__(); self.tokens = tokens
        self.arm = "shared32" if tokens == 32 else "fields112"
        self.field_embedding = nn.Parameter(torch.randn(6, 2048) * .02)
        self.field_scale = nn.Parameter(torch.randn(6, 2048) * .1)
        self.flat_projection = nn.Sequential(nn.LayerNorm(7500), nn.Linear(7500, 2048, bias=False), nn.GELU(),
                                             nn.Linear(2048, 16 * 2048, bias=False))
        self.stats_norm = nn.LayerNorm(2048)
        self.stats_projection = nn.Linear(2048, 16 * 2048, bias=False)
        self.output_norm = nn.LayerNorm(2048); self.ffn_norm = nn.LayerNorm(2048)
        self.ffn = nn.Sequential(nn.Linear(2048, 2048), nn.GELU(), nn.Linear(2048, 2048))
        nn.init.normal_(self.stats_projection.weight, std=.0001)

    def forward(self, bank, valid):
        assert bank.shape[1:] == (1, 150, 50), tuple(bank.shape)
        flat = (bank.float() * valid[:, None, :, None].float()).flatten(1)
        h = self.flat_projection[:3](flat)
        label = self.flat_projection[3](h).reshape(len(h), 16, 2048)
        context = self.stats_norm(F.gelu(h[:, None] * (1 + torch.tanh(self.field_scale))[None] + self.field_embedding[None]))
        fields = self.stats_projection(context).reshape(len(h), 6, 16, 2048)
        stats = fields.sum(1) / math.sqrt(6) if self.arm == "shared32" else fields.flatten(1, 2)
        p = torch.cat([label, stats], dim=1)
        p = self.output_norm(p)
        return p + self.ffn(self.ffn_norm(p))

    def warmstart(self):
        source = torch.load(OLD, map_location="cpu", weights_only=False)["state_dict"]
        own = self.state_dict()
        for name, tensor in own.items():
            if name in source:
                assert source[name].shape == tensor.shape, name
                tensor.copy_(source[name])
        self.load_state_dict(own)

def initial_stats(tok, embedding, base):
    rows = []
    scale = base["codes"].float().std().clamp_min(.02)
    with torch.no_grad():
        for field in FIELDS:
            values = []
            for value in VALUES:
                text = "EEG statistic " + field.replace("_", " ") + " has level " + value + "."
                x = embedding(torch.tensor(token_ids(tok, text), device=embedding.weight.device)).float()
                indices = torch.linspace(0, len(x) - 1, 16, device=x.device).round().long()
                x = x[indices]; x = (x - x.mean()) / x.std().clamp_min(1e-8) * scale
                values.append(x.cpu())
            rows.append(torch.stack(values))
    return torch.stack(rows)

def load_oracle(root, device, checkpoint=None):
    base = torch.load(root / "base_codes.pt", map_location="cpu", weights_only=False)
    init = torch.load(root / "initial_stats.pt", map_location="cpu", weights_only=False)
    model = StatOracle(base, init, config(root)["arm"]).to(device)
    if checkpoint: model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"])
    return model

def config(root):
    return json.loads((root / "protocol.json").read_text())

def hf_decoder():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True); tok.pad_token = tok.eos_token
    decoder = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True,
              torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    decoder.config.use_cache = False
    for p in decoder.parameters(): p.requires_grad_(False)
    return tok, decoder, decoder.get_input_embeddings()

def prompt(tok, embedding, q, prefix, answer=None):
    before, after = wrapper(tok); device = embedding.weight.device
    def embed(text):
        return embedding(torch.tensor(token_ids(tok, text), dtype=torch.long, device=device)).detach()
    parts = [embed(before + "EEG evidence sample 1:\n"), prefix.to(device=device, dtype=embedding.weight.dtype),
             embed("\n" + q["question"] + after)]
    n = sum(len(v) for v in parts)
    if answer is None: return torch.cat(parts)
    target_ids = torch.tensor(token_ids(tok, answer, eos=True), dtype=torch.long, device=device)
    parts.append(embedding(target_ids).detach())
    labels = torch.full((n + len(target_ids),), -100, dtype=torch.long, device=device); labels[n:] = target_ids
    return torch.cat(parts), labels

def teacher_batch(records, device):
    return (torch.tensor([bin_values(r) for r in records], dtype=torch.long, device=device),
            torch.tensor([label_id(r) for r in records], dtype=torch.long, device=device))

def sequence_logps(decoder, tok, embedding, questions, prefixes, answers, reduction="mean"):
    sequences, labels = [], []
    for q, prefix, answer in zip(questions, prefixes, answers):
        x, y = prompt(tok, embedding, q, prefix, answer); sequences.append(x); labels.append(y)
    x = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    y = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    lengths = torch.tensor([len(v) for v in sequences], device=x.device)
    attention = torch.arange(x.shape[1], device=x.device)[None] < lengths[:, None]
    logits = decoder(inputs_embeds=x, attention_mask=attention.long(), use_cache=False).logits[:, :-1].float()
    target = y[:, 1:]; mask = target != -100
    safe = target.masked_fill(~mask, 0)
    logps = F.log_softmax(logits, dim=-1).gather(-1, safe[:, :, None]).squeeze(-1) * mask
    if reduction == "sum": return logps.sum(-1)
    return logps.sum(-1) / mask.sum(-1).clamp_min(1)

def sft_loss(decoder, tok, embedding, qs, prefixes):
    answers = [json.dumps(q["answer"], separators=(",", ":")) for q in qs]
    return -sequence_logps(decoder, tok, embedding, qs, prefixes, answers).mean()

def save_checkpoint(path, state):
    if shutil.disk_usage(path.parent).free < 12 * 1024 ** 3:
        raise RuntimeError("Checkpoint requires at least 12 GiB free on its output filesystem")
    temp = path.with_name(path.name + ".tmp"); torch.save(state, temp); temp.replace(path)

def load_real(split):
    if not (DATA / "prepared.json").exists(): raise RuntimeError("statistics/data/prepared.json not ready")
    data = torch.load(DATA / (split + ".pt"), map_location="cpu", weights_only=False)
    records = json.loads((DATA / (split + "_records.json")).read_text())
    assert len(data["bank"]) == len(records)
    assert data["bank"].shape[1:] == (1, 150, 50)
    assert data["bins"].shape == (len(records), len(FIELDS))
    return data, records

def draw_indices(records, rng, batch):
    groups = collections.defaultdict(list)
    for i, r in enumerate(records): groups[(r["dataset"], label_id(r))].append(i)
    datasets = sorted({k[0] for k in groups})
    keys = {ds: [k for k in sorted(groups) if k[0] == ds] for ds in datasets}
    def draw():
        return [rng.choice(groups[rng.choice(keys[rng.choice(datasets)])]) for _ in range(batch)]
    return draw

def actual_indices(root, split, records):
    path = root / (split + "_indices.json")
    if path.exists(): return json.loads(path.read_text())
    cap = 16 if split == "val" else 128
    selected = []
    for ds in sorted({r["dataset"] for r in records}):
        indices = [i for i, r in enumerate(records) if r["dataset"] == ds]
        indices.sort(key=lambda i: hashlib.sha256((records[i]["id"] + ":statsem1014").encode()).hexdigest())
        selected += indices[:cap]
    write_json(path, selected)
    write_json(root / (split + "_selection.json"), {
        "selection": "SHA256 order by fixed ID and seed, per-dataset cap " + str(cap),
        "counts": dict(collections.Counter(records[i]["dataset"] for i in selected)),
        "ids": [records[i]["id"] for i in selected],
        "full": len(selected) == len(records), "not_historical_full_test": True})
    return selected

def train(root, phase, steps):
    seed_all(); out = root / phase; out.mkdir(exist_ok=True)
    tok, decoder, embedding = hf_decoder()
    base = torch.load(root / "base_codes.pt", map_location="cpu", weights_only=False)
    if not (root / "initial_stats.pt").exists():
        torch.save(initial_stats(tok, embedding, base), root / "initial_stats.pt")
    oracle = load_oracle(root, "cuda", root / "oracle/best.pt" if phase != "oracle" else None)
    if phase == "oracle": model = oracle; lr = .001; data = records = None
    else:
        oracle.eval()
        for p in oracle.parameters(): p.requires_grad_(False)
        data, records = load_real("train")
        model = StatStudent(oracle.tokens).cuda(); model.warmstart(); lr = .00001
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.0001)
    rng = random.Random(SEED); start = 0; best = float("inf"); history = []; used = set()
    if (out / "last.pt").exists():
        saved = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(saved["state_dict"]); optimizer.load_state_dict(saved["optimizer"])
        start = saved["step"]; best = saved["best"]; history = saved["history"]
        used = set(saved.get("used_train_indices", []))
        rng.setstate(saved["rng"]); torch.set_rng_state(saved["torch_rng"]); torch.cuda.set_rng_state(saved["cuda_rng"])
        if "numpy_rng" in saved: np.random.set_state(saved["numpy_rng"])
        print("RESUME", phase, start, flush=True)
    if phase == "oracle":
        vrecords = json.loads((root / "synthetic_val.json").read_text())
        val_pairs = [(r, fixed_questions(r)[i % len(TASKS)]) for i, r in enumerate(vrecords)]
    else:
        vdata, vrecords = load_real("val"); vi = actual_indices(root, "val", vrecords)
        val_pairs = [(i, fixed_questions(vrecords[i])[j % len(TASKS)]) for j, i in enumerate(vi[:128])]
        draw = draw_indices(records, rng, 8)
    initial = [p.detach().clone() for p in list(model.parameters())[:2]] if start < 30 else None
    began = time.time(); smoke = []
    def state(step):
        return {"format": "statsem_factorized_oracle_v1" if phase == "oracle" else "statsem_rawbank_heavy_v1",
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "optimizer": optimizer.state_dict(), "step": step, "best": best, "history": history,
                "rng": rng.getstate(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(), "numpy_rng": np.random.get_state(),
                "used_train_indices": sorted(used), "seed": SEED, "decoder_frozen": True,
                "tokens": oracle.tokens, "source_sha256": sha(__file__)}
    for step in range(start + 1, steps + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        if phase == "oracle":
            rs = [synthetic_record(rng, base["labels"], "train", step * 8 + k) for k in range(8)]
            bins, labs = teacher_batch(rs, "cuda")
        else:
            ii = draw(); used.update(ii); rs = [records[i] for i in ii]
            bins = data["bins"][ii].long().cuda(); labs = data["coarse"][ii].long().cuda()
        qs = [question(r, rng.choice(TASKS), rng) for r in rs]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if phase == "oracle": pref = model(bins, labs); align = model.anchor()
            else:
                pref = model(data["bank"][ii].cuda(), data["valid"][ii].cuda())
                with torch.no_grad(): target = oracle(bins, labs)
                nmse = (pref.float() - target.float()).square().mean() / target.float().square().mean().clamp_min(1e-8)
                cos = 1 - F.cosine_similarity(pref.float().flatten(1), target.float().flatten(1)).mean()
                align = .1 * (nmse + cos)
            ce = sft_loss(decoder, tok, embedding, qs, pref); loss = ce + align
        assert torch.isfinite(loss), "non-finite training loss"
        loss.backward(); grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        assert torch.isfinite(grad) and grad > 0, "invalid gradient"
        assert all(p.grad is None for p in decoder.parameters()), "decoder parameter gradient"
        optimizer.step()
        row = {"step": step, "loss": float(loss), "ce": float(ce), "alignment": float(align), "grad": float(grad)}
        if step <= 30: smoke.append(row)
        if step % 10 == 0: print(json.dumps(row), flush=True)
        if step == 30:
            changed = any(not torch.equal(a, b) for a, b in zip(initial, list(model.parameters())[:2]))
            assert changed, "smoke parameters unchanged"
            write_json(out / "smoke.json", {"status": "passed", "steps": 30, "decoder_frozen": True,
                "finite_gradients": True, "parameters_changed": changed, "losses": smoke})
        if step == 30 or step % 500 == 0 or step == steps:
            model.eval(); losses = []
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for offset in range(0, len(val_pairs), 8):
                    group = val_pairs[offset:offset + 8]
                    if phase == "oracle":
                        rr = [r for r, q in group]; bb, ll = teacher_batch(rr, "cuda"); pp = model(bb, ll)
                    else:
                        jj = [i for i, q in group]; pp = model(vdata["bank"][jj].cuda(), vdata["valid"][jj].cuda())
                    losses.append(float(sft_loss(decoder, tok, embedding, [q for r, q in group], pp)))
            val = float(np.mean(losses)); improved = val < best; best = min(best, val)
            history.append({"step": step, "val_tf_loss": val, "elapsed_seconds": time.time() - began})
            ckpt = state(step); save_checkpoint(out / "last.pt", ckpt)
            if improved: save_checkpoint(out / "best.pt", ckpt)
            write_json(out / "summary.json", {"status": "complete" if step >= steps else "running",
                "phase": phase, "step": step, "best_val_tf_loss": best, "history": history,
                "used_train_windows": len(used), "decoder_frozen": True,
                "selection": "Val TF only; promotion independently requires free deterministic AR",
                "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1024 ** 3})
            print("VALIDATION", history[-1], flush=True)

def parse_generation(text, q):
    try: value = json.loads(text.strip())
    except (json.JSONDecodeError, TypeError): return None, False, 0.
    expected = q["answer"]; valid = isinstance(value, dict)
    if not valid: return value, False, 0.
    if q["task"] == "report":
        valid = set(value) == {"label", "statistics"} and isinstance(value.get("label"), str)
        stats = value.get("statistics")
        valid = valid and isinstance(stats, dict) and set(stats) == set(FIELDS)
        valid = valid and all(stats[f] in VALUES for f in FIELDS)
        facts = sum(isinstance(stats, dict) and stats.get(f) == expected["statistics"][f] for f in FIELDS)
        facts += value.get("label") == expected["label"]
        return value, bool(valid), facts / 7.
    valid = set(value) == set(expected)
    if q["task"] == "label": valid = valid and isinstance(value.get("label"), str)
    elif q["task"] == "rule": valid = valid and value.get("decision") in ["review", "routine"]
    else: valid = valid and value.get(q["field"]) in VALUES
    return value, bool(valid), float(value == expected)

def metrics(rows):
    per = {}; summaries = []
    for ds in sorted({r["dataset"] for r in rows}):
        group = [r for r in rows if r["dataset"] == ds]; tasks = {}; features = []
        for task in TASKS:
            selected = [r for r in group if r["task"] == task]
            if not selected: continue
            acc = float(np.mean([r["correct"] for r in selected])); item = {"n": len(selected), "exact_accuracy": acc,
                   "format_valid_rate": float(np.mean([r["valid"] for r in selected]))}
            if task == "label":
                labels = sorted({r["expected"]["label"] for r in selected})
                confusion = np.zeros((len(labels), len(labels) + 1), dtype=int)
                for r in selected:
                    prediction = r["prediction"].get("label") if isinstance(r["prediction"], dict) else None
                    pred = labels.index(prediction) if prediction in labels else len(labels)
                    confusion[labels.index(r["expected"]["label"]), pred] += 1
                recalls = [confusion[k, k] / confusion[k].sum() for k in range(len(labels)) if confusion[k].sum()]
                f1 = [2 * confusion[k, k] / max(1, confusion[k].sum() + confusion[:, k].sum()) for k in range(len(labels))]
                item.update({"bacc": float(np.mean(recalls)), "macro_f1": float(np.mean(f1)),
                    "confusion": confusion.tolist(), "labels": labels, "columns": labels + ["invalid_or_unknown"],
                    "scope": "labels represented in this fixed evaluation subset"})
            if task.startswith("feature:"):
                field = task.split(":", 1)[1]; confusion = np.zeros((3, 4), dtype=int)
                for r in selected:
                    target = VALUES.index(r["expected"][field])
                    prediction = r["prediction"].get(field) if isinstance(r["prediction"], dict) else None
                    pred = VALUES.index(prediction) if prediction in VALUES else 3
                    confusion[target, pred] += 1
                recalls = [confusion[k, k] / confusion[k].sum() for k in range(3) if confusion[k].sum()]
                f1 = [2 * confusion[k, k] / max(1, confusion[k].sum() + confusion[:, k].sum()) for k in range(3)]
                item.update({"bacc": float(np.mean(recalls)), "macro_f1": float(np.mean(f1)),
                             "confusion": confusion.tolist(), "columns": VALUES + ["invalid"]})
                features.append(item["bacc"])
            if task == "report":
                item["fact_accuracy"] = float(np.mean([r["fact_score"] for r in selected]))
                item["wrong_or_missing_facts"] = sum(round((1 - r["fact_score"]) * 7) for r in selected)
                item["extra_report_fields"] = sum(len(set(r["prediction"]) - {"label", "statistics"}) +
                    (len(set(r["prediction"].get("statistics", {})) - set(FIELDS)) if isinstance(r["prediction"].get("statistics"), dict) else 0)
                    for r in selected if isinstance(r["prediction"], dict))
            tasks[task] = item
        feature = float(np.mean(features)) if features else 0.
        rule = tasks.get("rule", {}).get("exact_accuracy", 0.)
        report = tasks.get("report", {}).get("exact_accuracy", 0.)
        semantic = float(np.mean([feature, rule, report]))
        per[ds] = {"tasks": tasks, "feature_macro_bacc": feature, "rule_accuracy": rule,
                   "report_exact_accuracy": report, "semantic_family_macro": semantic}
        summaries.append(semantic)
        per[ds]["within_label_feature_exact"] = {
            label: {"n": len([r for r in group if r["gold_label"] == label and r["task"].startswith("feature:")]),
                    "accuracy": float(np.mean([r["correct"] for r in group if r["gold_label"] == label and r["task"].startswith("feature:")]))}
            for label in sorted({r["gold_label"] for r in group})}
    return {"per_dataset": per, "semantic_family_macro": float(np.mean(summaries)) if summaries else 0.,
            "format_valid_rate": float(np.mean([r["valid"] for r in rows])) if rows else 0.,
            "queries": len(rows), "repair_rate": 0.,
            "scope": "statistical semantics; label/feature/rule/report separated; not old classification ranking"}

def ar_components(root, mode):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from vllm import LLM
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    temporary = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, torch_dtype=torch.bfloat16).cpu()
    embedding = copy.deepcopy(temporary.get_input_embeddings()); del temporary
    oracle = load_oracle(root, "cuda", root / "oracle/best.pt").eval()
    if mode == "oracle": model = oracle
    else:
        model = StatStudent(oracle.tokens).cuda()
        checkpoint = root / ("dpo" if mode == "dpo" else "student") / "best.pt"
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]); model.eval()
    engine = LLM(model=str(MODEL), dtype="bfloat16", enable_prompt_embeds=True, max_model_len=2048,
                 max_num_seqs=16, gpu_memory_utilization=.35, enforce_eager=True,
                 disable_log_stats=True, seed=SEED)
    return tok, embedding, model, engine

def evaluate(root, mode, scope):
    from vllm import SamplingParams
    seed_all(); out = root / "evaluation"; out.mkdir(exist_ok=True)
    path = out / (scope + "_" + mode + ".jsonl")
    if path.with_suffix(".summary.json").exists(): return
    if path.exists():
        failure = out / "failed_attempts"; failure.mkdir(exist_ok=True)
        path.rename(failure / (path.name + "." + str(int(time.time()))))
    if scope.startswith("synthetic"):
        records = json.loads((root / ("synthetic_smoke.json" if scope == "synthetic_smoke" else "synthetic_val.json")).read_text())
        data = None; indices = list(range(len(records)))
    else:
        split = "test" if scope == "actual_test" else "val"
        if split == "test": assert (root / "test_freeze.json").exists(), "Test not frozen"
        data, records = load_real(split); indices = actual_indices(root, split, records)
    tok, embedding, model, engine = ar_components(root, mode)
    shuffled = {}
    if mode == "student_shuffled":
        groups = collections.defaultdict(list)
        for i, r in enumerate(records): groups[(r["dataset"], label_id(r))].append(i)
        for i in indices:
            candidates = [j for j in groups[(records[i]["dataset"], label_id(records[i]))]
                          if j != i and bin_values(records[j]) != bin_values(records[i])]
            candidates.sort(key=lambda j: (records[j]["subject"] == records[i]["subject"],
                hashlib.sha256((records[j]["id"] + records[i]["id"] + ":shuffle1014").encode()).hexdigest()))
            shuffled[i] = candidates[0] if candidates else i
    params = SamplingParams(temperature=0., max_tokens=384)
    pairs = [(i, q) for i in indices for q in fixed_questions(records[i])]
    rows = []
    with path.open("w") as stream:
        for offset in range(0, len(pairs), 16):
            batch = pairs[offset:offset + 16]; ii = [i for i, q in batch]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                if mode == "oracle":
                    bb, ll = teacher_batch([records[i] for i in ii], "cuda"); prefixes = model(bb, ll).cpu()
                else:
                    source_indices = [shuffled.get(i, i) for i in ii]
                    prefixes = model(data["bank"][source_indices].cuda(), data["valid"][source_indices].cuda()).cpu()
            prompts = [{"prompt_embeds": prompt(tok, embedding, q, p).to(torch.bfloat16)}
                       for (i, q), p in zip(batch, prefixes)]
            outputs = engine.generate(prompts, params, use_tqdm=False)
            for (i, q), generated in zip(batch, outputs):
                text = generated.outputs[0].text; pred, valid, fact = parse_generation(text, q)
                row = {"id": records[i]["id"], "dataset": records[i]["dataset"], "gold_label": records[i]["label"],
                       "gold_bins": bin_values(records[i]), "bank_source_id": records[shuffled.get(i, i)]["id"], "task": q["task"],
                       "mode": mode, "scope": scope, "question": q["question"], "expected": q["answer"],
                       "prediction": pred, "generation": text, "valid": valid, "correct": valid and pred == q["answer"],
                       "fact_score": fact, "oracle_uses_gold": mode == "oracle"}
                rows.append(row); stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            if offset % 256 == 0: print("AR", scope, mode, offset, len(pairs), flush=True)
    result = metrics(rows); result.update({"status": "complete", "mode": mode, "scope": scope,
        "deterministic_free_generation": True, "temperature": 0., "max_tokens": 384,
        "source_sha256": sha(__file__), "checkpoint_sha256": sha(root / ("oracle" if mode == "oracle" else "dpo" if mode == "dpo" else "student") / "best.pt")})
    write_json(path.with_suffix(".summary.json"), result)
    write_json(path.with_suffix(".failures.json"), [r for r in rows if not r["correct"]][:40])
    print(json.dumps({k: v for k, v in result.items() if k != "per_dataset"}), flush=True)

def gate(root, scope):
    result = json.loads((root / "evaluation" / (scope + "_oracle.summary.json")).read_text())
    score = result["semantic_family_macro"]; valid = result["format_valid_rate"]
    level = "target" if score >= .80 and valid >= .80 else "bounded_exploration" if score >= .65 and valid >= .80 else "failed"
    value = {"passed": level != "failed", "level": level, "score": score, "format_valid_rate": valid,
             "target": .80, "minimum_exploration": .65, "minimum_valid": .80, "scope": scope,
             "metric": "equal dataset macro of mean(feature macro BAcc, rule accuracy, complete report exact accuracy)"}
    write_json(root / (scope + "_gate.json"), value)
    if not value["passed"]: raise SystemExit(20)

def collect_preferences(root):
    from vllm import SamplingParams
    seed_all(); out = root / "dpo"; out.mkdir(exist_ok=True)
    if (out / "preferences.summary.json").exists(): return
    data, records = load_real("train")
    indices = list(range(len(records)))
    indices.sort(key=lambda i: hashlib.sha256((records[i]["id"] + ":dpo").encode()).hexdigest())
    indices = indices[:384]
    tok, embedding, model, engine = ar_components(root, "student")
    params = SamplingParams(temperature=.8, top_p=.95, n=2, max_tokens=384, seed=SEED)
    pairs = []; all_rows = []
    with (out / "candidate_generations.jsonl").open("w") as stream:
        for offset in range(0, len(indices), 16):
            ii = indices[offset:offset + 16]
            qs = [question(records[i], ["report", "rule", "feature:" + FIELDS[i % 6]][i % 3], random.Random(SEED + i)) for i in ii]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                prefixes = model(data["bank"][ii].cuda(), data["valid"][ii].cuda()).cpu()
            prompts = [{"prompt_embeds": prompt(tok, embedding, q, p).to(torch.bfloat16)} for q, p in zip(qs, prefixes)]
            outputs = engine.generate(prompts, params, use_tqdm=False)
            for i, q, generated in zip(ii, qs, outputs):
                candidates = []
                for v in generated.outputs:
                    parsed, valid, score = parse_generation(v.text, q)
                    candidates.append({"text": v.text, "valid": valid, "fact_score": score, "parsed": parsed})
                row = {"index": i, "id": records[i]["id"], "question": q, "candidates": candidates}
                stream.write(json.dumps(row) + "\n"); all_rows.append(row)
                ordered = sorted(candidates, key=lambda v: v["fact_score"], reverse=True)
                if len(ordered) >= 2 and ordered[0]["valid"] and ordered[-1]["valid"] and ordered[0]["fact_score"] - ordered[-1]["fact_score"] >= .15:
                    pairs.append({"index": i, "question": q, "chosen": ordered[0]["text"], "rejected": ordered[-1]["text"],
                                  "chosen_score": ordered[0]["fact_score"], "rejected_score": ordered[-1]["fact_score"]})
    write_json(out / "preferences.json", pairs)
    write_json(out / "preferences.summary.json", {"status": "complete", "sampled_train_windows": len(indices),
        "pairs": len(pairs), "minimum_pairs": 32, "usable": len(pairs) >= 32, "synthetic_corruptions": False,
        "source": "two actual sampled frozen-SFT EEG generations", "beta": .1, "steps": 500})

def dpo_train(root):
    out = root / "dpo"; preference = json.loads((out / "preferences.summary.json").read_text())
    if not preference["usable"]:
        write_json(out / "summary.json", {"status": "skipped", "reason": "fewer than 32 factual generated preference pairs", "pairs": preference["pairs"]})
        return
    seed_all(); tok, decoder, embedding = hf_decoder(); data, records = load_real("train")
    pairs = json.loads((out / "preferences.json").read_text())
    tokens = 32 if config(root)["arm"] == "shared32" else 112
    sft = torch.load(root / "student/best.pt", map_location="cpu", weights_only=False)
    model = StatStudent(tokens).cuda(); model.load_state_dict(sft["state_dict"])
    reference = StatStudent(tokens).cuda(); reference.load_state_dict(sft["state_dict"]); reference.eval()
    for p in reference.parameters(): p.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.000002, weight_decay=.0001)
    rng = random.Random(SEED); start = 0; history = []
    if (out / "last.pt").exists():
        saved = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(saved["state_dict"]); optimizer.load_state_dict(saved["optimizer"])
        start = saved["step"]; history = saved["history"]
        rng.setstate(saved["rng"]); torch.set_rng_state(saved["torch_rng"]); torch.cuda.set_rng_state(saved["cuda_rng"])
        if "numpy_rng" in saved: np.random.set_state(saved["numpy_rng"])
    began = time.time()
    for step in range(start + 1, 501):
        batch = [rng.choice(pairs) for _ in range(2)]; ii = [b["index"] for b in batch]
        qq = [b["question"] for b in batch]; chosen = [b["chosen"] for b in batch]; rejected = [b["rejected"] for b in batch]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pp = model(data["bank"][ii].cuda(), data["valid"][ii].cuda())
            with torch.no_grad():
                rr = reference(data["bank"][ii].cuda(), data["valid"][ii].cuda())
                ref_c = sequence_logps(decoder, tok, embedding, qq, rr, chosen, "sum")
                ref_r = sequence_logps(decoder, tok, embedding, qq, rr, rejected, "sum")
            policy_c = sequence_logps(decoder, tok, embedding, qq, pp, chosen, "sum")
            policy_r = sequence_logps(decoder, tok, embedding, qq, pp, rejected, "sum")
            margins = .1 * ((policy_c - policy_r) - (ref_c - ref_r))
            loss = F.softplus(-margins).mean()
        assert torch.isfinite(loss); loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); assert torch.isfinite(grad) and grad > 0
        assert all(p.grad is None for p in decoder.parameters())
        optimizer.step()
        if step % 10 == 0: print("DPO", step, float(loss), float(margins.mean()), flush=True)
        if step == 30: write_json(out / "smoke.json", {"status": "passed", "steps": 30, "decoder_frozen": True, "real_generated_pairs": len(pairs)})
        if step % 250 == 0 or step == 30:
            history.append({"step": step, "loss": float(loss), "elapsed_seconds": time.time() - began})
            saved = {"format": "statsem_dpo_rawbank_v1", "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                     "optimizer": optimizer.state_dict(), "step": step, "history": history,
                     "rng": rng.getstate(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(), "numpy_rng": np.random.get_state(),
                     "reference_sft_sha256": sha(root / "student/best.pt"), "beta": .1, "decoder_frozen": True,
                     "source_sha256": sha(__file__)}
            save_checkpoint(out / "last.pt", saved)
            # Fixed terminal DPO checkpoint; only Val AR decides SFT vs DPO for Test.
            if step == 500: save_checkpoint(out / "best.pt", saved)
            write_json(out / "summary.json", {"status": "complete" if step == 500 else "running",
                                             "step": step, "pairs": len(pairs), "history": history})

def freeze_test(root):
    candidates = {"student": root / "student/best.pt"}
    chosen = "student"
    sft = json.loads((root / "evaluation/actual_val_student.summary.json").read_text())
    dpo_path = root / "evaluation/actual_val_dpo.summary.json"
    if dpo_path.exists():
        dpo = json.loads(dpo_path.read_text())
        candidates["dpo"] = root / "dpo/best.pt"
        if dpo["semantic_family_macro"] >= sft["semantic_family_macro"] and dpo["format_valid_rate"] >= .80: chosen = "dpo"
    write_json(root / "test_freeze.json", {"status": "frozen", "selected_eeg_mode": chosen,
        "selection": "Val semantic family macro, DPO valid>=80%; no Test selection",
        "checkpoint_sha256": {k: sha(v) for k, v in candidates.items()},
        "oracle_sha256": sha(root / "oracle/best.pt"), "test_per_dataset_cap": 128, "frozen_at": time.time(),
        "program_truth_text": {"accuracy_by_definition": 1., "llm_executed": False}})

def final_report(root):
    out = root / "evaluation"; rows = []
    for path in sorted(out.glob("*.summary.json")):
        value = json.loads(path.read_text())
        if value.get("status") == "complete": rows.append((path.name, value))
    lines = ["# Statistical semantics arm report", "", "This is learned statistical-interface capacity, not clinical report validation or inherited native text reasoning.",
             "Gold Oracle uses true label/statistics; EEG modes use only frozen raw bank and valid mask. Program-to-text truth is 100% by definition; no LLM text-baseline measurement.", "",
             "| Scope / mode | Semantic family macro | JSON valid | Queries |", "|---|---:|---:|---:|"]
    for name, value in rows:
        lines.append("| " + name + " | " + str(round(100 * value["semantic_family_macro"], 3)) + " | " +
                     str(round(100 * value["format_valid_rate"], 3)) + " | " + str(value["queries"]) + " |")
    for name, value in rows:
        lines += ["", "## " + name, "", "| Dataset | Label BAcc | Feature BAcc | Rule exact | Report exact | Report facts |",
                  "|---|---:|---:|---:|---:|---:|"]
        for ds, result in value["per_dataset"].items():
            tasks = result["tasks"]
            vals = [tasks.get("label", {}).get("bacc", 0), result["feature_macro_bacc"],
                    result["rule_accuracy"], result["report_exact_accuracy"], tasks.get("report", {}).get("fact_accuracy", 0)]
            lines.append("| " + ds + " | " + " | ".join(str(round(100 * v, 3)) for v in vals) + " |")
        lines += ["", "| Dataset / field | BAcc | Macro-F1 | Exact | Valid |", "|---|---:|---:|---:|---:|"]
        for ds, result in value["per_dataset"].items():
            for field in FIELDS:
                item = result["tasks"].get("feature:" + field)
                if item:
                    vals = [item["bacc"], item["macro_f1"], item["exact_accuracy"], item["format_valid_rate"]]
                    lines.append("| " + ds + " / " + field + " | " + " | ".join(str(round(100 * v, 3)) for v in vals) + " |")
    lines += ["", "Per-dataset/field BAcc, Macro-F1, confusion matrices, label/rule/report exact and report fact accuracy are in each summary JSON; failure examples are paired *.failures.json.",
              "Capacity comparison changes shared fusion versus separated field layout as well as token count; no pure token-count causal claim.",
              "Synthetic combination holdouts and actual original subject splits are distinct evaluation scopes. Actual Test uses the fixed per-dataset cap and is not the historical full Test ranking."]
    (root / "final_report.md").write_text("\n".join(lines) + "\n")

def create(root, arm, gpu):
    root.mkdir(parents=True, exist_ok=True)
    if (root / "protocol.json").exists(): return
    value = {"id": root.name, "arm": arm, "gpu": gpu, "seed": SEED, "decoder": str(MODEL),
        "decoder_frozen": True, "fields": FIELDS, "main_variable": "32 shared versus 112 separated tokens; layout also changes",
        "tokens": 32 if arm == "shared32" else 112, "label_tokens": 16, "per_field_codes": [3, 16, 2048],
        "oracle_parameterization": "factorized field/value codes, question-independent; shared sum/sqrt6 versus concatenate; no full-combination table",
        "student_architecture": "original label16 heavy head plus shared field-conditioned stats16 head; equal student parameter count across arms",
        "label_source": str(BASE), "student_warmstart": str(OLD), "label_source_sha256": sha(BASE),
        "student_source_sha256": sha(OLD), "oracle_steps": 4000, "student_steps": 6000, "dpo_steps": 500,
        "batch": 8, "oracle_lr": .001, "student_lr": .00001, "dpo_lr": .000002, "dpo_beta": .1,
        "tasks": TASKS, "data": str(DATA), "synthetic_split": "bins SHA256 modulo5 train2/3/4 val0 test1",
        "actual_split": "original subject-disjoint manifests; Train-only thresholds; no Test tuning",
        "actual_val_cap_per_dataset": 16, "actual_test_cap_per_dataset": 128,
        "gates": {"target": .80, "bounded_exploration": .65, "minimum_valid": .80,
            "score": "equal dataset mean(feature macro BAcc, rule accuracy, complete report exact)"},
        "budget": {"worker_hours": 12, "stage_hours": 4}, "smoke_steps": 30,
        "test": "freeze checkpoints before one final actual Test; gold Oracle and actual EEG separated",
        "shuffle_control": "fixed bank exchange within same dataset and gold label, prefer different subject and different bins; evaluation control only",
        "text_control": "program truth reference 100% by definition; no executed LLM text baseline",
        "dpo": "two actual sampled Train EEG outputs; both valid; fact score gap>=.15; >=32 pairs else skip",
        "authorization": "User expressly authorized this new statistics/token-capacity/EEG/DPO exploration; ICL excluded",
        "source_sha256": sha(__file__),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
        "dirty_workspace_preexisting": True}
    write_json(root / "protocol.json", value)
    (root / "source_snapshot.py").write_text(Path(__file__).read_text())
    (root / "PLAN.md").write_text("# Statistics arm " + arm + "\n\nUser-authorized bounded exploration under the master PLAN.\n"
        "Hypothesis: factorized ordinal statistics can expose within-label EEG variation to the frozen decoder; longer separated field tokens may improve joint reading.\n"
        "Use the same six fields, seed, code initialization, questions, losses and budgets in both arms. Capacity also changes aggregation/layout, which limits attribution.\n"
        "Pipeline: prepare -> 30-step Oracle smoke -> free AR smoke -> Oracle4000 -> synthetic Val gate -> wait for data -> actual Val Oracle gate -> EEG6000 -> actual Val -> genuine generated preferences -> optional DPO500 -> freeze -> actual Test Oracle/EEG -> report.\n"
        "Raw-bank student never receives gold stats, true labels or predicted-label code replacement. Oracle truth is teacher/evaluation only.\n"
        "Synthetic combinations are dynamically sampled with Train/Val/Test partition. Actual EEG retains original subject split and Train-only bins. All outputs stay in this arm/project.\n"
        "Target80%, bounded65–80%; semantic<65 or valid<80 stops. Report exact, fact accuracy, each field, label and rule are separate. Only Val selects; Test is once after freeze.\n"
        "Per worker12h, per stage4h, GPU" + str(gpu) + ". Resume via launch.sh; best/last optimizer/RNG saved. Existing models untouched.\n")
    write_json(root / "status.json", {"state": "ready", "stage": "created", "updated": time.time()})

def pipeline(root):
    lock = (root / "pipeline.lock").open("w"); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    first = root / "first_launch.json"
    if not first.exists(): write_json(first, {"started": time.time()})
    began = json.loads(first.read_text())["started"]; logs = root / "logs"; logs.mkdir(exist_ok=True)
    write_json(root / "launch.json", {"pid": os.getpid(), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                     "started": began, "source_sha256": sha(__file__)})
    def run(name, args, done=None, ar=False, minimum_step=0):
        if done and done.exists():
            saved = json.loads(done.read_text()) if done.suffix == ".json" else {}
            complete = saved.get("status") in ["complete", "passed", "skipped"] or saved.get("status") == "frozen"
            complete = complete and int(saved.get("step", minimum_step)) >= minimum_step
            if complete:
                print("SKIP", name, flush=True); return
        if shutil.disk_usage(root).free < 12 * 1024 ** 3:
            raise RuntimeError("Stage requires at least 12 GiB free on output filesystem")
        remaining = 12 * 3600 - (time.time() - began)
        if remaining <= 0: raise RuntimeError("12h total arm budget exhausted")
        command = [AR_PYTHON if ar else sys.executable, "-B", str(Path(__file__).resolve()), *args, "--root", str(root)]
        log = logs / (name + ".log")
        if log.exists():
            older = logs / "previous_attempts"; older.mkdir(exist_ok=True); log.rename(older / (name + "." + str(int(time.time())) + ".log"))
        write_json(root / "status.json", {"state": "running", "stage": name, "command": command,
                                         "pipeline_pid": os.getpid(), "updated": time.time(), "log": str(log)})
        with log.open("w") as output:
            child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            write_json(root / "stage_process.json", {"pid": child.pid, "stage": name, "started": time.time(), "command": command})
            try: code = child.wait(timeout=min(4 * 3600, remaining))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try: child.wait(timeout=30)
                except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL); child.wait()
                raise RuntimeError(name + " exceeded stage/worker budget")
        if code:
            write_json(root / "status.json", {"state": "stopped_gate" if code == 20 else "failed",
                    "stage": name, "exit_code": code, "log": str(log), "updated": time.time()})
            final_report(root); raise SystemExit(code)
    try:
        run("prepare", ["prepare"], root / "prepared.json")
        if not (root / "oracle/smoke.json").exists():
            run("oracle_smoke", ["train", "--phase", "oracle", "--steps", "30"])
        run("oracle_smoke_ar", ["evaluate", "--mode", "oracle", "--scope", "synthetic_smoke"],
            root / "evaluation/synthetic_smoke_oracle.summary.json", ar=True)
        run("oracle_training", ["train", "--phase", "oracle", "--steps", "4000"], root / "oracle/summary.json", minimum_step=4000)
        run("oracle_synthetic_val", ["evaluate", "--mode", "oracle", "--scope", "synthetic_val"],
            root / "evaluation/synthetic_val_oracle.summary.json", ar=True)
        run("oracle_synthetic_gate", ["gate", "--scope", "synthetic_val"])
        while not (DATA / "prepared.json").exists():
            if time.time() - began > 12 * 3600: raise RuntimeError("data wait exhausted worker budget")
            write_json(root / "status.json", {"state": "waiting_data", "stage": "data", "updated": time.time()})
            time.sleep(30)
        run("oracle_actual_val", ["evaluate", "--mode", "oracle", "--scope", "actual_val"],
            root / "evaluation/actual_val_oracle.summary.json", ar=True)
        run("oracle_actual_gate", ["gate", "--scope", "actual_val"])
        run("student_training", ["train", "--phase", "student", "--steps", "6000"], root / "student/summary.json", minimum_step=6000)
        run("student_actual_val", ["evaluate", "--mode", "student", "--scope", "actual_val"],
            root / "evaluation/actual_val_student.summary.json", ar=True)
        run("student_shuffled_actual_val", ["evaluate", "--mode", "student_shuffled", "--scope", "actual_val"],
            root / "evaluation/actual_val_student_shuffled.summary.json", ar=True)
        run("preferences", ["preferences"], root / "dpo/preferences.summary.json", ar=True)
        run("dpo_training", ["dpo"], root / "dpo/summary.json")
        dpo_summary = json.loads((root / "dpo/summary.json").read_text())
        if dpo_summary["status"] == "complete":
            run("dpo_actual_val", ["evaluate", "--mode", "dpo", "--scope", "actual_val"],
                root / "evaluation/actual_val_dpo.summary.json", ar=True)
        run("test_freeze", ["freeze"], root / "test_freeze.json")
        while not (DATA / "test.pt").exists() or not (DATA / "test_records.json").exists():
            if time.time() - began > 12 * 3600: raise RuntimeError("Test cache wait exhausted worker budget")
            write_json(root / "status.json", {"state": "waiting_test_cache", "stage": "test_data", "updated": time.time()})
            time.sleep(30)
        run("oracle_actual_test", ["evaluate", "--mode", "oracle", "--scope", "actual_test"],
            root / "evaluation/actual_test_oracle.summary.json", ar=True)
        run("student_actual_test", ["evaluate", "--mode", "student", "--scope", "actual_test"],
            root / "evaluation/actual_test_student.summary.json", ar=True)
        run("student_shuffled_actual_test", ["evaluate", "--mode", "student_shuffled", "--scope", "actual_test"],
            root / "evaluation/actual_test_student_shuffled.summary.json", ar=True)
        if dpo_summary["status"] == "complete":
            run("dpo_actual_test", ["evaluate", "--mode", "dpo", "--scope", "actual_test"],
                root / "evaluation/actual_test_dpo.summary.json", ar=True)
        final_report(root)
        write_json(root / "status.json", {"state": "complete", "stage": "complete", "updated": time.time(),
                                         "elapsed_seconds": time.time() - began})
    except SystemExit: raise
    except BaseException as e:
        write_json(root / "status.json", {"state": "failed", "stage": "pipeline", "error": repr(e),
                                         "traceback": traceback.format_exc(), "updated": time.time()})
        raise

def main():
    p = argparse.ArgumentParser(); p.add_argument("command", choices=["create", "prepare", "train", "evaluate", "gate", "preferences", "dpo", "freeze", "report", "pipeline"])
    p.add_argument("--root", type=Path, required=True); p.add_argument("--arm", choices=["shared32", "fields112"])
    p.add_argument("--gpu", type=int); p.add_argument("--phase", choices=["oracle", "student"], default="oracle")
    p.add_argument("--steps", type=int, default=4000); p.add_argument("--mode", choices=["oracle", "student", "student_shuffled", "dpo"], default="oracle")
    p.add_argument("--scope", choices=["synthetic_smoke", "synthetic_val", "actual_val", "actual_test"], default="synthetic_val")
    a = p.parse_args(); assert a.root.resolve().is_relative_to(MASTER.resolve())
    if a.command == "create": create(a.root, a.arm, a.gpu)
    elif a.command == "prepare": prepare(a.root)
    elif a.command == "train": train(a.root, a.phase, a.steps)
    elif a.command == "evaluate": evaluate(a.root, a.mode, a.scope)
    elif a.command == "gate": gate(a.root, a.scope)
    elif a.command == "preferences": collect_preferences(a.root)
    elif a.command == "dpo": dpo_train(a.root)
    elif a.command == "freeze": freeze_test(a.root)
    elif a.command == "report": final_report(a.root)
    else: pipeline(a.root)

if __name__ == "__main__": main()
