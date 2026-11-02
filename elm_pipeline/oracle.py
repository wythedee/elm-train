"""Portable unassisted text Teacher, fresh Oracle training and AR diagnostics."""
from __future__ import annotations
from collections import Counter
import hashlib
import importlib.metadata
import os
from pathlib import Path
import time
import torch
from torch import nn
from torch.nn import functional as F
from . import oracle_tasks as ot, tasks
from .io import read, sha, write
from .training import seed, tensor_state, oracle_prefix, rng_pack, rng_restore
from .backends import validate_version, engine_options


def load_pool(cfg):
    out = {}
    for split, key in (("train", "teacher_train_queries"), ("val", "teacher_val_queries")):
        records = read(Path(cfg["features"]) / (split + "_index.json"))["records"]
        out[split] = ot.pool(records, split, cfg.get(key, 120 if split == "train" else 40), cfg.get("seed", 1014))
        for item in out[split]:ot.leak_check(item)
    return out


def frozen_identity(cfg):
    paths = {"init": Path(cfg["oracle_initial_checkpoint"]),
             "train": Path(cfg["features"]) / "train_index.json", "val": Path(cfg["features"]) / "val_index.json",
             "wordings": Path(__file__).resolve().parents[1] / "metadata/oracle_wordings.json"}
    for p in Path(__file__).parent.glob("*.py"):paths["source/"+p.name] = p
    model = Path(cfg["model"])
    shards = set(read(model / "model.safetensors.index.json")["weight_map"].values())
    for name in shards | {"config.json", "tokenizer.json", "tokenizer_config.json"}:paths["model/"+name] = model/name
    return {k: {"path": str(p), "sha256": sha(p)} for k,p in paths.items()}


def verify_identity(identity):
    for row in identity.values():
        if sha(row["path"]) != row["sha256"]:raise ValueError("Oracle dependency changed: "+row["path"])


def generate(cfg, config_sha, teacher=True):
    from safetensors import safe_open
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from .evaluation import metrics
    validate_version(cfg, importlib.metadata.version("vllm"))
    out = Path(cfg["output"]) / ("text_teacher" if teacher else "oracle_evaluation")
    out.mkdir(exist_ok=False)
    pool = load_pool(cfg);write(out / "plan.json", pool)
    identity = frozen_identity(cfg)
    if not teacher:identity["trained_oracle"] = {"path":cfg["oracle"],"sha256":sha(cfg["oracle"])}
    write(out / "identity.json", identity)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], local_files_only=True)
    emb = None;bank = None
    if not teacher:
        model = Path(cfg["model"]);wm = read(model / "model.safetensors.index.json")["weight_map"]
        key = "model.embed_tokens.weight"
        with safe_open(str(model/wm[key]), framework="pt", device="cpu") as f:
            emb = nn.Embedding.from_pretrained(f.get_tensor(key), freeze=True)
        bank = tensor_state(cfg["oracle"])
    engine = LLM(**engine_options(cfg, os.environ["ELM_AR_UTIL"]))
    params = SamplingParams(temperature=0,max_tokens=384,seed=cfg.get("seed",1014))
    pending = [(split,e) for split in (("train","val") if teacher else ("val",)) for e in pool[split]]
    raw = [];batches = 0
    for start in range(0,len(pending),16):
        if time.time()>float(os.environ["ELM_STAGE_DEADLINE"]):raise TimeoutError("Original stage deadline")
        part=pending[start:start+16];prompts=[]
        for split,item in part:
            if teacher:
                text=tokenizer.apply_chat_template(ot.messages(item,True),tokenize=False,add_generation_prompt=True,enable_thinking=False)
                prompts.append({"prompt_token_ids":tokenizer(text,add_special_tokens=False)["input_ids"]})
            else:
                with torch.no_grad():
                    pp=[oracle_prefix([r],bank,"cpu") for r in item["records"]]
                    prompts.append({"prompt_embeds":ot.render(tokenizer,emb,item,pp).bfloat16()})
        generated=engine.generate(prompts,params,use_tqdm=False);assert len(generated)==len(part)
        batches+=1
        for (split,item),p,g in zip(part,prompts,generated):
            text=g.outputs[0].text
            row={"id":item["id"],"task":item["task"],"dataset":item["record"]["dataset"],"split":split,
                 "question":item["question"],"expected":item["expected"],"text":text,
                 "prompt_sha256":hashlib.sha256(str(p["prompt_token_ids"]).encode()).hexdigest() if teacher else hashlib.sha256(p["prompt_embeds"].float().numpy().tobytes()).hexdigest(),
                 "source":"text_teacher" if teacher else "oracle", "config_sha256":config_sha, **ot.score(text,item)}
            raw.append(row)
            with (out/"raw.partial.jsonl").open("a") as f:f.write(tasks.dumps(row)+"\n")
        print("GENERATED",start+len(part),"/",len(pending),flush=True)
    if len(raw)!=len(pending):raise ValueError("Incomplete generation")
    verify_identity(identity)
    (out/"raw.partial.jsonl").rename(out/"raw.jsonl")
    groups={split:metrics([r for r in raw if r["split"]==split]) for split in sorted({r["split"] for r in raw})}
    if teacher:
        byid={e["id"]:e for e in pool["train"]}
        correct=[{**byid[r["id"]],"teacher_text":r["text"]} for r in raw if r["split"]=="train" and r["correct"]]
        if not correct:raise ValueError("No unassisted correct Train teacher rows")
        write(out/"correct_train.json",correct)
    result={"status":"complete","queries":len(raw),"generation_batches":batches,"metrics":groups,
            "unassisted_same_frozen_4B":teacher,"all_declared_Val_retained":True,"Test":"N/A",
            "correct_Train":len(correct) if teacher else None}
    write(out/"summary.json",result);return result


class OracleBank(nn.Module):
    def __init__(self, initial):
        super().__init__()
        self.label_codes=nn.Parameter(initial["label_codes"].float().clone())
        self.stat_codes=nn.Parameter(initial["stat_codes"].float().clone())
        self.register_buffer("label_position",initial["label_position"].float().clone())
        self.register_buffer("initial_label",self.label_codes.detach().clone(),persistent=False)
        self.register_buffer("initial_stats",self.stat_codes.detach().clone(),persistent=False)
        if self.label_codes.shape!=(31,32,2560) or self.stat_codes.shape!=(6,3,8,2560):raise ValueError("Fresh 80-token Oracle layout changed")
    def prefix(self,record):
        return torch.cat([
            self.label_codes[record["global_label_id"]]+self.label_position,
            torch.cat([self.stat_codes[i,tasks.LEVELS.index(record["bins"][f])] for i,f in enumerate(tasks.FIELDS)])])[None]
    def anchor(self):
        return ((self.label_codes-self.initial_label).square().mean()/self.initial_label.square().mean().clamp_min(1e-8)
                +(self.stat_codes-self.initial_stats).square().mean()/self.initial_stats.square().mean().clamp_min(1e-8))


def train(cfg,config_sha):
    import random
    import numpy as np
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM,AutoTokenizer
    out=Path(cfg["output"])/"oracle_training";out.mkdir(exist_ok=False)
    steps=cfg.get("oracle_steps",10)
    if not cfg.get("software_debug") or not 1<=steps<=10:raise ValueError("This new implementation is authorized for <=10 engineering steps only")
    seed(cfg.get("seed",1014));private=random.Random(cfg.get("seed",1014))
    total=torch.cuda.get_device_properties(0).total_memory/1024**3
    torch.cuda.set_per_process_memory_fraction(min(.98,float(os.environ["ELM_ALLOCATOR_GIB"])/total),0)
    identity=frozen_identity(cfg)
    initial=tensor_state(cfg["oracle_initial_checkpoint"]);net=OracleBank(initial).cuda()
    lm=AutoModelForCausalLM.from_pretrained(cfg["model"],local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation="sdpa").cuda().eval()
    lm.requires_grad_(False);lm.config.use_cache=False;emb=lm.get_input_embeddings()
    tok=AutoTokenizer.from_pretrained(cfg["model"],local_files_only=True)
    data=read(Path(cfg["output"])/"text_teacher/correct_train.json")
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
    all_slots=torch.zeros(80,device="cuda");order=[];cursor=0;trace=[]
    def logits(item,answer,prefixes=None,teacher=False):
        x=ot.render(tok,emb,item,prefixes,teacher)
        xx=torch.cat([x,emb(answer).detach()])[None]
        return lm(inputs_embeds=xx,attention_mask=torch.ones(xx.shape[:2],device=xx.device,dtype=torch.long),use_cache=False).logits[0,len(x)-1:len(x)+len(answer)-1].float()
    for step in range(steps):
        if time.time()>float(os.environ["ELM_STAGE_DEADLINE"]):raise TimeoutError("Original Oracle stage deadline")
        opt.zero_grad(set_to_none=True);losses=[];ids=[]
        for slot in range(8):
            if cursor>=len(order):order=list(range(len(data)));private.shuffle(order);cursor=0
            item=data[order[cursor]];cursor+=1;ids.append(item["id"])
            answer=torch.tensor(tok(item["teacher_text"],add_special_tokens=False)["input_ids"]+[tok.eos_token_id],device="cuda")
            with torch.autocast("cuda",dtype=torch.bfloat16):
                with torch.no_grad():teacher=logits(item,answer,teacher=True)
                pp=[net.prefix(r) for r in item["records"]]
                for p in pp:p.retain_grad()
                student=logits(item,answer,pp)
                ce=F.cross_entropy(student,answer)
                kl=F.kl_div(F.log_softmax(student,-1),F.softmax(teacher,-1),reduction="batchmean")
                loss=(ce+.5*kl+.0001*net.anchor())/8
            if not torch.isfinite(loss):raise ValueError("Non-finite Oracle loss")
            loss.backward()
            for p in pp:all_slots+=p.grad.detach().float().abs().sum((0,2))
            losses.append([float(ce.detach()),float(kl.detach())])
        if any(p.grad is not None or p.requires_grad for p in lm.parameters()):raise AssertionError("LM is not frozen")
        if not all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in net.parameters()):raise AssertionError("Oracle gradients missing")
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
        row={"step":step+1,"sample_ids":ids,"ce":sum(x[0] for x in losses)/8,"KL":sum(x[1] for x in losses)/8,"time":time.time()}
        trace.append(row);print(tasks.dumps(row),flush=True)
        with (out/"trace.jsonl").open("a") as f:f.write(tasks.dumps(row)+"\n")
    state={k:v.detach().cpu() for k,v in net.state_dict().items()}
    checkpoint={"state_dict":state,"optimizer":opt.state_dict(),"step":steps,"config_sha256":config_sha,
                "identity":identity,"order":order,"cursor":cursor,**rng_pack(private)}
    torch.save(checkpoint,out/"last.pt.partial");(out/"last.pt.partial").replace(out/"last.pt")
    check=torch.load(out/"last.pt",map_location="cpu",weights_only=False)
    restored=OracleBank(initial).cuda();restored.load_state_dict(check["state_dict"],strict=True)
    opt2=torch.optim.AdamW(restored.parameters(),lr=.001,weight_decay=.0001);opt2.load_state_dict(check["optimizer"])
    assert all(int(v["step"])==steps for v in opt2.state.values())
    assert all(torch.equal(v.detach().cpu(),check["state_dict"][k]) for k,v in restored.state_dict().items())
    rng_restore(check,private);a=(random.random(),torch.rand(4),torch.rand(4,device="cuda"),private.random(),np.random.random())
    rng_restore(check,private);b=(random.random(),torch.rand(4),torch.rand(4,device="cuda"),private.random(),np.random.random())
    assert a[0]==b[0] and torch.equal(a[1],b[1]) and torch.equal(a[2],b[2]) and a[3:]==b[3:]
    assert bool((all_slots>0).all())
    verify_identity(identity)
    dest=Path(cfg["oracle"]);dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():raise FileExistsError("Preserve existing Oracle export")
    tmp=dest.with_suffix(".partial.safetensors");save_file(state,str(tmp));tmp.replace(dest)
    result={"status":"committed","step":steps,"batch_size":8,"LM_frozen":True,"all_80_slots_received_gradients":True,
            "optimizer_all_rng_reload_passed":True,"fresh_init_sha256":sha(cfg["oracle_initial_checkpoint"]),
            "oracle_export_sha256":sha(dest),"training_examples":len(data),"Test":"N/A","on_policy_exercised":False,
            "scope":"10-step software implementation check, not long-training or scientific validation"}
    write(out/"result.json",result);return result
