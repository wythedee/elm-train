#!/usr/bin/env python3
"""Frozen real/synthetic candidates; per-example teacher filter, never template-wide filtering."""

if __name__ == "__main__":
    raise SystemExit("Reference snapshot only. Use tools/workbench.py for inspection; configure a fresh run before training.")

import collections,copy,hashlib,importlib.util,itertools,json,random,sys,time
from pathlib import Path
P=Path("/data/yuqiudeng/vstats-elm")
PLAN=P/"experiments/mainline_vrl/plans/diagnostics/oracle4b_learnability_20261024_s1014"
ROOT=P/"experiments/mainline_vrl/runs/oracle4b80_simple_20261024_s1014"
R=ROOT/"label32_stat8"
sys.path.insert(0,str(P/"experiments/mainline_vrl/scripts/python"))
import oracle_curriculum28_tasks_1016_v3 as legacy
base=legacy.base
DATA=legacy.old.DATA
FIELDS=legacy.FIELDS;LEVELS=["low","mid","high"];OPS=["read_level","compare_level","choose_higher","choose_lower"]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open("x") as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write("\n")
def key(x):return hashlib.sha256(str(x).encode()).hexdigest()
def compact(r):return {k:copy.deepcopy(r[k]) for k in ["id","dataset","subject","split","label","label_id","global_label_id","bins"] if k in r}
def synthetic(label_id,index,field=None,level=None):
 rng=random.Random(1014+index*731+label_id*211)
 labels=read(legacy.old.R/"protocol.json")["labels"]
 while True:
  r=base.synthetic_record(rng,labels,"train",index)
  r["bins"]={f:LEVELS[int(r["bins"][f])] for f in FIELDS}
  if field is None or r["bins"][field]==level:break
 r.update(label=labels[label_id],label_id=label_id,global_label_id=label_id,subject="synthetic_train",synthetic=True)
 return compact(r)
def build():
 train=read(DATA/"train_records.json");val=read(DATA/"val_records.json")
 assert {r["id"] for r in train}.isdisjoint(r["id"] for r in val)
 assert {(r["dataset"],r["subject"]) for r in train}.isdisjoint((r["dataset"],r["subject"]) for r in val)
 template=read(PLAN/"screen_qwen3_4b_v6/normalized/templates.json")["templates"]
 groups=read(PLAN/"screen_qwen3_4b_v6/inputs/wording_partition.json")
 # All sixteen Train and four Val wordings retained even if one exhaustive case failed.
 def choose(rs,field,level,seed,exclude=None):
  possible=[r for r in rs if r["bins"][field]==level and r["id"]!=exclude]
  assert possible
  return compact(possible[int(key(seed)[:12],16)%len(possible)])
 out={}
 for split,rs in [("train",train),("val",val)]:
  examples=[]
  for row in template:
   if groups[row["template_id"]]!=split:continue
   op=row["operation"]
   for fi,field in enumerate(FIELDS):
    for li,levels in enumerate(itertools.product(LEVELS,repeat=1 if op=="read_level" else 2)):
     seed=split+":"+row["template_id"]+":"+field+":"+str(levels)
     rec=[]
     for j,level in enumerate(levels):
      i=int(key(seed+":"+str(j))[:12],16)
      if split=="train" and i%4==0:r=synthetic(i%31,100000+i%10000000,field,level)
      else:r=choose(rs,field,level,seed+":"+str(j),rec[0]["id"] if rec else None)
      rec.append(r)
     e=dict(id="simple:"+split+":"+key(seed),kind="simple",operation=op,field=field,template_id=row["template_id"],
      template_wording=row["wording"],records=rec,split=split,scope="simple_unseen_wording" if split=="val" else "simple_train_wording",
      dataset=rec[0]["dataset"],wording_partition=split)
     examples.append(e)
  selected={}
  byds=collections.defaultdict(list);bylabel=collections.defaultdict(list)
  for r in rs:byds[r["dataset"]].append(r);bylabel[r["global_label_id"]].append(r)
  for ds,rr in sorted(byds.items()):
   for r in sorted(rr,key=lambda r:key(split+":basic:"+r["id"]))[:8]:selected[r["id"]]=compact(r)
  if split=="train":
   for ll,rr in sorted(bylabel.items()):
    for r in sorted(rr,key=lambda r:key("train:label:"+r["id"]))[:4]:selected[r["id"]]=compact(r)
   for ll in range(31):
    for k in range(3):
     r=synthetic(ll,20000000+ll*3+k);selected[r["id"]]=r
  for r in sorted(selected.values(),key=lambda r:r["id"]):
   for q in base.fixed_questions(r):
    if q["task"]=="rule":continue
    examples.append(dict(id="basic:"+split+":"+r["id"]+":"+q["task"],kind="base",q=q,records=[r],split=split,scope="original_base",dataset=r["dataset"]))
  assert len({e["id"] for e in examples})==len(examples)
  write(R/(split+"_candidates.json"),examples);out[split]=examples
 trainids={r["id"] for e in out["train"] for r in e["records"]};valids={r["id"] for e in out["val"] for r in e["records"]}
 assert trainids.isdisjoint(valids)
 result=dict(status="candidates_frozen_before_new4b_answers",time=time.time(),source_sha256=sha(__file__),
 counts={s:dict(collections.Counter(e["operation"] if e["kind"]=="simple" else e["q"]["task"] for e in ee)) for s,ee in out.items()},
 train_val_record_ids_disjoint=True,train_val_subjects_disjoint=True,all96_original27b_wording_preserved=True,
 sample_correct_filter_not_template_filter=True,val_not_filtered_by_text_correctness=True,
 real_labels=len({r["global_label_id"] for r in train}),synthetic_train_labels=31,
 candidate_sha256={s:sha(R/(s+"_candidates.json")) for s in out},
 template_sha256=sha(PLAN/"screen_qwen3_4b_v6/normalized/templates.json"),partition_sha256=sha(PLAN/"screen_qwen3_4b_v6/inputs/wording_partition.json"))
 write(R/"candidate_build.json",result);return result
if __name__=="__main__":print(json.dumps(build()))
