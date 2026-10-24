#!/usr/bin/env python3
"""Statistical semantic targets from exact current MoE windows; Train/Val only by default."""
from __future__ import annotations

if __name__ == "__main__":
    raise SystemExit("Reference snapshot only. Use tools/workbench.py for inspection; configure a fresh run before training.")

import argparse, collections, fcntl, hashlib, json, math, os, pickle, random, sys, time
from pathlib import Path
import h5py, lmdb, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample_poly, welch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
try:
 __import__("numpy._core.numeric")
except ModuleNotFoundError:
 sys.modules.setdefault("numpy._core",np.core)
 sys.modules.setdefault("numpy._core.numeric",np.core.numeric)
 sys.modules.setdefault("numpy._core.multiarray",np.core.multiarray)
FIELDS=["alpha_relative_power","theta_beta_ratio","spectral_entropy","hjorth_mobility","hjorth_complexity","line_length"]
KV=Path("/var/tmp/yuqiudeng_vstats/braincot16_icl_0907/kv/manifest.json")
RAW=Path("/home/workspace/datasets/yuqiudeng/vstats_experiments/braincot_expanded_moe16_0904/manifests")
OLD=Path("/var/tmp/yuqiudeng_vstats/braincot16_icl_0907/oracle/icl_oracle/codebook.pt")
SEED=1014
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 q=p.with_name(p.name+".tmp");q.write_text(json.dumps(x,indent=2,ensure_ascii=False)+"\n");q.replace(p)
def save(p,x):
 p=Path(p)
 estimated=sum(v.numel()*v.element_size() for v in x.values() if isinstance(v,torch.Tensor)) if isinstance(x,dict) else 0
 free=os.statvfs(p.parent).f_bavail*os.statvfs(p.parent).f_frsize
 if free < 12*1024**3+estimated:raise OSError(f"storage guard: free={free}, estimatedadditional={estimated}, reserve=12GiB")
 q=p.with_name(p.name+".tmp");torch.save(x,q);q.replace(p)
def hashids(xs):return hashlib.sha256("\n".join(xs).encode()).hexdigest()
def seed():
 random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(4)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(SEED)
def choose(rows,n):
 groups=collections.defaultdict(list)
 for r in rows:groups[r["label"]].append(r)
 for lab,g in groups.items():g.sort(key=lambda r:hashlib.sha256((str(SEED)+":"+r["id"]).encode()).hexdigest())
 result=[];keys=sorted(groups)
 for k in range(max(map(len,groups.values()))):
  for lab in keys:
   if k<len(groups[lab]):result.append(groups[lab][k])
   if len(result)>=n:return result
 return result
def load_signal(row,cache):
 if row.get("lmdb_path"):
  p=row["lmdb_path"]
  if p not in cache:cache[p]=lmdb.open(p,readonly=True,lock=False,readahead=False,meminit=False)
  with cache[p].begin() as t:v=t.get(str(row["lmdb_key"]).encode())
  if v is None:raise KeyError(row["id"])
  x=np.asarray(pickle.loads(v)["sample"],dtype=np.float32)
  x=x.reshape(x.shape[0],-1)*float(row.get("input_scale",1))
 else:
  p=row["h5_path"]
  if p not in cache:cache[p]=h5py.File(p,"r")
  g=cache[p][str(row["subject"])]["X"];i=int(row["trial_idx"]);n=int(row.get("time_concat_count") or 1)
  if n>1:
   z=np.asarray(g[i:i+n],dtype=np.float32);x=z.transpose(1,0,2).reshape(z.shape[1],-1)
  else:x=np.asarray(g[i],dtype=np.float32)
 start=int(row.get("window_start") or 0);length=row.get("window_length")
 if length is not None:
  assert 0<=start and start+int(length)<=x.shape[-1],row["id"]
  x=x[:,start:start+int(length)]
 sr=int(row.get("source_sampling_rate") or row.get("sampling_rate"))
 tr=int(row["sampling_rate"]);assert tr==200,(row["id"],tr)
 if row.get("resample_on_load") and sr!=tr:
  d=math.gcd(sr,tr);x=resample_poly(x,tr//d,sr//d,axis=-1).astype(np.float32)
 # Targets describe only measured valid samples, excluding explicit centered padding
 # and pre-existing all-channel zero margins; retain exact window bounds otherwise.
 allzero=np.all(x==0,axis=0)
 nz=np.flatnonzero(~allzero)
 assert len(nz)>4,"empty/constant EEG"
 l,r=int(nz[0]),int(nz[-1])+1;x=x[:,l:r]
 indices=row.get("canonical_channel_indices")
 channel_valid=np.isfinite(x).all(1)&(x.std(1)>1e-8)
 if indices is not None:
  assert len(indices)==x.shape[0]
  channel_valid&=(np.asarray(indices)>=0)&(np.asarray(indices)<75)
 x=x[channel_valid]
 assert len(x)>0 and x.shape[-1]>=100,"no valid channels/short window"
 # Per valid channel normalization prevents different input units and zero-channel padding
 # from changing the semantic target. This is normalized shape, not physical amplitude.
 x=(x-x.mean(1,keepdims=True))/(x.std(1,keepdims=True)+1e-6)
 info={"valid_channel_count":len(x),"source_channel_count":len(channel_valid),
       "valid_channel_indices":np.flatnonzero(channel_valid).tolist(),
       "valid_samples":x.shape[-1],"zero_margin_left":l,"zero_margin_right":int(len(allzero)-r),
       "declared_pad_to_length":row.get("pad_to_length"),"sampling_rate":200}
 return x.astype(np.float64),info
def numeric(x):
 freq,p=welch(x,fs=200,nperseg=min(400,x.shape[-1]),noverlap=min(200,min(400,x.shape[-1])//2),axis=-1)
 p=p.mean(0);valid=(freq>=1)&(freq<45);total=p[valid].sum()+1e-12
 def band(a,b):return p[(freq>=a)&(freq<b)].sum()/total
 dist=p[valid]/total
 dx=np.diff(x,axis=-1);dd=np.diff(dx,axis=-1);vx=x.var(1)+1e-12;vd=dx.var(1)+1e-12
 mobility=np.sqrt(vd/vx);complexity=np.sqrt((dd.var(1)+1e-12)/vd)/(mobility+1e-12)
 out=np.array([band(8,13),band(4,8)/(band(13,30)+1e-8),
 -(dist*np.log(dist+1e-12)).sum()/np.log(len(dist)),mobility.mean(),complexity.mean(),np.abs(dx).mean()])
 assert np.isfinite(out).all()
 return out
def threshold(train):
 pieces=[];weights=[]
 for x in train.values():
  v=x["features"].numpy();pieces.append(v);weights.append(np.full(len(v),1/len(v)))
 a=np.concatenate(pieces);w=np.concatenate(weights);out=[]
 for f in range(6):
  order=np.argsort(a[:,f]);vals=a[order,f];cdf=np.cumsum(w[order]);cdf/=cdf[-1]
  out.append([float(vals[min(len(vals)-1,np.searchsorted(cdf,q))]) for q in [1/3,2/3]])
 return np.asarray(out)
def counts(x):return {f:torch.bincount(x[:,i],minlength=3).tolist() for i,f in enumerate(FIELDS)}
def prepare(root,train_max=2048,val_max=128,test=False):
 root=Path(root);root.mkdir(parents=True,exist_ok=True)
 m=json.loads(KV.read_text());old=torch.load(OLD,map_location="cpu",weights_only=False)
 labels=list(old["slot_values"]["label"]);ds=m["datasets"]
 schema={"fields":FIELDS,"bin_labels":["low","mid","high"],"labels":labels,"datasets":ds,
 "dataset_labels":{d:m["dataset_manifests"][d]["labels"] for d in ds},
 "tensor_keys":{"bank":"float16[N,1,150,50]","valid":"bool[N,150]","coarse":"int64[N], original global31 label order",
 "features":"float32[N,6] continuous","bins":"int64[N,6] 0/1/2","dataset_ids":"int64[N]","local_labels":"int64[N]"},
 "records":"JSON list in exact tensor row order; features/bins dictionaries; coarse/global_label_id and label_id local",
 "target_semantics":"Welch PSD 1<=f<45Hz, all finite nonconstant canonical-valid channels equally pooled; measured window samples only; per valid-channel z-score; no adjacent-channel assumption; normalized signal shape, not absolute amplitude",
 "selection":"SHA256(seed:id) sorted within local class, round robin classes, per dataset capped",
 "source_sampling":"all original subject splits fixed; all sixteen sources treated as source Train (no claim of dataset OOD)",
 "test_policy":"Test targets produced only after schema/thresholds frozen; maximum 128/ds declared subset, never fullTest"}
 write(root/"data_schema.json",schema)
 if not (root/"base_codes.pt").exists():save(root/"base_codes.pt",{"codes":old["codes"]["label"].float(),"positions":old["sample_position_embeddings"].float(),"labels":labels})
 splits=["test"] if test else ["train","val"]
 allcache={};audit={"datasets":{},"seed":SEED,"source_kv":str(KV),"source_kv_sha256":sha(KV),"raw_root":str(RAW),"code_sha256":sha(__file__),"fields":FIELDS,"test_generated":test}
 root_stage=root/"partial";root_stage.mkdir(exist_ok=True)
 prepare_started=time.time()
 for split in splits:
  for di,d in enumerate(ds):
   if time.time()-prepare_started>4*3600:raise TimeoutError("data preparation stage4h")
   part=root_stage/(split+"_"+d+".pt");recs=root_stage/(split+"_"+d+"_records.json")
   raw=RAW/d/(split+".jsonl");rows=[json.loads(l) for l in raw.read_text().splitlines() if l.strip()]
   source_subjects={r["subject"] for r in rows};km=m["dataset_manifests"][d]
   assert len(rows)==km["splits"][split]["samples"],(d,split,len(rows),km["splits"][split]["samples"])
   assert set(map(str,source_subjects))==set(map(str,km["split_audit"][split]["subject_ids"])),(d,split,"subject source disagreement")
   selected=choose(rows,128 if split=="test" else train_max if split=="train" else val_max)
   wanted={r["id"]:r for r in selected};aligned={}
   if part.exists() and recs.exists():
    payload=torch.load(part,map_location="cpu",weights_only=False);records=json.loads(recs.read_text())
   else:
    for sh in km["splits"][split]["shards"]:
     blob=torch.load(sh["path"],map_location="cpu",weights_only=False)
     for i,id_ in enumerate(blob["ids"]):
      if id_ in wanted:
       r=wanted[id_];assert str(blob["subjects"][i])==str(r["subject"])
       local=int(blob["labels"][i]);assert km["labels"][local]==r["label"],(d,id_,"label mismatch")
       aligned[id_]=(blob["bank_kv"][i].to(torch.float16),blob["valid_mask"][i].bool(),local)
    assert set(wanted)==set(aligned),(d,split,"id coverage")
    bs=[];vs=[];cs=[];fs=[];ls=[];records=[];rejected=[]
    for r in selected:
     try:x,info=load_signal(r,allcache);feat=numeric(x)
     except Exception as e:rejected.append({"id":r["id"],"error":repr(e)});continue
     bank,valid,local=aligned[r["id"]];global_i=labels.index(r["label"])
     bs.append(bank);vs.append(valid);cs.append(global_i);fs.append(feat);ls.append(local)
     records.append({"id":r["id"],"dataset":d,"subject":str(r["subject"]),"split":split,"label":r["label"],
       "label_id":local,"coarse":global_i,"global_label_id":global_i,"features":dict(zip(FIELDS,map(float,feat))),"signal":info})
    assert records,(d,split,"no valid records")
    payload={"bank":torch.stack(bs),"valid":torch.stack(vs),"coarse":torch.tensor(cs),"features":torch.tensor(np.asarray(fs),dtype=torch.float32),
      "dataset_ids":torch.full((len(records),),di,dtype=torch.long),"local_labels":torch.tensor(ls)}
    save(part,payload);write(recs,records);write(root_stage/(split+"_"+d+"_rejected.json"),rejected)
   allcachekey=(split,d);audit["datasets"].setdefault(d,{})[split]={"source_count":len(rows),"selected_count":len(records),
    "source_subject_count":len(source_subjects),"selected_subject_count":len({r["subject"] for r in records}),
    "selected_ids_sha256":hashids([r["id"] for r in records]),"source_ids_sha256":hashids([r["id"] for r in rows]),"raw_manifest_sha256":sha(raw),
    "channel_valid_min":min(r["signal"]["valid_channel_count"] for r in records),"channel_valid_max":max(r["signal"]["valid_channel_count"] for r in records)}
   print(json.dumps({"stage":"features","split":split,"dataset":d,"selected":len(records)}),flush=True)
   write(root/"progress.json",{"stage":"prepare","split":split,"dataset":d,"updated":time.time()})
 if test:
  cuts=np.array(json.loads((root/"thresholds.json").read_text())["cuts"])
 else:
  tr={d:torch.load(root_stage/("train_"+d+".pt"),map_location="cpu",weights_only=False) for d in ds}
  cuts=threshold(tr);del tr
  assert (cuts[:,0]<cuts[:,1]).all(),cuts
  write(root/"thresholds.json",{"fields":FIELDS,"cuts":cuts.tolist(),"method":"weighted empirical pooled Train quantiles; each dataset total weight1; no Val/Test fit","train_max_per_dataset":train_max})
 manifest={"format":"statsem_1014_v1","datasets":ds,"labels":labels,"feature_names":FIELDS,"dataset_labels":schema["dataset_labels"],"splits":{},"source_kv_sha256":sha(KV),"thresholds_sha256":sha(root/"thresholds.json"),"code_sha256":sha(__file__)}
 if test and (root/"manifest.json").exists():manifest=json.loads((root/"manifest.json").read_text())
 for split in splits:
  pieces=[];records=[]
  for d in ds:
   p=torch.load(root_stage/(split+"_"+d+".pt"),map_location="cpu",weights_only=False)
   rr=json.loads((root_stage/(split+"_"+d+"_records.json")).read_text())
   a=p["features"].numpy();bins=((a>cuts[:,0])+(a>cuts[:,1]).astype(np.int64)).astype(np.int64)
   # Explicit integer addition: NumPy boolean addition would otherwise be logical OR.
   bins=(a>cuts[:,0]).astype(np.int64)+(a>cuts[:,1]).astype(np.int64)
   p["bins"]=torch.tensor(bins);pieces.append(p)
   for i,r in enumerate(rr):r["bins"]={f:["low","mid","high"][int(bins[i,j])] for j,f in enumerate(FIELDS)}
   records.extend(rr);audit["datasets"][d][split]["bins"]=counts(p["bins"])
  merged={k:torch.cat([p[k] for p in pieces]) for k in pieces[0]};del pieces
  assert torch.isfinite(merged["features"]).all() and torch.isfinite(merged["bank"]).all()
  save(root/(split+".pt"),merged);write(root/(split+"_records.json"),records)
  manifest["splits"][split]={"path":str(root/(split+".pt")),"records":str(root/(split+"_records.json")),"samples":len(records),"ids_sha256":hashids([r["id"] for r in records]),"dataset_counts":dict(collections.Counter(r["dataset"] for r in records))}
  print(json.dumps({"stage":"merged","split":split,"samples":len(records)}),flush=True)
 # Validate source subject isolation from original KV audit, without opening Test signals.
 for d in ds:
  sa=m["dataset_manifests"][d]["split_audit"]
  for a,b in [("train","val"),("train","test"),("val","test")]:
   assert not (set(map(str,sa[a]["subject_ids"]))&set(map(str,sa[b]["subject_ids"]))),(d,a,b,"subject overlap")
 audit["source_subject_isolation"]="passed";audit["thresholds_sha256"]=sha(root/"thresholds.json")
 write(root/("test_audit.json" if test else "audit.json"),audit);write(root/"manifest.json",manifest)
 write(root/("test_prepared.json" if test else "prepared.json"),{"status":"complete","splits":manifest["splits"],"thresholds_sha256":sha(root/"thresholds.json"),"source_kv_sha256":sha(KV),"schema_sha256":sha(root/"data_schema.json"),"finished":time.time()})
def metrics(pred,truth,records,labels):
 out={}
 for d in sorted({r["dataset"] for r in records}):
  ii=np.array([i for i,r in enumerate(records) if r["dataset"]==d]);g=truth[ii];p=pred[ii]
  fs={}
  for j,f in enumerate(FIELDS):
   gg=g[:,j];pp=p[:,j];present=np.unique(gg)
   b=float(balanced_accuracy_score(gg,pp))
   fs[f]={"bacc":b,"macro_f1":float(f1_score(gg,pp,labels=[0,1,2],average="macro",zero_division=0)),
    "support":np.bincount(gg,minlength=3).tolist(),"present_bin_count":len(present),"confusion":confusion_matrix(gg,pp,labels=[0,1,2]).tolist()}
  out[d]={"samples":len(ii),"features":fs,"feature_macro_bacc":float(np.mean([x["bacc"] for x in fs.values()]))}
 return {"dataset_macro_feature_bacc":float(np.mean([x["feature_macro_bacc"] for x in out.values()])),"per_dataset":out}
class Probe(nn.Module):
 def __init__(self,nlabels):
  super().__init__();self.body=nn.Sequential(nn.Linear(7500,256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,128),nn.GELU())
  self.stat=nn.Linear(128,18);self.label=nn.Linear(128,nlabels)
 def forward(self,x):
  h=self.body(x);return self.stat(h).reshape(-1,6,3),self.label(h)
def probe(root,steps=1200):
 root=Path(root);out=root/"probe";out.mkdir(exist_ok=True)
 assert (root/"prepared.json").exists();seed()
 m=json.loads((root/"manifest.json").read_text());train=torch.load(root/"train.pt",map_location="cpu",weights_only=False);val=torch.load(root/"val.pt",map_location="cpu",weights_only=False)
 records=json.loads((root/"val_records.json").read_text());trrecs=json.loads((root/"train_records.json").read_text())
 # Fixed dataset+label-only conditional mode baseline measures label shortcut in targets.
 maps={}
 for r,bb in zip(trrecs,train["bins"].numpy()):
  key=(r["dataset"],r["label"]);maps.setdefault(key,[]).append(bb)
 modes={k:np.array([np.bincount(np.array(v)[:,f],minlength=3).argmax() for f in range(6)]) for k,v in maps.items()}
 baseline=np.stack([modes[(r["dataset"],r["label"])] for r in records])
 write(out/"label_only_baseline.json",metrics(baseline,val["bins"].numpy(),records,m["labels"]))
 tx=(train["bank"]*train["valid"][:,None,:,None]).reshape(-1,7500).float();vx=(val["bank"]*val["valid"][:,None,:,None]).reshape(-1,7500).float()
 mean=tx.mean(0);std=tx.std(0).clamp_min(.01);tx=((tx-mean)/std).clamp(-10,10);vx=((vx-mean)/std).clamp(-10,10)
 device="cuda";net=Probe(len(m["labels"])).to(device);opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=.01)
 ty=train["bins"].to(device);tl=train["coarse"].to(device);tx=tx.to(device);vx=vx.to(device)
 groups=collections.defaultdict(list)
 for i,(ds,lab) in enumerate(zip(train["dataset_ids"].tolist(),train["local_labels"].tolist())):groups[(ds,lab)].append(i)
 dsgroups=collections.defaultdict(list)
 for k,v in groups.items():dsgroups[k[0]].append(v)
 dskeys=sorted(dsgroups);rng=random.Random(SEED)
 history=[];best=-1;start=time.time();begin=1
 if (out/"last.pt").exists():
  ck=torch.load(out/"last.pt",map_location="cpu",weights_only=False);net.load_state_dict(ck["model"]);opt.load_state_dict(ck["optimizer"]);rng.setstate(ck["rng"]);torch.set_rng_state(ck["torch_rng"]);torch.cuda.set_rng_state(ck["cuda_rng"]);begin=int(ck["step"])+1
  if (out/"status.json").exists():
   prior=json.loads((out/"status.json").read_text());history=prior.get("history",[]);best=prior.get("best_val_feature_bacc",-1)
  write(out/"resume.json",{"from_step":begin-1,"optimizer_rng_restored":True,"checkpoint_source_sha256":ck.get("source_sha256"),"time":time.time()})
 write(out/"metadata.json",{"steps":steps,"batch":128,"lr":3e-4,"seed":SEED,"GPU":"4","source_sha256":sha(__file__),"selection":"Val dataset/feature macro BAcc; Test unopened","input":"post-MoE bank valid-masked flattened only; Train-fitted standardization; no true label or stats in input","loss":"six equally weighted 3class CE +0.25 global-label CE","budget_hours":4})
 for step in range(begin,steps+1):
  net.train();ii=[]
  for _ in range(128):
   d=rng.choice(dskeys);g=rng.choice(dsgroups[d]);ii.append(rng.choice(g))
  ii=torch.tensor(ii,device=device);opt.zero_grad();s,l=net(tx[ii])
  loss=F.cross_entropy(s.reshape(-1,3),ty[ii].reshape(-1))+.25*F.cross_entropy(l,tl[ii])
  assert torch.isfinite(loss);loss.backward()
  assert sum(float(p.grad.abs().sum()) for p in net.parameters() if p.grad is not None)>0
  torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
  if step==30:write(out/"smoke.json",{"status":"passed","steps":30,"finite_loss":float(loss),"nonzero_gradients":True})
  if step%50==0:print(json.dumps({"stage":"probe","step":step,"loss":float(loss),"seconds":time.time()-start}),flush=True)
  if step%100==0 or step==steps:
   net.eval();ss=[];ll=[]
   with torch.no_grad():
    for i in range(0,len(vx),256):
     a,b=net(vx[i:i+256]);ss.append(a.argmax(-1).cpu());ll.append(b.argmax(-1).cpu())
   pred=torch.cat(ss).numpy();label=torch.cat(ll).numpy();report=metrics(pred,val["bins"].numpy(),records,m["labels"])
   report["label_per_dataset"]={}
   for d in dskeys:
    ids=np.where(val["dataset_ids"].numpy()==d)[0];allowed=[m["labels"].index(x) for x in m["dataset_labels"][m["datasets"][d]]]
    report["label_per_dataset"][m["datasets"][d]]={"bacc":float(balanced_accuracy_score(val["coarse"].numpy()[ids],label[ids])),"global_unrestricted":True}
   score=report["dataset_macro_feature_bacc"];history.append({"step":step,"loss":float(loss),"val_feature_bacc":score})
   ck={"model":net.state_dict(),"optimizer":opt.state_dict(),"mean":mean,"std":std,"step":step,"rng":rng.getstate(),"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state(),"source_sha256":sha(__file__)}
   save(out/"last.pt",ck);write(out/"last_val.json",report)
   if score>best:
    best=score;save(out/"best.pt",ck);write(out/"best_val.json",report)
   write(out/"status.json",{"status":"running","step":step,"steps":steps,"best_val_feature_bacc":best,"history":history})
  if time.time()-start>4*3600:raise TimeoutError("probe stage4h")
 write(out/"summary.json",{"status":"complete","steps":steps,"best_val_feature_bacc":best,"history":history,"seconds":time.time()-start,"test_used":False})
 write(out/"status.json",{"status":"complete","step":steps,"best_val_feature_bacc":best})
def main():
 a=argparse.ArgumentParser();a.add_argument("mode",choices=["prepare","test","probe","pipeline"]);a.add_argument("--root",required=True);a.add_argument("--steps",type=int,default=1200);a.add_argument("--train-max",type=int,default=2048);a.add_argument("--val-max",type=int,default=128);args=a.parse_args()
 seed();root=Path(args.root);root.mkdir(parents=True,exist_ok=True)
 lock=(root/"data_pipeline.lock").open("w");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 write(root/"launch.json",{"pid":os.getpid(),"mode":args.mode,"started":time.time(),"root":str(root),"source_sha256":sha(__file__)})
 try:
  if args.mode in ["prepare","pipeline"] and not (root/"prepared.json").exists():prepare(root,args.train_max,args.val_max)
  if args.mode=="test":prepare(root,args.train_max,args.val_max,test=True)
  if args.mode in ["probe","pipeline"] and not (root/"probe/summary.json").exists():probe(root,args.steps)
 except Exception as e:
  write(root/"failure.json",{"error":repr(e),"time":time.time()});raise
if __name__=="__main__":main()
