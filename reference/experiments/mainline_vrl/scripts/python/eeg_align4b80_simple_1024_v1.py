#!/usr/bin/env python3
"""Conditionally align frozenEEG features to a frozen80x2560 Oracle. No earlyGPU dispatch."""

if __name__ == "__main__":
    raise SystemExit("Reference snapshot only. Use tools/workbench.py for inspection; configure a fresh run before training.")

from pathlib import Path
import argparse,collections,copy,fcntl,gc,hashlib,importlib.util,json,math,os,random,shutil,signal,subprocess,sys,time,traceback
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import oracle4b80_simple_1024_v1 as oracle
P=Path("/data/yuqiudeng/vstats-elm")
PLAN=P/"experiments/mainline_vrl/plans/diagnostics/oracle4b_learnability_20261024_s1014"
UP=P/"experiments/mainline_vrl/runs/oracle4b80_simple_20261024_s1014"
ROOT=P/"experiments/mainline_vrl/runs/eeg_align4b80_simple_20261024_s1014";ARM="eeg_projector80";R=ROOT/ARM
DATA=P/"experiments/mainline_vrl/runs/stat_sleep_20261014_s1014/statistics/data"
EXPERT=P/"experiments/mainline_vrl/runs/stat_expert_20261015_s1014/banks"
tasks=oracle.tasks;base=oracle.base;ref=oracle.ref
read=oracle.read;write=oracle.write;sha=oracle.sha;complete=oracle.complete;SEED=1014
TRAINPY=str(P/".venv/bin/python");ARPY=str(P/"runtime/envs/eeglm-vllm/bin/python")
sp=importlib.util.spec_from_file_location("eeg80projector",PLAN/"eeg_projector80_4b_v1.py")
alignmod=importlib.util.module_from_spec(sp);sp.loader.exec_module(alignmod)


def proc(pid):
 q=Path('/proc')/str(pid)
 try:
  st=q.joinpath('stat').read_text().rsplit(')',1)[1].split()
  argv=[v.decode(errors='replace') for v in q.joinpath('cmdline').read_bytes().split(b'\0') if v]
  return {'pid':pid,'ppid':int(st[1]),'state':st[0],'start_ticks':int(st[19]),'argv':argv}
 except (FileNotFoundError,PermissionError,ProcessLookupError):return None

def cache_owned():
 out=[]
 for q in Path('/proc').iterdir():
  if not q.name.isdigit() or int(q.name)==os.getpid():continue
  try:
   pairs=q.joinpath('environ').read_bytes().split(b'\0')
   if ('HF_HOME='+str(ROOT/'cache/hf')).encode() not in pairs:continue
   x=proc(int(q.name))
   if x and x['state']!='Z':out.append(x)
  except (FileNotFoundError,PermissionError,ProcessLookupError):pass
 return out

def cleanup(child,identity):
 if child is not None and child.poll() is None:
  now=proc(child.pid);assert now and now['start_ticks']==identity['start_ticks'] and now['argv']==identity['argv'] and os.getpgid(child.pid)==child.pid
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=20)
  except subprocess.TimeoutExpired:
   now=proc(child.pid);assert now and now['start_ticks']==identity['start_ticks'];os.killpg(child.pid,signal.SIGKILL);child.wait()
 # Only this uniquely owned cache namespace may be cleaned after the worker exits.
 end=time.time()+8
 while cache_owned() and time.time()<end:time.sleep(.5)
 leftovers=cache_owned()
 for old in leftovers:
  now=proc(old['pid'])
  if now and now['start_ticks']==old['start_ticks'] and now['argv']==old['argv']:
   pairs=Path('/proc',str(old['pid']),'environ').read_bytes().split(b'\0')
   assert ('HF_HOME='+str(ROOT/'cache/hf')).encode() in pairs
   os.kill(old['pid'],signal.SIGTERM)
 end=time.time()+8
 while cache_owned() and time.time()<end:time.sleep(.5)
 write(ROOT/'audit/last_owned_cleanup.json',{'time':time.time(),'leftovers_signalled':leftovers,'remaining':cache_owned(),'no_other_cache_namespace_touched':True})
 assert not cache_owned(),'Owned auxiliary processes remain; no new arm may start'

def gpus():
 out=[]
 z=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 for line in z.splitlines():
  gpu,total,free,util=[int(x.strip()) for x in line.split(',')]
  if gpu in range(2,8):out.append((gpu,total,free,util))
 return sorted(out,key=lambda x:(-x[2],x[3],x[0]))

def lock_free(path):
 f=Path(path).open("a+")
 try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:f.close();return False
 fcntl.flock(f,fcntl.LOCK_UN);f.close();return True

def same_identity(old,new):
 return bool(new and new["state"]!="Z" and new["start_ticks"]==old["start_ticks"] and new["argv"]==old["argv"])

def spawn_identity(child,expected):
 end=time.time()+5
 while time.time()<end:
  now=proc(child.pid)
  if now and now["state"]!="Z" and now["argv"]==expected and now["ppid"]==os.getpid():
   return now
  if child.poll() is not None:raise RuntimeError("Owned child exited before stable argv capture")
  time.sleep(.01)
 raise RuntimeError("Owned child argv not stable; do not assume identity")

def guard(extra=0):
 if shutil.disk_usage(R).free<12*2**30+extra:raise OSError('12GiB disk and atomic-write reserve reached')

def save(p,x):
 guard(8*2**30);Path(p).parent.mkdir(parents=True,exist_ok=True);base.save_checkpoint(Path(p),x)

def config():return read(ROOT/'protocol.json')

def sha_text(s):return hashlib.sha256(s.encode()).hexdigest()

def ids(tok,text,eos=True):return torch.tensor(base.token_ids(tok,text,eos=eos),device='cuda',dtype=torch.long)

def logits_for(lm,emb,x,y):
 xx=torch.cat([x,emb(y).detach()])[None]
 assert len(xx[0])<=4096
 logits=lm(inputs_embeds=xx,attention_mask=torch.ones(xx.shape[:2],device=xx.device,dtype=torch.long),use_cache=False).logits[0,len(x)-1:len(x)+len(y)-1]
 return logits.float()

def engine_load():
 from transformers import AutoTokenizer,AutoModelForCausalLM
 from vllm import LLM,SamplingParams
 tok=AutoTokenizer.from_pretrained(base.MODEL,local_files_only=True)
 lm=AutoModelForCausalLM.from_pretrained(base.MODEL,torch_dtype=torch.bfloat16,local_files_only=True).cpu()
 emb=copy.deepcopy(lm.get_input_embeddings());del lm
 batch=int(os.environ.get('VSTATS_COMPACT_AR_BATCH','16'))
 engine=LLM(model=str(base.MODEL),dtype='bfloat16',enable_prompt_embeds=True,max_model_len=4096,max_num_seqs=batch,gpu_memory_utilization=float(os.environ.get('VSTATS_COMPACT_AR_MEMORY_UTILIZATION','.35')),enforce_eager=True,disable_log_stats=True,seed=SEED)
 return tok,emb,engine,SamplingParams,batch

def summarize(rows):
 groups=collections.defaultdict(list)
 for r in rows:groups[(r["mode"],r["task"])].append(r)
 out=[]
 for (mode,task),rr in sorted(groups.items()):
  ds={}
  for name in sorted({r["dataset"] for r in rr}):
   ss=[r for r in rr if r["dataset"]==name]
   ds[name]=dict(n=len(ss),correct=sum(r["correct"] for r in ss),valid=sum(r["valid"] for r in ss),
    exact=sum(r["correct"] for r in ss)/len(ss),format=sum(r["valid"] for r in ss)/len(ss))
  out.append(dict(mode=mode,task=task,n=len(rr),correct=sum(r["correct"] for r in rr),valid=sum(r["valid"] for r in rr),
   exact=sum(r["correct"] for r in rr)/len(rr),dataset_macro_exact=float(np.mean([d["exact"] for d in ds.values()])),per_dataset=ds))
 return dict(status="complete",queries=len(rows),metrics=out,no_cross_task_composite=True)

def stage(role,name,args,artifact):
 W=R/('worker_'+role);W.mkdir(parents=True,exist_ok=True);log=W/(name+'.log');meta_path=W/(name+'.attempt_state.json')
 if complete(R/artifact):print('SKIP',role,name,flush=True);return
 limit=read(ROOT/'first_launch.json')['started']+12*3600
 meta=read(meta_path) if meta_path.exists() else {'first_started':None,'attempts':0,'ar_batch':16,'ar_memory_utilization':.35,'micro_batch':1}
 ar=args[0] in ['teacher','eval'];need=(24 if ar else 26)*1024
 while True:
  deadline=min(limit,meta['first_started']+4*3600 if meta['first_started'] else limit)
  if time.time()>=deadline:raise TimeoutError('Original worker/stage budget '+name)
  lease=None;env=dict(os.environ);chosen=None;child=None;identity=None
  try:
   while lease is None:
    cards=[]
    q=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    for l in q.splitlines():
     gpu,total,free,util=[int(v.strip()) for v in l.split(',')]
     if gpu in range(2,8) and free>=need:cards.append((gpu,total,free,util))
    for gpu,total,free,util in sorted(cards,key=lambda x:(-x[2],x[3],x[0])):
     f=(ROOT/f'gpu{gpu}.lease').open('a+')
     try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
     except BlockingIOError:f.close();continue
     measurements=[]
     try:
      for sample in range(3):
       if time.time()>=deadline:raise TimeoutError('Original budget during stable admission')
       q=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip()
       gi,uuid,free_s=[x.strip() for x in q.split(',')];assert int(gi)==gpu
       measurements.append({'time':time.time(),'gpu':gpu,'uuid':uuid,'free_mib':int(free_s)})
       if int(free_s)<need:break
       if sample<2:time.sleep(10)
     except BaseException:
      fcntl.flock(f,fcntl.LOCK_UN);f.close();raise
     stable=len(measurements)==3 and all(x['free_mib']>=need for x in measurements)
     with (W/'admission_checks.jsonl').open('a') as admission:
      admission.write(tasks.dumps({'time':time.time(),'stage':name,'worker_pid':os.getpid(),'required_free_mib':need,'stable':stable,'samples':measurements,'source_sha256':sha(__file__)})+'\n')
     if not stable:fcntl.flock(f,fcntl.LOCK_UN);f.close();continue
     free=min(x['free_mib'] for x in measurements)
     lease=f;chosen=gpu;env.update(CUDA_VISIBLE_DEVICES=str(gpu),VSTATS_COMPACT_ALLOCATOR_BUDGET_MIB=str(free-10*1024),VSTATS_COMPACT_GPU_TOTAL_MIB=str(total));break
    if lease is None:
     write(W/'status.json',{'state':'waiting_gpu_memory','stage':name,'pipeline_pid':os.getpid(),'required_free_mib':need,'time':time.time()})
     if time.time()>=deadline:raise TimeoutError('Original budget waiting memory')
     time.sleep(10)
   guard()
   if meta['first_started'] is None:meta['first_started']=time.time()
   meta['attempts']+=1;write(meta_path,meta);deadline=min(limit,meta['first_started']+4*3600)
   env.update(VSTATS_STAGE_DEADLINE=str(deadline),VSTATS_COMPACT_MICRO_BATCH=str(meta['micro_batch']),VSTATS_COMPACT_AR_BATCH=str(meta['ar_batch']),VSTATS_COMPACT_AR_MEMORY_UTILIZATION=str(min(meta['ar_memory_utilization'],(free-10*1024)/total)))
   cmd=[ARPY if ar else TRAINPY,'-B',str(Path(__file__).resolve())]+args
   offset=log.stat().st_size if log.exists() else 0
   with log.open('a') as f:
    f.write(tasks.dumps({'time':time.time(),'attempt':meta,'gpu':chosen,'command':cmd,'allocator_budget_mib':env['VSTATS_COMPACT_ALLOCATOR_BUDGET_MIB']})+'\n');f.flush()
    child=subprocess.Popen(cmd,cwd=P,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    identity=spawn_identity(child,cmd)
    write(W/'status.json',{'state':'running','stage':name,'pipeline_pid':os.getpid(),'pid':child.pid,'gpu':chosen,'time':time.time(),'log':str(log),'command':cmd})
    below=0;reserve_samples=[]
    while child.poll() is None:
     try:child.wait(timeout=min(10,max(.1,deadline-time.time())))
     except subprocess.TimeoutExpired:pass
     if time.time()>=deadline:raise TimeoutError('Original worker/stage budget '+name)
     if child.poll() is not None:break
     q=subprocess.check_output(['nvidia-smi','-i',str(chosen),'--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip()
     gi,uuid,free_s=[x.strip() for x in q.split(',')];free_now=int(free_s)
     assert int(gi)==chosen and chosen in range(2,8)
     below=below+1 if free_now<8*1024 else 0
     if below:
      reserve_samples.append({'time':time.time(),'gpu':chosen,'uuid':uuid,'free_mib':free_now,'owned_child_pid':child.pid,'owned_child_pgid':os.getpgid(child.pid)})
     if below>=3:
      write(W/(name+'.physical_reserve_stop.json'),{'status':'safe_stop_required','reason':'Three consecutive physical free samples below8GiB; not OOM, no AR reduction','samples':reserve_samples,'strict_resume_required':True,'first_started':meta['first_started'],'deadline':deadline})
      raise RuntimeError('PHYSICAL_RESERVE_SAFE_STOP; preserve strict last and original budgets; see '+str(W/(name+'.physical_reserve_stop.json')))
    code=child.returncode
   if code==0:assert complete(R/artifact),'Missing complete artifact '+artifact;return
   with log.open() as f:f.seek(offset);tail=f.read().lower()
   oom=any(x in tail for x in ['cuda out of memory','torch.outofmemoryerror','cuda error: out of memory'])
   if not oom:raise RuntimeError(role+' '+name+' failed; see '+str(log))
   before=dict(meta)
   if not ar or (meta['ar_batch']==1 and meta['ar_memory_utilization']<=.2):raise RuntimeError('OOM at minimum allowed resource configuration; checkpoint retained')
   meta['ar_batch']=max(1,meta['ar_batch']//2);meta['ar_memory_utilization']=max(.2,round(meta['ar_memory_utilization']-.05,2));write(meta_path,meta)
   with (W/'resource_retries.jsonl').open('a') as f:f.write(tasks.dumps({'time':time.time(),'stage':name,'before':before,'after':meta,'failed_pid':child.pid,'unchanged_models_tasks_budget':True})+'\n')
  finally:
   if child is not None:cleanup(child,identity)
   if lease is not None:fcntl.flock(lease,fcntl.LOCK_UN);lease.close()

def verify():
 oracle.verify()
 d=config();assert sha(__file__)==d["source_sha256"]
 assert sha(PLAN/"eeg_projector80_4b_v1.py")==d["projector_source_sha256"]
 assert sha(UP/"oracle_freeze.json")==d["upstream_freeze_sha256"]
 assert sha(UP/"label32_stat8/curriculum/oracle/best.pt")==d["oracle_sha256"]
 for path,want in d["frozen_files"].items():assert sha(path)==want,path
 assert read(UP/"label32_stat8/oracle_promotion_gate.json")["passed"]
 assert read(UP/"audit/teacher_coverage.json")["all_label_field_level_direction_tie_covered"]


def initialize(device):
 net=alignmod.EEGProjector80()
 net.load_state_dict(torch.load(R/"projector_init.pt",map_location="cpu",weights_only=False)["state_dict"])
 return net.to(device)


_FEATURES={}
def load_features(split):
 if split not in _FEATURES:
  label=torch.load(DATA/(split+".pt"),map_location="cpu",weights_only=False,mmap=True)
  stat=torch.load(EXPERT/(split+".pt"),map_location="cpu",weights_only=False,mmap=True)
  records=read(DATA/(split+"_records.json"))
  assert len(label["bank"])==len(stat["bank"])==len(records)
  assert label["bank"].shape[1:]==stat["bank"].shape[1:]==(1,150,50)
  index={r["id"]:i for i,r in enumerate(records)}
  assert len(index)==len(records)
  _FEATURES[split]=(label,stat,records,index)
 return _FEATURES[split]
def eeg_prefix(net,e,split):
 label,stat,records,index=load_features(split)
 ii=[index[r["id"]] for r in e["records"]]
 return net(label["bank"][ii].to("cuda"),label["valid"][ii].to("cuda"),stat["bank"][ii].to("cuda"),stat["valid"][ii].to("cuda"))


def prepare():
 verify();base.seed_all();guard()
 if complete(R/"prepared.json"):return
 real=[]
 for j in range(6):
  for e in read(UP/"label32_stat8/teacher"/("round_%d_bank.json"%j)):
   if all(not r["id"].startswith("synthetic_") for r in e["records"]):real.append(e)
 assert len({e["id"] for e in real})==len(real)
 tr,st,records,index=load_features("train")
 for e in real:
  assert tasks.parse(e["teacher_trajectory"],e)["correct"]
  for r in e["records"]:
   assert r["id"] in index
   actual=records[index[r["id"]]]
   assert actual["label"]==r["label"] and actual["bins"]==r["bins"]
 coverage={op:len({e["template_id"] for e in real if e.get("operation")==op}) for op in ["read_level","compare_level","choose_higher","choose_lower"]}
 assert all(v>=8 for v in coverage.values()),coverage
 classes={r["global_label_id"] for e in real for r in e["records"]};assert len(classes)==30
 write(R/"train_pool.json",real);write(R/"val_plan.json",read(UP/"label32_stat8/val_plan.json"))
 write(R/"tf_val.json",read(UP/"label32_stat8/tf_val.json"))
 write(R/"val_text_raw.json",read(UP/"label32_stat8/teacher/val_text_raw.json"))
 va=load_features("val");assert {r["id"] for r in records}.isdisjoint(va[3])
 net=alignmod.EEGProjector80()
 assert sum(p.numel() for p in net.parameters())==253616432
 # Full production CPU forward/backward before any GPU alignment; no proxy hidden size here.
 bank=tr["bank"][:1].float();valid=tr["valid"][:1]
 z=net(bank,valid,st["bank"][:1].float(),st["valid"][:1]);assert z.shape==(1,80,2560)
 target=torch.randn_like(z,requires_grad=True);loss=alignmod.token_alignment(z,target)
 loss.backward();assert target.grad is None
 assert net.label_out.weight.grad.abs().sum()>0 and net.stat_out.weight.grad.abs().sum()>0
 net.zero_grad(set_to_none=True)
 save(R/"projector_init.pt",dict(state_dict=net.state_dict(),source_sha256=sha(PLAN/"eeg_projector80_4b_v1.py"),seed=1014))
 write(R/"audit/cpu_check.json",dict(status="passed",time=time.time(),real_train_examples=len(real),real_classes=30,synthetic_EEG_examples=0,
  wordings=coverage,production_shape=[1,80,2560],production_parameters=253616432,production_CPU_gradient_checked=True,
  target_stop_gradient=True,no_label_bin_or_question_forward_inputs=True,train_val_id_disjoint=True,
  source_sha256=sha(__file__),init_sha256=sha(R/"projector_init.pt")))
 write(R/"prepared.json",dict(status="complete",time=time.time(),init_sha256=sha(R/"projector_init.pt"),train_pool_sha256=sha(R/"train_pool.json")))


_DRAW_GROUPS=None
def draw_examples(rng,pool,step):
 global _DRAW_GROUPS
 if _DRAW_GROUPS is None:
  groups={}
  for e in pool:
   k=e.get("operation","base:"+e.get("q",{}).get("task",""))
   field=e.get("field","base");answer=tasks.target(e)
   ds=e["dataset"];label=e["records"][0]["global_label_id"]
   groups.setdefault(k,{}).setdefault(field,{}).setdefault(answer,{}).setdefault(ds,{}).setdefault(label,[]).append(e)
  _DRAW_GROUPS=groups
 ops=["read_level","compare_level","choose_higher","choose_lower"];bas=["base:label"]+["base:feature:"+f for f in tasks.FIELDS]+["base:report"]
 out=[]
 for i in range(8):
  n=(step-1)*8+i;k=bas[(n//5)%8] if n%5==0 else ops[(n-n//5-1)%4]
  g=_DRAW_GROUPS[k]
  for level in range(4):g=g[rng.choice(sorted(g))]
  out.append(rng.choice(g))
 return out


def train(round_,smoke=False):
 verify();guard();base.seed_all();out=R/"alignment"/("smoke" if smoke else "train");out.mkdir(parents=True,exist_ok=True)
 tok,lm,emb=base.hf_decoder();net=initialize("cuda")
 target_bank=oracle.initialize("cuda").eval().requires_grad_(False)
 target_bank.load_state_dict(torch.load(UP/"label32_stat8/curriculum/oracle/best.pt",map_location="cpu",weights_only=False)["state_dict"])
 opt=torch.optim.AdamW(net.parameters(),lr=1e-5,weight_decay=.0001);rng=random.Random(1014)
 pool=read(R/"train_pool.json");tf=read(R/"tf_val.json");start=0;best=math.inf;history=[];began=time.time()
 if not smoke and (out/"last.pt").exists():
  ck=torch.load(out/"last.pt",map_location="cpu",weights_only=False);assert ck["source_sha256"]==sha(__file__) and ck["oracle_sha256"]==config()["oracle_sha256"]
  assert ck["projector_init_sha256"]==sha(R/"projector_init.pt")
  assert ck["step"]<=(round_+1)*1000 and ck["step"]>=round_*1000
  net.load_state_dict(ck["state_dict"]);opt.load_state_dict(ck["optimizer"]);ref.restore(ck,rng)
  random.setstate(ck["python_global_rng"]);torch.cuda.set_rng_state_all(ck["all_cuda_rng"])
  start=ck["step"];best=ck["best"];history=ck["history"]
 end=30 if smoke else (round_+1)*1000
 before={k:p.detach().clone() for k,p in net.named_parameters()} if smoke else None
 all_slots=torch.zeros(80,dtype=torch.bool)
 def state(step):
  return dict(state_dict={k:v.detach().cpu() for k,v in net.state_dict().items()},optimizer=opt.state_dict(),step=step,best=best,history=history,
   **ref.rng_state(rng),python_global_rng=random.getstate(),all_cuda_rng=torch.cuda.get_rng_state_all(),source_sha256=sha(__file__),
   oracle_sha256=config()["oracle_sha256"],projector_init_sha256=sha(R/"projector_init.pt"),effective_batch=8,micro_batch=1)
 with (out/"trace.jsonl").open("a") as f:
  for step in range(start+1,end+1):
   if time.time()>float(os.environ["VSTATS_STAGE_DEADLINE"])-120:
    if step>start+1:save(out/"last.pt",state(step-1))
    raise TimeoutError("Original EEG stage budget; strict last retained")
   net.train();opt.zero_grad(set_to_none=True);ee=draw_examples(rng,pool,step);lossv=cev=alignv=0.
   for e in ee:
    with torch.autocast("cuda",dtype=torch.bfloat16):
     z=eeg_prefix(net,e,"train");z.retain_grad()
     with torch.no_grad():
      bins,labels=base.teacher_batch(e["records"],"cuda");target=target_bank(bins,labels)
     align=alignmod.token_alignment(z,target)
     y=ids(tok,e["teacher_trajectory"]);x=tasks.render(tok,emb,e,z);logits=logits_for(lm,emb,x,y)
     ce=F.cross_entropy(logits,y);loss=ce+align
    assert torch.isfinite(loss);(loss/8).backward()
    assert z.grad is not None;all_slots|=(z.grad.detach().float().square().sum((0,2)).cpu()>0)
    lossv+=float(loss)/8;cev+=float(ce)/8;alignv+=float(align)/8
   grad=nn.utils.clip_grad_norm_(net.parameters(),1.);assert torch.isfinite(grad) and grad>0
   assert all(p.grad is None for p in lm.parameters()) and all(p.grad is None for p in target_bank.parameters())
   opt.step();f.write(tasks.dumps(dict(step=step,ce=cev,alignment=alignv,loss=lossv,sample_ids=[e["id"] for e in ee]))+"\n");f.flush()
   if step%10==0:print(tasks.dumps(dict(step=step,ce=cev,alignment=alignv,grad=float(grad),seconds=time.time()-began)),flush=True)
   if smoke and step==30:
    assert all_slots.all()
    changed={name:not torch.equal(before[name],p) for name,p in net.named_parameters()};assert all(changed.values()),[n for n,x in changed.items() if not x]
    save(out/"last.pt",state(step))
    ck=torch.load(out/"last.pt",map_location="cpu",weights_only=False)
    restored=initialize("cuda");restored.load_state_dict(ck["state_dict"]);opt2=torch.optim.AdamW(restored.parameters(),lr=1e-5,weight_decay=.0001);opt2.load_state_dict(ck["optimizer"])
    for k,values in opt.state_dict()["state"].items():
     for name,val in values.items():
      other=opt2.state_dict()["state"][k][name]
      assert torch.equal(val.cpu(),other.cpu()) if isinstance(val,torch.Tensor) else val==other
    def draws():return [rng.random(),random.random(),float(np.random.rand()),torch.rand(5).tolist(),torch.rand(5,device="cuda").cpu().tolist()]
    ref.restore(ck,rng);random.setstate(ck["python_global_rng"]);torch.cuda.set_rng_state_all(ck["all_cuda_rng"]);one=draws()
    ref.restore(ck,rng);random.setstate(ck["python_global_rng"]);torch.cuda.set_rng_state_all(ck["all_cuda_rng"]);assert one==draws()
    write(out/"summary.json",dict(status="passed",time=time.time(),steps=30,all80_slot_gradients=True,all_trainable_parameters_changed=True,
     LM_and_Oracle_frozen=True,only_projector_trained=True,optimizer_and_private_global_python_numpy_torch_allCUDA_RNG_reload=True,smoke_weights_discarded=True,
     peak_torch_gib=torch.cuda.max_memory_allocated()/2**30))
   if not smoke and step%100==0:save(out/"last.pt",state(step))
   if not smoke and (step%500==0 or step==end):
    net.eval();val=0.
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
     for e in tf:
      z=eeg_prefix(net,e,"val");x=tasks.render(tok,emb,e,z);y=ids(tok,tasks.target(e));val+=float(F.cross_entropy(logits_for(lm,emb,x,y),y))/len(tf)
    improve=val<best;best=min(best,val);history.append(dict(step=step,val_tf_ce=val))
    ck=state(step);save(out/"last.pt",ck)
    if improve:save(out/"best.pt",ck)
    write(out/"summary.json",dict(status="complete" if step==6000 else "round_complete" if step==end else "running",steps=step,best_val_tf_ce=best,history=history))
 if not smoke:
  save(out/("round_%d_last.pt"%round_),state(end))
  write(out/("round_%d.complete.json"%round_),dict(status="complete",time=time.time(),step=end))


def evaluate(smoke=False,round_=None):
 verify();guard();out=R/"alignment/evaluation";out.mkdir(parents=True,exist_ok=True)
 suffix="smoke" if smoke else "round_%d_val"%round_ if round_ is not None else "val"
 if complete(out/(suffix+".summary.json")):return
 plan=read(R/"val_plan.json")
 if smoke:plan=sum([[e for e in plan if e.get("operation")==op][:4] for op in ["read_level","compare_level","choose_higher","choose_lower"]],[])+[e for e in plan if e["kind"]=="base" and e["q"]["task"]=="report"][:4]
 ckpt=R/"alignment/smoke/last.pt" if smoke else R/"alignment/train"/("round_%d_last.pt"%round_ if round_ is not None else "best.pt")
 checkpoint_sha256=sha(ckpt);state=torch.load(ckpt,map_location="cpu",weights_only=False)["state_dict"]
 tok,emb,engine,SamplingParams,batch=engine_load();net=initialize("cuda").eval().requires_grad_(False);net.load_state_dict(state)
 params=SamplingParams(temperature=0.,max_tokens=384,seed=1014);path=out/(suffix+".jsonl");existing={}
 if path.exists():
  for line in path.open():
   row=json.loads(line);assert row["id"] not in existing;existing[row["id"]]=row
 missing=[e for e in plan if e["id"] not in existing]
 with path.open("a") as f:
  for off in range(0,len(missing),batch):
   bb=missing[off:off+batch];requests=[]
   with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
    for e in bb:requests.append(dict(prompt_embeds=tasks.render(tok,emb,e,eeg_prefix(net,e,"val")).cpu().bfloat16()))
   outputs=engine.generate(requests,params,use_tqdm=False)
   for e,o in zip(bb,outputs):
    text=o.outputs[0].text;score=tasks.parse(text,e)
    row=dict(id=e["id"],mode="EEG",task=e.get("operation",e.get("q",{}).get("task")),kind=e["kind"],dataset=e["dataset"],
     question=tasks.question(e),expected=tasks.expected(e),generation=text,**score,field=e.get("field",e.get("q",{}).get("field")),
     sample_ids=[r["id"] for r in e["records"]],gold_label=e["records"][0]["label"],checkpoint_sha256=checkpoint_sha256,
     source_sha256=sha(__file__),input_interface="frozen_EEG_features_to80softtokens",reference_solution_supplied=False,no_gold_label_or_bins_as_projector_input=True)
    existing[e["id"]]=row;f.write(tasks.dumps(row)+"\n")
   f.flush()
   if off%160==0:print("EEG_AR",suffix,off+len(bb),len(missing),flush=True)
   if time.time()>float(os.environ["VSTATS_STAGE_DEADLINE"])-60:raise TimeoutError("Original EEG AR budget")
 rows=[existing[e["id"]] for e in plan];result=summarize(rows);result.update(checkpoint_sha256=checkpoint_sha256,source_sha256=sha(__file__),plan_sha256=sha(R/"val_plan.json"))
 write(out/(suffix+".summary.json"),result);del engine;del net


def pipeline():
 W=ROOT/"worker_master";W.mkdir(parents=True,exist_ok=True)
 with (W/"pipeline.lock").open("a+") as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);verify();prepare()
  if not (ROOT/"first_launch.json").exists():write(ROOT/"first_launch.json",dict(started=time.time(),budget_seconds=43200,smoke_included=True))
  write(W/"launch.json",dict(identity=proc(os.getpid()),source_sha256=sha(__file__),started=read(ROOT/"first_launch.json")["started"]))
  try:
   stage("alignment","smoke",["train","--smoke"],"alignment/smoke/summary.json")
   stage("alignment","AR_smoke",["eval","--smoke"],"alignment/evaluation/smoke.summary.json")
   for rr in range(6):
    stage("alignment","train_round_%d"%rr,["train","--round",str(rr)],"alignment/train/round_%d.complete.json"%rr)
    if rr in [2,5]:stage("alignment","round_%d_val"%rr,["eval","--round",str(rr)],"alignment/evaluation/round_%d_val.summary.json"%rr)
   stage("alignment","best_val",["eval"],"alignment/evaluation/val.summary.json")
   write(W/"complete.json",dict(status="complete",time=time.time()))
   write(W/"status.json",dict(state="complete_cpu_report_pending",time=time.time()))
  except BaseException:
   write(ROOT/"abort.json",dict(time=time.time(),error=traceback.format_exc()))
   write(W/"status.json",dict(state="stopped",time=time.time(),error=traceback.format_exc()));raise



def checkpoint_audit():
 folder=R/"alignment/train";last=folder/"last.pt"
 if not last.exists():return dict(committed_steps=0,checkpoint_files={},trace_verified=False)
 checkpoints={}
 init_hash=sha(R/"projector_init.pt")
 required=["optimizer","rng","numpy_rng","torch_rng","cuda_rng","all_cuda_rng","python_global_rng"]
 committed=0
 for path in sorted(folder.glob("*.pt")):
  ck=torch.load(path,map_location="cpu",weights_only=False,mmap=True)
  assert all(k in ck for k in required),str(path)
  assert ck["source_sha256"]==sha(__file__) and ck["oracle_sha256"]==config()["oracle_sha256"]
  assert ck["projector_init_sha256"]==init_hash
  assert 0<ck["step"]<=6000 and ck["effective_batch"]==8 and ck["micro_batch"]==1
  private=random.Random();private.setstate(ck["rng"])
  glob=random.Random();glob.setstate(ck["python_global_rng"])
  nrng=np.random.RandomState();nrng.set_state(ck["numpy_rng"])
  assert isinstance(ck["torch_rng"],torch.Tensor) and ck["torch_rng"].dtype==torch.uint8
  assert isinstance(ck["cuda_rng"],torch.Tensor) and len(ck["all_cuda_rng"])>=1
  assert all(isinstance(t,torch.Tensor) and t.dtype==torch.uint8 for t in ck["all_cuda_rng"])
  optimizer=ck["optimizer"];params=sum(len(g["params"]) for g in optimizer["param_groups"])
  assert len(optimizer["state"])==params and params>0
  for values in optimizer["state"].values():
   assert {"step","exp_avg","exp_avg_sq"}.issubset(values)
   assert int(values["step"])==ck["step"]
  checkpoints[path.name]=dict(step=ck["step"],sha256=sha(path),strict_optimizer_all_RNG_and_lineage_present=True)
  if path==last:committed=ck["step"]
  del ck
 traces={}
 for line in (folder/"trace.jsonl").open():
  row=json.loads(line)
  if row["step"]<=committed:
   if row["step"] in traces:assert traces[row["step"]]["sample_ids"]==row["sample_ids"]
   traces[row["step"]]=row
 assert sorted(traces)==list(range(1,committed+1))
 rng=random.Random(1014);pool=read(R/"train_pool.json")
 for step in range(1,committed+1):
  assert [e["id"] for e in draw_examples(rng,pool,step)]==traces[step]["sample_ids"],step
 return dict(committed_steps=committed,checkpoint_files=checkpoints,trace_verified=True,
  committed_trace_contiguous=True,all_committed_sample_IDs_recomputed=True,
  uncommitted_trace_not_counted=True,duplicate_trace_not_double_counted=True)

def finalize():
 assert not cache_owned()
 for n in ["worker_master/pipeline.lock"]+["gpu%d.lease"%g for g in range(2,8)]:assert lock_free(ROOT/n),n
 verify();evaluations={}
 training_audit=checkpoint_audit();write(ROOT/"audit/round_progress_verification.json",training_audit)
 plan={e["id"]:e for e in read(R/"val_plan.json")}
 for p in (R/"alignment/evaluation").glob("*.jsonl"):
  rows=[json.loads(l) for l in p.open()];assert len({r["id"] for r in rows})==len(rows)
  for r in rows:
   e=plan[r["id"]];assert r["expected"]==tasks.expected(e) and r["question"]==tasks.question(e)
   sc=tasks.parse(r["generation"],e);assert sc["valid"]==r["valid"] and sc["correct"]==r["correct"]
  evaluations[p.stem]=summarize(rows)
 history=read(R/"alignment/train/summary.json") if (R/"alignment/train/summary.json").exists() else {}
 history=dict(history,steps=training_audit["committed_steps"])
 paired={}
 if (R/"alignment/evaluation/val.jsonl").exists():
  eeg={r["id"]:r for r in [json.loads(l) for l in (R/"alignment/evaluation/val.jsonl").open()]}
  ora={r["id"]:r for r in [json.loads(l) for l in (UP/"label32_stat8/curriculum/evaluation/val.jsonl").open()]}
  text=read(R/"val_text_raw.json")
  for task in sorted({r["task"] for r in eeg.values()}):
   ids=[i for i,r in eeg.items() if r["task"]==task];both=onlyO=onlyE=neither=0
   for i in ids:
    assert eeg[i]["expected"]==ora[i]["expected"] and eeg[i]["question"]==ora[i]["question"]
    a,b=ora[i]["correct"],eeg[i]["correct"]
    both+=a and b;onlyO+=a and not b;onlyE+=b and not a;neither+=not a and not b
   tt=[i for i in ids if text[i]["correct"]]
   paired[task]=dict(n=len(ids),both=both,only_Oracle=onlyO,only_EEG=onlyE,neither=neither,
    text_correct_support=len(tt),EEG_correct_on_text_correct=sum(eeg[i]["correct"] for i in tt),
    Oracle_correct=sum(ora[i]["correct"] for i in ids),EEG_correct=sum(eeg[i]["correct"] for i in ids))
 result=dict(training=history,evaluations=evaluations,paired=paired,Test="N/A_Val_conditioned_diagnostic",no_clinical_or_general_language_claim=True)
 write(ROOT/"results_summary.json",result)
 write(ROOT/"audit/final_verification.json",dict(status="passed_for_available_artifacts",time=time.time(),process_exit_check_pending=False,
  all_predeclared_evaluations_complete=complete(ROOT/"worker_master/complete.json"),steps=history.get("steps",0),
  available_evaluation_counts={k:v["queries"] for k,v in evaluations.items()},source_sha256=sha(__file__),oracle_sha256=config()["oracle_sha256"]))
 lines=["# EEG逐token对齐","", "冻结4B、Oracle与两路既有EEG特征，只训练80-token投影器。Val为原1744题，未过滤文本错误。","", "正式步数："+str(history.get("steps",0))+" / 6000。","",
 "|题型|支持数|明文正确|Oracle正确|EEG正确|","|---|---:|---:|---:|---:|"]
 for task,d in paired.items():lines.append("|"+task+"|"+str(d["n"])+"|"+str(d["text_correct_support"])+"|"+str(d["Oracle_correct"])+"|"+str(d["EEG_correct"])+"|")
 lines+=["","缺失结果为N/A，不是完整Test或临床结论；原生明文ID和连续前缀执行路径不同。"]
 (ROOT/"final_report.md").write_text("\n".join(lines)+"\n")



def upstream_training_integrity():
 folder=UP/"label32_stat8/curriculum/oracle"
 ck=torch.load(folder/"last.pt",map_location="cpu",weights_only=False,mmap=True)
 assert ck["step"]==3600 and ck["source_sha256"]==sha(P/"experiments/mainline_vrl/scripts/python/oracle4b80_simple_1024_v1.py")
 assert ck["init_sha256"]==sha(UP/"label32_stat8/oracle_init.pt") and ck["tokens"]==80
 assert ck["micro_batch"]==1 and ck["effective_batch"]==8
 for key in ["optimizer","rng","numpy_rng","torch_rng","cuda_rng","all_cuda_rng","python_global_rng"]:assert key in ck,key
 assert ck["optimizer"]["state"] and all(int(v["step"])==3600 for v in ck["optimizer"]["state"].values())
 banks=[read(UP/"label32_stat8/teacher"/("round_%d_bank.json"%j)) for j in range(6)]
 assert ck["teacher_bank_hashes"]==[sha(UP/"label32_stat8/teacher"/("round_%d_bank.json"%j)) for j in range(6)]
 traces={}
 for line in (folder/"rollout_trace.jsonl").open():
  row=json.loads(line)
  if row["step"]<=3600:
   if row["step"] in traces:assert traces[row["step"]]["sample_ids"]==row["sample_ids"]
   traces[row["step"]]=row
 assert sorted(traces)==list(range(1,3601))
 for step in range(1,3601):
  rr=(step-1)//600
  assert traces[step]["sample_ids"]==[e["id"] for e in oracle.data_batch(step,banks,rr)]
  assert traces[step]["student_probability"]==[0,.25,.75,.75,.75,.75][rr]
 f=read(UP/"oracle_freeze.json")
 assert f["EEG_gate_passed"] and f["best_sha256"]==sha(folder/"best.pt") and f["init_sha256"]==ck["init_sha256"]
 result=dict(status="passed",time=time.time(),committed_steps=3600,all_committed_trace_and_sampler_IDs_verified=True,
  strict_optimizer_private_and_global_Python_NumPy_Torch_allCUDA_RNG_present=True,
  source_init_teacher_bank_and_selected_best_lineage_verified=True,last_checkpoint_sha256=sha(folder/"last.pt"),
  selected_Oracle_sha256=f["best_sha256"],process_exit_check_pending=False)
 write(PLAN/"conditional_EEG_alignment_v1/audit/upstream_training_integrity.json",result)

def conditional():
 B=PLAN/"conditional_EEG_alignment_v1";B.mkdir(parents=True,exist_ok=True)
 with (B/"bootstrap.lock").open("a+") as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  freeze=read(B/"audit/code_freeze.json")
  assert sha(UP/"protocol.json")==freeze["upstream_protocol_sha256"]
  assert sha(P/"experiments/mainline_vrl/scripts/python/oracle4b80_simple_1024_v1.py")==freeze["upstream_source_sha256"]
  assert sha(__file__)==freeze["source_sha256"]
  assert sha(PLAN/"eeg_projector80_4b_v1.py")==freeze["projector_source_sha256"]
  upid=read(UP/"dispatch.json")["identity"]
  upfirst=read(UP/"first_launch.json")["started"]
  while True:
   upid=read(UP/"dispatch.json")["identity"]
   live=same_identity(upid,proc(upid["pid"]))
   if not live and lock_free(UP/"worker_master/pipeline.lock"):break
   assert time.time()<upfirst+43200+1200,"Original Oracle deadline+CPU closure bound"
   write(B/"status.json",dict(time=time.time(),state="waiting_Oracle_completion_no_EEG_GPU",upstream_master_live=live));time.sleep(30)
  assert (UP/"audit/final_verification.json").exists() and not read(UP/"audit/final_verification.json")["process_exit_check_pending"]
  if not (UP/"label32_stat8/oracle_promotion_gate.json").exists() or not read(UP/"label32_stat8/oracle_promotion_gate.json")["passed"]:
   write(B/"status.json",dict(time=time.time(),state="Oracle_gate_not_passed_EEG_not_dispatched",EEG_dispatched=False))
   return
  assert complete(UP/"worker_master/complete.json")
  upstream_training_integrity()
  oracle.ready_to_finalize() if hasattr(oracle,"ready_to_finalize") else None
  assert not oracle.cache_owned()
  for g in range(2,8):assert lock_free(UP/("gpu%d.lease"%g))
  pre=read(UP/"audit/eeg_alignment_preparation_v1.json")
  assert sha(PLAN/"eeg_projector80_4b_v1.py")==pre["source_sha256"]
  for p,h in pre["frozen_feature_file_full_sha256"].items():assert sha(p)==h,p
  assert sha(__file__)==freeze["source_sha256"]
  oracle.verify()
  assert sha(UP/"protocol.json")==freeze["upstream_protocol_sha256"]
  assert sha(P/"experiments/mainline_vrl/scripts/python/oracle4b80_simple_1024_v1.py")==freeze["upstream_source_sha256"]
  assert not ROOT.exists(),"Unique EEGrun already exists; never duplicate"
  for folder in [ROOT/"audit",ROOT/"logs",ROOT/"worker_master",R/"audit",R/"worker_alignment"]:folder.mkdir(parents=True,exist_ok=True)
  frozen=dict(pre["frozen_feature_file_full_sha256"])
  frozen[str(P/"experiments/mainline_vrl/scripts/python/oracle4b80_simple_1024_v1.py")]=freeze["upstream_source_sha256"]
  proto=read(UP/"protocol.json")
  frozen.update({str(p):h for p,h in [(Path(k),v) for k,v in proto["models"].items()]})
  for p in [UP/"protocol.json",UP/"label32_stat8/val_plan.json",UP/"label32_stat8/tf_val.json",UP/"label32_stat8/teacher/summary.json",UP/"label32_stat8/teacher/val_text_raw.json",
   UP/"label32_stat8/curriculum/evaluation/val.jsonl"]:
   frozen[str(p)]=sha(p)
  for j in range(6):
   p=UP/"label32_stat8/teacher"/("round_%d_bank.json"%j);frozen[str(p)]=sha(p)
  conf=dict(source_sha256=sha(__file__),projector_source_sha256=sha(PLAN/"eeg_projector80_4b_v1.py"),
   upstream_freeze_sha256=sha(UP/"oracle_freeze.json"),oracle_sha256=sha(UP/"label32_stat8/curriculum/oracle/best.pt"),
   frozen_files=frozen,steps=6000,batch=8,micro=1,lr=1e-5,wd=.0001,loss="tokenwise normalized MSE+.1cosine+correctteacher answerCE",
   training_selection="RealTrain only; same correctteacher sample pool;20%basic80%fourtypes; field/answer/dataset/class balanced localRandom1014",
   master_hours=12,stage_hours=4,projector_parameters=253616432,hidden=2048,output_shape=[80,2560],
   experts_Oracle_LM_frozen=True,Oracle_gate_required_and_passed=True,Test="N/A_Val_diagnostic")
  write(ROOT/"protocol.json",conf)
  write(ROOT/"audit/authorization.json",dict(user_explicit="按这套自动推进（推荐）",condition="Original90percentOracleVal+fullreport gate passed",gate=read(UP/"label32_stat8/oracle_promotion_gate.json"),no_extra_scientific_settings=True))
  (ROOT/"PLAN.md").write_text("# 条件EEG80token对齐\n原Oracle90%Val门槛已通过后启动。冻结原4B/Oracle/原EEG特征，仅训练既有7500→2048双路投影器的80x2560适配。6000步，batch8micro1，LR1e-5wd.0001，逐token归一化MSE+.1cosine+答案CE。先CPUproduction结构梯度，再fresh30GPU训练+20AR后freshinit正式。原subjectsplit，完整Val1744不删题，Test=N/A。独立12h含Smoke，各stage4h，严格optimizer全部RNG恢复。\n")
  # All caches are process-local under the newEEG root; no CUDA13 compat in original4B environment.
  env=dict(os.environ,CUDA_VISIBLE_DEVICES="",HF_HOME=str(ROOT/"cache/hf"),XDG_CACHE_HOME=str(ROOT/"cache/xdg"),TORCH_HOME=str(ROOT/"cache/torch"),TORCH_EXTENSIONS_DIR=str(ROOT/"cache/extensions"),
   TRITON_CACHE_DIR=str(ROOT/"cache/triton"),VLLM_CACHE_ROOT=str(ROOT/"cache/vllm"),TORCHINDUCTOR_CACHE_DIR=str(ROOT/"cache/inductor"),NUMBA_CACHE_DIR=str(ROOT/"cache/numba"),CUDA_CACHE_PATH=str(ROOT/"cache/cuda"),
   TMPDIR=str(P/"tmp/eeg4simple1024"),VLLM_USE_V1="0",PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="4",MKL_NUM_THREADS="4")
  Path(env["TMPDIR"]).mkdir(parents=True,exist_ok=True)
  for k in ["LD_LIBRARY_PATH","CUDA_HOME","FLASHINFER_WORKSPACE_BASE"]:env.pop(k,None)
  cmd=[TRAINPY,"-B",str(Path(__file__).resolve()),"pipeline"]
  with (ROOT/"logs/pipeline.log").open("x") as log:
   ch=subprocess.Popen(cmd,cwd=P,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   ident=spawn_identity(ch,cmd)
  write(ROOT/"dispatch.json",dict(time=time.time(),identity=ident,argv=cmd,source_sha256=sha(__file__),protocol_sha256=sha(ROOT/"protocol.json")))
  write(B/"status.json",dict(time=time.time(),state="EEG_pipeline_dispatched_after_Oracle_gate",identity=ident,run=str(ROOT)))


def main():
 p=argparse.ArgumentParser();p.add_argument("command",choices=["conditional","pipeline","prepare","train","eval","finalize"])
 p.add_argument("--round",type=int,default=0);p.add_argument("--smoke",action="store_true");a=p.parse_args()
 signal.signal(signal.SIGTERM,lambda *x:(_ for _ in ()).throw(SystemExit("Owned EEG process stopped; preserve allartifacts")))
 if os.environ.get("CUDA_VISIBLE_DEVICES"):
  torch.cuda.set_per_process_memory_fraction(min(1.,int(os.environ["VSTATS_COMPACT_ALLOCATOR_BUDGET_MIB"])/int(os.environ["VSTATS_COMPACT_GPU_TOTAL_MIB"])))
 if a.command=="conditional":
  try:conditional()
  except BaseException:
   write(PLAN/"conditional_EEG_alignment_v1/error_conditional.json",dict(time=time.time(),error=traceback.format_exc(),no_assumed_EEG_dispatch=True))
   raise
 elif a.command=="pipeline":
  try:pipeline()
  finally:
   if (R/"val_plan.json").exists():finalize()
 elif a.command=="prepare":prepare()
 elif a.command=="train":train(a.round,a.smoke)
 elif a.command=="eval":evaluate(a.smoke,a.round if "--round" in sys.argv else None)
 elif a.command=="finalize":finalize()


if __name__=="__main__":main()
