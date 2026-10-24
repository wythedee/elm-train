#!/usr/bin/env python3
"""3600-step individual-correct simple-task Oracle; no closed run edits."""

if __name__ == "__main__":
    raise SystemExit("Reference snapshot only. Use tools/workbench.py for inspection; configure a fresh run before training.")

from pathlib import Path
import argparse,collections,copy,csv,fcntl,hashlib,itertools,json,math,os,random,shutil,signal,subprocess,sys,time,traceback,gc,importlib.util
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import oracle_curriculum28_tasks_1016_v3 as legacy
P=Path("/data/yuqiudeng/vstats-elm")
PLAN=P/"experiments/mainline_vrl/plans/diagnostics/oracle4b_learnability_20261024_s1014"
spec=importlib.util.spec_from_file_location("simple_tasks",PLAN/"simple_oracle_tasks_v2.py")
tasks=importlib.util.module_from_spec(spec);spec.loader.exec_module(tasks)
old=legacy.old;ref=old.ref;base=old.base
DATA=P/"experiments/mainline_vrl/runs/stat_sleep_20261014_s1014/statistics/data"
ROOT=P/"experiments/mainline_vrl/runs/oracle4b80_simple_20261024_s1014"
ARMS=["label32_stat8"];ARM="label32_stat8";R=ROOT/ARM
LABEL_TOKENS=32;STAT_TOKENS=8;TOKENS=80
LABEL_INDICES=[i%16 for i in range(32)];STAT_INDICES=[0,1]*4
SEED=1014;read=ref.read;write=ref.write;sha=ref.sha;complete=ref.complete
ARPY=str(P/"runtime/envs/eeglm-vllm/bin/python");TRAINPY=str(P/".venv/bin/python")
base.MODEL=P/"experiments/eeg_vrl_demo/models/Qwen/Qwen3-4B"


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
 guard(128*2**20);Path(p).parent.mkdir(parents=True,exist_ok=True);base.save_checkpoint(Path(p),x)

def config():return read(ROOT/'protocol.json')

class ExpandedOracle(nn.Module):
 def __init__(self):
  super().__init__()
  original=torch.load(ROOT/'native4b_init28.pt',map_location='cpu',weights_only=False)
  ix=torch.tensor(LABEL_INDICES,dtype=torch.long);sx=torch.tensor(STAT_INDICES,dtype=torch.long)
  self.label_codes=nn.Parameter(original['label_codes'][:,ix].float().clone())
  self.stat_codes=nn.Parameter(original['stat_codes'][:,:,sx].float().clone())
  self.register_buffer('label_position',original['label_position'].float().clone())
  self.register_buffer('initial_label',original['label_codes'][:,ix].float().clone(),persistent=False)
  self.register_buffer('initial_stats',original['stat_codes'][:,:,sx].float().clone(),persistent=False)
  self.tokens=TOKENS
 def forward(self,bins,labels):
  stats=torch.stack([self.stat_codes[j,bins[:,j]] for j in range(6)],1).flatten(1,2)
  label=self.label_codes[labels]+self.label_position[None,None]
  return torch.cat([label,stats],1)
 def anchor(self):
  a=(self.label_codes-self.initial_label).square().mean()/self.initial_label.square().mean().clamp_min(1e-8)
  b=(self.stat_codes-self.initial_stats).square().mean()/self.initial_stats.square().mean().clamp_min(1e-8)
  return .0001*(a+b)

def initialize(device):
 net=ExpandedOracle()
 net.load_state_dict(torch.load(R/'oracle_init.pt',map_location='cpu',weights_only=False)['state_dict'])
 return net.to(device)

def sha_text(s):return hashlib.sha256(s.encode()).hexdigest()

def engine_load():
 from transformers import AutoTokenizer,AutoModelForCausalLM
 from vllm import LLM,SamplingParams
 tok=AutoTokenizer.from_pretrained(base.MODEL,local_files_only=True)
 lm=AutoModelForCausalLM.from_pretrained(base.MODEL,torch_dtype=torch.bfloat16,local_files_only=True).cpu()
 emb=copy.deepcopy(lm.get_input_embeddings());del lm
 batch=int(os.environ.get('VSTATS_COMPACT_AR_BATCH','16'))
 engine=LLM(model=str(base.MODEL),dtype='bfloat16',enable_prompt_embeds=True,max_model_len=4096,max_num_seqs=batch,gpu_memory_utilization=float(os.environ.get('VSTATS_COMPACT_AR_MEMORY_UTILIZATION','.35')),enforce_eager=True,disable_log_stats=True,seed=SEED)
 return tok,emb,engine,SamplingParams,batch

def prefix(net,e):
 bb,ll=base.teacher_batch(e['records'],'cuda');return net(bb,ll)

def ids(tok,text,eos=True):return torch.tensor(base.token_ids(tok,text,eos=eos),device='cuda',dtype=torch.long)

def logits_for(lm,emb,x,y):
 xx=torch.cat([x,emb(y).detach()])[None]
 assert len(xx[0])<=4096
 logits=lm(inputs_embeds=xx,attention_mask=torch.ones(xx.shape[:2],device=xx.device,dtype=torch.long),use_cache=False).logits[0,len(x)-1:len(x)+len(y)-1]
 return logits.float()

def ce_objective(lm,tok,emb,net,e,trajectory=None):
 y=ids(tok,trajectory or e.get('teacher_trajectory',tasks.target(e)))
 x=tasks.render(tok,emb,e,prefix(net,e));s=logits_for(lm,emb,x,y)
 return F.cross_entropy(s,y),s,y

def distillation(lm,tok,emb,net,e,trajectory,gold_logits=None,gold_ids=None):
 y=ids(tok,trajectory) if gold_ids is None else gold_ids
 with torch.no_grad():
  x=tasks.render(tok,emb,e,teacher=True,reference=False)
  t=logits_for(lm,emb,x,y);lt=F.log_softmax(t,-1)
 if gold_logits is None:
  x=tasks.render(tok,emb,e,prefix(net,e));s=logits_for(lm,emb,x,y)
 else:s=gold_logits
 assert t.shape==s.shape
 # Chunk only the full-vocabulary computation; retain exact forward-KL objective.
 value=0.
 for off in range(0,len(y),32):
  teacher_log=lt[off:off+32];student_log=F.log_softmax(s[off:off+32],-1)
  value=value+(teacher_log.exp()*(teacher_log-student_log)).sum()/len(y)
 return value

def rollouts(lm,tok,emb,net,ee):
 if not ee:return []
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
  xx=[tasks.render(tok,emb,e,prefix(net,e)).detach() for e in ee]
  length=max(len(x) for x in xx);x=torch.zeros(len(xx),length,emb.weight.shape[1],device='cuda',dtype=emb.weight.dtype);mask=torch.zeros(len(xx),length,device='cuda',dtype=torch.long)
  for i,v in enumerate(xx):x[i,-len(v):]=v;mask[i,-len(v):]=1
  out=lm.generate(inputs_embeds=x,attention_mask=mask,use_cache=True,max_new_tokens=384,do_sample=True,temperature=.7,top_p=.95,pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id)
  assert out.shape[0]==len(ee) and out.shape[1]<=384,out.shape
  result=[tok.decode(row,skip_special_tokens=True) for row in out]
 return result

def train(role,round_,smoke=False):
 guard();base.seed_all();out=R/role/('smoke' if smoke else 'oracle');out.mkdir(parents=True,exist_ok=True)
 tok,lm,emb=base.hf_decoder();net=initialize('cuda');opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
 rng=random.Random(SEED+881);start=0;best=math.inf;history=[];began=time.time()
 micro=int(os.environ.get('VSTATS_COMPACT_MICRO_BATCH','1'));assert micro==1
 if not smoke and (out/'last.pt').exists():
  ck=torch.load(out/'last.pt',map_location='cpu',weights_only=False);net.load_state_dict(ck['state_dict']);opt.load_state_dict(ck['optimizer']);ref.restore(ck,rng);torch.cuda.set_rng_state_all(ck['all_cuda_rng']);random.setstate(ck['python_global_rng']);start=ck['step'];best=ck['best'];history=ck['history']
 initial={k:p.detach().clone() for k,p in net.named_parameters()};grad_slots={'label':torch.zeros(LABEL_TOKENS,dtype=torch.bool),'stats':torch.zeros(6,STAT_TOKENS,dtype=torch.bool)}
 banks=[read(R/'teacher'/f'round_{r}_bank.json') for r in range(6)]
 tf=read(R/'tf_val.json');end=30 if smoke else (round_+1)*600
 prob=.5 if smoke else [0.,.25,.75,.75,.75,.75][round_]
 audit_path=out/'rollout_trace.jsonl'
 def state(step):
  return {'state_dict':{k:v.detach().cpu() for k,v in net.state_dict().items()},'optimizer':opt.state_dict(),'all_cuda_rng':torch.cuda.get_rng_state_all(),'step':step,'best':best,'history':history,**ref.rng_state(rng),'python_global_rng':random.getstate(),'source_sha256':sha(__file__),'init_sha256':sha(R/'oracle_init.pt'),'role':role,'round':round_,'tokens':TOKENS,'micro_batch':micro,'effective_batch':8,'student_rollout_probability':prob if role=='curriculum' else 0.,'teacher_bank_hashes':[sha(R/'teacher'/f'round_{r}_bank.json') for r in range(6)]}
 with audit_path.open('a') as audit:
  for step in range(start+1,end+1):
   if time.time()>float(os.environ.get('VSTATS_STAGE_DEADLINE','inf'))-120:
    if step>start+1:save(out/'last.pt',state(step-1))
    raise TimeoutError('Cumulative train stage budget')
   net.train();opt.zero_grad(set_to_none=True);ee=data_batch(step,banks,round_)
   selected=[i for i in range(8) if role=='curriculum' and rng.random()<prob]
   generated=rollouts(lm,tok,emb,net,[ee[i] for i in selected]);student=dict(zip(selected,generated));cev=kdv=0.;events=[]
   for i,e in enumerate(ee):
    with torch.autocast('cuda',dtype=torch.bfloat16):
     ce,ss,yy=ce_objective(lm,tok,emb,net,e);kd=ce.new_zeros(())
     if role=='curriculum':
      if i in student:kd=distillation(lm,tok,emb,net,e,student[i])
      else:kd=distillation(lm,tok,emb,net,e,e.get('teacher_trajectory',tasks.target(e)),gold_logits=ss,gold_ids=yy)
     total=ce+.5*kd+net.anchor()
    assert torch.isfinite(total);(total/8).backward();cev+=float(ce)/8;kdv+=float(kd)/8
    if i in student:
     score=tasks.parse(student[i],e);events.append({'id':e['id'],'trajectory':student[i],'valid':score['valid'],'correct':score['correct']})
   grad_slots['label']|=(net.label_codes.grad.detach().square().sum((0,2)).cpu()>0);grad_slots['stats']|=(net.stat_codes.grad.detach().square().sum((1,3)).cpu()>0)
   grad=nn.utils.clip_grad_norm_(net.parameters(),1.);assert torch.isfinite(grad) and grad>0
   assert all(p.grad is None for p in lm.parameters());opt.step()
   audit.write(tasks.dumps({'step':step,'round':round_,'ce':cev,'kd':kdv,'student_probability':prob if role=='curriculum' else 0.,'sample_ids':[e['id'] for e in ee],'student_rollouts':events})+'\n');audit.flush()
   if step%10==0:print(tasks.dumps({'step':step,'round':round_,'ce':cev,'kd':kdv,'grad':float(grad),'student_rollouts':len(student),'seconds':time.time()-began}),flush=True)
   if not smoke and step%100==0:save(out/'last.pt',state(step))
   if smoke and step==end:
    changed={k:not torch.equal(v,dict(net.named_parameters())[k]) for k,v in initial.items()};assert all(changed.values()),changed
    save(out/'last.pt',state(step));restored=initialize('cuda');restored.load_state_dict(torch.load(out/'last.pt',map_location='cpu',weights_only=False)['state_dict'])
    assert all(torch.equal(a,b) for a,b in zip(net.state_dict().values(),restored.state_dict().values()))
    smoke_extra=smoke_roundtrip(out/'last.pt',net,opt,rng,grad_slots)
    write(out/'summary.json',{'extra_verification':smoke_extra,'status':'passed','steps':end,'changed':changed,'decoder_frozen':True,'checkpoint_roundtrip':True,'student_generate_exercised':bool(selected) if role=='curriculum' else None,'seconds':time.time()-began,'peak_torch_gib':torch.cuda.max_memory_allocated()/2**30})
   if not smoke and (step%500==0 or step==end):
    net.eval();v=0.
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
     for e in tf:v+=float(ce_objective(lm,tok,emb,net,e,tasks.target(e))[0])/len(tf)
    improve=v<best;best=min(best,v);history.append({'step':step,'val_tf_ce':v,'round':round_})
    ck=state(step);save(out/'last.pt',ck)
    if improve:save(out/'best.pt',ck)
    write(out/'summary.json',{'status':'complete' if step==3600 else 'round_complete' if step==end else 'running','steps':step,'round':round_,'best_val_tf_ce':best,'history':history,'seconds_this_stage':time.time()-began,'peak_torch_gib':torch.cuda.max_memory_allocated()/2**30,'trainable_parameters':sum(p.numel() for p in net.parameters())})
    print('VALIDATION',history[-1],flush=True)
 if not smoke:
  save(out/f'round_{round_}_last.pt',state(end));write(out/f'round_{round_}.complete.json',{'status':'complete','step':end,'checkpoint_sha256':sha(out/f'round_{round_}_last.pt')})

def smoke_roundtrip(path,net,opt,rng,coverage):
 assert all(bool(x.all()) for x in coverage.values()),coverage
 ck=torch.load(path,map_location='cpu',weights_only=False)
 restored=initialize('cuda');restored.load_state_dict(ck['state_dict'])
 opt2=torch.optim.AdamW(restored.parameters(),lr=.001,weight_decay=.0001);opt2.load_state_dict(ck['optimizer'])
 assert opt2.param_groups[0]['lr']==opt.param_groups[0]['lr']
 for key,value in opt.state_dict()['state'].items():
  for k,v in value.items():
   other=opt2.state_dict()['state'][key][k]
   assert torch.equal(v.cpu(),other.cpu()) if isinstance(v,torch.Tensor) else v==other
 def draws():return [rng.random(),random.random(),float(np.random.rand()),torch.rand(5).tolist(),torch.rand(5,device='cuda').cpu().tolist()]
 ref.restore(ck,rng);torch.cuda.set_rng_state_all(ck['all_cuda_rng']);random.setstate(ck['python_global_rng']);one=draws()
 ref.restore(ck,rng);torch.cuda.set_rng_state_all(ck['all_cuda_rng']);random.setstate(ck['python_global_rng']);two=draws()
 assert one==two
 bb=torch.tensor([[0,1,2,0,1,2]],device='cuda');ll=torch.tensor([0],device='cuda')
 assert net(bb,ll).shape==(1,TOKENS,2560)
 return {'all_token_slots_received_gradient':True,'optimizer_roundtrip':True,'private_global_python_numpy_torch_all_visible_cuda_rng_roundtrip':True,'tokens':TOKENS,'no_smoke_weight_reuse':True}

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
 d=config();assert sha(__file__)==d["source_sha256"]
 for p,h in d["dependencies"].items():assert sha(P/p)==h,p
 for p,h in d["models"].items():assert sha(p)==h,p
 for p,h in d["assets"].items():assert sha(p)==h,p
 for split in ["train","val"]:assert sha(DATA/(split+"_records.json"))==d["data"]["record_file_sha256"][split]
 if (ROOT/"audit/dispatch_freeze.json").exists():
  fr=read(ROOT/"audit/dispatch_freeze.json");assert fr["protocol_sha256"]==sha(ROOT/"protocol.json")
  for p,h in fr["teacher_and_plans"].items():assert sha(p)==h,p
 assert d["training"]["steps_total_per_arm"]==3600 and d["arms"][ARM]["total"]==80


def prepare():
 verify();base.seed_all();guard()
 if complete(R/"prepared.json"):return
 net=ExpandedOracle();z=net(torch.tensor([[0,1,2,0,1,2]]),torch.tensor([0]))
 assert z.shape==(1,80,2560) and sum(p.numel() for p in net.parameters())==2908160
 fresh=initialize("cpu");assert all(torch.equal(a,b) for a,b in zip(net.state_dict().values(),fresh.state_dict().values()))
 for j in range(6):
  bins=torch.tensor([[0,1,2,0,1,2]]);bins[0,j]=(bins[0,j]+1)%3;zz=net(bins,torch.tensor([0]))
  mask=torch.ones(80,dtype=torch.bool);mask[32+j*8:32+(j+1)*8]=False
  assert torch.equal(z[:,mask],zz[:,mask]) and not torch.equal(z[:,~mask],zz[:,~mask])
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(str(base.MODEL),local_files_only=True);emb=nn.Embedding(len(tok)+256,1)
 pools=[read(R/"teacher"/("round_%d_bank.json"%j)) for j in range(6)];val=read(R/"val_plan.json")
 longest=0;accepted=0;leaks=0
 for e in sum(pools,[])+val:
  assert tasks.parse(tasks.target(e),e)["correct"]
  if e.get("teacher_trajectory"):
   assert tasks.parse(e["teacher_trajectory"],e)["correct"];accepted+=1
  x=tasks.render(tok,emb,e,torch.zeros(len(e["records"]),80,1));longest=max(longest,len(x))
  assert len(x)+384<=4096
 for e in (sum(pools,[])+val)[::max(1,(accepted+len(val))//80)]:
  before=tasks.messages(e);changed=copy.deepcopy(e);changed["expected"]={"wrong":True};changed["teacher_trajectory"]="WRONG"
  for r in changed["records"]:r["label"]="WRONG";r["bins"]={f:"high" for f in tasks.FIELDS}
  assert tasks.messages(changed)==before;leaks+=1
 streams={}
 for step in [1,100,500,600,601,1200,1800,1801,2400,3000,3600]:
  ee=data_batch(step,pools,min(5,(step-1)//600));streams[str(step)]=sha_text(tasks.dumps([(e["id"],e["teacher_trajectory"]) for e in ee]))
 write(R/"audit/cpu_check.json",dict(status="passed",shape=list(z.shape),parameters=2908160,own_field_and_label_only=True,
  accepted_teacher_program_reverified=accepted,all_val_retained=len(val),max_prompt_tokens=longest,no_input_answer_leaks=leaks,
  sample_stream_hashes=streams,fresh_native80_init=True,source_sha256=sha(__file__)))
 write(R/"prepared.json",dict(status="complete",time=time.time(),init_sha256=sha(R/"oracle_init.pt"),teacher_summary_sha256=sha(R/"teacher/summary.json")))


_BATCH_CACHE={}
def data_batch(step,banks,round_):
 key=id(banks)
 if key not in _BATCH_CACHE:
  _BATCH_CACHE[key]=[{k:[e for e in bank if (e.get("operation") if e["kind"]=="simple" else "base:"+e["q"]["task"])==k]
   for k in sorted({e.get("operation") if e["kind"]=="simple" else "base:"+e["q"]["task"] for e in bank})} for bank in banks]
 indexed=_BATCH_CACHE[key];rng=random.Random(SEED+step*10007);out=[];ops=["read_level","compare_level","choose_higher","choose_lower"]
 basekeys=["base:label"]+["base:feature:"+f for f in tasks.FIELDS]+["base:report"]
 for i in range(8):
  n=(step-1)*8+i;rr=rng.randrange(round_+1) if round_ and rng.random()<.25 else round_
  k=basekeys[(n//5)%len(basekeys)] if n%5==0 else ops[(n-n//5-1)%4]
  assert indexed[rr].get(k),("Missing individually correct teacher type",rr,k)
  out.append(rng.choice(indexed[rr][k]))
 return out


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


def evaluate(role,split,round_=None,smoke=False,mode="best"):
 assert split=="val";verify();guard()
 out=R/role/"evaluation";out.mkdir(parents=True,exist_ok=True)
 ee=read(R/"val_plan.json")
 suffix="smoke" if smoke else "round_%d_val"%round_ if round_ is not None else "initial_val" if mode=="initial" else "val"
 summary=out/(suffix+".summary.json");path=out/(suffix+".jsonl")
 if complete(summary):return
 if smoke:
  ee=[e for op in ["read_level","compare_level","choose_higher","choose_lower"] for e in ee if e.get("operation")==op][:0] if False else sum([[e for e in ee if e.get("operation")==op][:4] for op in ["read_level","compare_level","choose_higher","choose_lower"]],[])+[e for e in ee if e["kind"]=="base" and e["q"]["task"]=="report"][:4]
 elif round_ is not None and round_ not in [2,5]:
  groups=collections.defaultdict(list)
  for e in ee:groups[e.get("operation","base:"+e.get("q",{}).get("task",""))].append(e)
  ee=sum([sorted(v,key=lambda e:sha_text(e["id"]+":mini"))[:48 if not k.startswith("base:") else 12] for k,v in sorted(groups.items())],[])
  assert len(ee)==288
 ckpt=R/"oracle_init.pt" if mode=="initial" else R/role/"smoke/last.pt" if smoke else R/role/"oracle"/("round_%d_last.pt"%round_ if round_ is not None else "best.pt")
 state=torch.load(ckpt,map_location="cpu",weights_only=False)["state_dict"]
 tok,emb,engine,SamplingParams,batch=engine_load();net=initialize("cuda").eval().requires_grad_(False);net.load_state_dict(state)
 params=SamplingParams(temperature=0.,max_tokens=384,seed=1014);existing={}
 if path.exists():
  for line in path.open():
   x=json.loads(line);assert x["id"] not in existing;existing[x["id"]]=x
 missing=[e for e in ee if e["id"] not in existing];began=time.time()
 with path.open("a") as f:
  for off in range(0,len(missing),batch):
   bb=missing[off:off+batch];requests=[]
   with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
    for e in bb:
     x=tasks.render(tok,emb,e,prefix(net,e)).cpu().bfloat16();assert len(x)+384<=4096;requests.append({"prompt_embeds":x})
   outputs=engine.generate(requests,params,use_tqdm=False)
   for e,o,request in zip(bb,outputs,requests):
    generation=o.outputs[0].text;score=tasks.parse(generation,e)
    row=dict(id=e["id"],mode=mode if round_ is None else "step_"+str((round_+1)*600),task=e.get("operation",e.get("q",{}).get("task")),
     kind=e["kind"],dataset=e["dataset"],sample_ids=[r["id"] for r in e["records"]],gold_label=e["records"][0]["label"],
     question=tasks.question(e),expected=tasks.expected(e),generation=generation,**score,
     field=e.get("field",e.get("q",{}).get("field")),template_id=e.get("template_id"),checkpoint_sha256=sha(ckpt),
     source_sha256=sha(__file__),prompt_tokens=len(request["prompt_embeds"]),reference_solution_supplied=False,input_interface="soft_prompt_embeds")
    existing[e["id"]]=row;f.write(tasks.dumps(row)+"\n")
   f.flush()
   if off%160==0:print("AR",suffix,off+len(bb),len(missing),flush=True)
   if time.time()>float(os.environ.get("VSTATS_STAGE_DEADLINE","inf"))-60:raise TimeoutError("OriginalAR stage budget; partial retained")
 rows=[existing[e["id"]] for e in ee];d=summarize(rows);d.update(seconds=time.time()-began,checkpoint_sha256=sha(ckpt),plan_sha256=sha(R/"val_plan.json"),source_sha256=sha(__file__))
 write(summary,d)
 del engine;del net


def promotion_gate():
 rows=[json.loads(l) for l in (R/"curriculum/evaluation/val.jsonl").open()]
 text=read(R/"teacher/val_text_raw.json");checks=[]
 for op in ["read_level","compare_level","choose_higher","choose_lower"]:
  rr=[r for r in rows if r["task"]==op];paired=[r for r in rr if text[r["id"]]["correct"]]
  value=sum(r["correct"] for r in paired)/len(paired) if paired else None
  checks.append(dict(name=op,support=len(paired),value=value,minimum=.9,minimum_support=16,passed=len(paired)>=16 and value>=.9))
 report=[r for r in rows if r["task"]=="report"]
 value=sum(r["correct"] for r in report)/len(report) if report else None
 checks.append(dict(name="basic_full_report_exact",support=len(report),value=value,minimum=.9,passed=bool(report) and value>=.9))
 coverage=read(R/"teacher/summary.json")
 passed=all(c["passed"] for c in checks) and len(coverage["all31_synthetic_or_real_label_ids"])==31
 result=dict(status="complete",time=time.time(),passed=passed,checks=checks,full_Val_not_text_filtered=True,
  coverage=coverage["accepted_train_wordings"],decision="EEG_alignment_authorized_only_if_passed",
  teacher_nativeID_vs_Oracle_promptEmbeds_interface_difference=True,not_a_clinical_or_continuous_value_claim=True)
 write(R/"oracle_promotion_gate.json",result);return passed


def pipeline():
 W=ROOT/"worker_master";W.mkdir(exist_ok=True)
 with (W/"pipeline.lock").open("a+") as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  teacher=PLAN/"real_teacher_4b_v1";di=read(teacher/"bootstrap_dispatch.json")["identity"]
  while True:
   live=same_identity(di,proc(di["pid"]))
   if (teacher/"audit/final_verification.json").exists() and not live and lock_free(teacher/"bootstrap.lock"):break
   if not live and (teacher/"error_coordinate.json").exists():raise RuntimeError("Fresh4B teacher stopped; inspect actualerror")
   assert time.time()<read(teacher/"bootstrap_dispatch.json")["time"]+7800
   write(W/"status.json",dict(state="waiting_fresh_unassisted4b_teacher",time=time.time(),teacher_live=live));time.sleep(10)
  assert complete(teacher/"audit/final_verification.json") and not read(teacher/"audit/final_verification.json")["process_exit_check_pending"]
  assert complete(R/"teacher/summary.json")
  paths=[R/"teacher"/("round_%d_bank.json"%r) for r in range(6)]+[R/"val_plan.json",R/"tf_val.json",R/"teacher/summary.json",R/"teacher/val_text_raw.json"]
  if not (ROOT/"audit/dispatch_freeze.json").exists():
   write(ROOT/"audit/dispatch_freeze.json",dict(time=time.time(),protocol_sha256=sha(ROOT/"protocol.json"),teacher_and_plans={str(p):sha(p) for p in paths},source_sha256=sha(__file__)))
  prepare()
  if not (ROOT/"first_launch.json").exists():write(ROOT/"first_launch.json",dict(started=time.time(),budget_seconds=43200,smoke_included=True))
  write(W/"launch.json",dict(identity=proc(os.getpid()),started=read(ROOT/"first_launch.json")["started"],source_sha256=sha(__file__)))
  try:
   stage("curriculum","smoke",["train","--smoke"],"curriculum/smoke/summary.json")
   stage("curriculum","ar_smoke",["eval","--smoke"],"curriculum/evaluation/smoke.summary.json")
   stage("curriculum","initial_val",["eval","--mode","initial"],"curriculum/evaluation/initial_val.summary.json")
   for rr in range(6):
    stage("curriculum","train_round_%d"%rr,["train","--round",str(rr)],"curriculum/oracle/round_%d.complete.json"%rr)
    stage("curriculum","progress_round_%d"%rr,["eval","--round",str(rr)],"curriculum/evaluation/round_%d_val.summary.json"%rr)
   stage("curriculum","selected_best_val",["eval"],"curriculum/evaluation/val.summary.json")
   passed=promotion_gate();verify()
   write(ROOT/"oracle_freeze.json",dict(time=time.time(),best_sha256=sha(R/"curriculum/oracle/best.pt"),init_sha256=sha(R/"oracle_init.pt"),
    source_sha256=sha(__file__),protocol_sha256=sha(ROOT/"protocol.json"),teacher_freeze_sha256=sha(ROOT/"audit/dispatch_freeze.json"),
    selection="fixedValTF240CE_only",EEG_gate_passed=passed))
   write(W/"complete.json",dict(status="complete",time=time.time(),EEG_alignment_gate_passed=passed,report_pending=True))
   write(W/"status.json",dict(state="Oracle_complete_EEG_gate_passed" if passed else "Oracle_complete_EEG_gate_failed",time=time.time()))
  except BaseException:
   write(ROOT/"abort.json",dict(time=time.time(),error=traceback.format_exc()))
   write(W/"status.json",dict(state="stopped_original_budget_or_error",time=time.time(),traceback=traceback.format_exc()));raise


def finalize():
 assert not cache_owned(),"Owned child remains"
 for name in ["worker_master/pipeline.lock"]+["gpu%d.lease"%g for g in range(2,8)]:assert lock_free(ROOT/name),name
 verify();evaluations={};rows_all=[]
 for p in (R/"curriculum/evaluation").glob("*.jsonl"):
  rows=[json.loads(l) for l in p.open()];plan={e["id"]:e for e in read(R/"val_plan.json")}
  assert len({r["id"] for r in rows})==len(rows)
  for r in rows:
   assert r["expected"]==tasks.expected(plan[r["id"]]) and r["question"]==tasks.question(plan[r["id"]])
   sc=tasks.parse(r["generation"],plan[r["id"]]);assert sc["valid"]==r["valid"] and sc["correct"]==r["correct"]
  evaluations[p.stem]=summarize(rows);rows_all+=rows
 gate=read(R/"oracle_promotion_gate.json") if (R/"oracle_promotion_gate.json").exists() else None
 history=read(R/"curriculum/oracle/summary.json") if (R/"curriculum/oracle/summary.json").exists() else {}
 write(ROOT/"audit/final_verification.json",dict(time=time.time(),status="passed_for_available_artifacts",process_exit_check_pending=False,
  available_evaluation_counts={k:v["queries"] for k,v in evaluations.items()},source_sha256=sha(__file__),
  init_sha256=sha(R/"oracle_init.pt"),formal_steps=history.get("steps",0),
  all_predeclared_Oracle_evaluations_complete=complete(ROOT/"worker_master/complete.json"),EEG_alignment_launched=False))
 write(ROOT/"results_summary.json",dict(training=history,evaluations=evaluations,gate=gate,Test="N/A_not_part_of_tonight_Val_conditioned_alignment_diagnostic"))
 lines=["# 4B简单题Oracle","", "冻结原始4B，只训练80-token编码表；逐条使用4B无答案明文答对的Train样本。完整Val保留全部未见题面。","",
  "正式提交步数："+str(history.get("steps",0))+" / 3600。1800和3600步在同一新题库对照。","",
  "原生明文token-ID与Oracle prompt_embeds执行路径不同；不把小题库Val结果当临床、连续值或任意语言能力。",""]
 for name,d in evaluations.items():
  lines+=["## "+name,"","|题型|正确/总数|正确率|","|---|---:|---:|"]
  for m in d["metrics"]:lines.append("|"+m["task"]+"|"+str(m["correct"])+"/"+str(m["n"])+"|"+format(m["exact"]*100,".1f")+"%|")
  lines+=[""]
 if gate:lines+=["EEG自动对齐门槛："+("通过" if gate["passed"] else "未通过")+"。",tasks.dumps(gate["checks"])]
 if (ROOT/"abort.json").exists():lines+=["","存在预算或运行中断；缺失结果保留N/A，详见abort.json。"]
 (ROOT/"final_report.md").write_text("\n".join(lines)+"\n")


def main():
 p=argparse.ArgumentParser();p.add_argument("command",choices=["pipeline","prepare","train","eval","finalize","check"])
 p.add_argument("--role",default="curriculum");p.add_argument("--round",type=int,default=None);p.add_argument("--split",default="val");p.add_argument("--smoke",action="store_true");p.add_argument("--mode",default="best");a=p.parse_args()
 signal.signal(signal.SIGTERM,lambda *x:(_ for _ in ()).throw(SystemExit("Owned process stopped; preserve artifacts")))
 if os.environ.get("CUDA_VISIBLE_DEVICES"):torch.cuda.set_per_process_memory_fraction(min(1.,int(os.environ["VSTATS_COMPACT_ALLOCATOR_BUDGET_MIB"])/int(os.environ["VSTATS_COMPACT_GPU_TOTAL_MIB"])))
 if a.command=="pipeline":
  try:pipeline()
  finally:
   if (R/"val_plan.json").exists():finalize()
 elif a.command=="prepare":prepare()
 elif a.command=="train":train(a.role,a.round or 0,a.smoke)
 elif a.command=="eval":evaluate(a.role,a.split,a.round,a.smoke,a.mode)
 elif a.command=="finalize":finalize()
 elif a.command=="check":print(tasks.dumps(dict(owned=cache_owned(),gpus=gpus(),master=read(ROOT/"worker_master/status.json") if (ROOT/"worker_master/status.json").exists() else None)))


if __name__=="__main__":main()
