#!/usr/bin/env python3
"""Compact28 factorized Oracle and fixed-stat-expert interface experiment."""

if __name__ == "__main__":
    raise SystemExit("Reference snapshot only. Use tools/workbench.py for inspection; configure a fresh run before training.")

from pathlib import Path
import argparse,collections,copy,csv,fcntl,hashlib,json,math,os,random,shutil,signal,subprocess,sys,time,traceback
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import statsem_1014_train as base
P=Path(__file__).resolve().parents[4]
R=P/'experiments/mainline_vrl/runs/stat_compact28_20261015_s1014'
SR=P/'experiments/mainline_vrl/runs/stat_expert_20261015_s1014'
OLD=P/'experiments/mainline_vrl/runs/stat_sleep_20261014_s1014/statistics'
DATA=OLD/'data';OROOT=OLD/'fields112'
ARPY='/home/workspace/yuqiudeng/envs/eeglm-vllm/bin/python'
TRAINPY=str(P/'.venv/bin/python')
FIELDS=base.FIELDS;SEED=1014


def read(p):return json.loads(Path(p).read_text())

def write(p,x):base.write_json(p,x)

def sha(p):return base.sha(p)

def seed():base.seed_all()

def guard(extra=0):
 free=shutil.disk_usage(R).free
 if free<12*2**30+extra:raise OSError(f'12GiB storage guard: free={free} extra={extra}')

def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 # Atomic saves must have temporary-write headroom in addition to the reserve.
 tensors=x.get('state_dict',x.get('model',{})) if isinstance(x,dict) else {}
 approx=sum(v.numel()*v.element_size() for v in tensors.values() if isinstance(v,torch.Tensor))
 guard(max(approx*3,64*2**20));base.save_checkpoint(p,x)

def rng_state(rng):
 return {'rng':rng.getstate(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'numpy_rng':np.random.get_state()}

def restore(ck,rng):
 rng.setstate(ck['rng']);torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state(ck['cuda_rng']);np.random.set_state(ck['numpy_rng'])

def records(split):
 if split=='test':assert (R/'test_freeze.json').exists(),'Global model freeze required'
 return read(DATA/(split+'_records.json'))

def old_data(split):
 if split=='test':assert (R/'test_freeze.json').exists()
 return torch.load(DATA/(split+'.pt'),map_location='cpu',weights_only=False)

class Readout(nn.Module):
 def __init__(self):
  super().__init__();self.body=nn.Sequential(nn.Linear(7500,256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,128),nn.GELU());self.stat=nn.Linear(128,18)
 def forward(self,x):return self.stat(self.body(x)).reshape(-1,6,3)

def pick(recs,rng,batch,homogeneous=False):
 groups=collections.defaultdict(lambda:collections.defaultdict(list))
 for i,r in enumerate(recs):groups[r['dataset']][r['label']].append(i)
 keys=sorted(groups)
 def draw():
  dd=[rng.choice(keys)]*batch if homogeneous else [rng.choice(keys) for _ in range(batch)]
  return [rng.choice(groups[d][rng.choice(sorted(groups[d]))]) for d in dd]
 return draw

def normalized(data,mean_=None,std=None):
 xx=(data['bank']*data['valid'][:,None,:,None]).flatten(1).float()
 if mean_ is None:mean_=xx.mean(0);std=xx.std(0).clamp_min(.01)
 return ((xx-mean_)/std).clamp(-10,10),mean_,std

def probe_model(arm):
 ck=torch.load(SR/('probe_'+arm)/'best.pt',map_location='cpu',weights_only=False);net=Readout().cuda().eval();net.load_state_dict(ck['state_dict'])
 return net,ck['mean'].cuda(),ck['std'].cuda()

class StatsHead(nn.Module):
 def __init__(self):
  super().__init__();self.field_embedding=nn.Parameter(torch.randn(6,2048)*.02);self.field_scale=nn.Parameter(torch.randn(6,2048)*.1)
  self.flat_projection=nn.Sequential(nn.LayerNorm(7500),nn.Linear(7500,2048,bias=False),nn.GELU());self.stats_norm=nn.LayerNorm(2048);self.stats_projection=nn.Linear(2048,2*2048,bias=False)
  self.output_norm=nn.LayerNorm(2048);self.ffn_norm=nn.LayerNorm(2048);self.ffn=nn.Sequential(nn.Linear(2048,2048),nn.GELU(),nn.Linear(2048,2048));nn.init.normal_(self.stats_projection.weight,std=.0001)
 def forward(self,bank,valid):
  h=self.flat_projection((bank.float()*valid[:,None,:,None]).flatten(1));c=self.stats_norm(F.gelu(h[:,None]*(1+torch.tanh(self.field_scale))[None]+self.field_embedding[None]))
  z=self.stats_projection(c).reshape(len(h),12,2048);z=self.output_norm(z);return z+self.ffn(self.ffn_norm(z))

class LabelPrefix(nn.Module):
 def __init__(self):
  super().__init__();m=base.StatStudent(112);m.load_state_dict(torch.load(OROOT/'student/best.pt',map_location='cpu',weights_only=False)['state_dict'])
  self.flat_projection=m.flat_projection;self.output_norm=m.output_norm;self.ffn_norm=m.ffn_norm;self.ffn=m.ffn
  for p in self.parameters():p.requires_grad_(False)
 def forward(self,bank,valid):
  flat=(bank.float()*valid[:,None,:,None]).flatten(1);p=self.flat_projection(flat).reshape(len(flat),16,2048);p=self.output_norm(p);return p+self.ffn(self.ffn_norm(p))

def llm_train(arm,smoke=False):
 seed();out=R/('llm_'+arm)/('smoke' if smoke else 'train');out.mkdir(parents=True,exist_ok=True)
 tok,lm,embedding=base.hf_decoder();label=LabelPrefix().cuda().eval();oracle=load_compact('cuda').eval()
 for p in oracle.parameters():p.requires_grad_(False)
 net=StatsHead().cuda();net.load_state_dict(torch.load(R/'llm_init.pt',map_location='cpu',weights_only=False)['state_dict'])
 tr=old_data('train');va=old_data('val');bt=bank_data(arm,'train');bv=bank_data(arm,'val');rr=records('train');vr=records('val')
 opt=torch.optim.AdamW(net.parameters(),lr=1e-5,weight_decay=.0001);rng=random.Random(SEED);draw=pick(rr,rng,8);steps=30 if smoke else 6000
 vi=base.actual_indices(R/('llm_'+arm),'val',vr);vpairs=[(i,base.fixed_questions(vr[i])[j%9]) for j,i in enumerate(vi)]
 start=0;best=math.inf;history=[];began=time.time();micro=int(os.environ.get('VSTATS_COMPACT_MICRO_BATCH','8'));assert micro in [1,2,4,8]
 if (out/'last.pt').exists():
  ck=torch.load(out/'last.pt',map_location='cpu',weights_only=False);net.load_state_dict(ck['state_dict']);opt.load_state_dict(ck['optimizer']);restore(ck,rng);start=ck['step'];best=ck['best'];history=ck['history']
 initial=net.field_embedding.detach().clone()
 def prefix(ii,banks,old):
  with torch.no_grad():lp=label(old['bank'][ii].cuda(),old['valid'][ii].cuda())
  return torch.cat([lp,net(banks['bank'][ii].cuda(),banks['valid'][ii].cuda())],1)
 def state(step):return {'state_dict':{k:v.detach().cpu() for k,v in net.state_dict().items()},'optimizer':opt.state_dict(),'step':step,'best':best,'history':history,**rng_state(rng),'source_sha256':sha(__file__),'effective_batch':8,'micro_batch':micro}
 for step in range(start+1,steps+1):
  net.train();opt.zero_grad(set_to_none=True);ii=draw();rs=[rr[i] for i in ii];qs=[base.question(r,rng.choice(base.TASKS),rng) for r in rs]
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
   bins,labs=base.teacher_batch(rs,'cuda');target=oracle(bins,labs)[:,16:]
  denominator=target.float().square().mean().clamp_min(1e-8);loss_value=ce_value=align_value=0.
  for off in range(0,8,micro):
   jj=ii[off:off+micro];qq=qs[off:off+micro];tt=target[off:off+micro]
   with torch.autocast('cuda',dtype=torch.bfloat16):
    pf=prefix(jj,bt,tr);z=pf[:,16:]
    align_part=.1*((z.float()-tt.float()).square().mean()/denominator+1-F.cosine_similarity(z.float().flatten(1),tt.float().flatten(1)).mean())
    ce_part=base.sft_loss(lm,tok,embedding,qq,pf);loss_part=ce_part+align_part
   assert torch.isfinite(loss_part);(loss_part*(len(qq)/8)).backward()
   loss_value+=float(loss_part)*len(qq)/8;ce_value+=float(ce_part)*len(qq)/8;align_value+=float(align_part)*len(qq)/8
  loss=torch.tensor(loss_value);ce=torch.tensor(ce_value);align=torch.tensor(align_value)
  grad=torch.nn.utils.clip_grad_norm_(net.parameters(),1.);assert torch.isfinite(grad) and grad>0
  assert all(p.grad is None for p in lm.parameters()) and all(p.grad is None for p in label.parameters());opt.step()
  if step%10==0:print(json.dumps({'step':step,'loss':float(loss),'ce':float(ce),'alignment':float(align),'grad':float(grad),'seconds':time.time()-began}),flush=True)
  if smoke and step==30:
   assert not torch.equal(initial,net.field_embedding);save(out/'last.pt',state(step));write(out/'smoke.json',{'status':'passed','steps':30,'decoder_frozen':True,'label_branch_frozen':True,'parameters_changed':True,'finite_gradients':True,'tokens':28,'effective_batch':8,'micro_batch':micro})
  if not smoke and (step%500==0 or step==steps):
   net.eval();losses=[]
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
    for off in range(0,len(vpairs),micro):
     batch=vpairs[off:off+micro];jj=[i for i,q in batch];losses.append(float(base.sft_loss(lm,tok,embedding,[q for i,q in batch],prefix(jj,bv,va))))
   val=float(np.mean(losses));improved=val<best;best=min(best,val);history.append({'step':step,'val_tf_loss':val,'seconds':time.time()-began});ck=state(step);save(out/'last.pt',ck)
   if improved:save(out/'best.pt',ck)
   write(out/'summary.json',{'status':'complete' if step==steps else 'running','step':step,'best_val_tf':best,'history':history,'peak_cuda_gib':torch.cuda.max_memory_allocated()/2**30,'trainable_parameters':sum(p.numel() for p in net.parameters()),'label_frozen':True,'lm_frozen':True});print('VALIDATION',history[-1],flush=True)
  if time.time()-began>3.85*3600:
   save(out/'last.pt',state(step));raise TimeoutError('LLM stage four-hour budget')

def llm_eval(arm,split,mode):
 from transformers import AutoTokenizer,AutoModelForCausalLM
 from vllm import LLM,SamplingParams
 seed();out=R/('llm_'+arm)/'evaluation';out.mkdir(exist_ok=True);path=out/(split+'_'+mode+'.jsonl')
 if path.with_suffix('.summary.json').exists():return
 if path.exists():
  fail=out/'failed_attempts';fail.mkdir(exist_ok=True);path.rename(fail/(path.name+'.'+str(int(time.time()))))
 if split=='test':assert (R/'test_freeze.json').exists()
 data=old_data(split);banks=bank_data(arm,split);rr=records(split);indices=base.actual_indices(R/('llm_'+arm),split,rr)
 tok=AutoTokenizer.from_pretrained(base.MODEL,local_files_only=True);lm=AutoModelForCausalLM.from_pretrained(base.MODEL,torch_dtype=torch.bfloat16,local_files_only=True).cpu();embedding=copy.deepcopy(lm.get_input_embeddings());del lm
 label=LabelPrefix().cuda().eval()
 if mode=='continuous':
  model=StatsHead().cuda().eval();model.load_state_dict(torch.load(R/('llm_'+arm)/'train/best.pt',map_location='cpu',weights_only=False)['state_dict'])
 else:
  model,mu,sd=probe_model(arm);oracle=load_compact('cuda').eval()
 engine=LLM(model=str(base.MODEL),dtype='bfloat16',enable_prompt_embeds=True,max_model_len=2048,max_num_seqs=int(os.environ.get('VSTATS_COMPACT_AR_BATCH','16')),gpu_memory_utilization=float(os.environ.get('VSTATS_COMPACT_AR_MEMORY_UTILIZATION','.35')),enforce_eager=True,disable_log_stats=True,seed=SEED)
 pairs=[(i,q) for i in indices for q in base.fixed_questions(rr[i])];rows=[]
 with path.open('w') as f:
  for off in range(0,len(pairs),int(os.environ.get('VSTATS_COMPACT_AR_BATCH','16'))):
   batch=pairs[off:off+int(os.environ.get('VSTATS_COMPACT_AR_BATCH','16'))];ii=[i for i,q in batch]
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
    lp=label(data['bank'][ii].cuda(),data['valid'][ii].cuda());b=banks['bank'][ii].cuda();v=banks['valid'][ii].cuda()
    if mode=='continuous':sp=model(b,v)
    else:
     xx=((b.float()*v[:,None,:,None]).flatten(1)-mu)/sd
     with torch.autocast('cuda',enabled=False):bins=model(xx.clamp(-10,10)).argmax(-1)
     sp=torch.stack([oracle.stat_codes[j,bins[:,j]] for j in range(6)],1).flatten(1,2)
    pf=torch.cat([lp,sp],1).cpu()
   requests=[{'prompt_embeds':base.prompt(tok,embedding,q,p).to(torch.bfloat16)} for (i,q),p in zip(batch,pf)]
   outputs=engine.generate(requests,SamplingParams(temperature=0.,max_tokens=384),use_tqdm=False)
   for (i,q),o in zip(batch,outputs):
    text=o.outputs[0].text;pred,valid,fact=base.parse_generation(text,q);row={'id':rr[i]['id'],'dataset':rr[i]['dataset'],'task':q['task'],'arm':arm,'mode':mode,'split':split,'question':q['question'],'expected':q['answer'],'prediction':pred,'generation':text,'valid':valid,'correct':valid and pred==q['answer'],'fact_score':fact,'oracle_uses_gold':False,'gold_label':rr[i]['label'],'gold_bins':base.bin_values(rr[i])};rows.append(row);f.write(json.dumps(row,ensure_ascii=False)+'\n')
   f.flush()
   if off%256==0:print('AR',arm,split,mode,off,len(pairs),flush=True)
 result=base.metrics(rows);result.update({'status':'complete','arm':arm,'mode':mode,'split':split,'tokens':28,'deterministic_free_generation':True,'temperature':0.,'source_sha256':sha(__file__),'predictor_sha256':sha(SR/('probe_'+arm)/'best.pt') if mode=='predicted_code' else None,'projector_sha256':sha(R/('llm_'+arm)/'train/best.pt') if mode=='continuous' else None})
 write(path.with_suffix('.summary.json'),result);write(path.with_suffix('.failures.json'),[x for x in rows if not x['correct']][:40])

def complete(p):
 if not p.exists():return False
 x=read(p);return x.get('status') in ['complete','passed','frozen']

def stage(role,name,args,artifact=None):
 root=R/'worker_main';root.mkdir(exist_ok=True);log=root/(name+'.log')
 if artifact and complete(R/artifact):print('SKIP',name,flush=True);return
 worker_deadline=float(os.environ['VSTATS_STAT_EXPERT_DEADLINE'])
 stage_meta=root/(name+'.attempt_state.json')
 meta=read(stage_meta) if stage_meta.exists() else {'first_started':None,'attempts':0,'micro_batch':8,'ar_batch':16,'ar_memory_utilization':.35}
 is_ar=args[0] in ['llm-eval','oracle-eval']
 is_gpu=is_ar or args[0] in ['oracle-train','llm-train']
 while True:
  deadline=min(worker_deadline,(meta['first_started']+4*3600) if meta['first_started'] is not None else worker_deadline)
  if time.time()>=deadline:raise TimeoutError('Original stage/worker budget reached before '+name)
  gpu_lock=None;env=dict(os.environ);chosen=None
  try:
   if is_gpu:
    need_mib=(24 if is_ar else 26)*1024
    while gpu_lock is None:
     q=subprocess.run(['nvidia-smi','--query-gpu=index,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
     cards=[]
     for line in q.stdout.splitlines():
      gpu,total,free,util=[int(v.strip()) for v in line.split(',')]
      if gpu in range(2,8) and free>=need_mib:cards.append((gpu,total,free,util))
     cards.sort(key=lambda x:(-x[2],x[3],x[0]))
     for gpu,total,free,util in cards:
      f=(R/('gpu'+str(gpu)+'.lease')).open('a+')
      try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
      except BlockingIOError:f.close();continue
      # Recheck after acquiring our lease; other users are allowed to share.
      check=subprocess.run(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
      free=int(check.stdout.strip())
      if free>=need_mib:
       gpu_lock=f;chosen=str(gpu);env['CUDA_VISIBLE_DEVICES']=chosen
       # Leave8GiB physical reserve plus2GiB allowance for non-Torch allocations.
       env['VSTATS_COMPACT_ALLOCATOR_BUDGET_MIB']=str(free-10*1024)
       env['VSTATS_COMPACT_GPU_TOTAL_MIB']=str(total)
       break
      fcntl.flock(f,fcntl.LOCK_UN);f.close()
     if gpu_lock is None:
      write(root/'status.json',{'state':'waiting_gpu_memory','stage':name,'gpu_candidates':list(range(2,8)),'required_free_mib':need_mib,'pipeline_pid':os.getpid(),'time':time.time(),'note':'User authorizes GPU sharing; wait only for enough free VRAM, never stop others.'})
      if time.time()>=deadline:raise TimeoutError('Original stage/worker budget waiting VRAM')
      time.sleep(10)
   env['VSTATS_COMPACT_MICRO_BATCH']=str(meta['micro_batch'])
   env['VSTATS_COMPACT_AR_BATCH']=str(meta['ar_batch'])
   env['VSTATS_COMPACT_AR_MEMORY_UTILIZATION']=str(meta['ar_memory_utilization'])
   guard()
   if meta['first_started'] is None:meta['first_started']=time.time()
   meta['attempts']+=1;write(stage_meta,meta)
   cmd=[ARPY if is_ar else TRAINPY,'-B',str(Path(__file__).resolve())]+args
   with log.open('a') as f:
    f.write(json.dumps({'stage_attempt':meta['attempts'],'time':time.time(),'gpu':chosen,'micro_batch':meta['micro_batch'],'ar_batch':meta['ar_batch'],'ar_memory_utilization':meta['ar_memory_utilization'],'allocator_budget_mib':env.get('VSTATS_COMPACT_ALLOCATOR_BUDGET_MIB')})+'\n');f.flush()
    child=subprocess.Popen(cmd,cwd=P,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=env)
    write(root/'status.json',{'state':'running','stage':name,'pid':child.pid,'pipeline_pid':os.getpid(),'gpu':chosen,'log':str(log),'command':cmd,'time':time.time(),'shared_gpu':True,'micro_batch':meta['micro_batch'],'effective_batch':8 if not is_ar else None,'ar_batch':meta['ar_batch'],'allocator_budget_mib':env.get('VSTATS_COMPACT_ALLOCATOR_BUDGET_MIB'),'physical_reserve_gib':8})
    try:code=child.wait(timeout=max(1,min(worker_deadline,meta['first_started']+4*3600)-time.time()))
    except subprocess.TimeoutExpired:
     os.killpg(child.pid,signal.SIGTERM)
     try:child.wait(timeout=30)
     except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
     raise TimeoutError('Original stage/worker budget '+name)
   if code==0:
    if artifact:assert complete(R/artifact),'Stage exited without completed required artifact '+str(artifact)
    return
   tail=log.read_text()[-16000:].lower()
   oom=any(x in tail for x in ['cuda out of memory','cuda error: out of memory','torch.outofmemoryerror'])
   pressure='resource_headroom_low' in tail
   if not (oom or pressure):raise RuntimeError(f'Stage {name} exit {code}; see {log}')
   event={'time':time.time(),'stage':name,'failed_child_pid':child.pid,'gpu':chosen,'exit_code':code,'oom':oom,'headroom_pressure':pressure,'attempt_before':dict(meta),'source_sha256':sha(__file__),'resume':'Existing formal last.pt restores optimizer and all RNG; absent checkpoint restarts fixed init. No other process stopped.'}
   if oom:
    if is_ar:
     if meta['ar_batch']==1 and meta['ar_memory_utilization']<=.2:raise RuntimeError('AR OOM at smallest resource configuration; stop for review')
     meta['ar_batch']=max(1,meta['ar_batch']//2);meta['ar_memory_utilization']=max(.2,round(meta['ar_memory_utilization']-.05,2))
    else:
     if meta['micro_batch']==1:raise RuntimeError('Training OOM at microbatch1; stop for review')
     meta['micro_batch']=max(1,meta['micro_batch']//2)
   event['attempt_next']=dict(meta);write(stage_meta,meta)
   with (root/'resource_retries.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
  finally:
   if gpu_lock is not None:fcntl.flock(gpu_lock,fcntl.LOCK_UN);gpu_lock.close()



def bank_data(arm,split):
 assert arm=='new'
 if split=='test':assert (R/'test_freeze.json').exists()
 return torch.load(SR/'banks'/(split+'.pt'),map_location='cpu',weights_only=False)

class CompactOracle(nn.Module):
 def __init__(self,base_codes,stats):
  super().__init__()
  self.label_codes=nn.Parameter(base_codes['codes'].float().clone())
  self.stat_codes=nn.Parameter(stats.float().clone())
  self.register_buffer('label_position',base_codes['positions'][0].float().clone())
  self.register_buffer('initial_label',base_codes['codes'].float().clone(),persistent=False)
  self.register_buffer('initial_stats',stats.float().clone(),persistent=False)
  self.tokens=28
 def forward(self,bins,labels):
  stats=torch.stack([self.stat_codes[j,bins[:,j]] for j in range(6)],1).flatten(1,2)
  label=self.label_codes[labels]+self.label_position[None,None]
  return torch.cat([label,stats],1)
 def anchor(self):
  a=(self.label_codes-self.initial_label).square().mean()/self.initial_label.square().mean().clamp_min(1e-8)
  b=(self.stat_codes-self.initial_stats).square().mean()/self.initial_stats.square().mean().clamp_min(1e-8)
  return .0001*(a+b)

def load_compact(device,checkpoint='best'):
 b=torch.load(R/'base_codes.pt',map_location='cpu',weights_only=False)
 st=torch.load(R/'initial_stats.pt',map_location='cpu',weights_only=False)
 m=CompactOracle(b,st).to(device)
 path=R/('oracle_init.pt' if checkpoint=='init' else 'oracle/'+checkpoint+'.pt')
 m.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['state_dict'])
 return m

def prepare():
 seed()
 base_codes=torch.load(OROOT/'base_codes.pt',map_location='cpu',weights_only=False)
 initial=torch.load(OROOT/'initial_stats.pt',map_location='cpu',weights_only=False)
 assert initial.shape==(6,3,16,2048)
 compact=initial.reshape(6,3,2,8,2048).mean(3)
 save(R/'base_codes.pt',base_codes);torch.save(compact,R/'initial_stats.pt')
 oracle=CompactOracle(base_codes,compact)
 save(R/'oracle_init.pt',{'state_dict':oracle.state_dict(),'seed':SEED,'tokens':28})
 # Preserve common projector initial weights, compress only the output projection.
 net=StatsHead()
 original=torch.load(SR/'llm_init.pt',map_location='cpu',weights_only=False)['state_dict']
 state=net.state_dict()
 for k in state:
  if k=='stats_projection.weight':
   state[k]=(original[k].reshape(2,8,2048,2048).mean(1)*math.sqrt(8)).reshape(4096,2048)
  else:
   assert state[k].shape==original[k].shape,k
   state[k]=original[k]
 net.load_state_dict(state);save(R/'llm_init.pt',{'state_dict':net.state_dict(),'seed':SEED,'tokens':28})
 shutil.copyfile(OROOT/'synthetic_val.json',R/'synthetic_val.json')
 write(R/'llm_new/protocol.json',{'arm':'new','actual_val_cap_per_dataset':16,'actual_test_cap_per_dataset':128,'tokens':28})
 bb=torch.tensor([[0,1,2,0,1,2]]);ll=torch.tensor([0]);pref=oracle(bb,ll)
 assert pref.shape==(1,28,2048)
 for j in range(6):
  changed=bb.clone();changed[0,j]=(changed[0,j]+1)%3
  other=oracle(changed,ll);mask=torch.ones(28,dtype=torch.bool);mask[16+2*j:18+2*j]=False
  assert torch.equal(pref[:,mask],other[:,mask])
  assert not torch.equal(pref[:,~mask],other[:,~mask])
 other=oracle(bb,torch.tensor([1]));assert torch.equal(pref[:,16:],other[:,16:])
 write(R/'audit/structural_smoke.json',{'status':'passed','tokens':28,'stat_codes_shape':list(oracle.stat_codes.shape),'single_field_changes_only_own_pair':True,'label_changes_do_not_change_stat_codes':True,'pair_tokens_may_cooperate':True,'not_a_gpu_training_smoke':True})
 write(R/'prepared.json',{'status':'complete','time':time.time(),'init_sha256':sha(R/'oracle_init.pt'),'projector_init_sha256':sha(R/'llm_init.pt')})

def oracle_train(smoke=False):
 seed();out=R/('oracle_smoke' if smoke else 'oracle');out.mkdir(exist_ok=True)
 tok,lm,embedding=base.hf_decoder();net=load_compact('cuda','init')
 opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
 rng=random.Random(SEED);labels=read(R/'protocol.json')['labels'];steps=30 if smoke else 4000
 vr=read(R/'synthetic_val.json');vp=[(r,base.fixed_questions(r)[i%9]) for i,r in enumerate(vr)]
 start=0;best=math.inf;history=[];began=time.time();micro=int(os.environ.get('VSTATS_COMPACT_MICRO_BATCH','8'));assert micro in [1,2,4,8]
 if (out/'last.pt').exists():
  ck=torch.load(out/'last.pt',map_location='cpu',weights_only=False);net.load_state_dict(ck['state_dict']);opt.load_state_dict(ck['optimizer']);restore(ck,rng);start=ck['step'];best=ck['best'];history=ck['history']
 initial=net.stat_codes.detach().clone()
 def state(step):
  return {'state_dict':{k:v.detach().cpu() for k,v in net.state_dict().items()},'optimizer':opt.state_dict(),'step':step,'best':best,'history':history,**rng_state(rng),'source_sha256':sha(__file__),'effective_batch':8,'micro_batch':micro,'tokens':28,'effective_batch':8,'micro_batch':micro,'decoder_frozen':True}
 for step in range(start+1,steps+1):
  net.train();opt.zero_grad(set_to_none=True)
  rs=[base.synthetic_record(rng,labels,'train',step*8+k) for k in range(8)]
  qs=[base.question(r,rng.choice(base.TASKS),rng) for r in rs];bb,ll=base.teacher_batch(rs,'cuda')
  ce_value=0.;loss_value=0.
  for off in range(0,8,micro):
   with torch.autocast('cuda',dtype=torch.bfloat16):
    pf=net(bb[off:off+micro],ll[off:off+micro]);ce_part=base.sft_loss(lm,tok,embedding,qs[off:off+micro],pf);loss_part=ce_part+net.anchor()
   assert torch.isfinite(loss_part);(loss_part*(len(qs[off:off+micro])/8)).backward()
   ce_value+=float(ce_part)*len(qs[off:off+micro])/8;loss_value+=float(loss_part)*len(qs[off:off+micro])/8
  ce=torch.tensor(ce_value);loss=torch.tensor(loss_value);grad=torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
  assert torch.isfinite(grad) and grad>0 and all(p.grad is None for p in lm.parameters());opt.step()
  if step%10==0:print(json.dumps({'step':step,'loss':float(loss),'ce':float(ce),'grad':float(grad),'seconds':time.time()-began}),flush=True)
  if smoke and step==30:
   assert not torch.equal(initial,net.stat_codes);save(out/'last.pt',state(step))
   write(out/'smoke.json',{'status':'passed','steps':30,'finite_gradients':True,'decoder_frozen':True,'stats_codes_changed':True,'tokens':28,'effective_batch':8,'micro_batch':micro,'peak_cuda_gib':torch.cuda.max_memory_allocated()/2**30})
  if not smoke and (step%500==0 or step==steps):
   net.eval();losses=[]
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
    for off in range(0,len(vp),micro):
     group=vp[off:off+micro];bb,ll=base.teacher_batch([r for r,q in group],'cuda')
     losses.append(float(base.sft_loss(lm,tok,embedding,[q for r,q in group],net(bb,ll))))
   val=float(np.mean(losses));improved=val<best;best=min(best,val);history.append({'step':step,'val_tf_loss':val,'seconds':time.time()-began})
   ck=state(step);save(out/'last.pt',ck)
   if improved:save(out/'best.pt',ck)
   write(out/'summary.json',{'status':'complete' if step==steps else 'running','step':step,'best_val_tf':best,'history':history,'tokens':28,'effective_batch':8,'micro_batch':micro,'peak_cuda_gib':torch.cuda.max_memory_allocated()/2**30})
  if time.time()-began>3.85*3600:save(out/'last.pt',state(step));raise TimeoutError('Oracle4h stage budget')

def oracle_eval(scope):
 from transformers import AutoTokenizer,AutoModelForCausalLM
 from vllm import LLM,SamplingParams
 seed();out=R/'evaluation';out.mkdir(exist_ok=True);path=out/(scope+'.jsonl')
 if path.with_suffix('.summary.json').exists():return
 if path.exists():
  failed=out/'failed_attempts';failed.mkdir(exist_ok=True);path.rename(failed/(path.name+'.'+str(int(time.time()))))
 intervention=scope=='independence_val';mixed=scope.startswith('mixed')
 if intervention or scope=='synthetic_val':rr=read(R/'synthetic_val.json')
 else:
  split='test' if scope.endswith('test') else 'val'
  rr=records(split);ii=base.actual_indices(R/'llm_new',split,rr);rr=[rr[i] for i in ii]
 tok=AutoTokenizer.from_pretrained(base.MODEL,local_files_only=True)
 lm=AutoModelForCausalLM.from_pretrained(base.MODEL,torch_dtype=torch.bfloat16,local_files_only=True).cpu()
 embedding=copy.deepcopy(lm.get_input_embeddings());del lm
 model=load_compact('cuda').eval();label=LabelPrefix().cuda().eval() if mixed else None
 if mixed:
  allrecs=records(split);loc={r['id']:i for i,r in enumerate(allrecs)};old=old_data(split)
 pairs=[]
 if intervention:
  labels=read(R/'protocol.json')['labels']
  for origin in rr[:32]:
   variants=[('base',-1,origin)]
   for field in range(6):
    r=copy.deepcopy(origin);r['id']=origin['id']+':field'+str(field);r['bins'][FIELDS[field]]=(base.bin_values(origin)[field]+1)%3
    variants.append(('field',field,r))
   r=copy.deepcopy(origin);idx=(base.label_id(origin)+1)%len(labels);r.update(id=origin['id']+':label',label=labels[idx],label_id=idx,global_label_id=idx)
   variants.append(('label',-1,r))
   for variant,changed,r in variants:
    for q in base.fixed_questions(r)[:7]:pairs.append((r,q,{'origin_id':origin['id'],'variant':variant,'changed_field':changed}))
 else:
  pairs=[(r,q,{}) for r in rr for q in base.fixed_questions(r)]
 engine=LLM(model=str(base.MODEL),dtype='bfloat16',enable_prompt_embeds=True,max_model_len=2048,max_num_seqs=int(os.environ.get('VSTATS_COMPACT_AR_BATCH','16')),gpu_memory_utilization=float(os.environ.get('VSTATS_COMPACT_AR_MEMORY_UTILIZATION','.35')),enforce_eager=True,disable_log_stats=True,seed=SEED)
 rows=[]
 with path.open('w') as f:
  for off in range(0,len(pairs),int(os.environ.get('VSTATS_COMPACT_AR_BATCH','16'))):
   batch=pairs[off:off+int(os.environ.get('VSTATS_COMPACT_AR_BATCH','16'))];rs=[r for r,q,meta in batch];bb,ll=base.teacher_batch(rs,'cuda')
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
    pf=model(bb,ll)
    if mixed:
     ii=[loc[r['id']] for r in rs];lp=label(old['bank'][ii].cuda(),old['valid'][ii].cuda());pf=torch.cat([lp,pf[:,16:]],1)
    pf=pf.cpu()
   requests=[{'prompt_embeds':base.prompt(tok,embedding,q,p).to(torch.bfloat16)} for (r,q,meta),p in zip(batch,pf)]
   outputs=engine.generate(requests,SamplingParams(temperature=0.,max_tokens=384),use_tqdm=False)
   for (r,q,meta),o in zip(batch,outputs):
    text=o.outputs[0].text;pred,valid,fact=base.parse_generation(text,q)
    row={'id':r['id'],'dataset':r['dataset'],'task':q['task'],'question':q['question'],'expected':q['answer'],'prediction':pred,'generation':text,'valid':valid,'correct':valid and pred==q['answer'],'fact_score':fact,'gold_label':r['label'],'gold_bins':base.bin_values(r),'oracle_uses_gold':True,'scope':scope,**meta}
    rows.append(row);f.write(json.dumps(row,ensure_ascii=False)+'\n')
   f.flush()
   if off%256==0:print('AR',scope,off,len(pairs),flush=True)
 result=base.metrics(rows);result.update(status='complete',scope=scope,tokens=28,oracle_uses_gold=True,oracle_sha256=sha(R/'oracle/best.pt'),source_sha256=sha(__file__))
 if intervention:
  original={(x['origin_id'],x['task']):x for x in rows if x['variant']=='base'}
  unchanged=[];changed=[];label_stable=[];stats_label_stable=[]
  for x in rows:
   if x['variant']=='base':continue
   base_row=original[x['origin_id'],x['task']]
   same=x['valid'] and base_row['valid'] and x['prediction']==base_row['prediction']
   if x['variant']=='field':
    if x['task']=='feature:'+FIELDS[x['changed_field']]:changed.append(x['correct'])
    elif x['task']=='label':label_stable.append(same)
    else:unchanged.append(same)
   elif x['task'].startswith('feature:'):stats_label_stable.append(same)
  result['interventions']={'base_records':32,'non_target_field_output_stability':float(np.mean(unchanged)),'changed_field_target_accuracy':float(np.mean(changed)),'label_stability_under_field_change':float(np.mean(label_stable)),'stat_stability_under_label_change':float(np.mean(stats_label_stable)),'note':'Base combinations are Val-heldout; single-field interventions can enter seen synthetic combinations, diagnostic only.'}
 write(path.with_suffix('.summary.json'),result);write(path.with_suffix('.failures.json'),[x for x in rows if not x['correct']][:40])

def gate():
 syn=read(R/'evaluation/synthetic_val.summary.json');real=read(R/'evaluation/oracle_val.summary.json');ind=read(R/'evaluation/independence_val.summary.json');v=ind['interventions']
 macro=lambda x:float(np.mean([z['feature_macro_bacc'] for z in x['per_dataset'].values()]))
 checks={'structural_factorization':read(R/'audit/structural_smoke.json')['status']=='passed','synthetic_fields_ge80':macro(syn)>=.8,'synthetic_each_field_ge60':min(z['bacc'] for v_ in syn['per_dataset'].values() for task,z in v_['tasks'].items() if task.startswith('feature:'))>=.6,'oracle_label_ge80':float(np.mean([v_['tasks']['label']['bacc'] for v_ in real['per_dataset'].values()]))>=.8,'label_stability_ge80':v['label_stability_under_field_change']>=.8,'real_val_fields_ge80':macro(real)>=.8,'non_target_stability_ge80':v['non_target_field_output_stability']>=.8,'changed_field_accuracy_ge80':v['changed_field_target_accuracy']>=.8,'stat_label_stability_ge80':v['stat_stability_under_label_change']>=.8,'valid_ge80':min(x['format_valid_rate'] for x in [syn,real,ind])>=.8}
 passed=all(checks.values());write(R/'oracle_gate.json',{'status':'passed' if passed else 'gate_failed','passed':passed,'checks':checks,'synthetic_field_bacc':macro(syn),'real_val_field_bacc':macro(real),'interventions':v,'time':time.time(),'no_test_used':True})
 if not passed:raise SystemExit(20)

def freeze_compact():
 files=['oracle/best.pt','llm_new/train/best.pt']
 write(R/'test_freeze.json',{'status':'frozen','time':time.time(),'checkpoints':{f:sha(R/f) for f in files},'stat_expert_sha256':sha(SR/'expert/best.pt'),'matched_readout_sha256':sha(SR/'probe_new/best.pt'),'label_prefix_source_sha256':sha(OROOT/'student/best.pt'),'test_previously_exposed':True,'selection':'Predeclared Val TF best; no Test selection','tokens':28})

def report():
 out=['# 28-token统计接口\n\n分类16＋六字段各2＝28token。固定上一轮统计expert/readout/原数据/seed/标签prefix/Qwen，重新训练紧凑Oracle和连续统计projector；旧112结果只作固定benchmark参考。Test为已查看的重复benchmark，每数据集128窗口。Oracle与mixed Oracle使用gold统计，仅作容量/兼容参考，不能混同真实EEG结果。\n']
 for p in sorted((R/'evaluation').glob('*.summary.json'))+sorted((R/'llm_new/evaluation').glob('*.summary.json')):
  x=read(p);out.append('\n## '+str(p.relative_to(R))+'\n\n|数据集|字段BAcc|标签BAcc|规则|报告exact|facts|格式|\n|---|---:|---:|---:|---:|---:|---:|\n')
  for ds,v in x['per_dataset'].items():
   ts=v['tasks'];out.append('|'+ds+'|'+'|'.join(f'{100*z:.2f}' for z in [v['feature_macro_bacc'],ts.get('label',{}).get('bacc',0),v['rule_accuracy'],v['report_exact_accuracy'],ts.get('report',{}).get('fact_accuracy',0),float(np.mean([z['format_valid_rate'] for z in ts.values()]))])+'|\n')
  out.append('\n逐字段BAcc/MF1/混淆、JSONL和失败例见同名产物。')
 out.append('\n独立性以输入字段分解和单字段替换读取稳定性检查；不强制同字段两个向量正交，不声称LLM注意力隔离。未做ICL/DPO，不作临床或保留原生推理声明。\n')
 (R/'final_report.md').write_text(''.join(out));write(R/'report_complete.json',{'status':'complete','time':time.time(),'source_sha256':sha(__file__)})

def pipeline():
 root=R/'worker_main';root.mkdir(exist_ok=True);lock=(root/'pipeline.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 first=root/'first_launch.json'
 if not first.exists():write(first,{'started':time.time()})
 os.environ['VSTATS_STAT_EXPERT_DEADLINE']=str(read(first)['started']+12*3600)
 write(root/'launch.json',{'pid':os.getpid(),'started':read(first)['started'],'source_sha256':sha(__file__)})
 try:
  stage('main','oracle_smoke',['oracle-train','--smoke'],'oracle_smoke/smoke.json')
  stage('main','oracle_training',['oracle-train'],'oracle/summary.json')
  for scope in ['synthetic_val','oracle_val','mixed_val','independence_val']:stage('main',scope,['oracle-eval','--scope',scope],'evaluation/'+scope+'.summary.json')
  stage('main','oracle_gate',['gate'],'oracle_gate.json')
  stage('main','llm_smoke',['llm-train','--arm','new','--smoke'],'llm_new/smoke/smoke.json')
  stage('main','llm_training',['llm-train','--arm','new'],'llm_new/train/summary.json')
  for mode in ['continuous','predicted_code']:stage('main','val_'+mode,['llm-eval','--arm','new','--split','val','--mode',mode],'llm_new/evaluation/val_'+mode+'.summary.json')
  stage('main','freeze',['freeze'],'test_freeze.json')
  for scope in ['oracle_test','mixed_test']:stage('main',scope,['oracle-eval','--scope',scope],'evaluation/'+scope+'.summary.json')
  for mode in ['continuous','predicted_code']:stage('main','test_'+mode,['llm-eval','--arm','new','--split','test','--mode',mode],'llm_new/evaluation/test_'+mode+'.summary.json')
  report();write(root/'status.json',{'state':'complete','time':time.time()})
 except Exception:
  failed=(R/'oracle_gate.json').exists() and not read(R/'oracle_gate.json')['passed']
  write(root/'status.json',{'state':'gate_failed' if failed else 'failed','time':time.time(),'traceback':traceback.format_exc()});raise

def main():
 a=argparse.ArgumentParser();a.add_argument('command');a.add_argument('--smoke',action='store_true');a.add_argument('--arm',default='new');a.add_argument('--split',default='val');a.add_argument('--mode',default='continuous');a.add_argument('--scope',default='oracle_val');x=a.parse_args()
 if x.command=='prepare':prepare()
 elif x.command=='pipeline':pipeline()
 elif x.command=='oracle-train':oracle_train(x.smoke)
 elif x.command=='oracle-eval':oracle_eval(x.scope)
 elif x.command=='gate':gate()
 elif x.command=='llm-train':llm_train('new',x.smoke)
 elif x.command=='llm-eval':llm_eval('new',x.split,x.mode)
 elif x.command=='freeze':freeze_compact()
 elif x.command=='report':report()
 else:raise ValueError(x.command)
if __name__=='__main__':
 if os.environ.get('VSTATS_COMPACT_ALLOCATOR_BUDGET_MIB'):
  budget=float(os.environ['VSTATS_COMPACT_ALLOCATOR_BUDGET_MIB'])
  total=float(os.environ['VSTATS_COMPACT_GPU_TOTAL_MIB'])
  torch.cuda.set_per_process_memory_fraction(min(.85,budget/total),0)
 main()
