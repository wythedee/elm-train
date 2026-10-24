#!/usr/bin/env python3
"""Jointly align label16+statistics12 to the fixed new28 Oracle; heldout prompt robustness."""

if __name__ == "__main__":
    raise SystemExit("Reference snapshot only. Use tools/workbench.py for inspection; configure a fresh run before training.")

from pathlib import Path
import argparse,ast,collections,copy,csv,fcntl,hashlib,json,math,os,random,shutil,signal,subprocess,sys,time,traceback
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import stat_compact28_1015_v3 as ref
base=ref.base;P=ref.P;CR=ref.R;SR=ref.SR;DATA=ref.DATA;OROOT=ref.OROOT
R=P/'experiments/mainline_vrl/runs/stat_joint28_20261015_s1014'
FIELDS=base.FIELDS;VALUES=base.VALUES;SEED=1014;ARPY=ref.ARPY;TRAINPY=ref.TRAINPY;ROLE='main'
read=ref.read;write=ref.write;sha=ref.sha;seed=ref.seed;complete=ref.complete
def guard(extra=0):
 free=shutil.disk_usage(R).free
 if free<12*2**30+extra:raise OSError(f'12GiB storage guard free={free} extra={extra}')
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 tensors=x.get('state_dict',{})
 approx=sum(v.numel()*v.element_size() for v in tensors.values() if isinstance(v,torch.Tensor))
 guard(max(approx*4,64*2**20));base.save_checkpoint(p,x)
def rng_state(rng):
 return {**ref.rng_state(rng),'python_global_rng':random.getstate()}
def restore(ck,rng):
 ref.restore(ck,rng);random.setstate(ck['python_global_rng'])
def records(split):
 if split=='test':assert (R/'test_freeze.json').exists(),'New-run freeze required before Test'
 return read(DATA/(split+'_records.json'))
def data(split):
 if split=='test':assert (R/'test_freeze.json').exists()
 return torch.load(DATA/(split+'.pt'),map_location='cpu',weights_only=False)
def banks(split):
 if split=='test':assert (R/'test_freeze.json').exists()
 return torch.load(SR/'banks'/(split+'.pt'),map_location='cpu',weights_only=False)
class JointPrefix(nn.Module):
 def __init__(self):
  super().__init__();self.label=ref.LabelPrefix();self.statistics=ref.StatsHead()
  self.statistics.load_state_dict(torch.load(CR/'llm_init.pt',map_location='cpu',weights_only=False)['state_dict'])
  for p in self.parameters():p.requires_grad_(True)
 def forward(self,label_bank,label_valid,stat_bank,stat_valid):
  return torch.cat([self.label(label_bank,label_valid),self.statistics(stat_bank,stat_valid)],1)
def initialized(device='cuda'):
 m=JointPrefix().to(device)
 m.load_state_dict(torch.load(R/'joint_init.pt',map_location='cpu',weights_only=False)['state_dict'])
 return m
def train(smoke=False):
 seed();out=R/('smoke' if smoke else 'train');out.mkdir(exist_ok=True)
 tok,lm,embedding=base.hf_decoder();oracle=ref.load_compact('cuda').eval()
 for p in oracle.parameters():p.requires_grad_(False)
 net=initialized();tr=data('train');va=data('val');bt=banks('train');bv=banks('val');rr=records('train');vr=records('val')
 opt=torch.optim.AdamW(net.parameters(),lr=1e-5,weight_decay=.0001);rng=random.Random(SEED);draw=ref.pick(rr,rng,8)
 vi=base.actual_indices(R/'evaluation','val',vr);vpairs=[(i,base.fixed_questions(vr[i])[j%9]) for j,i in enumerate(vi)]
 steps=30 if smoke else 6000;micro=int(os.environ.get('VSTATS_COMPACT_MICRO_BATCH','8'));assert micro in [1,2,4,8]
 start=0;best=math.inf;history=[];began=time.time()
 if (out/'last.pt').exists():
  ck=torch.load(out/'last.pt',map_location='cpu',weights_only=False);net.load_state_dict(ck['state_dict']);opt.load_state_dict(ck['optimizer']);restore(ck,rng)
  start=ck['step'];best=ck['best'];history=ck['history']
 initial={n:next(p for p in module.parameters() if p.requires_grad).detach().clone() for n,module in [('label',net.label),('statistics',net.statistics)]}
 def prefix(ii,st,old):
  return net(old['bank'][ii].cuda(),old['valid'][ii].cuda(),st['bank'][ii].cuda(),st['valid'][ii].cuda())
 def state(step):
  return {'state_dict':{k:v.detach().cpu() for k,v in net.state_dict().items()},'optimizer':opt.state_dict(),'step':step,'best':best,'history':history,**rng_state(rng),
   'source_sha256':sha(__file__),'oracle_sha256':sha(CR/'oracle/best.pt'),'effective_batch':8,'micro_batch':micro}
 for step in range(start+1,steps+1):
  net.train();opt.zero_grad(set_to_none=True);ii=draw();rs=[rr[i] for i in ii];qs=[base.question(r,rng.choice(base.TASKS),rng) for r in rs]
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
   bins_,labs=base.teacher_batch(rs,'cuda');target=oracle(bins_,labs)
  denominator=target.float().square().mean().clamp_min(1e-8);loss_value=ce_value=align_value=0.;branch_grad={}
  for off in range(0,8,micro):
   jj=ii[off:off+micro];qq=qs[off:off+micro];tt=target[off:off+micro]
   with torch.autocast('cuda',dtype=torch.bfloat16):
    pf=prefix(jj,bt,tr);assert pf.shape[1:]==(28,2048)
    align=.1*((pf.float()-tt.float()).square().mean()/denominator+1-F.cosine_similarity(pf.float().flatten(1),tt.float().flatten(1)).mean())
    ce=base.sft_loss(lm,tok,embedding,qq,pf);loss=ce+align
   assert torch.isfinite(loss);(loss*(len(qq)/8)).backward()
   loss_value+=float(loss)*len(qq)/8;ce_value+=float(ce)*len(qq)/8;align_value+=float(align)*len(qq)/8
  for name,module in [('label',net.label),('statistics',net.statistics)]:
   grads=[p.grad for p in module.parameters() if p.grad is not None]
   assert grads and all(torch.isfinite(g).all() for g in grads),name
   branch_grad[name]=float(torch.sqrt(sum(g.float().square().sum() for g in grads)))
   assert branch_grad[name]>0,name
  grad=torch.nn.utils.clip_grad_norm_(net.parameters(),1.);assert torch.isfinite(grad) and grad>0
  assert all(p.grad is None for p in lm.parameters()) and all(p.grad is None for p in oracle.parameters())
  opt.step()
  if step%10==0:print(json.dumps({'step':step,'loss':loss_value,'ce':ce_value,'alignment_all28':align_value,'grad':float(grad),'branch_grad':branch_grad,'seconds':time.time()-began}),flush=True)
  if smoke and step==30:
   changed={n:not torch.equal(initial[n],next(p for p in module.parameters() if p.requires_grad).detach()) for n,module in [('label',net.label),('statistics',net.statistics)]}
   assert all(changed.values()),changed
   save(out/'last.pt',state(step));write(out/'smoke.json',{'status':'passed','steps':30,'changed_branches':changed,'finite_gradients':True,'label_branch_trainable':True,'statistics_branch_trainable':True,'all28_alignment':True,'decoder_frozen':True,'oracle_frozen':True,'tokens':28,'effective_batch':8,'micro_batch':micro,'peak_cuda_gib':torch.cuda.max_memory_allocated()/2**30})
  if not smoke and (step%500==0 or step==steps):
   net.eval();losses=[]
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
    for off in range(0,len(vpairs),micro):
     group=vpairs[off:off+micro];jj=[i for i,q in group];losses.append(float(base.sft_loss(lm,tok,embedding,[q for i,q in group],prefix(jj,bv,va))))
   val=float(np.mean(losses));improved=val<best;best=min(best,val);history.append({'step':step,'val_tf_loss':val,'seconds':time.time()-began})
   ck=state(step);save(out/'last.pt',ck)
   if improved:save(out/'best.pt',ck)
   write(out/'summary.json',{'status':'complete' if step==steps else 'running','step':step,'best_val_tf':best,'history':history,'tokens':28,'trainable_parameters':sum(p.numel() for p in net.parameters()),'trainable_label_parameters':sum(p.numel() for p in net.label.parameters()),'trainable_stat_parameters':sum(p.numel() for p in net.statistics.parameters()),'label_branch_trainable':True,'statistics_branch_trainable':True,'all28_alignment':True,'lm_frozen':True,'oracle_frozen':True,'peak_cuda_gib':torch.cuda.max_memory_allocated()/2**30,'effective_batch':8,'micro_batch':micro})
   print('VALIDATION',history[-1],flush=True)
  if time.time()-began>3.85*3600:save(out/'last.pt',state(step));raise TimeoutError('Joint4h stage budget')
def robust_questions(r):
 bins_=dict(zip(FIELDS,[VALUES[i] for i in base.bin_values(r)]));seed_=int(hashlib.sha256((r['id']+':robust1015').encode()).hexdigest()[:12],16);rng=random.Random(seed_)
 result=[]
 def add(family,task,text,answer,specs):
  result.append({'family':family,'task':task,'question':text,'answer':answer,'leaf_specs':specs,'robust':True})
 def spec(path,kind):return {'path':path,'kind':kind}
 for family in ['paraphrase','context']:
  for field in ['label']+FIELDS:
   if field=='label':
    text='Identify the observed class carried by the EEG tokens. Emit exactly one JSON object with the key label and its class string.'
    answer={'label':r['label']};task='label';leaf=[spec(['label'],'label')]
   else:
    text='Decode only the '+field+' measurement from the EEG tokens. Emit a single JSON object whose only key is '+field+'; use low, mid or high for its level.'
    answer={field:bins_[field]};task='feature:'+field;leaf=[spec([field],'bin')]
   if family=='context':
    background='Administrative background (contains no EEG measurements):\n'+'\n'.join('Workflow note '+str(i)+': '+['The record identifier is for filing.','The viewer may export structured observations.','Scheduling notes do not describe the signal.','The technician requested a compact response.'][i%4] for i in range(24))
    text=background+'\nUse the EEG tokens as evidence for the measurement; the administrative notes do not alter the encoded facts.\n'+text
   add(family,task,text,answer,leaf)
 chosen=rng.sample(FIELDS,3)
 add('selective_report','report_subset','Create a partial EEG report. Return only JSON with label and statistics. The statistics object must contain only these requested fields: '+', '.join(chosen)+'. Each level is low, mid or high. Do not include the other measurements.',
  {'label':r['label'],'statistics':{f:bins_[f] for f in chosen}},[spec(['label'],'label')]+[spec(['statistics',f],'bin') for f in chosen])
 ordered=rng.sample(FIELDS,len(FIELDS))
 add('nested_report','report_nested','Serialize the encoded EEG facts in this exact structure: {"record":{"class":"CLASS","measurements":[{"name":"FIELD","level":"LEVEL"}]}}. Include all six measurements in this order: '+', '.join(ordered)+'. Each measurement item has only name and level; levels are low, mid or high. Return only JSON.',
  {'record':{'class':r['label'],'measurements':[{'name':f,'level':bins_[f]} for f in ordered]}},
  [spec(['record','class'],'label')]+[spec(['record','measurements',i,'level'],'bin') for i in range(6)])
 fs=rng.sample(FIELDS,3);op=rng.choice(['AND','OR']);desired=bool(rng.randrange(2))
 while True:
  vv=[rng.choice(VALUES) for _ in fs];conds=[bins_[f]==v for f,v in zip(fs,vv)];truth=all(conds) if op=='AND' else any(conds)
  if truth==desired:break
 text='Evaluate this three-measurement rule using the EEG tokens: '+(' '+op+' ').join('('+f+' is '+v+')' for f,v in zip(fs,vv))+'. Return only {"decision":"review"} when the rule is true, otherwise {"decision":"routine"}.'
 add('three_field_rule','rule_three',text,{'decision':'review' if truth else 'routine'},[spec(['decision'],'decision')])
 fs=rng.sample(FIELDS,3);followup=bool(rng.randrange(2));desired=bool(rng.randrange(2))
 while True:
  vv=[rng.choice(VALUES) for _ in fs];cond=[bins_[f]==v for f,v in zip(fs,vv)];truth=(followup and (cond[0] or cond[1])) or cond[2]
  if truth==desired:break
 text='External workflow context: follow_up_required='+str(followup).lower()+'. This is a policy input, not an EEG measurement. Apply the policy (follow_up_required AND ('+fs[0]+' is '+vv[0]+' OR '+fs[1]+' is '+vv[1]+')) OR ('+fs[2]+' is '+vv[2]+'). Read the measurements from EEG tokens. Return only {"decision":"review"} if true, otherwise {"decision":"routine"}.'
 add('text_conditioned_rule','rule_context',text,{'decision':'review' if truth else 'routine'},[spec(['decision'],'decision')])
 assert len(result)==18
 return result
def at(value,path):
 for k in path:
  if isinstance(k,int):
   if not isinstance(value,list) or k>=len(value):return None
  elif not isinstance(value,dict) or k not in value:return None
  value=value[k]
 return value
def structure(value,expected):
 if isinstance(expected,dict):return isinstance(value,dict) and set(value)==set(expected) and all(structure(value[k],v) for k,v in expected.items())
 if isinstance(expected,list):return isinstance(value,list) and len(value)==len(expected) and all(structure(a,b) for a,b in zip(value,expected))
 return isinstance(value,str)
def parse(text,q):
 if not q.get('robust'):return base.parse_generation(text,q)
 try:value=json.loads(text.strip())
 except (ValueError,TypeError):return None,False,0.
 valid=structure(value,q['answer']);correct_leaves=[]
 for leaf in q['leaf_specs']:
  pred=at(value,leaf['path']);target=at(q['answer'],leaf['path'])
  valid=valid and (pred in VALUES if leaf['kind']=='bin' else pred in ['review','routine'] if leaf['kind']=='decision' else isinstance(pred,str))
  correct_leaves.append(pred==target)
 # Field names/order are part of the output schema, rather than extra facts.
 if q['family']=='nested_report':
  valid=valid and [at(value,['record','measurements',i,'name']) for i in range(6)]==[at(q['answer'],['record','measurements',i,'name']) for i in range(6)]
 return value,bool(valid),float(np.mean(correct_leaves))
def robust_metrics(rows):
 out={}
 for family in sorted({r['family'] for r in rows}):
  group=[r for r in rows if r['family']==family];per={}
  for ds in sorted({r['dataset'] for r in group}):
   rr=[r for r in group if r['dataset']==ds]
   v={'n':len(rr),'exact_accuracy':float(np.mean([r['correct'] for r in rr])),'format_valid_rate':float(np.mean([r['valid'] for r in rr])),'fact_accuracy':float(np.mean([r['fact_score'] for r in rr]))}
   if family in ['paraphrase','context']:
    retrieval=base.metrics(rr)['per_dataset'][ds]
    v['retrieval_tasks']=retrieval['tasks'];v['field_macro_bacc']=retrieval['feature_macro_bacc'];v['label_bacc']=retrieval['tasks']['label']['bacc']
   per[ds]=v
  value={'queries':len(group),'exact_accuracy':float(np.mean([v['exact_accuracy'] for v in per.values()])),'format_valid_rate':float(np.mean([v['format_valid_rate'] for v in per.values()])),'fact_accuracy':float(np.mean([v['fact_accuracy'] for v in per.values()])),'per_dataset':per}
  if family in ['paraphrase','context']:
   value['field_macro_bacc']=float(np.mean([v['field_macro_bacc'] for v in per.values()]));value['label_macro_bacc']=float(np.mean([v['label_bacc'] for v in per.values()]))
  out[family]=value
 return {'status':'complete','queries':len(rows),'per_family':out,'no_hidden_composite':True,'scope':'Prompt/context/schema transfer diagnostic; all questions held out from Oracle and student training.'}
def eval_ar(split,mode,suite):
 from transformers import AutoTokenizer,AutoModelForCausalLM
 from vllm import LLM,SamplingParams
 seed();out=R/'evaluation';path=out/(split+'_'+suite+'_'+mode+'.jsonl')
 if complete(path.with_suffix('.summary.json')):return
 if path.exists():
  fail=out/'failed_attempts';fail.mkdir(exist_ok=True);path.rename(fail/(path.name+'.'+str(int(time.time()))))
 rr=records(split);indices=base.actual_indices(out,split,rr)
 if suite=='robust':
  cap=4 if split=='val' else 8;groups=collections.defaultdict(list)
  for i in indices:groups[rr[i]['dataset']].append(i)
  indices=[i for ds in sorted(groups) for i in groups[ds][:cap]]
 questions=robust_questions if suite=='robust' else base.fixed_questions
 pairs=[(i,q) for i in indices for q in questions(rr[i])]
 tok=AutoTokenizer.from_pretrained(base.MODEL,local_files_only=True);lm=AutoModelForCausalLM.from_pretrained(base.MODEL,torch_dtype=torch.bfloat16,local_files_only=True).cpu();embedding=copy.deepcopy(lm.get_input_embeddings());del lm
 old=data(split) if mode!='oracle' else None;st=banks(split) if mode!='oracle' else None
 if mode=='joint':
  model=initialized().eval();model.load_state_dict(torch.load(R/'train/best.pt',map_location='cpu',weights_only=False)['state_dict'])
 elif mode=='legacy':
  label=ref.LabelPrefix().cuda().eval();model=ref.StatsHead().cuda().eval();model.load_state_dict(torch.load(CR/'llm_new/train/best.pt',map_location='cpu',weights_only=False)['state_dict'])
 else:model=ref.load_compact('cuda').eval()
 arbatch=int(os.environ.get('VSTATS_COMPACT_AR_BATCH','16'))
 engine=LLM(model=str(base.MODEL),dtype='bfloat16',enable_prompt_embeds=True,max_model_len=2048,max_num_seqs=arbatch,gpu_memory_utilization=float(os.environ.get('VSTATS_COMPACT_AR_MEMORY_UTILIZATION','.35')),enforce_eager=True,disable_log_stats=True,seed=SEED)
 rows=[];max_prompt=0;began=time.time()
 with path.open('w') as f:
  for off in range(0,len(pairs),arbatch):
   group=pairs[off:off+arbatch];ii=[i for i,q in group];rs=[rr[i] for i in ii]
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
    if mode=='oracle':
     bb,ll=base.teacher_batch(rs,'cuda');pf=model(bb,ll)
    elif mode=='joint':pf=model(old['bank'][ii].cuda(),old['valid'][ii].cuda(),st['bank'][ii].cuda(),st['valid'][ii].cuda())
    else:pf=torch.cat([label(old['bank'][ii].cuda(),old['valid'][ii].cuda()),model(st['bank'][ii].cuda(),st['valid'][ii].cuda())],1)
    pf=pf.cpu()
   requests=[{'prompt_embeds':base.prompt(tok,embedding,q,p).to(torch.bfloat16)} for (i,q),p in zip(group,pf)]
   max_prompt=max(max_prompt,max(len(v['prompt_embeds']) for v in requests));assert max_prompt+384<=2048
   results=engine.generate(requests,SamplingParams(temperature=0.,max_tokens=384),use_tqdm=False)
   for (i,q),o in zip(group,results):
    r=rr[i];text=o.outputs[0].text;pred,valid,fact=parse(text,q)
    row={'id':r['id'],'dataset':r['dataset'],'task':q['task'],'family':q.get('family','base'),'mode':mode,'split':split,'suite':suite,'question':q['question'],'expected':q['answer'],'prediction':pred,'generation':text,'valid':valid,'correct':valid and pred==q['answer'],'fact_score':fact,'gold_label':r['label'],'gold_bins':base.bin_values(r),'oracle_uses_gold':mode=='oracle'}
    rows.append(row);f.write(json.dumps(row,ensure_ascii=False)+'\n')
   f.flush()
   if off%256==0:print('AR',split,suite,mode,off,len(pairs),flush=True)
 result=robust_metrics(rows) if suite=='robust' else base.metrics(rows)
 result.update(status='complete',mode=mode,split=split,suite=suite,tokens=28,source_sha256=sha(__file__),oracle_sha256=sha(CR/'oracle/best.pt'),projector_sha256=sha(R/'train/best.pt') if mode=='joint' else sha(CR/'llm_new/train/best.pt') if mode=='legacy' else None,oracle_uses_gold=mode=='oracle',deterministic_free_generation=True,temperature=0.,max_new_tokens=384,max_prompt_tokens=max_prompt,seconds=time.time()-began)
 write(path.with_suffix('.summary.json'),result);write(path.with_suffix('.failures.json'),[r for r in rows if not r['correct']][:40])
def verify_sources():
 c=read(R/'protocol.json');assert sha(c['source'])==c['source_sha256']
 for path,h in c['dependencies'].items():assert sha(P/path)==h,path
 for path,h in c['fixed_models'].items():assert sha(path)==h,path
 return True
def prepare():
 seed();verify_sources();net=JointPrefix()
 save(R/'joint_init.pt',{'state_dict':net.state_dict(),'seed':SEED,'tokens':28,'label_initialization':'Same original realEEG prefix as frozen28 control, now fully trainable.','statistics_initialization':str(CR/'llm_init.pt')})
 vr=records('val');vi=base.actual_indices(R/'evaluation','val',vr)
 assert vi==read(CR/'llm_new/val_indices.json')
 # CPU checks: shapes, trainable branches, target/schema generation, no real Test access.
 fake=torch.randn(2,1,150,50);valid=torch.ones(2,150)
 with torch.no_grad():pf=net(fake,valid,fake,valid)
 assert pf.shape==(2,28,2048) and torch.isfinite(pf).all()
 assert all(p.requires_grad for p in net.label.parameters()) and all(p.requires_grad for p in net.statistics.parameters())
 token_counts=[];families=collections.Counter()
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(base.MODEL,local_files_only=True)
 before,after=base.wrapper(tok)
 for i in vi:
  for q in robust_questions(vr[i]):
   pred,ok,fact=parse(json.dumps(q['answer']),q);assert ok and fact==1 and pred==q['answer']
   _,ok2,_=parse('{}',q);assert not ok2
   token_counts.append(len(base.token_ids(tok,before+'EEG evidence sample 1:\n'))+28+len(base.token_ids(tok,'\n'+q['question']+after)));families[q['family']]+=1
 assert max(token_counts)+384<=2048
 write(R/'audit/cpu_checks.json',{'status':'passed','shape':[2,28,2048],'trainable_label_parameters':sum(p.numel() for p in net.label.parameters()),'trainable_stat_parameters':sum(p.numel() for p in net.statistics.parameters()),'label_all_trainable':True,'statistics_all_trainable':True,'robust_schemas_gold_roundtrip':True,'empty_wrong_outputs_rejected':True,'val_questions_checked':sum(families.values()),'families':dict(families),'max_val_prompt_tokens':max(token_counts),'context_limit':2048,'test_read':False,'not_gpu_smoke':True})
 write(R/'prepared.json',{'status':'complete','time':time.time(),'init_sha256':sha(R/'joint_init.pt')})
def freeze():
 assert complete(R/'evaluation/val_base_joint.summary.json') and complete(R/'evaluation/val_robust_joint.summary.json')
 verify_sources();write(R/'test_freeze.json',{'status':'frozen','time':time.time(),'joint_checkpoint_sha256':sha(R/'train/best.pt'),'fixed_models':read(R/'protocol.json')['fixed_models'],'source_sha256':sha(__file__),'test_previously_exposed':True,'selection':'Predeclared Val TF best; no robustness metric or Test selection.','tokens':28})
def wait_for(path):
 deadline=float(os.environ['VSTATS_STAT_EXPERT_DEADLINE'])
 while not complete(path):
  if time.time()>=deadline:raise TimeoutError('Original worker12h waiting '+str(path))
  other=R/('worker_control' if ROLE=='main' else 'worker_main')/'status.json'
  if other.exists() and read(other).get('state')=='failed':raise RuntimeError('Other role failed; stop rather than wait indefinitely')
  write(R/('worker_'+ROLE)/'status.json',{'state':'waiting_dependency','path':str(path),'pipeline_pid':os.getpid(),'time':time.time()});time.sleep(10)
def report():
 main=read(R/'evaluation/test_base_joint.summary.json')
 out=['# 联合对齐28token与Oracle鲁棒性\n','分类16与统计12两路projector均训练，对齐固定新Oracle全部28token；EEG experts和LLM冻结。原16数据集切分/ID/六字段/Train阈值/seed1014不变。Test是重复benchmark，每数据集128窗口，非全量原始Test。\n',
 '## 基础任务Test\n','|路径|六字段BAcc %|标签BAcc %|规则 %|报告全对 %|报告facts %|格式 %|\n|---|---:|---:|---:|---:|---:|---:|\n']
 for name,p in [('旧分类前缀冻结28',CR/'llm_new/evaluation/test_continuous.summary.json'),('联合28',R/'evaluation/test_base_joint.summary.json')]:
  if not p.exists():continue
  x=read(p);v=list(x['per_dataset'].values());avg=lambda f:float(np.mean([f(z) for z in v]))
  vals=[avg(lambda z:z['feature_macro_bacc']),avg(lambda z:z['tasks']['label']['bacc']),avg(lambda z:z['rule_accuracy']),avg(lambda z:z['report_exact_accuracy']),avg(lambda z:z['tasks']['report']['fact_accuracy']),x['format_valid_rate']]
  out.append('|'+name+'|'+'|'.join(f'{100*z:.2f}' for z in vals)+'|\n')
 out+=['\n## 未训练过的问法/背景/输出结构\n','Oracle控制含gold事实，joint和legacy仅EEG。Robust Val64窗口、Test128窗口，各18题/窗口；每数据集4/8窗口，分项小样本诊断。\n',
 '|模式|变体|字段BAcc %|标签BAcc %|输出全对 %|事实 %|格式 %|\n|---|---|---:|---:|---:|---:|---:|\n']
 for mode in ['oracle','legacy','joint']:
  x=read(R/f'evaluation/test_robust_{mode}.summary.json')
  for fam,v in x['per_family'].items():
   out.append('|'+mode+'|'+fam+'|'+('|'.join(f'{100*v[k]:.2f}' if k in v else 'N/A' for k in ['field_macro_bacc','label_macro_bacc','exact_accuracy','fact_accuracy','format_valid_rate']))+'|\n')
 out+=['\n每个summary与JSONL保留逐数据集/字段、格式、事实及失败例。鲁棒性测评未参与训练或选模。联合分支按同256窗口Val TF loss选best，先完成Val AR再冻结Test。\n',
 'Oracle已学习接口的跨问法表现不等同保留原生文本推理；机器事实报告不是医生报告。原16域均有Train，不作新数据集OOD声明。程序统计文本为约定100%参考，未实测LLM文本基线。\n']
 (R/'final_report.md').write_text(''.join(out));write(R/'report_complete.json',{'status':'complete','time':time.time(),'source_sha256':sha(__file__),'needs_final_integrity_audit_and_details':True})
def pipeline(role):
 global ROLE
 ROLE=role;root=R/('worker_'+role);lock=(root/'pipeline.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 first=root/'first_launch.json'
 if not first.exists():write(first,{'started':time.time()})
 os.environ['VSTATS_STAT_EXPERT_DEADLINE']=str(read(first)['started']+12*3600)
 write(root/'launch.json',{'pid':os.getpid(),'started':read(first)['started'],'source_sha256':sha(__file__),'role':role})
 try:
  verify_sources()
  if role=='main':
   stage(role,'joint_smoke',['llm-train','--smoke'],'smoke/smoke.json')
   stage(role,'joint_training',['llm-train'],'train/summary.json')
   for suite in ['base','robust']:stage(role,'val_'+suite+'_joint',['llm-eval','--split','val','--suite',suite,'--mode','joint'],'evaluation/val_'+suite+'_joint.summary.json')
   stage(role,'test_freeze',['freeze'],'test_freeze.json')
   for suite in ['base','robust']:stage(role,'test_'+suite+'_joint',['llm-eval','--split','test','--suite',suite,'--mode','joint'],'evaluation/test_'+suite+'_joint.summary.json')
   wait_for(R/'worker_control/complete.json');report()
  else:
   for mode in ['oracle','legacy']:stage(role,'val_robust_'+mode,['llm-eval','--split','val','--suite','robust','--mode',mode],'evaluation/val_robust_'+mode+'.summary.json')
   wait_for(R/'test_freeze.json')
   for mode in ['oracle','legacy']:stage(role,'test_robust_'+mode,['llm-eval','--split','test','--suite','robust','--mode',mode],'evaluation/test_robust_'+mode+'.summary.json')
  write(root/'complete.json',{'status':'complete','time':time.time(),'role':role});write(root/'status.json',{'state':'complete','time':time.time(),'role':role})
 except Exception:
  write(root/'status.json',{'state':'failed','time':time.time(),'role':role,'traceback':traceback.format_exc()});raise
def stage(role,name,args,artifact=None):
 root=R/('worker_'+ROLE);root.mkdir(exist_ok=True);log=root/(name+'.log')
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
   log_offset=log.stat().st_size if log.exists() else 0
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
   with log.open('r') as latest:
    latest.seek(log_offset);tail=latest.read()[-16000:].lower()
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

def main():
 a=argparse.ArgumentParser();a.add_argument('command');a.add_argument('--role',choices=['main','control'],default='main');a.add_argument('--smoke',action='store_true');a.add_argument('--split',choices=['val','test'],default='val');a.add_argument('--mode',choices=['joint','oracle','legacy'],default='joint');a.add_argument('--suite',choices=['base','robust'],default='base');x=a.parse_args()
 if x.command=='prepare':prepare()
 elif x.command=='pipeline':pipeline(x.role)
 elif x.command=='llm-train':train(x.smoke)
 elif x.command=='llm-eval':eval_ar(x.split,x.mode,x.suite)
 elif x.command=='freeze':freeze()
 elif x.command=='report':report()
 else:raise ValueError(x.command)
if __name__=='__main__':
 if os.environ.get('VSTATS_COMPACT_ALLOCATOR_BUDGET_MIB'):
  torch.cuda.set_per_process_memory_fraction(min(.85,float(os.environ['VSTATS_COMPACT_ALLOCATOR_BUDGET_MIB'])/float(os.environ['VSTATS_COMPACT_GPU_TOTAL_MIB'])),0)
 main()
