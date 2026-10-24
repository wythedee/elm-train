#!/usr/bin/env python3
"""Three-round teacher-to-student Oracle distillation; isolated, bounded, no EEG updates."""

if __name__ == "__main__":
    raise SystemExit("Reference snapshot only. Use tools/workbench.py for inspection; configure a fresh run before training.")

from pathlib import Path
import argparse,collections,copy,csv,fcntl,hashlib,itertools,json,math,os,random,shutil,signal,subprocess,sys,time,traceback
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import oracle_curriculum28_tasks_1016_v3 as tasks
old=tasks.old;ref=old.ref;base=old.base;P=old.P;R=tasks.R;DATA=old.DATA
SEED=1014;read=ref.read;write=ref.write;sha=ref.sha;complete=ref.complete
ARPY=ref.ARPY;TRAINPY=ref.TRAINPY
def guard(extra=0):
 if shutil.disk_usage(R).free<12*2**30+extra:raise OSError('12GiB disk and atomic-write reserve reached')
def save(p,x):
 guard(128*2**20);Path(p).parent.mkdir(parents=True,exist_ok=True);base.save_checkpoint(Path(p),x)
def config():return read(R/'protocol.json')
def verify():
 d=config();assert sha(__file__)==d['source_sha256']
 for p,h in d['dependencies'].items():assert sha(P/p)==h,p
 for p,h in d['models'].items():assert sha(p)==h,p
 for split in ['train','val']:assert sha(DATA/(split+'_records.json'))==d['data']['record_file_sha256'][split],split
def initialize(device):
 net=ref.load_compact(device);net.load_state_dict(torch.load(R/'oracle_init.pt',map_location='cpu',weights_only=False)['state_dict']);return net
def evaluation_plan(split):
 path=R/(split+'_plan.json')
 if path.exists():return read(path)
 assert split=='val' or (R/'test_freeze.json').exists()
 rows=read(DATA/(split+'_records.json'));indices=read(ref.R/'llm_new'/(split+'_indices.json'))
 out=[]
 for i in indices:
  for j,q in enumerate(base.fixed_questions(rows[i])):
   out.append({'id':rows[i]['id']+':base:'+str(j),'records':[rows[i]],'q':q,'kind':'base','scope':'original_base','dataset':rows[i]['dataset'],'split':split})
 groups=read(old.PR/'statistics'/(split+'_plan.json'))['groups']
 if split=='val':groups=[g for g in groups if int(g['group_id'].rsplit(':',1)[1])<2]
 for gi,g in enumerate(groups):
  for j,fam in enumerate(tasks.FAMILIES):
   rs=g['records'] if j>=4 else g['records'][:1]
   spec=tasks.specification(rs,fam,random.Random(SEED+gi*211+j))
   paired={}
   for k,suite in enumerate(['seen_wording','unseen_wording','long_context','role_context']):
    rng=random.Random(SEED+gi*371+j*19+k)
    e=tasks.make(rs,copy.deepcopy(spec),rng,split,suite,g['group_id']+':'+suite+':'+fam,wording='train' if suite=='seen_wording' else split)
    def notes(text):return [line for line in text.splitlines() if line.startswith('Filing note ')]
    def set_notes(text,lines):
     body=[line for line in text.splitlines() if not line.startswith('Filing note ')]
     return '\n'.join(body[:1]+lines+body[1:])
    if suite=='unseen_wording':
     be=paired['seen_wording'];e['question']=set_notes(e['question'],notes(be['question']))
     e.update(system=be['system'],history=[],layout=be['layout'],context_lines=be['context_lines'])
    elif suite=='long_context':
     be=paired['unseen_wording'];e['question']=set_notes(be['question'],notes(e['question']))
     e.update(system=be['system'],history=[],layout=be['layout'])
    elif suite=='role_context':
     be=paired['unseen_wording'];e.update(question=be['question'],layout=be['layout'],context_lines=be['context_lines'])
    e['pair_id']=g['group_id']+':'+fam;paired[suite]=e;out.append(e)
  rng=random.Random(SEED+gi*151)
  rs=g['records'][:1];spec=tasks.specification(rs,'rule',rng,logic=split+'_holdout')
  out.append(tasks.make(rs,spec,rng,split,'logic_holdout',g['group_id']+':logic_holdout'))
  same_domain=[r for r in rows if r['dataset']==g['records'][0]['dataset']]
  same_domain.sort(key=lambda r:sha_text(r['id']+':length_holdout:'+g['group_id']))
  rs=same_domain[:3]
  for fam in tasks.FAMILIES[4:]:
   rng=random.Random(SEED+gi*173+tasks.FAMILIES.index(fam))
   out.append(tasks.make(rs,tasks.specification(rs,fam,rng),rng,split,'length_holdout',g['group_id']+':length_holdout:'+fam))
 write(path,out);return out
def sha_text(s):return hashlib.sha256(s.encode()).hexdigest()
def prepare():
 verify();base.seed_all();guard()
 if (R/'prepared.json').exists():return
 ck=torch.load(old.R/'oracle/best.pt',map_location='cpu',weights_only=False)
 if not (R/'oracle_init.pt').exists():save(R/'oracle_init.pt',{'state_dict':ck['state_dict'],'source_sha256':sha(old.R/'oracle/best.pt'),'phase':'new optimizer; all 28 Oracle slots train; LM frozen'})
 labels=config()['labels'];teacher=[]
 for r in range(3):
  ee=[tasks.draw_train(r,i,labels) for i in range(1024)]
  write(R/'teacher'/(f'round_{r}_plan.json'),ee);teacher+=ee
 val=evaluation_plan('val');rng=random.Random(SEED+7171);tf=[]
 for i in range(96):
  e=old.draw(rng,'val',i,'base');tf.append(tasks.canonical(e))
 for i in range(144):
  fam=tasks.FAMILIES[i%9];n=rng.choice([2,4]) if fam in tasks.FAMILIES[4:] else 1
  rs=[base.synthetic_record(rng,labels,'val',100000+i*4+j) for j in range(n)]
  tf.append(tasks.make(rs,tasks.specification(rs,fam,rng),rng,'val','unseen_wording','tf:'+str(i)))
 write(R/'tf_val.json',tf)
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(base.MODEL,local_files_only=True)
 fake=nn.Embedding(tok.vocab_size+256,1);maxlen=0
 for e in teacher+tf+val:
  if e['kind']=='process':
   pf=torch.zeros(len(e['records']),28,1);maxlen=max(maxlen,len(tasks.render(tok,fake,e,pf)))
   assert tasks.parse(tasks.target(e),e)['correct']
   assert tasks.parse(tasks.target(e),e)['trace_correct']
   for r in e['records']:
    if r.get('synthetic'):assert base.combo_partition(base.bin_values(r)) in ({2,3,4} if e['split']=='train' else {0})
  else:assert old.parse(tasks.target(e),e)[-1]
 assert maxlen+384<=4096,(maxlen,'context')
 # Independent boolean truth tables and ordinal count/selection fixtures.
 checks=0
 for m in itertools.product([False,True],repeat=3):
  for flag in [False,True]:
   for pattern in ['all','any','text_train','val_holdout','test_holdout']:
    expected={'all':sum(m)==3,'any':sum(m)>0,'text_train':(flag and (m[0] or m[1])) or m[2],'val_holdout':flag and (m[0] or (m[1] and m[2])),'test_holdout':(flag or m[0]) and (m[1] or m[2])}[pattern]
    assert tasks.rule_truth(pattern,m,flag)==expected;checks+=1
 for n in [2,3,4]:
  for vv in itertools.product(tasks.VALUES,repeat=n):
   for desired in tasks.VALUES:
    assert len([i for i,v in enumerate(vv) if v==desired])==vv.count(desired);checks+=1
 # Questions carry placeholders, not observed values or variable result-length hints.
 for e in teacher:
  q=e['question'];assert 'Verified reference solution' not in q
  if e['spec']['family']=='select':
   assert 'matching record letters, possibly empty' in q
 net=initialize('cpu');bb=torch.tensor([[0,1,2,0,1,2]]);ll=torch.tensor([0]);z=net(bb,ll)
 assert z.shape==(1,28,2048)
 for f in range(6):
  b2=bb.clone();b2[0,f]=(b2[0,f]+1)%3;z2=net(b2,ll);mask=torch.ones(28,dtype=torch.bool);mask[16+f*2:18+f*2]=False
  assert torch.equal(z[:,mask],z2[:,mask])
 assert torch.equal(z[:,16:],net(bb,torch.tensor([1]))[:,16:])
 leakage_checks=0
 for e in teacher[:27]:
  before=tasks.prompt(e['records'],e['spec'],random.Random(919),'train','teacher_train')
  for level in tasks.VALUES:
   changed=copy.deepcopy(e['records'])
   for rr in changed:rr['label']='different_class';rr['bins']={f:level for f in tasks.FIELDS}
   after=tasks.prompt(changed,e['spec'],random.Random(919),'train','teacher_train')
   assert before==after,(e['id'],level,'question leaks facts');leakage_checks+=1
 for e in teacher:
  if e['spec']['family']=='select':
   gold=tasks.expected(e);wrong=copy.deepcopy(gold);wrong['trace']['checks'][0]['value']=[];wrong['result']['records']=[]
   parsed=tasks.parse(tasks.dumps(wrong),e);assert parsed['valid']
   if gold['result']['records']:assert not parsed['correct']
 for ident in {e.get('pair_id') for e in val if e.get('pair_id')}:
  pp={e['scope']:e for e in val if e.get('pair_id')==ident};a,b,c,d=[pp[k] for k in ['seen_wording','unseen_wording','long_context','role_context']]
  assert a['spec']==b['spec']==c['spec']==d['spec'] and a['records']==b['records']==c['records']==d['records']
  assert a['layout']==b['layout']==c['layout']==d['layout']
  assert a['system']==b['system']==c['system'] and a['history']==b['history']==c['history']==[]
  assert b['question']==d['question'] and len(d['history'])==2
  assert [x for x in b['question'].splitlines() if not x.startswith('Filing note ')]==[x for x in c['question'].splitlines() if not x.startswith('Filing note ')]
 write(R/'audit/cpu_check.json',{'status':'passed','no_fact_leak_checks':leakage_checks,'paired_prompt_factors_verified':True,'wrong_selection_format_independent':True,'truth_checks':checks,'teacher_candidates':len(teacher),'val_queries':len(val),'tf_queries':len(tf),'max_student_prompt_tokens':maxlen,'oracle_shape':list(z.shape),'question_independent_factorization':True,'no_test_access':True,'grammar':'Finite compositional grammar; no claim of unrestricted language generalization.'})
 write(R/'prepared.json',{'status':'complete','time':time.time(),'init_sha256':sha(R/'oracle_init.pt'),'val_plan_sha256':sha(R/'val_plan.json'),'tf_sha256':sha(R/'tf_val.json')})
def engine_load():
 from transformers import AutoTokenizer,AutoModelForCausalLM
 from vllm import LLM,SamplingParams
 tok=AutoTokenizer.from_pretrained(base.MODEL,local_files_only=True)
 lm=AutoModelForCausalLM.from_pretrained(base.MODEL,torch_dtype=torch.bfloat16,local_files_only=True).cpu()
 emb=copy.deepcopy(lm.get_input_embeddings());del lm
 batch=int(os.environ.get('VSTATS_COMPACT_AR_BATCH','16'))
 engine=LLM(model=str(base.MODEL),dtype='bfloat16',enable_prompt_embeds=True,max_model_len=4096,max_num_seqs=batch,gpu_memory_utilization=float(os.environ.get('VSTATS_COMPACT_AR_MEMORY_UTILIZATION','.35')),enforce_eager=True,disable_log_stats=True,seed=SEED)
 return tok,emb,engine,SamplingParams,batch
def teacher_generate(pilot=False):
 guard();base.seed_all();out=R/'teacher';done=out/('pilot.summary.json' if pilot else 'summary.json')
 if complete(done):return
 tok,emb,engine,SamplingParams,batch=engine_load()
 plans=[read(out/f'round_{r}_plan.json') for r in range(3)]
 ee=plans[0][:64] if pilot else sum(plans,[])
 raw=out/('pilot.jsonl' if pilot else 'generations.jsonl');existing={}
 if raw.exists():
  for line in raw.read_text().splitlines():
   x=json.loads(line);existing[x['id']]=x
 started=time.time()
 with raw.open('a') as f:
  missing=[e for e in ee if e['id'] not in existing]
  for off in range(0,len(missing),batch):
   bb=missing[off:off+batch];requests=[]
   for e in bb:
    x=tasks.render(tok,emb,e,teacher=True,reference=True);assert len(x)+384<=4096
    requests.append({'prompt_embeds':x.bfloat16()})
   params=[SamplingParams(temperature=.7,top_p=.95,max_tokens=384,seed=SEED+int(e['id'].rsplit(':',1)[1])+10000*int(e['id'].split(':')[1][1:])) for e in bb]
   outputs=engine.generate(requests,params,use_tqdm=False)
   for e,o in zip(bb,outputs):
    rawtext=o.outputs[0].text;normalized,wrapper=tasks.normalize_teacher(rawtext);score=tasks.parse(normalized,e)
    row={'id':e['id'],'family':e['spec']['family'],'generation':rawtext,'trajectory':normalized,'wrapper_normalized':wrapper,'accepted':score['correct'],'valid':score['valid'],'finish_reason':o.outputs[0].finish_reason,'reference_aided':True,'plan_item_sha256':sha_text(tasks.dumps(e))}
    existing[e['id']]=row;f.write(tasks.dumps(row)+'\n')
   f.flush();print('TEACHER',pilot,off+len(bb),len(missing),flush=True)
   if time.time()>float(os.environ.get('VSTATS_STAGE_DEADLINE','inf'))-60:raise TimeoutError('Teacher cumulative stage budget; raw results retained')
 accepted=[existing[e['id']] for e in ee if existing[e['id']]['accepted']]
 result={'status':'complete','pilot':pilot,'candidates':len(ee),'accepted':len(accepted),'acceptance_rate':len(accepted)/len(ee),'seconds':time.time()-started,'reference_aided':True,'all_accepted_targets_program_verified':True}
 if pilot:
  result['passed']=len(accepted)>=32
  write(done,result)
  if not result['passed']:raise RuntimeError('Teacher pilot below fixed32/64 acceptance; do not train')
 else:
  per={}
  for r,plan in enumerate(plans):
   bank=[]
   for e in plan:
    if existing[e['id']]['accepted']:bank.append({**e,'teacher_trajectory':existing[e['id']]['trajectory']})
   counts=dict(collections.Counter(e['spec']['family'] for e in bank));ok=len(bank)>=512 and all(counts.get(k,0)>=32 for k in tasks.FAMILIES)
   write(out/f'round_{r}_bank.json',bank);per[str(r)]={'accepted':len(bank),'per_family':counts,'passed':ok,'bank_sha256':sha(out/f'round_{r}_bank.json')}
  result['rounds']=per;result['passed']=all(v['passed'] for v in per.values());write(done,result)
  if not result['passed']:raise RuntimeError('Teacher accepted bank below fixed coverage minimum')
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
  x=tasks.render(tok,emb,e,teacher=True,reference=True)
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
def data_batch(step,banks,round_):
 rng=random.Random(SEED+step*10007);ee=[]
 for i in range(8):
  if rng.random()<.2:
   e=old.draw(rng,'train',step*8+i,'base');ee.append(tasks.canonical(e))
  else:
   rr=rng.randrange(round_+1) if round_ and rng.random()<.25 else round_
   ee.append(rng.choice(banks[rr]))
 return ee
def train(role,round_,smoke=False):
 guard();base.seed_all();out=R/role/('smoke' if smoke else 'oracle');out.mkdir(parents=True,exist_ok=True)
 tok,lm,emb=base.hf_decoder();net=initialize('cuda');opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
 rng=random.Random(SEED+881);start=0;best=math.inf;history=[];began=time.time()
 micro=int(os.environ.get('VSTATS_COMPACT_MICRO_BATCH','1'));assert micro in [1,2]
 if not smoke and (out/'last.pt').exists():
  ck=torch.load(out/'last.pt',map_location='cpu',weights_only=False);net.load_state_dict(ck['state_dict']);opt.load_state_dict(ck['optimizer']);ref.restore(ck,rng);random.setstate(ck['python_global_rng']);start=ck['step'];best=ck['best'];history=ck['history']
 initial={k:p.detach().clone() for k,p in net.named_parameters()}
 banks=[read(R/'teacher'/f'round_{r}_bank.json') for r in range(3)]
 tf=read(R/'tf_val.json');end=30 if smoke else (round_+1)*600
 prob=.5 if smoke else [0.,.25,.75][round_]
 audit_path=out/'rollout_trace.jsonl'
 def state(step):
  return {'state_dict':{k:v.detach().cpu() for k,v in net.state_dict().items()},'optimizer':opt.state_dict(),'step':step,'best':best,'history':history,**ref.rng_state(rng),'python_global_rng':random.getstate(),'source_sha256':sha(__file__),'init_sha256':sha(R/'oracle_init.pt'),'role':role,'round':round_,'tokens':28,'micro_batch':micro,'effective_batch':8,'student_rollout_probability':prob if role=='curriculum' else 0.,'teacher_bank_hashes':[sha(R/'teacher'/f'round_{r}_bank.json') for r in range(3)]}
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
   grad=nn.utils.clip_grad_norm_(net.parameters(),1.);assert torch.isfinite(grad) and grad>0
   assert all(p.grad is None for p in lm.parameters());opt.step()
   audit.write(tasks.dumps({'step':step,'round':round_,'ce':cev,'kd':kdv,'student_probability':prob if role=='curriculum' else 0.,'sample_ids':[e['id'] for e in ee],'student_rollouts':events})+'\n');audit.flush()
   if step%10==0:print(tasks.dumps({'step':step,'round':round_,'ce':cev,'kd':kdv,'grad':float(grad),'student_rollouts':len(student),'seconds':time.time()-began}),flush=True)
   if not smoke and step%100==0:save(out/'last.pt',state(step))
   if smoke and step==end:
    changed={k:not torch.equal(v,dict(net.named_parameters())[k]) for k,v in initial.items()};assert all(changed.values()),changed
    save(out/'last.pt',state(step));restored=initialize('cuda');restored.load_state_dict(torch.load(out/'last.pt',map_location='cpu',weights_only=False)['state_dict'])
    assert all(torch.equal(a,b) for a,b in zip(net.state_dict().values(),restored.state_dict().values()))
    write(out/'summary.json',{'status':'passed','steps':end,'changed':changed,'decoder_frozen':True,'checkpoint_roundtrip':True,'student_generate_exercised':bool(selected) if role=='curriculum' else None,'seconds':time.time()-began,'peak_torch_gib':torch.cuda.max_memory_allocated()/2**30})
   if not smoke and (step%500==0 or step==end):
    net.eval();v=0.
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
     for e in tf:v+=float(ce_objective(lm,tok,emb,net,e,tasks.target(e))[0])/len(tf)
    improve=v<best;best=min(best,v);history.append({'step':step,'val_tf_ce':v,'round':round_})
    ck=state(step);save(out/'last.pt',ck)
    if improve:save(out/'best.pt',ck)
    write(out/'summary.json',{'status':'complete' if step==1800 else 'round_complete' if step==end else 'running','steps':step,'round':round_,'best_val_tf_ce':best,'history':history,'seconds_this_stage':time.time()-began,'peak_torch_gib':torch.cuda.max_memory_allocated()/2**30,'trainable_parameters':sum(p.numel() for p in net.parameters())})
    print('VALIDATION',history[-1],flush=True)
 if not smoke:
  save(out/f'round_{round_}_last.pt',state(end));write(out/f'round_{round_}.complete.json',{'status':'complete','step':end,'checkpoint_sha256':sha(out/f'round_{round_}_last.pt')})

def summarize(rows):
 canonical={m:base.metrics([r for r in rows if r['mode']==m and r['scope']=='original_base']) for m in sorted({r['mode'] for r in rows}) if any(r['mode']==m and r['scope']=='original_base' for r in rows)}
 groups=collections.defaultdict(list)
 for r in rows:
  if r['scope']!='original_base':groups[(r['mode'],r['scope'],r['family'])].append(r)
 result=[]
 for (mode,suite,fam),rr in sorted(groups.items()):
  per={}
  for ds in sorted({r['dataset'] for r in rr}):
   ss=[r for r in rr if r['dataset']==ds];dd=[r for r in ss if r['decision_score'] is not None]
   good=[r for r in dd if r['valid'] and r['fact_score']==1]
   per[ds]={'n':len(ss),'facts':float(np.mean([r['fact_score'] for r in ss])),'format':float(np.mean([r['valid'] for r in ss])),'exact':float(np.mean([r['correct'] for r in ss])),'trace':float(np.mean([r['trace_correct'] for r in ss])),'decision':float(np.mean([r['decision_score'] for r in dd])) if dd else None,'decision_given_all_facts':float(np.mean([r['decision_score'] for r in good])) if good else None,'conditional_support':len(good)}
  g={'mode':mode,'scope':suite,'family':fam,'n':len(rr),'per_dataset':per}
  for k in ['facts','format','exact','trace','decision']:
   vv=[v[k] for v in per.values() if v[k] is not None];g[k]=float(np.mean(vv)) if vv else None
  allgood=[r for r in rr if r['decision_score'] is not None and r['valid'] and r['fact_score']==1]
  g['conditional_support']=len(allgood);g['decision_given_all_facts']=float(np.mean([r['decision_score'] for r in allgood])) if allgood else None
  result.append(g)
 return {'status':'complete','queries':len(rows),'canonical':canonical,'task_metrics':result,'aggregation':'Dataset-macro for facts/format/exact/decision; conditional decision pooled with explicit support. No nonexistent task composite.'}
def evaluate(role,split,round_=None,smoke=False):
 verify();guard();out=R/role/'evaluation';out.mkdir(parents=True,exist_ok=True)
 suffix='smoke' if smoke else f'round_{round_}_val' if round_ is not None else split
 path=out/(suffix+'.jsonl');summary=out/(suffix+'.summary.json')
 if complete(summary):return
 ee=evaluation_plan(split)
 if smoke:ee=[e for e in ee if e['kind']=='process'][:18]
 elif round_ is not None:
  chosen=[]
  for ds in sorted({e['dataset'] for e in ee}):
   dd=[e for e in ee if e['dataset']==ds and e['scope'] in ['seen_wording','unseen_wording']]
   dd.sort(key=lambda e:sha_text(e['id']+':round_progress'));chosen+=dd[:18]
  ee=chosen
 modes=[role]
 if role=='ce' and split=='val' and round_ is None and not smoke:modes=['initial','ce']
 if role=='ce' and split=='test':modes=['initial','ce']
 # Initial Test uses only new suites; canonical initial Test remains N/A.
 pairs=[(mode,e) for mode in modes for e in ee if not (split=='test' and mode=='initial' and e['scope']=='original_base')]
 tok,emb,engine,SamplingParams,batch=engine_load();net=initialize('cuda').eval().requires_grad_(False)
 ckpt=(R/role/'smoke/last.pt') if smoke else (R/role/'oracle'/f'round_{round_}_last.pt') if round_ is not None else R/role/'oracle/best.pt'
 state=torch.load(ckpt,map_location='cpu',weights_only=False)['state_dict'];net.load_state_dict(state)
 params=SamplingParams(temperature=0.,max_tokens=384);began=time.time();existing={}
 if path.exists():
  for line in path.read_text().splitlines():
   x=json.loads(line);existing[(x['mode'],x['id'])]=x
 missing=[(m,e) for m,e in pairs if (m,e['id']) not in existing]
 with path.open('a') as f:
  for off in range(0,len(missing),batch):
   bb=missing[off:off+batch];requests=[]
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
    for mode,e in bb:
     if mode=='initial':net.load_state_dict(torch.load(R/'oracle_init.pt',map_location='cpu',weights_only=False)['state_dict'])
     else:net.load_state_dict(state)
     x=tasks.render(tok,emb,e,prefix(net,e)).cpu().bfloat16();assert len(x)+384<=4096
     requests.append({'prompt_embeds':x})
   outputs=engine.generate(requests,params,use_tqdm=False)
   for (mode,e),o,x in zip(bb,outputs,requests):
    text=o.outputs[0].text;score=tasks.parse(text,e)
    row={'mode':mode,'id':e['id'],'scope':e['scope'],'family':e['spec']['family'] if e['kind']=='process' else e['q']['task'],'task':'process:'+e['spec']['family'] if e['kind']=='process' else e['q']['task'],'kind':e['kind'],'dataset':e['dataset'],'sample_ids':[r['id'] for r in e['records']],'gold_label':e['records'][0]['label'],'question':e['question'] if e['kind']=='process' else e['q']['question'],'expected':tasks.expected(e),'generation':text,**score,'records':len(e['records']),'soft_tokens':28*len(e['records']),'prompt_tokens':len(x['prompt_embeds']),'context_lines':e.get('context_lines',0),'gold_input':True,'finish_reason':o.outputs[0].finish_reason,'model_sha256':sha(R/'oracle_init.pt') if mode=='initial' else sha(ckpt)}
    if e['kind']!='process' and 'field' in e['q']:row['field']=e['q']['field']
    existing[(mode,e['id'])]=row;f.write(tasks.dumps(row)+'\n')
   f.flush()
   if off%80==0:print('AR',role,suffix,off+len(bb),len(missing),flush=True)
   if time.time()>float(os.environ.get('VSTATS_STAGE_DEADLINE','inf'))-60:raise TimeoutError('Cumulative AR budget; partial raw retained')
 rows=[existing[(m,e['id'])] for m,e in pairs]
 result=summarize(rows);result.update({'source_sha256':sha(__file__),'plan_sha256':sha(R/(split+'_plan.json')),'checkpoint_sha256':sha(ckpt),'split':split,'seconds_this_attempt':time.time()-began,'peak_torch_gib':torch.cuda.max_memory_allocated()/2**30,'max_new_tokens':384,'temperature':0})
 write(summary,result);write(out/(suffix+'.failures.json'),[r for r in rows if not r['correct']][:60])
def promotion_gate():
 d=read(R/'curriculum/evaluation/val.summary.json');checks=[];per=d['canonical']['curriculum']['per_dataset']
 for task in ['label']+['feature:'+f for f in tasks.FIELDS]:
  checks.append({'name':'canonical:'+task+':bacc','value':float(np.mean([v['tasks'][task]['bacc'] for v in per.values()])),'minimum':.8})
 checks.append({'name':'canonical:report:facts','value':float(np.mean([v['tasks']['report']['fact_accuracy'] for v in per.values()])),'minimum':.8})
 for x in d['task_metrics']:
  for k in ['facts','format','decision']:
   if x[k] is not None:checks.append({'name':x['scope']+':'+x['family']+':'+k,'value':x[k],'minimum':.8})
 passed=all(x['value']>=x['minimum'] for x in checks)
 write(R/'oracle_promotion_gate.json',{'status':'complete','passed':passed,'checks':checks,'failed':[x for x in checks if x['value']<x['minimum']],'decision':'Ready as frozen target only when all predeclared Val checks pass; fixed diagnostic Test still runs after freeze. No automatic EEG training.'})
 return passed
def stage(role,name,args,artifact):
 W=R/('worker_'+role);W.mkdir(parents=True,exist_ok=True);log=W/(name+'.log');meta_path=W/(name+'.attempt_state.json')
 if complete(R/artifact):print('SKIP',role,name,flush=True);return
 limit=read(R/'first_launch.json')['started']+12*3600
 meta=read(meta_path) if meta_path.exists() else {'first_started':None,'attempts':0,'ar_batch':16,'ar_memory_utilization':.35,'micro_batch':1}
 ar=args[0] in ['teacher','eval'];need=(24 if ar else 26)*1024
 while True:
  deadline=min(limit,meta['first_started']+4*3600 if meta['first_started'] else limit)
  if time.time()>=deadline:raise TimeoutError('Original worker/stage budget '+name)
  lease=None;env=dict(os.environ);chosen=None;child=None
  try:
   while lease is None:
    cards=[]
    q=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    for l in q.splitlines():
     gpu,total,free,util=[int(v.strip()) for v in l.split(',')]
     if gpu in range(2,8) and free>=need:cards.append((gpu,total,free,util))
    for gpu,total,free,util in sorted(cards,key=lambda x:(-x[2],x[3],x[0])):
     f=(R/f'gpu{gpu}.lease').open('a+')
     try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
     except BlockingIOError:f.close();continue
     free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
     if free<need:fcntl.flock(f,fcntl.LOCK_UN);f.close();continue
     lease=f;chosen=gpu;env.update(CUDA_VISIBLE_DEVICES=str(gpu),VSTATS_COMPACT_ALLOCATOR_BUDGET_MIB=str(free-10*1024),VSTATS_COMPACT_GPU_TOTAL_MIB=str(total));break
    if lease is None:
     write(W/'status.json',{'state':'waiting_gpu_memory','stage':name,'pipeline_pid':os.getpid(),'required_free_mib':need,'time':time.time()})
     if time.time()>=deadline:raise TimeoutError('Original budget waiting memory')
     time.sleep(10)
   guard()
   if meta['first_started'] is None:meta['first_started']=time.time()
   meta['attempts']+=1;write(meta_path,meta);deadline=min(limit,meta['first_started']+4*3600)
   env.update(VSTATS_STAGE_DEADLINE=str(deadline),VSTATS_COMPACT_MICRO_BATCH=str(meta['micro_batch']),VSTATS_COMPACT_AR_BATCH=str(meta['ar_batch']),VSTATS_COMPACT_AR_MEMORY_UTILIZATION=str(meta['ar_memory_utilization']))
   cmd=[ARPY if ar else TRAINPY,'-B',str(Path(__file__).resolve())]+args
   offset=log.stat().st_size if log.exists() else 0
   with log.open('a') as f:
    f.write(tasks.dumps({'time':time.time(),'attempt':meta,'gpu':chosen,'command':cmd,'allocator_budget_mib':env['VSTATS_COMPACT_ALLOCATOR_BUDGET_MIB']})+'\n');f.flush()
    child=subprocess.Popen(cmd,cwd=P,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    write(W/'status.json',{'state':'running','stage':name,'pipeline_pid':os.getpid(),'pid':child.pid,'gpu':chosen,'time':time.time(),'log':str(log),'command':cmd})
    code=child.wait(timeout=max(1,deadline-time.time()))
   if code==0:assert complete(R/artifact),'Missing complete artifact '+artifact;return
   with log.open() as f:f.seek(offset);tail=f.read().lower()
   oom=any(x in tail for x in ['cuda out of memory','torch.outofmemoryerror','cuda error: out of memory'])
   if not oom:raise RuntimeError(role+' '+name+' failed; see '+str(log))
   before=dict(meta)
   if not ar or (meta['ar_batch']==1 and meta['ar_memory_utilization']<=.2):raise RuntimeError('OOM at minimum allowed resource configuration; checkpoint retained')
   meta['ar_batch']=max(1,meta['ar_batch']//2);meta['ar_memory_utilization']=max(.2,round(meta['ar_memory_utilization']-.05,2));write(meta_path,meta)
   with (W/'resource_retries.jsonl').open('a') as f:f.write(tasks.dumps({'time':time.time(),'stage':name,'before':before,'after':meta,'failed_pid':child.pid,'unchanged_models_tasks_budget':True})+'\n')
  finally:
   if child is not None and child.poll() is None:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=30)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
   if lease is not None:fcntl.flock(lease,fcntl.LOCK_UN);lease.close()
def worker(role):
 W=R/('worker_'+role);W.mkdir(exist_ok=True)
 with (W/'pipeline.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  write(W/'launch.json',{'pid':os.getpid(),'started':time.time(),'source_sha256':sha(__file__)})
  try:
   verify()
   for r in range(3):
    stage(role,f'train_round_{r}',['train','--role',role,'--round',str(r)],f'{role}/oracle/round_{r}.complete.json')
    stage(role,f'progress_round_{r}',['eval','--role',role,'--round',str(r)],f'{role}/evaluation/round_{r}_val.summary.json')
   stage(role,'val',['eval','--role',role],f'{role}/evaluation/val.summary.json')
   write(W/'val_complete.json',{'status':'complete','time':time.time(),'best_sha256':sha(R/role/'oracle/best.pt')})
   while not (R/'test_freeze.json').exists():
    if time.time()>read(R/'first_launch.json')['started']+12*3600:raise TimeoutError('Waiting global freeze in original budget')
    if (R/'abort.json').exists():raise RuntimeError('Coordinator aborted; do not access Test')
    write(W/'status.json',{'state':'waiting_test_freeze','pipeline_pid':os.getpid(),'time':time.time()});time.sleep(10)
   stage(role,'test',['eval','--role',role,'--split','test'],f'{role}/evaluation/test.summary.json')
   write(W/'complete.json',{'status':'complete','time':time.time()});write(W/'status.json',{'state':'complete','pipeline_pid':os.getpid(),'time':time.time()})
  except BaseException:
   write(W/'status.json',{'state':'failed','pipeline_pid':os.getpid(),'traceback':traceback.format_exc(),'time':time.time()});raise
def pipeline():
 W=R/'worker_master';W.mkdir(exist_ok=True)
 with (W/'pipeline.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if not (R/'first_launch.json').exists():write(R/'first_launch.json',{'started':time.time()})
  write(W/'launch.json',{'pid':os.getpid(),'source_sha256':sha(__file__),'started':read(R/'first_launch.json')['started']})
  children=[]
  try:
   verify();assert complete(R/'prepared.json')
   if (R/'abort.json').exists():
    (R/'abort.json').rename(R/'audit'/('prior_abort_'+str(int(time.time()))+'.json'))
   stage('master','teacher_pilot',['teacher','--smoke'],'teacher/pilot.summary.json')
   assert read(R/'teacher/pilot.summary.json')['passed']
   stage('master','teacher_bank',['teacher'],'teacher/summary.json');assert read(R/'teacher/summary.json')['passed']
   for role in ['curriculum','ce']:
    stage('master',role+'_smoke',['train','--role',role,'--smoke'],f'{role}/smoke/summary.json')
    stage('master',role+'_ar_smoke',['eval','--role',role,'--smoke'],f'{role}/evaluation/smoke.summary.json')
   for role in ['ce','curriculum']:
    if complete(R/('worker_'+role)/'complete.json'):continue
    f=(R/('worker_'+role)/'dispatch.log').open('a');p=subprocess.Popen([TRAINPY,'-B',str(Path(__file__).resolve()),'worker','--role',role],cwd=P,env=dict(os.environ),stdout=f,stderr=subprocess.STDOUT,start_new_session=True);f.close();children.append((role,p))
    write(R/('worker_'+role)/'dispatch.json',{'pid':p.pid,'time':time.time(),'source_sha256':sha(__file__)})
   deadline=read(R/'first_launch.json')['started']+12*3600
   while not all(complete(R/('worker_'+role)/'val_complete.json') for role in ['ce','curriculum']):
    if any(p.poll() not in [None,0] for role,p in children):raise RuntimeError('An owned worker failed; preserve logs, stop peer safely')
    if time.time()>deadline:raise TimeoutError('Original master12h budget')
    write(W/'status.json',{'state':'waiting_workers_val','pipeline_pid':os.getpid(),'children':{r:p.pid for r,p in children},'time':time.time()});time.sleep(10)
   promotion_gate();verify()
   if not (R/'test_freeze.json').exists():write(R/'test_freeze.json',{'status':'frozen','time':time.time(),'models':{role:sha(R/role/'oracle/best.pt') for role in ['ce','curriculum']},'init_sha256':sha(R/'oracle_init.pt'),'source_sha256':sha(__file__),'dependencies':config()['dependencies'],'selection':'Fixed240 Val TF CE only across1800; robust AR never selects. All real Test previously exposed, repeated benchmark.'})
   while any(p.poll() is None for role,p in children):
    if any(p.poll() not in [None,0] for role,p in children):raise RuntimeError('An owned Test worker failed')
    if time.time()>deadline:raise TimeoutError('Original master12h budget')
    write(W/'status.json',{'state':'waiting_workers_test','pipeline_pid':os.getpid(),'children':{r:p.pid for r,p in children},'time':time.time()});time.sleep(10)
   assert all(complete(R/('worker_'+role)/'complete.json') for role in ['ce','curriculum'])
   write(W/'status.json',{'state':'finalizing','pipeline_pid':os.getpid(),'time':time.time()})
   finalize(require_process_exit=False);write(W/'complete.json',{'status':'complete','time':time.time()});write(W/'status.json',{'state':'complete','time':time.time(),'pipeline_pid':os.getpid()})
  except BaseException:
   write(R/'abort.json',{'time':time.time(),'error':traceback.format_exc()});write(W/'status.json',{'state':'failed','pipeline_pid':os.getpid(),'traceback':traceback.format_exc(),'time':time.time()});raise
  finally:
   for role,p in children:
    if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
   for role,p in children:
    if p.poll() is None:
     try:p.wait(timeout=45)
     except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
def finalize(require_process_exit=True):
 verify();out=R/'details';out.mkdir(exist_ok=True);allrows=[];checks={}
 for role in ['ce','curriculum']:
  for split in ['val','test']:
   path=R/role/'evaluation'/(split+'.jsonl');rr=[json.loads(l) for l in path.read_text().splitlines()];plan={e['id']:e for e in evaluation_plan(split)}
   seen=set()
   for row in rr:
    key=(row['mode'],row['id']);assert key not in seen;seen.add(key);e=plan[row['id']]
    assert row['sample_ids']==[r['id'] for r in e['records']] and tasks.same(row['expected'],tasks.expected(e))
    parsed=tasks.parse(row['generation'],e)
    for k,v in parsed.items():assert tasks.same(row[k],v),(role,split,row['id'],k)
   recomputed=summarize(rr);saved=read(R/role/'evaluation'/(split+'.summary.json'))
   for k in ['queries','canonical','task_metrics']:assert recomputed[k]==saved[k],(role,split,k)
   checks[role+':'+split]={'queries':len(rr),'raw_sha256':sha(path),'summary_sha256':sha(R/role/'evaluation'/(split+'.summary.json'))}
   allrows.extend([{**r,'split':split} for r in rr])
 freeze=read(R/'test_freeze.json')
 for role in ['ce','curriculum']:assert freeze['models'][role]==sha(R/role/'oracle/best.pt')
 tasks_rows=[]
 for role in ['ce','curriculum']:
  for split in ['val','test']:
   for x in read(R/role/'evaluation'/(split+'.summary.json'))['task_metrics']:
    tasks_rows.append({'split':split,**{k:x[k] for k in ['mode','scope','family','n','facts','format','exact','trace','decision','decision_given_all_facts','conditional_support']}})
 with (out/'tasks.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(tasks_rows[0]));w.writeheader();w.writerows(tasks_rows)
 fields=[]
 for split in ['val','test']:
  for role in ['ce','curriculum']:
   c=read(R/role/'evaluation'/(split+'.summary.json'))['canonical']
   for mode,vv in c.items():
    for ds,v in vv['per_dataset'].items():
     for name,a in v['tasks'].items():fields.append({'split':split,'mode':mode,'dataset':ds,'task':name,'n':a['n'],'bacc':a.get('bacc'),'macro_f1':a.get('macro_f1'),'exact':a['exact_accuracy'],'facts':a.get('fact_accuracy'),'format':a['format_valid_rate'],'support_confusion':tasks.dumps(a.get('confusion'))})
 with (out/'canonical.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(fields[0]));w.writeheader();w.writerows(fields)
 write(out/'examples.json',{'success':[r for r in allrows if r['mode']=='curriculum' and r['correct'] and r['kind']=='process'][:12],'failures':[r for r in allrows if r['mode']=='curriculum' and not r['correct']][:24]})
 gate=read(R/'oracle_promotion_gate.json')
 lines=['# 28-token Oracle teacher→student课程蒸馏','','仅更新Oracle分类16＋统计12；Qwen3-1.7B与EEG模型冻结。先参考解答辅助teacher rollout，再按0/25/75%逐轮使用student rollout的分布蒸馏。每轮600步，3轮共1800步；CE对照使用完全相同teacher bank、采样和预算，但没有KD。主变量是整个蒸馏课程，不单独归因KL或采样比例。','','所有输入为gold标签和六字段档位，目标报告来自程序事实；不是EEG效果或临床报告。生成题库为有限组合语法，含独立问法、长背景、系统提示/无关对话及留出逻辑和3记录组合；不能声称接入任意General AI生态。Teacher看程序验证参考解答，接受率不是无辅助teacher准确率。','','原16域split/ID/阈值不变。Test为此前暴露的重复benchmark128/域；本轮Val冻结之后才访问。没有按robust AR或Test选模；best仅固定Val TF CE。','','|Split|模型|测试范围|任务|n|事实%|格式%|exact%|过程全对%|严格决策%|读全事实后决策%|支持数|','|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for x in tasks_rows:
  def fmt(k):return 'N/A' if x[k] is None else f'{100*x[k]:.2f}'
  lines.append('|'+ '|'.join([x['split'],x['mode'],x['scope'],x['family'],str(x['n'])]+[fmt(k) for k in ['facts','format','exact','trace','decision','decision_given_all_facts']]+[str(x['conditional_support'])])+'|')
 lines+=['','分类与六字段逐数据集BAcc/MF1/支持/混淆见 details/canonical.csv；原任务与新过程任务分别列示。对应同一记录/spec的seen_wording与其他prompt suite可配对比较；条件决策必须连支持数一起解读。','', 'Oracle晋级Val80%各项门槛：'+str(gate['passed'])+'。未通过则不提供已合格EEG对齐目标；无自动EEG训练。','', '失败项：'+tasks.dumps(gate['failed']),'','模型、训练曲线、每轮mini Val、teacher原始输出/过滤、student训练轨迹、资源日志保留在本run。']
 (R/'final_report.md').write_text('\n'.join(lines)+'\n')
 if gate['passed']:write(R/'encoder_target.json',{'status':'ready_for_alignment_review','oracle_checkpoint':str(R/'curriculum/oracle/best.pt'),'sha256':sha(R/'curriculum/oracle/best.pt'),'tokens':28,'question_independent':True,'fields':tasks.FIELDS,'labels':config()['labels'],'test_freeze_sha256':sha(R/'test_freeze.json'),'no_eeg_training_started':True})
 write(R/'audit/final_verification.json',{'status':'passed','time':time.time(),'raw_and_summaries':checks,'freeze_hashes_match':True,'process_exit_check_pending':not require_process_exit})
 write(R/'report_complete.json',{'status':'complete','time':time.time(),'report_sha256':sha(R/'final_report.md'),'promotion_gate':gate['passed']})
 ledger=P/'experiments/mainline_vrl/reports/20260808_experiment_ledger.md';key=R.name+' — Oracle课程蒸馏'
 if key not in ledger.read_text():
  with ledger.open('a') as f:f.write('\n\n### '+key+'\n\n3×600步Oracle-only；同预算CE与teacher→student KD；有限prompt语法、独立问法/背景/角色/逻辑与长度诊断。Val晋级='+str(gate['passed'])+'；重复Test，不写旧分类榜，不自动接EEG。见[报告](../runs/'+R.name+'/final_report.md)。\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('command');p.add_argument('--role',choices=['ce','curriculum'],default='curriculum');p.add_argument('--round',type=int,default=None);p.add_argument('--split',default='val');p.add_argument('--smoke',action='store_true');a=p.parse_args()
 def stop(*args):raise SystemExit('Owned worker received termination; preserve checkpoints and release locks')
 signal.signal(signal.SIGTERM,stop)
 if os.environ.get('CUDA_VISIBLE_DEVICES'):
  torch.cuda.set_per_process_memory_fraction(min(1.,int(os.environ['VSTATS_COMPACT_ALLOCATOR_BUDGET_MIB'])/int(os.environ['VSTATS_COMPACT_GPU_TOTAL_MIB'])))
 if a.command=='prepare':prepare()
 elif a.command=='teacher':teacher_generate(a.smoke)
 elif a.command=='train':train(a.role,a.round if a.round is not None else 0,a.smoke)
 elif a.command=='eval':evaluate(a.role,a.split,a.round,a.smoke)
 elif a.command=='worker':worker(a.role)
 elif a.command=='pipeline':pipeline()
 elif a.command=='finalize':finalize()
 else:raise ValueError(a.command)
if __name__=='__main__':main()
